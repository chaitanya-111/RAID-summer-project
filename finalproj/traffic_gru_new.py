"""
GRU dynamic EDGE-WEIGHT model (PyTorch) for SUMO data, for use with Dijkstra / A* routing.

Pipeline:
    load data -> [T, E, F] tensors -> time-aware split -> train-only scaling + train-only cost fit
    -> sliding windows -> encoder/decoder GRU -> positive edge weights -> train/validate
    -> evaluate vs. simple traffic-cost baselines -> predict_edge_weights() for inference.
NOT included: routing itself (Dijkstra / A*). Routing stays in your own program.

------------------------------------------------------------------------------------------
WHAT IS THE TRAINING TARGET?  (please read - this is the honest part)
------------------------------------------------------------------------------------------
A SUMO edge-data file has traffic measurements (travel time, waiting time, density, speed...).
It does NOT contain a ground-truth "optimal routing weight" per edge, and this file does NOT invent one.
So the model is NOT supervised by true optimal weights. Instead we define an explicit, documented
TRAFFIC COST from the measured features (see CostSpec):

    traffic_cost(e, t) = cost_base
        + travel_time_weight * travel_time / train_mean(travel_time)
        + waiting_time_weight* waiting_time/ train_mean(waiting_time)
        + time_loss_weight   * time_loss    / train_mean(time_loss)
        + density_weight     * density      / train_mean(density)
        + occupancy_weight   * occupancy    / train_mean(occupancy)
        + speed_weight       * (1 - speed / free_flow_speed(e))      # slow relative to the edge's own free flow

Why is this a useful routing weight?
  * It is >= cost_base (>0), so it is a valid Dijkstra/A* edge cost.
  * Long/slow edges (large travel time), congested edges (density, occupancy), and edges where vehicles
    wait or lose time all get a higher cost; fast free-flowing edges get a cost close to cost_base.
  * Every term is divided by a TRAIN-ONLY statistic, so the terms are unit-free and comparable, and the
    coefficients in Config (travel_time_weight, ...) say how much each phenomenon matters to YOU.

What does the GRU learn? Given the last L steps of traffic features (+ edge identity + blockage flags),
it outputs the traffic cost of each edge for the next H steps (FORECAST of the cost, so a router can
plan with what is about to happen) and, as an auxiliary task, the cost at the observed steps
(a "nowcast", so the model also learns "traffic state at t -> cost at t").
The value of the GRU over a simple formula is that it anticipates the cost (e.g. rising congestion) and
gives a smooth, positive, learned mapping. Whether it actually beats the simple baselines is measured in
evaluate_split() and printed - it is not assumed.

Blocked edges: their measured traffic says nothing about routing cost (an empty blocked road looks
"free-flow"!), so blocked cells are EXCLUDED from the loss, and the routing layer must hard-block them
(predict_edge_weights() returns blocked edges with weight = inf and lists them in "blocked_edge_ids").

Tensor shapes (B=batch, L=input steps, H=horizon steps, E=edges, F=features).  1 step = 60 s in your data
    raw scenario                 [T, E, F]
    model input x                [B, L, E, F]     scaled traffic features
    blocked history bx           [B, L, E]        0/1 flag
    blocked future by            [B, H, E]        0/1 flag (scenario condition, known in advance)
    inside the GRUs              [B*E, steps, ...] every edge becomes its own sequence
    encoder hidden states        [B*E, L, hidden]
    PRIMARY output w_fut         [B, H, E]        positive routing weights for the future steps
    auxiliary output w_now       [B, L, E]        positive weights at the observed steps

Usage:
    1) Edit Config below (column names!).   2) python traffic_gru.py --csv your_data.csv
"""
import argparse
import copy
import re
import random
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import Dataset, DataLoader

FORBIDDEN = {"departed", "arrived", "lanechangefrom", "lanechangeto",
             "lanechangedfrom", "lanechangedto"}  # never used anywhere (SUMO spells it "laneChangedFrom")


def is_forbidden(col) -> bool:
    return re.sub(r"[^a-z]", "", str(col).lower()) in FORBIDDEN


def norm_name(s) -> str:
    """'travelTime', 'travel_time', 'Travel Time' -> 'traveltime' (used to match column names)."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ----------------------------------------------------------------------------------------
# CONFIG  -> EDIT THE COLUMN NAMES TO MATCH YOUR FILE
# ----------------------------------------------------------------------------------------
@dataclass
class Config:
    # ---- data columns ----
    csv_path: str = "traffic.csv"            # .csv or .parquet
    time_col: str = "time"                   # numeric time (or parseable datetime)
    edge_col: str = "edge_id"                # edge / lane identifier (treated as categorical)
    scenario_col: str | None = None          # run/simulation id column; None = whole file is one run
    feature_cols: list | None = None         # None = every numeric column not used as id/time/blockage
    # Blockage -- use ONE of the two (or neither, then flags are all 0 with a warning):
    blocked_col: str | None = None           # per-row 0/1 flag: this edge is blocked at this time
    blocked_edge_col: str | None = None      # per-row id of the blocked edge in that run/time
    no_block_values: tuple = ("", "none", "nan", "null")  # values of blocked_edge_col meaning "no blockage"
    missing_fill: str = "ffill"              # "ffill" (carry last value) or "zero" (empty edge == 0)
    # If a missing (time, edge) row means "no vehicles" (free road), set missing_fill="zero" and
    # mask_unobserved_targets=False so those free edges are trained too. Otherwise they are ignored.
    mask_unobserved_targets: bool = True

    # ---- windowing (1 step = 60 s in your SUMO data: 5400 s / 60 s = 90 steps) ----
    input_len: int = 12                      # L: 12 x 60 s = 720 s  = 12 min of history
    horizon: int = 6                         # H:  6 x 60 s = 360 s  =  6 min of predicted weights
    train_frac: float = 0.70
    val_frac: float = 0.15                   # test = the rest (latest data)

    # ---- model ----
    emb_dim: int = 16                        # size of the learned edge-identity vector
    hidden: int = 128
    layers: int = 2
    dropout: float = 0.2
    min_weight: float = 0.01                 # weights are min_weight + softplus(.)  -> always > 0
    max_weight: float = 1000.0               # upper safety clamp -> always finite
    blocked_weight: float = float("inf")     # weight given to blocked edges at inference (hard block)

    # ---- optimisation ----
    batch_size: int = 16                     # effective GRU batch is batch_size * n_edges
    lr: float = 1e-3
    weight_decay: float = 1e-5               # L2-style regularisation on the model weights (AdamW)
    epochs: int = 100
    patience: int = 10                       # early stopping
    grad_clip: float = 1.0
    huber_beta: float = 1.0
    seed: int = 42
    model_path: str = "best_traffic_gru.pt"

    # ---- traffic-cost target (see CostSpec). Set a column to None to switch that term off. ----
    # Column names are matched ignoring case/underscores ("travelTime" == "travel_time").
    travel_time_col: str | None = "traveltime"
    waiting_time_col: str | None = "waitingTime"
    time_loss_col: str | None = "timeLoss"
    density_col: str | None = "density"      # use "laneDensity" if that is what your file has
    occupancy_col: str | None = None
    speed_col: str | None = "speed"
    travel_time_weight: float = 1.0
    waiting_time_weight: float = 1.0
    time_loss_weight: float = 0.5
    density_weight: float = 0.5
    occupancy_weight: float = 0.0
    speed_weight: float = 0.5
    cost_base: float = 1.0                   # free-flow edges get cost ~ cost_base (>0)
    cost_term_clip: float = 10.0             # each normalised term is clipped to [0, clip] -> finite, no spikes

    # ---- loss ----
    loss_in_log_space: bool = True           # Huber on log(weight): relative errors, robust to big costs
    nowcast_weight: float = 0.5              # weight of the auxiliary "cost at observed steps" loss
    smoothness_weight: float = 0.01          # weight of the temporal smoothness regulariser


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------------------
# DATA PREPARATION  (unchanged from the forecasting version)
# ----------------------------------------------------------------------------------------
def to_numeric_time(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return (pd.to_datetime(s) - pd.Timestamp("1970-01-01")).dt.total_seconds()


def load_df(path):
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    drop = [c for c in df.columns if is_forbidden(c)]
    return df.drop(columns=drop)  # forbidden columns removed completely, if present


def pick_features(df, cfg):
    if cfg.feature_cols:
        feats = [c for c in cfg.feature_cols if not is_forbidden(c)]
    else:
        skip = {cfg.time_col, cfg.edge_col, cfg.scenario_col, cfg.blocked_col, cfg.blocked_edge_col}
        feats = [c for c in df.columns
                 if c not in skip and not is_forbidden(c) and pd.api.types.is_numeric_dtype(df[c])]
    if not feats:
        raise ValueError("No feature columns found - check Config.")
    return feats


def valid_block_mask(s: pd.Series, no_vals) -> pd.Series:
    return s.notna() & ~s.astype(str).str.strip().str.lower().isin(no_vals)


def build_edge_vocab(df, cfg):
    ids = set(df[cfg.edge_col].astype(str))
    if cfg.blocked_edge_col:
        m = valid_block_mask(df[cfg.blocked_edge_col], cfg.no_block_values)
        ids |= set(df.loc[m, cfg.blocked_edge_col].astype(str))
    return {e: i for i, e in enumerate(sorted(ids))}   # identifiers only, so no leakage


def build_scenarios(df, cfg, feats, edge2idx):
    """Return list of dicts with X [T,E,F] (NaN = never seen), M [T,E] observed mask, B [T,E] blockage."""
    df = df.copy()
    df["_t"] = to_numeric_time(df[cfg.time_col])
    df["_e"] = df[cfg.edge_col].astype(str).map(edge2idx)
    df["_bidx"] = np.nan
    if cfg.blocked_col:
        m = df[cfg.blocked_col].fillna(0).astype(float) > 0
        df.loc[m, "_bidx"] = df.loc[m, "_e"]
    elif cfg.blocked_edge_col:
        m = valid_block_mask(df[cfg.blocked_edge_col], cfg.no_block_values)
        df.loc[m, "_bidx"] = df.loc[m, cfg.blocked_edge_col].astype(str).map(edge2idx)
    else:
        print("WARNING: no blockage column configured -> blockage flag is all zeros; "
              "the model cannot learn blockage effects.")
    if cfg.scenario_col is None:
        df["_s"] = 0
    else:
        df["_s"] = df[cfg.scenario_col]

    E, F = len(edge2idx), len(feats)
    out = []
    for sid, sd in df.groupby("_s", sort=True):
        times = np.sort(sd["_t"].unique())
        T = len(times)
        agg = sd.groupby(["_t", "_e"])[feats].mean().reset_index()   # duplicates -> mean
        ti = np.searchsorted(times, agg["_t"].values)
        ei = agg["_e"].values.astype(int)
        X = np.full((T, E, F), np.nan, dtype=np.float32)
        X[ti, ei] = agg[feats].values
        M = np.zeros((T, E), dtype=bool)
        M[ti, ei] = True
        B = np.zeros((T, E), dtype=np.float32)
        bd = sd[sd["_bidx"].notna()]
        if len(bd):
            B[np.searchsorted(times, bd["_t"].values), bd["_bidx"].values.astype(int)] = 1.0
        X = fill_missing(X, cfg.missing_fill)
        out.append(dict(sid=sid, X=X, M=M, B=B, times=times))
    return out


def fill_missing(X, mode):
    """ffill is causal (uses only the past) so it cannot leak future values."""
    if mode == "zero":
        return np.nan_to_num(X, nan=0.0)
    T, E, F = X.shape
    return pd.DataFrame(X.reshape(T, E * F)).ffill().values.reshape(T, E, F).astype(np.float32)
    # cells still NaN (edge never observed before) become the train mean (0 after scaling)


def time_split(scen, cfg):
    """Chronological split by TARGET time. Val/test windows may use the preceding period as input
    context (past -> future, no leakage); only their targets must lie inside the val/test period."""
    L, H = cfg.input_len, cfg.horizon
    parts = {"train": [], "val": [], "test": []}
    for s in scen:
        T = len(s["X"])
        a, b = int(T * cfg.train_frac), int(T * (cfg.train_frac + cfg.val_frac))
        spans = {"train": (0, a), "val": (max(0, a - L), b), "test": (max(0, b - L), T)}
        for name, (lo, hi) in spans.items():
            seg = {k: s[k][lo:hi] for k in ("X", "M", "B")}
            if len(seg["X"]) >= L + H:
                parts[name].append(seg)
    for k, v in parts.items():
        if not v:
            raise ValueError(
                f"'{k}' split too short. Scenario lengths (time steps): {[len(s['X']) for s in scen]}, "
                f"need train >= {L + H}, val/test >= {H} target steps. Reduce input_len/horizon, "
                f"use a finer sampling period, or add more runs.")
    return parts


class Scaler:
    """Per-feature standardisation fitted on TRAIN observed cells only."""
    def fit(self, segs):
        rows = np.concatenate([s["X"][s["M"]] for s in segs], axis=0)
        self.mean = np.nanmean(rows, axis=0).astype(np.float32)
        self.std = np.maximum(np.nanstd(rows, axis=0), 1e-6).astype(np.float32)
        return self

    def transform(self, X):
        return np.nan_to_num((X - self.mean) / self.std, nan=0.0).astype(np.float32)

    def inverse(self, Xs):
        return Xs * self.std + self.mean


# ----------------------------------------------------------------------------------------
# TRAFFIC COST  (the training target - defined from features, NOT a ground-truth optimum)
# ----------------------------------------------------------------------------------------
# (Config column attr, Config coefficient attr, kind)
#   "ratio": bigger value = worse traffic  -> term = value / train_mean(value), clipped to [0, clip]
#   "speed": bigger value = better traffic -> term = 1 - speed / free_flow_speed(edge), clipped to [0, 1]
COST_TERMS = [
    ("travel_time_col",  "travel_time_weight",  "ratio"),
    ("waiting_time_col", "waiting_time_weight", "ratio"),
    ("time_loss_col",    "time_loss_weight",    "ratio"),
    ("density_col",      "density_weight",      "ratio"),
    ("occupancy_col",    "occupancy_weight",    "ratio"),
    ("speed_col",        "speed_weight",        "speed"),
]


class CostSpec:
    """Builds traffic_cost [T,E] from ORIGINAL-unit features [T,E,F]. Statistics are fitted on TRAIN only.

    - Travel time is divided by ONE global train mean (not per edge) on purpose: a long edge really is
      more costly to traverse than a short one, and per-edge normalisation would erase that.
    - Free-flow speed of an edge = 95th percentile of its own train speeds (fallback: global 95th pct).
    - Empty edges (density == 0) often report speed 0/-1 in SUMO; that is NOT congestion, so the speed
      term is set to 0 for them when a density column exists.
    """
    def __init__(self, cfg, feats):
        lookup = {norm_name(f): i for i, f in enumerate(feats)}
        self.base, self.clip = float(cfg.cost_base), float(cfg.cost_term_clip)
        self.terms = []
        for col_attr, w_attr, kind in COST_TERMS:
            col, coef = getattr(cfg, col_attr), getattr(cfg, w_attr)
            if col is None or coef == 0:
                continue
            j = lookup.get(norm_name(col))
            if j is None:
                print(f"NOTE: cost term '{col_attr}'='{col}' not found among features {feats} -> term skipped.")
                continue
            self.terms.append(dict(name=feats[j], idx=j, coef=float(coef), kind=kind))
        if not self.terms:
            raise ValueError("No traffic-cost term could be built: check Config *_col names and *_weight values.")
        self.empty_idx = lookup.get(norm_name(cfg.density_col)) if cfg.density_col else None

    def fit(self, segs):
        X = np.concatenate([s["X"] for s in segs]); M = np.concatenate([s["M"] for s in segs])   # [T,E,F],[T,E]
        occupied = (X[..., self.empty_idx] > 0) if self.empty_idx is not None else np.ones_like(M)
        for t in self.terms:
            v = X[..., t["idx"]]
            if t["kind"] == "ratio":
                m = np.nanmean(np.clip(v[M], 0, None))
                t["scale"] = float(m) if np.isfinite(m) and m > 1e-6 else 1.0
            else:
                sp = np.where(M & occupied, v, np.nan)                                          # [T,E]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ref = np.nanpercentile(sp, 95, axis=0)                                      # [E]
                    glob = np.nanpercentile(sp, 95) if np.isfinite(sp).any() else 1.0
                ref = np.where(np.isfinite(ref), ref, glob)
                t["ref"] = np.maximum(ref, 1e-6).tolist()
        return self

    def __call__(self, X):
        """X: [T,E,F] original units -> cost [T,E]  (>= cost_base, finite)."""
        X = np.nan_to_num(X, nan=0.0)
        cost = np.full(X.shape[:2], self.base, dtype=np.float32)
        for t in self.terms:
            v = X[..., t["idx"]]
            if t["kind"] == "ratio":
                term = np.clip(v / t["scale"], 0.0, self.clip)
            else:
                term = np.clip(1.0 - v / np.asarray(t["ref"], np.float32)[None, :], 0.0, 1.0)
                if self.empty_idx is not None:
                    term = np.where(X[..., self.empty_idx] <= 0, 0.0, term)
            cost += t["coef"] * term.astype(np.float32)
        return cost

    def to_dict(self):
        return dict(base=self.base, clip=self.clip, terms=self.terms, empty_idx=self.empty_idx)

    @classmethod
    def from_dict(cls, d):
        o = cls.__new__(cls)
        o.base, o.clip, o.terms, o.empty_idx = d["base"], d["clip"], d["terms"], d["empty_idx"]
        return o


class WindowDS(Dataset):
    """Sliding windows inside each segment only (never across scenario/split boundaries).
    Each item is a dict (shapes without the batch dimension):
        x [L,E,F] scaled features        bx [L,E] blocked history      by [H,E] blocked future
        c_hist [L,E] traffic cost at observed steps   c_fut [H,E] traffic cost at future steps (TARGET)
        m_hist [L,E] / m_fut [H,E] observed masks     y_feat [H,E,F] scaled future features (evaluation only)
    """
    def __init__(self, segs, L, H):
        self.L, self.H = L, H
        self.X = [torch.from_numpy(s["X"]) for s in segs]
        self.M = [torch.from_numpy(s["M"]) for s in segs]
        self.B = [torch.from_numpy(s["B"]) for s in segs]
        self.C = [torch.from_numpy(s["C"]) for s in segs]
        self.idx = [(i, t) for i, s in enumerate(self.X) for t in range(len(s) - L - H + 1)]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i, t = self.idx[k]; L, H = self.L, self.H
        return dict(x=self.X[i][t:t + L], bx=self.B[i][t:t + L], by=self.B[i][t + L:t + L + H],
                    c_hist=self.C[i][t:t + L], c_fut=self.C[i][t + L:t + L + H],
                    m_hist=self.M[i][t:t + L].float(), m_fut=self.M[i][t + L:t + L + H].float(),
                    y_feat=self.X[i][t + L:t + L + H])


# ----------------------------------------------------------------------------------------
# MODEL
# ----------------------------------------------------------------------------------------
class WeightHead(nn.Module):
    """hidden state [..., hidden] -> ONE raw score [...]; the caller turns it into a positive weight."""
    def __init__(self, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden // 2, 1))

    def forward(self, h):
        return self.net(h).squeeze(-1)


class EdgeWeightGRU(nn.Module):
    """Many-to-many encoder/decoder GRU.
        encoder GRU reads the L history steps of every edge  -> hidden state at EVERY step + final state
        decoder GRU unrolls H future steps starting from the encoder's final state
        weight head turns EVERY GRU output step into a positive edge weight
    Edges are processed as independent sequences (batch dimension B*E); the learned edge embedding lets the
    shared GRU behave differently per edge.  NOTE: an edge does not "see" its neighbours' traffic."""
    def __init__(self, n_edges, n_feat, horizon, emb_dim=16, hidden=128, layers=2, dropout=0.2,
                 min_weight=0.01, max_weight=1000.0):
        super().__init__()
        self.E, self.F, self.H = n_edges, n_feat, horizon
        self.min_weight, self.max_weight = min_weight, max_weight
        self.edge_emb = nn.Embedding(n_edges, emb_dim)       # learned identity of each edge
        self.blk_emb = nn.Embedding(2, 4)                    # embedding of the 0/1 blockage flag
        gru_kw = dict(input_size=n_feat + 4 + emb_dim, hidden_size=hidden, num_layers=layers,
                      batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.encoder = nn.GRU(**gru_kw)                      # reads history
        self.decoder = nn.GRU(**gru_kw)                      # produces the future sequence
        self.now_head = WeightHead(hidden, dropout)          # weights at the observed steps (auxiliary)
        self.fut_head = WeightHead(hidden, dropout)          # weights at the future steps (primary)

    def to_weight(self, raw):
        # softplus(raw) > 0 and smooth; + min_weight keeps it strictly positive; clamp keeps it finite.
        return (self.min_weight + Fn.softplus(raw)).clamp(max=self.max_weight)

    def forward(self, x, bx, by):
        # x: [B, L, E, F]  B=batch, L=historical 60-s steps, E=edges, F=traffic features (scaled)
        # bx: [B, L, E] blocked flags in history;  by: [B, H, E] blocked flags in the future
        B, L, E, F = x.shape
        H = by.shape[1]
        emb = self.edge_emb(torch.arange(E, device=x.device))                          # [E, D]

        # ---- encoder: history -> hidden states ----
        z = torch.cat([x,
                       self.blk_emb(bx.long()),                                        # [B, L, E, 4]
                       emb[None, None].expand(B, L, E, -1)], dim=-1)                   # [B, L, E, F+4+D]
        z = z.permute(0, 2, 1, 3).reshape(B * E, L, -1)                                # [B*E, L, F+4+D]
        enc_out, h_n = self.encoder(z)                # enc_out [B*E, L, hidden]; h_n [layers, B*E, hidden]

        # auxiliary output: weight at every OBSERVED step ("traffic state at t -> cost at t")
        w_now = self.to_weight(self.now_head(enc_out))                                 # [B*E, L]
        w_now = w_now.view(B, E, L).permute(0, 2, 1)                                   # [B, L, E]

        # ---- decoder: future steps. Input per step = last observed features + future blockage flag + edge id
        last = x[:, -1].reshape(B * E, 1, F).expand(-1, H, -1)                         # [B*E, H, F]
        blk = self.blk_emb(by.long()).permute(0, 2, 1, 3).reshape(B * E, H, 4)         # [B*E, H, 4]
        emb_h = emb[None, :, None].expand(B, E, H, -1).reshape(B * E, H, -1)           # [B*E, H, D]
        dec_out, _ = self.decoder(torch.cat([last, blk, emb_h], dim=-1), h_n)          # [B*E, H, hidden]

        # primary output: weight at every FUTURE step
        w_fut = self.to_weight(self.fut_head(dec_out))                                 # [B*E, H]
        w_fut = w_fut.view(B, E, H).permute(0, 2, 1)                                   # [B, H, E]
        return w_fut, w_now


# ----------------------------------------------------------------------------------------
# LOSS
#   total = weight_prediction_loss(future) + nowcast_weight * weight_prediction_loss(observed steps)
#           + smoothness_weight * temporal_smoothness_loss
# ----------------------------------------------------------------------------------------
def masked_huber(pred, target, mask, beta):
    """Huber loss (quadratic for small errors, linear for big ones -> robust to congestion spikes),
    averaged only over valid cells (mask=1)."""
    l = Fn.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    return (l * mask).sum() / mask.sum().clamp(min=1.0)


def temporal_smoothness(w, mask):
    """mean((W_t - W_(t-1))^2) over consecutive steps where both cells are valid.
    w: [B, S, E] (S = time steps).  It discourages the predicted weights from jumping between
    consecutive 60-s steps unless the data-driven loss above says the traffic really changed."""
    if w.shape[1] < 2:
        return w.new_zeros(())
    d = w[:, 1:] - w[:, :-1]
    m = mask[:, 1:] * mask[:, :-1]
    return (d ** 2 * m).sum() / m.sum().clamp(min=1.0)


def compute_loss(w_fut, w_now, batch, cfg):
    # Blocked cells are excluded: their traffic says nothing about routing cost (routing hard-blocks them).
    mf = batch["m_fut"] * (1.0 - batch["by"])                                          # [B, H, E]
    mh = batch["m_hist"] * (1.0 - batch["bx"])                                         # [B, L, E]
    sp = (lambda t: torch.log(t.clamp(min=1e-6))) if cfg.loss_in_log_space else (lambda t: t)
    pf, pn = sp(w_fut), sp(w_now)
    fut = masked_huber(pf, sp(batch["c_fut"]), mf, cfg.huber_beta)                     # future weights vs target cost
    now = masked_huber(pn, sp(batch["c_hist"]), mh, cfg.huber_beta)                    # observed-step weights vs cost
    smooth = temporal_smoothness(pf, mf)                                               # stability of future weights
    total = fut + cfg.nowcast_weight * now + cfg.smoothness_weight * smooth
    return total, dict(total=total.item(), future=fut.item(), nowcast=now.item(), smooth=smooth.item())


# ----------------------------------------------------------------------------------------
# TRAIN
# ----------------------------------------------------------------------------------------
def to_device(batch):
    return {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}


def run_epoch(model, loader, cfg, opt=None):
    train = opt is not None
    model.train(train)
    sums, n = {}, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = to_device(batch)
            w_fut, w_now = model(batch["x"], batch["bx"], batch["by"])
            loss, parts = compute_loss(w_fut, w_now, batch, cfg)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
            bs = batch["x"].size(0)
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v * bs
            n += bs
    return {k: v / max(n, 1) for k, v in sums.items()}


# ----------------------------------------------------------------------------------------
# EVALUATION
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def collect(model, loader):
    """W [N,H,E] predicted weights, C [N,H,E] target cost, M [N,H,E] valid mask (observed & not blocked),
    LAST [N,E] cost at the last observed step (persistence baseline), YF [N,H,E,F] scaled future features."""
    model.eval()
    out = {k: [] for k in ("W", "C", "M", "LAST", "YF")}
    for b in loader:
        w_fut, _ = model(b["x"].to(DEVICE), b["bx"].to(DEVICE), b["by"].to(DEVICE))
        out["W"].append(w_fut.cpu().numpy())
        out["C"].append(b["c_fut"].numpy())
        out["M"].append(((b["m_fut"] * (1 - b["by"])) > 0.5).numpy())
        out["LAST"].append(b["c_hist"][:, -1].numpy())
        out["YF"].append(b["y_feat"].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def spearman(a, b):
    """Rank correlation (does a bigger value of a go with a bigger value of b?). nan if undefined."""
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(pd.Series(a).rank().values, pd.Series(b).rank().values)[0, 1])


def reg_metrics(p, y):
    err = p - y
    mse = float(np.mean(err ** 2)); sst = float(np.sum((y - y.mean()) ** 2))
    return dict(MAE=float(np.mean(np.abs(err))), RMSE=mse ** 0.5,
                R2=float(1 - np.sum(err ** 2) / sst) if sst > 0 else float("nan"), Spearman=spearman(p, y))


def step_change(P, M):
    """Mean |W_t - W_(t-1)| between consecutive prediction steps (valid pairs only)."""
    if P.shape[1] < 2:
        return float("nan")
    m = M[:, 1:] & M[:, :-1]
    return float(np.abs(P[:, 1:] - P[:, :-1])[m].mean()) if m.any() else float("nan")


def evaluate_split(name, model, loader, scaler, cost, edge_mean_cost):
    R = collect(model, loader)
    W, C, M, LAST, YF = R["W"], R["C"], R["M"], R["LAST"], R["YF"]
    H = W.shape[1]
    preds = {
        "GRU": W,
        "BaselineA persistence (manual cost at last observed step)": np.repeat(LAST[:, None], H, axis=1),
        "BaselineB static (train-mean manual cost per edge)": np.broadcast_to(edge_mean_cost[None, None, :], W.shape),
    }
    res = {"weight_stats": {}, "accuracy": {}, "stability": {}, "correlation": {}}
    print(f"\n===== {name} (valid cells: {int(M.sum())}) =====")
    if not M.any():
        print("No valid cells to evaluate."); return res

    w = W[M]
    res["weight_stats"] = dict(min=float(w.min()), max=float(w.max()), mean=float(w.mean()), std=float(w.std()))
    print("GRU weight stats:     " + ", ".join(f"{k}={v:.4f}" for k, v in res["weight_stats"].items()))
    print(f"target cost stats:    min={C[M].min():.4f}, max={C[M].max():.4f}, mean={C[M].mean():.4f}, std={C[M].std():.4f}")

    print("\n-- Agreement with the future traffic cost (cost units; Spearman = rank agreement) --")
    for tag, p in preds.items():
        res["accuracy"][tag] = reg_metrics(p[M], C[M])
        print(f"[{tag}] " + ", ".join(f"{k}={v:.4f}" for k, v in res["accuracy"][tag].items()))

    print("\n-- Temporal stability: mean |change| between consecutive steps within the horizon --")
    res["stability"]["GRU"] = step_change(W, M)
    res["stability"]["target cost (how much traffic really changes)"] = step_change(C, M)
    for k, v in res["stability"].items():
        print(f"{k}: {v:.4f}")
    print("(constant baselines change by 0 by construction; that is stability without any adaptation)")

    print("\n-- Rank correlation of weight with measured traffic (original units, target steps) --")
    print(f"{'feature':<16}{'expected':>9}{'GRU':>9}{'BaseA':>9}{'target':>9}")
    feat = scaler.inverse(YF)                                                      # [N,H,E,F]
    for t in cost.terms:
        v = feat[..., t["idx"]][M]
        row = dict(GRU=spearman(W[M], v), BaseA=spearman(preds[list(preds)[1]][M], v), target=spearman(C[M], v))
        res["correlation"][t["name"]] = row
        print(f"{t['name']:<16}{('-' if t['kind'] == 'speed' else '+'):>9}"
              f"{row['GRU']:>9.3f}{row['BaseA']:>9.3f}{row['target']:>9.3f}")

    g, a = res["accuracy"]["GRU"]["RMSE"], res["accuracy"][list(preds)[1]]["RMSE"]
    print(f"\nVerdict on RMSE vs future cost: GRU {g:.4f} vs persistence {a:.4f} -> "
          + ("GRU lower (better)." if g < a else "GRU NOT better than persistence."))
    return res


# ----------------------------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------------------------
def train_pipeline(cfg: Config):
    set_seed(cfg.seed)
    df = load_df(cfg.csv_path)
    feats = pick_features(df, cfg)
    print("Device:", DEVICE, "| features used:", feats)
    edge2idx = build_edge_vocab(df, cfg)
    scen = build_scenarios(df, cfg, feats, edge2idx)
    parts = time_split(scen, cfg)

    scaler = Scaler().fit(parts["train"])                                # TRAIN ONLY
    cost = CostSpec(cfg, feats).fit(parts["train"])                      # TRAIN ONLY (normalisation constants)
    print("Traffic-cost terms:", [(t["name"], t["coef"], t["kind"]) for t in cost.terms])

    for segs in parts.values():                                          # target cost from ORIGINAL units
        for s in segs:
            s["C"] = cost(s["X"])                                        # [T, E]
    num = sum((s["C"] * s["M"]).sum(0) for s in parts["train"])
    den = sum(s["M"].sum(0) for s in parts["train"])
    edge_mean_cost = np.where(den > 0, num / np.maximum(den, 1), cost.base).astype(np.float32)   # [E] baseline B

    for segs in parts.values():
        for s in segs:
            s["X"] = scaler.transform(s["X"])
            if not cfg.mask_unobserved_targets:
                s["M"] = np.ones_like(s["M"])

    dl = {k: DataLoader(WindowDS(v, cfg.input_len, cfg.horizon), batch_size=cfg.batch_size,
                        shuffle=(k == "train"), pin_memory=DEVICE.type == "cuda")
          for k, v in parts.items()}                                     # shuffling WINDOWS in train is fine
    print({k: len(v.dataset) for k, v in dl.items()}, "windows | edges:", len(edge2idx))

    kw = dict(n_edges=len(edge2idx), n_feat=len(feats), horizon=cfg.horizon, emb_dim=cfg.emb_dim,
              hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout,
              min_weight=cfg.min_weight, max_weight=cfg.max_weight)
    model = EdgeWeightGRU(**kw).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    best, bad, best_state = float("inf"), 0, None
    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch(model, dl["train"], cfg, opt)
        va = run_epoch(model, dl["val"], cfg)
        sched.step(va["total"])
        print(f"epoch {ep:3d} | train {tr['total']:.5f} (fut {tr['future']:.4f} now {tr['nowcast']:.4f} "
              f"smooth {tr['smooth']:.4f}) | val {va['total']:.5f} | lr {opt.param_groups[0]['lr']:.2e}")
        if va["total"] < best - 1e-6:
            best, bad, best_state = va["total"], 0, copy.deepcopy(model.state_dict())
            torch.save(dict(state_dict=best_state, model_kwargs=kw, features=feats, edge2idx=edge2idx,
                            scaler_mean=scaler.mean.tolist(), scaler_std=scaler.std.tolist(),
                            cost_spec=cost.to_dict(), edge_mean_cost=edge_mean_cost.tolist(),
                            cfg=asdict(cfg)), cfg.model_path)
        else:
            bad += 1
            if bad >= cfg.patience:
                print("Early stopping."); break

    model.load_state_dict(best_state)
    evaluate_split("VALIDATION", model, dl["val"], scaler, cost, edge_mean_cost)
    evaluate_split("TEST", model, dl["test"], scaler, cost, edge_mean_cost)
    print(f"\nBest model saved to {cfg.model_path}")
    return model


# ----------------------------------------------------------------------------------------
# LOAD + INFERENCE
# ----------------------------------------------------------------------------------------
def load_model(path, device=DEVICE):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = EdgeWeightGRU(**ck["model_kwargs"]).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, ck


def predict_edge_weights(history_df, model_path="best_traffic_gru.pt", horizon=None,
                         blocked_edges=(), mark_history_blocked=False, device=DEVICE):
    """
    history_df   : raw DataFrame (ONE scenario) with the same time/edge/feature columns as training,
                   containing at least `input_len` distinct timestamps. Only the latest input_len are used.
    horizon      : number of future steps to return (<= the horizon the model was trained with).
    blocked_edges: edge ids blocked over the prediction horizon. They enter the model as input AND are
                   hard-blocked in the output (weight = cfg.blocked_weight, default inf) - the routing layer
                   must not rely on the network to avoid them.
    Returns dict:
        "weights"        np.ndarray [H, E]  routing weights (blocked edges = inf)
        "weights_model"  np.ndarray [H, E]  raw model output before hard-blocking
        "times"           future timestamps [H]     "edge_ids" list of E ids (column order of "weights")
        "blocked_edge_ids" list                     "dataframe" long table: step, time, edge_id, weight, blocked
    """
    model, ck = load_model(model_path, device)
    cfg = Config(**{k: (tuple(v) if k == "no_block_values" else v) for k, v in ck["cfg"].items()})
    feats, edge2idx = ck["features"], ck["edge2idx"]
    mean, std = np.array(ck["scaler_mean"], np.float32), np.array(ck["scaler_std"], np.float32)
    L, Hm, E, F = cfg.input_len, cfg.horizon, len(edge2idx), len(feats)
    H = Hm if horizon is None else min(horizon, Hm)

    d = history_df.copy()
    d = d.drop(columns=[c for c in d.columns if is_forbidden(c)])
    d["_t"] = to_numeric_time(d[cfg.time_col])
    d["_e"] = d[cfg.edge_col].astype(str).map(edge2idx)
    d = d[d["_e"].notna()]
    times = np.sort(d["_t"].unique())
    if len(times) < L:
        raise ValueError(f"Need at least {L} timestamps of history, got {len(times)}.")
    times = times[-L:]
    d = d[d["_t"].isin(times)]
    agg = d.groupby(["_t", "_e"])[feats].mean().reset_index()
    X = np.full((L, E, F), np.nan, np.float32)
    X[np.searchsorted(times, agg["_t"].values), agg["_e"].values.astype(int)] = agg[feats].values
    X = fill_missing(X, cfg.missing_fill)
    Xs = np.nan_to_num((X - mean) / std, nan=0.0).astype(np.float32)             # [L, E, F]

    bx = np.zeros((L, E), np.float32); by = np.zeros((Hm, E), np.float32)
    blocked_idx = []
    for e in blocked_edges:
        if str(e) not in edge2idx:
            raise ValueError(f"Unknown blocked edge id '{e}'.")
        idx = edge2idx[str(e)]
        blocked_idx.append(idx)
        by[:, idx] = 1.0
        if mark_history_blocked:
            bx[:, idx] = 1.0

    with torch.no_grad():
        w_fut, _ = model(torch.from_numpy(Xs)[None].to(device), torch.from_numpy(bx)[None].to(device),
                         torch.from_numpy(by)[None].to(device))                  # [1, Hm, E]
    w_model = w_fut[0, :H].cpu().numpy().astype(np.float64)                      # [H, E]
    w = w_model.copy()
    w[:, blocked_idx] = cfg.blocked_weight                                       # HARD block

    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    ftimes = times[-1] + dt * np.arange(1, H + 1)
    inv = {i: e for e, i in edge2idx.items()}
    edge_ids = [inv[i] for i in range(E)]
    blocked_ids = [inv[i] for i in sorted(set(blocked_idx))]
    long = pd.DataFrame({"step": np.repeat(np.arange(1, H + 1), E), "time": np.repeat(ftimes, E),
                         "edge_id": edge_ids * H, "weight": w.reshape(-1),
                         "blocked": np.tile(np.isin(np.arange(E), blocked_idx), H)})
    return dict(weights=w, weights_model=w_model, times=ftimes, edge_ids=edge_ids,
                blocked_edge_ids=blocked_ids, dataframe=long)


def edge_weight_dict(out, step=1, drop_blocked=False):
    """{edge_id: weight} for prediction step `step` (1 = first future step, i.e. +60 s).
    drop_blocked=True omits blocked edges - use it if your router treats a missing edge as unusable
    (safest for networkx/Dijkstra, since an 'inf' edge could still be returned if no other path exists)."""
    blocked = set(out["blocked_edge_ids"]) if drop_blocked else set()
    return {e: float(w) for e, w in zip(out["edge_ids"], out["weights"][step - 1]) if e not in blocked}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    CFG = Config()   # <-- set time_col, edge_col, scenario_col, blocked_col / blocked_edge_col, cost columns here
    if a.csv: CFG.csv_path = a.csv
    if a.epochs: CFG.epochs = a.epochs
    train_pipeline(CFG)

    # Example inference (uncomment and adapt):
    # hist = pd.read_csv("some_recent_window.csv")
    # out = predict_edge_weights(hist, model_path=CFG.model_path, blocked_edges=["edge_42"])
    # w = edge_weight_dict(out, step=1, drop_blocked=True)      # {edge_id: weight} for Dijkstra / A*
    # out["weights"]      -> [H, E] array      out["dataframe"] -> time | edge_id | weight table