from __future__ import annotations

import asyncio
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL_PATH = Path(os.getenv("GRU_MODEL_PATH", str(ROOT / "best_traffic_gru.pt")))
HISTORY_PATH = Path(os.getenv("TRAFFIC_HISTORY_CSV", str(ROOT / "data" / "demo_traffic.csv")))
GRAPH_PATH = Path(os.getenv("GRAPH_JSON", str(ROOT / "data" / "graph.json")))

from repo import DisruptionEngine

try:
    from traffic_gru_new import predict_edge_weights
    GRU_IMPORT_ERROR = ""
except Exception as exc:
    predict_edge_weights = None
    GRU_IMPORT_ERROR = str(exc)

app = FastAPI(title="GRU Traffic Routing Simulation", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Graph / model data
# -----------------------------

def load_graph(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    data = pd.read_json(path) if False else __import__("json").loads(path.read_text())
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    graph: dict[str, list[tuple[str, float]]] = {str(n["id"]): [] for n in nodes}
    edge_by_id: dict[str, dict[str, Any]] = {}
    pair_to_edge: dict[tuple[str, str], str] = {}
    for e in edges:
        eid = str(e["id"])
        u, v = str(e["u"]), str(e["v"])
        w = float(e.get("weight", 1.0))
        graph.setdefault(u, []).append((v, w))
        if e.get("bidirectional", True):
            graph.setdefault(v, []).append((u, w))
        edge_by_id[eid] = {**e, "id": eid, "u": u, "v": v}
        pair_to_edge[(u, v)] = eid
        if e.get("bidirectional", True):
            pair_to_edge[(v, u)] = eid
    return graph, nodes, edges, edge_by_id, pair_to_edge


GRAPH, GRAPH_NODES, GRAPH_EDGES, EDGE_BY_ID, PAIR_TO_EDGE = load_graph(GRAPH_PATH)
BASE_GRAPH = {n: list(edges) for n, edges in GRAPH.items()}
DISRUPTIONS = DisruptionEngine(GRAPH)

try:
    HISTORY = pd.read_csv(HISTORY_PATH)
except Exception as exc:
    raise RuntimeError(f"Could not load traffic history {HISTORY_PATH}: {exc}")

if predict_edge_weights is None:
    raise RuntimeError(f"Could not import GRU inference: {GRU_IMPORT_ERROR}")
if not MODEL_PATH.exists():
    raise RuntimeError(f"GRU checkpoint not found: {MODEL_PATH}")

# -----------------------------
# Simulation state
# -----------------------------
RUNNING = False
SIMULATION_TIME = 0
PREDICTIONS: list[dict[str, Any]] = []
EDGE_WEIGHTS: dict[str, float] = {}
ROUTES: list[dict[str, Any]] = []
CARS: list[dict[str, Any]] = []
DISRUPTION_FACTORS: dict[tuple[str, str], float] = {}
LAST_PREDICTION_TICK = -1
PREDICTION_INTERVAL = int(os.getenv("PREDICTION_INTERVAL_SECONDS", "10"))
NUM_CARS = int(os.getenv("NUM_CARS", "24"))
RNG = random.Random(42)


def edge_is_active(u: str, v: str) -> bool:
    return any(n == v for n, _ in GRAPH.get(u, []))


def active_disruptions() -> list[dict[str, Any]]:
    result = []
    for (u, v), kind in DISRUPTIONS.active_disruptions.items():
        result.append({
            "u": str(u), "v": str(v), "type": kind,
            "factor": DISRUPTION_FACTORS.get((u, v), 1.0),
        })
    return result


def graph_for_frontend() -> dict[str, Any]:
    blocked_pairs = {
        frozenset((str(u), str(v)))
        for (u, v), kind in DISRUPTIONS.active_disruptions.items()
        if kind == "blocked"
    }
    edges = []
    for e in GRAPH_EDGES:
        item = dict(e)
        item["blocked"] = frozenset((str(e["u"]), str(e["v"]))) in blocked_pairs
        tw = EDGE_WEIGHTS.get(str(e["id"]), float(e.get("weight", 1.0)))
        item["traffic_weight"] = float(tw) if math.isfinite(float(tw)) else None
        edges.append(item)
    return {"nodes": GRAPH_NODES, "edges": edges}


def blocked_model_edges() -> list[str]:
    ids = []
    for (u, v), kind in DISRUPTIONS.active_disruptions.items():
        if kind != "blocked":
            continue
        for pair in ((str(u), str(v)), (str(v), str(u))):  # blocking removes both directions
            eid = PAIR_TO_EDGE.get(pair)
            if eid:
                ids.append(eid)
    return ids


def refresh_predictions() -> None:
    global PREDICTIONS, EDGE_WEIGHTS, LAST_PREDICTION_TICK
    out = predict_edge_weights(
        HISTORY,
        model_path=str(MODEL_PATH),
        horizon=6,
        blocked_edges=blocked_model_edges(),
        mark_history_blocked=False,
    )
    # First predicted minute is the routing cost used by the simulation.
    first = {str(e): float(w) for e, w in zip(out["edge_ids"], out["weights"][0])}
    EDGE_WEIGHTS = first
    for (u, v), kind in DISRUPTIONS.active_disruptions.items():
        if kind == "blocked":
            continue
        eid = PAIR_TO_EDGE.get((str(u), str(v)))
        if eid:
            EDGE_WEIGHTS[eid] *= DISRUPTION_FACTORS.get((u, v), 1.0)

    rows = out["dataframe"]
    # Dashboard shows a compact sample, while the complete weight map remains server-side.
    sample = []
    for _, row in rows.iterrows():
        if len(sample) >= 24:
            break
        sample.append({
            "edge_id": str(row["edge_id"]),
            "step": int(row["step"]),
            "weight": None if not math.isfinite(float(row["weight"])) else float(row["weight"]),
            "blocked": bool(row["blocked"]),
        })
    PREDICTIONS = sample
    LAST_PREDICTION_TICK = SIMULATION_TIME


def routing_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(GRAPH_NODES and [str(n["id"]) for n in GRAPH_NODES] or list(BASE_GRAPH))
    for u, edges in GRAPH.items():
        for v, _base in edges:
            eid = PAIR_TO_EDGE.get((u, v))
            if not eid:
                continue
            weight = EDGE_WEIGHTS.get(eid, 1.0)
            if not math.isfinite(weight):
                continue
            g.add_edge(u, v, weight=weight, edge_id=eid)
    return g


def calculate_routes() -> None:
    global ROUTES
    g = routing_graph()
    nodes = list(g.nodes)
    if len(nodes) < 2:
        ROUTES = []
        return
    routes = []
    for car in CARS:
        try:
            path = nx.shortest_path(g, car["source"], car["destination"], weight="weight")
            cost = nx.path_weight(g, path, weight="weight")
        except nx.NetworkXNoPath:
            path, cost = [], float("inf")
        car["route"] = path
        car["route_cost"] = None if not math.isfinite(cost) else float(cost)
        routes.append({
            "car_id": car["id"], "source": car["source"],
            "destination": car["destination"], "route": path,
            "cost": car["route_cost"],
        })
    ROUTES = routes


def initialise_cars() -> None:
    global CARS
    # Cars only use the largest strongly connected part of the road network,
    # so every source can always reach every destination (roads are one-way).
    dg = nx.DiGraph([(u, v) for u, es in BASE_GRAPH.items() for v, _ in es])
    nodes = sorted(max(nx.strongly_connected_components(dg), key=len)) if dg.number_of_nodes() else []
    CARS = []
    if len(nodes) < 2:
        return
    for i in range(NUM_CARS):
        source, destination = RNG.sample(nodes, 2)
        CARS.append({"id": f"car-{i+1}", "source": source, "destination": destination,
                     "route": [], "route_cost": None})


def stats() -> dict[str, float | int]:
    finite = [float(c["route_cost"]) for c in CARS if c["route_cost"] is not None]
    current = (sum(finite) / len(finite)) if finite else 0.0
    # The model's output is a routing cost, not necessarily seconds. We therefore
    # label the dashboard values as simulation estimates rather than claiming they
    # are measured travel times.
    normal = current
    factors = [d["factor"] for d in active_disruptions() if d["type"] != "blocked"]
    if factors:
        normal = current / max(1.0, sum(factors) / len(factors))
    delay = max(0.0, current - normal)
    return {"cars": len(CARS), "normal_travel_time": normal,
            "current_travel_time": current, "average_delay": delay}


def state() -> dict[str, Any]:
    return {
        "status": "running" if RUNNING else "paused" if SIMULATION_TIME else "stopped",
        "simulation_time": SIMULATION_TIME,
        "stats": stats(),
        "disruptions": active_disruptions(),
        "predictions": PREDICTIONS,
        "routes": ROUTES,
        "graph": graph_for_frontend(),
        "ml": {
            "model_loaded": MODEL_PATH.exists(),
            "history_loaded": HISTORY_PATH.exists(),
            "history_rows": len(HISTORY),
            "prediction_interval": PREDICTION_INTERVAL,
            "last_prediction_tick": LAST_PREDICTION_TICK,
            "model_import_error": GRU_IMPORT_ERROR,
        },
    }


def tick() -> None:
    global SIMULATION_TIME
    SIMULATION_TIME += 1
    if LAST_PREDICTION_TICK < 0 or SIMULATION_TIME - LAST_PREDICTION_TICK >= PREDICTION_INTERVAL:
        refresh_predictions()
    calculate_routes()


class DisruptionRequest(BaseModel):
    type: str
    u: str
    v: str
    factor: float | None = Field(default=None, gt=0)


class RouteRequest(BaseModel):
    source: str
    destination: str


@app.on_event("startup")
async def startup() -> None:
    initialise_cars()
    refresh_predictions()
    calculate_routes()


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": MODEL_PATH.exists(),
        "history": HISTORY_PATH.exists(),
        "graph": bool(GRAPH),
        "edges": len(EDGE_BY_ID),
    }


@app.get("/api/state")
def get_state():
    return state()


@app.post("/api/simulation/{command}")
def simulation_command(command: str):
    global RUNNING, SIMULATION_TIME, LAST_PREDICTION_TICK
    if command == "start":
        RUNNING = True
        if not EDGE_WEIGHTS:
            refresh_predictions()
        calculate_routes()
    elif command == "pause":
        RUNNING = False
    elif command == "reset":
        RUNNING = False
        SIMULATION_TIME = 0
        LAST_PREDICTION_TICK = -1
        DISRUPTIONS.restore_all()
        DISRUPTION_FACTORS.clear()
        initialise_cars()
        refresh_predictions()
        calculate_routes()
    else:
        raise HTTPException(status_code=400, detail="Unknown command. Use start, pause, or reset.")
    return state()


@app.post("/api/predictions/refresh")
def refresh_prediction_endpoint():
    refresh_predictions()
    calculate_routes()
    return state()


@app.post("/api/disruptions")
def create_disruption(request: DisruptionRequest):
    kind = request.type.lower()
    key = (request.u, request.v)
    factor = request.factor or ({"accident": 3.0, "construction": 2.0}.get(kind, 1.0))
    if kind == "blocked":
        ok = DISRUPTIONS.block_road(request.u, request.v)
    elif kind == "accident":
        ok = DISRUPTIONS.add_accident(request.u, request.v, factor)
    elif kind == "construction":
        ok = DISRUPTIONS.add_construction(request.u, request.v, factor)
    else:
        raise HTTPException(status_code=400, detail="type must be blocked, accident, or construction")
    if not ok:
        raise HTTPException(status_code=404, detail="Road was not found in the loaded graph")
    DISRUPTION_FACTORS[key] = factor
    if (request.v, request.u) in DISRUPTIONS.active_disruptions:
        DISRUPTION_FACTORS[(request.v, request.u)] = factor
    refresh_predictions()
    calculate_routes()
    return state()


@app.delete("/api/disruptions/{u}/{v}")
def restore_disruption(u: str, v: str):
    if not DISRUPTIONS.restore_road(u, v):
        raise HTTPException(status_code=404, detail="No active disruption for that road")
    DISRUPTION_FACTORS.pop((u, v), None)
    DISRUPTION_FACTORS.pop((v, u), None)
    refresh_predictions()
    calculate_routes()
    return state()


@app.post("/api/route")
def route(request: RouteRequest):
    g = routing_graph()
    if request.source not in g or request.destination not in g:
        raise HTTPException(status_code=404, detail="Unknown source or destination node")
    try:
        path = nx.shortest_path(g, request.source, request.destination, weight="weight")
        cost = nx.path_weight(g, path, weight="weight")
    except nx.NetworkXNoPath:
        raise HTTPException(status_code=409, detail="No available route")
    return {"source": request.source, "destination": request.destination,
            "route": path, "cost": float(cost)}


@app.websocket("/ws/state")
async def state_socket(websocket: WebSocket):
    global RUNNING
    await websocket.accept()
    try:
        while True:
            if RUNNING:
                tick()
            await websocket.send_json(state())
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return
