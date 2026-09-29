"""
TRAFFIC DISRUPTION ENGINE
=========================

What this program does
----------------------

1. Load normal SUMO traffic data.
2. Load the already-trained GRU.
3. Get the GRU edge weights for a future timestamp.
4. Randomly block one or more edges.
5. Take the traffic flow that was using the blocked edge.
6. Find alternative edges connected to the same junction.
7. Redistribute the blocked traffic among those edges.
8. Prefer edges with lower GRU weight and lower existing traffic.
9. Save the resulting traffic state.

IMPORTANT
---------

This program DOES NOT use:

    - Dijkstra
    - A*
    - source/destination
    - OD pairs
    - path finding

This is LOCAL TRAFFIC FLOW REDISTRIBUTION.

The cars continue as part of the existing traffic flow.
"""


# ============================================================
# IMPORTS
# ============================================================

import os
import random
import importlib.util

import numpy as np
import pandas as pd
import torch


# ============================================================
# USER CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Your files
# ------------------------------------------------------------

GRU_CHECKPOINT = "best_traffic_gru_2.pt"

GRU_SCRIPT = "traffic_gru_new (1)g.py"

TRAFFIC_FILE = "normal_traffic_median_filled.csv"

# Your SUMO network file.
#
# Example:
#
#     map.net.xml
#     osm.net.xml
#     jodhpur.net.xml
#
NETWORK_FILE = "your_network.net.xml"


# ------------------------------------------------------------
# Disruption settings
# ------------------------------------------------------------

# Number of random edges to block
NUMBER_OF_BLOCKED_EDGES = 1

# Random seed
RANDOM_SEED = 42

# GRU prediction step
#
# 1 = first future timestamp
# 2 = second future timestamp
# ...
#
# Your model has horizon = 6.
GRU_STEP = 1


# ------------------------------------------------------------
# Redistribution settings
# ------------------------------------------------------------

# Controls how strongly existing traffic is considered.
#
# Higher value:
#     avoid already congested edges more strongly.
#
TRAFFIC_PENALTY = 1.0

# Controls how strongly GRU weight is considered.
#
# Higher value:
#     prefer lower GRU-weight edges more strongly.
#
WEIGHT_PENALTY = 1.0

# Minimum amount of traffic that is considered significant.
EPSILON = 1e-8


# ============================================================
# LOAD YOUR GRU TRAINING SCRIPT
# ============================================================

def load_gru_module(script_path):

    if not os.path.exists(script_path):

        raise FileNotFoundError(
            f"\nGRU script not found:\n{script_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "traffic_gru_module",
        script_path
    )

    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    return module


# ============================================================
# LOAD TRAFFIC DATA
# ============================================================

def load_traffic():

    if not os.path.exists(TRAFFIC_FILE):

        raise FileNotFoundError(
            f"\nTraffic file not found:\n{TRAFFIC_FILE}"
        )

    df = pd.read_csv(TRAFFIC_FILE)

    required_columns = [

        "time",
        "edge_id",
        "flow"

    ]

    missing = [

        c for c in required_columns
        if c not in df.columns

    ]

    if missing:

        raise ValueError(
            f"\nTraffic CSV is missing:\n{missing}"
        )

    df["edge_id"] = (
        df["edge_id"]
        .astype(str)
    )

    return df


# ============================================================
# LOAD SUMO NETWORK
# ============================================================

def load_network(network_file):

    """
    Read SUMO .net.xml.

    We only use:

        edge_id
        from_node
        to_node

    No routing algorithm is used.
    """

    import xml.etree.ElementTree as ET

    if not os.path.exists(network_file):

        raise FileNotFoundError(
            f"\nSUMO network file not found:\n"
            f"{network_file}"
        )

    tree = ET.parse(network_file)

    root = tree.getroot()

    rows = []

    for edge in root.findall("edge"):

        edge_id = edge.get("id")

        # Ignore SUMO internal edges
        if edge.get("function") == "internal":

            continue

        from_node = edge.get("from")

        to_node = edge.get("to")

        if (
            edge_id is None
            or from_node is None
            or to_node is None
        ):

            continue

        rows.append({

            "edge_id": str(edge_id),

            "from_node": str(from_node),

            "to_node": str(to_node)

        })

    network = pd.DataFrame(rows)

    if len(network) == 0:

        raise ValueError(
            "No normal edges were found in the SUMO network."
        )

    return network


# ============================================================
# GET LATEST TRAFFIC STATE
# ============================================================

def get_latest_state(df):

    latest_time = df["time"].max()

    state = df[
        df["time"] == latest_time
    ].copy()

    return state, latest_time


# ============================================================
# GET TRAFFIC FLOW OF AN EDGE
# ============================================================

def get_flow(state, edge_id):

    rows = state[
        state["edge_id"].astype(str)
        == str(edge_id)
    ]

    if len(rows) == 0:

        return 0.0

    value = rows["flow"].sum()

    if pd.isna(value):

        return 0.0

    return float(value)


# ============================================================
# GET ALL EDGE FLOWS
# ============================================================

def get_flow_dictionary(state):

    flows = {}

    for _, row in state.iterrows():

        edge = str(row["edge_id"])

        flow = row["flow"]

        if pd.isna(flow):

            flow = 0.0

        flows[edge] = float(flow)

    return flows


# ============================================================
# FIND ALTERNATIVE EDGES
# ============================================================

def find_alternative_edges(
    blocked_edge,
    network,
    blocked_edges
):
    """
    Find outgoing edges from the same junction.

    Example:

                    E2
                    ↓
              ┌──── Junction
              │      ↓
        E1 → Junction → E3
              │
              ↓
                    E4

    If E1 is blocked, E2/E3/E4 can receive
    the displaced flow.

    We remove all currently blocked edges.
    """

    blocked_edges = set(

        str(x)
        for x in blocked_edges

    )

    blocked_rows = network[
        network["edge_id"].astype(str)
        == str(blocked_edge)
    ]

    if len(blocked_rows) == 0:

        return []

    blocked_row = blocked_rows.iloc[0]

    # The junction where the blocked edge ends.
    #
    # Traffic that would enter the blocked edge is
    # redirected at this junction.

    junction = blocked_row["to_node"]

    alternatives = network[
        network["from_node"].astype(str)
        == str(junction)
    ].copy()

    alternatives = alternatives[
        ~alternatives["edge_id"]
        .astype(str)
        .isin(blocked_edges)
    ]

    alternatives = alternatives[
        alternatives["edge_id"].astype(str)
        != str(blocked_edge)
    ]

    return alternatives["edge_id"].astype(str).tolist()


# ============================================================
# GET GRU EDGE WEIGHTS
# ============================================================

def get_gru_weights(
    gru_module,
    traffic_history
):

    print("\nRunning trained GRU...")

    # --------------------------------------------------------
    # The training script already contains:
    #
    #     predict_edge_weights()
    #
    # We use it directly.
    #
    # No blockage is supplied to the GRU.
    #
    # The disruption happens AFTER obtaining the weights.
    # --------------------------------------------------------

    output = (
        gru_module.predict_edge_weights(

            history_df=traffic_history,

            model_path=GRU_CHECKPOINT,

            horizon=None,

            blocked_edges=(),

            mark_history_blocked=False

        )
    )

    weights = output["weights"]

    edge_ids = output["edge_ids"]

    # Convert selected timestep into dictionary.
    #
    # weights shape:
    #
    #     [horizon, number_of_edges]
    #
    step_index = GRU_STEP - 1

    if step_index < 0:

        raise ValueError(
            "GRU_STEP must be >= 1."
        )

    if step_index >= weights.shape[0]:

        raise ValueError(
            f"GRU_STEP={GRU_STEP} is larger than "
            f"the available horizon={weights.shape[0]}."
        )

    selected_weights = weights[
        step_index
    ]

    edge_weights = {}

    for edge_id, weight in zip(
        edge_ids,
        selected_weights
    ):

        edge_weights[str(edge_id)] = float(weight)

    prediction_time = output[
        "times"
    ][step_index]

    print(
        f"GRU prediction timestamp: "
        f"{prediction_time}"
    )

    return edge_weights


# ============================================================
# CALCULATE REDISTRIBUTION SCORE
# ============================================================

def calculate_edge_score(
    edge,
    current_flow,
    gru_weight
):
    """
    Lower score = more attractive edge.

    We combine:

        GRU weight
        +
        current traffic

    The exact formula is:

        score =
            WEIGHT_PENALTY * normalized_weight
            +
            TRAFFIC_PENALTY * normalized_flow

    """

    if not np.isfinite(gru_weight):

        return np.inf

    return (
        WEIGHT_PENALTY * gru_weight
        +
        TRAFFIC_PENALTY * current_flow
    )


# ============================================================
# REDISTRIBUTE TRAFFIC
# ============================================================

def redistribute_traffic(
    blocked_edge,
    blocked_flow,
    alternative_edges,
    flow_dict,
    weight_dict
):
    """
    Redistribute blocked traffic.

    Important:

        We do NOT simply put all traffic on the
        lowest-weight edge.

    Instead, the traffic is distributed using
    the relative attractiveness of all available
    edges.

    Lower:

        GRU weight
        current traffic

    means:

        more traffic gets assigned.
    """

    if blocked_flow <= EPSILON:

        return {}

    if len(alternative_edges) == 0:

        print(
            "WARNING: No alternative edges available."
        )

        return {}

    # --------------------------------------------------------
    # Calculate scores
    # --------------------------------------------------------

    scores = {}

    for edge in alternative_edges:

        current_flow = flow_dict.get(
            edge,
            0.0
        )

        weight = weight_dict.get(
            edge,
            np.inf
        )

        score = calculate_edge_score(

            edge,

            current_flow,

            weight

        )

        scores[edge] = score

    # Remove unusable edges.
    scores = {

        edge: score

        for edge, score in scores.items()

        if np.isfinite(score)

    }

    if len(scores) == 0:

        print(
            "WARNING: No usable alternative edges."
        )

        return {}

    # --------------------------------------------------------
    # Convert scores into attractiveness.
    #
    # Lower score = more attractive.
    #
    # attractiveness = 1 / score
    # --------------------------------------------------------

    attractiveness = {}

    for edge, score in scores.items():

        attractiveness[edge] = (
            1.0 / max(score, EPSILON)
        )

    total_attractiveness = sum(
        attractiveness.values()
    )

    if total_attractiveness <= EPSILON:

        return {}

    # --------------------------------------------------------
    # Initial proportional distribution
    # --------------------------------------------------------

    distribution = {}

    for edge, value in attractiveness.items():

        fraction = (
            value
            / total_attractiveness
        )

        distribution[edge] = (
            blocked_flow * fraction
        )

    return distribution


# ============================================================
# APPLY DISRUPTION
# ============================================================

def apply_disruption(
    state,
    network,
    edge_weights,
    blocked_edges
):
    """
    Apply all disruptions to the current traffic state.
    """

    # Current traffic on every edge.
    flow_dict = get_flow_dictionary(
        state
    )

    # Save original values.
    original_flow = flow_dict.copy()

    redistribution_records = []

    # --------------------------------------------------------
    # Process each blocked edge
    # --------------------------------------------------------

    for blocked_edge in blocked_edges:

        blocked_flow = flow_dict.get(
            blocked_edge,
            0.0
        )

        print("\n" + "=" * 70)

        print(
            f"BLOCKED EDGE: {blocked_edge}"
        )

        print(
            f"Traffic on blocked edge: "
            f"{blocked_flow:.2f}"
        )

        if blocked_flow <= EPSILON:

            print(
                "No traffic to redistribute."
            )

            continue

        # ----------------------------------------------------
        # Find alternatives
        # ----------------------------------------------------

        alternatives = find_alternative_edges(

            blocked_edge,

            network,

            blocked_edges

        )

        print(
            f"Alternative edges: "
            f"{len(alternatives)}"
        )

        for edge in alternatives:

            print(
                f"    {edge}"
            )

        # ----------------------------------------------------
        # Remove traffic from blocked edge.
        # ----------------------------------------------------

        flow_dict[blocked_edge] = 0.0

        # ----------------------------------------------------
        # Redistribute
        # ----------------------------------------------------

        distribution = redistribute_traffic(

            blocked_edge,

            blocked_flow,

            alternatives,

            flow_dict,

            edge_weights

        )

        # ----------------------------------------------------
        # Add redistributed traffic
        # ----------------------------------------------------

        print("\nREDISTRIBUTION:")

        for edge, amount in distribution.items():

            flow_dict[edge] = (
                flow_dict.get(edge, 0.0)
                + amount
            )

            weight = edge_weights.get(
                edge,
                np.nan
            )

            redistribution_records.append({

                "blocked_edge":
                    blocked_edge,

                "alternative_edge":
                    edge,

                "blocked_flow":
                    blocked_flow,

                "diverted_flow":
                    amount,

                "existing_flow":
                    flow_dict[edge] - amount,

                "new_flow":
                    flow_dict[edge],

                "gru_weight":
                    weight

            })

            print(
                f"    {amount:10.2f} vehicles"
                f" -> {edge}"
                f" | weight = {weight:.4f}"
            )

    return (
        flow_dict,
        original_flow,
        redistribution_records
    )


# ============================================================
# RANDOM BLOCKAGE
# ============================================================

def choose_random_edges(
    network,
    number_of_edges,
    seed
):

    random.seed(seed)

    available_edges = (
        network["edge_id"]
        .astype(str)
        .tolist()
    )

    if number_of_edges > len(
        available_edges
    ):

        raise ValueError(
            "Number of blocked edges is larger "
            "than the number of network edges."
        )

    blocked = random.sample(

        available_edges,

        number_of_edges

    )

    return blocked


# ============================================================
# CREATE OUTPUT DATAFRAME
# ============================================================

def create_output(
    state,
    new_flows
):

    output = state.copy()

    output["original_flow"] = (
        output["edge_id"]
        .astype(str)
        .map(
            lambda x:
                float(
                    state[
                        state["edge_id"]
                        .astype(str)
                        == x
                    ]["flow"].sum()
                )
        )
    )

    output["new_flow"] = (
        output["edge_id"]
        .astype(str)
        .map(
            lambda x:
                new_flows.get(
                    x,
                    0.0
                )
        )
    )

    output["flow_change"] = (
        output["new_flow"]
        -
        output["original_flow"]
    )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("       TRAFFIC DISRUPTION ENGINE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load traffic
    # --------------------------------------------------------

    print("\n[1] Loading traffic data...")

    traffic = load_traffic()

    print(
        f"Traffic rows: {len(traffic)}"
    )

    # --------------------------------------------------------
    # 2. Load network
    # --------------------------------------------------------

    print("\n[2] Loading SUMO network...")

    network = load_network(
        NETWORK_FILE
    )

    print(
        f"Network edges: {len(network)}"
    )

    # --------------------------------------------------------
    # 3. Current traffic state
    # --------------------------------------------------------

    state, current_time = (
        get_latest_state(
            traffic
        )
    )

    print(
        f"\nCurrent timestamp: "
        f"{current_time}"
    )

    # --------------------------------------------------------
    # 4. Load GRU
    # --------------------------------------------------------

    print(
        "\n[3] Loading trained GRU..."
    )

    gru_module = load_gru_module(
        GRU_SCRIPT
    )

    # --------------------------------------------------------
    # 5. Get GRU weights
    # --------------------------------------------------------

    print(
        "\n[4] Predicting edge weights..."
    )

    edge_weights = get_gru_weights(

        gru_module,

        traffic

    )

    print(
        f"Received weights for "
        f"{len(edge_weights)} edges."
    )

    # --------------------------------------------------------
    # 6. Randomly block edges
    # --------------------------------------------------------

    print(
        "\n[5] Creating random disruption..."
    )

    blocked_edges = choose_random_edges(

        network,

        NUMBER_OF_BLOCKED_EDGES,

        RANDOM_SEED

    )

    print(
        "\nBLOCKED EDGES:"
    )

    for edge in blocked_edges:

        print(
            f"    {edge}"
        )

    # --------------------------------------------------------
    # 7. Redistribute traffic
    # --------------------------------------------------------

    print(
        "\n[6] Redistributing traffic..."
    )

    (
        new_flows,
        original_flows,
        records
    ) = apply_disruption(

        state,

        network,

        edge_weights,

        blocked_edges

    )

    # --------------------------------------------------------
    # 8. Create output
    # --------------------------------------------------------

    output = create_output(

        state,

        new_flows

    )

    # --------------------------------------------------------
    # 9. Save new traffic state
    # --------------------------------------------------------

    output_file = (
        "traffic_after_disruption.csv"
    )

    output.to_csv(

        output_file,

        index=False

    )

    # --------------------------------------------------------
    # 10. Save redistribution details
    # --------------------------------------------------------

    redistribution_file = (
        "redistribution_details.csv"
    )

    redistribution_df = pd.DataFrame(
        records
    )

    redistribution_df.to_csv(

        redistribution_file,

        index=False

    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("DISRUPTION COMPLETE")
    print("=" * 70)

    print(
        f"\nBlocked edges:"
    )

    for edge in blocked_edges:

        print(
            f"    {edge}"
        )

    print(
        f"\nOriginal total flow: "
        f"{sum(original_flows.values()):.2f}"
    )

    print(
        f"New total flow: "
        f"{sum(new_flows.values()):.2f}"
    )

    print(
        "\nFiles created:"
    )

    print(
        f"    {output_file}"
    )

    print(
        f"    {redistribution_file}"
    )

    print("\n")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

