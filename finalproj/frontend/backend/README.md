# Backend

This backend connects the React dashboard to the supplied GRU checkpoint and DisruptionEngine.

## Run

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r frontend/backend/requirements.txt
uvicorn frontend.backend.main:app --reload --host 0.0.0.0 --port 8000
```

The default test data is `data/demo_traffic.csv` and `data/demo_graph.json`.

For your real SUMO data, set:

```bash
export TRAFFIC_HISTORY_CSV=/absolute/path/to/traffic.csv
export GRAPH_JSON=/absolute/path/to/graph.json
export GRU_MODEL_PATH=/absolute/path/to/best_traffic_gru.pt
```

The traffic CSV must use the feature schema expected by the supplied checkpoint. The graph JSON must contain `nodes` and `edges`, with each edge's `id` matching an edge ID in the GRU checkpoint.
