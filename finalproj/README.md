# GRU Traffic Routing — Complete Test Integration

This package connects the supplied React frontend to the supplied `traffic_gru_new.py`, `best_traffic_gru.pt`, and `repo.py` disruption engine.

> The `code/` directory from the original upload is intentionally excluded.

## What is implemented

- React dashboard connected to FastAPI.
- WebSocket live simulation updates.
- Supplied GRU checkpoint used for edge-weight prediction.
- GRU edge IDs mapped to graph edges.
- Dijkstra routing through NetworkX using the GRU's predicted edge weights.
- Blocked roads removed from routing and hard-blocked in GRU predictions.
- Accident and construction factors applied to routing weights.
- Frontend controls for start/pause/reset and live disruptions.
- Live graph visualization with routed edges highlighted.
- Demo graph + synthetic history that match the supplied checkpoint, so the package can be tested immediately.

## 1. Backend

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r frontend/backend/requirements.txt
uvicorn frontend.backend.main:app --host 0.0.0.0 --port 8000
```

Check:

```bash
curl http://localhost:8000/health
```

You should see `ok: true`, `model: true`, `history: true`, and `graph: true`.

## 2. Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`.

## 3. Test sequence

1. Start backend.
2. Start frontend.
3. Confirm the map and GRU predictions appear.
4. Press **START**.
5. Watch simulation time and route state update every second.
6. Select a road and apply **Accident** or **Construction**.
7. Apply **Blocked** to remove that road from routing.
8. Press **RESTORE** to return the road to normal.
9. Press **RESET** to restore the whole simulation.

## 4. Replace demo data with your real data

The included demo files are only a runnable integration harness. For your real SUMO run, provide:

- a traffic CSV whose edge IDs match the checkpoint's `edge2idx` and whose feature columns match the checkpoint;
- a graph JSON whose edge `id` values match those same GRU edge IDs.

Then run, for example:

```bash
export TRAFFIC_HISTORY_CSV=/absolute/path/to/your/traffic.csv
export GRAPH_JSON=/absolute/path/to/your/graph.json
export GRU_MODEL_PATH=/absolute/path/to/best_traffic_gru.pt
uvicorn frontend.backend.main:app --host 0.0.0.0 --port 8000
```

The supplied model was trained with a 12-step history and predicts up to 6 future steps. The backend uses the first future step as the current routing weight and keeps the complete six-step prediction available to the backend.
