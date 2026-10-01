import { useCallback, useEffect, useState } from "react";
import "./App.css";

import Header from "./components/Header";
import TrafficMap from "./components/TrafficMap";
import TrafficStats from "./components/TrafficStats";
import DisruptionPanel from "./components/DisruptionPanel";
import PredictionPanel from "./components/PredictionPanel";
import SimulationControls from "./components/SimulationControls";
import { createStateSocket, getState } from "./services/api";

const EMPTY_STATE = {
  status: "stopped",
  simulation_time: 0,
  stats: {
    cars: 0,
    normal_travel_time: 0,
    current_travel_time: 0,
    average_delay: 0,
  },
  disruptions: [],
  predictions: [],
  graph: { nodes: [], edges: [] },
};

function App() {
  const [state, setState] = useState(EMPTY_STATE);
  const [error, setError] = useState("");

  const refreshState = useCallback(async () => {
    try {
      setState(await getState());
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refreshState();

    const socket = createStateSocket(
      (nextState) => {
        setState(nextState);
        setError("");
      },
      () => setError("Backend WebSocket is unavailable. Start the Python backend."),
    );

    return () => socket.close();
  }, [refreshState]);

  return (
    <div>
      <Header status={state.status} simulationTime={state.simulation_time} error={error} />

      <div className="main-layout">
        <TrafficMap graph={state.graph} routes={state.routes || []} running={state.status === "running"} />
        <TrafficStats stats={state.stats} />
      </div>

      <DisruptionPanel graph={state.graph} disruptions={state.disruptions} onStateChange={setState} />
      <PredictionPanel predictions={state.predictions} />
      <SimulationControls onStateChange={setState} />
    </div>
  );
}

export default App;
