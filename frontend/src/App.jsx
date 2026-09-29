import "./App.css";

import Header from "./components/Header";
import TrafficMap from "./components/TrafficMap";
import TrafficStats from "./components/TrafficStats";
import DisruptionPanel from "./components/DisruptionPanel";
import PredictionPanel from "./components/PredictionPanel";
import SimulationControls from "./components/SimulationControls";

function App() {
  return (
    <div>
      <Header />

      <div className="main-layout">
        <TrafficMap />
        <TrafficStats />
      </div>

      <DisruptionPanel />

      <PredictionPanel />

      <SimulationControls />
    </div>
  );
}

export default App;