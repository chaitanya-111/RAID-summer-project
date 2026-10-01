import { simulationCommand } from "../services/api";

function SimulationControls({ onStateChange }) {
  async function send(command) {
    try {
      onStateChange(await simulationCommand(command));
    } catch (error) {
      console.error(error);
    }
  }
  return (
    <section className="simulation-controls">
      <h2>Simulation Controls</h2>
      <button onClick={() => send("start")}>START</button>
      <button onClick={() => send("pause")}>PAUSE</button>
      <button onClick={() => send("reset")}>RESET</button>
    </section>
  );
}
export default SimulationControls;
