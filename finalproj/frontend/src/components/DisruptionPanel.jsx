import { useMemo, useState } from "react";
import { addDisruption, removeDisruption } from "../services/api";

function DisruptionPanel({ graph, disruptions, onStateChange }) {
  const [type, setType] = useState("accident");
  const [edgeId, setEdgeId] = useState("");
  const [factor, setFactor] = useState("3");
  const [busy, setBusy] = useState(false);
  const edges = useMemo(() => graph?.edges || [], [graph]);

  async function create() {
    const edge = edges.find((item) => String(item.id) === String(edgeId));
    if (!edge) return;
    setBusy(true);
    try {
      const next = await addDisruption({
        type,
        u: String(edge.u),
        v: String(edge.v),
        factor: type === "blocked" ? undefined : Number(factor),
      });
      onStateChange(next);
    } finally {
      setBusy(false);
    }
  }

  async function restore(item) {
    setBusy(true);
    try {
      onStateChange(await removeDisruption(item.u, item.v));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="disruption-panel">
      <h2>Disruptions</h2>
      <div className="control-row">
        <select value={type} onChange={(e) => setType(e.target.value)}>
          <option value="accident">Accident</option>
          <option value="construction">Construction</option>
          <option value="blocked">Blocked</option>
        </select>
        <select value={edgeId} onChange={(e) => setEdgeId(e.target.value)}>
          <option value="">Select road</option>
          {edges.map((edge) => (
            <option key={edge.id} value={edge.id}>
              {edge.id} ({edge.u} → {edge.v})
            </option>
          ))}
        </select>
        {type !== "blocked" && (
          <input value={factor} onChange={(e) => setFactor(e.target.value)} type="number" min="1" step="0.5" />
        )}
        <button disabled={!edgeId || busy} onClick={create}>APPLY</button>
      </div>

      {disruptions.length === 0 ? (
        <p className="empty-state">No active disruptions.</p>
      ) : (
        disruptions.map((item) => (
          <div className="disruption" key={`${item.u}-${item.v}`}>
            <div>
              <h3>{item.type}</h3>
              <p>{item.u} → {item.v}{item.factor > 1 ? ` · ×${item.factor}` : ""}</p>
            </div>
            <button disabled={busy} onClick={() => restore(item)}>RESTORE</button>
          </div>
        ))
      )}
    </section>
  );
}

export default DisruptionPanel;
