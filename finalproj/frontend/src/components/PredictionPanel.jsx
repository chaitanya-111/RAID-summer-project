function PredictionPanel({ predictions }) {
  return (
    <section className="prediction-panel">
      <h2>GRU Edge-Weight Predictions</h2>
      {predictions.length === 0 ? <p className="empty-state">No GRU predictions available.</p> : (
        <div className="prediction-grid">
          {predictions.map((prediction) => (
            <div className="prediction" key={`${prediction.edge_id}-${prediction.step}`}>
              <h3>{prediction.edge_id}</h3>
              <p>+{prediction.step} min</p>
              <strong>{prediction.blocked ? "BLOCKED" : Number(prediction.weight).toFixed(3)}</strong>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
export default PredictionPanel;
