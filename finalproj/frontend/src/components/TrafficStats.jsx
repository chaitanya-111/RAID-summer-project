function TrafficStats({ stats }) {
  const items = [
    ["Cars", stats.cars],
    ["Baseline Routing Cost", Number(stats.normal_travel_time || 0).toFixed(2)],
    ["Current Routing Cost", Number(stats.current_travel_time || 0).toFixed(2)],
    ["Added Routing Cost", Number(stats.average_delay || 0).toFixed(2)],
  ];

  return (
    <section className="traffic-stats">
      <h2>Traffic Statistics</h2>
      {items.map(([label, value]) => (
        <div className="stat" key={label}>
          <h3>{label}</h3>
          <p>{value}</p>
        </div>
      ))}
    </section>
  );
}
export default TrafficStats;
