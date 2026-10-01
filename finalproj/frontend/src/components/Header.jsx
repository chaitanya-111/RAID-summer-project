function Header({ status, simulationTime, error }) {
  return (
    <header className="header">
      <h1>Traffic Control System</h1>
      <p>Smart Traffic Monitoring &amp; Simulation</p>
      <div className="header-status">
        <span>Status: {status}</span>
        <span>Simulation time: {simulationTime}s</span>
      </div>
      {error && <p className="backend-error">{error}</p>}
    </header>
  );
}

export default Header;
