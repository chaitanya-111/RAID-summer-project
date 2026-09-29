function TrafficStats() {
    return (
        <section className="traffic-stats">
            <h2>Traffic Statistics</h2>

            <div className="stat">
                <h3>Cars</h3>
                <p>200</p>
            </div>

            <div className="stat">
                <h3>Normal Travel Time</h3>
                <p>72.4 s</p>
            </div>

            <div className="stat">
                <h3>Current Travel Time</h3>
                <p>91.8 s</p>
            </div>

            <div className="stat">
                <h3>Average Delay</h3>
                <p>19.4 s</p>
            </div>
        </section>
    );
}

export default TrafficStats;