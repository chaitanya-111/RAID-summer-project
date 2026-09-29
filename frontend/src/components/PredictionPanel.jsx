function PredictionPanel() {
    const predictions = [
        {
            route: "Route A",
            delay: 18.4,
        },
        {
            route: "Route B",
            delay: 4.2,
        },
        {
            route: "Route C",
            delay: null,
        },
    ];

    return (
        <section className="prediction-panel">
            <h2>Delay Prediction</h2>

            {predictions.map((prediction, index) => (
                <div className="prediction" key={index}>
                    <h3>{prediction.route}</h3>

                    {prediction.delay === null ? (
                        <p>BLOCKED</p>
                    ) : (
                        <p>+{prediction.delay} s</p>
                    )}
                </div>
            ))}
        </section>
    );
}

export default PredictionPanel;