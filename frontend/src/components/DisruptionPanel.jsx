function DisruptionPanel() {
    const disruptions = [
        {
            type: "Accident",
            road: "Road A → Road B",
        },
        {
            type: "Blocked",
            road: "Road C → Road D",
        },
        {
            type: "Construction",
            road: "Road E → Road F",
        },
    ];

    return (
        <section className="disruption-panel">
            <h2>Active Disruptions</h2>

            {disruptions.map((disruption, index) => (
                <div className="disruption" key={index}>
                    <h3>{disruption.type}</h3>
                    <p>{disruption.road}</p>
                </div>
            ))}
        </section>
    );
}

export default DisruptionPanel;