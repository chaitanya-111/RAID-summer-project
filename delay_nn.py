import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class DelayNN:

    def __init__(self):

        # ----------------------------------------------------
        # Neural Network
        # ----------------------------------------------------

        self.model = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42
        )

        # ----------------------------------------------------
        # Feature scaler
        # ----------------------------------------------------

        self.scaler = StandardScaler()

        self.trained = False

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self, X, y):

        """
        X = input features

        y = actual delay

        Features used:

            0 -> distance
            1 -> average speed
            2 -> normal travel time
            3 -> disruption factor
            4 -> disruption type
        """

        X = np.asarray(X, dtype=float)

        y = np.asarray(y, dtype=float)

        # ----------------------------------------------------
        # Split data
        # ----------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # ----------------------------------------------------
        # Scale input features
        # ----------------------------------------------------

        X_train_scaled = self.scaler.fit_transform(
            X_train
        )

        X_test_scaled = self.scaler.transform(
            X_test
        )

        # ----------------------------------------------------
        # Train NN
        # ----------------------------------------------------

        self.model.fit(
            X_train_scaled,
            y_train
        )

        self.trained = True

        # ----------------------------------------------------
        # Test prediction
        # ----------------------------------------------------

        predictions = self.model.predict(
            X_test_scaled
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            predictions
        )

        print("\n==============================")
        print("       NN TRAINING RESULT")
        print("==============================")

        print(
            "Training samples:",
            len(X_train)
        )

        print(
            "Testing samples :",
            len(X_test)
        )

        print(
            "MAE             :",
            round(mae, 4)
        )

        print(
            "RMSE            :",
            round(rmse, 4)
        )

        print(
            "R²              :",
            round(r2, 4)
        )

        print("==============================\n")

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }

    # ========================================================
    # PREDICT DELAY
    # ========================================================

    def predict_delay(
        self,
        distance,
        avg_speed,
        normal_time,
        disruption_factor,
        disruption_type
    ):

        if not self.trained:

            raise RuntimeError(
                "Train the neural network first."
            )

        # ----------------------------------------------------
        # Encode disruption type
        # ----------------------------------------------------

        if disruption_type == "normal":

            type_value = 0

        elif disruption_type == "accident":

            type_value = 1

        elif disruption_type == "construction":

            type_value = 2

        elif disruption_type == "blocked":

            type_value = 3

        else:

            type_value = 0

        # ----------------------------------------------------
        # Create feature vector
        # ----------------------------------------------------

        X = np.array([
            [
                distance,
                avg_speed,
                normal_time,
                disruption_factor,
                type_value
            ]
        ])

        # ----------------------------------------------------
        # Scale
        # ----------------------------------------------------

        X_scaled = self.scaler.transform(X)

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predicted_delay = self.model.predict(
            X_scaled
        )[0]

        # Delay cannot be negative
        predicted_delay = max(
            0.0,
            predicted_delay
        )

        return predicted_delay

    # ========================================================
    # CHOOSE LOWEST DELAY
    # ========================================================

    def choose_best_option(
        self,
        options
    ):

        """
        options is a list like:

        [
            {
                "name": "Route A",
                "distance": 500,
                "avg_speed": 10,
                "normal_time": 50,
                "disruption_factor": 3,
                "disruption_type": "accident"
            },

            {
                "name": "Route B",
                "distance": 700,
                "avg_speed": 15,
                "normal_time": 46.67,
                "disruption_factor": 1,
                "disruption_type": "normal"
            }
        ]
        """

        if not self.trained:

            raise RuntimeError(
                "Train the neural network first."
            )

        results = []

        for option in options:

            predicted_delay = self.predict_delay(

                option["distance"],

                option["avg_speed"],

                option["normal_time"],

                option["disruption_factor"],

                option["disruption_type"]
            )

            results.append({

                "name": option["name"],

                "predicted_delay":
                    predicted_delay
            })

        # ----------------------------------------------------
        # Find minimum predicted delay
        # ----------------------------------------------------

        best = min(
            results,
            key=lambda x: x["predicted_delay"]
        )

        return best, results


# ============================================================
# REAL TRAFFIC GRAPH TRAINING
# ============================================================

from traffic_simulation import (
    load_local_pbf,
    PBF_FILE_PATH
)

from disruption_engine import DisruptionEngine


def generate_training_data(
    graph,
    disruption_engine,
    samples_per_type=100
):

    X = []
    y = []

    edges = list(
        graph.edges()
    )

    if not edges:

        raise ValueError(
            "Graph contains no edges."
        )

    # --------------------------------------------------------
    # NORMAL ROAD SAMPLES
    # --------------------------------------------------------

    for _ in range(samples_per_type):

        u, v = edges[
            np.random.randint(
                0,
                len(edges)
            )
        ]

        edge = graph[u][v]

        distance = edge.get(
            "distance",
            0.0
        )

        avg_speed = edge.get(
            "avg_speed",
            0.0
        )

        normal_time = edge.get(
            "normal_time",
            0.0
        )

        # No disruption
        disruption_factor = 1

        disruption_type = 0

        # Delay = 0
        delay = 0.0

        X.append([
            distance,
            avg_speed,
            normal_time,
            disruption_factor,
            disruption_type
        ])

        y.append(
            delay
        )

    # --------------------------------------------------------
    # ACCIDENT SAMPLES
    # --------------------------------------------------------

    for _ in range(samples_per_type):

        u, v = edges[
            np.random.randint(
                0,
                len(edges)
            )
        ]

        edge = graph[u][v]

        distance = edge.get(
            "distance",
            0.0
        )

        avg_speed = edge.get(
            "avg_speed",
            0.0
        )

        normal_time = edge.get(
            "normal_time",
            0.0
        )

        disruption_factor = 3

        disruption_type = 1

        # ----------------------------------------------------
        # Actual accident
        # ----------------------------------------------------

        success = disruption_engine.add_accident(
            u,
            v,
            factor=3
        )

        if not success:
            continue

        disrupted_time = graph[u][v].get(
            "weight",
            normal_time
        )

        delay = max(
            0.0,
            disrupted_time - normal_time
        )

        X.append([
            distance,
            avg_speed,
            normal_time,
            disruption_factor,
            disruption_type
        ])

        y.append(
            delay
        )

        # Restore
        disruption_engine.restore_road(
            u,
            v
        )

    # --------------------------------------------------------
    # CONSTRUCTION SAMPLES
    # --------------------------------------------------------

    for _ in range(samples_per_type):

        u, v = edges[
            np.random.randint(
                0,
                len(edges)
            )
        ]

        edge = graph[u][v]

        distance = edge.get(
            "distance",
            0.0
        )

        avg_speed = edge.get(
            "avg_speed",
            0.0
        )

        normal_time = edge.get(
            "normal_time",
            0.0
        )

        disruption_factor = 2

        disruption_type = 2

        # ----------------------------------------------------
        # Actual construction
        # ----------------------------------------------------

        success = disruption_engine.add_construction(
            u,
            v,
            factor=2
        )

        if not success:
            continue

        disrupted_time = graph[u][v].get(
            "weight",
            normal_time
        )

        delay = max(
            0.0,
            disrupted_time - normal_time
        )

        X.append([
            distance,
            avg_speed,
            normal_time,
            disruption_factor,
            disruption_type
        ])

        y.append(
            delay
        )

        # Restore
        disruption_engine.restore_road(
            u,
            v
        )

    # --------------------------------------------------------
    # BLOCKED ROAD SAMPLES
    # --------------------------------------------------------

    for _ in range(samples_per_type):

        u, v = edges[
            np.random.randint(
                0,
                len(edges)
            )
        ]

        edge = graph[u][v]

        distance = edge.get(
            "distance",
            0.0
        )

        avg_speed = edge.get(
            "avg_speed",
            0.0
        )

        normal_time = edge.get(
            "normal_time",
            0.0
        )

        disruption_factor = 0

        disruption_type = 3

        # ----------------------------------------------------
        # A blocked edge has no finite edge travel time.
        #
        # We represent its delay as normal_time for this
        # edge. Route-level rerouting will be handled later.
        # ----------------------------------------------------

        delay = normal_time

        X.append([
            distance,
            avg_speed,
            normal_time,
            disruption_factor,
            disruption_type
        ])

        y.append(
            delay
        )

    return (
        np.array(X, dtype=float),
        np.array(y, dtype=float)
    )


# ============================================================
# TRAIN USING REAL ROAD DATA
# ============================================================

if __name__ == "__main__":

    print("\n==============================")
    print("   LOADING TRAFFIC GRAPH")
    print("==============================")

    graph, roundabout_nodes = (
        load_local_pbf(
            PBF_FILE_PATH
        )
    )

    print(
        f"Nodes: {len(graph.nodes)}"
    )

    print(
        f"Edges: {len(graph.edges)}"
    )

    # --------------------------------------------------------
    # Create disruption engine
    # --------------------------------------------------------

    disruption_engine = (
        DisruptionEngine(
            graph
        )
    )

    # --------------------------------------------------------
    # Generate REAL training data
    # --------------------------------------------------------

    print("\n==============================")
    print("   GENERATING TRAINING DATA")
    print("==============================")

    X, y = generate_training_data(
        graph,
        disruption_engine,
        samples_per_type=500
    )

    print(
        f"Training samples generated: {len(X)}"
    )

    print(
        f"Feature shape: {X.shape}"
    )

    print(
        f"Target shape: {y.shape}"
    )

    # --------------------------------------------------------
    # Create NN
    # --------------------------------------------------------

    nn = DelayNN()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    results = nn.train(
        X,
        y
    )

    # --------------------------------------------------------
    # Example prediction using REAL road data
    # --------------------------------------------------------

    test_edge = list(
        graph.edges()
    )[0]

    u, v = test_edge

    edge = graph[u][v]

    predicted_delay = nn.predict_delay(

        distance=edge.get(
            "distance",
            0.0
        ),

        avg_speed=edge.get(
            "avg_speed",
            0.0
        ),

        normal_time=edge.get(
            "normal_time",
            0.0
        ),

        disruption_factor=3,

        disruption_type="accident"
    )

    print("\n==============================")
    print("       REAL NN PREDICTION")
    print("==============================")

    print(
        "Road:",
        (u, v)
    )

    print(
        "Distance:",
        round(
            edge["distance"],
            2
        ),
        "m"
    )

    print(
        "Average speed:",
        round(
            edge["avg_speed"],
            2
        ),
        "m/s"
    )

    print(
        "Normal time:",
        round(
            edge["normal_time"],
            2
        ),
        "s"
    )

    print(
        "Predicted accident delay:",
        round(
            predicted_delay,
            2
        ),
        "s"
    )

    print("==============================")
