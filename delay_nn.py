import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from traffic_simulation import (
    load_local_pbf,
    PBF_FILE_PATH
)

from disruption_engine import DisruptionEngine


# ============================================================
# DELAY NEURAL NETWORK
# ============================================================

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
        Features:

            0 -> distance
            1 -> average speed
            2 -> normal travel time
            3 -> disruption factor
            4 -> disruption type

        Disruption types:

            0 -> normal
            1 -> accident
            3 -> blocked
        """

        X = np.asarray(
            X,
            dtype=float
        )

        y = np.asarray(
            y,
            dtype=float
        )

        # ----------------------------------------------------
        # Check training data
        # ----------------------------------------------------

        if len(X) == 0:

            raise ValueError(
                "No training data was generated."
            )

        if len(X) != len(y):

            raise ValueError(
                "X and y must have the same number of samples."
            )

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
        # Scale features
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

        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

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

        # ====================================================
        # NORMAL ROAD
        # ====================================================

        if disruption_type == "normal":

            return 0.0

        # ====================================================
        # BLOCKED ROAD
        # ====================================================

        if disruption_type == "blocked":

            # A blocked road cannot be travelled.
            #
            # We represent the edge delay as its normal
            # travel time. Actual route rerouting is handled
            # separately by the traffic graph.

            return max(
                0.0,
                float(normal_time)
            )

        # ====================================================
        # ENCODE DISRUPTION TYPE
        # ====================================================

        if disruption_type == "accident":

            type_value = 1

        else:

            type_value = 0

        # ====================================================
        # CREATE FEATURE VECTOR
        # ====================================================

        X = np.array([
            [
                distance,
                avg_speed,
                normal_time,
                disruption_factor,
                type_value
            ]
        ], dtype=float)

        # ====================================================
        # SCALE
        # ====================================================

        X_scaled = self.scaler.transform(
            X
        )

        # ====================================================
        # NN PREDICTION
        # ====================================================

        predicted_delay = self.model.predict(
            X_scaled
        )[0]

        # ====================================================
        # DELAY CANNOT BE NEGATIVE
        # ====================================================

        predicted_delay = max(
            0.0,
            float(predicted_delay)
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
        options example:

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
            },

            {
                "name": "Route C",
                "distance": 600,
                "avg_speed": 12,
                "normal_time": 50,
                "disruption_factor": 0,
                "disruption_type": "blocked"
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
# GENERATE REAL TRAFFIC TRAINING DATA
# ============================================================

def generate_training_data(
    graph,
    disruption_engine,
    samples_per_type=500
):

    X = []
    y = []

    # --------------------------------------------------------
    # Get graph edges
    # --------------------------------------------------------

    edges = list(
        graph.edges()
    )

    if not edges:

        raise ValueError(
            "Graph contains no edges."
        )

    # ========================================================
    # NORMAL ROAD SAMPLES
    # ========================================================

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
            edge.get(
                "weight",
                0.0
            )
        )

        disruption_factor = 1

        disruption_type = 0

        # Normal road has no delay.

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

    # ========================================================
    # ACCIDENT SAMPLES
    # ========================================================

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
            edge.get(
                "weight",
                0.0
            )
        )

        disruption_factor = 3

        disruption_type = 1

        # ----------------------------------------------------
        # Add accident
        # ----------------------------------------------------

        success = disruption_engine.add_accident(
            u,
            v,
            factor=3
        )

        if not success:

            continue

        # ----------------------------------------------------
        # Get disrupted travel time
        # ----------------------------------------------------

        disrupted_time = graph[u][v].get(
            "weight",
            normal_time
        )

        # ----------------------------------------------------
        # Calculate actual delay
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Restore road
        # ----------------------------------------------------

        disruption_engine.restore_road(
            u,
            v
        )

    # ========================================================
    # BLOCKED ROAD SAMPLES
    # ========================================================

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
            edge.get(
                "weight",
                0.0
            )
        )

        disruption_factor = 0

        disruption_type = 3

        # ----------------------------------------------------
        # Blocked road
        # ----------------------------------------------------
        #
        # There is no finite travel time for a blocked edge.
        #
        # We represent its edge-level delay as normal_time.
        #
        # The actual route-level behaviour is handled by
        # rerouting around the blocked road.
        # ----------------------------------------------------

        delay = max(
            0.0,
            normal_time
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

    # ========================================================
    # CONVERT TO NUMPY ARRAYS
    # ========================================================

    X = np.array(
        X,
        dtype=float
    )

    y = np.array(
        y,
        dtype=float
    )

    return X, y


# ============================================================
# TRAIN USING REAL ROAD DATA
# ============================================================

if __name__ == "__main__":

    print("\n==============================")
    print("   LOADING TRAFFIC GRAPH")
    print("==============================")

    # --------------------------------------------------------
    # Load graph
    # --------------------------------------------------------

    graph, roundabout_nodes = load_local_pbf(
        PBF_FILE_PATH
    )

    print(
        f"Nodes: {len(graph.nodes)}"
    )

    print(
        f"Edges: {len(graph.edges)}"
    )

    # ========================================================
    # CREATE DISRUPTION ENGINE
    # ========================================================

    disruption_engine = DisruptionEngine(
        graph
    )

    # ========================================================
    # GENERATE TRAINING DATA
    # ========================================================

    print("\n==============================")
    print("   GENERATING TRAINING DATA")
    print("==============================")

    X, y = generate_training_data(
        graph,
        disruption_engine,
        samples_per_type=500
    )

    print(
        "Training samples generated:",
        len(X)
    )

    print(
        "Feature shape:",
        X.shape
    )

    print(
        "Target shape:",
        y.shape
    )

    # ========================================================
    # CREATE NEURAL NETWORK
    # ========================================================

    nn = DelayNN()

    # ========================================================
    # TRAIN
    # ========================================================

    results = nn.train(
        X,
        y
    )

    # ========================================================
    # TEST ACCIDENT PREDICTION
    # ========================================================

    test_edge = list(
        graph.edges()
    )[0]

    u, v = test_edge

    edge = graph[u][v]

    # --------------------------------------------------------
    # Accident prediction
    # --------------------------------------------------------

    predicted_accident_delay = nn.predict_delay(

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

    # --------------------------------------------------------
    # Blocked prediction
    # --------------------------------------------------------

    predicted_blocked_delay = nn.predict_delay(

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

        disruption_factor=0,

        disruption_type="blocked"
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

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
            edge.get(
                "distance",
                0.0
            ),
            2
        ),
        "m"
    )

    print(
        "Average speed:",
        round(
            edge.get(
                "avg_speed",
                0.0
            ),
            2
        ),
        "m/s"
    )

    print(
        "Normal time:",
        round(
            edge.get(
                "normal_time",
                0.0
            ),
            2
        ),
        "s"
    )

    print(
        "Predicted accident delay:",
        round(
            predicted_accident_delay,
            2
        ),
        "s"
    )

    print(
        "Blocked road delay:",
        round(
            predicted_blocked_delay,
            2
        ),
        "s"
    )

    print("==============================")
