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

        Target:

            actual delay in seconds
        """

        X = np.asarray(
            X,
            dtype=float
        )

        y = np.asarray(
            y,
            dtype=float
        )

        if len(X) == 0:

            raise ValueError(
                "No training data available."
            )

        if len(X) != len(y):

            raise ValueError(
                "X and y must contain the same number of samples."
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
        # Scale input features
        # ----------------------------------------------------

        X_train_scaled = self.scaler.fit_transform(
            X_train
        )

        X_test_scaled = self.scaler.transform(
            X_test
        )

        # ----------------------------------------------------
        # Train neural network
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

        # Delay cannot be negative
        predictions = np.maximum(
            predictions,
            0.0
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

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
        # Print results
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

        # ----------------------------------------------------
        # Encode disruption type
        # ----------------------------------------------------

        if disruption_type == "normal":

            type_value = 0

        elif disruption_type == "accident":

            type_value = 1

        elif disruption_type == "blocked":

            type_value = 3

        else:

            raise ValueError(
                "Invalid disruption type. "
                "Use normal, accident, or blocked."
            )

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
        ], dtype=float)

        # ----------------------------------------------------
        # Scale
        # ----------------------------------------------------

        X_scaled = self.scaler.transform(
            X
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predicted_delay = self.model.predict(
            X_scaled
        )[0]

        # ----------------------------------------------------
        # Prevent negative delay
        # ----------------------------------------------------

        predicted_delay = max(
            0.0,
            float(predicted_delay)
        )

        return predicted_delay

    # ========================================================
    # CALCULATE ACTUAL EDGE DELAY
    # ========================================================

    def calculate_edge_delay(
        self,
        graph,
        u,
        v
    ):

        """
        Calculates actual delay of one road.

        Normal:
            delay = 0

        Accident:
            delay = current_time - normal_time

        Blocked:
            delay = normal_time

        This uses the same logic as DisruptionEngine.
        """

        if not graph.has_edge(u, v):

            return 0.0

        edge = graph[u][v]

        normal_time = edge.get(
            "normal_time",
            edge.get(
                "weight",
                0.0
            )
        )

        # ----------------------------------------------------
        # BLOCKED
        # ----------------------------------------------------

        if edge.get(
            "blocked",
            False
        ):

            return max(
                0.0,
                normal_time
            )

        # ----------------------------------------------------
        # CURRENT TRAVEL TIME
        # ----------------------------------------------------

        current_time = edge.get(
            "weight",
            normal_time
        )

        # ----------------------------------------------------
        # DELAY
        # ----------------------------------------------------

        delay = (
            current_time
            -
            normal_time
        )

        return max(
            0.0,
            delay
        )

    # ========================================================
    # CALCULATE ACTUAL ROUTE DELAY
    # ========================================================

    def calculate_route_delay(
        self,
        graph,
        route
    ):

        """
        Calculates total actual delay for a route.

        Returns None if the route contains a blocked road.
        """

        if route is None:

            return 0.0

        if len(route) < 2:

            return 0.0

        normal_total = 0.0

        current_total = 0.0

        for u, v in zip(
            route[:-1],
            route[1:]
        ):

            if not graph.has_edge(
                u,
                v
            ):

                return 0.0

            edge = graph[u][v]

            # ------------------------------------------------
            # BLOCKED ROAD
            # ------------------------------------------------

            if edge.get(
                "blocked",
                False
            ):

                return None

            # ------------------------------------------------
            # NORMAL TIME
            # ------------------------------------------------

            normal_time = edge.get(
                "normal_time",
                edge.get(
                    "weight",
                    0.0
                )
            )

            # ------------------------------------------------
            # CURRENT TIME
            # ------------------------------------------------

            current_time = edge.get(
                "weight",
                normal_time
            )

            normal_total += normal_time

            current_total += current_time

        # ----------------------------------------------------
        # TOTAL DELAY
        # ----------------------------------------------------

        delay = (
            current_total
            -
            normal_total
        )

        return max(
            0.0,
            delay
        )

    # ========================================================
    # CHOOSE LOWEST PREDICTED DELAY
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
            }
        ]
        """

        if not self.trained:

            raise RuntimeError(
                "Train the neural network first."
            )

        if not options:

            return None, []

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

                "predicted_delay": predicted_delay
            })

        # ----------------------------------------------------
        # Find lowest predicted delay
        # ----------------------------------------------------

        best = min(
            results,
            key=lambda x: x["predicted_delay"]
        )

        return best, results


# ============================================================
# GENERATE REAL TRAINING DATA
# ============================================================

def generate_training_data(
    graph,
    disruption_engine,
    samples_per_type=500
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

        # ----------------------------------------------------
        # NORMAL ROAD
        # ----------------------------------------------------

        disruption_factor = 1

        disruption_type = 0

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
        # Calculate actual delay
        # ----------------------------------------------------

        current_time = graph[u][v].get(
            "weight",
            normal_time
        )

        delay = max(
            0.0,
            current_time
            -
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
        # Block road
        # ----------------------------------------------------

        success = disruption_engine.block_road(
            u,
            v
        )

        if not success:

            continue

        # ----------------------------------------------------
        # Blocked road gets finite delay target
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

        # ----------------------------------------------------
        # Restore road
        # ----------------------------------------------------

        disruption_engine.restore_road(
            u,
            v
        )

    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    X = np.array(
        X,
        dtype=float
    )

    y = np.array(
        y,
        dtype=float
    )

    if len(X) == 0:

        raise ValueError(
            "No training samples were generated."
        )

    return X, y


# ============================================================
# TRAIN USING REAL ROAD DATA
# ============================================================

if __name__ == "__main__":

    print("\n==============================")
    print("   LOADING TRAFFIC GRAPH")
    print("==============================")

    graph, roundabout_nodes = load_local_pbf(
        PBF_FILE_PATH
    )

    print(
        "Nodes:",
        len(graph.nodes)
    )

    print(
        "Edges:",
        len(graph.edges)
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
    # CREATE NN
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
    # TEST REAL ACCIDENT PREDICTION
    # ========================================================

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
            predicted_delay,
            2
        ),
        "s"
    )

    print("==============================")
