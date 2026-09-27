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
# DEMO TRAINING
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Example training data
    #
    # Each row:
    #
    # distance
    # avg_speed
    # normal_time
    # disruption_factor
    # disruption_type
    # --------------------------------------------------------

    X = np.array([

        [500, 10, 50, 1, 0],
        [600, 12, 50, 1, 0],
        [800, 16, 50, 1, 0],
        [400, 8, 50, 3, 1],
        [600, 10, 60, 3, 1],
        [1000, 20, 50, 3, 1],
        [500, 10, 50, 2, 2],
        [700, 14, 50, 2, 2],
        [900, 18, 50, 2, 2],

        [400, 8, 50, 1, 0],
        [700, 14, 50, 1, 0],
        [900, 18, 50, 1, 0],

        [500, 10, 50, 3, 1],
        [800, 16, 50, 3, 1],
        [1200, 24, 50, 3, 1],

        [500, 10, 50, 2, 2],
        [800, 16, 50, 2, 2],
        [1200, 24, 50, 2, 2]
    ])

    # --------------------------------------------------------
    # Actual delay
    #
    # accident:
    #
    # delay = normal_time * 3 - normal_time
    #
    # construction:
    #
    # delay = normal_time * 2 - normal_time
    #
    # normal:
    #
    # delay = 0
    # --------------------------------------------------------

    y = np.array([

        0,
        0,
        0,

        100,
        120,
        100,

        50,
        50,
        50,

        0,
        0,
        0,

        100,
        100,
        100,

        50,
        50,
        50
    ])

    # --------------------------------------------------------
    # Create NN
    # --------------------------------------------------------

    nn = DelayNN()

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    nn.train(
        X,
        y
    )

    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    predicted = nn.predict_delay(

        distance=600,

        avg_speed=10,

        normal_time=60,

        disruption_factor=3,

        disruption_type="accident"
    )

    print(
        "Predicted delay:",
        round(predicted, 2),
        "seconds"
    )
