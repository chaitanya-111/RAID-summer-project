import random


class DisruptionEngine:

    def __init__(self, graph):
        """
        graph:
            NetworkX DiGraph used by the traffic simulator.

        Every edge contains:

            distance      -> physical road distance in meters
            avg_speed     -> average road speed in m/s
            normal_time   -> travel time without disruption
            weight        -> current travel time used by Dijkstra
            blocked       -> whether road is blocked
            highway       -> road type
            disruption   -> current disruption type
        """

        self.graph = graph

        # ====================================================
        # STORE ORIGINAL STATE OF EVERY EDGE
        # ====================================================

        self.original_graph = {}

        for u, v, data in graph.edges(data=True):

            self.original_graph[(u, v)] = {

                # Physical road distance
                "distance": data.get(
                    "distance",
                    1.0
                ),

                # Average speed of vehicles on this road
                "avg_speed": data.get(
                    "avg_speed",
                    1.0
                ),

                # Travel time without any disruption
                "normal_time": data.get(
                    "normal_time",
                    data.get(
                        "weight",
                        1.0
                    )
                ),

                # Original weight
                # Initially this is equal to normal_time
                "weight": data.get(
                    "weight",
                    1.0
                ),

                # Original blocked state
                "blocked": data.get(
                    "blocked",
                    False
                ),

                # Road type
                "highway": data.get(
                    "highway"
                ),

                # Original disruption state
                "disruption": data.get(
                    "disruption"
                )
            }

        # ====================================================
        # CURRENT ACTIVE DISRUPTIONS
        # ====================================================

        self.active_disruptions = {}

    # ========================================================
    # BLOCK ROAD
    # ========================================================

    def block_road(self, u, v):

        """
        Completely blocks a road.

        Both directions are blocked if the reverse
        edge exists.

        The normal_time is NOT changed.

        Only the blocked status changes.
        """

        if not self.graph.has_edge(u, v):

            return False

        # ----------------------------------------------------
        # Block u -> v
        # ----------------------------------------------------

        self.graph[u][v]["blocked"] = True

        self.graph[u][v]["disruption"] = "blocked"

        # ----------------------------------------------------
        # Block reverse direction if it exists
        # ----------------------------------------------------

        if self.graph.has_edge(v, u):

            self.graph[v][u]["blocked"] = True

            self.graph[v][u]["disruption"] = "blocked"

        # ----------------------------------------------------
        # Store disruption
        # ----------------------------------------------------

        self.active_disruptions[(u, v)] = {

            "type": "blocked",

            "factor": None,

            "delay_per_edge": float("inf")
        }

        return True

    # ========================================================
    # ADD ACCIDENT
    # ========================================================

    def add_accident(
        self,
        u,
        v,
        factor=3
    ):

        """
        Accident increases travel time.

        Example:

            normal_time = 20 seconds
            factor = 3

            disrupted_time = 60 seconds

        normal_time remains unchanged.
        weight becomes the disrupted travel time.
        """

        if not self.graph.has_edge(u, v):

            return False

        # ====================================================
        # FORWARD EDGE
        # ====================================================

        if (u, v) not in self.original_graph:

            return False

        original = self.original_graph[
            (u, v)
        ]

        normal_time = original[
            "normal_time"
        ]

        disrupted_time = (
            normal_time * factor
        )

        ##update current graph

        self.graph[u][v]["weight"] = (
            disrupted_time
        )

        self.graph[u][v]["blocked"] = False

        self.graph[u][v]["disruption"] = (
            "accident"
        )

        # ====================================================
        # REVERSE EDGE
        # ====================================================

        reverse_delay = (
            disrupted_time
            -
            normal_time
        )

        if self.graph.has_edge(v, u):

            if (v, u) in self.original_graph:

                reverse_original = (
                    self.original_graph[
                        (v, u)
                    ]
                )

                reverse_normal_time = (
                    reverse_original[
                        "normal_time"
                    ]
                )

                reverse_disrupted_time = (
                    reverse_normal_time
                    *
                    factor
                )

                self.graph[v][u][
                    "weight"
                ] = (
                    reverse_disrupted_time
                )

                self.graph[v][u][
                    "blocked"
                ] = False

                self.graph[v][u][
                    "disruption"
                ] = "accident"

        # ====================================================
        # STORE ACTIVE DISRUPTION
        # ====================================================

        self.active_disruptions[(u, v)] = {

            "type": "accident",

            "factor": factor,

            "delay_per_edge": reverse_delay
        }

        return True

    # ========================================================
    # ADD CONSTRUCTION
    # ========================================================

    def add_construction(
        self,
        u,
        v,
        factor=2
    ):

        """
        Construction increases travel time.

        Example:

            normal_time = 20 seconds
            factor = 2

            disrupted_time = 40 seconds
        """

        if not self.graph.has_edge(u, v):

            return False

        # ====================================================
        # FORWARD EDGE
        # ====================================================

        if (u, v) not in self.original_graph:

            return False

        original = self.original_graph[
            (u, v)
        ]

        normal_time = original[
            "normal_time"
        ]

        disrupted_time = (
            normal_time * factor
        )

        # ----------------------------------------------------
        # Update current graph
        # ----------------------------------------------------

        self.graph[u][v]["weight"] = (
            disrupted_time
        )

        self.graph[u][v]["blocked"] = False

        self.graph[u][v]["disruption"] = (
            "construction"
        )

        # ====================================================
        # REVERSE EDGE
        # ====================================================

        if self.graph.has_edge(v, u):

            if (v, u) in self.original_graph:

                reverse_original = (
                    self.original_graph[
                        (v, u)
                    ]
                )

                reverse_normal_time = (
                    reverse_original[
                        "normal_time"
                    ]
                )

                reverse_disrupted_time = (
                    reverse_normal_time
                    *
                    factor
                )

                self.graph[v][u][
                    "weight"
                ] = (
                    reverse_disrupted_time
                )

                self.graph[v][u][
                    "blocked"
                ] = False

                self.graph[v][u][
                    "disruption"
                ] = "construction"

        # ====================================================
        # STORE ACTIVE DISRUPTION
        # ====================================================

        self.active_disruptions[(u, v)] = {

            "type": "construction",

            "factor": factor,

            "delay_per_edge": (
                disrupted_time
                -
                normal_time
            )
        }

        return True

    # ========================================================
    # CALCULATE EDGE DELAY
    # ========================================================

    def get_edge_delay(self, u, v):

        """
        Returns:

            current travel time
            -
            normal travel time
        """

        if not self.graph.has_edge(u, v):

            return float("inf")

        current_weight = self.graph[u][v].get(
            "weight",
            0.0
        )

        normal_time = self.graph[u][v].get(
            "normal_time",
            current_weight
        )

        if self.graph[u][v].get(
            "blocked",
            False
        ):

            return float("inf")

        return max(
            0.0,
            current_weight
            -
            normal_time
        )

    # ========================================================
    # CALCULATE ROUTE DELAY
    # ========================================================

    def calculate_route_delay(
        self,
        route
    ):

        """
        Calculates total delay for an entire route.

        delay =
            disrupted route time
            -
            normal route time
        """

        if not route or len(route) < 2:

            return 0.0

        normal_time = 0.0

        current_time = 0.0

        for u, v in zip(
            route[:-1],
            route[1:]
        ):

            if not self.graph.has_edge(
                u,
                v
            ):

                return float("inf")

            edge = self.graph[u][v]

            # ------------------------------------------------
            # If road is blocked
            # ------------------------------------------------

            if edge.get(
                "blocked",
                False
            ):

                return float("inf")

            # ------------------------------------------------
            # Normal travel time
            # ------------------------------------------------

            normal_time += edge.get(
                "normal_time",
                edge.get(
                    "weight",
                    0.0
                )
            )

            # ------------------------------------------------
            # Current/disrupted travel time
            # ------------------------------------------------

            current_time += edge.get(
                "weight",
                edge.get(
                    "normal_time",
                    0.0
                )
            )

        # ----------------------------------------------------
        # Delay
        # ----------------------------------------------------

        delay = max(
            0.0,
            current_time
            -
            normal_time
        )

        return delay

    # ========================================================
    # GET ROUTE TIMES
    # ========================================================

    def get_route_times(
        self,
        route
    ):

        """
        Returns:

            normal_time
            current_time
            delay
        """

        if not route or len(route) < 2:

            return {
                "normal_time": 0.0,
                "current_time": 0.0,
                "delay": 0.0
            }

        normal_time = 0.0

        current_time = 0.0

        for u, v in zip(
            route[:-1],
            route[1:]
        ):

            if not self.graph.has_edge(
                u,
                v
            ):

                return {
                    "normal_time": float("inf"),
                    "current_time": float("inf"),
                    "delay": float("inf")
                }

            edge = self.graph[u][v]

            if edge.get(
                "blocked",
                False
            ):

                return {
                    "normal_time": (
                        normal_time
                        +
                        edge.get(
                            "normal_time",
                            0.0
                        )
                    ),
                    "current_time": float("inf"),
                    "delay": float("inf")
                }

            normal_time += edge.get(
                "normal_time",
                edge.get(
                    "weight",
                    0.0
                )
            )

            current_time += edge.get(
                "weight",
                edge.get(
                    "normal_time",
                    0.0
                )
            )

        delay = max(
            0.0,
            current_time
            -
            normal_time
        )

        return {
            "normal_time": normal_time,
            "current_time": current_time,
            "delay": delay
        }

    # ========================================================
    # RESTORE ROAD
    # ========================================================

    def restore_road(
        self,
        u,
        v
    ):

        """
        Restores the road completely to its original state.

        This restores:

            distance
            avg_speed
            normal_time
            weight
            blocked
            highway
            disruption
        """

        if (
            u,
            v
        ) not in self.active_disruptions:

            return False

        # ====================================================
        # RESTORE u -> v
        # ====================================================

        if self.graph.has_edge(
            u,
            v
        ):

            original = (
                self.original_graph.get(
                    (u, v)
                )
            )

            if original:

                self.graph[u][v][
                    "distance"
                ] = original[
                    "distance"
                ]

                self.graph[u][v][
                    "avg_speed"
                ] = original[
                    "avg_speed"
                ]

                self.graph[u][v][
                    "normal_time"
                ] = original[
                    "normal_time"
                ]

                self.graph[u][v][
                    "weight"
                ] = original[
                    "weight"
                ]

                self.graph[u][v][
                    "blocked"
                ] = original[
                    "blocked"
                ]

                self.graph[u][v][
                    "highway"
                ] = original[
                    "highway"
                ]

                self.graph[u][v][
                    "disruption"
                ] = original[
                    "disruption"
                ]

        # ====================================================
        # RESTORE v -> u
        # ====================================================

        if self.graph.has_edge(
            v,
            u
        ):

            original = (
                self.original_graph.get(
                    (v, u)
                )
            )

            if original:

                self.graph[v][u][
                    "distance"
                ] = original[
                    "distance"
                ]

                self.graph[v][u][
                    "avg_speed"
                ] = original[
                    "avg_speed"
                ]

                self.graph[v][u][
                    "normal_time"
                ] = original[
                    "normal_time"
                ]

                self.graph[v][u][
                    "weight"
                ] = original[
                    "weight"
                ]

                self.graph[v][u][
                    "blocked"
                ] = original[
                    "blocked"
                ]

                self.graph[v][u][
                    "highway"
                ] = original[
                    "highway"
                ]

                self.graph[v][u][
                    "disruption"
                ] = original[
                    "disruption"
                ]

        # ====================================================
        # REMOVE ACTIVE DISRUPTION
        # ====================================================

        del self.active_disruptions[
            (u, v)
        ]

        return True

    # ========================================================
    # RESTORE ALL
    # ========================================================

    def restore_all(self):

        """
        Restore every active disruption.
        """

        for edge in list(
            self.active_disruptions.keys()
        ):

            self.restore_road(
                edge[0],
                edge[1]
            )

        self.active_disruptions.clear()

    # ========================================================
    # RANDOM ROAD
    # ========================================================

    def get_random_road(self):

        """
        Returns a random road that currently
        has no active disruption.
        """

        available_edges = []

        for u, v in self.graph.edges():

            if (
                u,
                v
            ) not in self.active_disruptions:

                available_edges.append(
                    (u, v)
                )

        if not available_edges:

            return None

        return random.choice(
            available_edges
        )

    # ========================================================
    # RANDOM DISRUPTION
    # ========================================================

    def create_random_disruption(self):

        """
        Randomly creates one of:

            blocked
            accident
            construction
        """

        road = self.get_random_road()

        if road is None:

            return None

        u, v = road

        disruption_type = random.choice(
            [
                "blocked",
                "accident",
                "construction"
            ]
        )

        # ====================================================
        # BLOCKED
        # ====================================================

        if disruption_type == "blocked":

            success = self.block_road(
                u,
                v
            )

        # ====================================================
        # ACCIDENT
        # ====================================================

        elif disruption_type == "accident":

            success = self.add_accident(
                u,
                v,
                factor=3
            )

        # ====================================================
        # CONSTRUCTION
        # ====================================================

        else:

            success = self.add_construction(
                u,
                v,
                factor=2
            )

        # ====================================================
        # RETURN RESULT
        # ====================================================

        if success:

            return {
                "road": (u, v),
                "type": disruption_type
            }

        return None

    # ========================================================
    # GET ACTIVE DISRUPTIONS
    # ========================================================

    def get_active_disruptions(self):

        return self.active_disruptions.copy()
