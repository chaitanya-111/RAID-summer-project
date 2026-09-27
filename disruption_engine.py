import random


class DisruptionEngine:

    def __init__(self, graph):

        self.graph = graph

        # ====================================================
        # SAVE ORIGINAL EDGE DATA
        # ====================================================

        self.original_graph = {}

        for u, v, data in graph.edges(data=True):

            normal_time = data.get(
                "normal_time",
                data.get("weight", 1.0)
            )

            self.original_graph[(u, v)] = {
                "distance": data.get("distance", 1.0),
                "avg_speed": data.get("avg_speed", 1.0),
                "normal_time": normal_time,
                "weight": data.get("weight", normal_time),
                "blocked": data.get("blocked", False),
                "highway": data.get("highway"),
                "disruption": data.get("disruption")
            }

        # Currently active disruptions
        self.active_disruptions = {}

    # ========================================================
    # BLOCK ROAD
    # ========================================================

    def block_road(self, u, v):

        if not self.graph.has_edge(u, v):
            return False

        # Block forward edge
        self.graph[u][v]["blocked"] = True
        self.graph[u][v]["disruption"] = "blocked"

        # Block reverse edge if it exists
        if self.graph.has_edge(v, u):

            self.graph[v][u]["blocked"] = True
            self.graph[v][u]["disruption"] = "blocked"

        self.active_disruptions[(u, v)] = {
            "type": "blocked",
            "factor": None
        }

        return True

    # ========================================================
    # ADD ACCIDENT
    # ========================================================

    def add_accident(self, u, v, factor=3):

        if not self.graph.has_edge(u, v):
            return False

        if (u, v) not in self.original_graph:
            return False

        # ----------------------------------------------------
        # FORWARD EDGE
        # ----------------------------------------------------

        original = self.original_graph[(u, v)]

        normal_time = original["normal_time"]

        disrupted_time = normal_time * factor

        self.graph[u][v]["weight"] = disrupted_time
        self.graph[u][v]["blocked"] = False
        self.graph[u][v]["disruption"] = "accident"

        # ----------------------------------------------------
        # REVERSE EDGE
        # ----------------------------------------------------

        if self.graph.has_edge(v, u):

            if (v, u) in self.original_graph:

                reverse_original = self.original_graph[(v, u)]

                reverse_normal_time = (
                    reverse_original["normal_time"]
                )

                reverse_disrupted_time = (
                    reverse_normal_time * factor
                )

                self.graph[v][u]["weight"] = (
                    reverse_disrupted_time
                )

                self.graph[v][u]["blocked"] = False

                self.graph[v][u]["disruption"] = (
                    "accident"
                )

        # ----------------------------------------------------
        # STORE DISRUPTION
        # ----------------------------------------------------

        self.active_disruptions[(u, v)] = {
            "type": "accident",
            "factor": factor
        }

        return True

    # ========================================================
    # GET DELAY OF ONE EDGE
    # ========================================================

    def get_edge_delay(self, u, v):

        """
        Returns delay of one edge.

        Normal road:
            delay = 0

        Accident:
            delay = current_time - normal_time

        Blocked:
            returns normal_time

        The blocked value is used as a finite penalty for
        NN training. A blocked road itself cannot be travelled.
        """

        if not self.graph.has_edge(u, v):
            return 0.0

        edge = self.graph[u][v]

        normal_time = edge.get(
            "normal_time",
            edge.get("weight", 0.0)
        )

        current_time = edge.get(
            "weight",
            normal_time
        )

        # ----------------------------------------------------
        # BLOCKED ROAD
        # ----------------------------------------------------

        if edge.get("blocked", False):

            return max(
                0.0,
                normal_time
            )

        # ----------------------------------------------------
        # NORMAL / ACCIDENT
        # ----------------------------------------------------

        delay = current_time - normal_time

        return max(
            0.0,
            delay
        )

    # ========================================================
    # CALCULATE DELAY FOR A ROUTE
    # ========================================================

    def calculate_route_delay(self, route):

        """
        Calculates total delay of a route.

        For a usable route:

            delay =
                current route time
                -
                normal route time

        If route contains a blocked road:
            returns None

        This avoids infinity errors.
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

            if not self.graph.has_edge(u, v):

                return 0.0

            edge = self.graph[u][v]

            # ------------------------------------------------
            # BLOCKED ROUTE
            # ------------------------------------------------

            if edge.get("blocked", False):

                return None

            # ------------------------------------------------
            # NORMAL TIME
            # ------------------------------------------------

            normal_time = edge.get(
                "normal_time",
                edge.get("weight", 0.0)
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

        delay = current_total - normal_total

        return max(
            0.0,
            delay
        )

    # ========================================================
    # GET ROUTE TIMES
    # ========================================================

    def get_route_times(self, route):

        """
        Returns:

            normal_time
            current_time
            delay
        """

        if route is None or len(route) < 2:

            return {
                "normal_time": 0.0,
                "current_time": 0.0,
                "delay": 0.0
            }

        normal_total = 0.0
        current_total = 0.0

        for u, v in zip(
            route[:-1],
            route[1:]
        ):

            if not self.graph.has_edge(u, v):

                return {
                    "normal_time": 0.0,
                    "current_time": 0.0,
                    "delay": 0.0
                }

            edge = self.graph[u][v]

            # ------------------------------------------------
            # BLOCKED ROAD
            # ------------------------------------------------

            if edge.get("blocked", False):

                return {
                    "normal_time": normal_total,
                    "current_time": None,
                    "delay": None
                }

            normal_time = edge.get(
                "normal_time",
                edge.get("weight", 0.0)
            )

            current_time = edge.get(
                "weight",
                normal_time
            )

            normal_total += normal_time

            current_total += current_time

        delay = max(
            0.0,
            current_total - normal_total
        )

        return {
            "normal_time": normal_total,
            "current_time": current_total,
            "delay": delay
        }

    # ========================================================
    # RESTORE ONE ROAD
    # ========================================================

    def restore_road(self, u, v):

        if (u, v) not in self.active_disruptions:
            return False

        # ----------------------------------------------------
        # RESTORE u -> v
        # ----------------------------------------------------

        if self.graph.has_edge(u, v):

            original = self.original_graph.get((u, v))

            if original:

                self.graph[u][v]["distance"] = (
                    original["distance"]
                )

                self.graph[u][v]["avg_speed"] = (
                    original["avg_speed"]
                )

                self.graph[u][v]["normal_time"] = (
                    original["normal_time"]
                )

                self.graph[u][v]["weight"] = (
                    original["weight"]
                )

                self.graph[u][v]["blocked"] = (
                    original["blocked"]
                )

                self.graph[u][v]["highway"] = (
                    original["highway"]
                )

                self.graph[u][v]["disruption"] = (
                    original["disruption"]
                )

        # ----------------------------------------------------
        # RESTORE v -> u
        # ----------------------------------------------------

        if self.graph.has_edge(v, u):

            original = self.original_graph.get((v, u))

            if original:

                self.graph[v][u]["distance"] = (
                    original["distance"]
                )

                self.graph[v][u]["avg_speed"] = (
                    original["avg_speed"]
                )

                self.graph[v][u]["normal_time"] = (
                    original["normal_time"]
                )

                self.graph[v][u]["weight"] = (
                    original["weight"]
                )

                self.graph[v][u]["blocked"] = (
                    original["blocked"]
                )

                self.graph[v][u]["highway"] = (
                    original["highway"]
                )

                self.graph[v][u]["disruption"] = (
                    original["disruption"]
                )

        # Remove active disruption
        del self.active_disruptions[(u, v)]

        return True

    # ========================================================
    # RESTORE ALL
    # ========================================================

    def restore_all(self):

        active_edges = list(
            self.active_disruptions.keys()
        )

        for u, v in active_edges:

            self.restore_road(
                u,
                v
            )

        self.active_disruptions.clear()

    # ========================================================
    # GET RANDOM ROAD
    # ========================================================

    def get_random_road(self):

        available_edges = []

        for u, v in self.graph.edges():

            if (u, v) not in self.active_disruptions:

                available_edges.append(
                    (u, v)
                )

        if not available_edges:
            return None

        return random.choice(
            available_edges
        )

    # ========================================================
    # CREATE RANDOM DISRUPTION
    # ========================================================

    def create_random_disruption(self):

        """
        Randomly creates:

            accident
            OR
            blocked road
        """

        road = self.get_random_road()

        if road is None:
            return None

        u, v = road

        disruption_type = random.choice(
            [
                "accident",
                "blocked"
            ]
        )

        # ----------------------------------------------------
        # ACCIDENT
        # ----------------------------------------------------

        if disruption_type == "accident":

            success = self.add_accident(
                u,
                v,
                factor=3
            )

        # ----------------------------------------------------
        # BLOCKED
        # ----------------------------------------------------

        else:

            success = self.block_road(
                u,
                v
            )

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
