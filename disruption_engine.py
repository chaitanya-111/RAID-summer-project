import random


class DisruptionEngine:

    def __init__(self, graph):
        """
        graph:
            NetworkX DiGraph used by the traffic simulator.
        """

        self.graph = graph

        # Store the original state of every edge
        self.original_graph = {}

        for u, v, data in graph.edges(data=True):

            self.original_graph[(u, v)] = {
                "weight": data.get("weight", 1.0),
                "blocked": data.get("blocked", False),
                "highway": data.get("highway")
            }

        # Currently active disruptions
        #
        # {
        #     (u, v): {
        #         "type": "accident",
        #         "remaining": 300
        #     }
        # }
        self.active_disruptions = {}

    # ========================================================
    # BLOCK ROAD
    # ========================================================

    def block_road(self, u, v):

        if not self.graph.has_edge(u, v):
            return False

        self.graph[u][v]["blocked"] = True

        # If the road is two-way, block reverse direction too
        if self.graph.has_edge(v, u):
            self.graph[v][u]["blocked"] = True

        self.active_disruptions[(u, v)] = {
            "type": "blocked"
        }

        return True

    # ========================================================
    # ADD ACCIDENT
    # ========================================================

    def add_accident(self, u, v, factor=3):

        if not self.graph.has_edge(u, v):
            return False

        original_weight = self.original_graph[
            (u, v)
        ]["weight"]

        self.graph[u][v]["weight"] = (
            original_weight * factor
        )

        self.graph[u][v]["blocked"] = False

        if self.graph.has_edge(v, u):

            if (v, u) in self.original_graph:

                reverse_original_weight = (
                    self.original_graph[
                        (v, u)
                    ]["weight"]
                )

                self.graph[v][u]["weight"] = (
                    reverse_original_weight * factor
                )

                self.graph[v][u]["blocked"] = False

        self.active_disruptions[(u, v)] = {
            "type": "accident"
        }

        return True

    # ========================================================
    # ADD CONSTRUCTION
    # ========================================================

    def add_construction(self, u, v, factor=2):

        if not self.graph.has_edge(u, v):
            return False

        original_weight = self.original_graph[
            (u, v)
        ]["weight"]

        self.graph[u][v]["weight"] = (
            original_weight * factor
        )

        self.graph[u][v]["blocked"] = False

        if self.graph.has_edge(v, u):

            if (v, u) in self.original_graph:

                reverse_original_weight = (
                    self.original_graph[
                        (v, u)
                    ]["weight"]
                )

                self.graph[v][u]["weight"] = (
                    reverse_original_weight * factor
                )

                self.graph[v][u]["blocked"] = False

        self.active_disruptions[(u, v)] = {
            "type": "construction"
        }

        return True

    # ========================================================
    # RESTORE ROAD
    # ========================================================

    def restore_road(self, u, v):

        if (u, v) not in self.active_disruptions:
            return False

        # Restore u -> v
        if self.graph.has_edge(u, v):

            original = self.original_graph.get(
                (u, v)
            )

            if original:

                self.graph[u][v]["weight"] = (
                    original["weight"]
                )

                self.graph[u][v]["blocked"] = (
                    original["blocked"]
                )

        # Restore v -> u
        if self.graph.has_edge(v, u):

            original = self.original_graph.get(
                (v, u)
            )

            if original:

                self.graph[v][u]["weight"] = (
                    original["weight"]
                )

                self.graph[v][u]["blocked"] = (
                    original["blocked"]
                )

        del self.active_disruptions[(u, v)]

        return True

    # ========================================================
    # RESTORE ALL
    # ========================================================

    def restore_all(self):

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
    # RANDOM DISRUPTION
    # ========================================================

    def create_random_disruption(self):

        road = self.get_random_road()

        if road is None:
            return None

        u, v = road

        disruption_type = random.choice([
            "blocked",
            "accident",
            "construction"
        ])

        if disruption_type == "blocked":

            success = self.block_road(
                u,
                v
            )

        elif disruption_type == "accident":

            success = self.add_accident(
                u,
                v,
                factor=3
            )

        else:

            success = self.add_construction(
                u,
                v,
                factor=2
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