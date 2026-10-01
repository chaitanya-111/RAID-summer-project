class DisruptionEngine:

    def __init__(self, graph):
        self.graph = graph
        self.original_graph = {
            node: list(edges)
            for node, edges in graph.items()
        }
    #Creates a copy of the original graph.

        self.active_disruptions = {}
        #Dictionary storing currently active disruptions

    
    def block_road(self, u, v):

        if u not in self.graph:
            return False
        #If node  doesn't exist → stop

        found = False
        #Initially assume the road doesn't exist

        self.graph[u] = [
            (neighbor, weight)
            for neighbor, weight in self.graph[u]
            if not (neighbor == v and (found := True))
        ]
        #removes the road so both directions are removed

        if v in self.graph:
            self.graph[v] = [
                (neighbor, weight)
                for neighbor, weight in self.graph[v]
                if neighbor != u
            ]
        #Remove reverse road

        if found:
            self.active_disruptions[(u, v)] = "blocked"
            return True

        return False

        # Accident doesn't remove the road,
    # it makes the road more costly/slower
    def add_accident(self, u, v, factor=3):

        if u not in self.graph:
            return False

        found = False

        # Change u -> v
        new_edges = []

        for neighbor, weight in self.graph[u]:
            if neighbor == v:
                new_edges.append((neighbor, weight * factor))
                found = True
            else:
                new_edges.append((neighbor, weight))

        self.graph[u] = new_edges

        # Change v -> u
        if v in self.graph:
            new_edges = []

            for neighbor, weight in self.graph[v]:
                if neighbor == u:
                    new_edges.append((neighbor, weight * factor))
                else:
                    new_edges.append((neighbor, weight))

            self.graph[v] = new_edges

        if found:
            self.active_disruptions[(u, v)] = "accident"

        return found


    # Increases the road cost
    def add_construction(self, u, v, factor=2):

        if u not in self.graph:
            return False

        found = False

        # Change u -> v
        new_edges = []

        for neighbor, weight in self.graph[u]:
            if neighbor == v:
                new_edges.append((neighbor, weight * factor))
                found = True
            else:
                new_edges.append((neighbor, weight))

        self.graph[u] = new_edges

        # Change v -> u
        if v in self.graph:
            new_edges = []

            for neighbor, weight in self.graph[v]:
                if neighbor == u:
                    new_edges.append((neighbor, weight * factor))
                else:
                    new_edges.append((neighbor, weight))

            self.graph[v] = new_edges

        if found:
            self.active_disruptions[(u, v)] = "construction"

        return found

    #Restores a disrupted road to its original state
    def restore_road(self, u, v):

        if (u, v) not in self.active_disruptions:
            return False
     #there wasn't a disruption → nothing to restore

        if u in self.original_graph:

            original_edges = self.original_graph[u]

            self.graph[u] = [
                (neighbor, weight)
                for neighbor, weight in self.graph[u]
                if neighbor != v
            ]

            for neighbor, weight in original_edges:
                if neighbor == v:
                    self.graph[u].append((v, weight))

        if v in self.original_graph:

            original_edges = self.original_graph[v]

            self.graph[v] = [
                (neighbor, weight)
                for neighbor, weight in self.graph[v]
                if neighbor != u
            ]

            for neighbor, weight in original_edges:
                if neighbor == u:
                    self.graph[v].append((u, weight))
                    #restore removes the current version and puts back original values

        del self.active_disruptions[(u, v)]

        return True

    # -------------------------------
    # RESTORE ALL DISRUPTIONS
    # -------------------------------
    def restore_all(self):
        #Restores every road

        self.graph.clear()

        for node, edges in self.original_graph.items():
            self.graph[node] = list(edges)

        self.active_disruptions.clear()
