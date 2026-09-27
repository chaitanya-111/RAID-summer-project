import math
import random
import osmium
import networkx as nx
import pygame

from disruption_engine import DisruptionEngine


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800

NUM_CARS = 200

PBF_FILE_PATH = (
    "/Users/suryanshu/Downloads/"
    "8b5c1ba1-dd19-48de-8132-3814e65046a8.osm.pbf"
)


# ============================================================
# DISRUPTION CONFIGURATION
# ============================================================

# How often a new random disruption is created
DISRUPTION_INTERVAL = 10.0

# How long a disruption remains active
DISRUPTION_DURATION = 15.0

# Probability of each disruption type
BLOCK_PROBABILITY = 0.40
ACCIDENT_PROBABILITY = 0.35
CONSTRUCTION_PROBABILITY = 0.25

# Accident makes the road this many times more expensive
ACCIDENT_FACTOR = 3

# Construction makes the road this many times more expensive
CONSTRUCTION_FACTOR = 2


# ============================================================
# COLORS
# ============================================================

COLOR_BG = (25, 25, 30)

COLOR_ROAD = (80, 85, 95)

COLOR_ROAD_BLOCKED = (220, 50, 50)

COLOR_NODE = (100, 100, 110)

COLOR_TEXT = (255, 255, 255)

COLOR_PANEL = (35, 35, 42)

COLOR_ACCIDENT = (255, 140, 0)

COLOR_CONSTRUCTION = (255, 235, 59)

COLOR_BLOCKED = (220, 50, 50)


# ============================================================
# ROUTE COLORS
# ============================================================

ROUTE_COLORS = [
    (255, 87, 34),      # Bright Orange
    (156, 39, 176),     # Purple
    (0, 230, 118),      # Neon Green
    (255, 235, 59),     # Yellow
    (0, 176, 255),      # Cyan
    (255, 64, 129),     # Pink
    (121, 85, 72),      # Brown
    (63, 81, 181),      # Indigo
]


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lon1,
    lat1,
    lon2,
    lat2
):

    R = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(
        lat2 - lat1
    )

    dlambda = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(dphi / 2) ** 2
        +
        math.cos(phi1)
        *
        math.cos(phi2)
        *
        math.sin(dlambda / 2) ** 2
    )

    return (
        2
        * R
        * math.atan2(
            math.sqrt(a),
            math.sqrt(1 - a)
        )
    )


# ============================================================
# PBF HANDLER
# ============================================================

class PBFHandler(osmium.SimpleHandler):

    def __init__(self):

        super().__init__()

        self.nodes = {}

        self.graph = nx.DiGraph()

        self.roundabout_nodes = set()

    # --------------------------------------------------------
    # NODE
    # --------------------------------------------------------

    def node(self, n):

        self.nodes[n.id] = (
            n.location.lon,
            n.location.lat
        )

    # --------------------------------------------------------
    # WAY
    # --------------------------------------------------------

    def way(self, w):

        highway = w.tags.get(
            "highway"
        )

        junction = w.tags.get(
            "junction"
        )

        oneway_tag = w.tags.get(
            "oneway"
        )

        is_roundabout = (
            junction == "roundabout"
        )

        is_oneway = (
            is_roundabout
            or
            oneway_tag in [
                "yes",
                "1",
                "true"
            ]
        )

        if highway in [
            "primary",
            "secondary",
            "tertiary",
            "unclassified",
            "trunk",
            "motorway"
        ]:

            way_nodes = [
                node.ref
                for node in w.nodes
                if node.ref in self.nodes
            ]

            if len(way_nodes) < 2:
                return

            if is_roundabout:

                self.roundabout_nodes.update(
                    way_nodes
                )

            for u, v in zip(
                way_nodes[:-1],
                way_nodes[1:]
            ):

                lon1, lat1 = self.nodes[u]

                lon2, lat2 = self.nodes[v]

                dist = haversine_distance(
                    lon1,
                    lat1,
                    lon2,
                    lat2
                )

                self.graph.add_node(
                    u,
                    pos=self.nodes[u]
                )

                self.graph.add_node(
                    v,
                    pos=self.nodes[v]
                )

                # Forward direction
                self.graph.add_edge(
                    u,
                    v,
                    weight=dist,
                    blocked=False,
                    highway=highway,
                    disruption=None
                )

                # Reverse direction
                if not is_oneway:

                    self.graph.add_edge(
                        v,
                        u,
                        weight=dist,
                        blocked=False,
                        highway=highway,
                        disruption=None
                    )


# ============================================================
# LOAD LOCAL PBF
# ============================================================

def load_local_pbf(
    pbf_filepath
):

    print(
        f"Parsing local file: "
        f"{pbf_filepath} ..."
    )

    handler = PBFHandler()

    handler.apply_file(
        pbf_filepath,
        locations=True
    )

    G = handler.graph

    if len(G) > 0:

        largest_wcc = max(
            nx.weakly_connected_components(G),
            key=len
        )

        G_sub = nx.DiGraph()

        for node in largest_wcc:

            G_sub.add_node(
                node,
                pos=G.nodes[node]["pos"]
            )

        for u, v, data in G.edges(
            largest_wcc,
            data=True
        ):

            G_sub.add_edge(
                u,
                v,
                **data
            )

        G = G_sub

    valid_roundabouts = [
        n
        for n in handler.roundabout_nodes
        if n in G
    ]

    print(
        f"Done!"
    )

    print(
        f"Graph contains "
        f"{len(G.nodes)} nodes and "
        f"{len(G.edges)} directed road segments."
    )

    print(
        f"Found "
        f"{len(valid_roundabouts)} "
        f"roundabout nodes."
    )

    return (
        G,
        valid_roundabouts
    )


# ============================================================
# CONVERT GPS COORDINATES TO SCREEN COORDINATES
# ============================================================

def convert_coords_to_world(
    G,
    width,
    height,
    padding=50
):

    positions = nx.get_node_attributes(
        G,
        "pos"
    )

    lons = [
        p[0]
        for p in positions.values()
    ]

    lats = [
        p[1]
        for p in positions.values()
    ]

    min_lon = min(lons)
    max_lon = max(lons)

    min_lat = min(lats)
    max_lat = max(lats)

    world_pos = {}

    for node, (lon, lat) in positions.items():

        if max_lon == min_lon:

            x = width / 2

        else:

            x = (
                padding
                +
                (
                    (lon - min_lon)
                    /
                    (max_lon - min_lon)
                )
                *
                (width - 2 * padding)
            )

        if max_lat == min_lat:

            y = height / 2

        else:

            y = height - (
                padding
                +
                (
                    (lat - min_lat)
                    /
                    (max_lat - min_lat)
                )
                *
                (height - 2 * padding)
            )

        world_pos[node] = (
            x,
            y
        )

    return world_pos


# ============================================================
# CAR CLASS
# ============================================================

class Car:

    def __init__(
        self,
        graph,
        world_pos,
        start_node,
        dest_node,
        route_id,
        color
    ):

        self.graph = graph

        self.world_pos = world_pos

        self.start_node = start_node

        self.dest_node = dest_node

        self.route_id = route_id

        self.color = color

        # Random speed for each car
        self.speed = random.uniform(
            2.0,
            5.0
        )

        self.current_node = (
            start_node
        )

        self.path = []

        self.path_index = 0

        self.next_node = (
            start_node
        )

        self.pos = list(
            self.world_pos[
                self.current_node
            ]
        )

        self.recalculate_path()

    # ========================================================
    # RECALCULATE PATH
    # ========================================================

    def recalculate_path(self):

        if (
            self.current_node is None
            or self.dest_node is None
        ):

            self.path = []

            return

        def weight_func(
            u,
            v,
            d
        ):

            if d.get(
                "blocked",
                False
            ):

                return float("inf")

            return d.get(
                "weight",
                1.0
            )

        try:

            full_path = nx.dijkstra_path(
                self.graph,
                self.current_node,
                self.dest_node,
                weight=weight_func
            )

            self.path = full_path

            self.path_index = 0

            if len(self.path) > 1:

                self.next_node = (
                    self.path[1]
                )

            else:

                self.next_node = (
                    self.current_node
                )

        except (
            nx.NetworkXNoPath,
            nx.NodeNotFound
        ):

            self.path = [
                self.current_node
            ]

            self.next_node = (
                self.current_node
            )

    # ========================================================
    # UPDATE
    # ========================================================

    def update(self):

        # Destination reached
        if (
            not self.path
            or
            self.current_node
            == self.dest_node
        ):

            self.current_node = (
                self.start_node
            )

            self.pos = list(
                self.world_pos[
                    self.start_node
                ]
            )

            self.recalculate_path()

            return

        # ----------------------------------------------------
        # Check whether current road became blocked
        # ----------------------------------------------------

        if self.graph.has_edge(
            self.current_node,
            self.next_node
        ):

            if self.graph[
                self.current_node
            ][
                self.next_node
            ].get(
                "blocked",
                False
            ):

                self.recalculate_path()

                return

        # ----------------------------------------------------
        # Move toward next node
        # ----------------------------------------------------

        target_x, target_y = (
            self.world_pos[
                self.next_node
            ]
        )

        dx = (
            target_x
            - self.pos[0]
        )

        dy = (
            target_y
            - self.pos[1]
        )

        distance = math.hypot(
            dx,
            dy
        )

        if distance < self.speed:

            self.pos = [
                target_x,
                target_y
            ]

            self.current_node = (
                self.next_node
            )

            if (
                self.path_index + 1
                <
                len(self.path) - 1
            ):

                self.path_index += 1

                self.next_node = (
                    self.path[
                        self.path_index + 1
                    ]
                )

            else:

                self.current_node = (
                    self.dest_node
                )

        else:

            self.pos[0] += (
                dx / distance
            ) * self.speed

            self.pos[1] += (
                dy / distance
            ) * self.speed

    # ========================================================
    # SCREEN POSITION
    # ========================================================

    def get_screen_pos(
        self,
        zoom,
        pan_x,
        pan_y
    ):

        target_x, target_y = (
            self.world_pos[
                self.next_node
            ]
        )

        dx = (
            target_x
            - self.pos[0]
        )

        dy = (
            target_y
            - self.pos[1]
        )

        dist = math.hypot(
            dx,
            dy
        )

        offset_pixels = 3.0

        if dist > 0:

            nx_vec = (
                -dy / dist
            )

            ny_vec = (
                dx / dist
            )

        else:

            nx_vec = 0

            ny_vec = 0

        world_x = (
            self.pos[0]
            + nx_vec * offset_pixels
        )

        world_y = (
            self.pos[1]
            + ny_vec * offset_pixels
        )

        return (
            world_x * zoom + pan_x,
            world_y * zoom + pan_y
        )

    # ========================================================
    # DRAW
    # ========================================================

    def draw(
        self,
        surface,
        zoom,
        pan_x,
        pan_y
    ):

        if not self.path:
            return

        sx, sy = (
            self.get_screen_pos(
                zoom,
                pan_x,
                pan_y
            )
        )

        car_radius = max(
            2,
            int(
                4 * math.sqrt(zoom)
            )
        )

        pygame.draw.circle(
            surface,
            self.color,
            (
                int(sx),
                int(sy)
            ),
            car_radius
        )


# ============================================================
# GENERATE ROUTES
# ============================================================

def generate_clustered_routes(
    G,
    roundabout_nodes,
    num_cars
):

    """
    Uses at least 50% of the available road
    segments as starting points for route
    generation.

    Since NUM_CARS limits how many routes can
    simultaneously contain vehicles, the number
    of active routes is capped at NUM_CARS.
    """

    roads = list(
        G.edges()
    )

    total_roads = len(
        roads
    )

    if total_roads < 2:

        print(
            "Not enough roads "
            "to generate routes."
        )

        return [], []

    # --------------------------------------------------------
    # At least half the roads
    # --------------------------------------------------------

    requested_routes = math.ceil(
        total_roads * 0.5
    )

    num_routes = min(
        requested_routes,
        num_cars
    )

    # --------------------------------------------------------
    # Randomize roads
    # --------------------------------------------------------

    random.shuffle(
        roads
    )

    selected_roads = roads[
        :num_routes
    ]

    print(
        f"Total road segments: "
        f"{total_roads}"
    )

    print(
        f"Selected road segments: "
        f"{len(selected_roads)}"
    )

    print(
        f"Selected percentage: "
        f"{len(selected_roads) / total_roads * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Destination candidates
    # --------------------------------------------------------

    all_nodes = list(
        G.nodes()
    )

    routes = []

    # --------------------------------------------------------
    # Generate route for each selected road
    # --------------------------------------------------------

    for road in selected_roads:

        start_node = road[0]

        # Randomize destination order
        destinations = all_nodes.copy()

        random.shuffle(
            destinations
        )

        destination = None

        for candidate in destinations:

            if candidate == start_node:
                continue

            try:

                if nx.has_path(
                    G,
                    start_node,
                    candidate
                ):

                    destination = (
                        candidate
                    )

                    break

            except nx.NetworkXError:

                continue

        if destination is None:
            continue

        route_color = (
            ROUTE_COLORS[
                len(routes)
                %
                len(ROUTE_COLORS)
            ]
        )

        routes.append({

            "id": len(routes) + 1,

            "start": start_node,

            "dest": destination,

            "color": route_color

        })

    # --------------------------------------------------------
    # Randomize route order
    # --------------------------------------------------------

    random.shuffle(
        routes
    )

    for i, route in enumerate(
        routes
    ):

        route["id"] = i + 1

    if not routes:

        print(
            "No valid routes generated."
        )

        return [], []

    # --------------------------------------------------------
    # Assign cars
    # --------------------------------------------------------

    car_configs = []

    for i in range(num_cars):

        route = routes[
            i % len(routes)
        ]

        car_configs.append(
            route
        )

    print(
        f"Generated "
        f"{len(routes)} routes."
    )

    print(
        f"Assigned "
        f"{num_cars} cars."
    )

    return (
        car_configs,
        routes
    )


# ============================================================
# DRAW DISRUPTION STATUS
# ============================================================

def draw_disruption_panel(
    screen,
    font,
    disruption_engine
):

    active = (
        disruption_engine.get_active_disruptions()
    )

    panel_width = 270

    panel_height = 35 + (
        len(active) * 22
    )

    pygame.draw.rect(
        screen,
        COLOR_PANEL,
        (
            10,
            10,
            panel_width,
            panel_height
        )
    )

    title = font.render(
        f"Active Disruptions: {len(active)}",
        True,
        COLOR_TEXT
    )

    screen.blit(
        title,
        (
            20,
            18
        )
    )

    y = 42

    for road, data in active.items():

        disruption_type = data[
            "type"
        ]

        if disruption_type == "blocked":

            color = COLOR_BLOCKED

        elif disruption_type == "accident":

            color = COLOR_ACCIDENT

        else:

            color = COLOR_CONSTRUCTION

        text = font.render(
            disruption_type.upper(),
            True,
            color
        )

        screen.blit(
            text,
            (
                20,
                y
            )
        )

        y += 22


# ============================================================
# MAIN
# ============================================================

def main():

    pygame.init()

    font = pygame.font.SysFont(
        "Arial",
        12,
        bold=True
    )

    screen = pygame.display.set_mode(
        (
            WINDOW_WIDTH,
            WINDOW_HEIGHT
        )
    )

    pygame.display.set_caption(
        "Traffic Simulator + Disruption Engine"
    )

    clock = pygame.time.Clock()

    # ========================================================
    # LOAD MAP
    # ========================================================

    G, roundabout_nodes = (
        load_local_pbf(
            PBF_FILE_PATH
        )
    )

    # ========================================================
    # CREATE DISRUPTION ENGINE
    # ========================================================

    disruption_engine = (
        DisruptionEngine(G)
    )

    # ========================================================
    # WORLD COORDINATES
    # ========================================================

    world_pos = (
        convert_coords_to_world(
            G,
            WINDOW_WIDTH,
            WINDOW_HEIGHT
        )
    )

    # ========================================================
    # GENERATE ROUTES
    # ========================================================

    car_configs, active_routes = (
        generate_clustered_routes(
            G,
            roundabout_nodes,
            NUM_CARS
        )
    )

    # ========================================================
    # CREATE CARS
    # ========================================================

    cars = []

    for cfg in car_configs:

        car = Car(

            G,

            world_pos,

            cfg["start"],

            cfg["dest"],

            cfg["id"],

            cfg["color"]

        )

        cars.append(
            car
        )

    # ========================================================
    # CAMERA
    # ========================================================

    zoom = 1.0

    pan_x = 0.0

    pan_y = 0.0

    is_panning = False

    pan_start_pos = (
        0,
        0
    )

    # ========================================================
    # DISRUPTION TIMERS
    # ========================================================

    time_since_disruption = 0.0

    active_disruption_timers = {}

    # ========================================================
    # MAIN LOOP
    # ========================================================

    running = True

    while running:

        # Time passed since previous frame
        dt = (
            clock.get_time()
            / 1000.0
        )

        # ====================================================
        # EVENTS
        # ====================================================

        for event in pygame.event.get():

            # ------------------------------------------------
            # QUIT
            # ------------------------------------------------

            if event.type == pygame.QUIT:

                running = False

            # ------------------------------------------------
            # ZOOM
            # ------------------------------------------------

            elif event.type == pygame.MOUSEWHEEL:

                mx, my = (
                    pygame.mouse.get_pos()
                )

                zoom_factor = (
                    1.8
                    if event.y > 0
                    else 0.85
                )

                new_zoom = max(
                    0.2,
                    min(
                        zoom * zoom_factor,
                        15.0
                    )
                )

                pan_x = (
                    mx
                    -
                    (mx - pan_x)
                    *
                    (new_zoom / zoom)
                )

                pan_y = (
                    my
                    -
                    (my - pan_y)
                    *
                    (new_zoom / zoom)
                )

                zoom = new_zoom

            # ------------------------------------------------
            # MOUSE BUTTON DOWN
            # ------------------------------------------------

            elif event.type == pygame.MOUSEBUTTONDOWN:

                # ============================================
                # RIGHT CLICK = PAN
                # ============================================

                if event.button == 3:

                    is_panning = True

                    pan_start_pos = (
                        pygame.mouse.get_pos()
                    )

                # ============================================
                # ALT/SHIFT + LEFT CLICK = MANUAL BLOCK
                # ============================================

                elif event.button == 1:

                    mods = pygame.key.get_mods()

                    if (
                        mods & pygame.KMOD_ALT
                        or
                        mods & pygame.KMOD_SHIFT
                    ):

                        mx, my = (
                            pygame.mouse.get_pos()
                        )

                        world_mx = (
                            mx - pan_x
                        ) / zoom

                        world_my = (
                            my - pan_y
                        ) / zoom

                        closest_edge = None

                        min_dist = (
                            15.0 / zoom
                        )

                        # Find closest road
                        for u, v in G.edges():

                            p1 = world_pos[u]

                            p2 = world_pos[v]

                            mid_x = (
                                p1[0]
                                + p2[0]
                            ) / 2

                            mid_y = (
                                p1[1]
                                + p2[1]
                            ) / 2

                            dist = math.hypot(
                                world_mx - mid_x,
                                world_my - mid_y
                            )

                            if dist < min_dist:

                                min_dist = dist

                                closest_edge = (
                                    u,
                                    v
                                )

                        if closest_edge:

                            u, v = (
                                closest_edge
                            )

                            # --------------------------------
                            # If already blocked -> restore
                            # --------------------------------

                            if G[u][v].get(
                                "blocked",
                                False
                            ):

                                disruption_engine.restore_road(
                                    u,
                                    v
                                )

                                print(
                                    "Road manually restored!"
                                )

                            else:

                                disruption_engine.block_road(
                                    u,
                                    v
                                )

                                print(
                                    "Road manually blocked!"
                                )

                            # Recalculate routes
                            for car in cars:

                                car.recalculate_path()

            # ------------------------------------------------
            # MOUSE MOTION
            # ------------------------------------------------

            elif event.type == pygame.MOUSEMOTION:

                if is_panning:

                    mx, my = (
                        pygame.mouse.get_pos()
                    )

                    pan_x += (
                        mx
                        -
                        pan_start_pos[0]
                    )

                    pan_y += (
                        my
                        -
                        pan_start_pos[1]
                    )

                    pan_start_pos = (
                        mx,
                        my
                    )

            # ------------------------------------------------
            # MOUSE BUTTON UP
            # ------------------------------------------------

            elif event.type == pygame.MOUSEBUTTONUP:

                if event.button == 3:

                    is_panning = False

        # ====================================================
        # RANDOM DISRUPTION TIMER
        # ====================================================

        time_since_disruption += dt

        if (
            time_since_disruption
            >= DISRUPTION_INTERVAL
        ):

            time_since_disruption = 0.0

            disruption = (
                disruption_engine
                .create_random_disruption()
            )

            if disruption:

                road = disruption[
                    "road"
                ]

                disruption_type = (
                    disruption["type"]
                )

                # Store expiration time
                active_disruption_timers[
                    road
                ] = DISRUPTION_DURATION

                print(
                    f"[DISRUPTION] "
                    f"{disruption_type.upper()} "
                    f"on road {road}"
                )

                # Recalculate every car
                for car in cars:

                    car.recalculate_path()

        # ====================================================
        # UPDATE DISRUPTION TIMERS
        # ====================================================

        roads_to_restore = []

        for road in list(
            active_disruption_timers.keys()
        ):

            active_disruption_timers[
                road
            ] -= dt

            if (
                active_disruption_timers[
                    road
                ]
                <= 0
            ):

                roads_to_restore.append(
                    road
                )

        # ====================================================
        # RESTORE EXPIRED DISRUPTIONS
        # ====================================================

        for road in roads_to_restore:

            u, v = road

            disruption_engine.restore_road(
                u,
                v
            )

            del active_disruption_timers[
                road
            ]

            print(
                f"[DISRUPTION CLEARED] "
                f"Road {road} restored."
            )

            # Recalculate routes
            for car in cars:

                car.recalculate_path()

        # ====================================================
        # UPDATE CARS
        # ====================================================

        for car in cars:

            car.update()

        # ====================================================
        # RENDER
        # ====================================================

        screen.fill(
            COLOR_BG
        )

        # ----------------------------------------------------
        # Screen coordinates
        # ----------------------------------------------------

        screen_pos = {

            node: (
                pt[0] * zoom + pan_x,
                pt[1] * zoom + pan_y
            )

            for node, pt
            in world_pos.items()

        }

        # ====================================================
        # 1. DRAW BASE ROAD NETWORK
        # ====================================================

        for u, v, data in G.edges(
            data=True
        ):

            # -----------------------------------------------
            # Blocked road
            # -----------------------------------------------

            if data.get(
                "blocked",
                False
            ):

                color = (
                    COLOR_ROAD_BLOCKED
                )

                base_width = 5

            # -----------------------------------------------
            # Accident
            # -----------------------------------------------

            elif data.get(
                "disruption"
            ) == "accident":

                color = (
                    COLOR_ACCIDENT
                )

                base_width = 4

            # -----------------------------------------------
            # Construction
            # -----------------------------------------------

            elif data.get(
                "disruption"
            ) == "construction":

                color = (
                    COLOR_CONSTRUCTION
                )

                base_width = 4

            # -----------------------------------------------
            # Normal road
            # -----------------------------------------------

            else:

                color = COLOR_ROAD

                base_width = 2

            line_width = max(
                1,
                int(
                    base_width
                    *
                    math.sqrt(
                        zoom
                    )
                )
            )

            pygame.draw.line(
                screen,
                color,
                screen_pos[u],
                screen_pos[v],
                line_width
            )

        # ====================================================
        # 2. DRAW COLORED ROUTES
        # ====================================================

        drawn_routes = set()

        for car in cars:

            if (
                car.route_id
                not in drawn_routes
                and
                car.path
            ):

                drawn_routes.add(
                    car.route_id
                )

                for u, v in zip(
                    car.path[:-1],
                    car.path[1:]
                ):

                    pygame.draw.line(

                        screen,

                        car.color,

                        screen_pos[u],

                        screen_pos[v],

                        max(
                            2,
                            int(
                                3
                                *
                                math.sqrt(
                                    zoom
                                )
                            )
                        )

                    )

        # ====================================================
        # 3. START / DESTINATION MARKERS
        # ====================================================

        marker_radius = max(
            6,
            int(
                10
                *
                math.sqrt(
                    zoom
                )
            )
        )

        for route in active_routes:

            r_id = route[
                "id"
            ]

            color = route[
                "color"
            ]

            # ------------------------------------------------
            # START
            # ------------------------------------------------

            if route[
                "start"
            ] in screen_pos:

                sx, sy = screen_pos[
                    route["start"]
                ]

                pygame.draw.circle(
                    screen,
                    color,
                    (
                        int(sx),
                        int(sy)
                    ),
                    marker_radius
                )

                pygame.draw.circle(
                    screen,
                    (255, 255, 255),
                    (
                        int(sx),
                        int(sy)
                    ),
                    marker_radius,
                    2
                )

                txt = font.render(
                    f"S{r_id}",
                    True,
                    (255, 255, 255)
                )

                screen.blit(
                    txt,
                    (
                        int(sx) - 8,
                        int(sy) - 6
                    )
                )

            # ------------------------------------------------
            # DESTINATION
            # ------------------------------------------------

            if route[
                "dest"
            ] in screen_pos:

                dx, dy = screen_pos[
                    route["dest"]
                ]

                pygame.draw.circle(
                    screen,
                    color,
                    (
                        int(dx),
                        int(dy)
                    ),
                    marker_radius
                )

                pygame.draw.circle(
                    screen,
                    (0, 0, 0),
                    (
                        int(dx),
                        int(dy)
                    ),
                    marker_radius,
                    2
                )

                txt = font.render(
                    f"D{r_id}",
                    True,
                    (0, 0, 0)
                )

                screen.blit(
                    txt,
                    (
                        int(dx) - 8,
                        int(dy) - 6
                    )
                )

        # ====================================================
        # 4. DRAW CARS
        # ====================================================

        for car in cars:

            car.draw(
                screen,
                zoom,
                pan_x,
                pan_y
            )

        # ====================================================
        # 5. DISRUPTION PANEL
        # ====================================================

        draw_disruption_panel(
            screen,
            font,
            disruption_engine
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        pygame.display.flip()

        clock.tick(
            60
        )

    # ========================================================
    # EXIT
    # ========================================================

    pygame.quit()


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    main()