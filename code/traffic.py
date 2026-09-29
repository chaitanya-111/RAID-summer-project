import pygame
import math
import subprocess
import random #added

pygame.init()

WIDTH = 1100
HEIGHT = 760

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traffic Control Simulator")

BACKGROUND = (234, 242, 255)
WHITE = (255, 255, 255)
DARK_BLUE = (23, 74, 139)
TEXT_BLUE = (40, 75, 110)
ROAD = (150, 175, 200)
BLOCKED = (224, 82, 82)
ROUTE = (23, 105, 170)
NODE = (217, 233, 255)
NODE_BORDER = (55, 107, 158)
START = (39, 121, 189)
END = (127, 184, 232)
MAP_BACKGROUND = (247, 251, 255)
BORDER = (184, 203, 227)

font = pygame.font.SysFont("arial", 18)
small_font = pygame.font.SysFont("arial", 14)
title_font = pygame.font.SysFont("arial", 30, bold=True)
button_font = pygame.font.SysFont("arial", 17, bold=True)


def load_nodes(filename):
    nodes = {}

    with open(filename, "r") as file:
        for line in file:
            parts = line.split()

            if len(parts) >= 3 and not parts[0].startswith("#"):
                node_id = parts[0]
                lat = float(parts[1])
                lon = float(parts[2])

                nodes[node_id] = {
                    "lat": lat,
                    "lon": lon,
                    "x": 0,
                    "y": 0
                }

    return nodes


def load_edges(filename):
    edges = []

    with open(filename, "r") as file:
        for line in file:
            parts = line.split()

            if len(parts) >= 3:
                edges.append(
                    (parts[0], parts[1], float(parts[2]))
                )

    return edges


nodes = load_nodes("nodes_data.txt")
edges = load_edges("graph_data.txt")

# CONVERT GPS TO SCREEN COORDINATES

min_lat = min(node["lat"] for node in nodes.values())
max_lat = max(node["lat"] for node in nodes.values())

min_lon = min(node["lon"] for node in nodes.values())
max_lon = max(node["lon"] for node in nodes.values())

MAP_LEFT = 30
MAP_TOP = 130
MAP_RIGHT = WIDTH - 30
MAP_BOTTOM = HEIGHT - 90

map_width = MAP_RIGHT - MAP_LEFT
map_height = MAP_BOTTOM - MAP_TOP

for node in nodes.values():

    node["x"] = (MAP_LEFT+
        ((node["lon"] - min_lon) / (max_lon - min_lon))
        * map_width)

    node["y"] = ( MAP_BOTTOM - 
        ((node["lat"] - min_lat) / (max_lat - min_lat))
        * map_height)

# STATE
blocked = set()

start = None
end = None

shortest_path = []
total_distance = 0

#added
cars =[]
simulation_running = False

# PYTHON CALLS C++
def run_cpp_dijkstra():

    global shortest_path
    global total_distance

    shortest_path = []
    total_distance = 0

    if start is None or end is None:
        return

    # Give C++ the roads that the user has blocked.
    with open("blocked_edges.txt", "w") as file:

        for a, b in blocked:
            file.write(a + " " + b + "\n")

    try:

        result = subprocess.run(
            ["./traffic", start, end],
            capture_output=True,
            text=True
        )

    except FileNotFoundError:

        print("Could not find ./traffic.")
        print("Compile code.cpp first:")
        print("g++ code.cpp -o traffic")
        return

    if result.returncode != 0:
        print(result.stderr)
        return

    output = result.stdout.strip()

    if output == "NO_ROUTE":
        return

    for line in output.splitlines():

        parts = line.split()

        if not parts:
            continue

        if parts[0] == "PATH":
            shortest_path = parts[1:]

        elif parts[0] == "DISTANCE":
            total_distance = float(parts[1])


# NODE CLICK
def find_node(mouse_pos):

    mx, my = mouse_pos

    closest = None
    closest_distance = 10

    for node_id, node in nodes.items():

        distance = math.hypot(
            mx - node["x"],
            my - node["y"]
        )

        if distance < closest_distance:
            closest_distance = distance
            closest = node_id

    return closest

# added
def create_cars():

    cars.clear()

    node_ids = list(nodes.keys())

    for i in range(100):

        start_node = random.choice(node_ids)
        end_node = random.choice(node_ids)

        while end_node == start_node:
            end_node = random.choice(node_ids)

        cars.append({
            "id": i,
            "start": start_node,
            "end": end_node,
            "path": [],
            "current": 0,
            "progress": 0.0
        })
        
def run_car_simulation():
    
    with open("cars.txt", "w") as file:
        
        for car in cars:
            file.write(
                f"{car['id']}"
                f"{car['start']}"
                f"{car['end']}\n"
            )
    with open("blocked_edges.txt", "w") as file:
        for a,b in blocked:
            file.write(a + " " + b + "\n")
            
    result = subprocess.run(
        ["./traffic"], capture_output=True, text = True
    )
    if result.returncode != 0:
        print(result.stderr)
        return

    current_car = None

    for line in result.stdout.splitlines():

        parts = line.split()

        if not parts:
            continue

        if parts[0] == "CAR":
            current_car = int(parts[1])

        elif parts[0] == "PATH":
            if current_car is not None:
                cars[current_car]["path"] = parts[1:]
    
def move_cars():

    for car in cars:

        path = car["path"]

        if len(path) < 2:
            continue

        if car["current"] >= len(path) - 1:
            continue

        a = path[car["current"]]
        b = path[car["current"] + 1]

        car["progress"] += 0.005

        if car["progress"] >= 1:
            car["progress"] = 0
            car["current"] += 1
            
def draw_cars():

    for car in cars:

        path = car["path"]

        if len(path) < 2:
            continue

        if car["current"] >= len(path) - 1:
            continue

        a = path[car["current"]]
        b = path[car["current"] + 1]

        node_a = nodes[a]
        node_b = nodes[b]

        x = node_a["x"] + car["progress"] * (
            node_b["x"] - node_a["x"]
        )

        y = node_a["y"] + car["progress"] * (
            node_b["y"] - node_a["y"]
        )

        pygame.draw.circle(
            screen,
            DARK_BLUE,
            (int(x), int(y)),
            5
        )
   
# ROAD CLICK
def distance_to_line(px, py, x1, y1, x2, y2):

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)

    t = (
        (px - x1) * dx +
        (py - y1) * dy
    ) / (dx * dx + dy * dy)

    t = max(0, min(1, t))

    closest_x = x1 + t * dx
    closest_y = y1 + t * dy

    return math.hypot(
        px - closest_x,
        py - closest_y
    )


def find_road(mouse_pos):

    mx, my = mouse_pos

    closest_edge = None
    closest_distance = 7

    for a, b, weight in edges:

        if a not in nodes or b not in nodes:
            continue

        node_a = nodes[a]
        node_b = nodes[b]

        distance = distance_to_line(
            mx,
            my,
            node_a["x"],
            node_a["y"],
            node_b["x"],
            node_b["y"]
        )

        if distance < closest_distance:

            closest_distance = distance
            closest_edge = (a, b)

    return closest_edge


# PATH EDGE CHECK
def edge_in_path(a, b):

    for i in range(len(shortest_path) - 1):

        if (
            shortest_path[i] == a
            and shortest_path[i + 1] == b
        ) or (
            shortest_path[i] == b
            and shortest_path[i + 1] == a
        ):
            return True

    return False

def draw_button():

    rect = pygame.Rect(
        WIDTH // 2 - 70,
        HEIGHT - 55,
        140,
        38
    )

    pygame.draw.rect(
        screen,
        DARK_BLUE,
        rect,
        border_radius=7
    )

    text = button_font.render(
        "Reset",
        True,
        WHITE
    )

    screen.blit(
        text,
        (
            rect.centerx - text.get_width() // 2,
            rect.centery - text.get_height() // 2
        )
    )

    return rect

def draw():

    screen.fill(BACKGROUND)

    container = pygame.Rect(
        20,
        15,
        WIDTH - 40,
        HEIGHT - 30
    )

    pygame.draw.rect(
        screen,
        WHITE,
        container,
        border_radius=12
    )

    title = title_font.render(
        "Traffic Control Simulator",
        True,
        DARK_BLUE
    )

    screen.blit(
        title,
        (
            WIDTH // 2 - title.get_width() // 2,
            28
        )
    )

    instruction = font.render(
        "Click a node for start, another node for destination. "
        "Click a road to block/unblock it.",
        True,
        TEXT_BLUE
    )

    screen.blit(
        instruction,
        (
            WIDTH // 2 - instruction.get_width() // 2,
            72
        )
    )

    map_rect = pygame.Rect(
        MAP_LEFT,
        MAP_TOP,
        map_width,
        MAP_BOTTOM - MAP_TOP
    )

    pygame.draw.rect(
        screen,
        MAP_BACKGROUND,
        map_rect,
        border_radius=10
    )

    pygame.draw.rect(
        screen,
        BORDER,
        map_rect,
        width=1,
        border_radius=10
    )

    # ROADS
    for a, b, weight in edges:

        if a not in nodes or b not in nodes:
            continue

        node_a = nodes[a]
        node_b = nodes[b]

        key = frozenset((a, b))

        if key in blocked:

            color = BLOCKED
            width = 3

        elif edge_in_path(a, b):

            color = ROUTE
            width = 4

        else:

            color = ROAD
            width = 1

        pygame.draw.line(
            screen,
            color,
            (node_a["x"], node_a["y"]),
            (node_b["x"], node_b["y"]),
            width
        )

    for node_id, node in nodes.items():

        radius = 2
        color = NODE

        if node_id == start:
            radius = 8
            color = START

        elif node_id == end:
            radius = 8
            color = END

        pygame.draw.circle(
            screen,
            color,
            (int(node["x"]), int(node["y"])),
            radius
        )

        if node_id == start or node_id == end:

            pygame.draw.circle(
                screen,
                DARK_BLUE,
                (int(node["x"]), int(node["y"])),
                radius,
                2
            )

    if start is None:

        message = "Select a start and destination."

    elif end is None:

        message = (
            "Start = " + start +
            "    |    Now select destination."
        )

    elif not shortest_path:

        message = "No route available."

    else:

        message = (
            "Shortest route found    |    "
            "Start: " + start +
            "    Destination: " + end +
            "    |    Road segments: " +
            str(len(shortest_path) - 1) +
            "    |    Total distance: " +
            f"{total_distance:.2f}"
        )
        
    draw_cars()

    result_text = small_font.render(
        message,
        True,
        DARK_BLUE
    )

    screen.blit(
        result_text,
        (
            WIDTH // 2 - result_text.get_width() // 2,
            HEIGHT - 83
        )
    )


    return draw_button()

running = True
clock = pygame.time.Clock()

while running:

    if simulation_running:
        move_cars()
        
    button_rect = draw()
    pygame.display.flip()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            if event.key == pygame.K_SPACE:

                create_cars()
                run_car_simulation()
                simulation_running = True
        
        elif event.type == pygame.MOUSEBUTTONDOWN:

            mouse_pos = event.pos

            # Reset
            if button_rect.collidepoint(mouse_pos):

                start = None
                end = None
                blocked.clear()
                shortest_path = []
                total_distance = 0

            else:

                clicked_node = find_node(mouse_pos)

                if clicked_node is not None:

                    if start is None:

                        start = clicked_node

                    elif end is None and clicked_node != start:

                        end = clicked_node

                    else:

                        start = clicked_node
                        end = None

                    run_cpp_dijkstra()

                else:

                    clicked_edge = find_road(mouse_pos)

                    if clicked_edge is not None:

                        a, b = clicked_edge
                        key = frozenset((a, b))

                        if key in blocked:
                            blocked.remove(key)
                        else:
                            blocked.add(key)

                        run_cpp_dijkstra()

    clock.tick(60)

pygame.quit()