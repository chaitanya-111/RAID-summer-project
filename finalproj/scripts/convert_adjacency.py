"""Build data/graph.json from the SUMO network (data/osm.net.xml.gz).

Real junction coordinates and road shapes come from the net file, and the edge ids
are the SUMO edge ids used by the GRU, so predicted weights land on the right roads.
"""
import gzip, json
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root = ET.parse(gzip.open(ROOT / "data/osm.net.xml.gz")).getroot()
adj = json.loads((ROOT / "data/adjacency_list.json").read_text())  # used only to pick the junctions you kept

junc = {j.get("id"): j for j in root.findall("junction")}
edges = []
used = set()
for e in root.findall("edge"):
    if e.get("function") == "internal":
        continue
    lanes = e.findall("lane")
    lane = lanes[0]  # rightmost lane; its shape is already offset from the centre line
    pts = [[round(float(a), 2), round(float(b), 2)] for a, b in (p.split(",") for p in lane.get("shape").split())]
    length, speed = float(lane.get("length")), float(lane.get("speed"))
    edges.append({"id": e.get("id"), "u": e.get("from"), "v": e.get("to"), "name": e.get("name", ""),
                  "length": round(length, 1), "speed": round(speed, 2), "lanes": len(lanes),
                  "weight": round(length / max(speed, 0.1), 2),  # free-flow travel time (s)
                  "bidirectional": False, "shape": pts})
    used.update((e.get("from"), e.get("to")))

nodes = [{"id": n, "x": float(junc[n].get("x")), "y": float(junc[n].get("y"))} for n in sorted(used | set(adj))]
(ROOT / "data/graph.json").write_text(json.dumps({"nodes": nodes, "edges": edges}))
print(len(nodes), "nodes,", len(edges), "edges -> data/graph.json")
