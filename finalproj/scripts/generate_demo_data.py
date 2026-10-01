import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
ck=torch.load(ROOT/'best_traffic_gru.pt',map_location='cpu',weights_only=False)
edge_ids=list(ck['edge2idx'].keys())
features=list(ck['features'])

# 10 x 20 grid = 370 undirected connections; use 369 model edge IDs.
rows, cols = 10, 20
nodes=[]
for r in range(rows):
    for c in range(cols):
        nodes.append({'id': f'n{r}_{c}', 'x': c, 'y': r})
connections=[]
for r in range(rows):
    for c in range(cols-1):
        connections.append((f'n{r}_{c}', f'n{r}_{c+1}'))
for r in range(rows-1):
    for c in range(cols):
        connections.append((f'n{r}_{c}', f'n{r+1}_{c}'))
connections=connections[:len(edge_ids)]
edges=[]
for eid,(u,v) in zip(edge_ids,connections):
    edges.append({'id':eid,'u':u,'v':v,'weight':1.0,'bidirectional':True})
(ROOT/'data').mkdir(exist_ok=True)
(ROOT/'data/demo_graph.json').write_text(json.dumps({'nodes':nodes,'edges':edges},indent=2))

# Deterministic 24-step synthetic SUMO-like history. It has exactly the checkpoint features.
rng=np.random.default_rng(42)
base={f:np.ones(len(edge_ids),dtype=np.float32) for f in features}
base['sampledSeconds'][:]=60
base['traveltime'][:]=12
base['density'][:]=8
base['laneDensity'][:]=8
base['occupancy'][:]=0.12
base['waitingTime'][:]=1.0
base['timeLoss'][:]=1.0
base['speed'][:]=7.0
base['speedRelative'][:]=0.85
base['entered'][:]=2.0
# 'left' and 'flow' are harmless extras and may be dropped by the checkpoint's forbidden-column logic.
records=[]
for t in range(24):
    rush=1.0 + 0.25*math.sin(t/4)
    for i,eid in enumerate(edge_ids):
        local=1.0 + 0.12*math.sin(i/17+t/3)
        congestion=max(0.4, rush*local)
        records.append({
            'time':t,'edge_id':eid,
            'sampledSeconds':60.0,
            'traveltime':12.0*congestion,
            'density':8.0*congestion,
            'laneDensity':8.0*congestion,
            'occupancy':min(0.95,0.12*congestion),
            'waitingTime':1.0*max(0.2,congestion-0.2),
            'timeLoss':1.0*max(0.2,congestion-0.1),
            'speed':max(1.0,7.0/congestion),
            'speedRelative':max(0.1,min(1.2,0.85/congestion)),
            'entered':max(0.0,2.0*congestion+rng.normal(0,0.05)),
            'left':max(0.0,2.0*congestion+rng.normal(0,0.05)),
            'flow':max(0.0,2.0*congestion+rng.normal(0,0.05)),
        })
pd.DataFrame(records).to_csv(ROOT/'data/demo_traffic.csv',index=False)
print('generated',len(nodes),'nodes',len(edges),'edges',len(records),'traffic rows')
