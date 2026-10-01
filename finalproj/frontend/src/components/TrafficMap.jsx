import { useEffect, useMemo, useRef } from "react";

const PAD = 20;
const TIME_SCALE = 4; // cars move this many times faster than real time so motion is visible
const ROUTE_COLORS = ["#38bdf8", "#f59e0b", "#a78bfa", "#34d399", "#f472b6", "#facc15"];

// point at distance d along a polyline with precomputed cumulative lengths
function pointAt(seg, d) {
  const { pts, cum } = seg;
  const total = cum[cum.length - 1];
  const dist = Math.min(Math.max(d, 0), total);
  let i = 1;
  while (i < cum.length - 1 && cum[i] < dist) i += 1;
  const span = cum[i] - cum[i - 1] || 1;
  const f = (dist - cum[i - 1]) / span;
  return [pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * f, pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * f];
}

function TrafficMap({ graph, routes = [], running = false }) {
  const carLayerRef = useRef(null);
  const carsRef = useRef(new Map());
  const runningRef = useRef(running);
  runningRef.current = running;
  const edgeStateRef = useRef({});

  // static geometry, rebuilt only when the road network itself changes
  const geo = useMemo(() => {
    const nodes = graph?.nodes || [];
    const edges = graph?.edges || [];
    if (!nodes.length) return null;
    const pts = [
      ...nodes.map((n) => [Number(n.x), Number(n.y)]),
      ...edges.flatMap((e) => e.shape || []),
    ];
    const minX = Math.min(...pts.map((p) => p[0])), maxX = Math.max(...pts.map((p) => p[0]));
    const minY = Math.min(...pts.map((p) => p[1])), maxY = Math.max(...pts.map((p) => p[1]));
    const spanX = Math.max(maxX - minX, 1), spanY = Math.max(maxY - minY, 1);
    const scale = 1000 / spanX; // fixed viewBox width, height follows the real aspect ratio
    const W = 1000 + 2 * PAD, H = spanY * scale + 2 * PAD;
    const tx = ([x, y]) => [PAD + (x - minX) * scale, PAD + (maxY - y) * scale]; // flip y for screen
    const nodePos = {};
    nodes.forEach((n) => { nodePos[String(n.id)] = tx([Number(n.x), Number(n.y)]); });
    const seg = {};
    edges.forEach((e) => {
      const raw = e.shape?.length >= 2 ? e.shape : [nodePos[String(e.u)], nodePos[String(e.v)]].filter(Boolean);
      const p = e.shape?.length >= 2 ? raw.map(tx) : raw;
      if (p.length < 2) return;
      const cum = [0];
      for (let i = 1; i < p.length; i += 1) cum.push(cum[i - 1] + Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]));
      seg[`${e.u}::${e.v}`] = { pts: p, cum, len: cum[cum.length - 1], edge: e, mpp: (e.length || 1) / (cum[cum.length - 1] || 1) };
    });
    return { nodes, edges, nodePos, seg, W, H, scale };
  }, [graph?.nodes, graph?.edges?.length]); // eslint-disable-line react-hooks/exhaustive-deps

  // latest blocked / predicted-weight state, read by the animation loop
  useEffect(() => {
    const m = {};
    (graph?.edges || []).forEach((e) => { m[`${e.u}::${e.v}`] = e; });
    edgeStateRef.current = m;
  }, [graph]);

  // sync routes coming from the backend into per-car animation state
  useEffect(() => {
    if (!geo) return;
    const cars = carsRef.current;
    const live = new Set();
    routes.forEach((r, idx) => {
      const route = (r.route || []).map(String);
      live.add(r.car_id);
      const car = cars.get(r.car_id);
      if (route.length < 2) { if (car) car.tail = []; return; }
      if (!car) {
        cars.set(r.car_id, {
          id: r.car_id, color: ROUTE_COLORS[idx % ROUTE_COLORS.length], source: route[0],
          from: route[0], to: route[1], d: Math.random() * 0.5, tail: route.slice(2), parked: false,
        });
        return;
      }
      car.source = route[0];
      if (car.parked) { // finished last trip: start the new route
        Object.assign(car, { from: route[0], to: route[1], d: 0, tail: route.slice(2), parked: false });
        return;
      }
      const i = route.indexOf(car.to);
      if (i >= 0) car.tail = route.slice(i + 1);                 // keep current road, follow the new route after it
      else Object.assign(car, { from: route[0], to: route[1], d: 0, tail: route.slice(2) }); // road no longer on route
    });
    for (const id of [...cars.keys()]) if (!live.has(id)) cars.delete(id);
  }, [routes, geo]);

  // animation loop: moves the SVG circles directly, no React re-render per frame
  useEffect(() => {
    if (!geo) return undefined;
    const layer = carLayerRef.current;
    const els = new Map();
    let raf, last = performance.now();
    const frame = (now) => {
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      const state = edgeStateRef.current;
      const ws = Object.values(state).map((e) => e.traffic_weight).filter((w) => w > 0);
      const meanW = ws.length ? ws.reduce((a, b) => a + b, 0) / ws.length : 1;

      carsRef.current.forEach((car) => {
        let s = geo.seg[`${car.from}::${car.to}`];
        if (runningRef.current && s && !car.parked) {
          const e = state[`${car.from}::${car.to}`];
          const slow = e?.traffic_weight > 0 ? Math.min(1.5, Math.max(0.25, meanW / e.traffic_weight)) : 1;
          const speedMps = (s.edge.speed || 8) * slow * TIME_SCALE;          // metres per second on screen
          car.d += (speedMps * dt) / s.mpp;                                   // metres -> map units
          while (s && car.d >= s.len) {
            car.d -= s.len;
            if (car.tail.length) { car.from = car.to; car.to = car.tail.shift(); s = geo.seg[`${car.from}::${car.to}`]; }
            else { car.parked = true; car.d = 0; break; }
          }
        }
        s = geo.seg[`${car.from}::${car.to}`];
        let el = els.get(car.id);
        if (!el) {
          el = document.createElementNS("http://www.w3.org/2000/svg", "circle");
          el.setAttribute("r", "4");
          el.setAttribute("fill", car.color);
          el.setAttribute("stroke", "#0b0f14");
          el.setAttribute("stroke-width", "1.2");
          layer.appendChild(el);
          els.set(car.id, el);
        }
        if (s) {
          const [x, y] = pointAt(s, car.d);
          el.setAttribute("cx", x); el.setAttribute("cy", y); el.style.display = "";
        } else el.style.display = "none";
      });
      for (const [id, el] of els) if (!carsRef.current.has(id)) { el.remove(); els.delete(id); }
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => { cancelAnimationFrame(raf); els.forEach((el) => el.remove()); };
  }, [geo]);

  const routePairs = useMemo(() => {
    const set = new Set();
    routes.forEach((r) => { for (let i = 1; i < (r.route || []).length; i += 1) set.add(`${r.route[i - 1]}::${r.route[i]}`); });
    return set;
  }, [routes]);

  return (
    <section className="traffic-map">
      <h2>Live Traffic Network</h2>
      <div className="map-placeholder" style={geo ? { aspectRatio: `${geo.W} / ${geo.H}`, height: "auto" } : undefined}>
        {!geo ? <p>Waiting for graph data from the Python backend.</p> : (
          <svg viewBox={`0 0 ${geo.W} ${geo.H}`} className="network-svg" role="img" aria-label="Traffic network with moving cars">
            {geo.edges.map((edge) => {
              const s = geo.seg[`${edge.u}::${edge.v}`];
              if (!s) return null;
              const cls = edge.blocked ? "edge blocked" : routePairs.has(`${edge.u}::${edge.v}`) ? "edge route" : "edge";
              return <polyline key={edge.id} points={s.pts.map((p) => p.join(",")).join(" ")} className={cls} fill="none" />;
            })}
            <g ref={carLayerRef} />
          </svg>
        )}
      </div>
    </section>
  );
}

export default TrafficMap;
