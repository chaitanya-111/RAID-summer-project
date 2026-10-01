const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function getState() {
  const response = await fetch(`${API_BASE}/api/state`);
  if (!response.ok) throw new Error(`State request failed: ${response.status}`);
  return response.json();
}

export async function simulationCommand(command) {
  const response = await fetch(`${API_BASE}/api/simulation/${command}`, {
    method: "POST",
  });
  if (!response.ok) throw new Error(`Simulation command failed: ${response.status}`);
  return response.json();
}

export async function addDisruption(payload) {
  const response = await fetch(`${API_BASE}/api/disruptions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Disruption request failed: ${response.status}`);
  }
  return response.json();
}

export async function removeDisruption(u, v) {
  const response = await fetch(`${API_BASE}/api/disruptions/${encodeURIComponent(u)}/${encodeURIComponent(v)}`, {
    method: "DELETE",
  });
  if (!response.ok) throw new Error(`Restore request failed: ${response.status}`);
  return response.json();
}

export function createStateSocket(onMessage, onError) {
  const wsBase = API_BASE.replace(/^http/, "ws");
  const socket = new WebSocket(`${wsBase}/ws/state`);
  socket.onmessage = (event) => onMessage(JSON.parse(event.data));
  socket.onerror = onError;
  return socket;
}

export { API_BASE };
