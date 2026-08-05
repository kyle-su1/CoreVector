// Thin wrapper around the BFF endpoints. All requests go through the Vite
// proxy (/api -> http://localhost:8000).

async function jsonFetch(url, options) {
  const resp = await fetch(url, options);
  if (!resp.ok) {
    const detail = await resp.json().catch(() => ({}));
    throw new Error(detail.detail || `${resp.status} ${resp.statusText}`);
  }
  return resp.json();
}

export function getVectors() {
  return jsonFetch("/api/vectors");
}

export function search({ id, query, k }) {
  return jsonFetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id, query, k }),
  });
}

export function insert(vectors) {
  return jsonFetch("/api/insert", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vectors }),
  });
}

// Generate clustered demo vectors client-side for the "Seed demo data" button.
export function makeDemoVectors(n = 300, dim = 128) {
  const payloads = [
    "neural networks and deep learning",
    "italian restaurants downtown",
    "quantum computing basics",
    "stock market news today",
    "sourdough bread recipe",
    "memory-mapped IO in operating systems",
    "training a transformer from scratch",
    "best hiking trails in the alps",
  ];
  const centers = payloads.map(() =>
    Array.from({ length: dim }, () => Math.random() * 2 - 1)
  );
  // Box-Muller for gaussian jitter around each cluster center.
  const gauss = () => {
    const u = 1 - Math.random();
    const v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const vectors = [];
  for (let i = 0; i < n; i++) {
    const c = i % payloads.length;
    const values = centers[c].map((x) => x + gauss() * 0.15);
    vectors.push({ values, payload: payloads[c] });
  }
  return vectors;
}
