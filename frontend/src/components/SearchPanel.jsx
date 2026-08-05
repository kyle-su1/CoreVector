import React from "react";

export default function SearchPanel({
  queryId,
  setQueryId,
  k,
  setK,
  onSearch,
  onSeed,
  onReload,
  count,
  busy,
  error,
}) {
  return (
    <div className="panel">
      <h1>CoreVector</h1>
      <p className="subtitle">Embedding explorer — PCA 128D → 3D</p>

      <div className="stat">
        <span className="stat-num">{count}</span> vectors in index
      </div>

      <div className="field">
        <label>Query by point id</label>
        <input
          type="number"
          value={queryId}
          onChange={(e) => setQueryId(e.target.value)}
          placeholder="e.g. 0"
        />
      </div>

      <div className="field">
        <label>k (neighbors)</label>
        <input
          type="number"
          min="1"
          max="50"
          value={k}
          onChange={(e) => setK(Number(e.target.value))}
        />
      </div>

      <button className="primary" onClick={onSearch} disabled={busy}>
        {busy ? "Searching…" : "Search nearest neighbors"}
      </button>

      <div className="secondary-actions">
        <button onClick={onSeed} disabled={busy}>
          Seed demo data
        </button>
        <button onClick={onReload} disabled={busy}>
          Reload cloud
        </button>
      </div>

      {error && <div className="error">{error}</div>}

      <p className="hint">
        Tip: hover a point to see its payload. Orbit with drag, zoom with scroll.
      </p>
    </div>
  );
}
