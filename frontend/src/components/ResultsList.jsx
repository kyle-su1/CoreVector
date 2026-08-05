import React from "react";

export default function ResultsList({ neighbors }) {
  if (!neighbors || !neighbors.length) return null;
  return (
    <div className="results">
      <h2>Nearest neighbors</h2>
      <ol>
        {neighbors.map((n) => (
          <li key={n.id}>
            <span className="rid">#{n.id}</span>
            <span className="rdist">d²={n.distance.toFixed(3)}</span>
            <span className="rpay">{n.payload || <em>(no payload)</em>}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
