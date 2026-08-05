import React, { useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import { OrbitView } from "@deck.gl/core";
import { ScatterplotLayer, LineLayer, TextLayer } from "@deck.gl/layers";

const AXIS_COLORS = {
  pc1: [224, 108, 108], // red
  pc2: [104, 205, 137], // green
  pc3: [110, 158, 240], // blue
};

// Round a raw grid step up to a "nice" 1/2/5 * 10^n value.
function niceStep(raw) {
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10;
  return nice * mag;
}

// Reference axes (through the origin, where the mean-centered cloud sits) plus
// a faint ground grid on the PC2=0 plane to anchor depth perception.
function buildReferenceLayers(points) {
  if (!points.length) return [];
  let maxAbs = 1e-3;
  for (const p of points) {
    maxAbs = Math.max(maxAbs, Math.abs(p.x), Math.abs(p.y), Math.abs(p.z));
  }
  const L = maxAbs * 1.15;
  const step = niceStep(L / 5);

  const axisData = [
    { from: [-L, 0, 0], to: [L, 0, 0], color: AXIS_COLORS.pc1 },
    { from: [0, -L, 0], to: [0, L, 0], color: AXIS_COLORS.pc2 },
    { from: [0, 0, -L], to: [0, 0, L], color: AXIS_COLORS.pc3 },
  ];

  const labelData = [
    { pos: [L, 0, 0], text: "PC1", color: AXIS_COLORS.pc1 },
    { pos: [0, L, 0], text: "PC2", color: AXIS_COLORS.pc2 },
    { pos: [0, 0, L], text: "PC3", color: AXIS_COLORS.pc3 },
  ];

  const gridLines = [];
  for (let t = -L; t <= L + 1e-6; t += step) {
    gridLines.push({ from: [t, 0, -L], to: [t, 0, L] });
    gridLines.push({ from: [-L, 0, t], to: [L, 0, t] });
  }

  return [
    new LineLayer({
      id: "grid",
      data: gridLines,
      getSourcePosition: (d) => d.from,
      getTargetPosition: (d) => d.to,
      getColor: [90, 100, 115, 55],
      getWidth: 1,
    }),
    new LineLayer({
      id: "axes",
      data: axisData,
      getSourcePosition: (d) => d.from,
      getTargetPosition: (d) => d.to,
      getColor: (d) => d.color,
      getWidth: 2,
    }),
    new TextLayer({
      id: "axis-labels",
      data: labelData,
      getPosition: (d) => d.pos,
      getText: (d) => d.text,
      getColor: (d) => d.color,
      getSize: 15,
      sizeUnits: "pixels",
      billboard: true,
      getTextAnchor: "middle",
      getAlignmentBaseline: "center",
    }),
  ];
}

const BASE_COLOR = [120, 144, 180];
const NEIGHBOR_COLOR = [255, 159, 64];
const QUERY_COLOR = [80, 220, 130];

// Fit an OrbitView so the whole cloud is visible: center on the centroid and
// pick a zoom from the point spread.
function fitViewState(points) {
  if (!points.length) {
    return { target: [0, 0, 0], zoom: 3, rotationX: 20, rotationOrbit: 20 };
  }
  const c = [0, 0, 0];
  for (const p of points) {
    c[0] += p.x;
    c[1] += p.y;
    c[2] += p.z;
  }
  c[0] /= points.length;
  c[1] /= points.length;
  c[2] /= points.length;

  let maxR = 1e-3;
  for (const p of points) {
    const d = Math.hypot(p.x - c[0], p.y - c[1], p.z - c[2]);
    if (d > maxR) maxR = d;
  }
  // Larger spread -> zoom out. Empirical mapping that frames most clouds well.
  const zoom = Math.log2(60 / maxR);
  return { target: c, zoom, rotationX: 25, rotationOrbit: 30, minZoom: -5, maxZoom: 20 };
}

export default function Cloud3D({ points, neighborIds, query }) {
  const [hover, setHover] = useState(null);
  const initialViewState = useMemo(() => fitViewState(points), [points]);

  const neighborSet = useMemo(() => new Set(neighborIds), [neighborIds]);

  const baseLayer = new ScatterplotLayer({
    id: "points",
    data: points,
    getPosition: (d) => [d.x, d.y, d.z],
    getFillColor: (d) => (neighborSet.has(d.id) ? NEIGHBOR_COLOR : BASE_COLOR),
    getRadius: (d) => (neighborSet.has(d.id) ? 1.6 : 0.7),
    radiusUnits: "common",
    billboard: true,
    pickable: true,
    opacity: 0.85,
    onHover: (info) => setHover(info.object ? info : null),
    updateTriggers: {
      getFillColor: [neighborIds],
      getRadius: [neighborIds],
    },
  });

  const layers = [...buildReferenceLayers(points), baseLayer];

  if (query) {
    layers.push(
      new ScatterplotLayer({
        id: "query",
        data: [query],
        getPosition: (d) => [d.x, d.y, d.z],
        getFillColor: QUERY_COLOR,
        getRadius: 2.6,
        radiusUnits: "common",
        billboard: true,
      })
    );
    const links = points
      .filter((p) => neighborSet.has(p.id))
      .map((p) => ({ from: query, to: p }));
    layers.push(
      new LineLayer({
        id: "links",
        data: links,
        getSourcePosition: (d) => [d.from.x, d.from.y, d.from.z],
        getTargetPosition: (d) => [d.to.x, d.to.y, d.to.z],
        getColor: [255, 159, 64, 160],
        getWidth: 1.5,
      })
    );
  }

  return (
    <DeckGL
      views={new OrbitView({ orbitAxis: "Y" })}
      initialViewState={initialViewState}
      controller={true}
      layers={layers}
    >
      {hover && hover.object && (
        <div
          className="tooltip"
          style={{ left: hover.x + 12, top: hover.y + 12 }}
        >
          <div className="tooltip-id">#{hover.object.id}</div>
          <div>{hover.object.payload || <em>(no payload)</em>}</div>
        </div>
      )}
    </DeckGL>
  );
}
