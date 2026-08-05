import React, { useCallback, useEffect, useState } from "react";
import Cloud3D from "./components/Cloud3D.jsx";
import SearchPanel from "./components/SearchPanel.jsx";
import ResultsList from "./components/ResultsList.jsx";
import { getVectors, search, insert, makeDemoVectors } from "./api.js";

export default function App() {
  const [points, setPoints] = useState([]);
  const [neighbors, setNeighbors] = useState([]);
  const [query, setQuery] = useState(null);
  const [queryId, setQueryId] = useState("0");
  const [k, setK] = useState(5);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const reload = useCallback(async () => {
    setError(null);
    try {
      const data = await getVectors();
      setPoints(data.points);
    } catch (e) {
      setError(e.message);
    }
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  const onSearch = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const id = queryId === "" ? undefined : Number(queryId);
      const res = await search({ id, k });
      setNeighbors(res.neighbors);
      setQuery(res.query);
    } catch (e) {
      setError(e.message);
      setNeighbors([]);
      setQuery(null);
    } finally {
      setBusy(false);
    }
  }, [queryId, k]);

  const onSeed = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      await insert(makeDemoVectors(300));
      setNeighbors([]);
      setQuery(null);
      await reload();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }, [reload]);

  const neighborIds = neighbors.map((n) => n.id);

  return (
    <div className="app">
      <div className="sidebar">
        <SearchPanel
          queryId={queryId}
          setQueryId={setQueryId}
          k={k}
          setK={setK}
          onSearch={onSearch}
          onSeed={onSeed}
          onReload={reload}
          count={points.length}
          busy={busy}
          error={error}
        />
        <ResultsList neighbors={neighbors} />
      </div>
      <div className="canvas-wrap">
        {points.length === 0 ? (
          <div className="empty">
            No vectors yet — click <strong>Seed demo data</strong> to populate
            the index.
          </div>
        ) : (
          <Cloud3D points={points} neighborIds={neighborIds} query={query} />
        )}
      </div>
    </div>
  );
}
