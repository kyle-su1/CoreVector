#!/usr/bin/env python3
"""
CoreVector Visualization BFF (Backend-For-Frontend)

Bridges the browser (HTTP/JSON) to the C++ gRPC server, and does the
128-D -> 3-D dimensionality reduction (PCA via numpy SVD) so the frontend
only ever receives 3 coordinates per point.

Run (from project root, with the venv active):
    ./build/corevector_server                 # gRPC backend on :50051
    uvicorn bff.app:app --port 8000 --reload  # this service on :8000
"""

import os
import sys
import threading

import grpc
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Reuse the generated gRPC stubs from scripts/ (same pattern as scripts/client.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import vector_db_pb2  # noqa: E402
import vector_db_pb2_grpc  # noqa: E402

GRPC_TARGET = os.environ.get("COREVECTOR_GRPC", "localhost:50051")

app = FastAPI(title="CoreVector Visualization BFF")
app.add_middleware(
    CORSMiddleware,
    # Vite dev server (port floats to 5174+ if 5173 is taken). Requests through
    # the Vite proxy are same-origin and don't need this, but a direct browser
    # hit to :8000 does.
    allow_origin_regex=r"http://localhost:\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

_channel = grpc.insecure_channel(GRPC_TARGET)
_stub = vector_db_pb2_grpc.VectorDBStub(_channel)

# Cached PCA basis + the raw vectors it was fit on. Guarded by a lock because
# uvicorn may serve requests from multiple threads.
_lock = threading.Lock()
_pca = None  # dict: {mean, components (3 x dim), ids, values (n x dim)}


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class VectorIn(BaseModel):
    values: list[float]
    payload: str = ""


class InsertIn(BaseModel):
    vectors: list[VectorIn]


class SearchIn(BaseModel):
    query: list[float] | None = None
    id: int | None = None
    k: int = 5


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fit_pca():
    """Pull every vector from the backend and fit a 3-component PCA basis.

    Returns the cached dict, or None if the index is empty. Result is memoized
    in the module-level _pca until an insert invalidates it.
    """
    global _pca
    resp = _stub.GetAllVectors(vector_db_pb2.GetAllVectorsRequest())
    n = len(resp.records)
    if n == 0:
        _pca = None
        return None

    dim = resp.dim
    values = np.empty((n, dim), dtype=np.float32)
    ids = np.empty(n, dtype=np.int64)
    payloads = []
    for row, rec in enumerate(resp.records):
        values[row] = np.asarray(rec.values, dtype=np.float32)
        ids[row] = rec.id
        payloads.append(rec.payload)

    mean = values.mean(axis=0)
    centered = values - mean
    # Top-3 right singular vectors = principal directions. With < 3 samples we
    # pad the basis with zero rows so the projection is still well-defined.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    k = min(3, vt.shape[0])
    components = np.zeros((3, dim), dtype=np.float32)
    components[:k] = vt[:k]

    _pca = {
        "mean": mean,
        "components": components,  # (3, dim)
        "ids": ids,
        "values": values,  # (n, dim) — reused to look up a query by id
        "payloads": payloads,
    }
    return _pca


def _project(vecs: np.ndarray, pca) -> np.ndarray:
    """Project (n, dim) rows into the cached 3-D PCA space -> (n, 3)."""
    return (vecs - pca["mean"]) @ pca["components"].T


def _grpc_call(fn):
    try:
        return fn()
    except grpc.RpcError as exc:  # pragma: no cover - passthrough of backend errors
        raise HTTPException(status_code=502, detail=f"gRPC error: {exc.details()}")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/vectors")
def get_vectors():
    """All vectors projected to 3-D, ready to plot."""
    with _lock:
        pca = _pca if _pca is not None else _fit_pca()
        if pca is None:
            return {"count": 0, "dim": 0, "points": []}
        coords = _project(pca["values"], pca)
        points = [
            {
                "id": int(pca["ids"][i]),
                "x": float(coords[i, 0]),
                "y": float(coords[i, 1]),
                "z": float(coords[i, 2]),
                "payload": pca["payloads"][i],
            }
            for i in range(len(pca["ids"]))
        ]
    return {"count": len(points), "dim": int(pca["components"].shape[1]), "points": points}


@app.post("/api/search")
def search(req: SearchIn):
    """k-NN search. Query by explicit vector or by an existing point id.

    Returns the neighbors plus the query projected into the same 3-D PCA basis
    so the frontend can plot it alongside the cloud.
    """
    with _lock:
        pca = _pca if _pca is not None else _fit_pca()
        if pca is None:
            raise HTTPException(status_code=409, detail="Index is empty")

        if req.query is not None:
            query = np.asarray(req.query, dtype=np.float32)
        elif req.id is not None:
            match = np.where(pca["ids"] == req.id)[0]
            if len(match) == 0:
                raise HTTPException(status_code=404, detail=f"No vector with id {req.id}")
            query = pca["values"][match[0]]
        else:
            raise HTTPException(status_code=400, detail="Provide 'query' or 'id'")

        query_xyz = _project(query.reshape(1, -1), pca)[0]

    resp = _grpc_call(
        lambda: _stub.Search(
            vector_db_pb2.SearchRequest(query=query.tolist(), k=req.k)
        )
    )
    neighbors = [
        {"id": int(r.id), "distance": float(r.distance), "payload": r.payload}
        for r in resp.results
    ]
    return {
        "neighbors": neighbors,
        "query": {
            "x": float(query_xyz[0]),
            "y": float(query_xyz[1]),
            "z": float(query_xyz[2]),
        },
    }


@app.post("/api/insert")
def insert(req: InsertIn):
    """Insert vectors, then invalidate the cached PCA basis."""
    global _pca
    vectors = [
        vector_db_pb2.VectorData(values=v.values, payload=v.payload)
        for v in req.vectors
    ]
    resp = _grpc_call(
        lambda: _stub.Insert(vector_db_pb2.InsertRequest(vectors=vectors))
    )
    with _lock:
        _pca = None  # refit lazily on next /api/vectors
    return {"total_vectors": int(resp.total_vectors)}


@app.get("/api/health")
def health():
    return {"ok": True, "grpc_target": GRPC_TARGET}


@app.get("/")
def root():
    """This is the API service, not the web app. Point users to the frontend."""
    return {
        "service": "CoreVector Visualization BFF",
        "note": "This is the API, not the web UI. Open the frontend dev server instead.",
        "frontend": "http://localhost:5173  (or 5174 if 5173 is taken)",
        "endpoints": ["/api/vectors", "/api/search", "/api/insert", "/api/health"],
    }
