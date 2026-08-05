#!/usr/bin/env python3
"""
Seed the running index with demo vectors through the BFF's /api/insert endpoint.

Usage (server + BFF must be running):
    python3 bff/seed.py            # inserts ~300 vectors in a few clusters
    python3 bff/seed.py 500 http://localhost:8000
"""
import json
import sys
import urllib.request

import numpy as np

SAMPLE_PAYLOADS = [
    "neural networks and deep learning",
    "italian restaurants downtown",
    "quantum computing basics",
    "stock market news today",
    "sourdough bread recipe",
    "memory-mapped IO in operating systems",
    "training a transformer from scratch",
    "best hiking trails in the alps",
]


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    base = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    dim = 128
    rng = np.random.default_rng(42)

    # A few gaussian clusters so the 3-D projection shows visible structure.
    num_clusters = len(SAMPLE_PAYLOADS)
    centers = rng.uniform(-1.0, 1.0, size=(num_clusters, dim))

    vectors = []
    for i in range(n):
        c = i % num_clusters
        vec = centers[c] + rng.normal(0.0, 0.15, size=dim)
        vectors.append(
            {"values": vec.astype(float).tolist(), "payload": SAMPLE_PAYLOADS[c]}
        )

    body = json.dumps({"vectors": vectors}).encode()
    req = urllib.request.Request(
        f"{base}/api/insert", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.load(resp)
    print(f"Inserted {n} vectors. Total now: {result['total_vectors']}")


if __name__ == "__main__":
    main()
