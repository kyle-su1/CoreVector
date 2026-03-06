#!/usr/bin/env python3
"""
CoreVector gRPC Client Demo

Usage:
    1. Start the server:   ./build/corevector_server
    2. Run this script:    python3 scripts/client.py

Prerequisites:
    pip install grpcio grpcio-tools

Generate Python stubs (run from project root):
    python3 -m grpc_tools.protoc \
        --proto_path=proto \
        --python_out=scripts \
        --grpc_python_out=scripts \
        proto/vector_db.proto
"""

import sys
import os
import random
import time

# Add generated proto stubs to path
sys.path.insert(0, os.path.dirname(__file__))

import grpc
import vector_db_pb2
import vector_db_pb2_grpc


def run():
    channel = grpc.insecure_channel("localhost:50051")
    stub = vector_db_pb2_grpc.VectorDBStub(channel)

    DIM = 128
    K = 5

    # Sample payloads — sentences that would represent real embedded text
    sample_payloads = [
        "The quick brown fox jumps over the lazy dog",
        "Apple announces new MacBook Pro with M4 chip",
        "How to train a neural network from scratch",
        "Best Italian restaurants near downtown San Francisco",
        "Introduction to quantum computing for beginners",
        "NASA discovers water on distant exoplanet",
        "Python vs C++ performance comparison guide",
        "Stock market reaches all-time high today",
        "Recipe for homemade sourdough bread",
        "Understanding memory-mapped I/O in operating systems",
    ]

    NUM_VECTORS = len(sample_payloads)

    # ----- Insert with Payloads -----
    print(f"Inserting {NUM_VECTORS} vectors with text payloads (dim={DIM})...")
    vectors = []
    for i, payload in enumerate(sample_payloads):
        random.seed(i)  # Reproducible vectors
        vec = vector_db_pb2.VectorData(
            values=[random.uniform(-1.0, 1.0) for _ in range(DIM)],
            payload=payload,
        )
        vectors.append(vec)

    start = time.time()
    response = stub.Insert(vector_db_pb2.InsertRequest(vectors=vectors))
    elapsed = time.time() - start
    print(f"  Inserted! Total vectors in index: {response.total_vectors}")
    print(f"  Insert latency: {elapsed * 1000:.2f} ms")

    # ----- Search -----
    random.seed(0)  # Use vector 0's seed to find "The quick brown fox"
    query = [random.uniform(-1.0, 1.0) for _ in range(DIM)]
    print(f"\nSearching for top {K} nearest neighbors...")

    start = time.time()
    response = stub.Search(vector_db_pb2.SearchRequest(query=query, k=K))
    elapsed = time.time() - start

    print(f"  Search latency: {elapsed * 1000:.2f} ms")
    print(f"  Results:")
    for r in response.results:
        print(f"    ID: {r.id:>5}  Distance: {r.distance:.6f}  Payload: \"{r.payload}\"")

    # ----- Save -----
    print("\nSaving index to disk...")
    response = stub.Save(vector_db_pb2.SaveRequest(filename="demo_index.bin"))
    print(f"  {response.message}")

    # ----- Load -----
    print("Loading index from disk...")
    response = stub.Load(vector_db_pb2.LoadRequest(filename="demo_index.bin"))
    print(f"  {response.message} ({response.total_vectors} vectors)")

    # ----- Search again after load -----
    print(f"\nSearching again after reload (top {K})...")
    response = stub.Search(vector_db_pb2.SearchRequest(query=query, k=K))
    print(f"  Results:")
    for r in response.results:
        print(f"    ID: {r.id:>5}  Distance: {r.distance:.6f}  Payload: \"{r.payload}\"")

    print("\n✅ All operations completed successfully!")


if __name__ == "__main__":
    run()

