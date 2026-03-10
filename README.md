# CoreVector

A vector database built from scratch in C++. I use vector databases and embedding APIs in full-stack projects all the time, and wanted to understand what's actually happening under the hood — how vectors get stored, indexed, and searched efficiently.

## Features

- **Two index types** — exact brute-force (FlatIndex) and approximate nearest neighbor (HnswIndex)
- **SIMD-accelerated math** — L2 and cosine distance with xsimd, with scalar fallback
- **Vector payloads** — attach a text string to each vector
- **Persistence** — save/load indexes to binary files
- **Memory-mapped loading** — zero-copy index loading via mmap
- **Parallel search** — multi-threaded brute-force search
- **gRPC API** — remote insert, search, save, and load over the network
- **Python client** — example gRPC client for interacting with the server

## Tech Stack

| Layer | Technology |
|---|---|
| Language | C++20 |
| Build | CMake 3.20+ |
| SIMD | xsimd 11.1.0 |
| Networking | gRPC + Protobuf |
| Testing | Google Test |
| Benchmarking | Google Benchmark |
| Python client | grpcio, protobuf |

## Architecture

**FlatIndex** does an exhaustive O(N) linear scan over all vectors — guaranteed exact results, optionally parallelized across CPU cores. **HnswIndex** builds a hierarchical graph (Hierarchical Navigable Small World) for approximate O(log N) search, trading a small amount of recall for dramatically faster queries at scale.

Both indexes support the same interface: `Add`, `Search`, `Save`, and `Load`.

## Build

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

Requires gRPC and Protobuf installed on your system.

## Usage

### C++ API

```cpp
#include "flat_index.hpp"
#include "hnsw_index.hpp"

// Exact search
FlatIndex flat(128);
flat.Add(vec, "optional payload");
auto results = flat.Search(query, /*k=*/5);
auto results = flat.SearchParallel(query, 5);  // multi-threaded

// Approximate search
HnswIndex hnsw(128, /*M=*/16, /*ef_construction=*/200);
hnsw.Add(vec, "payload");
auto results = hnsw.Search(query, 5, /*ef_search=*/50);

// Persistence
flat.Save("index.bin");
flat.Load("index.bin");
flat.MmapLoad("index.bin");  // zero-copy
```

### gRPC Server

```bash
./build/corevector_server        # default: 128-dimensional vectors
./build/corevector_server 256    # custom dimension

python3 scripts/client.py        # example Python client
```

The server exposes `Insert`, `Search`, `Save`, and `Load` RPCs on port 50051.

## Tests

```bash
cd build && ctest
# or individually:
./test_flat_index
./test_hnsw
./test_vector_math
./test_serialization
./test_mmap
```

## Benchmarks

```bash
./build/benchmark_math         # SIMD vs naive distance functions
./build/benchmark_hnsw         # FlatIndex vs HnswIndex scaling (10K–100K vectors)
./build/benchmark_concurrency  # single-threaded vs parallel search

# Network overhead (start server first)
./build/corevector_server &
./build/benchmark_network      # local in-process vs gRPC round-trip latency
```
