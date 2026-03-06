/**
 * BENCHMARK 4: Algorithmic Complexity
 *
 * Compares FlatIndex (brute-force O(N)) vs. HnswIndex (approximate O(log N))
 * on a large dataset to demonstrate the algorithmic scaling advantage.
 */
#include "flat_index.hpp"
#include "hnsw_index.hpp"
#include <benchmark/benchmark.h>
#include <random>

using namespace corevector;

static constexpr size_t DIM = 128;
static constexpr size_t K = 10;

// Helper: generate a random vector
static Vector MakeRandomVector(size_t dim, unsigned seed) {
  Vector v(dim);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < dim; ++i) {
    v.data[i] = dist(gen);
  }
  return v;
}

// ---------------------------------------------------------------------------
// FlatIndex Benchmark (Brute-Force O(N))
// ---------------------------------------------------------------------------

static void BM_FlatIndex_Search(benchmark::State &state) {
  const size_t num_vectors = state.range(0);

  static FlatIndex *flat_ptr = nullptr;
  static size_t last_flat_size = 0;
  if (!flat_ptr || last_flat_size != num_vectors) {
    delete flat_ptr;
    flat_ptr = new FlatIndex(DIM);
    for (size_t i = 0; i < num_vectors; ++i) {
      flat_ptr->Add(MakeRandomVector(DIM, i));
    }
    last_flat_size = num_vectors;
  }

  Vector query = MakeRandomVector(DIM, 99999);

  for (auto _ : state) {
    auto results = flat_ptr->Search(query, K);
    benchmark::DoNotOptimize(results);
  }
}
BENCHMARK(BM_FlatIndex_Search)
    ->RangeMultiplier(10)
    ->Range(10000, 100000)
    ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// HnswIndex Benchmark (Approximate O(log N))
// ---------------------------------------------------------------------------

static void BM_HnswIndex_Search(benchmark::State &state) {
  const size_t num_vectors = state.range(0);

  static HnswIndex *hnsw_ptr = nullptr;
  static size_t last_hnsw_size = 0;
  if (!hnsw_ptr || last_hnsw_size != num_vectors) {
    delete hnsw_ptr;
    hnsw_ptr = new HnswIndex(DIM, /*M=*/16, /*ef_construction=*/200);
    for (size_t i = 0; i < num_vectors; ++i) {
      hnsw_ptr->Add(MakeRandomVector(DIM, i));
    }
    last_hnsw_size = num_vectors;
  }

  Vector query = MakeRandomVector(DIM, 99999);

  for (auto _ : state) {
    auto results = hnsw_ptr->Search(query, K, /*ef_search=*/50);
    benchmark::DoNotOptimize(results);
  }
}
BENCHMARK(BM_HnswIndex_Search)
    ->RangeMultiplier(10)
    ->Range(10000, 100000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
