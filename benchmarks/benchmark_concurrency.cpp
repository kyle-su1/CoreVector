#include "flat_index.hpp"
#include <benchmark/benchmark.h>
#include <random>

using namespace corevector;

// Helper to generate a random index
FlatIndex GenerateRandomIndex(size_t num_vectors, size_t dim) {
  FlatIndex index(dim);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < num_vectors; ++i) {
    Vector v(dim);
    for (size_t j = 0; j < dim; ++j) {
      v.data[j] = dist(gen);
    }
    index.Add(v);
  }
  return index;
}

// Helper to generate a random query vector
Vector GenerateRandomQuery(size_t dim) {
  Vector v(dim);
  std::mt19937 gen(1337); // Different seed for queries
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < dim; ++i) {
    v.data[i] = dist(gen);
  }
  return v;
}

// -----------------------------------------------------------------------------
// Search Benchmarks (Varying Dataset Size)
// -----------------------------------------------------------------------------
static void BM_SearchSequential(benchmark::State &state) {
  const size_t num_vectors = state.range(0);
  const size_t dim = 128; // Standard embedding size
  const size_t k = 10;

  FlatIndex index = GenerateRandomIndex(num_vectors, dim);
  Vector query = GenerateRandomQuery(dim);

  for (auto _ : state) {
    auto results = index.Search(query, k);
    benchmark::DoNotOptimize(results);
  }
}
BENCHMARK(BM_SearchSequential)
    ->RangeMultiplier(10)
    ->Range(1000, 100000)
    ->Unit(benchmark::kMillisecond);

static void BM_SearchParallel(benchmark::State &state) {
  const size_t num_vectors = state.range(0);
  const size_t dim = 128; // Standard embedding size
  const size_t k = 10;

  FlatIndex index = GenerateRandomIndex(num_vectors, dim);
  Vector query = GenerateRandomQuery(dim);

  for (auto _ : state) {
    auto results = index.SearchParallel(query, k);
    benchmark::DoNotOptimize(results);
  }
}
// We expect parallel overhead to dominate for small datasets, but win on large
// datasets
BENCHMARK(BM_SearchParallel)
    ->RangeMultiplier(10)
    ->Range(1000, 100000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
