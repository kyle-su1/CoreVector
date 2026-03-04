#include "math_simd.hpp"
#include "vector.hpp"
#include <benchmark/benchmark.h>
#include <random>

using namespace corevector;

// Helper to generate random vectors
Vector GenerateRandomVector(size_t dim) {
  Vector v(dim);
  std::mt19937 gen(42); // Fixed seed for reproducibility
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < dim; ++i) {
    v.data[i] = dist(gen);
  }
  return v;
}

// -----------------------------------------------------------------------------
// L2 Distance Benchmarks
// -----------------------------------------------------------------------------
static void BM_NaiveL2(benchmark::State &state) {
  const size_t dim = state.range(0);
  Vector a = GenerateRandomVector(dim);
  Vector b = GenerateRandomVector(dim);

  for (auto _ : state) {
    float dist = math::naive::L2(a, b);
    benchmark::DoNotOptimize(dist);
  }
}
BENCHMARK(BM_NaiveL2)->RangeMultiplier(2)->Range(128, 4096);

static void BM_SimdL2(benchmark::State &state) {
  const size_t dim = state.range(0);
  Vector a = GenerateRandomVector(dim);
  Vector b = GenerateRandomVector(dim);

  for (auto _ : state) {
    float dist = math::simd::L2(a, b);
    benchmark::DoNotOptimize(dist);
  }
}
BENCHMARK(BM_SimdL2)->RangeMultiplier(2)->Range(128, 4096);

// -----------------------------------------------------------------------------
// Cosine distance benchmarks
// -----------------------------------------------------------------------------
static void BM_NaiveCosine(benchmark::State &state) {
  const size_t dim = state.range(0);
  Vector a = GenerateRandomVector(dim);
  Vector b = GenerateRandomVector(dim);

  for (auto _ : state) {
    float dist = math::naive::CosineDistance(a, b);
    benchmark::DoNotOptimize(dist);
  }
}
BENCHMARK(BM_NaiveCosine)->RangeMultiplier(2)->Range(128, 4096);

static void BM_SimdCosine(benchmark::State &state) {
  const size_t dim = state.range(0);
  Vector a = GenerateRandomVector(dim);
  Vector b = GenerateRandomVector(dim);

  for (auto _ : state) {
    float dist = math::simd::CosineDistance(a, b);
    benchmark::DoNotOptimize(dist);
  }
}
BENCHMARK(BM_SimdCosine)->RangeMultiplier(2)->Range(128, 4096);

BENCHMARK_MAIN();
