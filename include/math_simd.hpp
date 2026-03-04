#pragma once

#include "vector.hpp"
#include <xsimd/xsimd.hpp>

namespace corevector {
namespace math {
namespace simd {

// Computes the squared L2 distance between two vectors using xsimd
inline float L2Sqr(const Vector &a, const Vector &b) {
  if (a.dim() != b.dim()) {
    throw std::invalid_argument("Vector dimensions must match");
  }

  const size_t size = a.dim();
  constexpr size_t simd_size = xsimd::simd_type<float>::size;
  const size_t vec_size = size - (size % simd_size);

  auto va = xsimd::batch<float>::broadcast(0.0f);
  auto dist_batch = xsimd::batch<float>::broadcast(0.0f);

  size_t i = 0;
  for (; i < vec_size; i += simd_size) {
    auto batch_a = xsimd::load_unaligned(&a.data[i]);
    auto batch_b = xsimd::load_unaligned(&b.data[i]);
    auto diff = batch_a - batch_b;
    dist_batch += diff * diff;
  }

  float distance = xsimd::reduce_add(dist_batch);

  // Scalar fallback for remaining elements
  for (; i < size; ++i) {
    float diff = a.data[i] - b.data[i];
    distance += diff * diff;
  }

  return distance;
}

// Computes the Euclidean (L2) distance between two vectors using xsimd
inline float L2(const Vector &a, const Vector &b) {
  return std::sqrt(L2Sqr(a, b));
}

// Computes the cosine similarity between two vectors using xsimd
// Returns a value between -1.0 and 1.0
inline float CosineSimilarity(const Vector &a, const Vector &b) {
  if (a.dim() != b.dim()) {
    throw std::invalid_argument("Vector dimensions must match");
  }

  const size_t size = a.dim();
  constexpr size_t simd_size = xsimd::simd_type<float>::size;
  const size_t vec_size = size - (size % simd_size);

  auto dot_batch = xsimd::batch<float>::broadcast(0.0f);
  auto norm_a_batch = xsimd::batch<float>::broadcast(0.0f);
  auto norm_b_batch = xsimd::batch<float>::broadcast(0.0f);

  size_t i = 0;
  for (; i < vec_size; i += simd_size) {
    auto batch_a = xsimd::load_unaligned(&a.data[i]);
    auto batch_b = xsimd::load_unaligned(&b.data[i]);

    dot_batch += batch_a * batch_b;
    norm_a_batch += batch_a * batch_a;
    norm_b_batch += batch_b * batch_b;
  }

  float dot_product = xsimd::reduce_add(dot_batch);
  float norm_a_sqr = xsimd::reduce_add(norm_a_batch);
  float norm_b_sqr = xsimd::reduce_add(norm_b_batch);

  // Scalar fallback for remaining elements
  for (; i < size; ++i) {
    dot_product += a.data[i] * b.data[i];
    norm_a_sqr += a.data[i] * a.data[i];
    norm_b_sqr += b.data[i] * b.data[i];
  }

  if (norm_a_sqr == 0.0f || norm_b_sqr == 0.0f) {
    return 0.0f; // Handle zero vector case
  }

  return dot_product / (std::sqrt(norm_a_sqr) * std::sqrt(norm_b_sqr));
}

// Computes the cosine distance between two vectors (1.0 - CosineSimilarity)
// Returns a value between 0.0 and 2.0 (smaller is closer)
inline float CosineDistance(const Vector &a, const Vector &b) {
  return 1.0f - CosineSimilarity(a, b);
}

} // namespace simd
} // namespace math
} // namespace corevector
