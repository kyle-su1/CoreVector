#pragma once

#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace corevector {

struct Vector {
  std::vector<float> data;

  Vector() = default;

  explicit Vector(size_t dim) : data(dim, 0.0f) {}
  Vector(std::initializer_list<float> values) : data(values) {}

  size_t dim() const { return data.size(); }
};

namespace math {
namespace naive {

// Computes the squared L2 distance between two vectors
inline float L2Sqr(const Vector &a, const Vector &b) {
  if (a.dim() != b.dim()) {
    throw std::invalid_argument("Vector dimensions must match");
  }

  float distance = 0.0f;
  for (size_t i = 0; i < a.dim(); ++i) {
    float diff = a.data[i] - b.data[i];
    distance += diff * diff;
  }
  return distance;
}

// Computes the Euclidean (L2) distance between two vectors
inline float L2(const Vector &a, const Vector &b) {
  return std::sqrt(L2Sqr(a, b));
}

// Computes the cosine similarity between two vectors
// Returns a value between -1.0 and 1.0
inline float CosineSimilarity(const Vector &a, const Vector &b) {
  if (a.dim() != b.dim()) {
    throw std::invalid_argument("Vector dimensions must match");
  }

  float dot_product = 0.0f;
  float norm_a_sqr = 0.0f;
  float norm_b_sqr = 0.0f;

  for (size_t i = 0; i < a.dim(); ++i) {
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

} // namespace naive
} // namespace math

} // namespace corevector
