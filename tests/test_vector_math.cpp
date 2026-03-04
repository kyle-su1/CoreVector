#include "math_simd.hpp"
#include "vector.hpp"
#include <gtest/gtest.h>
#include <random>

using namespace corevector;

// ============================================================================
// L2 Squared Distance Tests
// ============================================================================

TEST(VectorMathTest, L2Sqr_IdenticalVectors) {
  Vector a(3);
  a.data = {1.0f, 2.0f, 3.0f};

  EXPECT_FLOAT_EQ(math::naive::L2Sqr(a, a), 0.0f);
  EXPECT_FLOAT_EQ(math::simd::L2Sqr(a, a), 0.0f);
}

TEST(VectorMathTest, L2Sqr_KnownValues) {
  // [1, 0] vs [0, 1] → (1-0)^2 + (0-1)^2 = 1 + 1 = 2.0
  Vector a(2), b(2);
  a.data = {1.0f, 0.0f};
  b.data = {0.0f, 1.0f};

  EXPECT_FLOAT_EQ(math::naive::L2Sqr(a, b), 2.0f);
  EXPECT_FLOAT_EQ(math::simd::L2Sqr(a, b), 2.0f);

  // [3, 4] vs [0, 0] → 9 + 16 = 25.0
  Vector c(2), d(2);
  c.data = {3.0f, 4.0f};
  d.data = {0.0f, 0.0f};

  EXPECT_FLOAT_EQ(math::naive::L2Sqr(c, d), 25.0f);
  EXPECT_FLOAT_EQ(math::naive::L2(c, d), 5.0f); // sqrt(25) = 5
}

// ============================================================================
// Cosine Similarity Tests
// ============================================================================

TEST(VectorMathTest, CosineSimilarity_Parallel) {
  // Identical direction → similarity = 1.0
  Vector a(3), b(3);
  a.data = {1.0f, 2.0f, 3.0f};
  b.data = {2.0f, 4.0f, 6.0f}; // Same direction, different magnitude

  EXPECT_NEAR(math::naive::CosineSimilarity(a, b), 1.0f, 1e-5f);
  EXPECT_NEAR(math::simd::CosineSimilarity(a, b), 1.0f, 1e-5f);
}

TEST(VectorMathTest, CosineSimilarity_Orthogonal) {
  // Perpendicular vectors → similarity = 0.0
  Vector a(2), b(2);
  a.data = {1.0f, 0.0f};
  b.data = {0.0f, 1.0f};

  EXPECT_NEAR(math::naive::CosineSimilarity(a, b), 0.0f, 1e-5f);
  EXPECT_NEAR(math::simd::CosineSimilarity(a, b), 0.0f, 1e-5f);
}

TEST(VectorMathTest, CosineSimilarity_Opposite) {
  // Opposite direction → similarity = -1.0
  Vector a(3), b(3);
  a.data = {1.0f, 2.0f, 3.0f};
  b.data = {-1.0f, -2.0f, -3.0f};

  EXPECT_NEAR(math::naive::CosineSimilarity(a, b), -1.0f, 1e-5f);
  EXPECT_NEAR(math::simd::CosineSimilarity(a, b), -1.0f, 1e-5f);
}

TEST(VectorMathTest, CosineDistance_KnownValues) {
  // Same direction → distance = 0.0
  Vector a(2), b(2);
  a.data = {1.0f, 0.0f};
  b.data = {3.0f, 0.0f};

  EXPECT_NEAR(math::naive::CosineDistance(a, b), 0.0f, 1e-5f);
  EXPECT_NEAR(math::simd::CosineDistance(a, b), 0.0f, 1e-5f);

  // Opposite direction → distance = 2.0
  Vector c(2), d(2);
  c.data = {1.0f, 0.0f};
  d.data = {-1.0f, 0.0f};

  EXPECT_NEAR(math::naive::CosineDistance(c, d), 2.0f, 1e-5f);
  EXPECT_NEAR(math::simd::CosineDistance(c, d), 2.0f, 1e-5f);
}

// ============================================================================
// Cross-Implementation Agreement (Naive vs SIMD)
// ============================================================================

TEST(VectorMathTest, NaiveVsSimd_L2Sqr) {
  const size_t dim = 256;
  Vector a(dim), b(dim);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  for (size_t i = 0; i < dim; ++i) {
    a.data[i] = dist(gen);
    b.data[i] = dist(gen);
  }

  float naive_result = math::naive::L2Sqr(a, b);
  float simd_result = math::simd::L2Sqr(a, b);

  EXPECT_NEAR(naive_result, simd_result,
              1e-2f); // Allow small floating point drift
}

TEST(VectorMathTest, NaiveVsSimd_Cosine) {
  const size_t dim = 256;
  Vector a(dim), b(dim);
  std::mt19937 gen(42);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  for (size_t i = 0; i < dim; ++i) {
    a.data[i] = dist(gen);
    b.data[i] = dist(gen);
  }

  float naive_result = math::naive::CosineSimilarity(a, b);
  float simd_result = math::simd::CosineSimilarity(a, b);

  EXPECT_NEAR(naive_result, simd_result, 1e-5f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(VectorMathTest, ZeroVector) {
  Vector a(3), zero(3);
  a.data = {1.0f, 2.0f, 3.0f};
  zero.data = {0.0f, 0.0f, 0.0f};

  // Cosine with a zero vector should return 0.0 (not NaN or crash)
  EXPECT_FLOAT_EQ(math::naive::CosineSimilarity(a, zero), 0.0f);
  EXPECT_FLOAT_EQ(math::simd::CosineSimilarity(a, zero), 0.0f);
}

TEST(VectorMathTest, DimensionMismatch) {
  Vector a(3), b(5);
  a.data = {1.0f, 2.0f, 3.0f};
  b.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  EXPECT_THROW(math::naive::L2Sqr(a, b), std::invalid_argument);
  EXPECT_THROW(math::simd::L2Sqr(a, b), std::invalid_argument);
  EXPECT_THROW(math::naive::CosineSimilarity(a, b), std::invalid_argument);
  EXPECT_THROW(math::simd::CosineSimilarity(a, b), std::invalid_argument);
}
