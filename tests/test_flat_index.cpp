#include "flat_index.hpp"
#include <gtest/gtest.h>
#include <random>

using namespace corevector;

// ============================================================================
// Search Correctness Tests
// ============================================================================

TEST(FlatIndexTest, Search_KnownNearest) {
  // Create 3 vectors, query is closest to vec[1]
  FlatIndex index(3);

  Vector v0(3), v1(3), v2(3);
  v0.data = {10.0f, 10.0f, 10.0f}; // Far away
  v1.data = {1.0f, 1.0f, 1.0f};    // Closest to query
  v2.data = {5.0f, 5.0f, 5.0f};    // Medium distance

  index.Add(v0);
  index.Add(v1);
  index.Add(v2);

  Vector query(3);
  query.data = {1.1f, 1.1f, 1.1f}; // Very close to v1

  auto results = index.Search(query, 1);

  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].id, 1); // v1 should be the nearest
}

TEST(FlatIndexTest, Search_ReturnsKResults) {
  FlatIndex index(2);

  for (int i = 0; i < 20; ++i) {
    Vector v(2);
    v.data = {static_cast<float>(i), static_cast<float>(i)};
    index.Add(v);
  }

  auto results = index.Search(Vector(2), 5);
  EXPECT_EQ(results.size(), 5);
}

TEST(FlatIndexTest, Search_ResultsSortedByDistance) {
  FlatIndex index(2);

  // Add vectors at increasing distances from origin
  for (int i = 0; i < 10; ++i) {
    Vector v(2);
    v.data = {static_cast<float>(i * 3), static_cast<float>(i * 3)};
    index.Add(v);
  }

  Vector query(2);
  query.data = {0.0f, 0.0f}; // Origin

  auto results = index.Search(query, 5);

  // Results should be sorted by distance (ascending)
  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_LE(results[i - 1].distance, results[i].distance)
        << "Results not sorted at index " << i;
  }
}

TEST(FlatIndexTest, SearchParallel_MatchesSequential) {
  FlatIndex index(64);

  std::mt19937 gen(99);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Add 1000 random vectors
  for (int i = 0; i < 1000; ++i) {
    Vector v(64);
    for (size_t j = 0; j < 64; ++j) {
      v.data[j] = dist(gen);
    }
    index.Add(v);
  }

  // Create a query
  Vector query(64);
  for (size_t j = 0; j < 64; ++j) {
    query.data[j] = dist(gen);
  }

  auto seq_results = index.Search(query, 10);
  auto par_results = index.SearchParallel(query, 10);

  ASSERT_EQ(seq_results.size(), par_results.size());
  for (size_t i = 0; i < seq_results.size(); ++i) {
    EXPECT_EQ(seq_results[i].id, par_results[i].id)
        << "Mismatch at result index " << i;
    EXPECT_FLOAT_EQ(seq_results[i].distance, par_results[i].distance);
  }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(FlatIndexTest, Search_EmptyIndex) {
  FlatIndex index(128);
  Vector query(128);

  auto results = index.Search(query, 5);
  EXPECT_TRUE(results.empty());
}

TEST(FlatIndexTest, Search_KZero) {
  FlatIndex index(2);
  Vector v(2);
  v.data = {1.0f, 2.0f};
  index.Add(v);

  Vector query(2);
  query.data = {0.0f, 0.0f};

  auto results = index.Search(query, 0);
  EXPECT_TRUE(results.empty());
}

TEST(FlatIndexTest, Search_KLargerThanSize) {
  FlatIndex index(2);

  // Add only 3 vectors
  for (int i = 0; i < 3; ++i) {
    Vector v(2);
    v.data = {static_cast<float>(i), 0.0f};
    index.Add(v);
  }

  Vector query(2);
  query.data = {0.0f, 0.0f};

  // Request 100, should get only 3
  auto results = index.Search(query, 100);
  EXPECT_EQ(results.size(), 3);
}

TEST(FlatIndexTest, Add_DimensionMismatch) {
  FlatIndex index(3);

  Vector wrong_dim(5);
  wrong_dim.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  EXPECT_THROW(index.Add(wrong_dim), std::invalid_argument);
}

TEST(FlatIndexTest, Search_DimensionMismatch) {
  FlatIndex index(3);

  Vector v(3);
  v.data = {1.0f, 2.0f, 3.0f};
  index.Add(v);

  Vector wrong_query(5);
  wrong_query.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  EXPECT_THROW(index.Search(wrong_query, 1), std::invalid_argument);
}
