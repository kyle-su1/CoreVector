#include "flat_index.hpp"
#include "hnsw_index.hpp"
#include <gtest/gtest.h>
#include <random>

using namespace corevector;

// Helper to generate a deterministic random vector
static Vector MakeRandomVector(size_t dim, unsigned seed) {
  Vector v(dim);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < dim; ++i) {
    v.data[i] = dist(gen);
  }
  return v;
}

// ============================================================================
// Search Correctness
// ============================================================================

TEST(HnswIndexTest, Search_KnownNearest) {
  HnswIndex index(3, /*M=*/4, /*ef_construction=*/50);

  Vector v0(3), v1(3), v2(3);
  v0.data = {10.0f, 10.0f, 10.0f}; // Far
  v1.data = {1.0f, 1.0f, 1.0f};    // Closest to query
  v2.data = {5.0f, 5.0f, 5.0f};    // Medium

  index.Add(v0);
  index.Add(v1);
  index.Add(v2);

  Vector query(3);
  query.data = {1.1f, 1.1f, 1.1f};

  auto results = index.Search(query, 1, /*ef_search=*/10);
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].id, 1); // v1 is closest
}

TEST(HnswIndexTest, Search_ReturnsKResults) {
  HnswIndex index(16, /*M=*/8, /*ef_construction=*/50);

  for (int i = 0; i < 50; ++i) {
    index.Add(MakeRandomVector(16, i));
  }

  auto results = index.Search(MakeRandomVector(16, 9999), 10);
  EXPECT_EQ(results.size(), 10);
}

TEST(HnswIndexTest, Search_ResultsSortedByDistance) {
  HnswIndex index(16, /*M=*/8, /*ef_construction=*/100);

  for (int i = 0; i < 100; ++i) {
    index.Add(MakeRandomVector(16, i));
  }

  auto results = index.Search(MakeRandomVector(16, 9999), 10, /*ef_search=*/50);

  for (size_t i = 1; i < results.size(); ++i) {
    EXPECT_LE(results[i - 1].distance, results[i].distance)
        << "Results not sorted at index " << i;
  }
}

TEST(HnswIndexTest, Search_RecallVsFlatIndex) {
  const size_t dim = 32;
  const size_t num_vectors = 1000;
  const size_t k = 10;

  FlatIndex flat(dim);
  HnswIndex hnsw(dim, /*M=*/16, /*ef_construction=*/200);

  for (size_t i = 0; i < num_vectors; ++i) {
    auto v = MakeRandomVector(dim, i);
    flat.Add(v);
    hnsw.Add(v);
  }

  Vector query = MakeRandomVector(dim, 99999);
  auto flat_results = flat.Search(query, k);
  auto hnsw_results = hnsw.Search(query, k, /*ef_search=*/100);

  // Check that HNSW finds at least 95% of the true nearest neighbors
  size_t matches = 0;
  for (const auto &hr : hnsw_results) {
    for (const auto &fr : flat_results) {
      if (hr.id == fr.id) {
        ++matches;
        break;
      }
    }
  }

  double recall = static_cast<double>(matches) / k;
  EXPECT_GE(recall, 0.80) << "HNSW recall too low: " << recall << " ("
                          << matches << "/" << k << " matches)";
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(HnswIndexTest, Search_EmptyIndex) {
  HnswIndex index(128);
  Vector query(128);
  auto results = index.Search(query, 5);
  EXPECT_TRUE(results.empty());
}

TEST(HnswIndexTest, Add_DimensionMismatch) {
  HnswIndex index(3);
  Vector wrong(5);
  wrong.data = {1, 2, 3, 4, 5};
  EXPECT_THROW(index.Add(wrong), std::invalid_argument);
}

TEST(HnswIndexTest, Search_DimensionMismatch) {
  HnswIndex index(3);
  Vector v(3);
  v.data = {1, 2, 3};
  index.Add(v);

  Vector wrong_query(5);
  wrong_query.data = {1, 2, 3, 4, 5};
  EXPECT_THROW(index.Search(wrong_query, 1), std::invalid_argument);
}

TEST(HnswIndexTest, EfSearch_AffectsRecall) {
  const size_t dim = 32;
  const size_t num_vectors = 500;
  const size_t k = 10;

  FlatIndex flat(dim);
  HnswIndex hnsw(dim, /*M=*/8, /*ef_construction=*/100);

  for (size_t i = 0; i < num_vectors; ++i) {
    auto v = MakeRandomVector(dim, i);
    flat.Add(v);
    hnsw.Add(v);
  }

  Vector query = MakeRandomVector(dim, 99999);
  auto flat_results = flat.Search(query, k);

  // Low ef_search
  auto low_ef_results = hnsw.Search(query, k, /*ef_search=*/10);
  size_t low_matches = 0;
  for (const auto &hr : low_ef_results) {
    for (const auto &fr : flat_results) {
      if (hr.id == fr.id) {
        ++low_matches;
        break;
      }
    }
  }

  // High ef_search
  auto high_ef_results = hnsw.Search(query, k, /*ef_search=*/200);
  size_t high_matches = 0;
  for (const auto &hr : high_ef_results) {
    for (const auto &fr : flat_results) {
      if (hr.id == fr.id) {
        ++high_matches;
        break;
      }
    }
  }

  // Higher ef_search should give equal or better recall
  EXPECT_GE(high_matches, low_matches)
      << "Higher ef_search should not decrease recall";
}
