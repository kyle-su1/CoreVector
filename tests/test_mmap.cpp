#include "flat_index.hpp"
#include <cstdio>
#include <gtest/gtest.h>
#include <random>

using namespace corevector;

// Helper to generate a random query vector
Vector GenerateRandomVector(size_t dim) {
  Vector v(dim);
  std::mt19937 gen(1337);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < dim; ++i) {
    v.data[i] = dist(gen);
  }
  return v;
}

TEST(FlatIndexTest, MmapLoad) {
  const size_t dim = 128;
  FlatIndex original_index(dim);

  // Add 100 random vectors
  for (int i = 0; i < 100; ++i) {
    original_index.Add(GenerateRandomVector(dim));
  }

  // Save to file
  const std::string filename = "test_mmap_index.bin";
  original_index.Save(filename);

  // Load using MMAP
  FlatIndex mmap_index(dim);
  mmap_index.MmapLoad(filename);

  // Check metadata
  EXPECT_EQ(mmap_index.Size(), original_index.Size());
  EXPECT_EQ(mmap_index.Dim(), original_index.Dim());

  // Run a search query on both to ensure identical results
  Vector query = GenerateRandomVector(dim);
  auto results_orig = original_index.Search(query, 5);
  auto results_mmap = mmap_index.Search(query, 5);

  EXPECT_EQ(results_orig.size(), results_mmap.size());
  for (size_t i = 0; i < results_orig.size(); ++i) {
    EXPECT_EQ(results_orig[i].id, results_mmap[i].id);
    EXPECT_FLOAT_EQ(results_orig[i].distance, results_mmap[i].distance);
  }

  // Parallel search test
  auto results_orig_par = original_index.SearchParallel(query, 5);
  auto results_mmap_par = mmap_index.SearchParallel(query, 5);

  EXPECT_EQ(results_orig_par.size(), results_mmap_par.size());
  for (size_t i = 0; i < results_orig_par.size(); ++i) {
    EXPECT_EQ(results_orig_par[i].id, results_mmap_par[i].id);
    EXPECT_FLOAT_EQ(results_orig_par[i].distance, results_mmap_par[i].distance);
  }

  // Cleanup
  std::remove(filename.c_str());
}
