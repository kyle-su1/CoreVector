#include "flat_index.hpp"
#include <cstdio>
#include <gtest/gtest.h>
#include <random>

using namespace corevector;

// Helper to generate a random query vector
Vector GenerateRandomVector(size_t dim) {
  Vector v(dim);
  static std::mt19937 gen(1337);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  for (size_t i = 0; i < dim; ++i) {
    v.data[i] = dist(gen);
  }
  return v;
}

TEST(FlatIndexTest, SaveAndLoad) {
  const size_t dim = 128;
  FlatIndex original_index(dim);

  // Add 100 random vectors
  for (int i = 0; i < 100; ++i) {
    original_index.Add(GenerateRandomVector(dim));
  }

  // Save to file
  const std::string filename = "test_index.bin";
  original_index.Save(filename);

  // Load to a new index
  FlatIndex loaded_index(dim);
  loaded_index.Load(filename);

  // Check metadata
  EXPECT_EQ(loaded_index.Size(), original_index.Size());
  EXPECT_EQ(loaded_index.Dim(), original_index.Dim());

  // Run a search query on both to ensure identical results
  Vector query = GenerateRandomVector(dim);
  auto results_orig = original_index.Search(query, 5);
  auto results_loaded = loaded_index.Search(query, 5);

  EXPECT_EQ(results_orig.size(), results_loaded.size());
  for (size_t i = 0; i < results_orig.size(); ++i) {
    EXPECT_EQ(results_orig[i].id, results_loaded[i].id);
    EXPECT_FLOAT_EQ(results_orig[i].distance, results_loaded[i].distance);
  }

  // Cleanup
  std::remove(filename.c_str());
}

TEST(FlatIndexTest, SaveAndLoad_WithPayloads) {
  const size_t dim = 16;
  FlatIndex original_index(dim);

  // Add vectors with payloads
  std::vector<std::string> expected_payloads = {
      "The quick brown fox",     "Apple MacBook Pro", "Neural network training",
      "Quantum computing intro", "Memory-mapped I/O",
  };

  for (size_t i = 0; i < expected_payloads.size(); ++i) {
    original_index.Add(GenerateRandomVector(dim), expected_payloads[i]);
  }

  // Save to file
  const std::string filename = "test_payload_index.bin";
  original_index.Save(filename);

  // Load into a new index
  FlatIndex loaded_index(dim);
  loaded_index.Load(filename);

  EXPECT_EQ(loaded_index.Size(), original_index.Size());

  // Search and verify payloads survive the round trip
  Vector query = GenerateRandomVector(dim);
  auto results_orig = original_index.Search(query, 5);
  auto results_loaded = loaded_index.Search(query, 5);

  ASSERT_EQ(results_orig.size(), results_loaded.size());
  for (size_t i = 0; i < results_orig.size(); ++i) {
    EXPECT_EQ(results_orig[i].id, results_loaded[i].id);
    EXPECT_FLOAT_EQ(results_orig[i].distance, results_loaded[i].distance);
    EXPECT_EQ(results_orig[i].payload, results_loaded[i].payload)
        << "Payload mismatch at result index " << i;
  }

  // Cleanup
  std::remove(filename.c_str());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
