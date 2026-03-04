/**
 * BENCHMARK 3: Network Overhead
 *
 * Compares the latency of a local FlatIndex::Search() call vs. the same
 * query sent over the gRPC network boundary (localhost).
 *
 * Usage:
 *   1. Start the server:   ./corevector_server
 *   2. Run the benchmark:  ./benchmark_network
 *
 * The server must be running and pre-populated BEFORE the benchmark starts.
 * This benchmark will insert vectors via gRPC, then compare search latencies.
 */
#include "flat_index.hpp"
#include "vector_db.grpc.pb.h"

#include <benchmark/benchmark.h>
#include <grpcpp/grpcpp.h>
#include <random>

using namespace corevector;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static Vector GenerateRandomVector(size_t dim, unsigned seed) {
  Vector v(dim);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < dim; ++i) {
    v.data[i] = dist(gen);
  }
  return v;
}

// Shared constants
static constexpr size_t DIM = 128;
static constexpr size_t K = 10;
static constexpr size_t NUM_VECTORS = 10000;

// ---------------------------------------------------------------------------
// 1) LOCAL benchmark: FlatIndex::Search directly in-process
// ---------------------------------------------------------------------------

static void BM_SearchLocal(benchmark::State &state) {
  // Build a local index with the same data the server has
  static FlatIndex *index = nullptr;
  static Vector *query = nullptr;
  if (!index) {
    index = new FlatIndex(DIM);
    for (size_t i = 0; i < NUM_VECTORS; ++i) {
      index->Add(GenerateRandomVector(DIM, i));
    }
    query = new Vector(GenerateRandomVector(DIM, 99999));
  }

  for (auto _ : state) {
    auto results = index->Search(*query, K);
    benchmark::DoNotOptimize(results);
  }
}
BENCHMARK(BM_SearchLocal)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// 2) gRPC benchmark: same search sent over the network to corevector_server
// ---------------------------------------------------------------------------

static void BM_SearchGrpc(benchmark::State &state) {
  // Create a gRPC channel with increased max message size (64 MB)
  grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(64 * 1024 * 1024);
  args.SetMaxSendMessageSize(64 * 1024 * 1024);
  static auto channel = grpc::CreateCustomChannel(
      "localhost:50051", grpc::InsecureChannelCredentials(), args);
  static auto stub = api::VectorDB::NewStub(channel);

  // Insert vectors into the remote server in batches (only once)
  static bool populated = false;
  if (!populated) {
    const size_t BATCH_SIZE = 500;
    for (size_t batch_start = 0; batch_start < NUM_VECTORS;
         batch_start += BATCH_SIZE) {
      api::InsertRequest insert_req;
      size_t batch_end = std::min(batch_start + BATCH_SIZE, NUM_VECTORS);
      for (size_t i = batch_start; i < batch_end; ++i) {
        auto *vec_data = insert_req.add_vectors();
        Vector v = GenerateRandomVector(DIM, i);
        for (size_t j = 0; j < DIM; ++j) {
          vec_data->add_values(v.data[j]);
        }
      }
      grpc::ClientContext ctx;
      api::InsertResponse insert_resp;
      auto status = stub->Insert(&ctx, insert_req, &insert_resp);
      if (!status.ok()) {
        state.SkipWithError("Failed to populate server: " +
                            status.error_message());
        return;
      }
    }
    populated = true;
  }

  // Build the query request
  Vector query_vec = GenerateRandomVector(DIM, 99999);
  api::SearchRequest search_req;
  for (size_t j = 0; j < DIM; ++j) {
    search_req.add_query(query_vec.data[j]);
  }
  search_req.set_k(K);

  // Benchmark the gRPC round-trip latency
  for (auto _ : state) {
    grpc::ClientContext ctx;
    api::SearchResponse search_resp;
    auto status = stub->Search(&ctx, search_req, &search_resp);
    benchmark::DoNotOptimize(search_resp);
    if (!status.ok()) {
      state.SkipWithError("gRPC search failed: " + status.error_message());
      return;
    }
  }
}
BENCHMARK(BM_SearchGrpc)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
