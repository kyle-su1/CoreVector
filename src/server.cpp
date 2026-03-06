#include "flat_index.hpp"
#include "types.hpp"
#include "vector_db.grpc.pb.h"
#include "vector_db.pb.h"

#include <grpcpp/grpcpp.h>
#include <iostream>
#include <memory>
#include <string>

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;

namespace {

class VectorDBServiceImpl final : public corevector::api::VectorDB::Service {
public:
  explicit VectorDBServiceImpl(size_t dim) : index_(dim), dim_(dim) {}

  // Insert one or more vectors into the index
  Status Insert(ServerContext *context,
                const corevector::api::InsertRequest *request,
                corevector::api::InsertResponse *response) override {
    for (const auto &vec_data : request->vectors()) {
      if (static_cast<size_t>(vec_data.values_size()) != dim_) {
        return Status(grpc::INVALID_ARGUMENT,
                      "Vector dimension mismatch. Expected " +
                          std::to_string(dim_) + ", got " +
                          std::to_string(vec_data.values_size()));
      }

      ::corevector::Vector v(dim_);
      for (size_t i = 0; i < dim_; ++i) {
        v.data[i] = vec_data.values(i);
      }
      index_.Add(v, vec_data.payload());
    }

    response->set_total_vectors(index_.Size());
    return Status::OK;
  }

  // Search for the k nearest neighbors
  Status Search(ServerContext *context,
                const corevector::api::SearchRequest *request,
                corevector::api::SearchResponse *response) override {
    if (static_cast<size_t>(request->query_size()) != dim_) {
      return Status(grpc::INVALID_ARGUMENT,
                    "Query dimension mismatch. Expected " +
                        std::to_string(dim_) + ", got " +
                        std::to_string(request->query_size()));
    }

    ::corevector::Vector query(dim_);
    for (size_t i = 0; i < dim_; ++i) {
      query.data[i] = request->query(i);
    }

    auto results = index_.Search(query, request->k());

    for (const auto &result : results) {
      auto *sr = response->add_results();
      sr->set_id(result.id);
      sr->set_distance(result.distance);
      sr->set_payload(result.payload);
    }

    return Status::OK;
  }

  // Save the index to disk
  Status Save(ServerContext *context,
              const corevector::api::SaveRequest *request,
              corevector::api::SaveResponse *response) override {
    try {
      index_.Save(request->filename());
      response->set_success(true);
      response->set_message("Index saved to " + request->filename());
    } catch (const std::exception &e) {
      response->set_success(false);
      response->set_message(e.what());
    }
    return Status::OK;
  }

  // Load the index from disk
  Status Load(ServerContext *context,
              const corevector::api::LoadRequest *request,
              corevector::api::LoadResponse *response) override {
    try {
      index_.Load(request->filename());
      response->set_success(true);
      response->set_message("Index loaded from " + request->filename());
      response->set_total_vectors(index_.Size());
    } catch (const std::exception &e) {
      response->set_success(false);
      response->set_message(e.what());
    }
    return Status::OK;
  }

private:
  ::corevector::FlatIndex index_;
  size_t dim_;
};

} // namespace

int main(int argc, char *argv[]) {
  // Default dimension: 128, configurable via command line
  size_t dim = 128;
  if (argc > 1) {
    dim = std::stoul(argv[1]);
  }

  std::string server_address("0.0.0.0:50051");
  VectorDBServiceImpl service(dim);

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "CoreVector gRPC server listening on " << server_address
            << " (dim=" << dim << ")" << std::endl;
  server->Wait();

  return 0;
}
