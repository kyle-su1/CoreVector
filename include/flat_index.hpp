#pragma once

#include "types.hpp"
#include "vector.hpp"
#include <algorithm>
#include <fcntl.h>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace corevector {

class FlatIndex {
public:
  explicit FlatIndex(size_t dim) : dim_(dim) {}

  // Add a vector to the index with an optional text payload
  void Add(const Vector &vec, const std::string &payload = "") {
    if (vec.dim() != dim_) {
      throw std::invalid_argument(
          "Vector dimension does not match index dimension");
    }
    vectors_.push_back(vec);
    payloads_.push_back(payload);
  }

  // Add multiple vectors to the index (no payloads)
  void Add(const std::vector<Vector> &vecs) {
    for (const auto &vec : vecs) {
      Add(vec);
    }
  }

  // Search top-k nearest neighbors using single-threaded search (Brute-Force)
  std::vector<SearchResult> Search(const Vector &query, size_t k) const {
    if (query.dim() != dim_) {
      throw std::invalid_argument(
          "Query dimension does not match index dimension");
    }
    if (Size() == 0 || k == 0)
      return {};

    std::priority_queue<SearchResult>
        pq; // Max-heap to keep track of the k nearest

    for (size_t i = 0; i < Size(); ++i) {
      float dist;
      if (use_mmap_) {
        const float *v_data = mmap_data_ + (i * dim_);
        dist = 0.0f;
        for (size_t j = 0; j < dim_; ++j) {
          float d = query.data[j] - v_data[j];
          dist += d * d;
        }
      } else {
        dist = math::naive::L2Sqr(
            query, vectors_[i]); // Use squared distance for speed
      }

      if (pq.size() < k) {
        pq.push({i, dist, ""});
      } else if (dist < pq.top().distance) {
        pq.pop();
        pq.push({i, dist, ""});
      }
    }

    return ExtractResults(pq);
  }

  // Search top-k nearest neighbors using multi-threaded search
  // (std::thread chunking)
  std::vector<SearchResult> SearchParallel(const Vector &query,
                                           size_t k) const {
    if (query.dim() != dim_) {
      throw std::invalid_argument(
          "Query dimension does not match index dimension");
    }
    if (Size() == 0 || k == 0)
      return {};

    std::vector<float> distances(Size());

    // Manual thread chunking
    const size_t num_threads = std::thread::hardware_concurrency();
    const size_t chunk_size = Size() / num_threads;
    std::vector<std::thread> threads;

    for (size_t t = 0; t < num_threads; ++t) {
      size_t start = t * chunk_size;
      size_t end = (t == num_threads - 1) ? Size() : (t + 1) * chunk_size;

      threads.emplace_back([this, &query, &distances, start, end]() {
        for (size_t i = start; i < end; ++i) {
          if (use_mmap_) {
            const float *v_data = mmap_data_ + (i * dim_);
            float dist = 0.0f;
            for (size_t j = 0; j < dim_; ++j) {
              float d = query.data[j] - v_data[j];
              dist += d * d;
            }
            distances[i] = dist;
          } else {
            distances[i] = math::naive::L2Sqr(query, vectors_[i]);
          }
        }
      });
    }

    for (auto &thread : threads) {
      thread.join();
    }

    // We use an index array to track the original vector IDs.
    std::vector<size_t> indices(distances.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort indices array based on distances, taking only the top k
    k = std::min(k, Size());
    std::partial_sort(
        indices.begin(), indices.begin() + k, indices.end(),
        [&](size_t i, size_t j) { return distances[i] < distances[j]; });

    std::vector<SearchResult> results(k);
    for (size_t i = 0; i < k; ++i) {
      size_t id = indices[i];
      std::string pl =
          (!use_mmap_ && id < payloads_.size()) ? payloads_[id] : "";
      results[i] = {id, distances[id], std::move(pl)};
    }

    return results;
  }

  // Save the index to a binary file
  void Save(const std::string &filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
      throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Write metadata (dim, num_vectors)
    size_t num_vectors = vectors_.size();
    out.write(reinterpret_cast<const char *>(&dim_), sizeof(size_t));
    out.write(reinterpret_cast<const char *>(&num_vectors), sizeof(size_t));

    // Write contiguous vector data
    for (const auto &vec : vectors_) {
      out.write(reinterpret_cast<const char *>(vec.data.data()),
                dim_ * sizeof(float));
    }

    // Write payloads: for each, write string length then string bytes
    for (const auto &payload : payloads_) {
      size_t len = payload.size();
      out.write(reinterpret_cast<const char *>(&len), sizeof(size_t));
      out.write(payload.data(), len);
    }

    if (!out) {
      throw std::runtime_error("Failed to write data to file: " + filename);
    }
  }

  // Load the index from a binary file
  void Load(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
      throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    // Read metadata
    size_t file_dim = 0;
    size_t num_vectors = 0;
    if (!in.read(reinterpret_cast<char *>(&file_dim), sizeof(size_t))) {
      throw std::runtime_error("Failed to read dimension from file: " +
                               filename);
    }
    if (!in.read(reinterpret_cast<char *>(&num_vectors), sizeof(size_t))) {
      throw std::runtime_error("Failed to read vector count from file: " +
                               filename);
    }

    if (file_dim != dim_ && vectors_.empty()) {
      dim_ = file_dim; // Auto-configure dimension if empty
    } else if (file_dim != dim_) {
      throw std::runtime_error("File dimension does not match index dimension");
    }

    vectors_.resize(num_vectors, Vector(dim_));

    // Read contiguous vector data
    for (size_t i = 0; i < num_vectors; ++i) {
      if (!in.read(reinterpret_cast<char *>(vectors_[i].data.data()),
                   dim_ * sizeof(float))) {
        throw std::runtime_error("Failed to read vector data from file: " +
                                 filename);
      }
    }

    // Read payloads (if present in the file)
    payloads_.resize(num_vectors);
    for (size_t i = 0; i < num_vectors; ++i) {
      size_t len = 0;
      if (!in.read(reinterpret_cast<char *>(&len), sizeof(size_t))) {
        // No payloads in file (legacy format) — leave empty strings
        break;
      }
      payloads_[i].resize(len);
      if (len > 0) {
        in.read(&payloads_[i][0], len);
      }
    }
  }

  // Memory Map the index directly from disk (Zero-Copy)
  void MmapLoad(const std::string &filename) {
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
      throw std::runtime_error("Failed to open file for mmap: " + filename);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
      close(fd);
      throw std::runtime_error("Failed to stat file: " + filename);
    }
    mmap_size_ = sb.st_size;

    mmap_ptr_ = ::mmap(nullptr, mmap_size_, PROT_READ, MAP_SHARED, fd, 0);
    close(fd); // Can close fd after mmap

    if (mmap_ptr_ == MAP_FAILED) {
      mmap_ptr_ = nullptr;
      mmap_size_ = 0;
      throw std::runtime_error("Failed to mmap file: " + filename);
    }

    const char *ptr = static_cast<const char *>(mmap_ptr_);

    // Read metadata
    size_t file_dim = *reinterpret_cast<const size_t *>(ptr);
    ptr += sizeof(size_t);
    size_t num_vectors_disk = *reinterpret_cast<const size_t *>(ptr);
    ptr += sizeof(size_t);

    if (file_dim != dim_ && num_vectors_ == 0 && vectors_.empty()) {
      dim_ = file_dim;
    } else if (file_dim != dim_) {
      throw std::runtime_error(
          "Mmap file dimension does not match index dimension");
    }

    num_vectors_ = num_vectors_disk;
    mmap_data_ = reinterpret_cast<const float *>(ptr);
    use_mmap_ = true;

    // Clear any RAM vectors since we are mapping
    vectors_.clear();
  }

  ~FlatIndex() {
    if (use_mmap_ && mmap_ptr_ != nullptr) {
      ::munmap(mmap_ptr_, mmap_size_);
    }
  }

  size_t Size() const { return use_mmap_ ? num_vectors_ : vectors_.size(); }
  size_t Dim() const { return dim_; }

private:
  size_t dim_;
  std::vector<Vector> vectors_;
  std::vector<std::string> payloads_; // Parallel array of text metadata

  // mmap properties
  bool use_mmap_ = false;
  size_t num_vectors_ = 0;
  void *mmap_ptr_ = nullptr;
  size_t mmap_size_ = 0;
  const float *mmap_data_ = nullptr;

  std::vector<SearchResult>
  ExtractResults(std::priority_queue<SearchResult> &pq) const {
    std::vector<SearchResult> results(pq.size());
    // Priority queue pops the LARGEST element first, so we fill the array
    // backwards
    for (int i = static_cast<int>(pq.size()) - 1; i >= 0; --i) {
      auto r = pq.top();
      pq.pop();
      // Attach payload if available
      if (!use_mmap_ && r.id < payloads_.size()) {
        r.payload = payloads_[r.id];
      }
      results[i] = std::move(r);
    }
    return results;
  }
};

} // namespace corevector
