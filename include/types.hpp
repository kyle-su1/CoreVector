#pragma once

#include <string>

namespace corevector {

// Result of a nearest neighbor search
struct SearchResult {
  size_t id;
  float distance;
  std::string payload;

  bool operator<(const SearchResult &other) const {
    return distance < other.distance; // Max-heap: largest distance on top
  }
};

} // namespace corevector
