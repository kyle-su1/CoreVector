#pragma once

#include "vector.hpp"
#include <algorithm>
#include <cmath>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace corevector {

// Forward declaration — SearchResult is defined in flat_index.hpp
// We redefine it here so hnsw_index.hpp is self-contained
#ifndef COREVECTOR_SEARCH_RESULT_DEFINED
#define COREVECTOR_SEARCH_RESULT_DEFINED
struct SearchResult {
  size_t id;
  float distance;

  bool operator<(const SearchResult &other) const {
    return distance < other.distance; // Max-heap: largest distance on top
  }
};
#endif

class HnswIndex {
public:
  /**
   * @param dim         Vector dimensionality
   * @param M           Max connections per node per layer (default: 16)
   * @param ef_construction  Beam width during insert (default: 200)
   */
  HnswIndex(size_t dim, size_t M = 16, size_t ef_construction = 200)
      : dim_(dim), M_(M), M_max0_(M * 2), ef_construction_(ef_construction),
        m_L_(1.0 / std::log(static_cast<double>(M))), max_level_(0),
        entry_point_(0), level_gen_(42), level_dist_(0.0, 1.0) {}

  // =========================================================================
  // Public API
  // =========================================================================

  /**
   * Insert a vector into the HNSW graph.
   */
  void Add(const Vector &vec) {
    if (vec.dim() != dim_) {
      throw std::invalid_argument(
          "Vector dimension does not match index dimension");
    }

    size_t new_id = nodes_.size();
    size_t new_level = RandomLevel();

    // Create the new node
    HnswNode node;
    node.vector = vec;
    node.max_layer = new_level;
    node.neighbors.resize(new_level + 1);
    nodes_.push_back(std::move(node));

    // First node: just set it as entry point
    if (new_id == 0) {
      entry_point_ = 0;
      max_level_ = new_level;
      return;
    }

    size_t curr_node = entry_point_;

    // Phase 1: Greedily traverse layers above the new node's max layer
    // (just finding the closest entry point, no connections made)
    for (int layer = static_cast<int>(max_level_);
         layer > static_cast<int>(new_level); --layer) {
      curr_node = GreedyClosest(vec, curr_node, layer);
    }

    // Phase 2: For each layer from new_level down to 0,
    // find neighbors and create bidirectional connections
    for (int layer = static_cast<int>(std::min(new_level, max_level_));
         layer >= 0; --layer) {
      // Search this layer for the ef_construction nearest candidates
      auto candidates = SearchLayer(vec, curr_node, ef_construction_, layer);

      // Select the best M neighbors from the candidates
      size_t M_layer = (layer == 0) ? M_max0_ : M_;
      auto neighbors = SelectNeighbors(vec, candidates, M_layer);

      // Connect new_id → neighbors (forward links)
      nodes_[new_id].neighbors[layer] = neighbors;

      // Connect neighbors → new_id (backward links) + prune if needed
      for (size_t neighbor_id : neighbors) {
        nodes_[neighbor_id].neighbors[layer].push_back(new_id);

        // Prune if this neighbor now has too many connections
        if (nodes_[neighbor_id].neighbors[layer].size() > M_layer) {
          PruneConnections(neighbor_id, layer, M_layer);
        }
      }

      // Use the closest candidate as entry point for the next layer down
      if (!candidates.empty()) {
        curr_node = candidates.front().id;
      }
    }

    // Update the global entry point if the new node reaches a higher level
    if (new_level > max_level_) {
      max_level_ = new_level;
      entry_point_ = new_id;
    }
  }

  /**
   * Search for the k approximate nearest neighbors.
   * @param ef_search  Beam width (higher = better recall, slower). Default: 50.
   */
  std::vector<SearchResult> Search(const Vector &query, size_t k,
                                   size_t ef_search = 50) const {
    if (query.dim() != dim_) {
      throw std::invalid_argument(
          "Query dimension does not match index dimension");
    }
    if (nodes_.empty() || k == 0) {
      return {};
    }

    // Ensure ef_search >= k
    ef_search = std::max(ef_search, k);

    size_t curr_node = entry_point_;

    // Phase 1: Greedily traverse from top layer down to layer 1
    for (int layer = static_cast<int>(max_level_); layer > 0; --layer) {
      curr_node = GreedyClosest(query, curr_node, layer);
    }

    // Phase 2: Thorough beam search at layer 0
    auto candidates = SearchLayer(query, curr_node, ef_search, 0);

    // Return top-k from candidates (already sorted by distance)
    if (candidates.size() > k) {
      candidates.resize(k);
    }
    return candidates;
  }

  size_t Size() const { return nodes_.size(); }
  size_t Dim() const { return dim_; }

private:
  // =========================================================================
  // Internal Data Structures
  // =========================================================================

  struct HnswNode {
    Vector vector;
    std::vector<std::vector<size_t>>
        neighbors; // neighbors[layer] = {id, id, ...}
    size_t max_layer = 0;

    HnswNode() : vector(0) {} // Default constructor
  };

  // =========================================================================
  // Core Graph Operations
  // =========================================================================

  /**
   * Greedy search within a single layer to find the single closest node.
   * Used during the upper-layer traversal phase.
   */
  size_t GreedyClosest(const Vector &query, size_t entry, int layer) const {
    float best_dist = math::naive::L2Sqr(query, nodes_[entry].vector);
    size_t best_id = entry;
    bool improved = true;

    while (improved) {
      improved = false;
      if (layer >= static_cast<int>(nodes_[best_id].neighbors.size())) {
        break;
      }
      for (size_t neighbor_id : nodes_[best_id].neighbors[layer]) {
        float dist = math::naive::L2Sqr(query, nodes_[neighbor_id].vector);
        if (dist < best_dist) {
          best_dist = dist;
          best_id = neighbor_id;
          improved = true;
        }
      }
    }
    return best_id;
  }

  /**
   * Beam search within a single layer. Returns up to `ef` nearest candidates,
   * sorted by distance (ascending).
   */
  std::vector<SearchResult> SearchLayer(const Vector &query, size_t entry,
                                        size_t ef, int layer) const {
    // Min-heap: candidates to explore (closest first)
    auto cmp_min = [](const SearchResult &a, const SearchResult &b) {
      return a.distance > b.distance;
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>,
                        decltype(cmp_min)>
        candidates(cmp_min);

    // Max-heap: best results found so far (farthest on top for easy eviction)
    std::priority_queue<SearchResult> results;

    std::unordered_set<size_t> visited;

    float entry_dist = math::naive::L2Sqr(query, nodes_[entry].vector);
    candidates.push({entry, entry_dist});
    results.push({entry, entry_dist});
    visited.insert(entry);

    while (!candidates.empty()) {
      auto current = candidates.top();
      candidates.pop();

      // If the closest candidate is farther than the farthest result,
      // we can't improve → stop
      if (current.distance > results.top().distance) {
        break;
      }

      // Explore neighbors of the current node
      if (layer < static_cast<int>(nodes_[current.id].neighbors.size())) {
        for (size_t neighbor_id : nodes_[current.id].neighbors[layer]) {
          if (visited.count(neighbor_id)) {
            continue;
          }
          visited.insert(neighbor_id);

          float dist = math::naive::L2Sqr(query, nodes_[neighbor_id].vector);

          if (results.size() < ef || dist < results.top().distance) {
            candidates.push({neighbor_id, dist});
            results.push({neighbor_id, dist});

            if (results.size() > ef) {
              results.pop(); // Evict the farthest
            }
          }
        }
      }
    }

    // Extract results sorted by distance
    std::vector<SearchResult> sorted_results;
    sorted_results.reserve(results.size());
    while (!results.empty()) {
      sorted_results.push_back(results.top());
      results.pop();
    }
    // Max-heap pops largest first → reverse for ascending order
    std::reverse(sorted_results.begin(), sorted_results.end());
    return sorted_results;
  }

  /**
   * Select the best M neighbors from a candidate list.
   * Uses the simple heuristic: pick the M closest.
   */
  std::vector<size_t>
  SelectNeighbors(const Vector &query,
                  const std::vector<SearchResult> &candidates,
                  size_t M_target) const {
    std::vector<size_t> result;
    result.reserve(M_target);

    // Candidates are already sorted by distance (ascending)
    for (const auto &c : candidates) {
      if (result.size() >= M_target)
        break;
      result.push_back(c.id);
    }
    return result;
  }

  /**
   * Prune a node's connections down to M_target by keeping only the closest.
   */
  void PruneConnections(size_t node_id, int layer, size_t M_target) {
    auto &neighbors = nodes_[node_id].neighbors[layer];
    const auto &node_vec = nodes_[node_id].vector;

    // Compute distances to all current neighbors
    std::vector<SearchResult> scored;
    scored.reserve(neighbors.size());
    for (size_t nid : neighbors) {
      float dist = math::naive::L2Sqr(node_vec, nodes_[nid].vector);
      scored.push_back({nid, dist});
    }

    // Sort by distance and keep only the closest M_target
    std::sort(scored.begin(), scored.end(),
              [](const SearchResult &a, const SearchResult &b) {
                return a.distance < b.distance;
              });

    neighbors.clear();
    for (size_t i = 0; i < std::min(M_target, scored.size()); ++i) {
      neighbors.push_back(scored[i].id);
    }
  }

  /**
   * Randomly assign a layer level for a new node.
   * Uses exponential distribution: most nodes → layer 0, few → higher layers.
   */
  size_t RandomLevel() {
    double r = level_dist_(level_gen_);
    // Clamp to avoid log(0)
    if (r == 0.0)
      r = 1e-9;
    size_t level = static_cast<size_t>(-std::log(r) * m_L_);
    return level;
  }

  // =========================================================================
  // Member Variables
  // =========================================================================

  size_t dim_;             // Vector dimensionality
  size_t M_;               // Max connections per node per layer
  size_t M_max0_;          // Max connections per node in layer 0 (2 * M)
  size_t ef_construction_; // Beam width during insertion
  double m_L_;             // Level generation multiplier (1 / ln(M))

  std::vector<HnswNode> nodes_; // All nodes in the graph
  size_t entry_point_;          // ID of the entry point node
  size_t max_level_;            // Current max layer in the graph

  // Random number generator for level assignment
  std::mt19937 level_gen_;
  std::uniform_real_distribution<double> level_dist_;
};

} // namespace corevector
