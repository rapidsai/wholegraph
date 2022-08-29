/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "graph_builder.h"

#include <unistd.h>

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "file_utils.h"
#include "macros.h"

#include "whole_graph_mixed_graph.cuh"

namespace whole_graph {

namespace {

class GraphBuilderThreadPool {
 public:
  GraphBuilderThreadPool();
  size_t Size() const {
    return pool_.size();
  }
  void Start();
  void RunAsync(std::function<void(int, int)> fn);
  void WaitDone();
  void Run(std::function<void(int, int)> fn);
  void Stop();
  ~GraphBuilderThreadPool();

 private:
  void ThreadFunction(int rank, int size);
  std::vector<std::unique_ptr<std::thread>> pool_;
  volatile enum {
    ST_INITING_ = 0,
    ST_WAITING_TASK_ = 1,
    ST_RUNNING_TASK_ = 2,
    ST_STOPPING_ = 3,
    ST_STOPPED_ = 4,
  } status_ = ST_INITING_;
  std::mutex mu_;
  pthread_barrier_t barrier_{};
  std::function<void(int, int)> fn_;
};

void GraphBuilderThreadPool::ThreadFunction(int rank, int size) {
  WM_CHECK((size_t) size == pool_.size());
  while (true) {
    pthread_barrier_wait(&barrier_);
    //WM_CHECK(status_ == ST_RUNNING_TASK_ || status_ == ST_STOPPING_);
    if (status_ == ST_STOPPING_) break;
    fn_(rank, size);
    pthread_barrier_wait(&barrier_);
  }
  pthread_barrier_wait(&barrier_);
}
GraphBuilderThreadPool::GraphBuilderThreadPool() {
  int cpu_count = (int) sysconf(_SC_NPROCESSORS_ONLN);
  int pool_size = cpu_count / 2;
  if (pool_size < 1) pool_size = 1;
  pool_.resize(pool_size);
  pthread_barrier_init(&barrier_, nullptr, pool_size + 1);
}
void GraphBuilderThreadPool::Start() {
  mu_.lock();
  WM_CHECK(status_ == ST_INITING_);
  status_ = ST_WAITING_TASK_;
  for (int i = 0; i < (int) pool_.size(); i++) {
    pool_[i] = std::make_unique<std::thread>([i, this]() {
      this->ThreadFunction(i, (int) pool_.size());
    });
  }
  mu_.unlock();
}
void GraphBuilderThreadPool::RunAsync(std::function<void(int, int)> fn) {
  mu_.lock();
  WM_CHECK(status_ == ST_WAITING_TASK_);
  status_ = ST_RUNNING_TASK_;
  fn_ = std::move(fn);
  pthread_barrier_wait(&barrier_);
  mu_.unlock();
}
void GraphBuilderThreadPool::WaitDone() {
  mu_.lock();
  WM_CHECK(status_ == ST_RUNNING_TASK_);
  status_ = ST_WAITING_TASK_;
  pthread_barrier_wait(&barrier_);
  mu_.unlock();
}
void GraphBuilderThreadPool::Run(std::function<void(int, int)> fn) {
  RunAsync(std::move(fn));
  WaitDone();
}
void GraphBuilderThreadPool::Stop() {
  mu_.lock();
  WM_CHECK(status_ == ST_WAITING_TASK_);
  status_ = ST_STOPPING_;
  pthread_barrier_wait(&barrier_);
  pthread_barrier_wait(&barrier_);
  status_ = ST_STOPPED_;
  mu_.unlock();
}
GraphBuilderThreadPool::~GraphBuilderThreadPool() {
  if (status_ == ST_RUNNING_TASK_) {
    WaitDone();
  }
  if (status_ == ST_WAITING_TASK_) {
    Stop();
  }
  WM_CHECK(status_ == ST_STOPPED_ || status_ == ST_INITING_);
  if (status_ == ST_STOPPED_) {
    for (auto &t : pool_) {
      t->join();
      t.reset();
    }
  }
  pool_.clear();
  pthread_barrier_destroy(&barrier_);
}

}// namespace

struct EdgeType {
  EdgeType() = default;
  explicit EdgeType(const std::vector<std::string> &edge) {
    WM_CHECK(edge.size() == 3);
    src = edge[0];
    dst = edge[2];
    rel = edge[1];
  }
  std::string src;
  std::string dst;
  std::string rel;
};

struct EdgeTypeComparator {
  bool operator()(const EdgeType &lhs, const EdgeType &rhs) const {
    if (lhs.src < rhs.src) return true;
    if (lhs.src > rhs.src) return false;

    if (lhs.dst < rhs.dst) return true;
    if (lhs.dst > rhs.dst) return false;

    if (lhs.rel < rhs.rel) return true;
    if (lhs.rel > rhs.rel) return false;

    return false;
  }
};

struct EdgeData {
  EdgeData() {
    edge_buffer = nullptr;
    feature_buffer = nullptr;
    count = 0;
  }
  uint8_t *edge_buffer;
  uint8_t *feature_buffer;
  int64_t count;
};

class GraphBuilder {
 public:
  GraphBuilder(const std::vector<std::string> &node_type_names,
               const std::vector<std::vector<std::string>> &relations,
               WMType id_type);
  ~GraphBuilder();
  void LoadEdgeDataFromFileList(const std::vector<std::string> &edge_desc,
                                const std::string &file_prefix,
                                bool reverse,
                                WMType file_id_type,
                                size_t edge_feature_size);
  int GetNodeIdx(const std::string &node_name) const {
    auto it = node_type_name_to_idx_.find(node_name);
    WM_CHECK(it != node_type_name_to_idx_.end());
    return it->second;
  }
  void AsHomoGraph() {
    WM_CHECK(graph_type == GT_None);
    WM_CHECK(node_type_names_.size() == 1);
    WM_CHECK(edge_types_.size() == 1);
    graph_type = GT_Homo;
  }
  void AsHeterGraph() {
    WM_CHECK(graph_type == GT_None);
    WM_CHECK(node_type_names_.size() == 2);
    WM_CHECK(edge_types_.size() == 1);
    graph_type = GT_Heter;
  }
  void AsMixedHomoGraph() {
    WM_CHECK(graph_type == GT_None);
    graph_type = GT_Mixed;
  }
  void SetNodeCounts(const std::vector<int64_t> &node_counts) {
    WM_CHECK(node_counts.size() == node_counts_.size());
    for (size_t i = 0; i < node_counts.size(); i++) {
      if (node_counts[i] > 0) {
        node_counts_[i] = node_counts[i];
      }
    }
  }
  void SetShuffleID(bool shuffle_id) {
    shuffle_id_ = shuffle_id;
  }
  void SetEdgeConfig(const std::vector<std::string> &relation,
                     bool as_undirected,
                     bool add_self_loop,
                     bool build_both_direction) {
    int eidx = 0;
    if (!relation.empty()) {
      WM_CHECK(relation.size() == 3);
      EdgeType find_edge_type(relation);
      auto it = edge_type_to_idx_.find(find_edge_type);
      WM_CHECK(it != edge_type_to_idx_.end());
      eidx = it->second;
    }
    EdgeType edge_type = edge_types_[eidx];
    if (as_undirected || add_self_loop) {
      WM_CHECK(edge_type.src == edge_type.dst);
      WM_CHECK(graph_type == GT_Homo || graph_type == GT_Mixed);
    }
    edge_configs_[eidx].as_undirected = as_undirected;
    edge_configs_[eidx].add_self_loop = add_self_loop;
    if (build_both_direction) {
      WM_CHECK(graph_type == GT_Heter || graph_type == GT_Mixed);
    }
    edge_configs_[eidx].build_both_direction = build_both_direction;
  }
  void SetGraphSaveFile(const std::string &csr_row_ptr_filename,
                        const std::string &csr_col_idx_filename,
                        const std::string &id_mapping_prefix) {
    csr_row_ptr_filename_ = csr_row_ptr_filename;
    csr_col_idx_filename_ = csr_col_idx_filename;
    id_mapping_prefix_ = id_mapping_prefix;
  }
  void Build();

 private:
  void CreateEdgeData(int idx, int64_t edge_count, size_t feature_size) {
    edge_data_[idx].edge_buffer = (uint8_t *) malloc(GetWMTSize(id_type_) * 2 * edge_count);
    if (feature_size > 0) {
      edge_data_[idx].feature_buffer = (uint8_t *) malloc(feature_size * edge_count);
    }
    edge_data_[idx].count = edge_count;
  }
  void FillNodeCounts();
  void GenerateMixedNodeConvertTable(bool shuffle);
  void SaveMapping(const std::vector<int64_t> &save_vector, const std::string &name, bool force_int64 = false);

  void BuildHomo();
  void BuildMixed();

  enum GraphType {
    GT_None = 0,
    GT_Homo = 1,
    GT_Heter = 2,
    GT_Mixed = 3,
  } graph_type = GT_None;

  struct EdgeConfig {
    EdgeConfig() = default;
    bool as_undirected = false;      // applies only for same node type
    bool add_self_loop = false;      // applies only for same node type
    bool build_both_direction = true;// applies for Mixed different node type
  };

  bool shuffle_id_ = true;

  WMType id_type_;
  std::vector<std::string> node_type_names_;
  std::vector<EdgeType> edge_types_;
  std::map<std::string, int> node_type_name_to_idx_;
  std::map<EdgeType, int, EdgeTypeComparator> edge_type_to_idx_;
  std::vector<EdgeData> edge_data_;

  std::vector<std::vector<int64_t>> to_mixed_id;
  std::vector<TypedNodeID> to_typed_id;
  std::unique_ptr<GraphBuilderThreadPool> pool_;

  std::vector<int64_t> node_counts_;
  std::vector<EdgeConfig> edge_configs_;

  std::string csr_row_ptr_filename_;
  std::string csr_col_idx_filename_;
  std::string id_mapping_prefix_;

  template<typename IdType>
  friend void GraphBuilderBuildMixed(GraphBuilder *graph_builder);
};

GraphBuilder *CreateMixedGraphBuilder(const std::vector<std::string> &node_type_names,
                                      const std::vector<std::vector<std::string>> &relations,
                                      WMType id_type) {
  GraphBuilder *graph_builder = new GraphBuilder(node_type_names, relations, id_type);
  graph_builder->AsMixedHomoGraph();
  return graph_builder;
}

GraphBuilder *CreateHomoGraphBuilder(WMType id_type) {
  GraphBuilder *graph_builder = new GraphBuilder({"n"}, {{"n", "r", "n"}}, id_type);
  graph_builder->AsHomoGraph();
  return graph_builder;
}

void DestroyGraphBuilder(GraphBuilder *graph_builder) {
  delete graph_builder;
}

void GraphBuilderSetNodeCounts(GraphBuilder *graph_builder, const std::vector<int64_t> &node_counts) {
  graph_builder->SetNodeCounts(node_counts);
}

void GraphBuilderLoadEdgeDataFromFileList(GraphBuilder *graph_builder,
                                          const std::vector<std::string> &relations,
                                          const std::string &file_prefix,
                                          bool reverse,
                                          WMType file_id_type,
                                          size_t edge_feature_size) {
  graph_builder->LoadEdgeDataFromFileList(relations,
                                          file_prefix,
                                          reverse,
                                          file_id_type,
                                          edge_feature_size);
}

void GraphBuilderSetEdgeConfig(GraphBuilder *graph_builder,
                               const std::vector<std::string> &relation,
                               bool as_undirected,
                               bool add_self_loop,
                               bool build_both_direction) {
  graph_builder->SetEdgeConfig(relation, as_undirected, add_self_loop, build_both_direction);
}

void GraphBuilderSetShuffleID(GraphBuilder *graph_builder,
                              bool shuffle_id) {
  graph_builder->SetShuffleID(shuffle_id);
}

void GraphBuilderSetGraphSaveFile(GraphBuilder *graph_builder,
                                  const std::string &csr_row_ptr_filename,
                                  const std::string &csr_col_idx_filename,
                                  const std::string &id_mapping_prefix) {
  graph_builder->SetGraphSaveFile(csr_row_ptr_filename, csr_col_idx_filename, id_mapping_prefix);
}

void GraphBuilderBuild(GraphBuilder *graph_builder) {
  graph_builder->Build();
}

GraphBuilder::GraphBuilder(const std::vector<std::string> &node_type_names,
                           const std::vector<std::vector<std::string>> &relations,
                           WMType id_type) {
  id_type_ = id_type;
  node_type_names_ = node_type_names;
  pool_ = std::make_unique<GraphBuilderThreadPool>();
  for (int i = 0; i < (int) node_type_names_.size(); i++) {
    auto it = node_type_name_to_idx_.find(node_type_names_[i]);
    WM_CHECK(it == node_type_name_to_idx_.end());
    node_type_name_to_idx_.emplace(std::make_pair(node_type_names_[i], i));
  }
  for (int i = 0; i < (int) relations.size(); i++) {
    WM_CHECK(relations[i].size() == 3);
    EdgeType et(relations[i]);
    WM_CHECK(node_type_name_to_idx_.find(et.src) != node_type_name_to_idx_.end());
    WM_CHECK(node_type_name_to_idx_.find(et.dst) != node_type_name_to_idx_.end());
    edge_types_.push_back(et);
    edge_type_to_idx_.emplace(std::make_pair(et, i));
  }
  edge_data_.resize(relations.size());
  edge_configs_.resize(relations.size());
  node_counts_.resize(node_type_names_.size(), 0);
  pool_->Start();
}

GraphBuilder::~GraphBuilder() {
  pool_.reset();
  for (auto &ed : edge_data_) {
    if (ed.edge_buffer) {
      free(ed.edge_buffer);
      ed.edge_buffer = nullptr;
    }
    if (ed.feature_buffer) {
      free(ed.feature_buffer);
      ed.feature_buffer = nullptr;
    }
    ed.count = 0;
  }
}

template<typename FileIdType, typename GraphIdType>
void LoadEdgeDataFromSingleFile(FILE *fp,
                                bool reverse,
                                int64_t file_start_edge_id,
                                int64_t local_edge_start,
                                int64_t file_edge_count,
                                size_t single_edge_size_bytes,
                                int64_t local_edge_count,
                                int64_t max_edge_count,
                                size_t edge_feature_size,
                                char *edge_load_buffer,
                                EdgeData edge_data) {
  GraphIdType *local_edge_buffer = (GraphIdType *) edge_data.edge_buffer + 2 * local_edge_start;
  char *local_feature_buffer = (char *) edge_data.feature_buffer + edge_feature_size * local_edge_start;
  int64_t edge_offset_in_file = 0;
  if (file_start_edge_id < local_edge_start) {
    edge_offset_in_file += local_edge_start - file_start_edge_id;
    WM_CHECK(fseeko64(fp, edge_offset_in_file * single_edge_size_bytes, SEEK_SET) == 0);
  }
  int src_offset = reverse ? 1 : 0;
  int dst_offset = reverse ? 0 : 1;
  while (edge_offset_in_file < file_edge_count
         && edge_offset_in_file + file_start_edge_id < local_edge_start + local_edge_count) {
    int64_t read_edge_count = std::min(file_edge_count - edge_offset_in_file,
                                       local_edge_start + local_edge_count - file_start_edge_id
                                           - edge_offset_in_file);
    if (read_edge_count > max_edge_count) read_edge_count = max_edge_count;
    WM_CHECK(fread(edge_load_buffer, single_edge_size_bytes, read_edge_count, fp) == read_edge_count);
    int64_t start_id = file_start_edge_id + edge_offset_in_file;
    int64_t local_start_id = start_id - local_edge_start;
    for (int read_id = 0; read_id < read_edge_count; read_id++) {
      local_edge_buffer[(local_start_id + read_id) * 2 + src_offset] = (GraphIdType)
          * (FileIdType *) (edge_load_buffer + single_edge_size_bytes * read_id);
      local_edge_buffer[(local_start_id + read_id) * 2 + dst_offset] = (GraphIdType)
          * (FileIdType *) (edge_load_buffer + single_edge_size_bytes * read_id + sizeof(FileIdType));
      char *cpsrc = edge_load_buffer + single_edge_size_bytes * read_id + sizeof(FileIdType) * 2;
      char *cpdst = local_feature_buffer + edge_feature_size * (read_id + local_start_id);
      if (edge_feature_size > 0) memcpy(cpdst, cpsrc, edge_feature_size);
    }
    edge_offset_in_file += read_edge_count;
  }
}

REGISTER_DISPATCH_TWO_TYPES(LoadEdgeDataFromSingleFile, LoadEdgeDataFromSingleFile, SINT3264, SINT3264)

void GraphBuilder::LoadEdgeDataFromFileList(const std::vector<std::string> &edge_desc,
                                            const std::string &file_prefix,
                                            bool reverse,
                                            WMType file_id_type,
                                            size_t edge_feature_size) {
  std::string src_node_name, relation, dst_node_name;
  if (edge_desc.size() == 3) {
    src_node_name = edge_desc[0];
    relation = edge_desc[1];
    dst_node_name = edge_desc[2];
  } else {
    WM_CHECK(edge_desc.size() == 0);
    WM_CHECK(edge_types_.size() == 1);
    src_node_name = edge_types_[0].src;
    dst_node_name = edge_types_[0].dst;
    relation = edge_types_[0].rel;
  }
  std::string real_src_name = src_node_name;
  std::string real_dst_name = dst_node_name;
  if (reverse) {
    std::swap(real_src_name, real_dst_name);
  }
  EdgeType edge_type({real_src_name, relation, real_dst_name});
  WM_CHECK(edge_type_to_idx_.find(edge_type) != edge_type_to_idx_.end());
  int eidx = edge_type_to_idx_[edge_type];
  WM_CHECK(file_id_type == WMT_Int32 || file_id_type == WMT_Int64);
  size_t single_edge_size_bytes = 2 * GetWMTSize(file_id_type) + edge_feature_size;
  std::vector<std::string> filelist;
  if (!GetPartFileListFromPrefix(file_prefix, &filelist)) {
    fprintf(stderr, "[WmmpLoadToCSRGraphTraverseEdgeFileList] GetPartFileListFromPrefix from prefix %s failed.\n",
            file_prefix.c_str());
    abort();
  }
  std::vector<int64_t> file_start_edge_ids, edge_counts;
  int64_t start_edge_id = 0;
  for (const auto &filename : filelist) {
    size_t file_size = StatFileSize(filename);
    if (file_size % single_edge_size_bytes != 0) {
      fprintf(stderr, "File %s size is %ld, but type %s, edge_feature_size=%ld, single_edge_size_bytes=%ld\n",
              filename.c_str(), file_size, GetWMTName(file_id_type), edge_feature_size, single_edge_size_bytes);
      abort();
    }
    int64_t file_edge_count = file_size / single_edge_size_bytes;
    file_start_edge_ids.push_back(start_edge_id);
    start_edge_id += file_edge_count;
    edge_counts.push_back(file_edge_count);
  }
  int64_t total_raw_edge_count = start_edge_id;
  CreateEdgeData(eidx, total_raw_edge_count, edge_feature_size);
  EdgeData edge_data = edge_data_[eidx];
  pool_->Run([&filelist, &file_start_edge_ids, &edge_counts, reverse, file_id_type, edge_feature_size,
              total_raw_edge_count, single_edge_size_bytes, edge_data, this](int rank, int size) {
    int64_t local_edge_start = total_raw_edge_count * rank / size;
    int64_t local_edge_end = total_raw_edge_count * (rank + 1) / size;
    int64_t local_edge_count = local_edge_end - local_edge_start;
    const int64_t kMemoryBlockSize = 64 * 1024 * 1024;
    int64_t max_edge_count = kMemoryBlockSize / single_edge_size_bytes;
    auto *edge_load_buffer = (char *) malloc(kMemoryBlockSize);

    for (int fidx = 0; fidx < (int) filelist.size(); fidx++) {
      int64_t file_start_edge_id = file_start_edge_ids[fidx];
      int64_t file_edge_count = edge_counts[fidx];
      if (file_start_edge_id + file_edge_count < local_edge_start) continue;
      if (file_start_edge_id >= local_edge_start + local_edge_count) break;
      FILE *fp = fopen(filelist[fidx].c_str(), "rb");
      if (fp == nullptr) {
        fprintf(stderr, "Open file %s failed.\n", filelist[fidx].c_str());
        abort();
      }
      printf("Rank=%d, loading from file %s\n", rank, filelist[fidx].c_str());
      DISPATCH_TWO_TYPES(file_id_type, this->id_type_, LoadEdgeDataFromSingleFile,
                         fp, reverse, file_start_edge_id, local_edge_start,
                         file_edge_count, single_edge_size_bytes, local_edge_count,
                         max_edge_count, edge_feature_size, edge_load_buffer, edge_data);
      fclose(fp);
    }
    free(edge_load_buffer);
  });
}
template<typename GraphIDType>
void FindMaxNodeID(int64_t *src_max_id, int64_t *dst_max_id, const EdgeData &ed, int rank, int size) {
  int64_t count = ed.count;
  int64_t start = rank * count / size;
  int64_t end = (rank + 1) * count / size;
  const GraphIDType *edge_buffer = (const GraphIDType *) ed.edge_buffer;
  int64_t smax = -1;
  int64_t dmax = -1;
  for (int64_t offset = start; offset < end; offset++) {
    GraphIDType sid = edge_buffer[2 * offset];
    GraphIDType did = edge_buffer[2 * offset + 1];
    if (sid > smax) smax = sid;
    if (did > dmax) dmax = did;
  }
  *src_max_id = smax;
  *dst_max_id = dmax;
}
REGISTER_DISPATCH_ONE_TYPE(FindMaxNodeID, FindMaxNodeID, SINT3264)
void GraphBuilder::FillNodeCounts() {
  std::vector<int64_t> max_node_ids(node_type_names_.size(), -1);
  for (int i = 0; i < (int) edge_types_.size(); i++) {
    EdgeData edge_data = edge_data_[i];
    if (edge_data.edge_buffer == nullptr || edge_data.count == 0) continue;
    std::vector<int64_t> src_max_node_id(pool_->Size(), 0), dst_max_node_id(pool_->Size(), 0);
    pool_->Run([&src_max_node_id, &dst_max_node_id, edge_data, this](int rank, int size) {
      DISPATCH_ONE_TYPE(this->id_type_,
                        FindMaxNodeID,
                        &src_max_node_id[rank],
                        &dst_max_node_id[rank],
                        edge_data,
                        rank,
                        size);
    });
    int64_t src_maxid = -1;
    int64_t dst_maxid = -1;
    for (int j = 0; j < (int) pool_->Size(); j++) {
      if (src_max_node_id[j] > src_maxid) src_maxid = src_max_node_id[j];
      if (dst_max_node_id[j] > dst_maxid) dst_maxid = dst_max_node_id[j];
    }
    int src_node_type_idx = GetNodeIdx(edge_types_[i].src);
    int dst_node_type_idx = GetNodeIdx(edge_types_[i].dst);
    fprintf(stderr, "edge[%d] src_idx=%d, src_max=%ld, dst_idx=%d, dst_max=%ld\n",
            i, src_node_type_idx, src_maxid, dst_node_type_idx, dst_maxid);
    if (max_node_ids[src_node_type_idx] < src_maxid) max_node_ids[src_node_type_idx] = src_maxid;
    if (max_node_ids[dst_node_type_idx] < dst_maxid) max_node_ids[dst_node_type_idx] = dst_maxid;
  }
  for (int i = 0; i < (int) node_counts_.size(); i++) {
    if (node_counts_[i] == 0) {
      node_counts_[i] = max_node_ids[i] + 1;
    } else {
      fprintf(stderr, "node_counts_[%d]=%ld, max_node_ids=%ld\n", i, node_counts_[i], max_node_ids[i]);
      WM_CHECK(node_counts_[i] > max_node_ids[i]);
    }
    fprintf(stderr, "node_counts_[%d]=%ld\n", i, node_counts_[i]);
  }
}
void GraphBuilder::GenerateMixedNodeConvertTable(bool shuffle) {
  fprintf(stderr, "Start generating Mixed node convert table.\n");
  FillNodeCounts();
  to_mixed_id.resize(node_counts_.size());
  int64_t all_node_count = 0;
  for (auto type_node_count : node_counts_) {
    all_node_count += type_node_count;
  }
  to_typed_id.resize(all_node_count);
  int64_t type_start_idx = 0;
  for (int type_id = 0; type_id < (int) node_counts_.size(); type_id++) {
    int64_t type_node_count = node_counts_[type_id];
    to_mixed_id[type_id].resize(type_node_count);

    pool_->Run([type_id, type_start_idx, type_node_count, this](int rank, int size) {
      int64_t start = rank * type_node_count / size;
      int64_t end = (rank + 1) * type_node_count / size;
      for (int64_t id = start; id < end; id++) {
        TypedNodeID typed_node_id = MakeTypedID(type_id, id);
        to_typed_id[id + type_start_idx] = typed_node_id;
      }
    });

    type_start_idx += type_node_count;
  }
  if (shuffle) {
    std::shuffle(to_typed_id.begin(), to_typed_id.end(), std::mt19937(std::random_device()()));
  }
  pool_->Run([all_node_count, this](int rank, int size) {
    int64_t start = rank * all_node_count / size;
    int64_t end = (rank + 1) * all_node_count / size;
    for (int64_t mix_id = start; mix_id < end; mix_id++) {
      TypedNodeID typed_node_id = to_typed_id[mix_id];
      int type_id = TypeOfTypedID(typed_node_id);
      int id = IDOfTypedID(typed_node_id);
      to_mixed_id[type_id][id] = mix_id;
    }
  });
  fprintf(stderr, "Done generating Mixed node convert table.\n");
}
template<typename IdType>
void SaveVectorToFile(const std::vector<int64_t> &save_vector, FILE *fp) {
  const int kBufferSize = 1024 * 1024 * 8;
  std::vector<IdType> buffer(kBufferSize);
  for (size_t start = 0; start < save_vector.size(); start += kBufferSize) {
    size_t batch_size = save_vector.size() - start;
    if (batch_size > kBufferSize) batch_size = kBufferSize;
    for (size_t i = 0; i < batch_size; i++) {
      buffer[i] = (IdType) save_vector[i + start];
    }
    ssize_t wret = fwrite(buffer.data(), sizeof(IdType), batch_size, fp);
    WM_CHECK(wret == batch_size);
  }
}
REGISTER_DISPATCH_ONE_TYPE(SaveVectorToFile, SaveVectorToFile, SINT3264)
void GraphBuilder::SaveMapping(const std::vector<int64_t> &save_vector, const std::string &name, bool force_int64) {
  std::string mapping_filename = id_mapping_prefix_;
  mapping_filename.append("_").append(name);
  FILE *fp = fopen(mapping_filename.c_str(), "wb");
  if (fp == nullptr) {
    fprintf(stderr, "Open file %s failed for write.\n", mapping_filename.c_str());
    abort();
  }
  if (force_int64) {
    SaveVectorToFile<int64_t>(save_vector, fp);
  } else {
    DISPATCH_ONE_TYPE(id_type_, SaveVectorToFile, save_vector, fp);
  }
  fclose(fp);
}
void GraphBuilder::Build() {
  // currently only mixed type is supported.
  WM_CHECK(graph_type == GT_Mixed || graph_type == GT_Homo);
  if (graph_type == GT_Mixed) {
    BuildMixed();
  } else if (graph_type == GT_Homo) {
    BuildHomo();
  }
}

template<typename IdType>
void GraphBuilderBuildMixed(GraphBuilder *graph_builder) {
  fprintf(stderr, "Starting GraphBuilderBuildMixed...\n");
  WM_CHECK(graph_builder->edge_types_.size() == graph_builder->edge_data_.size());
  WM_CHECK(graph_builder->edge_types_.size() == graph_builder->edge_configs_.size());
  int64_t final_node_count = 0;
  for (long node_count : graph_builder->node_counts_) {
    final_node_count += node_count;
  }
  fprintf(stderr, "GraphBuilderBuildMixed final_node_count=%ld\n", final_node_count);
  std::vector<std::vector<std::vector<std::array<IdType, 2>>>> bucket_thread_edges(graph_builder->pool_->Size());
  for (auto &v : bucket_thread_edges) {
    v.resize(graph_builder->pool_->Size());
  }
  graph_builder->pool_->Run([graph_builder, &bucket_thread_edges, final_node_count](int rank, int size) {
    int64_t bucket_size = (final_node_count + size - 1) / size;
    for (int edge_type_idx = 0; edge_type_idx < (int) graph_builder->edge_types_.size(); edge_type_idx++) {
      GraphBuilder::EdgeConfig &edge_config = graph_builder->edge_configs_[edge_type_idx];
      EdgeData &edge_data = graph_builder->edge_data_[edge_type_idx];
      EdgeType &edge_type = graph_builder->edge_types_[edge_type_idx];
      int src_type_idx, dst_type_idx;
      auto it = graph_builder->node_type_name_to_idx_.find(edge_type.src);
      WM_CHECK(it != graph_builder->node_type_name_to_idx_.end());
      src_type_idx = it->second;
      it = graph_builder->node_type_name_to_idx_.find(edge_type.dst);
      WM_CHECK(it != graph_builder->node_type_name_to_idx_.end());
      dst_type_idx = it->second;
      int64_t edge_start = edge_data.count * rank / size;
      int64_t edge_end = edge_data.count * (rank + 1) / size;
      auto *edge_buffer = (IdType *) edge_data.edge_buffer;
      for (int64_t edge_idx = edge_start; edge_idx < edge_end; edge_idx++) {
        IdType src_id = edge_buffer[edge_idx * 2];
        IdType dst_id = edge_buffer[edge_idx * 2 + 1];
        IdType src_mid = graph_builder->to_mixed_id[src_type_idx][src_id];
        IdType dst_mid = graph_builder->to_mixed_id[dst_type_idx][dst_id];
        int src_bucket = src_mid / bucket_size;
        int dst_bucket = dst_mid / bucket_size;
        std::array<IdType, 2> edge({src_mid, dst_mid});
        std::vector<std::array<IdType, 2>> &v_src = bucket_thread_edges[src_bucket][rank];
        v_src.push_back(edge);
        if (edge_config.build_both_direction || edge_config.as_undirected) {
          edge[0] = dst_mid;
          edge[1] = src_mid;
          std::vector<std::array<IdType, 2>> &v_dst = bucket_thread_edges[dst_bucket][rank];
          v_dst.push_back(edge);
        }
      }
      if (edge_config.add_self_loop) {
        int64_t node_count = graph_builder->node_counts_[src_type_idx];
        IdType node_start = (IdType)(rank * node_count / size);
        IdType node_end = (IdType)(rank * node_count / size);
        for (IdType node_id = node_start; node_id < node_end; node_id++) {
          IdType src_mid = graph_builder->to_mixed_id[src_type_idx][node_id];
          int src_bucket = src_mid / bucket_size;
          std::array<IdType, 2> edge({src_mid, src_mid});
          std::vector<std::array<IdType, 2>> &v_src = bucket_thread_edges[src_bucket][rank];
          v_src.push_back(edge);
        }
      }
    }
  });
  fprintf(stderr, "Finished bucket edges.\n");
  std::vector<std::vector<IdType>> all_edges(final_node_count);
  graph_builder->pool_->Run([graph_builder, final_node_count, &bucket_thread_edges, &all_edges](int rank, int size) {
    for (int i = 0; i < size; i++) {
      auto &bucket = bucket_thread_edges[rank][i];
      for (auto &edge : bucket) {
        all_edges[edge[0]].push_back(edge[1]);
      }
      std::vector<std::array<IdType, 2>> empty;
      bucket.swap(empty);
    }
  });
  fprintf(stderr, "Finished building Mixed graph.\n");
  std::vector<int64_t> node_edge_count(final_node_count);
  graph_builder->pool_->Run([&all_edges, &node_edge_count, final_node_count](int rank, int size) {
    int64_t start = rank * final_node_count / size;
    int64_t end = (rank + 1) * final_node_count / size;
    for (int64_t nid = start; nid < end; nid++) {
      node_edge_count[nid] = all_edges[nid].size();
    }
  });
  std::vector<int64_t> node_edge_start(final_node_count + 1);
  node_edge_start[0] = 0;
  for (int64_t nid = 0; nid < final_node_count; nid++) {
    node_edge_start[nid + 1] = node_edge_start[nid] + node_edge_count[nid];
  }
  int64_t final_edge_count = node_edge_start[final_node_count];
  fprintf(stderr, "Mixed graph CSR row ready, final_edge_count=%ld.\n", final_edge_count);
  WM_CHECK(!graph_builder->csr_row_ptr_filename_.empty());
  FILE *fp = fopen(graph_builder->csr_row_ptr_filename_.c_str(), "wb");
  WM_CHECK(fp != nullptr);
  size_t fret = fwrite(node_edge_start.data(), sizeof(int64_t), final_node_count + 1, fp);
  WM_CHECK(fret == final_node_count + 1);
  fclose(fp);
  fprintf(stderr, "Mixed graph CSR row write_done.\n");
  fp = nullptr;
  WM_CHECK(!graph_builder->csr_col_idx_filename_.empty());
  fp = fopen(graph_builder->csr_col_idx_filename_.c_str(), "wb");
  WM_CHECK(fp != nullptr);
  fseeko64(fp, final_edge_count * sizeof(IdType), SEEK_SET);
  int fd = fileno(fp);

  graph_builder->pool_->Run([fd, &node_edge_start, &all_edges, final_node_count, final_edge_count](int rank, int size) {
    int64_t node_id_start = final_node_count * rank / size;
    int64_t node_id_end = final_node_count * (rank + 1) / size;
    int64_t file_offset = node_edge_start[node_id_start] * sizeof(IdType);
    const int64_t kBufferEltCount = 1024 * 1024 * 4;
    int64_t buffer_offset = 0;
    IdType *buffer = (IdType *) malloc(kBufferEltCount * sizeof(IdType));

    for (int64_t node_id = node_id_start; node_id < node_id_end; node_id++) {
      int64_t node_edge_count = all_edges[node_id].size();
      if (node_edge_count + buffer_offset >= kBufferEltCount && buffer_offset > 0) {
        size_t write_size = buffer_offset * sizeof(IdType);
        ssize_t bytes = pwrite64(fd, buffer, write_size, file_offset);
        WM_CHECK(bytes == write_size);
        size_t new_file_offset = node_edge_start[node_id] * sizeof(IdType);
        WM_CHECK(write_size + file_offset == new_file_offset);
        file_offset = new_file_offset;
        buffer_offset = 0;
      }
      if (node_edge_count + buffer_offset <= kBufferEltCount) {
        memcpy(buffer + buffer_offset, all_edges[node_id].data(), node_edge_count * sizeof(IdType));
        buffer_offset += node_edge_count;
      } else {
        WM_CHECK(node_edge_count > kBufferEltCount);
        WM_CHECK(buffer_offset == 0);
        WM_CHECK(file_offset == node_edge_start[node_id] * sizeof(IdType));
        ssize_t bytes = pwrite64(fd, all_edges[node_id].data(), node_edge_count * sizeof(IdType), file_offset);
        if (bytes != node_edge_count * sizeof(IdType)) {
          fprintf(stderr, "pwrite64 returned %ld, but node_edge_count = %ld, sizeof(IdType)=%ld\n",
                  bytes, node_edge_count, sizeof(IdType));
          abort();
        }
        file_offset += bytes;
      }
    }
    if (buffer_offset > 0) {
      size_t write_size = buffer_offset * sizeof(IdType);
      ssize_t bytes = pwrite64(fd, buffer, write_size, file_offset);
      WM_CHECK(bytes == write_size);
      size_t new_file_offset = node_edge_start[node_id_end] * sizeof(IdType);
      WM_CHECK(write_size + file_offset == new_file_offset);
    }

    free(buffer);
  });

  fclose(fp);
  if (graph_builder->shuffle_id_) {
    fprintf(stderr, "Mixed graph CSR col write_done.\n");
    graph_builder->SaveMapping(graph_builder->to_typed_id, "mixed_to_typed", true);
    fprintf(stderr, "mixed to typed id mapping saved.\n");
    for (int i = 0; i < (int) graph_builder->node_type_names_.size(); i++) {
      graph_builder->SaveMapping(graph_builder->to_mixed_id[i], graph_builder->node_type_names_[i]);
      fprintf(stderr, "type %s to mixed id saved\n", graph_builder->node_type_names_[i].c_str());
    }
  }
}

REGISTER_DISPATCH_ONE_TYPE(GraphBuilderBuildMixed, GraphBuilderBuildMixed, SINT3264)

void GraphBuilder::BuildMixed() {
  GenerateMixedNodeConvertTable(shuffle_id_);
  DISPATCH_ONE_TYPE(id_type_, GraphBuilderBuildMixed, this);
}

void GraphBuilder::BuildHomo() {
  GenerateMixedNodeConvertTable(shuffle_id_);
  DISPATCH_ONE_TYPE(id_type_, GraphBuilderBuildMixed, this);
}

}// namespace whole_graph