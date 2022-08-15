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
#pragma once

#include <string>

#include "data_type.h"

namespace whole_graph {

class GraphBuilder;

GraphBuilder *CreateMixedGraphBuilder(const std::vector<std::string> &node_type_names,
                                      const std::vector<std::vector<std::string>> &relations,
                                      WMType id_type);

GraphBuilder *CreateHomoGraphBuilder(WMType id_type);

void DestroyGraphBuilder(GraphBuilder *graph_builder);

void GraphBuilderSetNodeCounts(GraphBuilder *graph_builder, const std::vector<int64_t> &node_counts);

void GraphBuilderLoadEdgeDataFromFileList(GraphBuilder *graph_builder,
                                          const std::vector<std::string> &relations,
                                          const std::string &file_prefix,
                                          bool reverse,
                                          WMType file_id_type,
                                          size_t edge_feature_size);

void GraphBuilderSetEdgeConfig(GraphBuilder *graph_builder,
                               const std::vector<std::string> &relation,
                               bool as_undirected,
                               bool add_self_loop,
                               bool build_both_direction);

// default is shuffle
void GraphBuilderSetShuffleID(GraphBuilder *graph_builder,
                              bool shuffle_id);

void GraphBuilderSetGraphSaveFile(GraphBuilder *graph_builder,
                                  const std::string &csr_row_ptr_filename,
                                  const std::string &csr_col_idx_filename,
                                  const std::string &id_mapping_prefix);

void GraphBuilderBuild(GraphBuilder *graph_builder);

}// namespace whole_graph