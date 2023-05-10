/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include "append_unique_test_utils.hpp"
#include <gtest/gtest.h>
#include <iterator>
#include <random>
#include <unordered_map>
#include <vector>
#include <wholememory/tensor_description.h>

namespace graph_ops {
namespace testing {
template <typename DataType>
void host_get_random_node_ids(void* nodes, int64_t node_count, int64_t range, bool unique)
{
  DataType* nodes_ptr = static_cast<DataType*>(nodes);
  std::random_device rand_dev;
  std::mt19937 gen(rand_dev());
  std::uniform_int_distribution<int64_t> distr(0, range);
  if (!unique) {
    for (int64_t i = 0; i < node_count; i++) {
      nodes_ptr[i] = distr(gen);
    }
  } else {
    std::vector<DataType> tmp_array_in(range);
    for (int64_t i = 0; i < range; i++) {
      tmp_array_in[i] = i;
    }

    std::sample(tmp_array_in.begin(), tmp_array_in.end(), nodes_ptr, node_count, gen);
  }
}

void gen_node_ids(void* host_target_nodes_ptr,
                  wholememory_array_description_t node_desc,
                  int64_t range,
                  bool unique)
{
  int64_t node_count = node_desc.size;
  if (node_desc.dtype == WHOLEMEMORY_DT_INT) {
    host_get_random_node_ids<int>(host_target_nodes_ptr, node_count, range, unique);
  } else if (node_desc.dtype == WHOLEMEMORY_DT_INT64) {
    host_get_random_node_ids<int64_t>(host_target_nodes_ptr, node_count, range, unique);
  }
}

template <typename DataType>
void insert_nodes_to_append_unique_hash_table(
  std::unordered_map<DataType, int>& append_unique_hash_table,
  DataType* node_ptr,
  int64_t node_count,
  int* unique_count_ptr,
  bool target)
{
  if (target) {
    for (int64_t i = 0; i < node_count; i++) {
      DataType key = node_ptr[i];
      append_unique_hash_table.insert(std::make_pair(key, i));
    }
    return;
  } else {
    int unique_count = *unique_count_ptr;
    for (int64_t i = 0; i < node_count; i++) {
      DataType key = node_ptr[i];
      if (append_unique_hash_table.find(key) == append_unique_hash_table.end()) {
        append_unique_hash_table.insert(std::make_pair(key, unique_count));
        unique_count++;
      }
    }
    *unique_count_ptr = unique_count;
  }
}

template <typename DataType>
void host_get_append_unique(void* target_nodes_ptr,
                            wholememory_array_description_t target_nodes_desc,
                            void* neighbor_nodes_ptr,
                            wholememory_array_description_t neighbor_nodes_desc,
                            int* host_total_unique_count,
                            void** host_output_unique_nodes_ptr,
                            void** host_output_neighbor_raw_to_unique_ptr,
                            wholememory_array_description_t output_neighbor_raw_to_unique_desc)
{
  std::unordered_map<DataType, int> append_unique_hash_table;
  int unique_count = target_nodes_desc.size;
  insert_nodes_to_append_unique_hash_table<DataType>(append_unique_hash_table,
                                                     static_cast<DataType*>(target_nodes_ptr),
                                                     target_nodes_desc.size,
                                                     &unique_count,
                                                     true);
  insert_nodes_to_append_unique_hash_table<DataType>(append_unique_hash_table,
                                                     static_cast<DataType*>(neighbor_nodes_ptr),
                                                     neighbor_nodes_desc.size,
                                                     &unique_count,
                                                     false);

  *host_output_unique_nodes_ptr = (DataType*)malloc(unique_count * sizeof(DataType));
  *host_total_unique_count      = unique_count;

  for (auto iter = append_unique_hash_table.begin(); iter != append_unique_hash_table.end();
       iter++) {
    DataType key                                                 = iter->first;
    int index                                                    = iter->second;
    static_cast<DataType*>(*host_output_unique_nodes_ptr)[index] = key;
  }
  for (int64_t i = 0; i < neighbor_nodes_desc.size; i++) {
    DataType key = static_cast<DataType*>(neighbor_nodes_ptr)[i];
    static_cast<int*>(*host_output_neighbor_raw_to_unique_ptr)[i] = append_unique_hash_table[key];
  }
}

void host_append_unique(void* target_nodes_ptr,
                        wholememory_array_description_t target_nodes_desc,
                        void* neighbor_nodes_ptr,
                        wholememory_array_description_t neighbor_nodes_desc,
                        int* host_total_unique_count,
                        void** host_output_unique_nodes_ptr,
                        void** host_output_neighbor_raw_to_unique_ptr,
                        wholememory_array_description_t output_neighbor_raw_to_unique_desc)
{
  EXPECT_EQ(target_nodes_desc.dtype, neighbor_nodes_desc.dtype);
  *host_output_neighbor_raw_to_unique_ptr =
    (void*)malloc(wholememory_get_memory_size_from_array(&output_neighbor_raw_to_unique_desc));

  if (target_nodes_desc.dtype == WHOLEMEMORY_DT_INT) {
    host_get_append_unique<int>(target_nodes_ptr,
                                target_nodes_desc,
                                neighbor_nodes_ptr,
                                neighbor_nodes_desc,
                                host_total_unique_count,
                                host_output_unique_nodes_ptr,
                                host_output_neighbor_raw_to_unique_ptr,
                                output_neighbor_raw_to_unique_desc);
  } else if (target_nodes_desc.dtype == WHOLEMEMORY_DT_INT64) {
    host_get_append_unique<int64_t>(target_nodes_ptr,
                                    target_nodes_desc,
                                    neighbor_nodes_ptr,
                                    neighbor_nodes_desc,
                                    host_total_unique_count,
                                    host_output_unique_nodes_ptr,
                                    host_output_neighbor_raw_to_unique_ptr,
                                    output_neighbor_raw_to_unique_desc);
  }
}

}  // namespace testing
}  // namespace graph_ops
