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
#include "../wholegraph_ops/graph_sampling_test_utils.hpp"
#include "csr_add_self_loop_utils.hpp"
#include <experimental/random>
#include <gtest/gtest.h>
#include <random>
#include <wholememory/graph_op.h>
#include <wholememory_ops/register.hpp>

namespace graph_ops {
namespace testing {

template <typename RowPtrType, typename ColIdType>
void host_get_local_csr_graph(int row_num,
                              int col_num,
                              int graph_edge_num,
                              void* host_csr_row_ptr,
                              wholememory_array_description_t csr_row_ptr_desc,
                              void* host_csr_col_ptr,
                              wholememory_array_description_t csr_col_ptr_desc)
{
  RowPtrType* csr_row_ptr  = static_cast<RowPtrType*>(host_csr_row_ptr);
  ColIdType* csr_col_ptr   = static_cast<ColIdType*>(host_csr_col_ptr);
  int average_edge_per_row = graph_edge_num / row_num;
  std::default_random_engine generator;
  std::binomial_distribution<int> distribution(average_edge_per_row, 1);
  int total_edge = 0;
  for (int i = 0; i < row_num; i++) {
    while (true) {
      int random_num = distribution(generator);
      if (random_num >= 0 && random_num <= col_num) {
        csr_row_ptr[i] = random_num;
        total_edge += random_num;
        break;
      }
    }
  }

  int adjust_edge = std::abs(total_edge - graph_edge_num);
  std::random_device rand_dev;
  std::mt19937 gen(rand_dev());
  std::uniform_int_distribution<int> distr(0, row_num - 1);

  if (total_edge > graph_edge_num) {
    for (int i = 0; i < adjust_edge; i++) {
      while (true) {
        int random_row_id = distr(gen);
        if (csr_row_ptr[random_row_id] > 0) {
          csr_row_ptr[random_row_id]--;
          break;
        }
      }
    }
  }
  if (total_edge < graph_edge_num) {
    for (int i = 0; i < adjust_edge; i++) {
      while (true) {
        int random_row_id = distr(gen);
        if (csr_row_ptr[random_row_id] < col_num) {
          csr_row_ptr[random_row_id]++;
          break;
        }
      }
    }
  }
  wholegraph_ops::testing::host_prefix_sum_array(host_csr_row_ptr, csr_row_ptr_desc);
  EXPECT_TRUE(csr_row_ptr[row_num] == graph_edge_num);

  for (int i = 0; i < row_num; i++) {
    int start      = csr_row_ptr[i];
    int end        = csr_row_ptr[i + 1];
    int edge_count = end - start;
    if (edge_count == 0) continue;

    std::vector<int> array_in(col_num);
    for (int i = 0; i < col_num; i++) {
      array_in[i] = i;
    }
    std::sample(array_in.begin(), array_in.end(), &csr_col_ptr[start], edge_count, gen);
  }
}

REGISTER_DISPATCH_TWO_TYPES(HOSTGETLOCALCSRGRAPH, host_get_local_csr_graph, SINT3264, SINT3264)

template <typename DataType>
void get_random_float_array(void* host_csr_weight_ptr,
                            wholememory_array_description_t graph_csr_weight_ptr_desc)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<DataType> dis(1.0, 20.0);
  for (int64_t i = 0; i < graph_csr_weight_ptr_desc.size; i++) {
    static_cast<DataType*>(host_csr_weight_ptr)[i] = (DataType)dis(gen);
  }
}

void gen_local_csr_graph(int row_num,
                         int col_num,
                         int graph_edge_num,
                         void* host_csr_row_ptr,
                         wholememory_array_description_t csr_row_ptr_desc,
                         void* host_csr_col_ptr,
                         wholememory_array_description_t csr_col_ptr_desc,
                         void* host_csr_weight_ptr,
                         wholememory_array_description_t csr_weight_ptr_desc)
{
  DISPATCH_TWO_TYPES(csr_row_ptr_desc.dtype,
                     csr_col_ptr_desc.dtype,
                     HOSTGETLOCALCSRGRAPH,
                     row_num,
                     col_num,
                     graph_edge_num,
                     host_csr_row_ptr,
                     csr_row_ptr_desc,
                     host_csr_col_ptr,
                     csr_col_ptr_desc);
  if (host_csr_weight_ptr != nullptr) {
    if (csr_weight_ptr_desc.dtype == WHOLEMEMORY_DT_FLOAT) {
      get_random_float_array<float>(host_csr_weight_ptr, csr_weight_ptr_desc);
    } else if (csr_weight_ptr_desc.dtype == WHOLEMEMORY_DT_DOUBLE) {
      get_random_float_array<double>(host_csr_weight_ptr, csr_weight_ptr_desc);
    }
  }
}

void host_get_csr_add_self_loop(int* host_csr_row_ptr,
                                wholememory_array_description_t csr_row_ptr_array_desc,
                                int* host_csr_col_ptr,
                                wholememory_array_description_t csr_col_ptr_array_desc,
                                int* host_ref_output_csr_row_ptr,
                                wholememory_array_description_t output_csr_row_ptr_array_desc,
                                int* host_ref_output_csr_col_ptr,
                                wholememory_array_description_t output_csr_col_ptr_array_desc)
{
  for (int64_t row_id = 0; row_id < csr_row_ptr_array_desc.size - 1; row_id++) {
    int start                                   = host_csr_row_ptr[row_id];
    int end                                     = host_csr_row_ptr[row_id + 1];
    host_ref_output_csr_row_ptr[row_id]         = start + row_id;
    host_ref_output_csr_col_ptr[start + row_id] = row_id;
    for (int64_t j = start; j < end; j++) {
      host_ref_output_csr_col_ptr[j + row_id + 1] = host_csr_col_ptr[j];
    }
  }
  host_ref_output_csr_row_ptr[csr_row_ptr_array_desc.size - 1] =
    host_csr_row_ptr[csr_row_ptr_array_desc.size - 1] + csr_row_ptr_array_desc.size - 1;
}

void host_csr_add_self_loop(void* host_csr_row_ptr,
                            wholememory_array_description_t csr_row_ptr_array_desc,
                            void* host_csr_col_ptr,
                            wholememory_array_description_t csr_col_ptr_array_desc,
                            void* host_ref_output_csr_row_ptr,
                            wholememory_array_description_t output_csr_row_ptr_array_desc,
                            void* host_ref_output_csr_col_ptr,
                            wholememory_array_description_t output_csr_col_ptr_array_desc)
{
  EXPECT_EQ(csr_row_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(csr_col_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(output_csr_row_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(output_csr_col_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(csr_row_ptr_array_desc.size, output_csr_row_ptr_array_desc.size);
  EXPECT_EQ(csr_col_ptr_array_desc.size + csr_row_ptr_array_desc.size - 1,
            output_csr_col_ptr_array_desc.size);

  host_get_csr_add_self_loop(static_cast<int*>(host_csr_row_ptr),
                             csr_row_ptr_array_desc,
                             static_cast<int*>(host_csr_col_ptr),
                             csr_col_ptr_array_desc,
                             static_cast<int*>(host_ref_output_csr_row_ptr),
                             output_csr_row_ptr_array_desc,
                             static_cast<int*>(host_ref_output_csr_col_ptr),
                             output_csr_col_ptr_array_desc);
}

}  // namespace testing
}  // namespace graph_ops
