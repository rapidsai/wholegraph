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
#include "spmm_csr_no_weight_utils.hpp"
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

void gen_features(void* feature_ptr, wholememory_matrix_description_t feature_desc)
{
  int64_t feature_size = feature_desc.sizes[0] * feature_desc.sizes[1];
  wholememory_array_description_t tmp_array_desc =
    wholememory_create_array_desc(feature_size, 0, feature_desc.dtype);
  if (feature_desc.dtype == WHOLEMEMORY_DT_FLOAT) {
    get_random_float_array<float>(feature_ptr, tmp_array_desc);

  } else if (feature_desc.dtype == WHOLEMEMORY_DT_DOUBLE) {
    get_random_float_array<double>(feature_ptr, tmp_array_desc);
  }
}

void gen_features(void* feature_ptr, wholememory_tensor_description_t feature_desc)
{
  int64_t feature_size = 1;
  for (int i = 0; i < feature_desc.dim; i++) {
    feature_size = feature_desc.sizes[i];
  }
  wholememory_array_description_t tmp_array_desc =
    wholememory_create_array_desc(feature_size, 0, feature_desc.dtype);
  if (feature_desc.dtype == WHOLEMEMORY_DT_FLOAT) {
    get_random_float_array<float>(feature_ptr, tmp_array_desc);

  } else if (feature_desc.dtype == WHOLEMEMORY_DT_DOUBLE) {
    get_random_float_array<double>(feature_ptr, tmp_array_desc);
  }
}

template <typename FeatureType>
void get_spmm_csr_no_weight_forward(void* host_csr_row_ptr,
                                    wholememory_array_description_t csr_row_ptr_desc,
                                    void* host_csr_col_ptr,
                                    wholememory_array_description_t csr_col_ptr_desc,
                                    void* input_feature_ptr,
                                    wholememory_matrix_description_t input_feature_desc,
                                    int aggregator,
                                    void* host_output_feature,
                                    wholememory_matrix_description_t output_feature_desc)
{
  int* csr_row_ptr            = static_cast<int*>(host_csr_row_ptr);
  int* csr_col_ptr            = static_cast<int*>(host_csr_col_ptr);
  FeatureType* input_feature  = static_cast<FeatureType*>(input_feature_ptr);
  FeatureType* output_feature = static_cast<FeatureType*>(host_output_feature);
  int row_num                 = csr_row_ptr_desc.size - 1;
  int feature_dim             = input_feature_desc.sizes[1];
  int feature_stride          = input_feature_desc.stride;

  for (int i = 0; i < row_num; i++) {
    int start = csr_row_ptr[i];
    int end   = csr_row_ptr[i + 1];

    for (int k = 0; k < feature_dim; k++) {
      FeatureType sum = 0;
      if (aggregator == GCN_AGGREGATOR) { sum += input_feature[i * feature_stride + k]; }
      for (int j = start; j < end; j++) {
        int col_id = csr_col_ptr[j];
        sum += input_feature[col_id * feature_stride + k];
      }
      if (aggregator == GCN_AGGREGATOR) { sum /= (end - start + 1); }
      if (aggregator == MEAN_AGGREGATOR) { sum /= (end - start); }
      output_feature[i * feature_stride + k] = sum;
    }
  }
}

void host_spmm_csr_no_weight_forward(void* host_csr_row_ptr,
                                     wholememory_array_description_t csr_row_ptr_desc,
                                     void* host_csr_col_ptr,
                                     wholememory_array_description_t csr_col_ptr_desc,
                                     void* input_feature_ptr,
                                     wholememory_matrix_description_t input_feature_desc,
                                     int aggregator,
                                     void* host_output_feature,
                                     wholememory_matrix_description_t output_feature_desc)
{
  EXPECT_EQ(csr_row_ptr_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(csr_col_ptr_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(input_feature_desc.dtype, output_feature_desc.dtype);
  EXPECT_EQ(output_feature_desc.sizes[1], input_feature_desc.sizes[1]);

  if (input_feature_desc.dtype == WHOLEMEMORY_DT_FLOAT) {
    get_spmm_csr_no_weight_forward<float>(host_csr_row_ptr,
                                          csr_row_ptr_desc,
                                          host_csr_col_ptr,
                                          csr_col_ptr_desc,
                                          input_feature_ptr,
                                          input_feature_desc,
                                          aggregator,
                                          host_output_feature,
                                          output_feature_desc);
  } else if (input_feature_desc.dtype == WHOLEMEMORY_DT_DOUBLE) {
    get_spmm_csr_no_weight_forward<double>(host_csr_row_ptr,
                                           csr_row_ptr_desc,
                                           host_csr_col_ptr,
                                           csr_col_ptr_desc,
                                           input_feature_ptr,
                                           input_feature_desc,
                                           aggregator,
                                           host_output_feature,
                                           output_feature_desc);
  }
}

template <typename T>
void check_float_matrix_same(void* input,
                             wholememory_matrix_description_t input_matrix_desc,
                             void* input_ref,
                             wholememory_matrix_description_t input_ref_matrix_desc,
                             double epsilon = 1e-3)
{
  int64_t diff_count = 0;
  for (int64_t i = 0; i < input_matrix_desc.sizes[0]; i++) {
    for (int64_t j = 0; j < input_matrix_desc.sizes[1]; j++) {
      T value     = static_cast<T*>(input)[i * input_matrix_desc.stride + j];
      T ref_value = static_cast<T*>(input_ref)[i * input_matrix_desc.stride + j];
      if (std::abs(value - ref_value) > epsilon) { diff_count++; }
      if (diff_count < 5 && diff_count > 0) {
        printf(
          "row=%ld, col=%ld, got (float %f), but should be (float %f)\n", i, j, value, ref_value);
        fflush(stdout);
      }
    }
  }
  EXPECT_EQ(diff_count, 0);
}

void host_check_float_matrix_same(void* input,
                                  wholememory_matrix_description_t input_matrix_desc,
                                  void* input_ref,
                                  wholememory_matrix_description_t input_ref_matrix_desc)
{
  EXPECT_EQ(input_matrix_desc.dtype, input_ref_matrix_desc.dtype);
  EXPECT_EQ(input_matrix_desc.sizes[0], input_ref_matrix_desc.sizes[0]);
  EXPECT_EQ(input_matrix_desc.sizes[1], input_matrix_desc.sizes[1]);
  if (input_matrix_desc.dtype == WHOLEMEMORY_DT_FLOAT) {
    check_float_matrix_same<float>(input, input_matrix_desc, input_ref, input_ref_matrix_desc);
  } else if (input_matrix_desc.dtype == WHOLEMEMORY_DT_DOUBLE) {
    check_float_matrix_same<double>(input, input_matrix_desc, input_ref, input_ref_matrix_desc);
  }
}

template <typename T>
void check_float_array_same(T* input, T* input_ref, int64_t total_element, double epsilon = 1e-3)
{
  int64_t diff_count = 0;
  for (int64_t i = 0; i < total_element; i++) {
    T value     = input[i];
    T ref_value = input_ref[i];
    if (std::abs(value - ref_value) > epsilon) { diff_count++; }
    if (diff_count < 5 && diff_count > 0) {
      printf("index=%ld, got (float %f), but should be (float %f)\n", i, value, ref_value);
      fflush(stdout);
    }
  }
}

void host_check_float_tensor_same(void* input,
                                  wholememory_tensor_description_t input_tensor_desc,
                                  void* input_ref,
                                  wholememory_tensor_description_t input_ref_tensor_desc)
{
  EXPECT_EQ(input_tensor_desc.dtype, input_ref_tensor_desc.dtype);
  EXPECT_EQ(input_tensor_desc.dim, input_ref_tensor_desc.dim);
  int dim = input_tensor_desc.dim;
  for (int i = 0; i < dim; i++) {
    EXPECT_EQ(input_tensor_desc.sizes[i], input_ref_tensor_desc.sizes[i]);
  }
  int64_t total_ele_size = 1;
  for (int i = 0; i < dim; i++) {
    total_ele_size *= input_tensor_desc.sizes[i];
  }
  if (input_tensor_desc.dtype == WHOLEMEMORY_DT_FLOAT) {
    check_float_array_same<float>(
      static_cast<float*>(input), static_cast<float*>(input_ref), total_ele_size);
  } else if (input_tensor_desc.dtype == WHOLEMEMORY_DT_DOUBLE) {
    check_float_array_same<double>(
      static_cast<double*>(input), static_cast<double*>(input_ref), total_ele_size);
  }
}

template <typename WeightType>
void host_get_spmm_csr_no_weight_backward(int* host_csr_row_ptr,
                                          wholememory_array_description_t csr_row_ptr_desc,
                                          int* host_csr_col_ptr,
                                          wholememory_array_description_t csr_col_ptr_desc,
                                          void* host_input_grad_feature_ptr,
                                          wholememory_matrix_description_t input_grad_feature_desc,
                                          int aggregator,
                                          void* host_ref_output_grad_feature,
                                          wholememory_matrix_description_t output_feature_desc)
{
  WeightType* input_grad_feature_ptr  = static_cast<WeightType*>(host_input_grad_feature_ptr);
  WeightType* output_grad_feature_ptr = static_cast<WeightType*>(host_ref_output_grad_feature);
  int64_t emb_dim                     = input_grad_feature_desc.sizes[1];

  int64_t row_num = csr_row_ptr_desc.size - 1;
  for (int64_t i = 0; i < row_num; i++) {
    int start     = host_csr_row_ptr[i];
    int end       = host_csr_row_ptr[i + 1];
    float scale   = 1.0;
    int agg_count = end - start;
    if (aggregator == GCN_AGGREGATOR) {
      if (agg_count > 0) scale /= (end - start + 1);
    } else if (aggregator == MEAN_AGGREGATOR) {
      if (agg_count > 0) scale /= (end - start);
    }
    for (int emb_id = 0; emb_id < emb_dim; emb_id++) {
      WeightType value =
        static_cast<WeightType>(input_grad_feature_ptr[i * emb_dim + emb_id] * scale);
      if (aggregator == GCN_AGGREGATOR) { output_grad_feature_ptr[i * emb_dim + emb_id] += value; }
      for (int j = start; j < end; j++) {
        int col_id = host_csr_col_ptr[j];
        output_grad_feature_ptr[col_id * emb_dim + emb_id] += value;
      }
    }
  }
}
void host_spmm_csr_no_weight_backward(void* host_csr_row_ptr,
                                      wholememory_array_description_t csr_row_ptr_desc,
                                      void* host_csr_col_ptr,
                                      wholememory_array_description_t csr_col_ptr_desc,
                                      void* host_input_grad_feature_ptr,
                                      wholememory_matrix_description_t input_grad_feature_desc,
                                      int aggregator,
                                      void* host_ref_output_grad_feature,
                                      wholememory_matrix_description_t output_feature_desc)
{
  EXPECT_EQ(csr_row_ptr_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(csr_col_ptr_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(input_grad_feature_desc.dtype, output_feature_desc.dtype);
  EXPECT_EQ(output_feature_desc.sizes[1], input_grad_feature_desc.sizes[1]);
  memset(
    host_ref_output_grad_feature, 0, wholememory_get_memory_size_from_matrix(&output_feature_desc));

  if (input_grad_feature_desc.dtype == WHOLEMEMORY_DT_FLOAT) {
    host_get_spmm_csr_no_weight_backward<float>(static_cast<int*>(host_csr_row_ptr),
                                                csr_row_ptr_desc,
                                                static_cast<int*>(host_csr_col_ptr),
                                                csr_col_ptr_desc,
                                                host_input_grad_feature_ptr,
                                                input_grad_feature_desc,
                                                aggregator,
                                                host_ref_output_grad_feature,
                                                output_feature_desc);
  }
}

}  // namespace testing
}  // namespace graph_ops
