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
#include <gtest/gtest.h>
#include <random>

#include <wholememory/tensor_description.h>
#include <wholememory/wholegraph_op.h>
#include <wholememory/wholememory.h>

#include "parallel_utils.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/env_func_ptrs.hpp"
#include "wholememory/initialize.hpp"

#include "../wholememory/wholememory_test_utils.hpp"
#include "graph_sampling_test_utils.hpp"

typedef struct WholeGraphCSRUnweightedSampleWithoutReplacementTestParam {
  wholememory_array_description_t get_csr_row_ptr_desc() const
  {
    return wholememory_create_array_desc(graph_node_count + 1, 0, csr_row_ptr_dtype);
  }

  wholememory_array_description_t get_csr_col_ptr_desc() const
  {
    return wholememory_create_array_desc(graph_edge_count, 0, csr_col_ptr_dtype);
  }

  wholememory_array_description_t get_center_node_desc() const
  {
    return wholememory_create_array_desc(center_node_count, 0, center_node_dtype);
  }

  wholememory_array_description_t get_output_sample_offset_desc() const
  {
    return wholememory_create_array_desc(center_node_count + 1, 0, output_sample_offset_dtype);
  }

  int64_t get_graph_node_count() const { return graph_node_count; }
  int64_t get_graph_edge_count() const { return graph_edge_count; }
  int64_t get_max_sample_count() const { return max_sample_count; }

  WholeGraphCSRUnweightedSampleWithoutReplacementTestParam& set_memory_type(
    wholememory_memory_type_t new_memory_type)
  {
    memory_type = new_memory_type;
    return *this;
  };
  WholeGraphCSRUnweightedSampleWithoutReplacementTestParam& set_memory_location(
    wholememory_memory_location_t new_memory_location)
  {
    memory_location = new_memory_location;
    return *this;
  };

  WholeGraphCSRUnweightedSampleWithoutReplacementTestParam& set_max_sample_count(
    int new_sample_count)
  {
    max_sample_count = new_sample_count;
    return *this;
  }

  WholeGraphCSRUnweightedSampleWithoutReplacementTestParam& set_center_node_count(
    int new_center_node_count)
  {
    center_node_count = new_center_node_count;
    return *this;
  }
  WholeGraphCSRUnweightedSampleWithoutReplacementTestParam& set_graph_node_count(
    int new_graph_node_count)
  {
    graph_node_count = new_graph_node_count;
    return *this;
  }
  WholeGraphCSRUnweightedSampleWithoutReplacementTestParam& set_graph_edge_couont(
    int new_graph_edge_count)
  {
    graph_edge_count = new_graph_edge_count;
    return *this;
  }

  WholeGraphCSRUnweightedSampleWithoutReplacementTestParam& set_center_node_type(
    wholememory_dtype_t new_center_node_dtype)
  {
    center_node_dtype = new_center_node_dtype;
    return *this;
  }

  wholememory_memory_type_t memory_type                 = WHOLEMEMORY_MT_CHUNKED;
  wholememory_memory_location_t memory_location         = WHOLEMEMORY_ML_DEVICE;
  int64_t max_sample_count                              = 50;
  int64_t center_node_count                             = 512;
  int64_t graph_node_count                              = 9703LL;
  int64_t graph_edge_count                              = 104323L;
  wholememory_dtype_t csr_row_ptr_dtype                 = WHOLEMEMORY_DT_INT64;
  wholememory_dtype_t csr_col_ptr_dtype                 = WHOLEMEMORY_DT_INT;
  wholememory_dtype_t center_node_dtype                 = WHOLEMEMORY_DT_INT;
  wholememory_dtype_t output_sample_offset_dtype        = WHOLEMEMORY_DT_INT;
  wholememory_dtype_t output_dest_node_dtype            = center_node_dtype;
  wholememory_dtype_t output_center_node_local_id_dtype = WHOLEMEMORY_DT_INT;
  wholememory_dtype_t output_globla_edge_id_dtype       = WHOLEMEMORY_DT_INT64;

} WholeGraphCSRUnweightedSampleWithoutReplacementTestParam;

class WholeGraphCSRUnweightedSampleWithoutReplacementParameterTests
  : public ::testing::TestWithParam<WholeGraphCSRUnweightedSampleWithoutReplacementTestParam> {};

TEST_P(WholeGraphCSRUnweightedSampleWithoutReplacementParameterTests, UnWeightedSampleTest)
{
  auto params   = GetParam();
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  auto graph_node_count       = params.get_graph_node_count();
  auto graph_edge_count       = params.get_graph_edge_count();
  auto graph_csr_row_ptr_desc = params.get_csr_row_ptr_desc();
  auto graph_csr_col_ptr_desc = params.get_csr_col_ptr_desc();

  void* host_csr_row_ptr =
    (void*)malloc(wholememory_get_memory_size_from_array(&graph_csr_row_ptr_desc));
  void* host_csr_col_ptr =
    (void*)malloc(wholememory_get_memory_size_from_array(&graph_csr_col_ptr_desc));

  wholegraph_ops::testing::gen_csr_graph(graph_node_count,
                                         graph_edge_count,
                                         host_csr_row_ptr,
                                         graph_csr_row_ptr_desc,
                                         host_csr_col_ptr,
                                         graph_csr_col_ptr_desc);

  MultiProcessRun(
    dev_count,
    [&params, &pipes, host_csr_row_ptr, host_csr_col_ptr](int world_rank, int world_size) {
      thread_local std::random_device rd;
      thread_local std::mt19937 gen(rd());
      thread_local std::uniform_int_distribution<unsigned long long> distrib;
      unsigned long long random_seed = distrib(gen);

      EXPECT_EQ(wholememory_init(0), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(cudaSetDevice(world_rank), cudaSuccess);

      wholememory_comm_t wm_comm = create_communicator_by_pipes(pipes, world_rank, world_size);

      if (wholememory_communicator_support_type_location(
            wm_comm, params.memory_type, params.memory_location) != WHOLEMEMORY_SUCCESS) {
        EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);
        EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);
        WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
        if (world_rank == 0) GTEST_SKIP_("Skip due to not supported.");
        return;
      }

      auto csr_row_ptr_desc          = params.get_csr_row_ptr_desc();
      auto csr_col_ptr_desc          = params.get_csr_col_ptr_desc();
      auto center_node_desc          = params.get_center_node_desc();
      auto output_sample_offset_desc = params.get_output_sample_offset_desc();
      auto max_sample_count          = params.get_max_sample_count();
      int64_t graph_node_count       = params.get_graph_node_count();
      int64_t graph_edge_count       = params.get_graph_edge_count();

      size_t center_node_size = wholememory_get_memory_size_from_array(&center_node_desc);
      size_t output_sample_offset_size =
        wholememory_get_memory_size_from_array(&output_sample_offset_desc);

      cudaStream_t stream;
      EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

      void *host_ref_output_sample_offset, *host_ref_output_dest_nodes,
        *host_ref_output_center_nodes_local_id, *host_ref_output_global_edge_id;

      void *host_center_nodes, *host_output_sample_offset, *host_output_dest_nodes,
        *host_output_center_nodes_local_id, *host_output_global_edge_id;
      void *dev_center_nodes, *dev_output_sample_offset;

      wholememory_handle_t csr_row_ptr_memory_handle;
      wholememory_handle_t csr_col_ptr_memory_handle;

      EXPECT_EQ(wholememory_malloc(&csr_row_ptr_memory_handle,
                                   wholememory_get_memory_size_from_array(&csr_row_ptr_desc),
                                   wm_comm,
                                   params.memory_type,
                                   params.memory_location,
                                   wholememory_dtype_get_element_size(csr_row_ptr_desc.dtype)),
                WHOLEMEMORY_SUCCESS);
      EXPECT_EQ(wholememory_malloc(&csr_col_ptr_memory_handle,
                                   wholememory_get_memory_size_from_array(&csr_col_ptr_desc),
                                   wm_comm,
                                   params.memory_type,
                                   params.memory_location,
                                   wholememory_dtype_get_element_size(csr_col_ptr_desc.dtype)),
                WHOLEMEMORY_SUCCESS);

      wholegraph_ops::testing::copy_host_array_to_wholememory(
        host_csr_row_ptr, csr_row_ptr_memory_handle, csr_row_ptr_desc, stream);
      wholegraph_ops::testing::copy_host_array_to_wholememory(
        host_csr_col_ptr, csr_col_ptr_memory_handle, csr_col_ptr_desc, stream);

      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      wholememory_communicator_barrier(wm_comm);

      EXPECT_EQ(cudaSetDevice(world_rank), cudaSuccess);
      EXPECT_EQ(cudaMallocHost(&host_center_nodes, center_node_size), cudaSuccess);
      EXPECT_EQ(cudaMallocHost(&host_output_sample_offset, output_sample_offset_size), cudaSuccess);

      EXPECT_EQ(cudaMalloc(&dev_center_nodes, center_node_size), cudaSuccess);
      EXPECT_EQ(cudaMalloc(&dev_output_sample_offset, output_sample_offset_size), cudaSuccess);

      wholegraph_ops::testing::host_random_init_array(
        host_center_nodes, center_node_desc, 0, graph_node_count - 1);
      EXPECT_EQ(cudaMemcpyAsync(dev_center_nodes,
                                host_center_nodes,
                                wholememory_get_memory_size_from_array(&center_node_desc),
                                cudaMemcpyHostToDevice,
                                stream),
                cudaSuccess);

      wholememory_tensor_t wm_csr_row_ptr_tensor, wm_csr_col_ptr_tensor;
      wholememory_tensor_description_t wm_csr_row_ptr_tensor_desc, wm_csr_col_ptr_tensor_desc;
      wholememory_copy_array_desc_to_tensor(&wm_csr_row_ptr_tensor_desc, &csr_row_ptr_desc);
      wholememory_copy_array_desc_to_tensor(&wm_csr_col_ptr_tensor_desc, &csr_col_ptr_desc);
      EXPECT_EQ(wholememory_make_tensor_from_handle(
                  &wm_csr_row_ptr_tensor, csr_row_ptr_memory_handle, &wm_csr_row_ptr_tensor_desc),
                WHOLEMEMORY_SUCCESS);
      EXPECT_EQ(wholememory_make_tensor_from_handle(
                  &wm_csr_col_ptr_tensor, csr_col_ptr_memory_handle, &wm_csr_col_ptr_tensor_desc),
                WHOLEMEMORY_SUCCESS);

      wholememory_tensor_t center_nodes_tensor, output_sample_offset_tensor;
      wholememory_tensor_description_t center_nodes_tensor_desc, output_sample_offset_tensor_desc;
      wholememory_copy_array_desc_to_tensor(&center_nodes_tensor_desc, &center_node_desc);
      wholememory_copy_array_desc_to_tensor(&output_sample_offset_tensor_desc,
                                            &output_sample_offset_desc);
      EXPECT_EQ(wholememory_make_tensor_from_pointer(
                  &center_nodes_tensor, dev_center_nodes, &center_nodes_tensor_desc),
                WHOLEMEMORY_SUCCESS);
      EXPECT_EQ(wholememory_make_tensor_from_pointer(&output_sample_offset_tensor,
                                                     dev_output_sample_offset,
                                                     &output_sample_offset_tensor_desc),
                WHOLEMEMORY_SUCCESS);

      wholememory_env_func_t* default_env_func = wholememory::get_default_env_func();
      wholememory::default_memory_context_t output_dest_mem_ctx, output_center_localid_mem_ctx,
        output_edge_gid_mem_ctx;

      EXPECT_EQ(wholegraph_csr_unweighted_sample_without_replacement(wm_csr_row_ptr_tensor,
                                                                     wm_csr_col_ptr_tensor,
                                                                     center_nodes_tensor,
                                                                     max_sample_count,
                                                                     output_sample_offset_tensor,
                                                                     &output_dest_mem_ctx,
                                                                     &output_center_localid_mem_ctx,
                                                                     &output_edge_gid_mem_ctx,
                                                                     random_seed,
                                                                     default_env_func,
                                                                     stream),
                WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(cudaGetLastError(), cudaSuccess);
      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      wholememory_communicator_barrier(wm_comm);

      EXPECT_EQ(output_dest_mem_ctx.desc.dim, 1);
      EXPECT_EQ(output_center_localid_mem_ctx.desc.dim, 1);
      EXPECT_EQ(output_edge_gid_mem_ctx.desc.dim, 1);

      EXPECT_EQ(output_dest_mem_ctx.desc.dtype, csr_col_ptr_desc.dtype);
      EXPECT_EQ(output_center_localid_mem_ctx.desc.dtype, WHOLEMEMORY_DT_INT);
      EXPECT_EQ(output_edge_gid_mem_ctx.desc.dtype, WHOLEMEMORY_DT_INT64);

      EXPECT_EQ(output_dest_mem_ctx.desc.sizes[0], output_center_localid_mem_ctx.desc.sizes[0]);
      EXPECT_EQ(output_dest_mem_ctx.desc.sizes[0], output_edge_gid_mem_ctx.desc.sizes[0]);

      int64_t total_sample_count = output_dest_mem_ctx.desc.sizes[0];

      host_output_dest_nodes =
        malloc(total_sample_count * wholememory_dtype_get_element_size(csr_col_ptr_desc.dtype));
      host_output_center_nodes_local_id = malloc(total_sample_count * sizeof(int));
      host_output_global_edge_id        = malloc(total_sample_count * sizeof(int64_t));

      EXPECT_EQ(cudaMemcpyAsync(host_output_sample_offset,
                                dev_output_sample_offset,
                                output_sample_offset_size,
                                cudaMemcpyDeviceToHost,
                                stream),
                cudaSuccess);
      EXPECT_EQ(cudaMemcpyAsync(
                  host_output_dest_nodes,
                  output_dest_mem_ctx.ptr,
                  total_sample_count * wholememory_dtype_get_element_size(csr_col_ptr_desc.dtype),
                  cudaMemcpyDeviceToHost,
                  stream),
                cudaSuccess);
      EXPECT_EQ(cudaMemcpyAsync(host_output_center_nodes_local_id,
                                output_center_localid_mem_ctx.ptr,
                                total_sample_count * sizeof(int),
                                cudaMemcpyDeviceToHost,
                                stream),
                cudaSuccess);
      EXPECT_EQ(cudaMemcpyAsync(host_output_global_edge_id,
                                output_edge_gid_mem_ctx.ptr,
                                total_sample_count * sizeof(int64_t),
                                cudaMemcpyDeviceToHost,
                                stream),
                cudaSuccess);

      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      wholememory_communicator_barrier(wm_comm);

      int host_total_sample_count;
      wholegraph_ops::testing::wholegraph_csr_unweighted_sample_without_replacement_cpu(
        host_csr_row_ptr,
        csr_row_ptr_desc,
        host_csr_col_ptr,
        csr_col_ptr_desc,
        host_center_nodes,
        center_node_desc,
        max_sample_count,
        &host_ref_output_sample_offset,
        output_sample_offset_desc,
        &host_ref_output_dest_nodes,
        &host_ref_output_center_nodes_local_id,
        &host_ref_output_global_edge_id,
        &host_total_sample_count,
        random_seed);

      EXPECT_EQ(total_sample_count, host_total_sample_count);

      wholegraph_ops::testing::host_check_two_array_same(host_output_sample_offset,
                                                         output_sample_offset_desc,
                                                         host_ref_output_sample_offset,
                                                         output_sample_offset_desc);
      wholegraph_ops::testing::host_check_two_array_same(
        host_output_dest_nodes,
        wholememory_create_array_desc(host_total_sample_count, 0, csr_col_ptr_desc.dtype),
        host_ref_output_dest_nodes,
        wholememory_create_array_desc(host_total_sample_count, 0, csr_col_ptr_desc.dtype));

      wholegraph_ops::testing::host_check_two_array_same(
        host_output_center_nodes_local_id,
        wholememory_create_array_desc(host_total_sample_count, 0, WHOLEMEMORY_DT_INT),
        host_ref_output_center_nodes_local_id,
        wholememory_create_array_desc(host_total_sample_count, 0, WHOLEMEMORY_DT_INT));

      wholegraph_ops::testing::host_check_two_array_same(
        host_output_global_edge_id,
        wholememory_create_array_desc(host_total_sample_count, 0, WHOLEMEMORY_DT_INT64),
        host_ref_output_global_edge_id,
        wholememory_create_array_desc(host_total_sample_count, 0, WHOLEMEMORY_DT_INT64));

      (default_env_func->output_fns).free_fn(&output_dest_mem_ctx, nullptr);
      (default_env_func->output_fns).free_fn(&output_center_localid_mem_ctx, nullptr);
      (default_env_func->output_fns).free_fn(&output_edge_gid_mem_ctx, nullptr);

      EXPECT_EQ(wholememory_free(csr_row_ptr_memory_handle), WHOLEMEMORY_SUCCESS);
      EXPECT_EQ(wholememory_free(csr_col_ptr_memory_handle), WHOLEMEMORY_SUCCESS);

      if (host_ref_output_sample_offset != nullptr) free(host_ref_output_sample_offset);
      if (host_ref_output_dest_nodes != nullptr) free(host_ref_output_dest_nodes);
      if (host_ref_output_center_nodes_local_id != nullptr)
        free(host_ref_output_center_nodes_local_id);
      if (host_ref_output_global_edge_id != nullptr) free(host_ref_output_global_edge_id);

      EXPECT_EQ(cudaFreeHost(host_center_nodes), cudaSuccess);
      EXPECT_EQ(cudaFreeHost(host_output_sample_offset), cudaSuccess);
      EXPECT_EQ(cudaFree(dev_center_nodes), cudaSuccess);
      EXPECT_EQ(cudaFree(dev_output_sample_offset), cudaSuccess);
      EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);
      WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
    },
    true);

  if (host_csr_row_ptr != nullptr) free(host_csr_row_ptr);
  if (host_csr_col_ptr != nullptr) free(host_csr_col_ptr);
}

INSTANTIATE_TEST_SUITE_P(
  WholeGraphCSRUnweightedSampleWithoutReplacementOpTests,
  WholeGraphCSRUnweightedSampleWithoutReplacementParameterTests,
  ::testing::Values(WholeGraphCSRUnweightedSampleWithoutReplacementTestParam().set_memory_type(
                      WHOLEMEMORY_MT_CONTINUOUS),
                    WholeGraphCSRUnweightedSampleWithoutReplacementTestParam().set_memory_type(
                      WHOLEMEMORY_MT_CHUNKED),
                    WholeGraphCSRUnweightedSampleWithoutReplacementTestParam()
                      .set_memory_type(WHOLEMEMORY_MT_CONTINUOUS)
                      .set_memory_location(WHOLEMEMORY_ML_HOST),
                    WholeGraphCSRUnweightedSampleWithoutReplacementTestParam()
                      .set_memory_type(WHOLEMEMORY_MT_CHUNKED)
                      .set_memory_location(WHOLEMEMORY_ML_HOST),
                    WholeGraphCSRUnweightedSampleWithoutReplacementTestParam()
                      .set_memory_type(WHOLEMEMORY_MT_CONTINUOUS)
                      .set_max_sample_count(10)
                      .set_center_node_count(35)
                      .set_graph_node_count(23289)
                      .set_graph_edge_couont(689403),
                    WholeGraphCSRUnweightedSampleWithoutReplacementTestParam()
                      .set_memory_type(WHOLEMEMORY_MT_CHUNKED)
                      .set_center_node_type(WHOLEMEMORY_DT_INT64)));
