/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>
#include <wholememory/wholememory_op.h>

#include "parallel_utils.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/env_func_ptrs.hpp"
#include "wholememory/initialize.hpp"

#include "../wholememory/wholememory_test_utils.hpp"
#include "embedding_test_utils.hpp"

static int g_dev_count = 0;

typedef struct WholeMemoryGatherTestParam {
  wholememory_matrix_description_t get_embedding_desc() const
  {
    int64_t matrix_sizes[2] = {embedding_entry_count, embedding_dim};
    return wholememory_create_matrix_desc(
      matrix_sizes, embedding_stride, embedding_storage_offset, embedding_type);
  }
  wholememory_array_description_t get_indices_desc() const
  {
    return wholememory_create_array_desc(indices_count, indices_storage_offset, indices_type);
  }
  wholememory_matrix_description_t get_output_desc() const
  {
    int64_t output_sizes[2] = {indices_count, embedding_dim};
    return wholememory_create_matrix_desc(
      output_sizes, output_stride, output_storage_offset, output_type);
  }
  int64_t get_embedding_granularity() const
  {
    return embedding_stride * wholememory_dtype_get_element_size(embedding_type);
  }
  int get_rank_partition_method() const { return rank_partition_method; }
  WholeMemoryGatherTestParam& set_memory_type(wholememory_memory_type_t new_memory_type)
  {
    memory_type = new_memory_type;
    return *this;
  }
  WholeMemoryGatherTestParam& set_memory_location(wholememory_memory_location_t new_memory_location)
  {
    memory_location = new_memory_location;
    return *this;
  }
  WholeMemoryGatherTestParam& set_entry_count(int64_t entry_count)
  {
    embedding_entry_count = entry_count;
    return *this;
  }
  WholeMemoryGatherTestParam& set_embedding_dim(int64_t new_embedding_dim)
  {
    embedding_dim = new_embedding_dim;
    if (embedding_stride < embedding_dim) embedding_stride = embedding_dim;
    if (output_stride < embedding_dim) output_stride = embedding_dim;
    return *this;
  }
  WholeMemoryGatherTestParam& set_embedding_stride(int64_t new_embedding_stride)
  {
    embedding_stride = new_embedding_stride;
    return *this;
  }
  WholeMemoryGatherTestParam& set_output_stride(int64_t new_output_stride)
  {
    output_stride = new_output_stride;
    return *this;
  }
  WholeMemoryGatherTestParam& set_indices_count(int64_t new_indices_count)
  {
    indices_count = new_indices_count;
    return *this;
  }
  WholeMemoryGatherTestParam& set_embedding_type(wholememory_dtype_t new_embedding_type)
  {
    embedding_type = new_embedding_type;
    return *this;
  }
  WholeMemoryGatherTestParam& set_indices_type(wholememory_dtype_t new_indices_type)
  {
    indices_type = new_indices_type;
    return *this;
  }
  WholeMemoryGatherTestParam& set_output_type(wholememory_dtype_t new_output_type)
  {
    output_type = new_output_type;
    return *this;
  }
  WholeMemoryGatherTestParam& set_distributed_backend(
    wholememory_distributed_backend_t new_distributed_backend)
  {
    distributed_backend = new_distributed_backend;
    return *this;
  }
  WholeMemoryGatherTestParam& use_random_partition()
  {
    rank_partition_method = 1;
    return *this;
  }
  wholememory_memory_type_t memory_type                 = WHOLEMEMORY_MT_CHUNKED;
  wholememory_memory_location_t memory_location         = WHOLEMEMORY_ML_DEVICE;
  int64_t embedding_entry_count                         = 1000000LL;
  int64_t embedding_dim                                 = 32;
  int64_t embedding_stride                              = 32;
  int64_t indices_count                                 = 100000;
  int64_t output_stride                                 = 32;
  wholememory_dtype_t embedding_type                    = WHOLEMEMORY_DT_FLOAT;
  wholememory_dtype_t indices_type                      = WHOLEMEMORY_DT_INT;
  wholememory_dtype_t output_type                       = WHOLEMEMORY_DT_FLOAT;
  int64_t embedding_storage_offset                      = 0;
  int64_t indices_storage_offset                        = 0;
  int64_t output_storage_offset                         = 0;
  wholememory_distributed_backend_t distributed_backend = WHOLEMEMORY_DB_NCCL;
  int rank_partition_method                             = 0;  // 0-default, 1-random
} WholeMemoryGatherTestParam;

class WholeMemoryGatherParameterTests
  : public ::testing::TestWithParam<WholeMemoryGatherTestParam> {};

TEST_P(WholeMemoryGatherParameterTests, GatherTest)
{
  auto params = GetParam();
  EXPECT_GE(g_dev_count, 1);
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, g_dev_count);
  MultiProcessRun(
    g_dev_count,
    [&params, &pipes](int world_rank, int world_size) {
      EXPECT_EQ(wholememory_init(0), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(cudaSetDevice(world_rank), cudaSuccess);

      wholememory_comm_t wm_comm = create_communicator_by_pipes(pipes, world_rank, world_size);

#ifdef WITH_NVSHMEM_SUPPORT
      if (params.distributed_backend == WHOLEMEMORY_DB_NVSHMEM) {
        EXPECT_EQ(wholememory_communicator_set_distributed_backend(wm_comm, WHOLEMEMORY_DB_NVSHMEM),
                  WHOLEMEMORY_SUCCESS);
      }
#endif
      if (wholememory_communicator_support_type_location(
            wm_comm, params.memory_type, params.memory_location) != WHOLEMEMORY_SUCCESS) {
        EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);
        EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);
        WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
        if (world_rank == 0) GTEST_SKIP_("Skip due to not supported.");
        return;
      }

      wholememory_handle_t embedding_handle;

      auto embedding_desc         = params.get_embedding_desc();
      auto indices_desc           = params.get_indices_desc();
      auto output_desc            = params.get_output_desc();
      size_t embedding_entry_size = params.get_embedding_granularity();
      std::vector<size_t> rank_partition(world_size);
      wholememory_ops::testing::host_random_partition(
        rank_partition.data(), embedding_desc.sizes[0], world_size);
      size_t* rank_partition_ptr = nullptr;
      if (params.get_rank_partition_method() == 1) { rank_partition_ptr = rank_partition.data(); }
      EXPECT_EQ(wholememory_malloc(&embedding_handle,
                                   wholememory_get_memory_size_from_matrix(&embedding_desc),
                                   wm_comm,
                                   params.memory_type,
                                   params.memory_location,
                                   embedding_entry_size,
                                   rank_partition_ptr),
                WHOLEMEMORY_SUCCESS);
      cudaStream_t stream;
      EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

      void *dev_indices = nullptr, *dev_gather_buffer = nullptr, *dev_reference_buffer = nullptr;
      void *host_indices = nullptr, *host_gather_buffer = nullptr, *host_reference_buffer = nullptr;
      size_t gather_buffer_size  = wholememory_get_memory_size_from_matrix(&output_desc);
      size_t indices_buffer_size = wholememory_get_memory_size_from_array(&indices_desc);

      EXPECT_EQ(cudaMallocHost(&host_indices, indices_buffer_size), cudaSuccess);
      EXPECT_EQ(cudaMalloc(&dev_indices, indices_buffer_size), cudaSuccess);
      EXPECT_EQ(cudaMalloc(&dev_gather_buffer, gather_buffer_size), cudaSuccess);
      EXPECT_EQ(cudaMalloc(&dev_reference_buffer, gather_buffer_size), cudaSuccess);
      EXPECT_EQ(cudaMallocHost(&host_gather_buffer, gather_buffer_size), cudaSuccess);
      EXPECT_EQ(cudaMallocHost(&host_reference_buffer, gather_buffer_size), cudaSuccess);

      wholememory_ops::testing::device_random_init_local_embedding_table(
        embedding_handle, embedding_desc, stream);
      wholememory_ops::testing::host_random_init_indices(
        host_indices, indices_desc, embedding_desc.sizes[0]);
      EXPECT_EQ(cudaMemcpyAsync(dev_indices,
                                host_indices,
                                wholememory_get_memory_size_from_array(&indices_desc),
                                cudaMemcpyHostToDevice,
                                stream),
                cudaSuccess);

      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      wholememory_communicator_barrier(wm_comm);

      wholememory_tensor_t embedding_tensor;
      wholememory_tensor_description_t embedding_tensor_desc;
      wholememory_copy_matrix_desc_to_tensor(&embedding_tensor_desc, &embedding_desc);
      EXPECT_EQ(wholememory_make_tensor_from_handle(
                  &embedding_tensor, embedding_handle, &embedding_tensor_desc),
                WHOLEMEMORY_SUCCESS);

      wholememory_tensor_t indices_tensor, output_tensor;
      wholememory_tensor_description_t indices_tensor_desc, output_tensor_desc;
      wholememory_copy_array_desc_to_tensor(&indices_tensor_desc, &indices_desc);
      wholememory_copy_matrix_desc_to_tensor(&output_tensor_desc, &output_desc);
      EXPECT_EQ(
        wholememory_make_tensor_from_pointer(&indices_tensor, dev_indices, &indices_tensor_desc),
        WHOLEMEMORY_SUCCESS);
      EXPECT_EQ(wholememory_make_tensor_from_pointer(
                  &output_tensor, dev_gather_buffer, &output_tensor_desc),
                WHOLEMEMORY_SUCCESS);
      EXPECT_EQ(wholememory_gather(embedding_tensor,
                                   indices_tensor,
                                   output_tensor,
                                   wholememory::get_default_env_func(),
                                   stream),
                WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(cudaGetLastError(), cudaSuccess);
      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      EXPECT_EQ(wholememory_destroy_tensor(indices_tensor), WHOLEMEMORY_SUCCESS);
      EXPECT_EQ(wholememory_destroy_tensor(output_tensor), WHOLEMEMORY_SUCCESS);

      wholememory_ops::testing::device_get_expected_embedding(dev_reference_buffer,
                                                              output_desc,
                                                              embedding_desc.dtype,
                                                              dev_indices,
                                                              indices_desc,
                                                              wholememory::get_default_env_func(),
                                                              stream);
      EXPECT_EQ(cudaMemcpyAsync(host_gather_buffer,
                                dev_gather_buffer,
                                wholememory_get_memory_size_from_matrix(&output_desc),
                                cudaMemcpyDeviceToHost,
                                stream),
                cudaSuccess);
      EXPECT_EQ(cudaMemcpyAsync(host_reference_buffer,
                                dev_reference_buffer,
                                wholememory_get_memory_size_from_matrix(&output_desc),
                                cudaMemcpyDeviceToHost,
                                stream),
                cudaSuccess);
      EXPECT_EQ(cudaGetLastError(), cudaSuccess);
      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

      wholememory_ops::testing::host_check_embedding_same(
        host_gather_buffer, output_desc, host_reference_buffer, output_desc);

      EXPECT_EQ(cudaFreeHost(host_indices), cudaSuccess);
      EXPECT_EQ(cudaFree(dev_indices), cudaSuccess);
      EXPECT_EQ(cudaFree(dev_gather_buffer), cudaSuccess);
      EXPECT_EQ(cudaFree(dev_reference_buffer), cudaSuccess);
      EXPECT_EQ(cudaFreeHost(host_gather_buffer), cudaSuccess);
      EXPECT_EQ(cudaFreeHost(host_reference_buffer), cudaSuccess);

      EXPECT_EQ(wholememory_destroy_tensor(embedding_tensor), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(wholememory_free(embedding_handle), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);
      WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
    },
    true);
}

INSTANTIATE_TEST_SUITE_P(
  WholeMemoryGatherOpTests,
  WholeMemoryGatherParameterTests,
  ::testing::Values(
#if 0
    WholeMemoryGatherTestParam()
      .set_memory_location(WHOLEMEMORY_ML_DEVICE)
      .set_indices_type(WHOLEMEMORY_DT_INT64)
      .set_entry_count((1LL << 23LL) + 131)
      .set_embedding_dim(1024)
      .set_indices_count(100005),
#endif
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_HIERARCHY),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_indices_count(0),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_indices_count(0),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_indices_count(0),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_HIERARCHY).set_indices_count(0),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CONTINUOUS)
      .set_memory_location(WHOLEMEMORY_ML_HOST),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CHUNKED)
      .set_memory_location(WHOLEMEMORY_ML_HOST),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_memory_location(WHOLEMEMORY_ML_HOST),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_HIERARCHY)
      .set_memory_location(WHOLEMEMORY_ML_HOST),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).use_random_partition(),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).use_random_partition(),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_HIERARCHY).use_random_partition(),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_memory_location(WHOLEMEMORY_ML_HOST)
      .use_random_partition(),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_HIERARCHY)
      .set_memory_location(WHOLEMEMORY_ML_HOST)
      .use_random_partition(),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CONTINUOUS)
      .set_memory_location(WHOLEMEMORY_ML_HOST)
      .set_embedding_dim(1)
      .set_indices_type(WHOLEMEMORY_DT_INT64),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CHUNKED)
      .set_memory_location(WHOLEMEMORY_ML_HOST)
      .set_embedding_dim(1)
      .set_indices_type(WHOLEMEMORY_DT_INT64),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CONTINUOUS)
      .set_embedding_dim(11)
      .set_embedding_stride(12)
      .set_indices_count(100005),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CHUNKED)
      .set_embedding_dim(11)
      .set_embedding_stride(12)
      .set_indices_count(100005),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CHUNKED)
      .set_embedding_dim(1)
      .set_embedding_stride(1)
      .set_indices_count(100005),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CHUNKED)
      .set_embedding_dim(1)
      .set_embedding_stride(2)
      .set_indices_count(100005),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_embedding_dim(11)
      .set_embedding_stride(12)
      .set_indices_count(100005),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_HIERARCHY)
      .set_embedding_dim(11)
      .set_embedding_stride(12)
      .set_indices_count(100005),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_embedding_dim(128),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_embedding_dim(128),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_embedding_dim(128),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_HIERARCHY).set_embedding_dim(128),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_embedding_dim(127),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_embedding_dim(127),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_embedding_dim(127),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_HIERARCHY).set_embedding_dim(127),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_embedding_dim(129),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_embedding_dim(129),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_embedding_dim(129),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_HIERARCHY).set_embedding_dim(129),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_embedding_dim(513),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_embedding_dim(513),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_embedding_dim(513),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_HIERARCHY).set_embedding_dim(513),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CONTINUOUS)
      .set_embedding_type(WHOLEMEMORY_DT_HALF),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CHUNKED)
      .set_embedding_type(WHOLEMEMORY_DT_HALF),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_embedding_type(WHOLEMEMORY_DT_HALF),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CONTINUOUS)
      .set_output_type(WHOLEMEMORY_DT_HALF),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CHUNKED)
      .set_output_type(WHOLEMEMORY_DT_HALF),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_output_type(WHOLEMEMORY_DT_HALF),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_HIERARCHY)
      .set_output_type(WHOLEMEMORY_DT_HALF),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CONTINUOUS)
      .set_embedding_type(WHOLEMEMORY_DT_HALF)
      .set_output_type(WHOLEMEMORY_DT_HALF),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CHUNKED)
      .set_embedding_type(WHOLEMEMORY_DT_HALF)
      .set_output_type(WHOLEMEMORY_DT_HALF),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_embedding_type(WHOLEMEMORY_DT_HALF)
      .set_output_type(WHOLEMEMORY_DT_HALF),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_HIERARCHY)
      .set_embedding_type(WHOLEMEMORY_DT_HALF)
      .set_output_type(WHOLEMEMORY_DT_HALF),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CONTINUOUS)
      .set_indices_type(WHOLEMEMORY_DT_INT64),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CHUNKED)
      .set_indices_type(WHOLEMEMORY_DT_INT64),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_indices_type(WHOLEMEMORY_DT_INT64),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_HIERARCHY)
      .set_indices_type(WHOLEMEMORY_DT_INT64),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CONTINUOUS)
      .set_embedding_stride(33),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_embedding_stride(33),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_embedding_stride(33),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_HIERARCHY).set_embedding_stride(33),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CONTINUOUS).set_output_stride(33),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_CHUNKED).set_output_stride(33),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED).set_output_stride(33),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_HIERARCHY).set_output_stride(33),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CONTINUOUS)
      .set_embedding_type(WHOLEMEMORY_DT_HALF)
      .set_embedding_stride(33),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_CHUNKED)
      .set_embedding_type(WHOLEMEMORY_DT_HALF)
      .set_embedding_stride(33),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_embedding_type(WHOLEMEMORY_DT_HALF)
      .set_embedding_stride(33),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_HIERARCHY)
      .set_embedding_type(WHOLEMEMORY_DT_HALF)
      .set_embedding_stride(33),
    WholeMemoryGatherTestParam().set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
#ifdef WITH_NVSHMEM_SUPPORT
      ,
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_distributed_backend(WHOLEMEMORY_DB_NVSHMEM),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_distributed_backend(WHOLEMEMORY_DB_NVSHMEM)
      .use_random_partition(),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_embedding_dim(11)
      .set_embedding_stride(12)
      .set_indices_count(100005)
      .set_distributed_backend(WHOLEMEMORY_DB_NVSHMEM),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_embedding_dim(128)
      .set_distributed_backend(WHOLEMEMORY_DB_NVSHMEM),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_embedding_dim(127)
      .set_distributed_backend(WHOLEMEMORY_DB_NVSHMEM),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_embedding_dim(129)
      .set_distributed_backend(WHOLEMEMORY_DB_NVSHMEM),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_embedding_dim(513)
      .set_distributed_backend(WHOLEMEMORY_DB_NVSHMEM),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_distributed_backend(WHOLEMEMORY_DB_NVSHMEM)
      .set_embedding_type(WHOLEMEMORY_DT_HALF),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_distributed_backend(WHOLEMEMORY_DB_NVSHMEM)
      .set_output_type(WHOLEMEMORY_DT_HALF),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_distributed_backend(WHOLEMEMORY_DB_NVSHMEM)
      .set_embedding_type(WHOLEMEMORY_DT_HALF)
      .set_output_type(WHOLEMEMORY_DT_HALF),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_distributed_backend(WHOLEMEMORY_DB_NVSHMEM)
      .set_indices_type(WHOLEMEMORY_DT_INT64),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_embedding_stride(33)
      .set_distributed_backend(WHOLEMEMORY_DB_NVSHMEM),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_output_stride(33)
      .set_distributed_backend(WHOLEMEMORY_DB_NVSHMEM),
    WholeMemoryGatherTestParam()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_embedding_type(WHOLEMEMORY_DT_HALF)
      .set_embedding_stride(33)
      .set_distributed_backend(WHOLEMEMORY_DB_NVSHMEM)
#endif
      ));

class GlobalEnvironment : public ::testing::Environment {
 public:
  void SetUp() override { g_dev_count = ForkGetDeviceCount(); }
  void TearDown() override {}
};

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);

  ::testing::AddGlobalTestEnvironment(new GlobalEnvironment);

  return RUN_ALL_TESTS();
}
