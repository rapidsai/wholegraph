#pragma once

#include "cuda_macros.hpp"
#include "wholememory/env_func_ptrs.h"
#include "wholememory/tensor_description.h"
#include "wholememory_ops/register.hpp"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace wholememory_ops {
template <typename _T>
struct pair {
  _T value;
  _T raw_idx;
};

template <typename IndexT>

__global__ void generate_pair_for_index_kernel(const IndexT* indices,
                                               int64_t indice_count,
                                               pair<IndexT>* indices_pair)
{
  for (int64_t id = threadIdx.x + blockDim.x * blockIdx.x; id < indice_count;
       id += (static_cast<int64_t>(gridDim.x) * blockDim.x)) {
    IndexT value     = indices[id];
    IndexT raw_idx   = static_cast<IndexT>(id);
    indices_pair[id] = pair<IndexT>{value, raw_idx};
  }
}

template <typename IndexT>
struct partition_op {
  IndexT partition_min;
  IndexT partition_max;
  __device__ __host__ __forceinline__ explicit partition_op(IndexT partition_min,
                                                            IndexT partition_max)
    : partition_min(partition_min), partition_max(partition_max)
  {
  }
  __device__ __host__ __forceinline__ bool operator()(const pair<IndexT>& a) const
  {
    return !((a.value >= partition_min) && (a.value < partition_max));
  }
};

// min<=value <max
template <typename IndexT>
void generate_partition_pair_index(const IndexT* indices,
                                   int64_t indice_count,
                                   IndexT partition_min,
                                   IndexT partition_max,
                                   pair<IndexT>* output_partition_idx_pair,
                                   int64_t* num_selected,
                                   wm_thrust_allocator* p_thrust_allocator,

                                   wholememory_env_func_t* p_env_fns,
                                   cudaStream_t stream)
{
  //  wm_thrust_allocator thrust_allocator(p_env_fns);
  wm_thrust_allocator& allocator = *p_thrust_allocator;
  temp_memory_handle temp_indices_pair_handle(p_env_fns);
  temp_memory_handle d_num_selected_handle(p_env_fns);

  pair<IndexT>* temp_index_pair_ptr = static_cast<pair<IndexT>*>(
    temp_indices_pair_handle.device_malloc(2 * indice_count, get_wholememory_dtype<IndexT>()));
  int64_t* d_num_selected =
    static_cast<int64_t*>(d_num_selected_handle.device_malloc(1, WHOLEMEMORY_DT_INT64));
  int threads             = 128;
  int64_t block_count_128 = (indice_count + threads - 1) / threads;
  int block_count =
    ((block_count_128 >= INT_MAX) ? INT_MAX / 4 : static_cast<int>(block_count_128));

  generate_pair_for_index_kernel<<<block_count, threads, 0, stream>>>(
    indices, indice_count, temp_index_pair_ptr);

  void* cub_temp_storage    = nullptr;
  size_t temp_storage_bytes = 0;
  partition_op<IndexT> partition_op_instance{partition_min, partition_max};
  cub::DevicePartition::If(cub_temp_storage,
                           temp_storage_bytes,
                           temp_index_pair_ptr,
                           output_partition_idx_pair,
                           d_num_selected,
                           indice_count,
                           partition_op_instance,
                           stream);
  cub_temp_storage = allocator.allocate(temp_storage_bytes);
  cub::DevicePartition::If(cub_temp_storage,
                           temp_storage_bytes,
                           temp_index_pair_ptr,
                           output_partition_idx_pair,
                           d_num_selected,
                           indice_count,
                           partition_op_instance,
                           stream);

  WM_CUDA_CHECK_NO_THROW(
    cudaMemcpyAsync(num_selected, d_num_selected, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
  WM_CUDA_CHECK_NO_THROW(cudaStreamSynchronize(stream));
}

}  // namespace wholememory_ops
