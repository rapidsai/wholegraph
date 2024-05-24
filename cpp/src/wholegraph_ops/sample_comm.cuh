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
#pragma once
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>

#include <wholememory/device_reference.cuh>
#include <wholememory/env_func_ptrs.h>
#include <wholememory/global_reference.h>
#include <wholememory/tensor_description.h>

namespace wholegraph_ops {
template <typename IdType, typename LocalIdType, typename WMIdType, typename WMOffsetType>
__global__ void sample_all_kernel(wholememory_gref_t wm_csr_row_ptr,
                                  wholememory_array_description_t wm_csr_row_ptr_desc,
                                  wholememory_gref_t wm_csr_col_ptr,
                                  wholememory_array_description_t wm_csr_col_ptr_desc,
                                  const IdType* input_nodes,
                                  const int input_node_count,
                                  const int* output_sample_offset,
                                  wholememory_array_description_t output_sample_offset_desc,
                                  WMIdType* output_dest_node_ptr,
                                  int* output_center_localid_ptr,
                                  int64_t* output_edge_gid_ptr)
{
  int input_idx = blockIdx.x;
  if (input_idx >= input_node_count) return;

  wholememory::device_reference<WMOffsetType> wm_csr_row_ptr_dev_ref(wm_csr_row_ptr);
  wholememory::device_reference<WMIdType> wm_csr_col_ptr_ref(wm_csr_col_ptr);

  IdType nid         = input_nodes[input_idx];
  int64_t start      = wm_csr_row_ptr_dev_ref[nid];
  int64_t end        = wm_csr_row_ptr_dev_ref[nid + 1];
  int neighbor_count = (int)(end - start);
  if (neighbor_count <= 0) return;
  int offset = output_sample_offset[input_idx];
  for (int sample_id = threadIdx.x; sample_id < neighbor_count; sample_id += blockDim.x) {
    int neighbor_idx                         = sample_id;
    IdType gid                               = wm_csr_col_ptr_ref[start + neighbor_idx];
    output_dest_node_ptr[offset + sample_id] = gid;
    if (output_center_localid_ptr)
      output_center_localid_ptr[offset + sample_id] = (LocalIdType)input_idx;
    if (output_edge_gid_ptr) {
      output_edge_gid_ptr[offset + sample_id] = (int64_t)(start + neighbor_idx);
    }
  }
}

__device__ __forceinline__ int log2_up_device(int x)
{
  if (x <= 2) return x - 1;
  return 32 - __clz(x - 1);
}
template <typename IdType>
struct ExpandWithOffsetFunc {
  const IdType* indptr;
  IdType* indptr_shift;
  int length;
  __host__ __device__ auto operator()(int64_t tIdx) {
    indptr_shift[tIdx] = indptr[tIdx % length] + tIdx / length;
  }
};

template <typename WMIdType, typename DegreeType>
struct ReduceForDegrees {
  WMIdType* rowoffsets;
  DegreeType* in_degree_ptr;
  int length;
  __host__ __device__ auto operator()(int64_t tIdx) {
    in_degree_ptr[tIdx] = rowoffsets[tIdx + length] - rowoffsets[tIdx];
  }
};

template <typename IdType>
struct DebugPrint
{
  const IdType* indptr;
  __host__ __device__
  void operator()(int64_t tIdx)
  {
    if constexpr (std::is_same<IdType, int>::value) {
      printf("Value output and indx : %d, %lld\n", indptr[tIdx], tIdx);
    } else if constexpr (std::is_same<IdType, int64_t>::value) {
      printf("Value output and indx : %lld, %lld\n", indptr[tIdx], tIdx);
    } else {
      printf("Unsupported type\n");
    }
  }
};

template <typename WMIdType, typename IdType>
struct MappedSlice {
  const WMIdType* indptr_sta;
  const IdType* sampling_center_nodes;
  int64_t* sliced_csr_row_ptr;
  int cnt;
  int* in_degree;
  size_t offset;
  __host__ __device__ auto operator()(int64_t tIdx) {
    // for last element, sliced_csr_row_ptr stores the offet of global edge id
    if (tIdx == cnt) {in_degree[tIdx] = 0; sliced_csr_row_ptr[tIdx] = indptr_sta[0]; return;}
    const auto out_row = sampling_center_nodes[tIdx] - offset;
    const auto indptr_val = indptr_sta[out_row];  // global edge offset of this row
    const auto degree = indptr_sta[out_row + 1] - indptr_val;
    in_degree[tIdx] = degree;
    sliced_csr_row_ptr[tIdx] = indptr_val; // local edge offset of this row
  }
};

template <typename DegreeType>
struct MinInDegreeFanout {
  int max_sample_count;
  __host__ __device__ auto operator()(DegreeType degree) {
    return min(static_cast<int>(degree), max_sample_count);
  }
};

template <typename WMIdType>
struct ReorderSampledEdgeOp {
  int* csr_ptr_sta;
  int* ordered_csr_ptr;
  int64_t* map_loc_row_id;  //dev_raw_indice_ptr
  int center_node_size_plus_1;
  WMIdType* input;
  WMIdType* output;
  __host__ __device__ auto operator()(int64_t tIdx) {
    int loc_row_id = thrust::upper_bound(thrust::seq, csr_ptr_sta, csr_ptr_sta + center_node_size_plus_1, tIdx) - csr_ptr_sta - 1; // local edge id to local row id
    auto offset = tIdx - csr_ptr_sta[loc_row_id]; // local edge id
    auto ordered_row_id = map_loc_row_id[loc_row_id];
    output[ordered_csr_ptr[ordered_row_id] + offset] = input[tIdx];
  }
};

struct ExpandLocRowOp {
  int* ordered_csr_ptr;
  int center_node_size_plus_1;
  int* output;
  __host__ __device__ auto operator()(int64_t tIdx) {
    int loc_row_id = thrust::upper_bound(thrust::seq, ordered_csr_ptr, ordered_csr_ptr + center_node_size_plus_1, tIdx) - ordered_csr_ptr - 1; // local edge id to local row id
    output[tIdx] = loc_row_id;
  }
};

}  // namespace wholegraph_ops
