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
#include "whole_memory_embedding.h"

#include <cuda_runtime_api.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <functional>
#include <utility>

#include "cuda_env_fns.h"
#include "file_utils.h"
#include "macros.h"
#include "optimizer.cuh"
#include "whole_chunked_memory.cuh"
#include "whole_memory.h"
#include "whole_memory_communicator.h"
#include "whole_nccl_memory.h"

namespace whole_graph {

template<typename EmbType, typename SrcType, typename WMEmbType = EmbType>
__global__ void CopyEmbeddingFromMemoryBlockKernel(WMEmbType *wm_embedding,
                                                   int64_t storage_offset,
                                                   int64_t table_size,
                                                   int64_t embedding_dim,
                                                   int64_t embedding_stride,
                                                   const SrcType *src_embedding,
                                                   int64_t src_start_id,
                                                   int64_t src_stride) {
  int block_idx = blockIdx.x;
  int tidx = threadIdx.x;
  whole_graph::PtrGen<WMEmbType, EmbType> ptr_gen(wm_embedding, storage_offset);
  auto emb_id = block_idx + src_start_id;
  assert(emb_id >= 0 && emb_id < table_size);
  EmbType *output_ptr = ptr_gen.At(emb_id * embedding_stride);
  const SrcType *input_ptr = src_embedding + block_idx * src_stride;
  for (int i = tidx; i < embedding_dim; i += blockDim.x) {
    output_ptr[i] = (EmbType) input_ptr[i];
  }
}

template<typename EmbType, typename SrcType>
void CopyEmbeddingFromMemoryBlock(void *wm_embedding,
                                  int64_t storage_offset_bytes,
                                  int64_t table_size,
                                  int64_t embedding_dim,
                                  int64_t embedding_stride,
                                  const void *src,
                                  int64_t src_start_id,
                                  int64_t src_table_size,
                                  int64_t src_stride,
                                  cudaStream_t stream) {
  auto src_embedding = (const SrcType *) src;
  int block_size = embedding_dim > 256 ? 256 : embedding_dim;
  CopyEmbeddingFromMemoryBlockKernel<EmbType, SrcType, EmbType><<<src_table_size, block_size, 0, stream>>>(
      (EmbType *) wm_embedding,
      storage_offset_bytes,
      table_size,
      embedding_dim,
      embedding_stride,
      src_embedding,
      src_start_id,
      src_stride);
}

REGISTER_DISPATCH_TWO_TYPES(CopyEmbeddingFromMemoryBlock,
                            CopyEmbeddingFromMemoryBlock,
                            FLOAT_DOUBLE,
                            FLOAT_DOUBLE)

void WmmpLoadEmbeddingFromMemory(void *wm_embedding,
                                 int64_t storage_offset,
                                 int64_t table_size,
                                 int64_t embedding_dim,
                                 int64_t embedding_stride,
                                 const void *src,
                                 int64_t src_stride,
                                 WMType src_type,
                                 WMType emb_type,
                                 int64_t src_start_id,
                                 int64_t src_table_size,
                                 cudaStream_t stream) {
  DISPATCH_TWO_TYPES(emb_type,
                     src_type,
                     CopyEmbeddingFromMemoryBlock,
                     wm_embedding,
                     storage_offset,
                     table_size,
                     embedding_dim,
                     embedding_stride,
                     src,
                     src_start_id,
                     src_table_size,
                     src_stride,
                     stream);
}

template<typename EmbType, typename SrcType>
void CopyChunkedEmbeddingFromMemoryBlock(void *wm_embedding,
                                         int64_t storage_offset,
                                         int64_t table_size,
                                         int64_t embedding_dim,
                                         int64_t embedding_stride,
                                         const void *src,
                                         int64_t src_start_id,
                                         int64_t src_table_size,
                                         int64_t src_stride,
                                         cudaStream_t stream) {
  auto src_embedding = (const SrcType *) src;
  int block_size = embedding_dim > 256 ? 256 : embedding_dim;
  CopyEmbeddingFromMemoryBlockKernel<EmbType,
                                     SrcType,
                                     const WholeChunkedMemoryHandle><<<src_table_size, block_size, 0, stream>>>(
      (const WholeChunkedMemoryHandle *) wm_embedding,
      storage_offset,
      table_size,
      embedding_dim,
      embedding_stride,
      src_embedding,
      src_start_id,
      src_stride);
  WM_CUDA_CHECK(cudaGetLastError());
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));
}

REGISTER_DISPATCH_TWO_TYPES(CopyChunkedEmbeddingFromMemoryBlock,
                            CopyChunkedEmbeddingFromMemoryBlock,
                            FLOAT_DOUBLE,
                            FLOAT_DOUBLE)

void WmmpLoadChunkedEmbeddingFromMemory(WholeChunkedMemory_t wm_embedding,
                                        int64_t storage_offset,
                                        int64_t table_size,
                                        int64_t embedding_dim,
                                        int64_t embedding_stride,
                                        const void *src,
                                        int64_t src_stride,
                                        WMType src_type,
                                        WMType emb_type,
                                        int64_t src_start_id,
                                        int64_t src_table_size,
                                        cudaStream_t stream) {
  int dev_id;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  WholeChunkedMemoryHandle *wmth = GetDeviceChunkedHandle(wm_embedding, dev_id);
  DISPATCH_TWO_TYPES(emb_type,
                     src_type,
                     CopyChunkedEmbeddingFromMemoryBlock,
                     wmth,
                     storage_offset,
                     table_size,
                     embedding_dim,
                     embedding_stride,
                     src,
                     src_start_id,
                     src_table_size,
                     src_stride,
                     stream);
}

void WmmpStoreLocalEmbeddingToFile(WMType emb_type,
                                   const void *emb_ptr,
                                   int64_t embedding_count,
                                   int64_t embedding_dim,
                                   int64_t embedding_stride,
                                   const std::string &filename) {
  size_t elt_size = GetWMTSize(emb_type);
  const int kBufferSize = 8 * 1024 * 1024;
  WM_CHECK(kBufferSize > embedding_dim * elt_size);
  void *host_buffer;
  WM_CUDA_CHECK(cudaMallocHost(&host_buffer, kBufferSize));
  int64_t max_batch_size = kBufferSize / (embedding_dim * elt_size);
  FILE *fp = fopen(filename.c_str(), "wb");
  if (fp == nullptr) {
    fprintf(stderr, "Open output file %s failed.\n", filename.c_str());
    abort();
  }
  for (int64_t start_embedding = 0; start_embedding < embedding_count; start_embedding += max_batch_size) {
    int64_t batch_size = embedding_count - start_embedding;
    if (batch_size > max_batch_size) batch_size = max_batch_size;
    WM_CUDA_CHECK(cudaMemcpy2D(host_buffer,
                               embedding_dim * elt_size,
                               (const char *) emb_ptr + start_embedding * embedding_stride * elt_size,
                               embedding_stride * elt_size,
                               embedding_dim * elt_size,
                               batch_size,
                               cudaMemcpyDeviceToHost));
    WM_CHECK(fwrite(host_buffer, embedding_dim * elt_size, batch_size, fp) == batch_size);
  }
  fclose(fp);
  fprintf(stderr, "Done storing local embedding to file %s\n", filename.c_str());
  WM_CUDA_CHECK(cudaFreeHost(host_buffer));
}

void WmmpLoadLocalEmbeddingFromFile(WMType emb_type,
                                    void *emb_ptr,
                                    int64_t embedding_count,
                                    int64_t embedding_dim,
                                    int64_t embedding_stride,
                                    const std::string &file_prefix,
                                    int part_count,
                                    BootstrapCommunicator *bootstrap_communicator) {
  bool use_part_file = part_count > 0;
  if (part_count == 0) part_count = 1;
  int rank = bootstrap_communicator->Rank();
  size_t elt_size = GetWMTSize(emb_type);
  const int kBufferSize = 8 * 1024 * 1024;
  WM_CHECK(kBufferSize > embedding_dim * elt_size);
  void *host_buffer;
  WM_CUDA_CHECK(cudaMallocHost(&host_buffer, kBufferSize));
  int64_t max_batch_size = kBufferSize / (embedding_dim * elt_size);
  std::vector<int64_t> emb_count_vec(bootstrap_communicator->Size());
  CollAllGather(embedding_count, &emb_count_vec, bootstrap_communicator);
  WM_CHECK(emb_count_vec[rank] == embedding_count);
  std::vector<int64_t> emb_start_vec(bootstrap_communicator->Size(), 0);
  int64_t embedding_start_idx = 0;
  for (int i = 0; i < bootstrap_communicator->Size(); i++) {
    emb_start_vec[i] = embedding_start_idx;
    embedding_start_idx += emb_count_vec[i];
  }
  int64_t total_vec_count = embedding_start_idx;
  std::vector<int64_t> file_emb_count_vec(part_count), file_emb_start_vec(part_count);
  embedding_start_idx = 0;
  for (int i = 0; i < part_count; i++) {
    std::string filename = file_prefix;
    if (use_part_file) filename = GetPartFileName(file_prefix, i, part_count);
    auto file_size = StatFileSize(filename);
    WM_CHECK(file_size % (embedding_dim * elt_size) == 0);
    int64_t current_file_emb_count = file_size / (embedding_dim * elt_size);
    file_emb_count_vec[i] = current_file_emb_count;
    file_emb_start_vec[i] = embedding_start_idx;
    embedding_start_idx += current_file_emb_count;
  }
  int64_t total_file_vec_count = embedding_start_idx;
  WM_CHECK(total_vec_count == total_file_vec_count);
  int64_t rank_start_idx = emb_start_vec[rank];
  int64_t rank_end_idx = rank_start_idx + embedding_count;
  for (int i = 0; i < part_count; i++) {
    std::string filename = file_prefix;
    if (use_part_file) filename = GetPartFileName(file_prefix, i, part_count);
    int64_t file_start_idx = file_emb_start_vec[i];
    int64_t file_emb_count = file_emb_count_vec[i];
    int64_t file_end_idx = file_emb_start_vec[i] + file_emb_count;
    if (file_start_idx >= rank_end_idx || file_end_idx <= rank_start_idx)
      continue;
    int64_t intersect_start_idx = std::max(file_start_idx, rank_start_idx);
    int64_t intersect_end_idx = std::min(file_end_idx, rank_end_idx);
    int64_t intersect_count = intersect_end_idx - intersect_start_idx;
    int64_t file_idx_offset = intersect_start_idx - file_start_idx;
    int64_t rank_idx_offset = intersect_start_idx - rank_start_idx;
    FILE *fp = fopen(filename.c_str(), "rb");
    if (fp == nullptr) {
      fprintf(stderr, "Open file %s failed.\n", filename.c_str());
      abort();
    }
    WM_CHECK(fseeko(fp, file_idx_offset * embedding_dim * elt_size, SEEK_SET) == 0);

    for (int64_t start_embedding = 0; start_embedding < intersect_count; start_embedding += max_batch_size) {
      int64_t batch_size = embedding_count - start_embedding;
      if (batch_size > max_batch_size) batch_size = max_batch_size;
      int ret = fread(host_buffer, embedding_dim * elt_size, batch_size, fp);
      if (ret != batch_size) {
        fprintf(stderr, "reading from file %s, batchsize=%ld, embedding_dim=%ld, returned %d, error=%s\n",
                filename.c_str(), batch_size, embedding_dim, ret, strerror(errno));
      }
      WM_CHECK(ret == batch_size);
      WM_CUDA_CHECK(cudaMemcpy2D((char *) emb_ptr + (start_embedding + rank_idx_offset) * embedding_stride * elt_size,
                                 embedding_stride * elt_size,
                                 host_buffer,
                                 embedding_dim * elt_size,
                                 embedding_dim * elt_size,
                                 batch_size,
                                 cudaMemcpyHostToDevice));
    }
    fclose(fp);
    fprintf(stderr,
            "Rank=%d done reading %ld embedding vectors from file %s\n",
            rank,
            intersect_count,
            filename.c_str());
  }
}

#if 0
template <typename OutputT, typename ParamT, typename IdxT, typename ParamHandleT = ParamT>
__global__ void WholeMemoryGatherKernelOld(OutputT *__restrict__ output,
                                        const ParamHandleT *__restrict__ parameter,
                                        const IdxT *__restrict__ indice,
                                        size_t storage_offset,
                                        int64_t embedding_dim,
                                        int64_t embedding_stride,
                                        int64_t output_stride) {
  IdxT idx = indice[blockIdx.x];
  whole_graph::PtrGen<const ParamHandleT, ParamT> ptr_gen(parameter, storage_offset);
  const ParamT *param_ptr = ptr_gen.At((size_t) idx * embedding_stride);
  OutputT *output_ptr = output + (size_t) blockIdx.x * output_stride;
  for (int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
    output_ptr[i] = (OutputT)param_ptr[i];
  }
}
#endif

template<typename T>
__device__ __forceinline__ void MovTypedData(T *to, const T *from) {
  *to = *from;
}
template<int DataSize>
__device__ __forceinline__ void MovData(void *to, const void *from) {
  char *ptr_to = (char *) to;
  const char *ptr_from = (const char *) from;
  for (int i = 0; i < DataSize; i++) {
    ptr_to[i] = ptr_from[i];
  }
}
template<>
__device__ __forceinline__ void MovData<1>(void *to, const void *from) {
  MovTypedData((int8_t *) to, (const int8_t *) from);
}
template<>
__device__ __forceinline__ void MovData<2>(void *to, const void *from) {
  MovTypedData((int16_t *) to, (const int16_t *) from);
}
template<>
__device__ __forceinline__ void MovData<4>(void *to, const void *from) {
  MovTypedData((int32_t *) to, (const int32_t *) from);
}
template<>
__device__ __forceinline__ void MovData<8>(void *to, const void *from) {
  MovTypedData((int64_t *) to, (const int64_t *) from);
}
template<>
__device__ __forceinline__ void MovData<16>(void *to, const void *from) {
  MovTypedData((int4 *) to, (const int4 *) from);
}
template<>
__device__ __forceinline__ void MovData<32>(void *to, const void *from) {
  MovTypedData((int4 *) to, (const int4 *) from);
  MovTypedData(((int4 *) to) + 1, ((const int4 *) from) + 1);
}
template<>
__device__ __forceinline__ void MovData<64>(void *to, const void *from) {
  MovTypedData((int4 *) to, (const int4 *) from);
  MovTypedData(((int4 *) to) + 1, ((const int4 *) from) + 1);
  MovTypedData(((int4 *) to) + 2, ((const int4 *) from) + 2);
  MovTypedData(((int4 *) to) + 3, ((const int4 *) from) + 3);
}

template<typename OutputT, typename ParamT, typename IdxT, typename ParamHandleT, int Alignment = 1>
__global__ void WholeMemoryGatherKernel(OutputT *__restrict__ output,
                                        const ParamHandleT *__restrict__ parameter,
                                        const IdxT *__restrict__ indice,
                                        size_t storage_offset,
                                        int64_t embedding_dim,
                                        int64_t embedding_stride,
                                        int64_t output_stride,
                                        int64_t start_indice) {
  IdxT idx = indice[blockIdx.x] - start_indice;
  assert(idx >= 0);
  whole_graph::PtrGen<const ParamHandleT, ParamT> ptr_gen(parameter, storage_offset);
  const ParamT *param_ptr = ptr_gen.At((size_t) idx * embedding_stride);
  ParamT tmp[Alignment];
  OutputT cvt_tmp[Alignment];
  OutputT *output_ptr = output + (size_t) blockIdx.x * output_stride;
  for (int i = threadIdx.x * Alignment; i < embedding_dim; i += blockDim.x * Alignment) {
    MovData<Alignment * sizeof(ParamT)>(&tmp[0], &param_ptr[i]);
    for (int a = 0; a < Alignment; a++) cvt_tmp[a] = (OutputT) tmp[a];
    MovData<Alignment * sizeof(OutputT)>(&output_ptr[i], &cvt_tmp[0]);
  }
}

template<typename OutputT, typename ParamT>
int DetermineAlignmentEltCount(const void *output,
                               size_t storage_elt_offset,
                               int64_t embedding_dim,
                               int64_t embedding_stride,
                               int64_t output_stride) {
  int src_alignment = 16 / sizeof(ParamT);
  for (; src_alignment > 1; src_alignment /= 2) {
    if (storage_elt_offset % src_alignment == 0 && embedding_dim % src_alignment == 0
        && embedding_stride % src_alignment == 0)
      break;
  }
  int dst_alignment = 16 / sizeof(OutputT);
  for (; dst_alignment > 1; dst_alignment /= 2) {
    if ((int64_t) output % (dst_alignment * sizeof(OutputT)) == 0 && embedding_dim % dst_alignment == 0
        && output_stride % dst_alignment == 0)
      break;
  }
  return std::min(src_alignment, dst_alignment);
}

template<typename OutputT, typename ParamT, typename IdxT>
void WholeMemoryGatherFunc(void *output,
                           const void *parameter,
                           const void *indice,
                           size_t storage_offset,
                           int64_t indice_count,
                           int64_t embedding_dim,
                           int64_t embedding_stride,
                           int64_t output_stride,
                           int64_t start_indice,
                           cudaStream_t stream) {
  if (indice_count == 0) return;
  int alignment = DetermineAlignmentEltCount<OutputT, ParamT>(output,
                                                              storage_offset,
                                                              embedding_dim,
                                                              embedding_stride,
                                                              output_stride);
  int thread_count = embedding_dim / alignment;
  if (thread_count > 256) thread_count = 256;
  void (*kernel_fn)(OutputT *, const ParamT *, const IdxT *, size_t, int64_t, int64_t, int64_t, int64_t) = nullptr;
  WM_CHECK(alignment == 1 || alignment == 2 || alignment == 4 || alignment == 8);
  if (alignment == 1) {
    kernel_fn = WholeMemoryGatherKernel<OutputT, ParamT, IdxT, ParamT, 1>;
  } else if (alignment == 2) {
    kernel_fn = WholeMemoryGatherKernel<OutputT, ParamT, IdxT, ParamT, 2>;
  } else if (alignment == 4) {
    kernel_fn = WholeMemoryGatherKernel<OutputT, ParamT, IdxT, ParamT, 4>;
  } else if (alignment == 8) {
    kernel_fn = WholeMemoryGatherKernel<OutputT, ParamT, IdxT, ParamT, 8>;
  }
  kernel_fn<<<indice_count, thread_count, 0, stream>>>(
      (OutputT *) output,
      (const ParamT *) parameter,
      (const IdxT *) indice,
      storage_offset,
      embedding_dim,
      embedding_stride,
      output_stride,
      start_indice);
  WM_CUDA_CHECK(cudaGetLastError());
}
REGISTER_DISPATCH_THREE_TYPES(WholeMemoryGatherFunc,
                              WholeMemoryGatherFunc,
                              HALF_FLOAT_DOUBLE,
                              HALF_FLOAT_DOUBLE,
                              SINT3264)

REGISTER_DISPATCH_THREE_TYPES(WholeMemoryGatherIntFunc,
                              WholeMemoryGatherFunc,
                              SINT,
                              SINT,
                              SINT3264)

template<typename OutputT, typename ParamT, typename IdxT>
void WholeMemoryChunkedGatherFunc(void *output,
                                  const void *parameter,
                                  const void *indice,
                                  size_t storage_offset,
                                  int64_t indice_count,
                                  int64_t embedding_dim,
                                  int64_t embedding_stride,
                                  int64_t output_stride,
                                  cudaStream_t stream) {
  if (indice_count == 0) return;
  int alignment = DetermineAlignmentEltCount<OutputT, ParamT>(output,
                                                              storage_offset,
                                                              embedding_dim,
                                                              embedding_stride,
                                                              output_stride);
  int thread_count = embedding_dim / alignment;
  if (thread_count > 256) thread_count = 256;
  void (*kernel_fn)(OutputT *,
                    const WholeChunkedMemoryHandle *,
                    const IdxT *,
                    size_t,
                    int64_t,
                    int64_t,
                    int64_t,
                    int64_t) = nullptr;
  WM_CHECK(alignment == 1 || alignment == 2 || alignment == 4 || alignment == 8);
  if (alignment == 1) {
    kernel_fn = WholeMemoryGatherKernel<OutputT, ParamT, IdxT, WholeChunkedMemoryHandle, 1>;
  } else if (alignment == 2) {
    kernel_fn = WholeMemoryGatherKernel<OutputT, ParamT, IdxT, WholeChunkedMemoryHandle, 2>;
  } else if (alignment == 4) {
    kernel_fn = WholeMemoryGatherKernel<OutputT, ParamT, IdxT, WholeChunkedMemoryHandle, 4>;
  } else if (alignment == 8) {
    kernel_fn = WholeMemoryGatherKernel<OutputT, ParamT, IdxT, WholeChunkedMemoryHandle, 8>;
  }
  int dev_id = -1;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  WholeChunkedMemoryHandle *parameter_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) parameter, dev_id);
  kernel_fn<<<indice_count, thread_count, 0, stream>>>(
      (OutputT *) output,
      (const WholeChunkedMemoryHandle *) parameter_handle,
      (const IdxT *) indice,
      storage_offset,
      embedding_dim,
      embedding_stride,
      output_stride,
      0);
  WM_CUDA_CHECK(cudaGetLastError());
}

REGISTER_DISPATCH_THREE_TYPES(WholeMemoryChunkedGatherFunc,
                              WholeMemoryChunkedGatherFunc,
                              HALF_FLOAT_DOUBLE,
                              HALF_FLOAT_DOUBLE,
                              SINT3264)
REGISTER_DISPATCH_THREE_TYPES(WholeMemoryChunkedGatherIntFunc,
                              WholeMemoryChunkedGatherFunc,
                              SINT,
                              SINT,
                              SINT3264)

void WholeMemoryGather(WMType output_t,
                       WMType param_t,
                       WMType index_t,
                       void *output,
                       const void *parameter,
                       const void *indice,
                       size_t storage_offset,
                       int64_t indice_count,
                       int64_t embedding_dim,
                       int64_t embedding_stride,
                       int64_t output_stride,
                       cudaStream_t stream) {
  bool param_is_int = param_t == WMT_Int8 || param_t == WMT_Int16 || param_t == WMT_Int32 || param_t == WMT_Int64;
  bool param_is_float = param_t == WMT_Half || param_t == WMT_Float || param_t == WMT_Double;
  bool output_is_int = output_t == WMT_Int8 || output_t == WMT_Int16 || output_t == WMT_Int32 || output_t == WMT_Int64;
  bool output_is_float = output_t == WMT_Half || output_t == WMT_Float || output_t == WMT_Double;
  WM_CHECK(param_is_int || param_is_float);
  WM_CHECK(output_is_int || output_is_float);
  WM_CHECK(param_is_int == output_is_int && param_is_float == output_is_float);
  if (param_is_int) {
    DISPATCH_THREE_TYPES(output_t,
                         param_t,
                         index_t,
                         WholeMemoryGatherIntFunc,
                         output,
                         parameter,
                         indice,
                         storage_offset,
                         indice_count,
                         embedding_dim,
                         embedding_stride,
                         output_stride,
                         0,
                         stream);
  } else {
    DISPATCH_THREE_TYPES(output_t,
                         param_t,
                         index_t,
                         WholeMemoryGatherFunc,
                         output,
                         parameter,
                         indice,
                         storage_offset,
                         indice_count,
                         embedding_dim,
                         embedding_stride,
                         output_stride,
                         0,
                         stream);
  }
}

void WholeMemoryChunkedGather(WMType output_t,
                              WMType param_t,
                              WMType index_t,
                              void *output,
                              const void *parameter,
                              const void *indice,
                              size_t storage_offset,
                              int64_t indice_count,
                              int64_t embedding_dim,
                              int64_t embedding_stride,
                              int64_t output_stride,
                              cudaStream_t stream) {
  bool param_is_int = param_t == WMT_Int8 || param_t == WMT_Int16 || param_t == WMT_Int32 || param_t == WMT_Int64;
  bool param_is_float = param_t == WMT_Half || param_t == WMT_Float || param_t == WMT_Double;
  bool output_is_int = output_t == WMT_Int8 || output_t == WMT_Int16 || output_t == WMT_Int32 || output_t == WMT_Int64;
  bool output_is_float = output_t == WMT_Half || output_t == WMT_Float || output_t == WMT_Double;
  WM_CHECK(param_is_int || param_is_float);
  WM_CHECK(output_is_int || output_is_float);
  WM_CHECK(param_is_int == output_is_int && param_is_float == output_is_float);
  if (param_is_int) {
    DISPATCH_THREE_TYPES(output_t,
                         param_t,
                         index_t,
                         WholeMemoryChunkedGatherIntFunc,
                         output,
                         parameter,
                         indice,
                         storage_offset,
                         indice_count,
                         embedding_dim,
                         embedding_stride,
                         output_stride,
                         stream);
  } else {
    DISPATCH_THREE_TYPES(output_t,
                         param_t,
                         index_t,
                         WholeMemoryChunkedGatherFunc,
                         output,
                         parameter,
                         indice,
                         storage_offset,
                         indice_count,
                         embedding_dim,
                         embedding_stride,
                         output_stride,
                         stream);
  }
}

template<typename IdxT>
__global__ void WholeMemoryNCCLNodeCountForRanksKernel(const IdxT *indices,
                                                       int *rank_count,
                                                       int64_t indice_count,
                                                       int comm_size,
                                                       int64_t rank_interval_width) {
  extern __shared__ int rank_count_shared[];
  for (int idx = threadIdx.x; idx < comm_size; idx += blockDim.x) {
    rank_count_shared[idx] = 0;
  }
  __syncthreads();
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < indice_count; idx += blockDim.x * gridDim.x) {
    IdxT node_idx = indices[idx];
    int rank = node_idx / rank_interval_width;
    assert(rank >= 0 && rank < comm_size);
    atomicAdd_block(&rank_count_shared[rank], 1);
  }
  __syncthreads();
  for (int idx = threadIdx.x; idx < comm_size; idx += blockDim.x) {
    atomicAdd(rank_count + idx, rank_count_shared[idx]);
  }
}

template<typename IdxT>
void WholeMemoryNCCLNodeCountForRanks(const IdxT *indices,
                                      int *rank_count,
                                      int64_t indice_count,
                                      int comm_size,
                                      int64_t rank_interval_width,
                                      cudaStream_t stream) {
  int sm_count = 0;
  int dev_id = -1;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  WM_CHECK(dev_id >= 0);
  WM_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id));
  WM_CHECK(sm_count > 0);
  static constexpr int block_size = 512;
  int block_count = DivUp(indice_count, block_size);
  block_count = std::max(block_count, sm_count * 2);
  cudaMemsetAsync(rank_count, 0, sizeof(int) * comm_size, stream);
  WholeMemoryNCCLNodeCountForRanksKernel<<<block_count, block_size, comm_size * sizeof(int), stream>>>(indices,
                                                                                                       rank_count,
                                                                                                       indice_count,
                                                                                                       comm_size,
                                                                                                       rank_interval_width);
  WM_CUDA_CHECK(cudaGetLastError());
}

template<typename InputT, typename ParamT, typename IdxT, typename ParamHandleT, int Alignment = 1>
__global__ void WholeMemoryScatterKernel(const InputT *__restrict__ input,
                                         ParamHandleT *__restrict__ parameter,
                                         const IdxT *__restrict__ indice,
                                         size_t storage_offset,
                                         int64_t embedding_dim,
                                         int64_t embedding_stride,
                                         int64_t input_stride,
                                         int64_t start_indice) {
  IdxT idx = indice[blockIdx.x] - start_indice;
  assert(idx >= 0);
  whole_graph::PtrGen<ParamHandleT, ParamT> ptr_gen(parameter, storage_offset);
  ParamT *param_ptr = ptr_gen.At((size_t) idx * embedding_stride);
  InputT tmp[Alignment];
  ParamT cvt_tmp[Alignment];
  const InputT *input_ptr = input + (size_t) blockIdx.x * input_stride;
  for (int i = threadIdx.x * Alignment; i < embedding_dim; i += blockDim.x * Alignment) {
    MovData<Alignment * sizeof(InputT)>(&tmp[0], &input_ptr[i]);
    for (int a = 0; a < Alignment; a++) cvt_tmp[a] = (ParamT) tmp[a];
    MovData<Alignment * sizeof(ParamT)>(&param_ptr[i], &cvt_tmp[0]);
  }
}

template<typename ParamT, typename IdxT>
void WholeMemoryScatterFunc(const void *input,
                            void *parameter,
                            const void *indice,
                            size_t storage_offset,
                            int64_t indice_count,
                            int64_t embedding_dim,
                            int64_t embedding_stride,
                            int64_t input_stride,
                            int64_t start_indice,
                            cudaStream_t stream) {
  if (indice_count == 0) return;
  int alignment = DetermineAlignmentEltCount<ParamT, ParamT>(input,
                                                             storage_offset,
                                                             embedding_dim,
                                                             embedding_stride,
                                                             input_stride);
  int thread_count = embedding_dim / alignment;
  if (thread_count > 256) thread_count = 256;
  void (*kernel_fn)(const ParamT *, ParamT *, const IdxT *, size_t, int64_t, int64_t, int64_t, int64_t) = nullptr;
  WM_CHECK(alignment == 1 || alignment == 2 || alignment == 4 || alignment == 8);
  if (alignment == 1) {
    kernel_fn = WholeMemoryScatterKernel<ParamT, ParamT, IdxT, ParamT, 1>;
  } else if (alignment == 2) {
    kernel_fn = WholeMemoryScatterKernel<ParamT, ParamT, IdxT, ParamT, 2>;
  } else if (alignment == 4) {
    kernel_fn = WholeMemoryScatterKernel<ParamT, ParamT, IdxT, ParamT, 4>;
  } else if (alignment == 8) {
    kernel_fn = WholeMemoryScatterKernel<ParamT, ParamT, IdxT, ParamT, 8>;
  }
  kernel_fn<<<indice_count, thread_count, 0, stream>>>(
      (const ParamT *) input,
      (ParamT *) parameter,
      (const IdxT *) indice,
      storage_offset,
      embedding_dim,
      embedding_stride,
      input_stride,
      start_indice);
  WM_CUDA_CHECK(cudaGetLastError());
}
REGISTER_DISPATCH_TWO_TYPES(WholeMemoryScatterFuncFloat,
                            WholeMemoryScatterFunc,
                            HALF_FLOAT_DOUBLE,
                            SINT3264)
REGISTER_DISPATCH_TWO_TYPES(WholeMemoryScatterFuncInt,
                            WholeMemoryScatterFunc,
                            ALLINT,
                            SINT3264)

template<typename ParamT, typename IdxT>
void WholeMemoryChunkedScatterFunc(const void *input,
                                   void *parameter,
                                   const void *indice,
                                   size_t storage_offset,
                                   int64_t indice_count,
                                   int64_t embedding_dim,
                                   int64_t embedding_stride,
                                   int64_t input_stride,
                                   cudaStream_t stream) {
  if (indice_count == 0) return;
  int alignment = DetermineAlignmentEltCount<ParamT, ParamT>(input,
                                                             storage_offset,
                                                             embedding_dim,
                                                             embedding_stride,
                                                             input_stride);
  int thread_count = embedding_dim / alignment;
  if (thread_count > 256) thread_count = 256;
  void (*kernel_fn)(const ParamT *,
                    WholeChunkedMemoryHandle *,
                    const IdxT *,
                    size_t,
                    int64_t,
                    int64_t,
                    int64_t,
                    int64_t) = nullptr;
  WM_CHECK(alignment == 1 || alignment == 2 || alignment == 4 || alignment == 8);
  if (alignment == 1) {
    kernel_fn = WholeMemoryScatterKernel<ParamT, ParamT, IdxT, WholeChunkedMemoryHandle, 1>;
  } else if (alignment == 2) {
    kernel_fn = WholeMemoryScatterKernel<ParamT, ParamT, IdxT, WholeChunkedMemoryHandle, 2>;
  } else if (alignment == 4) {
    kernel_fn = WholeMemoryScatterKernel<ParamT, ParamT, IdxT, WholeChunkedMemoryHandle, 4>;
  } else if (alignment == 8) {
    kernel_fn = WholeMemoryScatterKernel<ParamT, ParamT, IdxT, WholeChunkedMemoryHandle, 8>;
  }
  int dev_id = -1;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  WholeChunkedMemoryHandle *parameter_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t) parameter, dev_id);
  kernel_fn<<<indice_count, thread_count, 0, stream>>>(
      (const ParamT *) input,
      (WholeChunkedMemoryHandle *) parameter_handle,
      (const IdxT *) indice,
      storage_offset,
      embedding_dim,
      embedding_stride,
      input_stride,
      0);
  WM_CUDA_CHECK(cudaGetLastError());
}

REGISTER_DISPATCH_TWO_TYPES(WholeMemoryChunkedScatterFuncFloat,
                            WholeMemoryChunkedScatterFunc,
                            HALF_FLOAT_DOUBLE,
                            SINT3264)
REGISTER_DISPATCH_TWO_TYPES(WholeMemoryChunkedScatterFuncInt,
                            WholeMemoryChunkedScatterFunc,
                            ALLINT,
                            SINT3264)

void WholeMemoryScatter(WMType param_t,
                        WMType index_t,
                        const void *input,
                        void *parameter,
                        const void *indice,
                        size_t storage_offset,
                        int64_t indice_count,
                        int64_t embedding_dim,
                        int64_t embedding_stride,
                        int64_t input_stride,
                        cudaStream_t stream) {
  bool is_float = param_t == WMT_Float || param_t == WMT_Half || param_t == WMT_Double;
  bool is_int = param_t == WMT_Int8 || param_t == WMT_Int16 || param_t == WMT_Int32 || param_t == WMT_Int64;
  WM_CHECK(is_float || is_int);
  if (is_float) {
    DISPATCH_TWO_TYPES(param_t,
                       index_t,
                       WholeMemoryScatterFuncFloat,
                       input,
                       parameter,
                       indice,
                       storage_offset,
                       indice_count,
                       embedding_dim,
                       embedding_stride,
                       input_stride,
                       0,
                       stream);
  } else {
    DISPATCH_TWO_TYPES(param_t,
                       index_t,
                       WholeMemoryScatterFuncInt,
                       input,
                       parameter,
                       indice,
                       storage_offset,
                       indice_count,
                       embedding_dim,
                       embedding_stride,
                       input_stride,
                       0,
                       stream);
  }
}

void WholeMemoryChunkedScatter(WMType param_t,
                               WMType index_t,
                               const void *input,
                               void *parameter,
                               const void *indice,
                               size_t storage_offset,
                               int64_t indice_count,
                               int64_t embedding_dim,
                               int64_t embedding_stride,
                               int64_t input_stride,
                               cudaStream_t stream) {
  bool is_float = param_t == WMT_Float || param_t == WMT_Half || param_t == WMT_Double;
  bool is_int = param_t == WMT_Int8 || param_t == WMT_Int16 || param_t == WMT_Int32 || param_t == WMT_Int64;
  WM_CHECK(is_float || is_int);
  if (is_float) {
    DISPATCH_TWO_TYPES(param_t,
                       index_t,
                       WholeMemoryChunkedScatterFuncFloat,
                       input,
                       parameter,
                       indice,
                       storage_offset,
                       indice_count,
                       embedding_dim,
                       embedding_stride,
                       input_stride,
                       stream);
  } else {
    DISPATCH_TWO_TYPES(param_t,
                       index_t,
                       WholeMemoryChunkedScatterFuncInt,
                       input,
                       parameter,
                       indice,
                       storage_offset,
                       indice_count,
                       embedding_dim,
                       embedding_stride,
                       input_stride,
                       stream);
  }
}

WMType NormalizeNCCLWMType(WMType param_t) {
  size_t elt_size = GetWMTSize(param_t);
  switch (elt_size) {
    case 1: {
      param_t = WMT_Int8;
      break;
    }
    case 2: {
      param_t = WMT_Int16;
      break;
    }
    case 4: {
      param_t = WMT_Int32;
      break;
    }
    case 8: {
      param_t = WMT_Int64;
      break;
    }
    default: {
      WM_CHECK(false);
      break;
    }
  }
  return param_t;
}

template<typename ParamT, typename IdxT>
void WholeMemoryNCCLGatherFunc(void *output,
                               whole_graph::WholeNCCLMemory_t parameter_wnmt,
                               const void *indice,
                               size_t storage_offset,
                               int64_t indice_count,
                               int64_t embedding_dim,
                               int64_t embedding_stride,
                               int64_t output_stride,
                               const CUDAEnvFns &cuda_env_fns,
                               cudaStream_t stream) {
  WMThrustAllocator allocator(cuda_env_fns);
  const ParamT *parameter_local_ptr;
  size_t parameter_local_size;
  WnmmpGetLocalMemory(parameter_wnmt, (void **) &parameter_local_ptr, &parameter_local_size);
  parameter_local_size = WnmmpGetChunkSize(parameter_wnmt);
  auto *bcomm = WnmmpGetBootstrapCommunicator(parameter_wnmt);
  const IdxT *indice_ptr = (const IdxT *) indice;
  int64_t local_node_count = parameter_local_size / sizeof(ParamT) / embedding_stride;
  TempMemoryHandle rank_count_ptr_tmh, rank_count_host_tmh, rank_offset_host_tmh, recv_rank_count_host_tmh,
      sorted_indice_tmh, reverse_indice_tmh, recv_ids_tmh, local_output_tmh, unshuffled_output_tmh;
  int *rank_count = (int *) cuda_env_fns.allocate_temp_fn(bcomm->Size() * sizeof(int), &rank_count_ptr_tmh);
  int *rank_count_host = (int *) cuda_env_fns.allocate_host_temp_fn(bcomm->Size() * sizeof(int), &rank_count_host_tmh);
  int *rank_offset_host = (int *) cuda_env_fns.allocate_host_temp_fn((bcomm->Size() + 1) * sizeof(int), &rank_offset_host_tmh);
  int *recv_rank_count_host = (int *) cuda_env_fns.allocate_host_temp_fn(bcomm->Size() * sizeof(int), &recv_rank_count_host_tmh);
  IdxT *sorted_indice = (IdxT *) cuda_env_fns.allocate_temp_fn(indice_count * sizeof(IdxT), &sorted_indice_tmh);
  int *reverse_indice = (int *) cuda_env_fns.allocate_temp_fn(indice_count * sizeof(int), &reverse_indice_tmh);
  // Exchange node count
  WholeMemoryNCCLNodeCountForRanks(indice_ptr, rank_count, indice_count, bcomm->Size(), local_node_count, stream);
  WM_CUDA_CHECK(cudaMemcpyAsync(rank_count_host, rank_count, bcomm->Size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
  WM_CUDA_CHECK(cudaGetLastError());
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
  bcomm->AllToAll(rank_count_host, sizeof(int), recv_rank_count_host, sizeof(int));
  bcomm->RecordEvent();
  rank_offset_host[0] = 0;
  for (int i = 0; i < bcomm->Size(); i++) {
    rank_offset_host[i + 1] = rank_offset_host[i] + rank_count_host[i];
  }
  bcomm->Synchronize();
  // Exchange ids
  WM_CUDA_CHECK(cudaMemcpyAsync(sorted_indice, indice, sizeof(IdxT) * indice_count, cudaMemcpyDeviceToDevice, stream));
  thrust::sequence(thrust::cuda::par(allocator).on(stream),
                   reverse_indice,
                   reverse_indice + indice_count,
                   0);
  thrust::sort_by_key(thrust::cuda::par(allocator).on(stream),
                      sorted_indice,
                      sorted_indice + indice_count,
                      reverse_indice);
  int total_recv_count = 0;
  for (int i = 0; i < bcomm->Size(); i++) {
    total_recv_count += recv_rank_count_host[i];
  }
  IdxT *recv_indice = (IdxT *) cuda_env_fns.allocate_temp_fn(total_recv_count * sizeof(IdxT), &recv_ids_tmh);
  ParamT *local_output = (ParamT *) cuda_env_fns.allocate_temp_fn(total_recv_count * sizeof(ParamT) * embedding_dim, &local_output_tmh);
  ParamT *unshuffled_output = (ParamT *) cuda_env_fns.allocate_temp_fn(indice_count * sizeof(ParamT) * embedding_dim, &unshuffled_output_tmh);
  std::vector<void *> send_ptrs(bcomm->Size()), recv_ptrs(bcomm->Size());
  std::vector<int> send_sizes(bcomm->Size()), recv_sizes(bcomm->Size());
  int64_t all_recv_idx = 0;
  for (int i = 0; i < bcomm->Size(); i++) {
    send_ptrs[i] = sorted_indice + rank_offset_host[i];
    recv_ptrs[i] = recv_indice + all_recv_idx;
    send_sizes[i] = rank_count_host[i] * sizeof(IdxT);
    recv_sizes[i] = recv_rank_count_host[i] * sizeof(IdxT);
    all_recv_idx += recv_rank_count_host[i];
  }
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
  bcomm->AllToAllV((const void **) send_ptrs.data(), send_sizes.data(), recv_ptrs.data(), recv_sizes.data());
  bcomm->RecordEvent();
  bcomm->Synchronize();
  // Local Gather
  WholeMemoryGatherFunc<ParamT, ParamT, IdxT>(local_output,
                                              parameter_local_ptr,
                                              recv_indice,
                                              storage_offset,
                                              total_recv_count,
                                              embedding_dim,
                                              embedding_stride,
                                              embedding_dim,
                                              bcomm->Rank() * local_node_count,
                                              stream);
  WM_CUDA_CHECK(cudaGetLastError());
  all_recv_idx = 0;
  for (int i = 0; i < bcomm->Size(); i++) {
    send_ptrs[i] = local_output + all_recv_idx * embedding_dim;
    send_sizes[i] = recv_rank_count_host[i] * embedding_dim * sizeof(ParamT);
    all_recv_idx += recv_rank_count_host[i];
    recv_ptrs[i] = unshuffled_output + rank_offset_host[i] * embedding_dim;
    recv_sizes[i] = rank_count_host[i] * embedding_dim * sizeof(ParamT);
  }
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
  // AllToAllV for embeddings
  bcomm->AllToAllV((const void **) send_ptrs.data(), send_sizes.data(), recv_ptrs.data(), recv_sizes.data());
  bcomm->RecordEvent();
  bcomm->Synchronize();
  // Local Reorder
  WholeMemoryScatterFunc<ParamT, int>(unshuffled_output,
                                      (ParamT *) output,
                                      reverse_indice,
                                      0,
                                      indice_count,
                                      embedding_dim,
                                      embedding_dim,
                                      embedding_dim,
                                      0,
                                      stream);
  WM_CUDA_CHECK(cudaGetLastError());
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
  cuda_env_fns.free_temp_fn(&sorted_indice_tmh);
  cuda_env_fns.free_temp_fn(&rank_count_ptr_tmh);
  cuda_env_fns.free_host_temp_fn(&rank_count_host_tmh);
  cuda_env_fns.free_host_temp_fn(&rank_offset_host_tmh);
  cuda_env_fns.free_host_temp_fn(&recv_rank_count_host_tmh);
  cuda_env_fns.free_temp_fn(&reverse_indice_tmh);
  cuda_env_fns.free_temp_fn(&recv_ids_tmh);
  cuda_env_fns.free_temp_fn(&local_output_tmh);
  cuda_env_fns.free_temp_fn(&unshuffled_output_tmh);
  allocator.deallocate_all();
}

REGISTER_DISPATCH_TWO_TYPES(WholeMemoryNCCLGatherFunc,
                            WholeMemoryNCCLGatherFunc,
                            SINT,
                            SINT3264)

void WholeMemoryNCCLGather(WMType param_t,
                           WMType index_t,
                           void *output,
                           whole_graph::WholeNCCLMemory_t parameter_wnmt,
                           const void *indice,
                           size_t storage_offset,
                           int64_t indice_count,
                           int64_t embedding_dim,
                           int64_t embedding_stride,
                           int64_t output_stride,
                           CUDAEnvFns cuda_env_fns,
                           cudaStream_t stream) {
  param_t = NormalizeNCCLWMType(param_t);
  DISPATCH_TWO_TYPES(param_t,
                     index_t,
                     WholeMemoryNCCLGatherFunc,
                     output,
                     parameter_wnmt,
                     indice,
                     storage_offset,
                     indice_count,
                     embedding_dim,
                     embedding_stride,
                     output_stride,
                     cuda_env_fns,
                     stream);
}

template<typename ParamT, typename IdxT>
void WholeMemoryNCCLScatterFunc(const void *input,
                                whole_graph::WholeNCCLMemory_t parameter_wnmt,
                                const void *indice,
                                size_t storage_offset,
                                int64_t indice_count,
                                int64_t embedding_dim,
                                int64_t embedding_stride,
                                int64_t input_stride,
                                const CUDAEnvFns &cuda_env_fns,
                                cudaStream_t stream) {
  WMThrustAllocator allocator(cuda_env_fns);
  ParamT *parameter_local_ptr;
  size_t parameter_local_size;
  WnmmpGetLocalMemory(parameter_wnmt, (void **) &parameter_local_ptr, &parameter_local_size);
  parameter_local_size = WnmmpGetChunkSize(parameter_wnmt);
  auto *bcomm = WnmmpGetBootstrapCommunicator(parameter_wnmt);
  const IdxT *indice_ptr = (const IdxT *) indice;
  int64_t local_node_count = parameter_local_size / sizeof(ParamT) / embedding_stride;

  TempMemoryHandle rank_count_ptr_tmh, rank_count_host_tmh, rank_offset_host_tmh, recv_rank_count_host_tmh,
      sorted_indice_tmh, reverse_indice_tmh, recv_ids_tmh, reordered_input_tmh, recv_embedding_tmh;
  int *rank_count = (int *) cuda_env_fns.allocate_temp_fn(bcomm->Size() * sizeof(int), &rank_count_ptr_tmh);
  int *rank_count_host = (int *) cuda_env_fns.allocate_host_temp_fn(bcomm->Size() * sizeof(int), &rank_count_host_tmh);
  int *rank_offset_host = (int *) cuda_env_fns.allocate_host_temp_fn((bcomm->Size() + 1) * sizeof(int), &rank_offset_host_tmh);
  int *recv_rank_count_host = (int *) cuda_env_fns.allocate_host_temp_fn(bcomm->Size() * sizeof(int), &recv_rank_count_host_tmh);
  IdxT *sorted_indice = (IdxT *) cuda_env_fns.allocate_temp_fn(indice_count * sizeof(IdxT), &sorted_indice_tmh);
  int *reverse_indice = (int *) cuda_env_fns.allocate_temp_fn(indice_count * sizeof(int), &reverse_indice_tmh);
  // Exchange node count
  WholeMemoryNCCLNodeCountForRanks(indice_ptr, rank_count, indice_count, bcomm->Size(), local_node_count, stream);
  WM_CUDA_CHECK(cudaMemcpyAsync(rank_count_host, rank_count, bcomm->Size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
  WM_CUDA_CHECK(cudaGetLastError());
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
  bcomm->AllToAll(rank_count_host, sizeof(int), recv_rank_count_host, sizeof(int));
  bcomm->RecordEvent();
  rank_offset_host[0] = 0;
  for (int i = 0; i < bcomm->Size(); i++) {
    rank_offset_host[i + 1] = rank_offset_host[i] + rank_count_host[i];
  }
  bcomm->Synchronize();
  // Exchange ids
  WM_CUDA_CHECK(cudaMemcpyAsync(sorted_indice, indice, sizeof(IdxT) * indice_count, cudaMemcpyDeviceToDevice, stream));
  thrust::sequence(thrust::cuda::par(allocator).on(stream),
                   reverse_indice,
                   reverse_indice + indice_count,
                   0);
  thrust::sort_by_key(thrust::cuda::par(allocator).on(stream),
                      sorted_indice,
                      sorted_indice + indice_count,
                      reverse_indice);
  int total_recv_count = 0;
  for (int i = 0; i < bcomm->Size(); i++) {
    total_recv_count += recv_rank_count_host[i];
  }
  IdxT *recv_indice = (IdxT *) cuda_env_fns.allocate_temp_fn(total_recv_count * sizeof(IdxT), &recv_ids_tmh);
  std::vector<void *> send_ptrs(bcomm->Size()), recv_ptrs(bcomm->Size());
  std::vector<int> send_sizes(bcomm->Size()), recv_sizes(bcomm->Size());
  int64_t all_recv_idx = 0;
  for (int i = 0; i < bcomm->Size(); i++) {
    send_ptrs[i] = sorted_indice + rank_offset_host[i];
    recv_ptrs[i] = recv_indice + all_recv_idx;
    send_sizes[i] = rank_count_host[i] * sizeof(IdxT);
    recv_sizes[i] = recv_rank_count_host[i] * sizeof(IdxT);
    all_recv_idx += recv_rank_count_host[i];
  }
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
  bcomm->AllToAllV((const void **) send_ptrs.data(), send_sizes.data(), recv_ptrs.data(), recv_sizes.data());
  bcomm->RecordEvent();
  bcomm->Synchronize();
  // Local Reorder
  ParamT *reordered_input = (ParamT *) cuda_env_fns.allocate_temp_fn(indice_count * sizeof(ParamT) * embedding_dim, &reordered_input_tmh);
  WholeMemoryGatherFunc<ParamT, ParamT, int>(reordered_input,
                                             (ParamT *) input,
                                             reverse_indice,
                                             0,
                                             indice_count,
                                             embedding_dim,
                                             input_stride,
                                             embedding_dim,
                                             0,
                                             stream);
  WM_CUDA_CHECK(cudaGetLastError());
  ParamT *recv_embedding = (ParamT *) cuda_env_fns.allocate_temp_fn(total_recv_count * sizeof(ParamT) * embedding_dim, &recv_embedding_tmh);
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
  all_recv_idx = 0;
  for (int i = 0; i < bcomm->Size(); i++) {
    send_ptrs[i] = reordered_input + rank_offset_host[i] * embedding_dim;
    recv_ptrs[i] = recv_embedding + all_recv_idx * embedding_dim;
    send_sizes[i] = rank_count_host[i] * embedding_dim * sizeof(ParamT);
    recv_sizes[i] = recv_rank_count_host[i] * embedding_dim * sizeof(ParamT);
    all_recv_idx += recv_rank_count_host[i];
  }
  int64_t all_recv_count = all_recv_idx;
  // AllToAllV for embeddings
  bcomm->AllToAllV((const void **) send_ptrs.data(), send_sizes.data(), recv_ptrs.data(), recv_sizes.data());
  bcomm->RecordEvent();
  bcomm->Synchronize();
  // Local Scatter
  WholeMemoryScatterFunc<ParamT, IdxT>(recv_embedding,
                                       (ParamT *) parameter_local_ptr,
                                       recv_indice,
                                       storage_offset,
                                       all_recv_count,
                                       embedding_dim,
                                       embedding_stride,
                                       embedding_dim,
                                       bcomm->Rank() * local_node_count,
                                       stream);
  WM_CUDA_CHECK(cudaGetLastError());
  CUDA_STREAM_SYNC(cuda_env_fns, stream);

  cuda_env_fns.free_temp_fn(&sorted_indice_tmh);
  cuda_env_fns.free_temp_fn(&rank_count_ptr_tmh);
  cuda_env_fns.free_host_temp_fn(&rank_count_host_tmh);
  cuda_env_fns.free_host_temp_fn(&rank_offset_host_tmh);
  cuda_env_fns.free_host_temp_fn(&recv_rank_count_host_tmh);
  cuda_env_fns.free_temp_fn(&reverse_indice_tmh);
  cuda_env_fns.free_temp_fn(&recv_ids_tmh);
  cuda_env_fns.free_temp_fn(&reordered_input_tmh);
  cuda_env_fns.free_temp_fn(&recv_embedding_tmh);
  allocator.deallocate_all();
}

REGISTER_DISPATCH_TWO_TYPES(WholeMemoryNCCLScatterFunc,
                            WholeMemoryNCCLScatterFunc,
                            SINT,
                            SINT3264)

void WholeMemoryNCCLScatter(WMType param_t,
                            WMType index_t,
                            const void *input,
                            whole_graph::WholeNCCLMemory_t parameter_wnmt,
                            const void *indice,
                            size_t storage_offset,
                            int64_t indice_count,
                            int64_t embedding_dim,
                            int64_t embedding_stride,
                            int64_t input_stride,
                            const CUDAEnvFns &cuda_env_fns,
                            cudaStream_t stream) {
  param_t = NormalizeNCCLWMType(param_t);
  DISPATCH_TWO_TYPES(param_t,
                     index_t,
                     WholeMemoryNCCLScatterFunc,
                     input,
                     parameter_wnmt,
                     indice,
                     storage_offset,
                     indice_count,
                     embedding_dim,
                     embedding_stride,
                     input_stride,
                     cuda_env_fns,
                     stream);
}

template<typename IdxT>
__global__ void CopyAndConvertToLocalIndice(IdxT *output_indices, const IdxT *recv_indice, int unique_count, int64_t local_entry_start) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (tidx >= unique_count) return;
  IdxT idx = recv_indice[tidx];
  assert(idx >= (IdxT) local_entry_start);
  output_indices[tidx] = idx - (IdxT) local_entry_start;
}

template<typename GradT>
__global__ void AggregateGradientKernel(const int *csr_row_ptr,
                                        const int *csr_col_ind,
                                        const GradT *input_grads,
                                        GradT *output_grads,
                                        int unique_count,
                                        int embedding_dim) {
  int row_idx = blockIdx.x;
  int col_start = csr_row_ptr[row_idx];
  int col_end = csr_row_ptr[row_idx + 1];
  int col_count = col_end - col_start;
  int wave_count = DivUp(embedding_dim, blockDim.x);
  float emb_value;
  for (int w = 0; w < wave_count; w++) {
    emb_value = 0.0f;
    int embedding_idx = w * blockDim.x + threadIdx.x;
    for (int col_idx = 0; col_idx < col_count; col_idx++) {
      int col_id = csr_col_ind[col_idx + col_start];
      if (embedding_idx < embedding_dim) emb_value += (float) input_grads[(int64_t) col_id * embedding_dim + embedding_idx];
    }
    output_grads[(int64_t) row_idx * embedding_dim + embedding_idx] = (GradT) emb_value;
  }
}

template<typename GradT, typename IdxT>
void WholeMemoryExchangeEmbeddingGradsFunc(std::function<void *(size_t)> local_indice_allocator,
                                           std::function<void *(size_t)> local_grad_allocator,
                                           const void *sparse_indices,
                                           const void *sparse_grads,
                                           int64_t indice_count,
                                           int64_t embedding_dim,
                                           int64_t embedding_stride,
                                           int64_t total_entry_count,
                                           BootstrapCommunicator *bootstrap_communicator,
                                           const CUDAEnvFns &cuda_env_fns,
                                           cudaStream_t stream) {
  WMThrustAllocator allocator(cuda_env_fns);
  BootstrapCommunicator *bcomm = bootstrap_communicator;
  int64_t local_entry_count = DivUp(total_entry_count, bootstrap_communicator->Size());
  const IdxT *indice_ptr = (const IdxT *) sparse_indices;
  TempMemoryHandle rank_count_ptr_tmh, rank_count_host_tmh, rank_offset_host_tmh, recv_rank_count_host_tmh,
      sorted_indice_tmh, reverse_indice_tmh, recv_ids_tmh, reordered_input_tmh, recv_embedding_tmh;
  int *rank_count = (int *) cuda_env_fns.allocate_temp_fn(bcomm->Size() * sizeof(int), &rank_count_ptr_tmh);
  int *rank_count_host = (int *) cuda_env_fns.allocate_host_temp_fn(bcomm->Size() * sizeof(int), &rank_count_host_tmh);
  int *rank_offset_host = (int *) cuda_env_fns.allocate_host_temp_fn((bcomm->Size() + 1) * sizeof(int), &rank_offset_host_tmh);
  int *recv_rank_count_host = (int *) cuda_env_fns.allocate_host_temp_fn(bcomm->Size() * sizeof(int), &recv_rank_count_host_tmh);
  IdxT *sorted_indice = (IdxT *) cuda_env_fns.allocate_temp_fn(indice_count * sizeof(IdxT), &sorted_indice_tmh);
  int *reverse_indice = (int *) cuda_env_fns.allocate_temp_fn(indice_count * sizeof(int), &reverse_indice_tmh);
  // Exchange node count
  WholeMemoryNCCLNodeCountForRanks(indice_ptr, rank_count, indice_count, bcomm->Size(), local_entry_count, stream);
  WM_CUDA_CHECK(cudaMemcpyAsync(rank_count_host, rank_count, bcomm->Size() * sizeof(int), cudaMemcpyDeviceToHost, stream));
  WM_CUDA_CHECK(cudaGetLastError());
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
  bcomm->AllToAll(rank_count_host, sizeof(int), recv_rank_count_host, sizeof(int));
  bcomm->RecordEvent();
  rank_offset_host[0] = 0;
  for (int i = 0; i < bcomm->Size(); i++) {
    rank_offset_host[i + 1] = rank_offset_host[i] + rank_count_host[i];
  }
  bcomm->Synchronize();
  // Exchange ids
  WM_CUDA_CHECK(cudaMemcpyAsync(sorted_indice, sparse_indices, sizeof(IdxT) * indice_count, cudaMemcpyDeviceToDevice, stream));
  thrust::sequence(thrust::cuda::par(allocator).on(stream),
                   reverse_indice,
                   reverse_indice + indice_count,
                   0);
  thrust::sort_by_key(thrust::cuda::par(allocator).on(stream),
                      sorted_indice,
                      sorted_indice + indice_count,
                      reverse_indice);
  int total_recv_count = 0;
  for (int i = 0; i < bcomm->Size(); i++) {
    total_recv_count += recv_rank_count_host[i];
  }
  IdxT *recv_indice = (IdxT *) cuda_env_fns.allocate_temp_fn(total_recv_count * sizeof(IdxT), &recv_ids_tmh);
  std::vector<void *> send_ptrs(bcomm->Size()), recv_ptrs(bcomm->Size());
  std::vector<int> send_sizes(bcomm->Size()), recv_sizes(bcomm->Size());
  int64_t all_recv_idx = 0;
  for (int i = 0; i < bcomm->Size(); i++) {
    send_ptrs[i] = sorted_indice + rank_offset_host[i];
    recv_ptrs[i] = recv_indice + all_recv_idx;
    send_sizes[i] = rank_count_host[i] * sizeof(IdxT);
    recv_sizes[i] = recv_rank_count_host[i] * sizeof(IdxT);
    all_recv_idx += recv_rank_count_host[i];
  }
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
  bcomm->AllToAllV((const void **) send_ptrs.data(), send_sizes.data(), recv_ptrs.data(), recv_sizes.data());
  bcomm->RecordEvent();
  bcomm->Synchronize();
  // Local Reorder
  GradT *reordered_input = (GradT *) cuda_env_fns.allocate_temp_fn(indice_count * sizeof(GradT) * embedding_dim, &reordered_input_tmh);
  WholeMemoryGatherFunc<GradT, GradT, int>(reordered_input,
                                           (GradT *) sparse_grads,
                                           reverse_indice,
                                           0,
                                           indice_count,
                                           embedding_dim,
                                           embedding_stride,
                                           embedding_dim,
                                           0,
                                           stream);
  WM_CUDA_CHECK(cudaGetLastError());
  GradT *recv_embedding = (GradT *) cuda_env_fns.allocate_temp_fn(total_recv_count * sizeof(GradT) * embedding_dim, &recv_embedding_tmh);
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
  cuda_env_fns.free_temp_fn(&sorted_indice_tmh);
  cuda_env_fns.free_temp_fn(&reverse_indice_tmh);
  all_recv_idx = 0;
  for (int i = 0; i < bcomm->Size(); i++) {
    send_ptrs[i] = reordered_input + rank_offset_host[i] * embedding_dim;
    recv_ptrs[i] = recv_embedding + all_recv_idx * embedding_dim;
    send_sizes[i] = rank_count_host[i] * embedding_dim * sizeof(GradT);
    recv_sizes[i] = recv_rank_count_host[i] * embedding_dim * sizeof(GradT);
    all_recv_idx += recv_rank_count_host[i];
  }
  // AllToAllV for embeddings
  bcomm->AllToAllV((const void **) send_ptrs.data(), send_sizes.data(), recv_ptrs.data(), recv_sizes.data());
  bcomm->RecordEvent();
  bcomm->Synchronize();
  // Aggregation
  TempMemoryHandle unique_indice_tmh;
  int *unique_indice = (int *) cuda_env_fns.allocate_temp_fn((total_recv_count + 1) * sizeof(int), &unique_indice_tmh);

  reverse_indice = (int *) cuda_env_fns.allocate_temp_fn(total_recv_count * sizeof(int), &reverse_indice_tmh);
  thrust::sequence(thrust::cuda::par(allocator).on(stream),
                   reverse_indice,
                   reverse_indice + total_recv_count,
                   0);
  thrust::sequence(thrust::cuda::par(allocator).on(stream),
                   unique_indice,
                   unique_indice + total_recv_count + 1,
                   0);
  thrust::sort_by_key(thrust::cuda::par(allocator).on(stream),
                      recv_indice,
                      recv_indice + total_recv_count,
                      reverse_indice);
  auto its = thrust::unique_by_key(thrust::cuda::par(allocator).on(stream),
                                   recv_indice,
                                   recv_indice + total_recv_count,
                                   unique_indice);
  CUDA_STREAM_SYNC(cuda_env_fns, stream);
  int64_t unique_count = its.first - recv_indice;
  IdxT *output_indices = (IdxT *) local_indice_allocator(unique_count);
  GradT *output_grads = (GradT *) local_grad_allocator(unique_count * embedding_dim);
  CopyAndConvertToLocalIndice<IdxT><<<DivUp(unique_count, 64), 64, 0, stream>>>(output_indices,
                                                                                recv_indice,
                                                                                unique_count,
                                                                                bcomm->Rank() * local_entry_count);
  WM_CUDA_CHECK(cudaMemcpyAsync(unique_indice + unique_count,
                                unique_indice + total_recv_count,
                                sizeof(int),
                                cudaMemcpyDeviceToDevice,
                                stream));

  int thread_count = embedding_dim;
  if (thread_count > 256) thread_count = 256;
  AggregateGradientKernel<GradT><<<unique_count, thread_count, 0, stream>>>(unique_indice, reverse_indice, recv_embedding, output_grads, unique_count, embedding_dim);
  WM_CUDA_CHECK(cudaGetLastError());
  CUDA_STREAM_SYNC(cuda_env_fns, stream);

  cuda_env_fns.free_temp_fn(&rank_count_ptr_tmh);
  cuda_env_fns.free_host_temp_fn(&rank_count_host_tmh);
  cuda_env_fns.free_host_temp_fn(&rank_offset_host_tmh);
  cuda_env_fns.free_host_temp_fn(&recv_rank_count_host_tmh);
  //cuda_env_fns.free_temp_fn(&sorted_indice_tmh);
  cuda_env_fns.free_temp_fn(&reverse_indice_tmh);
  cuda_env_fns.free_temp_fn(&recv_ids_tmh);
  cuda_env_fns.free_temp_fn(&reordered_input_tmh);
  cuda_env_fns.free_temp_fn(&recv_embedding_tmh);
  cuda_env_fns.free_temp_fn(&unique_indice_tmh);
  allocator.deallocate_all();
}

REGISTER_DISPATCH_TWO_TYPES(WholeMemoryExchangeEmbeddingGradsFunc,
                            WholeMemoryExchangeEmbeddingGradsFunc,
                            HALF_FLOAT,
                            SINT3264)

void WholeMemoryExchangeEmbeddingGrads(WMType index_t,
                                       WMType grad_t,
                                       std::function<void *(size_t)> local_indice_allocator,
                                       std::function<void *(size_t)> local_grad_allocator,
                                       const void *sparse_indices,
                                       const void *sparse_grads,
                                       int64_t indice_count,
                                       int64_t embedding_dim,
                                       int64_t embedding_stride,
                                       int64_t total_entry_count,
                                       BootstrapCommunicator *bootstrap_communicator,
                                       const CUDAEnvFns &cuda_env_fns,
                                       cudaStream_t stream) {
  DISPATCH_TWO_TYPES(grad_t,
                     index_t,
                     WholeMemoryExchangeEmbeddingGradsFunc,
                     local_indice_allocator,
                     local_grad_allocator,
                     sparse_indices,
                     sparse_grads,
                     indice_count,
                     embedding_dim,
                     embedding_stride,
                     total_entry_count,
                     bootstrap_communicator,
                     cuda_env_fns,
                     stream);
}

template<typename EmbType, typename StateType, typename UpdateAlgo>
__global__ void WholeMemoryEmbeddingLocalApplyGradientsKernel(OptimizerInfo opt_info,
                                                              const int *local_update_list,
                                                              EmbType *embedding,
                                                              const EmbType *grad,
                                                              StateType *per_element_state_0,
                                                              StateType *per_element_state_1,
                                                              StateType *per_embedding_state,
                                                              int64_t update_count,
                                                              int64_t embedding_table_size,
                                                              int embedding_dim,
                                                              int embedding_stride,
                                                              int grad_per_element_state_stride) {
  int entry_idx = blockIdx.x;
  int local_entry_id = local_update_list[entry_idx];
  UpdateAlgo algo;
  algo.run(opt_info,
           local_entry_id,
           entry_idx,
           embedding,
           grad,
           per_element_state_0,
           per_element_state_1,
           per_embedding_state,
           embedding_dim,
           embedding_stride,
           grad_per_element_state_stride);
}
template<typename EmbType, typename StateType>
void WholeMemoryEmbeddingLocalApplyGradientsFunc(OptimizerInfo opt_info,
                                                 const int *local_update_list,
                                                 void *embedding,
                                                 const void *grad,
                                                 void *per_element_state_0,
                                                 void *per_element_state_1,
                                                 void *per_embedding_state,
                                                 int64_t update_count,
                                                 int64_t embedding_table_size,
                                                 int embedding_dim,
                                                 int embedding_stride,
                                                 int grad_per_element_state_stride,
                                                 cudaStream_t stream) {
  int block_count = update_count;
  int thread_countx = AlignUp(embedding_dim, 32);
  if (thread_countx > 256) thread_countx = 256;
  void (*kernel_fn)(OptimizerInfo,
                    const int *,
                    EmbType *,
                    const EmbType *,
                    StateType *,
                    StateType *,
                    StateType *,
                    int64_t,
                    int64_t,
                    int,
                    int,
                    int) = nullptr;
  switch (opt_info.type) {
    case OPT_TYPE_SGD: {
      kernel_fn =
          WholeMemoryEmbeddingLocalApplyGradientsKernel<EmbType, StateType, SGDSparseEmbOptimizer<EmbType, StateType>>;
      break;
    }
    case OPT_TYPE_LAZY_ADAM: {
      kernel_fn =
          WholeMemoryEmbeddingLocalApplyGradientsKernel<EmbType, StateType, LazyAdamSparseEmbOptimizer<EmbType, StateType>>;
      break;
    }
    case OPT_TYPE_ADAGRAD: {
      kernel_fn =
          WholeMemoryEmbeddingLocalApplyGradientsKernel<EmbType, StateType, AdaGradSparseEmbOptimizer<EmbType, StateType>>;
      break;
    }
    case OPT_TYPE_RMSPROP: {
      kernel_fn =
          WholeMemoryEmbeddingLocalApplyGradientsKernel<EmbType, StateType, RMSPropSparseEmbOptimizer<EmbType, StateType>>;
      break;
    }
    default: {
      fprintf(stderr, "Optimizer type %d not supported.\n", (int) opt_info.type);
      abort();
    }
  }
  int thread_block = thread_countx;
  kernel_fn<<<block_count, thread_block, 0, stream>>>(opt_info,
                                                      local_update_list,
                                                      (EmbType *) embedding,
                                                      (const EmbType *) grad,
                                                      (StateType *) per_element_state_0,
                                                      (StateType *) per_element_state_1,
                                                      (StateType *) per_embedding_state,
                                                      update_count,
                                                      embedding_table_size,
                                                      embedding_dim,
                                                      embedding_stride,
                                                      grad_per_element_state_stride);
  WM_CUDA_CHECK(cudaGetLastError());
}

REGISTER_DISPATCH_TWO_TYPES(WholeMemoryEmbeddingLocalApplyGradientsFunc,
                            WholeMemoryEmbeddingLocalApplyGradientsFunc,
                            HALF_FLOAT,
                            HALF_FLOAT)

void WholeMemoryEmbeddingLocalApplyGradients(WMType grad_t,
                                             WMType state_t,
                                             OptimizerInfo opt_info,
                                             const int *update_list,
                                             void *embedding,
                                             void *grad,
                                             void *per_element_state_0,
                                             void *per_element_state_1,
                                             void *per_embedding_state,
                                             int64_t update_count,
                                             int64_t embedding_table_size,
                                             int64_t embedding_dim,
                                             int64_t embedding_stride,
                                             int64_t grad_per_element_state_stride,
                                             cudaStream_t stream) {
  DISPATCH_TWO_TYPES(grad_t, state_t, WholeMemoryEmbeddingLocalApplyGradientsFunc,
                     opt_info,
                     update_list,
                     embedding,
                     grad,
                     per_element_state_0,
                     per_element_state_1,
                     per_embedding_state,
                     update_count,
                     embedding_table_size,
                     embedding_dim,
                     embedding_stride,
                     grad_per_element_state_stride,
                     stream);
}

}// namespace whole_graph