#include "whole_graph_embedding.h"

#include <cuda_runtime_api.h>

#include <functional>

#include "whole_graph.h"
#include "whole_chunked_memory.cuh"
#include "file_utils.h"
#include "whole_graph_communicator.h"
#include "macros.h"

namespace whole_graph {

template<typename EmbType, typename SrcType, typename WMEmbType = EmbType>
__global__ void CopyEmbeddingFromMemoryBlockKernel(WMEmbType* wm_embedding,
                                                   int64_t storage_offset,
                                                   int64_t table_size,
                                                   int64_t embedding_dim,
                                                   int64_t embedding_stride,
                                                   const SrcType* src_embedding,
                                                   int64_t src_start_id,
                                                   int64_t src_stride) {
  int block_idx = blockIdx.x;
  int tidx = threadIdx.x;
  whole_graph::PtrGen<WMEmbType, EmbType> ptr_gen(wm_embedding, storage_offset);
  auto emb_id = block_idx + src_start_id;
  assert(emb_id >= 0 && emb_id < table_size);
  EmbType* output_ptr = ptr_gen.At(emb_id * embedding_stride);
  const SrcType* input_ptr = src_embedding + block_idx * src_stride;
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
  auto src_embedding = (const SrcType*) src;
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

void WmmpLoadChunkedEmbeddingFromMemory(void *wm_embedding,
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
                     CopyChunkedEmbeddingFromMemoryBlock,
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

static void WmmpEmbeddingTraverseFileListsByPrefix(const std::string &file_prefix,
                                                   int64_t embedding_dim,
                                                   int64_t table_size,
                                                   WMType src_type,
                                                   std::function<void(const void*, int64_t, int64_t)> copy_block_fn,
                                                   const int *ranks,
                                                   int rank_count) {
  std::vector<std::string> filelist;
  if (!GetPartFileListFromPrefix(file_prefix, &filelist)) {
    std::cerr << "GetPartFileListFromPrefix from prefix " << file_prefix << " failed.\n";
    abort();
  }
  const int64_t magic_num = 0x10adeebdf70e1157LL;
  CollCheckAllSame<int64_t>(magic_num, ranks, rank_count);
  CollCheckAllSame(filelist.size(), ranks, rank_count);
  size_t src_embedding_bytes = embedding_dim * GetWMTSize(src_type);
  std::vector<int64_t> file_start_vec_ids, vec_counts;
  int64_t start_vec = 0;
  for (const auto& filename : filelist) {
    size_t file_size = StatFileSize(filename);
    if (file_size % src_embedding_bytes != 0) {
      std::cerr << "File " << filename << " size is " << file_size << ", but type " << GetWMTName(src_type)
          << ", embedding_dim=" << embedding_dim << std::endl;
      abort();
    }
    int64_t vec_count = file_size / src_embedding_bytes;
    CollCheckAllSame(vec_count, ranks, rank_count);
    file_start_vec_ids.push_back(start_vec);
    start_vec += vec_count;
    vec_counts.push_back(vec_count);
  }
  assert (start_vec == table_size);
  int64_t local_size, local_start;
  local_start = WmmpCollOffsetAndSize(table_size, reinterpret_cast<size_t *>(&local_size), ranks, rank_count);
  const int64_t kMemoryBlockSize = 64 * 1024 * 1024;
  int64_t max_vec_count = kMemoryBlockSize / src_embedding_bytes;
  void *vec_h;
  WM_CUDA_CHECK(cudaMallocHost(&vec_h, kMemoryBlockSize));
  for (int fidx = 0; fidx < (int)filelist.size(); fidx++) {
    int64_t file_start_vec_id = file_start_vec_ids[fidx];
    int64_t file_vec_count = vec_counts[fidx];
    if (file_start_vec_id + file_vec_count < local_start) continue;
    if (file_start_vec_id >= local_start + local_size) break;
    FILE* fp = fopen(filelist[fidx].c_str(), "rb");
    if (fp == nullptr) {
      std::cerr << "Open file " << filelist[fidx] << " failed.\n";
      abort();
    }
    int64_t vec_offset_in_file = 0;
    if (file_start_vec_id < local_start) {
      vec_offset_in_file += local_start - file_start_vec_id;
      assert(fseeko64(fp, vec_offset_in_file * src_embedding_bytes, SEEK_SET) == 0);
    }
    while (vec_offset_in_file < file_vec_count && vec_offset_in_file + file_start_vec_id < local_start + local_size) {
      int64_t read_vec_count = std::min(file_vec_count - vec_offset_in_file, local_start + local_size - file_start_vec_id - vec_offset_in_file);
      if (read_vec_count > max_vec_count) read_vec_count = max_vec_count;
      assert(fread(vec_h, src_embedding_bytes, read_vec_count, fp) == read_vec_count);
      int64_t start_id = file_start_vec_ids[fidx] + vec_offset_in_file;
      copy_block_fn(vec_h, start_id, read_vec_count);
      vec_offset_in_file += read_vec_count;
    }
    fclose(fp);
  }
  WM_CUDA_CHECK(cudaFreeHost(vec_h));
}

void WmmpLoadEmbeddingFromFilelists(void *wm_embedding,
                                    int64_t storage_offset,
                                    int64_t table_size,
                                    int64_t embedding_dim,
                                    int64_t embedding_stride,
                                    const std::string &file_prefix,
                                    WMType src_type,
                                    WMType emb_type,
                                    cudaStream_t stream,
                                    const int* ranks,
                                    int rank_count) {
  WmmpEmbeddingTraverseFileListsByPrefix(file_prefix,
                                         embedding_dim,
                                         table_size,
                                         src_type,
                                         [=](
                                             const void *buf,
                                             int64_t start_id,
                                             int64_t src_size) {
                                           WmmpLoadEmbeddingFromMemory(wm_embedding,
                                                                       storage_offset,
                                                                       table_size,
                                                                       embedding_dim,
                                                                       embedding_stride,
                                                                       buf,
                                                                       embedding_dim,
                                                                       src_type,
                                                                       emb_type,
                                                                       start_id,
                                                                       src_size,
                                                                       stream);
                                         },
                                         ranks,
                                         rank_count);
}

void WmmpLoadChunkedEmbeddingFromFilelists(void *wm_embedding,
                                           int64_t storage_offset,
                                           int64_t table_size,
                                           int64_t embedding_dim,
                                           int64_t embedding_stride,
                                           const std::string &file_prefix,
                                           WMType src_type,
                                           WMType emb_type,
                                           cudaStream_t stream,
                                           const int *ranks,
                                           int rank_count) {
  int dev_id;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  WholeChunkedMemoryHandle* wmth = GetDeviceChunkedHandle((WholeChunkedMemory_t)wm_embedding, dev_id);
  WmmpEmbeddingTraverseFileListsByPrefix(file_prefix,
                                         embedding_dim,
                                         table_size,
                                         src_type,
                                         [=](
                                             const void *buf,
                                             int64_t start_id,
                                             int64_t src_size) {
                                           WmmpLoadChunkedEmbeddingFromMemory(wmth,
                                                                              storage_offset,
                                                                              table_size,
                                                                              embedding_dim,
                                                                              embedding_stride,
                                                                              buf,
                                                                              embedding_dim,
                                                                              src_type,
                                                                              emb_type,
                                                                              start_id,
                                                                              src_size,
                                                                              stream);
                                         },
                                         ranks,
                                         rank_count);
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

template <typename T>
__device__ __forceinline__ void MovTypedData(T* to, const T* from) {
  *to = *from;
}
template <int DataSize>
__device__ __forceinline__ void MovData(void* to, const void* from) {
  char* ptr_to = (char*)to;
  const char* ptr_from = (const char*)from;
  for (int i = 0; i < DataSize; i++) {
    ptr_to[i] = ptr_from[i];
  }
}
template<>
__device__ __forceinline__ void MovData<1>(void* to, const void* from) {
  MovTypedData((int8_t*)to, (const int8_t*)from);
}
template<>
__device__ __forceinline__ void MovData<2>(void* to, const void* from) {
  MovTypedData((int16_t*)to, (const int16_t*)from);
}
template<>
__device__ __forceinline__ void MovData<4>(void* to, const void* from) {
  MovTypedData((int32_t*)to, (const int32_t*)from);
}
template<>
__device__ __forceinline__ void MovData<8>(void* to, const void* from) {
  MovTypedData((int64_t*)to, (const int64_t*)from);
}
template<>
__device__ __forceinline__ void MovData<16>(void* to, const void* from) {
  MovTypedData((int4*)to, (const int4*)from);
}
template<>
__device__ __forceinline__ void MovData<32>(void* to, const void* from) {
  MovTypedData((int4*)to, (const int4*)from);
  MovTypedData(((int4*)to) + 1, ((const int4*)from) + 1);
}
template<>
__device__ __forceinline__ void MovData<64>(void* to, const void* from) {
  MovTypedData((int4*)to, (const int4*)from);
  MovTypedData(((int4*)to) + 1, ((const int4*)from) + 1);
  MovTypedData(((int4*)to) + 2, ((const int4*)from) + 2);
  MovTypedData(((int4*)to) + 3, ((const int4*)from) + 3);
}

template <typename OutputT, typename ParamT, typename IdxT, typename ParamHandleT, int Alignment = 1>
__global__ void WholeMemoryGatherKernel(OutputT *__restrict__ output,
                                        const ParamHandleT *__restrict__ parameter,
                                        const IdxT *__restrict__ indice,
                                        size_t storage_offset,
                                        int64_t embedding_dim,
                                        int64_t embedding_stride,
                                        int64_t output_stride) {
  IdxT idx = indice[blockIdx.x];
  whole_graph::PtrGen<const ParamHandleT, ParamT> ptr_gen(parameter, storage_offset);
  const ParamT *param_ptr = ptr_gen.At((size_t) idx * embedding_stride);
  ParamT tmp[Alignment];
  OutputT cvt_tmp[Alignment];
  OutputT *output_ptr = output + (size_t) blockIdx.x * output_stride;
  for (int i = threadIdx.x * Alignment; i < embedding_dim; i += blockDim.x * Alignment) {
    MovData<Alignment * sizeof(ParamT)>(&tmp[0], &param_ptr[i]);
    for (int a = 0; a < Alignment; a++) cvt_tmp[a] = (OutputT)tmp[a];
    MovData<Alignment * sizeof(OutputT)>(&output_ptr[i], &cvt_tmp[0]);
  }
}

template <typename OutputT, typename ParamT>
int DetermineAlignmentEltCount(void* output, size_t storage_elt_offset, int64_t embedding_dim, int64_t embedding_stride, int64_t output_stride) {
  int src_alignment = 16 / sizeof(ParamT);
  for (; src_alignment > 1; src_alignment /= 2) {
    if (storage_elt_offset % src_alignment == 0 && embedding_dim % src_alignment == 0 && embedding_stride % src_alignment == 0)
      break;
  }
  int dst_alignment = 16 / sizeof(OutputT);
  for (; dst_alignment > 1; dst_alignment /= 2) {
    if ((int64_t)output % (dst_alignment * sizeof(OutputT)) == 0 && embedding_dim % dst_alignment == 0 && output_stride % dst_alignment == 0)
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
                           cudaStream_t stream) {
  int alignment = DetermineAlignmentEltCount<OutputT, ParamT>(output, storage_offset, embedding_dim, embedding_stride, output_stride);
  int thread_count = embedding_dim / alignment;
  if (thread_count > 256) thread_count = 256;
  void (*kernel_fn)(OutputT *, const ParamT *, const IdxT *, size_t, int64_t, int64_t, int64_t) = nullptr;
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
      output_stride);
  WM_CUDA_CHECK(cudaGetLastError());
}
REGISTER_DISPATCH_THREE_TYPES(WholeMemoryGatherFunc, WholeMemoryGatherFunc, HALF_FLOAT_DOUBLE, HALF_FLOAT_DOUBLE, SINT3264)

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
  int alignment = DetermineAlignmentEltCount<OutputT, ParamT>(output, storage_offset, embedding_dim, embedding_stride, output_stride);
  int thread_count = embedding_dim / alignment;
  if (thread_count > 256) thread_count = 256;
  void (*kernel_fn)(OutputT *, const WholeChunkedMemoryHandle *, const IdxT *, size_t, int64_t, int64_t, int64_t) = nullptr;
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
  WholeChunkedMemoryHandle *parameter_handle = GetDeviceChunkedHandle((WholeChunkedMemory_t)parameter, dev_id);
  kernel_fn<<<indice_count, thread_count, 0, stream>>>(
      (OutputT *) output,
      (const WholeChunkedMemoryHandle *) parameter_handle,
      (const IdxT *) indice,
      storage_offset,
      embedding_dim,
      embedding_stride,
      output_stride);
  WM_CUDA_CHECK(cudaGetLastError());
}

REGISTER_DISPATCH_THREE_TYPES(WholeMemoryChunkedGatherFunc, WholeMemoryChunkedGatherFunc, HALF_FLOAT_DOUBLE, HALF_FLOAT_DOUBLE, SINT3264)

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
                       stream);
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