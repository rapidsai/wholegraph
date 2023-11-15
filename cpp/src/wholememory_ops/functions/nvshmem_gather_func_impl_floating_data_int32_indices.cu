#ifdef WITH_NVSHMEM_SUPPORT

#include "nvshmem_gather_scatter_func.cuh"
#include <wholememory/wholememory.h>

#include "logger.hpp"
#include "wholememory_ops/register.hpp"
namespace wholememory_ops {

template <typename EmbeddingT, typename OutputT>
void nvshmem_gather_floating_int32_temp_func(wholememory_comm_t wm_comm,
                                             wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
                                             wholememory_matrix_description_t embedding_desc,
                                             const void* indices,
                                             int64_t indice_count,
                                             void* output,
                                             void* temp_output,
                                             wholememory_matrix_description_t output_desc,
                                             size_t embedding_entry_count_per_rank,
                                             wholememory_env_func_t* p_env_fns,
                                             cudaStream_t stream)
{
  nvshmem_gather_temp_get_mem_sort_idx_func<EmbeddingT, int32_t, OutputT>(
    wm_comm,
    embeding_nvshmem_ptr,
    embedding_desc,
    indices,
    indice_count,
    output,
    temp_output,
    output_desc,
    embedding_entry_count_per_rank,
    p_env_fns,
    stream);
}

REGISTER_DISPATCH_TWO_TYPES(NvshmemGatherFuncFloatingInt32,
                            nvshmem_gather_floating_int32_temp_func,
                            HALF_FLOAT_DOUBLE,
                            HALF_FLOAT_DOUBLE)

wholememory_error_code_t nvshmem_gather_floating_int32_func(
  wholememory_comm_t wm_comm,
  wholememory_nvshmem_ref_t embeding_nvshmem_ptr,
  wholememory_matrix_description_t embedding_desc,
  const void* indices,
  wholememory_array_description_t indices_desc,
  void* output,
  void* temp_output,
  wholememory_matrix_description_t output_desc,
  size_t embedding_entry_count_per_rank,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  try {
    WHOLEMEMORY_CHECK(wholememory_dtype_is_floating_number(embedding_desc.dtype));
    WHOLEMEMORY_CHECK(wholememory_dtype_is_floating_number(output_desc.dtype));
    WHOLEMEMORY_CHECK(indices_desc.dtype == WHOLEMEMORY_DT_INT);

    DISPATCH_TWO_TYPES(embedding_desc.dtype,
                       output_desc.dtype,
                       NvshmemGatherFuncFloatingInt32,
                       wm_comm,
                       embeding_nvshmem_ptr,
                       embedding_desc,
                       indices,
                       indices_desc.size,
                       output,
                       temp_output,
                       output_desc,
                       embedding_entry_count_per_rank,
                       p_env_fns,
                       stream);
  } catch (const wholememory::cuda_error& wle) {
    WHOLEMEMORY_ERROR("gather CUDA LOGIC Error %s\n", wle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    WHOLEMEMORY_ERROR("gather CUDA LOGIC Error %s\n", le.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}
};  // namespace wholememory_ops

#endif
