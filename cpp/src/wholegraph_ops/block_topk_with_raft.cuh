#pragma once
#include <raft/matrix/detail/select_k-inl.cuh>
namespace wholegraph_ops {
template <typename T, typename IdxT>
constexpr auto calc_smem_size_for_block_wide(int num_of_warp, int k) -> int
{
  return raft::Pow2<256>::roundUp(raft::ceildiv(num_of_warp, 2) * sizeof(T) * k) +
         raft::ceildiv(num_of_warp, 2) * sizeof(IdxT) * k;
}

template <typename KeyT,
          int BLOCK_SIZE,
          int ITEMS_PER_THREAD,
          int MAXK,
          bool ASCENDING  = false,
          typename ValueT = cub::NullType,
          template <int, bool, typename, typename> class WarpSortClassT =
            raft::matrix::detail::select::warpsort::warp_sort_distributed_ext>
class BlockTopkRaftWarpSort {
  static_assert(MAXK <= raft::matrix::detail::select::warpsort::kMaxCapacity,
                "MAXK should be smaller than warpsort::kMaxCapacity");
  static_assert(MAXK >= 1 && raft::is_a_power_of_two(MAXK),
                "MAXK should >=1 and is a power of two ");

  using bq_t = raft::matrix::detail::select::warpsort::
    block_sort<WarpSortClassT, MAXK, ASCENDING, KeyT, ValueT>;

  static constexpr int WARP_SIZE = 32;
  static constexpr int CAL_SMEM_SIZE =
    calc_smem_size_for_block_wide<KeyT, ValueT>(BLOCK_SIZE / WARP_SIZE, MAXK);
  static constexpr int SMEM_REQUIRED = bq_t::queue_t::mem_required(BLOCK_SIZE);
  struct _TempStorage {
    union {
      __align__(256) uint8_t smem_buf_bytes0[CAL_SMEM_SIZE];
      __align__(256) uint8_t smem_buf_bytes1[SMEM_REQUIRED];
      struct {
        KeyT store_keys[MAXK];
        ValueT store_values[MAXK];
      };
    };
  };

 public:
  struct TempStorage : cub::Uninitialized<_TempStorage> {};

  __device__ __forceinline__ BlockTopkRaftWarpSort(TempStorage& temp_storage)
    : temp_storage_{temp_storage.Alias()}, tid_(threadIdx.x){};

  __device__ __forceinline__ void TopKToStriped(KeyT (&keys)[ITEMS_PER_THREAD],
                                                ValueT (&values)[ITEMS_PER_THREAD],
                                                const int k,
                                                const int valid_count)
  {
    bq_t queue(k, temp_storage_.smem_buf_bytes1);

#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      KeyT key = (i * BLOCK_SIZE + tid_) < valid_count
                   ? keys[i]
                   : WarpSortClassT<MAXK, ASCENDING, KeyT, ValueT>::kDummy;
      queue.add(key, values[i]);
    }
    queue.done(temp_storage_.smem_buf_bytes0);
    __syncthreads();
    queue.store(temp_storage_.store_keys, temp_storage_.store_values);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      int id = i * BLOCK_SIZE + tid_;
      if (id < k) {
        keys[i]   = temp_storage_.store_keys[id];
        values[i] = temp_storage_.store_values[id];
      }
    }
  }

  __device__ __forceinline__ void TopKToStriped(KeyT (&keys)[ITEMS_PER_THREAD],

                                                const int k,
                                                const int valid_count)
  {
    bq_t queue(k, temp_storage_.smem_buf_bytes1);

#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      KeyT key = (i * BLOCK_SIZE + tid_) < valid_count
                   ? keys[i]
                   : WarpSortClassT<MAXK, ASCENDING, KeyT, ValueT>::kDummy;
      queue.add(key, i);
    }
    queue.done(temp_storage_.smem_buf_bytes0);
    __syncthreads();

    queue.store(temp_storage_.store_keys, temp_storage_.store_values);
    __syncthreads();

#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
      int id = i * BLOCK_SIZE + tid_;
      if (id < k) { keys[i] = temp_storage_.store_keys[id]; }
    }
  }

 private:
  _TempStorage& temp_storage_;
  int tid_;
};

};  // namespace wholegraph_ops
