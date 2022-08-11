#include "whole_chunked_pytorch_tensor.h"

namespace whole_graph {

namespace pytorch {

StorageImpl::StorageImpl(size_t size_bytes, size_t min_granularity, const std::vector<int>& ranks)
    : size_bytes_(size_bytes) {
  whole_graph::WcmmpMalloc(&wcmt_, size_bytes, min_granularity, ranks.data(), ranks.size());
}

StorageImpl::~StorageImpl() {
  whole_graph::WcmmpFree(wcmt_);
}

void ChunkedTensorImpl::set_sizes_and_strides(torch::IntArrayRef new_size, torch::IntArrayRef new_stride) {
  TORCH_CHECK(
      new_size.size() == new_stride.size(),
      "dimensionality of sizes (",
      new_size.size(),
      ") must match dimensionality of strides (",
      new_stride.size(),
      ")");
  TORCH_CHECK(new_size.size() <= 2,
              "[ChunkedTensorImpl::set_sizes_and_strides] new_size.size()=",
              new_size.size(),
              ", current only size<=2 is supported.");
  TORCH_CHECK(new_size.size() <= 2,
              "[ChunkedTensorImpl::set_sizes_and_strides] new_size.size()=",
              new_size.size(),
              ", current only new_dim<=2 is supported.");
  const auto new_dim = new_size.size();

  sizes_and_strides_.set_sizes(new_size);

  if (new_dim > 0) {
    for (size_t dim = new_dim - 1;; dim--) {
      if (new_stride[dim] >= 0) {
        sizes_and_strides_.stride_at_unchecked(dim) = new_stride[dim];
      } else {
        // XXX: This behavior is surprising and may need to be removed to
        // support negative strides. Some pytorch functions rely on it:
        // for example, torch.cat (run TestTorch.test_cat_empty).
        if (dim == new_dim - 1) {
          sizes_and_strides_.stride_at_unchecked(dim) = 1;
        } else {
          // Keep stride monotonically increasing to match NumPy.
          sizes_and_strides_.stride_at_unchecked(dim) =
              std::max<int64_t>(
                  sizes_and_strides_.size_at_unchecked(dim + 1), 1) *
                  sizes_and_strides_.stride_at_unchecked(dim + 1);
        }
      }
      if (dim == 0)
        break;
    }
  }
  refresh_numel();
}

}

}