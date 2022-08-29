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
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/script.h>

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../macros.h"
#include "gnn_ops.h"
#include "graph_builder.h"
#include "pytorch_cuda_env_fns.h"
#include "whole_chunked_memory.h"
#include "whole_chunked_pytorch_tensor.h"
#include "whole_memory.h"
#include "whole_memory_embedding.h"
#include "whole_memory_graph.h"
#include "whole_nccl_pytorch_tensor.h"

namespace {

c10::Device GetCurrentCUDADevice() {
  int dev_id = -1;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  return {c10::DeviceType::CUDA, (int8_t) dev_id};
}

struct WholeMemoryInfo {
  c10::weak_intrusive_ptr<c10::StorageImpl> storage_impl{c10::intrusive_ptr<c10::StorageImpl>()};
  bool is_unified = false;
};

std::mutex wholememory_info_mutex;
std::unordered_map<void *, WholeMemoryInfo> wholememory_info_map;

void DeleteWholeMemoryMemory(void *ptr) {
  std::unique_lock<std::mutex> lock(wholememory_info_mutex);
  whole_graph::WmmpFree(ptr);
  wholememory_info_map.erase(ptr);
}

class WholeMemoryDeviceAllocator : public c10::Allocator {
 public:
  WholeMemoryDeviceAllocator() = default;
  static c10::DataPtr allocate_with_comm(size_t nbytes, whole_graph::BootstrapCommunicator *bc_ptr) {
    void *ptr = nullptr;
    whole_graph::WmmpMalloc(&ptr, nbytes, bc_ptr);
    c10::Device device = GetCurrentCUDADevice();
    //(void* data, void* ctx, DeleterFnPtr ctx_deleter, Device device)
    return {ptr, ptr, &DeleteWholeMemoryMemory, device};
  }
  c10::DataPtr allocate(size_t nbytes) const override {
    c10::Device device = GetCurrentCUDADevice();
    fprintf(stderr, "Only allocate_with_comm should be called.\n");
    abort();
    return {nullptr, nullptr, &DeleteWholeMemoryMemory, device};
  }
};

WholeMemoryDeviceAllocator *GetWholeMemoryDeviceAllocator() {
  static WholeMemoryDeviceAllocator whole_memory_device_allocator;
  return &whole_memory_device_allocator;
}

class WholeMemoryUnifiedAllocator : public c10::Allocator {
 public:
  WholeMemoryUnifiedAllocator() = default;
  static c10::DataPtr allocate_with_comm(size_t nbytes, whole_graph::BootstrapCommunicator *bc_ptr) {
    void *ptr = nullptr;
    whole_graph::WmmpMallocHost(&ptr, nbytes, bc_ptr);
    c10::Device device = GetCurrentCUDADevice();
    //(void* data, void* ctx, DeleterFnPtr ctx_deleter, Device device)
    return {ptr, ptr, &DeleteWholeMemoryMemory, device};
  }
  c10::DataPtr allocate(size_t nbytes) const override {
    c10::Device device = GetCurrentCUDADevice();
    fprintf(stderr, "Only allocate_with_comm should be called.\n");
    abort();
    return {nullptr, nullptr, &DeleteWholeMemoryMemory, device};
  }
};

WholeMemoryUnifiedAllocator *GetWholeMemoryUnifiedAllocator() {
  static WholeMemoryUnifiedAllocator whole_memory_unified_allocator;
  return &whole_memory_unified_allocator;
}

size_t CheckSizesAndStrides(const std::vector<int64_t> &sizes,
                            const std::vector<int64_t> &strides,
                            std::vector<int64_t> *fixed_strides) {
  size_t dim = sizes.size();
  size_t total_size = 1;
  if (!strides.empty()) {
    WM_CHECK(sizes.size() == strides.size());
    for (size_t i = 0; i < dim; i++) {
      WM_CHECK(strides[dim - i - 1] >= (int64_t) total_size);
      WM_CHECK(sizes[dim - i - 1] > 0);
      total_size *= sizes[dim - i - 1];
    }
    *fixed_strides = strides;
  } else {
    fixed_strides->resize(dim);
    for (size_t i = 0; i < dim; i++) {
      WM_CHECK(sizes[dim - i - 1] > 0);
      (*fixed_strides)[dim - i - 1] = total_size;
      total_size *= sizes[dim - i - 1];
    }
  }
  size_t storage_size = 1;
  if (dim >= 1) storage_size = sizes[0] * (*fixed_strides)[0];
  return storage_size;
}

}// namespace

void WholeMemoryInitLib() {
  whole_graph::WholeMemoryInit();
}

std::vector<int64_t> WholeMemoryGetUniqueID() {
  std::vector<int64_t> unique_id(AlignUp(sizeof(whole_graph::WmmpUniqueId), sizeof(int64_t)));
  whole_graph::WmmpGetUniqueId((whole_graph::WmmpUniqueId *) unique_id.data());
  return unique_id;
}
int64_t WholeMemoryCreateCommunicator(int64_t size, std::vector<int64_t> &unique_id_array, int64_t rank) {
  whole_graph::WmmpUniqueId unique_id;
  memcpy(&unique_id, unique_id_array.data(), sizeof(unique_id));
  auto *bc_ptr = whole_graph::WmmpCreateCommunicator(size, unique_id, rank);
  return (int64_t) bc_ptr;
}
void WholeMemoryDestroyCommunicator(int64_t comm) {
  auto *bc_ptr = (whole_graph::BootstrapCommunicator *) comm;
  whole_graph::WmmpDestroyCommunicator(bc_ptr);
}

void WholeMemoryFinalizeLib() {
  whole_graph::WholeMemoryFinalize();
}

torch::Tensor WholeMemoryCreateTensorScalarType(const std::vector<int64_t> &sizes,
                                                const std::vector<int64_t> &strides,
                                                torch::ScalarType type,
                                                bool is_unified,
                                                int64_t comm) {
  size_t elt_size = c10::elementSize(type);
  std::vector<int64_t> fixed_strides;
  size_t storage_elt_count = CheckSizesAndStrides(sizes, strides, &fixed_strides);
  c10::DataPtr data_ptr;
  auto *bc_ptr = (whole_graph::BootstrapCommunicator *) comm;
  if (!is_unified) {
    data_ptr = GetWholeMemoryDeviceAllocator()->allocate_with_comm(storage_elt_count * elt_size, bc_ptr);
  } else {
    data_ptr = GetWholeMemoryUnifiedAllocator()->allocate_with_comm(storage_elt_count * elt_size, bc_ptr);
  }
  void *data_ptr_void = data_ptr.get();
  c10::intrusive_ptr<c10::StorageImpl>
      storage_impl_ptr = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t{},
                                                               storage_elt_count * elt_size,
                                                               std::move(data_ptr),
                                                               nullptr,
                                                               false);
  {
    std::unique_lock<std::mutex> lock(wholememory_info_mutex);
    WholeMemoryInfo wmi;
    wmi.is_unified = is_unified;
    wmi.storage_impl = c10::weak_intrusive_ptr<c10::StorageImpl>(storage_impl_ptr);
    wholememory_info_map.emplace(data_ptr_void, wmi);
  }
  c10::Storage storage(storage_impl_ptr);
  torch::DispatchKeySet dispatch_key_set({c10::DispatchKey::CUDA});
  caffe2::TypeMeta data_type;
  data_type = type;
  torch::Tensor t =
      torch::Tensor::wrap_tensor_impl(c10::intrusive_ptr<torch::TensorImpl, torch::UndefinedTensorImpl>::make(std::move(
                                                                                                                  storage),
                                                                                                              dispatch_key_set, data_type));
  t.getIntrusivePtr()->set_sizes_and_strides(c10::IntArrayRef({sizes}), c10::IntArrayRef({fixed_strides}));
  return t;
}

torch::Tensor WholeMemoryCreateTensor(const std::vector<int64_t> &sizes,
                                      const std::vector<int64_t> &strides,
                                      py::object dtype,
                                      bool is_unified,
                                      int64_t comm) {
  torch::ScalarType type = torch::python::detail::py_object_to_dtype(std::move(dtype));
  return WholeMemoryCreateTensorScalarType(sizes, strides, type, is_unified, comm);
}

std::vector<int64_t> WholeMemoryGetRankIdxAndRankSizeOfTensor(torch::Tensor &t) {
  void *void_ptr = t.data_ptr();
  int rank_idx, rank_size;
  whole_graph::WmmpGetRankIdxAndRankSizeOfPtr(&rank_idx, &rank_size, void_ptr);
  return {rank_idx, rank_size};
}

int64_t WholeMemoryGetCommunicator(const torch::Tensor &t) {
  char *data_ptr = (char *) t.data_ptr();
  size_t elt_size = c10::elementSize(t.dtype().toScalarType());
  data_ptr -= t.storage_offset() * elt_size;
  return (int64_t) whole_graph::WmmpGetBootstrapCommunicator(data_ptr);
}

torch::Tensor WholeMemoryTensorViewFromDevice(torch::Tensor t, torch::Device dev) {
  //std::cerr << "[WholeMemoryTensorViewFromDevice] dev=" << dev << std::endl;
  void *data_ptr = t.data_ptr();
  c10::intrusive_ptr<c10::StorageImpl> storage_impl_ptr;
  {
    std::unique_lock<std::mutex> lock(wholememory_info_mutex);
    auto it = wholememory_info_map.find(data_ptr);
    WM_CHECK(it != wholememory_info_map.end());
    if (!it->second.is_unified) {
      WM_CHECK(dev.type() != c10::DeviceType::CPU);
    }
    storage_impl_ptr = it->second.storage_impl.lock();
    WM_CHECK(storage_impl_ptr.get() != nullptr);
  }
  int dev_idx = dev.has_index() ? dev.index() : -1;
  WM_CHECK(whole_graph::WmmpCanAccessFrom(storage_impl_ptr->data(), dev_idx));
  storage_impl_ptr->data_ptr().unsafe_set_device(dev);
  c10::Storage storage(storage_impl_ptr);
  caffe2::TypeMeta data_type;
  data_type = t.scalar_type();
  torch::DispatchKeySet
      dispatch_key_set({dev.type() == c10::DeviceType::CPU ? c10::DispatchKey::CPU : c10::DispatchKey::CUDA});
  torch::Tensor new_t =
      torch::Tensor::wrap_tensor_impl(c10::intrusive_ptr<torch::TensorImpl, torch::UndefinedTensorImpl>::make(std::move(
                                                                                                                  storage),
                                                                                                              dispatch_key_set, data_type));
  new_t.getIntrusivePtr()->set_sizes_and_strides(t.sizes(), t.strides());
  return new_t;
}

bool WholeMemoryIsUnifiedTensor(torch::Tensor t) {
  char *data_ptr = (char *) t.data_ptr();
  size_t offset = t.storage_offset() * t.dtype().itemsize();
  data_ptr -= offset;
  {
    std::unique_lock<std::mutex> lock(wholememory_info_mutex);
    auto it = wholememory_info_map.find(data_ptr);
    WM_CHECK(it != wholememory_info_map.end());
    return it->second.is_unified;
  }
}

py::tuple WholeMemoryAggregateSize(size_t local_size, int64_t comm) {
  auto *bootstrap_communicator = (whole_graph::BootstrapCommunicator *) comm;
  size_t total_size, offset;
  total_size = whole_graph::WmmpAggregateSize(local_size, &offset, bootstrap_communicator);
  return py::make_tuple(total_size, offset);
}

void WholeMemoryBarrier(int64_t comm) {
  auto *bootstrap_communicator = (whole_graph::BootstrapCommunicator *) comm;
  whole_graph::WmmpBarrier(bootstrap_communicator);
}

int64_t WholeMemoryGetRank(int64_t comm) {
  auto *bootstrap_communicator = (whole_graph::BootstrapCommunicator *) comm;
  return bootstrap_communicator->Rank();
}

int64_t WholeMemoryGetSize(int64_t comm) {
  auto *bootstrap_communicator = (whole_graph::BootstrapCommunicator *) comm;
  return bootstrap_communicator->Size();
}

whole_graph::pytorch::ChunkedTensor WholeMemoryCreateChunkedTensorScalarType(const std::vector<int64_t> &sizes,
                                                                             const std::vector<int64_t> &strides,
                                                                             torch::ScalarType type,
                                                                             int64_t comm) {
  auto *bc_ptr = (whole_graph::BootstrapCommunicator *) comm;
  size_t elt_size = c10::elementSize(type);
  std::vector<int64_t> fixed_strides;
  size_t storage_elt_count = CheckSizesAndStrides(sizes, strides, &fixed_strides);
  size_t min_granularity = elt_size;
  if (fixed_strides.size() > 1) min_granularity *= fixed_strides[fixed_strides.size() - 2];
  c10::intrusive_ptr<whole_graph::pytorch::ChunkedStorageImpl>
      storage_impl_ptr = c10::make_intrusive<whole_graph::pytorch::ChunkedStorageImpl>(storage_elt_count * elt_size,
                                                                                       min_granularity,
                                                                                       bc_ptr);
  whole_graph::pytorch::ChunkedStorage storage(storage_impl_ptr);
  caffe2::TypeMeta data_type;
  data_type = type;
  whole_graph::pytorch::ChunkedTensor t = whole_graph::pytorch::ChunkedTensor::wrap_tensor_impl(
      c10::intrusive_ptr<whole_graph::pytorch::ChunkedTensorImpl>::make(std::move(storage), data_type));
  t.getIntrusivePtr()->set_sizes_and_strides(c10::IntArrayRef({sizes}), c10::IntArrayRef({fixed_strides}));
  return t;
}

whole_graph::pytorch::ChunkedTensor WholeMemoryCreateChunkedTensor(const std::vector<int64_t> &sizes,
                                                                   const std::vector<int64_t> &strides,
                                                                   py::object dtype,
                                                                   int64_t comm) {
  torch::ScalarType type = torch::python::detail::py_object_to_dtype(std::move(dtype));
  return WholeMemoryCreateChunkedTensorScalarType(sizes, strides, type, comm);
}

int64_t WholeMemoryGetChunkedCommunicator(const whole_graph::pytorch::ChunkedTensor &ct) {
  auto wcmt = ct.GetChunkedMemory();
  return (int64_t) whole_graph::WcmmpGetBootstrapCommunicator(wcmt);
}

whole_graph::pytorch::ChunkedTensor WholeMemoryGetSubChunkedTensor(const whole_graph::pytorch::ChunkedTensor &ct,
                                                                   const std::vector<int64_t> &start,
                                                                   const std::vector<int64_t> &count) {
  int64_t dim = ct.dim();
  TORCH_CHECK(start.size() <= (size_t) dim,
              "[WholeMemoryGetSubChunkTensor] start should <= ct.dim(): ",
              start.size(),
              " v.s. ",
              dim);
  TORCH_CHECK(count.size() <= (size_t) dim,
              "[WholeMemoryGetSubChunkTensor] count should <= ct.dim(): ",
              start.size(),
              " v.s. ",
              dim);
  TORCH_CHECK(start.size() >= count.size(), "[WholeMemoryGetSubChunkTensor] count.size() should <= start.size().");
  std::vector<int64_t> new_size(dim), new_stride(dim);
  int64_t offset = ct.storage_offset();
  for (size_t i = 0; i < (size_t) dim; i++) {
    int64_t size_i = ct.size(i);
    int64_t start_i = 0;
    if (i < start.size()) {
      TORCH_CHECK(start[i] >= 0, "[WholeMemoryGetSubChunkTensor] start should >= 0");
      start_i = start[i];
      size_i = ct.size(i) - start[i];
      if (i < count.size()) {
        TORCH_CHECK(count[i] >= -1, "[WholeMemoryGetSubChunkTensor] count should >= -1");
        TORCH_CHECK(count[i] <= size_i, "[WholeMemoryGetSubChunkTensor] count[", i, "]=", count[i],
                    " should <= size_i=", size_i, ", ct.size(i)=", ct.size(i), ", start[i]=", start[i]);
        if (count[i] >= 0) {
          size_i = count[i];
        }
      }
    }
    new_size[i] = size_i;
    new_stride[i] = ct.stride(i);
    offset += start_i * new_stride[i];
  }
  whole_graph::pytorch::ChunkedStorage storage = ct.storage();
  whole_graph::pytorch::ChunkedTensor new_t = whole_graph::pytorch::ChunkedTensor::wrap_tensor_impl(
      c10::intrusive_ptr<whole_graph::pytorch::ChunkedTensorImpl>::make(std::move(storage), ct.dtype()));
  new_t.getIntrusivePtr()->set_sizes_and_strides(c10::IntArrayRef({new_size}), c10::IntArrayRef({new_stride}));
  new_t.getIntrusivePtr()->set_storage_offset(offset);
  return new_t;
}

void WholeChunkedMemoryDummyDelete(void *ptr) {
}

class WholeChunkedMemoryLocalDeviceAllocator : public c10::Allocator {
 public:
  WholeChunkedMemoryLocalDeviceAllocator() = default;
  static c10::DataPtr allocate_no_delete(void *ptr) {
    c10::Device device = GetCurrentCUDADevice();
    //(void* data, void* ctx, DeleterFnPtr ctx_deleter, Device device)
    return {ptr, ptr, &WholeChunkedMemoryDummyDelete, device};
  }
  c10::DataPtr allocate(size_t nbytes) const override {
    fprintf(stderr, "[WholeChunkedMemoryLocalDeviceAllocator] allocate should not be called.\n");
    abort();
    c10::Device device = GetCurrentCUDADevice();
    return {nullptr, nullptr, nullptr, device};
  }
};

WholeChunkedMemoryLocalDeviceAllocator *GetWholeChunkedMemoryLocalDeviceAllocator() {
  static WholeChunkedMemoryLocalDeviceAllocator whole_chunked_memory_local_device_allocator;
  return &whole_chunked_memory_local_device_allocator;
}

torch::Tensor WholeMemoryChunkedGetLocalTensor(const whole_graph::pytorch::ChunkedTensor &ct, int split_dim) {
  int64_t dim = ct.dim();
  auto cm = ct.GetChunkedMemory();
  TORCH_CHECK(ct.storage_offset() == 0, "Only storage offset=0 is supported.");
  TORCH_CHECK(ct.stride(dim - 1) == 1, "Chunked Tensor last stride should be 1.");
  TORCH_CHECK(split_dim >= 0 && split_dim < dim, "split_dim=", split_dim, " but dim=", dim);
  size_t split_stride = ct.stride(split_dim);
  void *ptr;
  size_t size;
  whole_graph::WcmmpGetLocalMemory(cm, &ptr, &size);
  size_t elt_size = whole_graph::GetWMTSize(whole_graph::pytorch::C10ScalarToWMType(ct.dtype().toScalarType()));
  TORCH_CHECK(size % (elt_size * split_stride) == 0,
              "elt_size=",
              elt_size,
              ", split_dim=",
              split_dim,
              ", but size=",
              size);
  size_t pre_dim = size / (elt_size * split_stride);
  std::vector<int64_t> local_strides;
  std::vector<int64_t> local_sizes;
  local_strides.push_back(split_stride);
  local_sizes.push_back(pre_dim);
  for (int64_t i = split_dim + 1; i < dim; i++) {
    local_strides.push_back(ct.stride(i));
    local_sizes.push_back(ct.size(i));
  }
  c10::DataPtr data_ptr = GetWholeChunkedMemoryLocalDeviceAllocator()->allocate_no_delete(ptr);
  c10::intrusive_ptr<c10::StorageImpl>
      storage_impl_ptr = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t{},
                                                               size,
                                                               std::move(data_ptr),
                                                               nullptr,
                                                               false);
  c10::Storage storage(storage_impl_ptr);
  torch::DispatchKeySet dispatch_key_set({c10::DispatchKey::CUDA});
  caffe2::TypeMeta data_type;
  data_type = ct.dtype().toScalarType();
  torch::Tensor t =
      torch::Tensor::wrap_tensor_impl(c10::intrusive_ptr<torch::TensorImpl, torch::UndefinedTensorImpl>::make(std::move(
                                                                                                                  storage),
                                                                                                              dispatch_key_set, data_type));
  t.getIntrusivePtr()->set_sizes_and_strides(c10::IntArrayRef({local_sizes}), c10::IntArrayRef({local_strides}));
  return t;
}

whole_graph::pytorch::NCCLTensor WholeMemoryCreateNCCLTensorScalarType(const std::vector<int64_t> &sizes,
                                                                       const std::vector<int64_t> &strides,
                                                                       torch::ScalarType type,
                                                                       int64_t comm) {
  auto *bc_ptr = (whole_graph::BootstrapCommunicator *) comm;
  size_t elt_size = c10::elementSize(type);
  std::vector<int64_t> fixed_strides;
  size_t storage_elt_count = CheckSizesAndStrides(sizes, strides, &fixed_strides);
  size_t min_granularity = elt_size;
  if (fixed_strides.size() > 1) min_granularity *= fixed_strides[fixed_strides.size() - 2];
  c10::intrusive_ptr<whole_graph::pytorch::NCCLStorageImpl>
      storage_impl_ptr = c10::make_intrusive<whole_graph::pytorch::NCCLStorageImpl>(storage_elt_count * elt_size,
                                                                                    min_granularity,
                                                                                    bc_ptr);
  whole_graph::pytorch::NCCLStorage storage(storage_impl_ptr);
  caffe2::TypeMeta data_type;
  data_type = type;
  whole_graph::pytorch::NCCLTensor t = whole_graph::pytorch::NCCLTensor::wrap_tensor_impl(
      c10::intrusive_ptr<whole_graph::pytorch::NCCLTensorImpl>::make(std::move(storage), data_type));
  t.getIntrusivePtr()->set_sizes_and_strides(c10::IntArrayRef({sizes}), c10::IntArrayRef({fixed_strides}));
  return t;
}

whole_graph::pytorch::NCCLTensor WholeMemoryCreateNCCLTensor(const std::vector<int64_t> &sizes,
                                                             const std::vector<int64_t> &strides,
                                                             py::object dtype,
                                                             int64_t comm) {
  torch::ScalarType type = torch::python::detail::py_object_to_dtype(std::move(dtype));
  return WholeMemoryCreateNCCLTensorScalarType(sizes, strides, type, comm);
}

int64_t WholeMemoryGetNCCLCommunicator(const whole_graph::pytorch::NCCLTensor &ct) {
  auto wnmt = ct.GetNCCLMemory();
  return (int64_t) whole_graph::WnmmpGetBootstrapCommunicator(wnmt);
}

whole_graph::pytorch::NCCLTensor WholeMemoryGetSubNCCLTensor(const whole_graph::pytorch::NCCLTensor &ct,
                                                             const std::vector<int64_t> &start,
                                                             const std::vector<int64_t> &count) {
  int64_t dim = ct.dim();
  TORCH_CHECK(start.size() <= (size_t) dim,
              "[WholeMemoryGetSubChunkTensor] start should <= ct.dim(): ",
              start.size(),
              " v.s. ",
              dim);
  TORCH_CHECK(count.size() <= (size_t) dim,
              "[WholeMemoryGetSubChunkTensor] count should <= ct.dim(): ",
              start.size(),
              " v.s. ",
              dim);
  TORCH_CHECK(start.size() >= count.size(), "[WholeMemoryGetSubChunkTensor] count.size() should <= start.size().");
  std::vector<int64_t> new_size(dim), new_stride(dim);
  int64_t offset = ct.storage_offset();
  for (size_t i = 0; i < (size_t) dim; i++) {
    int64_t size_i = ct.size(i);
    int64_t start_i = 0;
    if (i < start.size()) {
      TORCH_CHECK(start[i] >= 0, "[WholeMemoryGetSubChunkTensor] start should >= 0");
      start_i = start[i];
      size_i = ct.size(i) - start[i];
      if (i < count.size()) {
        TORCH_CHECK(count[i] >= -1, "[WholeMemoryGetSubChunkTensor] count should >= -1");
        TORCH_CHECK(count[i] <= size_i, "[WholeMemoryGetSubChunkTensor] count[", i, "]=", count[i],
                    " should <= size_i=", size_i, ", ct.size(i)=", ct.size(i), ", start[i]=", start[i]);
        if (count[i] >= 0) {
          size_i = count[i];
        }
      }
    }
    new_size[i] = size_i;
    new_stride[i] = ct.stride(i);
    offset += start_i * new_stride[i];
  }
  whole_graph::pytorch::NCCLStorage storage = ct.storage();
  whole_graph::pytorch::NCCLTensor new_t = whole_graph::pytorch::NCCLTensor::wrap_tensor_impl(
      c10::intrusive_ptr<whole_graph::pytorch::NCCLTensorImpl>::make(std::move(storage), ct.dtype()));
  new_t.getIntrusivePtr()->set_sizes_and_strides(c10::IntArrayRef({new_size}), c10::IntArrayRef({new_stride}));
  new_t.getIntrusivePtr()->set_storage_offset(offset);
  return new_t;
}

void WholeNCCLMemoryDummyDelete(void *ptr) {
}

class WholeNCCLMemoryLocalDeviceAllocator : public c10::Allocator {
 public:
  WholeNCCLMemoryLocalDeviceAllocator() = default;
  static c10::DataPtr allocate_no_delete(void *ptr) {
    c10::Device device = GetCurrentCUDADevice();
    //(void* data, void* ctx, DeleterFnPtr ctx_deleter, Device device)
    return {ptr, ptr, &WholeNCCLMemoryDummyDelete, device};
  }
  c10::DataPtr allocate(size_t nbytes) const override {
    fprintf(stderr, "[WholeNCCLMemoryLocalDeviceAllocator] allocate should not be called.\n");
    abort();
    c10::Device device = GetCurrentCUDADevice();
    return {nullptr, nullptr, nullptr, device};
  }
};

WholeNCCLMemoryLocalDeviceAllocator *GetWholeNCCLMemoryLocalDeviceAllocator() {
  static WholeNCCLMemoryLocalDeviceAllocator whole_chunked_memory_local_device_allocator;
  return &whole_chunked_memory_local_device_allocator;
}

torch::Tensor WholeMemoryNCCLGetLocalTensor(const whole_graph::pytorch::NCCLTensor &ct, int split_dim) {
  int64_t dim = ct.dim();
  auto cm = ct.GetNCCLMemory();
  TORCH_CHECK(ct.storage_offset() == 0, "Only storage offset=0 is supported.");
  TORCH_CHECK(ct.stride(dim - 1) == 1, "NCCL Tensor last stride should be 1.");
  TORCH_CHECK(split_dim >= 0 && split_dim < dim, "split_dim=", split_dim, " but dim=", dim);
  size_t split_stride = ct.stride(split_dim);
  void *ptr;
  size_t size;
  whole_graph::WnmmpGetLocalMemory(cm, &ptr, &size);
  size_t elt_size = whole_graph::GetWMTSize(whole_graph::pytorch::C10ScalarToWMType(ct.dtype().toScalarType()));
  TORCH_CHECK(size % (elt_size * split_stride) == 0,
              "elt_size=",
              elt_size,
              ", split_dim=",
              split_dim,
              ", but size=",
              size);
  size_t pre_dim = size / (elt_size * split_stride);
  std::vector<int64_t> local_strides;
  std::vector<int64_t> local_sizes;
  local_strides.push_back(split_stride);
  local_sizes.push_back(pre_dim);
  for (int64_t i = split_dim + 1; i < dim; i++) {
    local_strides.push_back(ct.stride(i));
    local_sizes.push_back(ct.size(i));
  }
  c10::DataPtr data_ptr = GetWholeNCCLMemoryLocalDeviceAllocator()->allocate_no_delete(ptr);
  c10::intrusive_ptr<c10::StorageImpl>
      storage_impl_ptr = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t{},
                                                               size,
                                                               std::move(data_ptr),
                                                               nullptr,
                                                               false);
  c10::Storage storage(storage_impl_ptr);
  torch::DispatchKeySet dispatch_key_set({c10::DispatchKey::CUDA});
  caffe2::TypeMeta data_type;
  data_type = ct.dtype().toScalarType();
  torch::Tensor t =
      torch::Tensor::wrap_tensor_impl(c10::intrusive_ptr<torch::TensorImpl, torch::UndefinedTensorImpl>::make(std::move(
                                                                                                                  storage),
                                                                                                              dispatch_key_set, data_type));
  t.getIntrusivePtr()->set_sizes_and_strides(c10::IntArrayRef({local_sizes}), c10::IntArrayRef({local_strides}));
  return t;
}

int64_t WholeMemoryStatFilelistEltCount(const std::string &file_prefix,
                                        py::object id_dtype) {
  torch::ScalarType id_scalar_type = torch::python::detail::py_object_to_dtype(std::move(id_dtype));
  c10::ScalarType id_type;
  id_type = id_scalar_type;
  return whole_graph::StatFilelistEltCount(file_prefix,
                                           whole_graph::pytorch::C10ScalarToWMType(id_type));
}

torch::Tensor WholeMemoryCreateJumpCOORow(const torch::Tensor &wm_csr_row_ptr,
                                          const torch::Tensor &wm_csr_col_idx,
                                          bool is_unified) {
  TORCH_CHECK(wm_csr_row_ptr.dim() == 1, "wm_csr_row_ptr should be 1D tensor.");
  TORCH_CHECK(wm_csr_row_ptr.dtype() == torch::kInt64,
              "wm_csr_row_ptr should be int64 tensor.");
  TORCH_CHECK(wm_csr_col_idx.dim() == 1, "wm_csr_col_idx should be 1D tensor.");
  TORCH_CHECK(wm_csr_col_idx.dtype() == torch::kInt32 || wm_csr_col_idx.dtype() == torch::kInt64,
              "wm_csr_col_idx should be int32 or int64 tensor.");
  auto *bc_ptr =
      whole_graph::WmmpGetBootstrapCommunicator((char *) wm_csr_row_ptr.data_ptr() - wm_csr_row_ptr.storage_offset());
  int64_t total_node_count = wm_csr_row_ptr.size(0) - 1;
  int64_t total_edge_count = wm_csr_col_idx.size(0);
  int64_t wm_jump_coo_size = whole_graph::WmmpGetJumpCOORowSize(total_edge_count);
  auto wm_jump_coo_row = WholeMemoryCreateTensorScalarType({wm_jump_coo_size},
                                                           {},
                                                           wm_csr_col_idx.dtype().toScalarType(),
                                                           is_unified,
                                                           (int64_t) bc_ptr);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  whole_graph::WmmpGenerateJumpCOORow(wm_jump_coo_row.data_ptr(),
                                      wm_csr_row_ptr.data_ptr(),
                                      total_node_count,
                                      total_edge_count,
                                      whole_graph::pytorch::C10ScalarToWMType(wm_csr_col_idx.dtype().toScalarType()),
                                      stream);
  return wm_jump_coo_row;
}

whole_graph::pytorch::ChunkedTensor WholeMemoryCreateChunkedJumpCOORow(
    whole_graph::pytorch::ChunkedTensor &wm_csr_row_ptr,
    whole_graph::pytorch::ChunkedTensor &wm_csr_col_idx) {
  TORCH_CHECK(wm_csr_row_ptr.dim() == 1, "wm_csr_row_ptr should be 1D tensor.");
  TORCH_CHECK(wm_csr_row_ptr.dtype() == torch::kInt64,
              "wm_csr_row_ptr should be int64 tensor.");
  TORCH_CHECK(wm_csr_col_idx.dim() == 1, "wm_csr_col_idx should be 1D tensor.");
  TORCH_CHECK(wm_csr_col_idx.dtype() == torch::kInt32 || wm_csr_col_idx.dtype() == torch::kInt64,
              "wm_csr_col_idx should be int32 or int64 tensor.");
  auto *bc_ptr = whole_graph::WcmmpGetBootstrapCommunicator(wm_csr_row_ptr.GetChunkedMemory());
  int64_t total_node_count = wm_csr_row_ptr.size(0) - 1;
  int64_t total_edge_count = wm_csr_col_idx.size(0);
  int64_t wm_jump_coo_size = whole_graph::WmmpGetJumpCOORowSize(total_edge_count);
  auto wm_jump_coo_row = WholeMemoryCreateChunkedTensorScalarType({wm_jump_coo_size},
                                                                  {},
                                                                  wm_csr_col_idx.dtype().toScalarType(),
                                                                  (int64_t) bc_ptr);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  whole_graph::WmmpGenerateChunkedJumpCOORow(wm_jump_coo_row.GetChunkedMemory(),
                                             wm_csr_row_ptr.GetChunkedMemory(),
                                             total_node_count,
                                             total_edge_count,
                                             whole_graph::pytorch::C10ScalarToWMType(wm_csr_col_idx.dtype().toScalarType()),
                                             stream);
  return wm_jump_coo_row;
}

std::vector<torch::Tensor> WholeMemoryGetEdgeNodesFromEid(const torch::Tensor &wm_csr_row_ptr,
                                                          const torch::Tensor &wm_csr_col_idx,
                                                          const torch::Tensor &wm_jump_coo_row,
                                                          const torch::Tensor &edge_idx_list,
                                                          bool need_src,
                                                          bool need_dst) {
  TORCH_CHECK(wm_csr_row_ptr.dim() == 1, "wm_csr_row_ptr should be 1D tensor.");
  TORCH_CHECK(wm_csr_row_ptr.dtype() == torch::kInt64,
              "wm_csr_row_ptr should be int64 tensor.");
  TORCH_CHECK(wm_csr_col_idx.dim() == 1, "wm_csr_col_idx should be 1D tensor.");
  TORCH_CHECK(wm_csr_col_idx.dtype() == torch::kInt32 || wm_csr_col_idx.dtype() == torch::kInt64,
              "wm_csr_col_idx should be int32 or int64 tensor.");
  TORCH_CHECK(wm_jump_coo_row.dim() == 1, "wm_jump_coo_row should be 1D tensor.");
  TORCH_CHECK(wm_jump_coo_row.dtype() == wm_csr_col_idx.dtype(),
              "wm_jump_coo_row should be same dtype as wm_csr_col_idx.");
  TORCH_CHECK(edge_idx_list.dim() == 1, "edge_idx_list should be 1D tensor.");
  TORCH_CHECK(edge_idx_list.dtype() == torch::kInt64, "edge_idx_list should be int64 tensor.");
  int64_t total_src_node_count = wm_csr_row_ptr.size(0) - 1;
  int64_t total_edge_count = wm_csr_col_idx.size(0);
  int64_t edge_list_count = edge_idx_list.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  torch::TensorOptions options;

  options = options.dtype(wm_csr_col_idx.dtype())
                .device(edge_idx_list.device())
                .requires_grad(false);
  std::vector<int64_t> size{edge_list_count};
  torch::Tensor src_nids, dst_nids;
  void *src_ptr = nullptr;
  void *dst_ptr = nullptr;
  if (need_src) {
    src_nids = torch::empty(size, options);
    src_ptr = src_nids.data_ptr();
  }
  if (need_dst) {
    dst_nids = torch::empty(size, options);
    dst_ptr = dst_nids.data_ptr();
  }

  whole_graph::WmmpGetEdgeNodesFromEid(wm_csr_row_ptr.data_ptr(),
                                       wm_csr_col_idx.data_ptr(),
                                       wm_jump_coo_row.data_ptr(),
                                       edge_idx_list.data_ptr<int64_t>(),
                                       whole_graph::pytorch::C10ScalarToWMType(wm_csr_col_idx.dtype().toScalarType()),
                                       total_src_node_count,
                                       total_edge_count,
                                       src_ptr,
                                       dst_ptr,
                                       edge_list_count,
                                       stream);

  if (need_src && need_dst) {
    return {src_nids, dst_nids};
  } else if (need_src && !need_dst) {
    return {src_nids};
  } else if (!need_src && need_dst) {
    return {dst_nids};
  } else {
    return {};
  }
}

std::vector<torch::Tensor> WholeMemoryGetEdgeNodesFromEidChunked(whole_graph::pytorch::ChunkedTensor &wm_csr_row_ptr,
                                                                 whole_graph::pytorch::ChunkedTensor &wm_csr_col_idx,
                                                                 whole_graph::pytorch::ChunkedTensor &wm_jump_coo_row,
                                                                 const torch::Tensor &edge_idx_list,
                                                                 bool need_src,
                                                                 bool need_dst) {
  TORCH_CHECK(wm_csr_row_ptr.dim() == 1, "wm_csr_row_ptr should be 1D tensor.");
  TORCH_CHECK(wm_csr_row_ptr.dtype() == torch::kInt64,
              "wm_csr_row_ptr should be int64 tensor.");
  TORCH_CHECK(wm_csr_col_idx.dim() == 1, "wm_csr_col_idx should be 1D tensor.");
  TORCH_CHECK(wm_csr_col_idx.dtype() == torch::kInt32 || wm_csr_col_idx.dtype() == torch::kInt64,
              "wm_csr_col_idx should be int32 or int64 tensor.");
  TORCH_CHECK(wm_jump_coo_row.dim() == 1, "wm_jump_coo_row should be 1D tensor.");
  TORCH_CHECK(wm_jump_coo_row.dtype() == wm_csr_col_idx.dtype(),
              "wm_jump_coo_row should be same dtype as wm_csr_col_idx.");
  TORCH_CHECK(edge_idx_list.dim() == 1, "edge_idx_list should be 1D tensor.");
  TORCH_CHECK(edge_idx_list.dtype() == torch::kInt64, "edge_idx_list should be int64 tensor.");
  int64_t total_src_node_count = wm_csr_row_ptr.size(0) - 1;
  int64_t total_edge_count = wm_csr_col_idx.size(0);
  int64_t edge_list_count = edge_idx_list.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  torch::TensorOptions options;

  options = options.dtype(wm_csr_col_idx.dtype())
                .device(edge_idx_list.device())
                .requires_grad(false);
  std::vector<int64_t> size{edge_list_count};
  torch::Tensor src_nids, dst_nids;
  void *src_ptr = nullptr;
  void *dst_ptr = nullptr;
  if (need_src) {
    src_nids = torch::empty(size, options);
    src_ptr = src_nids.data_ptr();
  }
  if (need_dst) {
    dst_nids = torch::empty(size, options);
    dst_ptr = dst_nids.data_ptr();
  }

  whole_graph::WmmpGetEdgeNodesFromEidChunked(wm_csr_row_ptr.GetChunkedMemory(),
                                              wm_csr_col_idx.GetChunkedMemory(),
                                              wm_jump_coo_row.GetChunkedMemory(),
                                              edge_idx_list.data_ptr<int64_t>(),
                                              whole_graph::pytorch::C10ScalarToWMType(wm_csr_col_idx.dtype().toScalarType()),
                                              total_src_node_count,
                                              total_edge_count,
                                              src_ptr,
                                              dst_ptr,
                                              edge_list_count,
                                              stream);

  if (need_src && need_dst) {
    return {src_nids, dst_nids};
  } else if (need_src && !need_dst) {
    return {src_nids};
  } else if (!need_src && need_dst) {
    return {dst_nids};
  } else {
    return {};
  }
}

static PyObject *ScaleTypeToDtype(c10::ScalarType st) {
  std::string name;
  switch (st) {
    case c10::ScalarType::Byte: {
      name = "uint8";
      break;
    }
    case c10::ScalarType::Char: {
      name = "int8";
      break;
    }
    case c10::ScalarType::Short: {
      name = "int16";
      break;
    }
    case c10::ScalarType::Int: {
      name = "int32";
      break;
    }
    case c10::ScalarType::Long: {
      name = "int64";
      break;
    }
    case c10::ScalarType::Half: {
      name = "float16";
      break;
    }
    case c10::ScalarType::Float: {
      name = "float32";
      break;
    }
    case c10::ScalarType::Double: {
      name = "float64";
      break;
    }
    case c10::ScalarType::BFloat16: {
      name = "bfloat16";
      break;
    }
    default: {
      printf("[ScaleTypeToDtype] Type not supported.\n");
      abort();
    }
  }
  PyObject *po = THPDtype_New(st, name);
  return po;
}

whole_graph::WMType PytorchDtypeToWMType(py::object pdtype) {
  torch::ScalarType torch_scalar_type = torch::python::detail::py_object_to_dtype(std::move(pdtype));
  c10::ScalarType scalar_type;
  scalar_type = torch_scalar_type;
  whole_graph::WMType wmtype = whole_graph::pytorch::C10ScalarToWMType(scalar_type);
  return wmtype;
}

void PyTorchWholeMemoryChunked2DSubTensorAssign(const torch::Tensor &t,
                                                const whole_graph::pytorch::ChunkedTensor &ct,
                                                int64_t start_idx) {
  TORCH_CHECK(t.dtype() == ct.dtype(), "PyTorchWholeMemoryChunkedSubTensorAssign should have same type.");
  TORCH_CHECK(t.dim() == ct.dim(), "PyTorchWholeMemoryChunkedSubTensorAssign t only dim 2 supported.");
  TORCH_CHECK(t.dim() == 2, "PyTorchWholeMemoryChunkedSubTensorAssign t only dim 2 supported now.");
  TORCH_CHECK(ct.dim() == 2, "PyTorchWholeMemoryChunkedSubTensorAssign ct only dim 2 supported now.");
  TORCH_CHECK(t.stride(1) == 1, "PyTorchWholeMemoryChunkedSubTensorAssign t only continuous dim1 supported.");
  TORCH_CHECK(ct.stride(1) == 1, "PyTorchWholeMemoryChunkedSubTensorAssign t only continuous dim1 supported.");
  TORCH_CHECK(ct.size(1) == t.size(1), "PyTorchWholeMemoryChunkedSubTensorAssign size 1 not same.");
  TORCH_CHECK(t.size(0) + start_idx <= ct.size(0), "PyTorchWholeMemoryChunkedSubTensorAssign copy not in range");
  whole_graph::WMType ctdtype = whole_graph::pytorch::C10ScalarToWMType(ct.dtype().toScalarType());
  whole_graph::WMType tdtype = whole_graph::pytorch::C10ScalarToWMType(t.dtype().toScalarType());
  whole_graph::WholeChunkedMemory_t wct = ct.GetChunkedMemory();
  void *tptr = t.data_ptr();
  whole_graph::WmmpLoadChunkedEmbeddingFromMemory(wct,
                                                  ct.storage_offset(),
                                                  ct.size(0),
                                                  ct.size(1),
                                                  ct.stride(0),
                                                  tptr,
                                                  t.stride(0),
                                                  tdtype,
                                                  ctdtype,
                                                  start_idx,
                                                  t.size(0),
                                                  at::cuda::getCurrentCUDAStream());
}

int64_t PyTorchWholeMemoryGetPtr(const torch::Tensor &t) {
  return (int64_t) t.data_ptr();
}

int64_t PyTorchWholeMemoryGetStorageOffset(const torch::Tensor &t) {
  return t.storage_offset();
}

void PytorchWholeMemoryEmbeddingLocalApplyGradients(
    int64_t optimizer_type,
    float learning_rate,
    std::vector<float> &optimizer_data,
    const torch::Tensor &local_sparse_indice,
    const torch::Tensor &grad,
    const torch::Tensor &emb,
    const std::vector<torch::Tensor> &per_element_states,
    const std::vector<torch::Tensor> &per_embedding_states) {
  WM_CHECK(local_sparse_indice.dim() == 1);
  WM_CHECK(emb.dim() == 2);
  WM_CHECK(grad.dim() == 2);
  WM_CHECK(local_sparse_indice.dtype() == torch::kInt32);
  WM_CHECK(emb.dtype() == torch::kFloat32 || emb.dtype() == torch::kFloat16);
  WM_CHECK(grad.dtype() == torch::kFloat32 || grad.dtype() == torch::kFloat16);
  WM_CHECK(emb.dtype() == grad.dtype());
  int64_t embedding_table_size = emb.size(0);
  int64_t embedding_dim = emb.size(1);
  int64_t embedding_stride = emb.stride(0);
  int64_t grad_per_element_state_stride = grad.stride(0);
  WM_CHECK(grad.size(0) == local_sparse_indice.size(0));
  WM_CHECK(grad.size(1) == embedding_dim);
  WM_CHECK(per_element_states.size() <= 2);
  WM_CHECK(per_embedding_states.size() <= 1);

  auto grad_type = whole_graph::pytorch::C10ScalarToWMType(grad.dtype().toScalarType());
  auto state_type = whole_graph::WMT_Float;

  auto *local_sparse_indice_ptr = local_sparse_indice.data_ptr<int32_t>();
  void *embedding = nullptr;
  void *gradient = nullptr;
  void *per_element_state_0 = nullptr;
  void *per_element_state_1 = nullptr;
  void *per_embedding_state = nullptr;
  if (!per_element_states.empty()) {
    state_type = whole_graph::pytorch::C10ScalarToWMType(per_element_states[0].dtype().toScalarType());
    for (auto &per_element_state : per_element_states) {
      auto current_state_type = whole_graph::pytorch::C10ScalarToWMType(per_element_state.dtype().toScalarType());
      WM_CHECK(current_state_type == state_type);
      WM_CHECK(per_element_state.dim() == 2);
      WM_CHECK(per_element_state.size(0) == embedding_table_size);
      WM_CHECK(per_element_state.size(1) == embedding_dim);
      WM_CHECK(per_element_state.stride(0) == grad_per_element_state_stride);
    }
  }
  if (!per_element_states.empty()) {
    per_element_state_0 = per_element_states[0].data_ptr();
    if (per_element_states.size() > 1) {
      per_element_state_1 = per_element_states[1].data_ptr();
    }
  }
  if (!per_embedding_states.empty()) {
    per_embedding_state = per_embedding_states[0].data_ptr();
  }

  embedding = emb.data_ptr();
  gradient = grad.data_ptr();
  auto optimizer_type_enum = (whole_graph::OptimizerType) optimizer_type;
  whole_graph::OptimizerInfo optimizer_info{};
  optimizer_info.type = optimizer_type_enum;
  optimizer_info.lr = learning_rate;
  WM_CHECK(optimizer_data.size() * sizeof(float) <= sizeof(optimizer_info.private_info));
  memcpy(&optimizer_info.private_info, optimizer_data.data(), optimizer_data.size() * sizeof(float));

  WholeMemoryEmbeddingLocalApplyGradients(grad_type,
                                          state_type,
                                          optimizer_info,
                                          local_sparse_indice_ptr,
                                          embedding,
                                          gradient,
                                          per_element_state_0,
                                          per_element_state_1,
                                          per_embedding_state,
                                          local_sparse_indice.size(0),
                                          embedding_table_size,
                                          embedding_dim,
                                          embedding_stride,
                                          grad_per_element_state_stride,
                                          at::cuda::getCurrentCUDAStream());
}

void WholeMemoryStoreLocalEmbeddingTensorToFile(const torch::Tensor &t,
                                                const std::string &filename) {
  WM_CHECK(t.dim() == 2);
  int64_t embedding_count = t.size(0);
  int64_t embedding_dim = t.size(1);
  int64_t embedding_stride = t.stride(0);
  auto emb_type = whole_graph::pytorch::C10ScalarToWMType(t.dtype().toScalarType());
  void *src_ptr = t.data_ptr();
  whole_graph::WmmpStoreLocalEmbeddingToFile(emb_type,
                                             src_ptr,
                                             embedding_count,
                                             embedding_dim,
                                             embedding_stride,
                                             filename);
}

void WholeMemoryLoadLocalEmbeddingTensorFromFile(torch::Tensor t,
                                                 const std::string &file_prefix,
                                                 int64_t part_count,
                                                 int64_t comm) {
  WM_CHECK(t.dim() == 2);
  int64_t embedding_count = t.size(0);
  int64_t embedding_dim = t.size(1);
  int64_t embedding_stride = t.stride(0);
  auto emb_type = whole_graph::pytorch::C10ScalarToWMType(t.dtype().toScalarType());
  void *src_ptr = t.data_ptr();
  auto *bc_ptr = (whole_graph::BootstrapCommunicator *) comm;
  whole_graph::WmmpLoadLocalEmbeddingFromFile(emb_type,
                                              src_ptr,
                                              embedding_count,
                                              embedding_dim,
                                              embedding_stride,
                                              file_prefix,
                                              part_count,
                                              bc_ptr);
}

int64_t PythonCreateMixedGraphBuilder(const std::vector<std::string> &node_type_names,
                                      const std::vector<std::vector<std::string>> &relations,
                                      py::object dtype) {
  torch::ScalarType type = torch::python::detail::py_object_to_dtype(std::move(dtype));
  whole_graph::WMType id_type = whole_graph::pytorch::C10ScalarToWMType(type);
  auto ptr = (int64_t) whole_graph::CreateMixedGraphBuilder(node_type_names, relations, id_type);
  return ptr;
}

int64_t PythonCreateHomoGraphBuilder(py::object dtype) {
  torch::ScalarType type = torch::python::detail::py_object_to_dtype(std::move(dtype));
  whole_graph::WMType id_type = whole_graph::pytorch::C10ScalarToWMType(type);
  auto ptr = (int64_t) whole_graph::CreateHomoGraphBuilder(id_type);
  return ptr;
}

void PythonDestroyGraphBuilder(int64_t graph_builder) {
  auto *ptr = (whole_graph::GraphBuilder *) graph_builder;
  whole_graph::DestroyGraphBuilder(ptr);
}

void PythonGraphBuilderSetNodeCounts(int64_t graph_builder, const std::vector<int64_t> &node_counts) {
  auto *ptr = (whole_graph::GraphBuilder *) graph_builder;
  whole_graph::GraphBuilderSetNodeCounts(ptr, node_counts);
}

void PythonGraphBuilderLoadEdgeDataFromFileList(int64_t graph_builder,
                                                const std::vector<std::string> &relations,
                                                const std::string &file_prefix,
                                                bool reverse,
                                                py::object dtype,
                                                int64_t edge_feature_size) {
  torch::ScalarType type = torch::python::detail::py_object_to_dtype(std::move(dtype));
  whole_graph::WMType file_id_type = whole_graph::pytorch::C10ScalarToWMType(type);
  auto *ptr = (whole_graph::GraphBuilder *) graph_builder;
  whole_graph::GraphBuilderLoadEdgeDataFromFileList(ptr,
                                                    relations,
                                                    file_prefix,
                                                    reverse,
                                                    file_id_type,
                                                    edge_feature_size);
}

void PythonGraphBuilderSetEdgeConfig(int64_t graph_builder,
                                     const std::vector<std::string> &relation,
                                     bool as_undirected,
                                     bool add_self_loop,
                                     bool build_both_direction) {
  auto *ptr = (whole_graph::GraphBuilder *) graph_builder;
  whole_graph::GraphBuilderSetEdgeConfig(ptr, relation, as_undirected, add_self_loop, build_both_direction);
}

void PythonGraphBuilderSetShuffleID(int64_t graph_builder,
                                    bool shuffle_id) {
  auto *ptr = (whole_graph::GraphBuilder *) graph_builder;
  whole_graph::GraphBuilderSetShuffleID(ptr, shuffle_id);
}

void PythonGraphBuilderSetGraphSaveFile(int64_t graph_builder,
                                        const std::string &csr_row_ptr_filename,
                                        const std::string &csr_col_idx_filename,
                                        const std::string &id_mapping_prefix) {
  auto *ptr = (whole_graph::GraphBuilder *) graph_builder;
  whole_graph::GraphBuilderSetGraphSaveFile(ptr, csr_row_ptr_filename, csr_col_idx_filename, id_mapping_prefix);
}

void PythonGraphBuilderBuildGraph(int64_t graph_builder) {
  auto *ptr = (whole_graph::GraphBuilder *) graph_builder;
  whole_graph::GraphBuilderBuild(ptr);
}

void PyTorchMixedGraphSGC(const torch::Tensor &param,
                          const torch::Tensor &csr_row_ptr,
                          const torch::Tensor &csr_col_idx,
                          const torch::Tensor &to_typed,
                          int64_t target_type,
                          int64_t neighbor_type) {
  TORCH_CHECK(param.dtype() == torch::ScalarType::BFloat16 || param.dtype() == torch::ScalarType::Half
                  || param.dtype() == torch::ScalarType::Float || param.dtype() == torch::ScalarType::Double,
              "param should be float, double, half or bfloat16 tensor.");
  int64_t node_count = param.size(0);
  int64_t embedding_dim = param.size(1);
  int64_t embedding_stride = param.stride(0);
  TORCH_CHECK(param.dim() == 2, "embedding shoulde be 2-D Tensor");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "csr_row_ptr should be 1-D Tensor");
  TORCH_CHECK(csr_row_ptr.dtype() == torch::ScalarType::Long, "csr_row_ptr shoulde be 1-D Tensor");
  TORCH_CHECK(csr_row_ptr.size(0) == node_count + 1, "csr_row_ptr should be (node_count + 1, ) Tensor");
  TORCH_CHECK(csr_col_idx.dim() == 1, "csr_col_idx should be 1-D Tensor");
  TORCH_CHECK(csr_col_idx.dtype() == torch::ScalarType::Long || csr_col_idx.dtype() == torch::ScalarType::Int,
              "csr_col_idx csr_col_idx be 1-D Tensor");
  TORCH_CHECK(to_typed.dtype() == torch::ScalarType::Long, "to_typed should be Long Tensor");
  TORCH_CHECK(to_typed.dim() == 1, "to_typed should be 1-D Tensor");
  TORCH_CHECK(to_typed.size(0) == node_count, "to_typed should be (node_count, ) Tensor");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  whole_graph::MixedGraphSGC(whole_graph::pytorch::C10ScalarToWMType(param.dtype().toScalarType()),
                             whole_graph::pytorch::C10ScalarToWMType(csr_col_idx.dtype().toScalarType()),
                             param.data_ptr(),
                             csr_row_ptr.data_ptr<int64_t>(),
                             csr_col_idx.data_ptr(),
                             to_typed.data_ptr<int64_t>(),
                             target_type,
                             neighbor_type,
                             0,
                             embedding_dim,
                             embedding_stride,
                             node_count,
                             stream);
}

void PyTorchMixedGraphSGCChunked(const whole_graph::pytorch::ChunkedTensor &param,
                                 const whole_graph::pytorch::ChunkedTensor &csr_row_ptr,
                                 const whole_graph::pytorch::ChunkedTensor &csr_col_idx,
                                 const whole_graph::pytorch::ChunkedTensor &to_typed,
                                 int64_t target_type,
                                 int64_t neighbor_type) {
  TORCH_CHECK(param.dtype() == torch::ScalarType::BFloat16 || param.dtype() == torch::ScalarType::Half
                  || param.dtype() == torch::ScalarType::Float || param.dtype() == torch::ScalarType::Double,
              "param should be float, double, half or bfloat16 tensor.");
  int64_t node_count = param.size(0);
  int64_t embedding_dim = param.size(1);
  int64_t embedding_stride = param.stride(0);
  TORCH_CHECK(param.dim() == 2, "embedding shoulde be 2-D Tensor");
  TORCH_CHECK(csr_row_ptr.dim() == 1, "csr_row_ptr should be 1-D Tensor");
  TORCH_CHECK(csr_row_ptr.dtype() == torch::ScalarType::Long, "csr_row_ptr shoulde be 1-D Tensor");
  TORCH_CHECK(csr_row_ptr.size(0) == node_count + 1, "csr_row_ptr should be (node_count + 1, ) Tensor");
  TORCH_CHECK(csr_col_idx.dim() == 1, "csr_col_idx should be 1-D Tensor");
  TORCH_CHECK(csr_col_idx.dtype() == torch::ScalarType::Long || csr_col_idx.dtype() == torch::ScalarType::Int,
              "csr_col_idx csr_col_idx be 1-D Tensor");
  TORCH_CHECK(to_typed.dtype() == torch::ScalarType::Long, "to_typed should be Long Tensor");
  TORCH_CHECK(to_typed.dim() == 1, "to_typed should be 1-D Tensor");
  TORCH_CHECK(to_typed.size(0) == node_count, "to_typed should be (node_count, ) Tensor");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  whole_graph::MixedGraphSGCChunked(whole_graph::pytorch::C10ScalarToWMType(param.dtype().toScalarType()),
                                    whole_graph::pytorch::C10ScalarToWMType(csr_col_idx.dtype().toScalarType()),
                                    param.GetChunkedMemory(),
                                    csr_row_ptr.GetChunkedMemory(),
                                    csr_col_idx.GetChunkedMemory(),
                                    to_typed.GetChunkedMemory(),
                                    target_type,
                                    neighbor_type,
                                    param.storage_offset(),
                                    embedding_dim,
                                    embedding_stride,
                                    node_count,
                                    stream);
}

PYBIND11_MODULE(wholegraph_pytorch, m) {
  py::class_<whole_graph::pytorch::ChunkedTensor>(m, "ChunkedTensor")
      .def(py::init<const whole_graph::pytorch::ChunkedTensor &>())
      .def("get_ptr", &whole_graph::pytorch::ChunkedTensor::get_ptr)
      .def("dim", &whole_graph::pytorch::ChunkedTensor::dim)
      .def_property_readonly("shape", &whole_graph::pytorch::ChunkedTensor::sizes)
      .def("stride", &whole_graph::pytorch::ChunkedTensor::strides)
      .def_property_readonly("dtype", [](const whole_graph::pytorch::ChunkedTensor &ct) {
        PyObject *po = ScaleTypeToDtype(ct.dtype().toScalarType());
        py::handle pyh(po);
        return py::reinterpret_steal<py::object>(pyh);
        //return ct.dtype().toScalarType();
      });
  py::class_<whole_graph::pytorch::NCCLTensor>(m, "NCCLTensor")
      .def(py::init<const whole_graph::pytorch::NCCLTensor &>())
      .def("get_ptr", &whole_graph::pytorch::NCCLTensor::get_ptr)
      .def("dim", &whole_graph::pytorch::NCCLTensor::dim)
      .def_property_readonly("shape", &whole_graph::pytorch::NCCLTensor::sizes)
      .def("stride", &whole_graph::pytorch::NCCLTensor::strides)
      .def_property_readonly("dtype", [](const whole_graph::pytorch::NCCLTensor &nt) {
        PyObject *po = ScaleTypeToDtype(nt.dtype().toScalarType());
        py::handle pyh(po);
        return py::reinterpret_steal<py::object>(pyh);
      });

  m.doc() = "PyTorch API for WholeMemory.";
  m.def("init_lib", &WholeMemoryInitLib, "Init WholeMemory Lib");
  m.def("finalize_lib", &WholeMemoryFinalizeLib, "Finalize WholeMemory Lib");

  m.def("get_unique_id", &WholeMemoryGetUniqueID, "get unique_id for communicator");
  m.def("create_communicator", &WholeMemoryCreateCommunicator, "create communicator");
  m.def("destroy_communicator", &WholeMemoryDestroyCommunicator, "destroy communicator");
  m.def("create_tensor", &WholeMemoryCreateTensor, "Create WholeMemory Tensor.");
  m.def("get_rank_idx_and_rank_size_of_tensor",
        &WholeMemoryGetRankIdxAndRankSizeOfTensor,
        "Get rank index and size of WholeMemory Tensor");
  m.def("get_tensor_communicator", &WholeMemoryGetCommunicator, "get communicator");
  m.def("get_tensor_view", &WholeMemoryTensorViewFromDevice, "Get view from device for WholeMemory Tensor");
  m.def("is_unified_tensor", &WholeMemoryIsUnifiedTensor, "Is unified tensor");
  m.def("aggregate_size", &WholeMemoryAggregateSize, "Aggregate sizes from ranks");
  m.def("get_rank", WholeMemoryGetRank, "get rank");
  m.def("get_size", WholeMemoryGetSize, "get size");
  m.def("barrier", &WholeMemoryBarrier, "Barrier for ranks");

  m.def("create_chunked_tensor", &WholeMemoryCreateChunkedTensor, "Create WholeMemory Chunked Tensor.");
  m.def("get_chunked_tensor_communicator", &WholeMemoryGetChunkedCommunicator, "get chunked communicator");
  m.def("get_sub_chunked_tensor", &WholeMemoryGetSubChunkedTensor, "Get WholeMemory subtensor from Chunked Tensor.");
  m.def("get_local_tensor_from_chunked_tensor",
        &WholeMemoryChunkedGetLocalTensor,
        "Get local Tensor from Chunked Tensor.");

  m.def("create_nccl_tensor", &WholeMemoryCreateNCCLTensor, "Create WholeMemory NCCL Tensor.");
  m.def("get_nccl_tensor_communicator", &WholeMemoryGetNCCLCommunicator, "get nccl communicator");
  m.def("get_sub_nccl_tensor", &WholeMemoryGetSubNCCLTensor, "Get WholeMemory subtensor from NCCL Tensor.");
  m.def("get_local_tensor_from_nccl_tensor",
        &WholeMemoryNCCLGetLocalTensor,
        "Get local Tensor from nccl Tensor.");

  m.def("stat_filelist_element_count", &WholeMemoryStatFilelistEltCount, "stat filelist element count");
  m.def("create_jump_coo_row", &WholeMemoryCreateJumpCOORow, "create jump coo row");
  m.def("create_chunked_jump_coo_row", &WholeMemoryCreateChunkedJumpCOORow, "create jump coo row");
  m.def("get_edge_src_dst_from_eid", &WholeMemoryGetEdgeNodesFromEid, "get edge src or dst node from eid");
  m.def("get_edge_src_dst_from_eid_chunked",
        &WholeMemoryGetEdgeNodesFromEidChunked,
        "get edge src or dst node from eid");

  m.def("tptr", &PyTorchWholeMemoryGetPtr, "");
  m.def("toffset", &PyTorchWholeMemoryGetStorageOffset, "");
  m.def("chunked_embedding_2d_sub_tensor_assign", &PyTorchWholeMemoryChunked2DSubTensorAssign, "2d sub tensor assign");

  m.def("embedding_apply_gradients_collective", &PytorchWholeMemoryEmbeddingLocalApplyGradients, "apply gradients locally");

  m.def("store_local_tensor_to_embedding_file",
        &WholeMemoryStoreLocalEmbeddingTensorToFile,
        "store local tensor of WholeChunkedTensor or Tensor to file.");
  m.def("load_local_tensor_from_embedding_file",
        &WholeMemoryLoadLocalEmbeddingTensorFromFile,
        "load local tensor of WholeChunkedTensor or Tensor from file.");

  m.def("create_mixed_graph_builder", &PythonCreateMixedGraphBuilder, "create Mixed GraphBuilder.");
  m.def("create_homograph_builder", &PythonCreateHomoGraphBuilder, "create Homo GraphBuilder.");
  m.def("destroy_graph_builder", &PythonDestroyGraphBuilder, "destroy Mixed GraphBuilder.");
  m.def("graph_builder_set_node_counts", &PythonGraphBuilderSetNodeCounts, "set node count.");
  m.def("graph_builder_load_edge_data", &PythonGraphBuilderLoadEdgeDataFromFileList, "set node count.");
  m.def("graph_builder_set_edge_config", &PythonGraphBuilderSetEdgeConfig, "set edge config.");
  m.def("graph_builder_set_shuffle_id", &PythonGraphBuilderSetShuffleID, "set whether to shuffle id.");
  m.def("graph_builder_set_graph_save_file", &PythonGraphBuilderSetGraphSaveFile, "set graph save file.");
  m.def("graph_builder_build", &PythonGraphBuilderBuildGraph, "build");

  m.def("mixed_graph_sgc", &PyTorchMixedGraphSGC, "SGC for mixed graph");
  m.def("mixed_graph_sgc_chunked", &PyTorchMixedGraphSGCChunked, "chunked SGC for mixed graph");
}
