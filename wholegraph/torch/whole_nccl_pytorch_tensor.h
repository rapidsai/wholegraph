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
#pragma once

#include <pybind11/pybind11.h>
#include <torch/script.h>

#include "data_type.h"
#include "pytorch_dtype.h"
#include "whole_nccl_memory.h"

namespace whole_graph {

namespace pytorch {

struct C10_API NCCLStorageImpl final : public c10::intrusive_ptr_target {
 public:
  explicit NCCLStorageImpl(size_t size_bytes, size_t min_granularity, whole_graph::BootstrapCommunicator *bc_ptr);
  ~NCCLStorageImpl() override;
  size_t nbytes() const {
    return size_bytes_;
  }
  whole_graph::WholeNCCLMemory_t GetNCCLMemory() const {
    return wnmt_;
  }

 private:
  size_t size_bytes_ = 0;
  whole_graph::WholeNCCLMemory_t wnmt_ = nullptr;
};

struct C10_API NCCLStorage {
 public:
  NCCLStorage() = default;
  explicit NCCLStorage(c10::intrusive_ptr<NCCLStorageImpl> ptr)
      : storage_impl_(std::move(ptr)) {}
  NCCLStorageImpl *unsafeGetNCCLStorageImpl() const noexcept {
    return storage_impl_.get();
  }
  whole_graph::WholeNCCLMemory_t GetNCCLMemory() const {
    return storage_impl_->GetNCCLMemory();
  }

 protected:
  c10::intrusive_ptr<NCCLStorageImpl> storage_impl_;
};

struct NCCLTensorImpl : public c10::intrusive_ptr_target {
 public:
  NCCLTensorImpl(NCCLStorage &&storage, const caffe2::TypeMeta data_type)
      : storage_(storage), data_type_(data_type) {
  }
  NCCLTensorImpl(const NCCLTensorImpl &) = delete;
  NCCLTensorImpl &operator=(const NCCLTensorImpl &) = delete;
  NCCLTensorImpl(NCCLTensorImpl &&) = delete;
  NCCLTensorImpl &operator=(NCCLTensorImpl &&) = delete;
  void set_sizes_and_strides(torch::IntArrayRef new_size, torch::IntArrayRef new_stride);
  int64_t dim() const {
    return sizes_and_strides_.size();
  }
  torch::IntArrayRef sizes() const {
    return sizes_and_strides_.sizes_arrayref();
  }
  torch::IntArrayRef strides() const {
    return sizes_and_strides_.strides_arrayref();
  }
  int64_t size(int64_t d) const {
    return sizes()[d];
  }
  int64_t stride(int64_t d) const {
    return strides()[d];
  }
  int64_t numel() const {
    return numel_;
  }
  void set_storage_offset(int64_t storage_offset) {
    storage_offset_ = storage_offset;
  }
  caffe2::TypeMeta dtype() const {
    return data_type_;
  }
  size_t itemsize() const {
    return data_type_.itemsize();
  }
  void set_dtype(
      const caffe2::TypeMeta data_type) {
    data_type_ = data_type;
  }
  int64_t storage_offset() const {
    return storage_offset_;
  }
  const NCCLStorage &storage() const {
    return storage_;
  }
  whole_graph::WholeNCCLMemory_t GetNCCLMemory() const {
    return storage_.GetNCCLMemory();
  }

 protected:
  NCCLTensorImpl() = default;
  void refresh_numel() {
    numel_ = compute_numel();
  }

 private:
  int64_t compute_numel() const {
    int64_t n = 1;
    for (auto s : sizes()) {
      n *= s;
    }
    return n;
  }
  c10::impl::SizesAndStrides sizes_and_strides_;
  NCCLStorage storage_;
  int64_t storage_offset_ = 0;
  int64_t numel_ = 1;
  caffe2::TypeMeta data_type_;
};

class NCCLTensor {
 public:
  explicit NCCLTensor(c10::intrusive_ptr<NCCLTensorImpl> chunked_tensor_impl)
      : impl_(std::move(chunked_tensor_impl)) {
    if (impl_.get() == nullptr) {
      throw std::runtime_error("TensorImpl with nullptr is not supported");
    }
  }
  NCCLTensor(const NCCLTensor &) = default;
  NCCLTensor(NCCLTensor &&) = default;
  const c10::intrusive_ptr<NCCLTensorImpl> &getIntrusivePtr() const {
    return impl_;
  }
  caffe2::TypeMeta dtype() const noexcept {
    return impl_->dtype();
  }
  int64_t dim() const {
    return impl_->dim();
  }
  torch::IntArrayRef sizes() const {
    return impl_->sizes();
  }
  torch::IntArrayRef strides() const {
    return impl_->strides();
  }
  int64_t size(int64_t d) const {
    return sizes()[d];
  }
  int64_t stride(int64_t d) const {
    return strides()[d];
  }
  int64_t numel() const {
    return impl_->numel();
  }
  const NCCLStorage &storage() const {
    return impl_->storage();
  }
  int64_t storage_offset() const {
    return impl_->storage_offset();
  }
  int64_t get_ptr() {
    return (int64_t) this;
  }
  static NCCLTensor wrap_tensor_impl(
      c10::intrusive_ptr<NCCLTensorImpl> tensor_impl) {
    NCCLTensor r(std::move(tensor_impl));
    return r;
  }
  whole_graph::WholeNCCLMemory_t GetNCCLMemory() const {
    return impl_->GetNCCLMemory();
  }

 private:
  c10::intrusive_ptr<NCCLTensorImpl> impl_;
};

}// namespace pytorch

}// namespace whole_graph