#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAStream.h>

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "whole_graph.h"
#include "whole_chunked_memory.h"
#include "whole_graph_embedding.h"
#include "whole_graph_graph.h"
#include "whole_chunked_pytorch_tensor.h"
#include "../macros.h"

namespace {

c10::Device GetCurrentCUDADevice() {
  int dev_id = -1;
  WM_CUDA_CHECK(cudaGetDevice(&dev_id));
  return {c10::DeviceType::CUDA, (int8_t)dev_id};
}

struct WholeMemoryInfo {
  c10::weak_intrusive_ptr<c10::StorageImpl> storage_impl{c10::intrusive_ptr<c10::StorageImpl>()};
  bool is_unified = false;
};

std::mutex wholememory_info_mutex;
std::unordered_map<void*, WholeMemoryInfo> wholememory_info_map;

void DeleteWholeMemoryMemory(void* ptr) {
  std::unique_lock<std::mutex> lock(wholememory_info_mutex);
  whole_graph::WmmpFree(ptr);
  wholememory_info_map.erase(ptr);
}

class WholeMemoryDeviceAllocator : public c10::Allocator {
 public:
  WholeMemoryDeviceAllocator() = default;
  static c10::DataPtr allocate_with_ranks(size_t nbytes, const std::vector<int>& ranks) {
    void* ptr = nullptr;
    whole_graph::WmmpMalloc(&ptr, nbytes, ranks.data(), ranks.size());
    c10::Device device = GetCurrentCUDADevice();
    //(void* data, void* ctx, DeleterFnPtr ctx_deleter, Device device)
    return {ptr, ptr, &DeleteWholeMemoryMemory, device};
  }
  c10::DataPtr allocate(size_t nbytes) const override {
    return allocate_with_ranks(nbytes, std::vector<int>());
  }
};

WholeMemoryDeviceAllocator* GetWholeMemoryDeviceAllocator() {
  static WholeMemoryDeviceAllocator whole_graph_device_allocator;
  return &whole_graph_device_allocator;
}

class WholeMemoryUnifiedAllocator : public c10::Allocator {
 public:
  WholeMemoryUnifiedAllocator() = default;
  static c10::DataPtr allocate_with_ranks(size_t nbytes, const std::vector<int>& ranks) {
    void* ptr = nullptr;
    whole_graph::WmmpMallocHost(&ptr, nbytes, ranks.data(), ranks.size());
    c10::Device device = GetCurrentCUDADevice();
    //(void* data, void* ctx, DeleterFnPtr ctx_deleter, Device device)
    return {ptr, ptr, &DeleteWholeMemoryMemory, device};
  }
  c10::DataPtr allocate(size_t nbytes) const override {
    return allocate_with_ranks(nbytes, std::vector<int>());
  }
};

WholeMemoryUnifiedAllocator* GetWholeMemoryUnifiedAllocator() {
  static WholeMemoryUnifiedAllocator whole_graph_unified_allocator;
  return &whole_graph_unified_allocator;
}

size_t CheckSizesAndStrides(const std::vector<int64_t> &sizes,
                            const std::vector<int64_t> &strides,
                            std::vector<int64_t> *fixed_strides) {
  size_t dim = sizes.size();
  size_t total_size = 1;
  if (!strides.empty()) {
    assert(sizes.size() == strides.size());
    for (size_t i = 0; i < dim; i++) {
      assert(strides[dim - i - 1] >= (int64_t)total_size);
      assert(sizes[dim - i -  1] > 0);
      total_size *= sizes[dim - i - 1];
    }
    *fixed_strides = strides;
  } else {
    fixed_strides->resize(dim);
    for (size_t i = 0; i < dim; i++) {
      assert(sizes[dim - i -  1] > 0);
      (*fixed_strides)[dim - i - 1] = total_size;
      total_size *= sizes[dim - i - 1];
    }
  }
  size_t storage_size = 1;
  if (dim >= 1) storage_size = sizes[0] * (*fixed_strides)[0];
  return storage_size;
}

}

void WholeMemoryInitLib() {
  whole_graph::WholeMemoryInit();
}

void WholeMemoryMultiProcessInit(int rank, int size) {
  assert (rank >= 0 && rank < size);
  whole_graph::WmmpInit(rank, size, nullptr);
}

void WholeMemoryMultiProcessFinalize() {
  whole_graph::WmmpFinalize();
}

void WholeMemoryFinalizeLib() {
  whole_graph::WholeMemoryFinalize();
}

torch::Tensor WholeMemoryCreateTensorScalarType(const std::vector<int64_t> &sizes,
                                      const std::vector<int64_t> &strides,
                                      torch::ScalarType type,
                                      bool is_unified,
                                      const std::vector<int> &ranks) {
  size_t elt_size = c10::elementSize(type);
  std::vector<int64_t> fixed_strides;
  size_t storage_elt_count = CheckSizesAndStrides(sizes, strides, &fixed_strides);
  c10::DataPtr data_ptr;
  if (!is_unified) {
    data_ptr = GetWholeMemoryDeviceAllocator()->allocate_with_ranks(storage_elt_count * elt_size, ranks);
  } else {
    data_ptr = GetWholeMemoryUnifiedAllocator()->allocate_with_ranks(storage_elt_count * elt_size, ranks);
  }
  void* data_ptr_void = data_ptr.get();
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
  torch::Tensor t = torch::Tensor::wrap_tensor_impl(c10::intrusive_ptr<torch::TensorImpl, torch::UndefinedTensorImpl>::make(std::move(storage), dispatch_key_set, data_type));
  t.getIntrusivePtr()->set_sizes_and_strides(c10::IntArrayRef({sizes}), c10::IntArrayRef({fixed_strides}));
  return t;
}

torch::Tensor WholeMemoryCreateTensor(const std::vector<int64_t> &sizes,
                                      const std::vector<int64_t> &strides,
                                      py::object dtype,
                                      bool is_unified,
                                      const std::vector<int> &ranks) {
  torch::ScalarType type = torch::python::detail::py_object_to_dtype(std::move(dtype));
  return WholeMemoryCreateTensorScalarType(sizes, strides, type, is_unified, ranks);
}

torch::Tensor WholeMemoryTensorViewFromDevice(torch::Tensor t, torch::Device dev) {
  //std::cerr << "[WholeMemoryTensorViewFromDevice] dev=" << dev << std::endl;
  void* data_ptr = t.data_ptr();
  c10::intrusive_ptr<c10::StorageImpl> storage_impl_ptr;
  {
    std::unique_lock<std::mutex> lock(wholememory_info_mutex);
    auto it = wholememory_info_map.find(data_ptr);
    assert(it != wholememory_info_map.end());
    if (!it->second.is_unified) {
      assert(dev.type() != c10::DeviceType::CPU);
    }
    storage_impl_ptr = it->second.storage_impl.lock();
    assert(storage_impl_ptr.get() != nullptr);
  }
  int dev_idx = dev.has_index() ? dev.index() : -1;
  assert(whole_graph::WmmpCanAccessFrom(storage_impl_ptr->data(), dev_idx));
  storage_impl_ptr->data_ptr().unsafe_set_device(dev);
  c10::Storage storage(storage_impl_ptr);
  caffe2::TypeMeta data_type;
  data_type = t.scalar_type();
  torch::DispatchKeySet dispatch_key_set({dev.type() == c10::DeviceType::CPU ? c10::DispatchKey::CPU : c10::DispatchKey::CUDA});
  torch::Tensor new_t = torch::Tensor::wrap_tensor_impl(c10::intrusive_ptr<torch::TensorImpl, torch::UndefinedTensorImpl>::make(std::move(storage), dispatch_key_set, data_type));
  new_t.getIntrusivePtr()->set_sizes_and_strides(t.sizes(), t.strides());
  return new_t;
}

py::tuple WholeMemoryAggregateSize(size_t local_size, const std::vector<int> &ranks) {
  size_t total_size, offset;
  total_size = whole_graph::WmmpAggregateSize(local_size, &offset, ranks.data(), ranks.size());
  return py::make_tuple(total_size, offset);
}

void WholeMemoryBarrier(std::vector<int>& ranks) {
  whole_graph::WmmpBarrier(ranks.data(), ranks.size());
}

whole_graph::pytorch::ChunkedTensor WholeMemoryCreateChunkedTensorScalarType(const std::vector<int64_t> &sizes,
                                                                    const std::vector<int64_t> &strides,
                                                                    torch::ScalarType type,
                                                                    const std::vector<int> &ranks) {
  size_t elt_size = c10::elementSize(type);
  std::vector<int64_t> fixed_strides;
  size_t storage_elt_count = CheckSizesAndStrides(sizes, strides, &fixed_strides);
  size_t min_granularity = elt_size;
  if (fixed_strides.size() > 1) min_granularity *= fixed_strides[fixed_strides.size() - 2];
  c10::intrusive_ptr<whole_graph::pytorch::StorageImpl>
      storage_impl_ptr = c10::make_intrusive<whole_graph::pytorch::StorageImpl>(storage_elt_count * elt_size,
                                                                                 min_granularity,
                                                                                 ranks);
  whole_graph::pytorch::Storage storage(storage_impl_ptr);
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
                                                                    const std::vector<int> &ranks) {
  torch::ScalarType type = torch::python::detail::py_object_to_dtype(std::move(dtype));
  return WholeMemoryCreateChunkedTensorScalarType(sizes, strides, type, ranks);
}

whole_graph::pytorch::ChunkedTensor WholeMemoryGetSubChunkedTensor(const whole_graph::pytorch::ChunkedTensor& ct,
                                                                    const std::vector<int64_t> &start,
                                                                    const std::vector<int64_t> &count) {
  int64_t dim = ct.dim();
  TORCH_CHECK(start.size() <= (size_t)dim,
              "[WholeMemoryGetSubChunkTensor] start should <= ct.dim(): ",
              start.size(),
              " v.s. ",
              dim);
  TORCH_CHECK(count.size() <= (size_t)dim,
              "[WholeMemoryGetSubChunkTensor] count should <= ct.dim(): ",
              start.size(),
              " v.s. ",
              dim);
  TORCH_CHECK(start.size() >= count.size(), "[WholeMemoryGetSubChunkTensor] count.size() should <= start.size().");
  std::vector<int64_t> new_size(dim), new_stride(dim);
  int64_t offset = ct.storage_offset();
  for (size_t i = 0; i < (size_t)dim; i++) {
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
  whole_graph::pytorch::Storage storage = ct.storage();
  whole_graph::pytorch::ChunkedTensor new_t = whole_graph::pytorch::ChunkedTensor::wrap_tensor_impl(
      c10::intrusive_ptr<whole_graph::pytorch::ChunkedTensorImpl>::make(std::move(storage), ct.dtype()));
  new_t.getIntrusivePtr()->set_sizes_and_strides(c10::IntArrayRef({new_size}), c10::IntArrayRef({new_stride}));
  new_t.getIntrusivePtr()->set_storage_offset(offset);
  return new_t;
}

void WholeMemoryLoadEmbeddingToTensorFromFilePrefix(torch::Tensor t,
                                                    py::object src_dtype,
                                                    py::object emb_dtype,
                                                    int64_t table_size,
                                                    int64_t embedding_dim,
                                                    const std::string &file_prefix) {
  torch::ScalarType src_scalar_type = torch::python::detail::py_object_to_dtype(std::move(src_dtype));
  torch::ScalarType emb_scalar_type = torch::python::detail::py_object_to_dtype(std::move(emb_dtype));
  void *wm_embedding = t.data_ptr();
  int64_t storage_offset = t.storage_offset();
  TORCH_CHECK(t.dim() == 2, "input wholememory tensor should be 2-D tensor");
  int64_t embedding_stride = t.stride(0);
  TORCH_CHECK(embedding_stride >= embedding_dim, "input wholememory embedding stride ", embedding_stride, " should >= embedding_dim ", embedding_dim);
  c10::ScalarType src_type, emb_type;
  src_type = src_scalar_type;
  emb_type = emb_scalar_type;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  whole_graph::WmmpLoadEmbeddingFromFilelists(wm_embedding,
                                               storage_offset,
                                               table_size,
                                               embedding_dim,
                                               embedding_stride,
                                               file_prefix,
                                               whole_graph::pytorch::C10ScalarToWMType(src_type),
                                               whole_graph::pytorch::C10ScalarToWMType(emb_type),
                                               stream);
}
void WholeMemoryLoadEmbeddingToChunkedTensorFromFilePrefix(const whole_graph::pytorch::ChunkedTensor& ct,
                                                           py::object src_dtype,
                                                           py::object emb_dtype,
                                                           int64_t table_size,
                                                           int64_t embedding_dim,
                                                           const std::string &file_prefix) {
  torch::ScalarType src_scalar_type = torch::python::detail::py_object_to_dtype(std::move(src_dtype));
  torch::ScalarType emb_scalar_type = torch::python::detail::py_object_to_dtype(std::move(emb_dtype));
  whole_graph::WholeChunkedMemory_t wmt_embedding = ct.GetChunkedMemory();
  int64_t storage_offset = ct.storage_offset();
  TORCH_CHECK(ct.dim() == 2, "input wholememory tensor should be 2-D tensor");
  int64_t embedding_stride = ct.stride(0);
  TORCH_CHECK(embedding_stride >= embedding_dim, "input wholememory embedding stride ", embedding_stride, " should >= embedding_dim ", embedding_dim);
  c10::ScalarType src_type, emb_type;
  src_type = src_scalar_type;
  emb_type = emb_scalar_type;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  whole_graph::WmmpLoadChunkedEmbeddingFromFilelists(wmt_embedding,
                                                      storage_offset,
                                                      table_size,
                                                      embedding_dim,
                                                      embedding_stride,
                                                      file_prefix,
                                                      whole_graph::pytorch::C10ScalarToWMType(src_type),
                                                      whole_graph::pytorch::C10ScalarToWMType(emb_type),
                                                      stream);
}

std::vector<int64_t> WholeMemoryLoadEdgeIndexFromFilePrefix(const std::string &file_prefix,
                                                            py::object id_dtype,
                                                            int64_t edge_embedding_dim) {
  torch::ScalarType id_scalar_type = torch::python::detail::py_object_to_dtype(std::move(id_dtype));
  c10::ScalarType id_type;
  id_type = id_scalar_type;
  void* local_edge_buffer;
  void* local_feature_buffer;
  int64_t total_edge_count, src_node_count = -1, dst_node_count = -1;
  int64_t local_edge_count = whole_graph::WmmpTraverseLoadEdgeDataFromFileList(&local_edge_buffer,
                                                                                &local_feature_buffer,
                                                                                &total_edge_count,
                                                                                &src_node_count,
                                                                                &dst_node_count,
                                                                                file_prefix,
                                                                                whole_graph::pytorch::C10ScalarToWMType(id_type),
                                                                                edge_embedding_dim);
  return {local_edge_count, (int64_t) local_edge_buffer, (int64_t) local_feature_buffer, total_edge_count, src_node_count, dst_node_count};
}

std::vector<torch::Tensor> WholeMemoryAllocateCSRGraph(const std::vector<int64_t> &edge_index,
                                                       py::object id_dtype,
                                                       bool directed,
                                                       bool reverse_edge,
                                                       bool add_self_loop,
                                                       bool use_host) {
  TORCH_CHECK(edge_index.size() == 6, "WholeMemoryAllocateCSRGraph edge_index should have size of 6.");

  int64_t total_src_node_count;
  if (!directed) {
    total_src_node_count = std::max(edge_index[4], edge_index[5]);
  } else {
    if (!reverse_edge) {
      total_src_node_count = edge_index[4];
    } else {
      total_src_node_count = edge_index[5];
    }
  }

  int64_t est_edge_count = whole_graph::EstimateEdgeCount(total_src_node_count, edge_index[3], directed, add_self_loop);
  return {WholeMemoryCreateTensorScalarType({total_src_node_count + 1}, {}, torch::ScalarType::Long, use_host, {}),
          WholeMemoryCreateTensor({est_edge_count}, {}, id_dtype, use_host, {})};
}

std::vector<whole_graph::pytorch::ChunkedTensor> WholeMemoryAllocateChunkedCSRGraph(const std::vector<int64_t> &edge_index,
                                                                                     py::object id_dtype,
                                                                                     bool directed,
                                                                                     bool reverse_edge,
                                                                                     bool add_self_loop) {
  TORCH_CHECK(edge_index.size() == 6, "WholeMemoryAllocateChunkedCSRGraph edge_index should have size of 6.");

  int64_t total_src_node_count;
  if (!directed) {
    total_src_node_count = std::max(edge_index[4], edge_index[5]);
  } else {
    if (!reverse_edge) {
      total_src_node_count = edge_index[4];
    } else {
      total_src_node_count = edge_index[5];
    }
  }

  int64_t
      est_edge_count = whole_graph::EstimateEdgeCount(total_src_node_count, edge_index[3], directed, add_self_loop);
  return {WholeMemoryCreateChunkedTensorScalarType({total_src_node_count + 1}, {}, torch::ScalarType::Long, {}),
          WholeMemoryCreateChunkedTensor({est_edge_count}, {}, std::move(id_dtype), {})};
}

int64_t WholeMemoryLoadToCSRGraphFromEdgeBuffer(const std::vector<int64_t> &edge_index,
                                                const torch::Tensor& wm_csr_row_ptr,
                                                const torch::Tensor& wm_csr_col_idx,
                                                py::object id_dtype,
                                                bool directed,
                                                bool reverse_edge,
                                                bool add_self_loop) {
  torch::ScalarType id_scalar_type = torch::python::detail::py_object_to_dtype(std::move(id_dtype));
  c10::ScalarType id_type;
  id_type = id_scalar_type;
  TORCH_CHECK(edge_index.size() == 6, "WholeMemoryLoadToCSRGraphFromEdgeBuffer edge_index should have size of 6.");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t final_total_edge_count;
  int64_t total_src_node_count;
  if (!directed) {
    total_src_node_count = std::max(edge_index[4], edge_index[5]);
  } else {
    if (!reverse_edge) {
      total_src_node_count = edge_index[4];
    } else {
      total_src_node_count = edge_index[5];
    }
  }
  //{local_edge_count, (int64_t) local_edge_buffer, (int64_t) local_feature_buffer, total_edge_count, src_node_count, dst_node_count};
  whole_graph::WmmpLoadToCSRGraphFromEdgeBuffer(wm_csr_row_ptr.data_ptr(),
                                                 wm_csr_col_idx.data_ptr(),
                                                 (void *) edge_index[1],
                                                 edge_index[0],
                                                 edge_index[4],
                                                 total_src_node_count,
                                                 &final_total_edge_count,
                                                 whole_graph::pytorch::C10ScalarToWMType(id_type),
                                                 directed,
                                                 reverse_edge,
                                                 add_self_loop,
                                                 stream);
  return final_total_edge_count;
}

int64_t WholeMemoryLoadToChunkedCSRGraphFromEdgeBuffer(const std::vector<int64_t> &edge_index,
                                                       const whole_graph::pytorch::ChunkedTensor& wm_csr_row_ptr,
                                                       const whole_graph::pytorch::ChunkedTensor& wm_csr_col_idx,
                                                       py::object id_dtype,
                                                       bool directed,
                                                       bool reverse_edge,
                                                       bool add_self_loop) {
  torch::ScalarType id_scalar_type = torch::python::detail::py_object_to_dtype(std::move(id_dtype));
  c10::ScalarType id_type;
  id_type = id_scalar_type;
  TORCH_CHECK(edge_index.size() == 6, "WholeMemoryLoadToChunkedCSRGraphFromEdgeBuffer edge_index should have size of 6.");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int64_t final_total_edge_count;
  int64_t total_src_node_count;
  if (!directed) {
    total_src_node_count = std::max(edge_index[4], edge_index[5]);
  } else {
    if (!reverse_edge) {
      total_src_node_count = edge_index[4];
    } else {
      total_src_node_count = edge_index[5];
    }
  }
  auto wmt_csr_row_ptr = wm_csr_row_ptr.GetChunkedMemory();
  auto wmt_csr_col_idx = wm_csr_col_idx.GetChunkedMemory();
  whole_graph::WmmpLoadToChunkedCSRGraphFromEdgeBuffer(wmt_csr_row_ptr,
                                                        wmt_csr_col_idx,
                                                        (void *) edge_index[1],
                                                        edge_index[0],
                                                        edge_index[4],
                                                        total_src_node_count,
                                                        &final_total_edge_count,
                                                        whole_graph::pytorch::C10ScalarToWMType(id_type),
                                                        directed,
                                                        reverse_edge,
                                                        add_self_loop,
                                                        stream);
  return final_total_edge_count;
}

void WholeMemoryFreeEdgeIndex(const std::vector<int64_t>& edge_index) {
  TORCH_CHECK(edge_index.size() == 6, "WholeMemoryFreeEdgeIndex edge_index should have size of 6.");
  whole_graph::WmmpFreeEdgeData((void*)edge_index[1], (void*)edge_index[2]);
}

static PyObject* ScaleTypeToDtype(c10::ScalarType st) {
  std::string name;
  switch(st) {
    case c10::ScalarType::Byte: { name = "uint8"; break; }
    case c10::ScalarType::Char: { name = "int8"; break; }
    case c10::ScalarType::Short: { name = "int16"; break; }
    case c10::ScalarType::Int: { name = "int32"; break; }
    case c10::ScalarType::Long: { name = "int64"; break; }
    case c10::ScalarType::Half: { name = "float16"; break; }
    case c10::ScalarType::Float: { name = "float32"; break; }
    case c10::ScalarType::Double: { name = "float64"; break; }
    case c10::ScalarType::BFloat16: { name = "bfloat16"; break; }
    default: { printf("[ScaleTypeToDtype] Type not supported.\n"); abort();}
  }
  PyObject* po = THPDtype_New(st, name);
  return po;
}

PYBIND11_MODULE(wholememory_pytorch, m) {
  py::class_<whole_graph::pytorch::ChunkedTensor>(m, "ChunkedTensor")
      .def(py::init<const whole_graph::pytorch::ChunkedTensor&>())
      .def("get_ptr", &whole_graph::pytorch::ChunkedTensor::get_ptr)
      .def_property_readonly("shape", &whole_graph::pytorch::ChunkedTensor::sizes)
      .def_property_readonly("stride", &whole_graph::pytorch::ChunkedTensor::strides)
      .def_property_readonly("dtype", [](const whole_graph::pytorch::ChunkedTensor& ct) {
        PyObject* po = ScaleTypeToDtype(ct.dtype().toScalarType());
        py::handle pyh(po);
        return py::reinterpret_steal<py::object>(pyh);
        //return ct.dtype().toScalarType();
      });
  m.doc() = "PyTorch API for WholeMemory.";
  m.def("init_lib", &WholeMemoryInitLib, "Init WholeMemory Lib");
  m.def("finalize_lib", &WholeMemoryFinalizeLib, "Finalize WholeMemory Lib");
  m.def("mp_init", &WholeMemoryMultiProcessInit, "Init Multi-Process for WholeMemory Lib");
  //m.def("mp_finalize", &WholeMemoryMultiProcessFinalize, "Finalize Multi-Process for WholeMemory");
  m.def("create_tensor", &WholeMemoryCreateTensor, "Create WholeMemory Tensor.");
  m.def("get_tensor_view", &WholeMemoryTensorViewFromDevice, "Get view from device for WholeMemory Tensor");
  m.def("aggregate_size", &WholeMemoryAggregateSize, "Aggregate sizes from ranks");
  m.def("barrier", &WholeMemoryBarrier, "Barrier for ranks");
  m.def("create_chunked_tensor", &WholeMemoryCreateChunkedTensor, "Create WholeMemory Chunked Tensor.");
  m.def("get_sub_chunked_tensor", &WholeMemoryGetSubChunkedTensor, "Get WholeMemory subtensor from Chunked Tensor.");
  m.def("load_embedding_to_tensor_from_file_prefix", &WholeMemoryLoadEmbeddingToTensorFromFilePrefix,
        "Load embedding to WholeMemory Tensor from file prefix.");
  m.def("load_embedding_to_chunked_tensor_from_file_prefix", &WholeMemoryLoadEmbeddingToChunkedTensorFromFilePrefix,
        "Load embedding to WholeMemory Chunked Tensor from file prefix.");
  m.def("load_edge_index_from_file_prefix", &WholeMemoryLoadEdgeIndexFromFilePrefix, "load edge index");
  m.def("allocate_csr_graph", &WholeMemoryAllocateCSRGraph, "allocate CSR graph");
  m.def("allocate_chunked_csr_graph", &WholeMemoryAllocateChunkedCSRGraph, "allocate Chunked CSR graph");
  m.def("load_to_csr_graph_from_edge_buffer", &WholeMemoryLoadToCSRGraphFromEdgeBuffer,
        "load graph to WholeMemory Tensor");
  m.def("load_to_chunked_csr_graph_from_edge_buffer", &WholeMemoryLoadToChunkedCSRGraphFromEdgeBuffer,
        "load graph to WholeMemory ChunkedTensor");
  m.def("free_edge_index", &WholeMemoryFreeEdgeIndex, "free edge index");
}
