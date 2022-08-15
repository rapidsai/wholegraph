#set(PYTORCH_TARGET_LIB "pytorch")

include_directories(SYSTEM ${TORCH_INCLUDE_DIRS})
list(APPEND TORCH_LINKER_LIBS ${TORCH_LIBRARIES})
include_directories(${PYTHON_INCLUDE_DIRS})

pybind11_add_module(wholegraph_pytorch
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/pytorch_cuda_env_fns.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/pytorch_dtype.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/whole_chunked_pytorch_tensor.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/whole_nccl_pytorch_tensor.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/gather_gpu.cu
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/graph_sampler_gpu.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/gnn_ops_gpu.cc
        ${PROJECT_SOURCE_DIR}/wholegraph/torch/whole_graph_pytorch_wrapper.cc)
target_link_libraries(wholegraph_pytorch PUBLIC pybind11::module ${LINKER_LIBS} ${TORCH_LINKER_LIBS} ${PYTHON_LIBRARIES} whole_graph)
set_property(TARGET wholegraph_pytorch PROPERTY CUDA_ARCHITECTURES 70-real 80-real 86-real 70-virtual)

