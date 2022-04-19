set(PYTORCH_TARGET_LIB "pytorch")

# Find PyTorch
find_package(Pytorch "1.2.0" "REQUIRED")
if(NOT PYTORCH_FOUND)
    return()
endif()

include_directories(SYSTEM ${Pytorch_INCLUDE_DIRS})
list(APPEND PYTORCH_LINKER_LIBS ${Pytorch_LIBRARIES})
include_directories(${PYTHON_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${Pytorch_COMPILE_FLAGS} ${CMAKE_CXX_FLAGS}")

pybind11_add_module(wholememory_pytorch
        ${PROJECT_SOURCE_DIR}/wholememory/torch/pytorch_cuda_env_fns.cc
        ${PROJECT_SOURCE_DIR}/wholememory/torch/pytorch_dtype.cc
        ${PROJECT_SOURCE_DIR}/wholememory/torch/whole_chunked_pytorch_tensor.cc
        ${PROJECT_SOURCE_DIR}/wholememory/torch/gather_gpu.cu
        ${PROJECT_SOURCE_DIR}/wholememory/torch/graph_sampler_gpu.cc
        ${PROJECT_SOURCE_DIR}/wholememory/torch/gnn_ops_gpu.cc
        ${PROJECT_SOURCE_DIR}/wholememory/torch/whole_memory_pytorch_wrapper.cc)
target_link_libraries(wholememory_pytorch PUBLIC pybind11::module ${LINKER_LIBS} ${PYTORCH_LINKER_LIBS} ${PYTHON_LIBRARIES} whole_memory)

