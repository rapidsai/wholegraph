#=============================================================================
# Copyright (c) 2018-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================
# Find PyTorch
# Get PyTorch cmake path
set(PY_EXE ${Python_EXECUTABLE})
execute_process(COMMAND ${PY_EXE} -c "import torch.utils; print(torch.utils.cmake_prefix_path)"
        OUTPUT_VARIABLE TORCH_CMAKE_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
message("TORCH_CMAKE_PREFIX=${TORCH_CMAKE_PREFIX}")
set(Torch_ROOT "${TORCH_CMAKE_PREFIX}/Torch")
set(TORCH_CUDA_ARCH_LIST "7.0;8.0;8.6")
find_package(Torch "1.8.0" "REQUIRED")
#execute_process(COMMAND ${PY_EXE} -c "from torch.utils.cpp_extension import CUDAExtension as ext; e = ext('', []); print(';'.join(e.library_dirs))"
#        OUTPUT_VARIABLE Torch_LIBRARY_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
#message(STATUS "Torch_LIBRARY_DIRS=${Torch_LIBRARY_DIRS}")
#string(REGEX REPLACE "No CUDA runtime[^\n]*\n?" "" Torch_LIBRARY_DIRS "${Torch_LIBRARY_DIRS}")
#execute_process(COMMAND ${PY_EXE} -c "from torch.utils.cpp_extension import CUDAExtension as ext; e = ext('', []); print(';'.join(e.libraries))"
#        OUTPUT_VARIABLE _Torch_LIBRARIES OUTPUT_STRIP_TRAILING_WHITESPACE)
#string(REGEX REPLACE "No CUDA runtime[^\n]*\n?" "" _Torch_LIBRARIES "${_Torch_LIBRARIES}")
#foreach (_TLIB IN LISTS _Torch_LIBRARIES)
#    find_library(FOUND_LIB_${_TLIB}
#            NAMES ${_TLIB}
#            HINTS ${Torch_LIBRARY_DIRS})
#    list(APPEND TORCH_LIBRARIES ${FOUND_LIB_${_TLIB}})
#endforeach ()
if (NOT TORCH_FOUND)
    message(FATAL_ERROR "Torch not found.")
    return()
endif ()
#execute_process(COMMAND ${PY_EXE} -c "import torch; print(torch.torch.compiled_with_cxx11_abi())"
#        OUTPUT_VARIABLE Torch_CXX11 OUTPUT_STRIP_TRAILING_WHITESPACE)
#string(TOUPPER ${Torch_CXX11} Torch_CXX11)
#message(STATUS "Torch_CXX11: ${Torch_CXX11}")
#set(USE_CXX11_ABI ${Torch_CXX11})

file(GLOB WHOLEGRAPH_PYTORCH_SOURCES
        "*.cpp"
        "ops/*.cpp")

message(STATUS "TORCH_CXX_FLAGS=${TORCH_CXX_FLAGS}")

add_library(wholegraph_torch_cxx11abi SHARED "")
target_sources(wholegraph_torch_cxx11abi PRIVATE ${WHOLEGRAPH_PYTORCH_SOURCES})
set_property(TARGET torch PROPERTY INTERFACE_COMPILE_OPTIONS "-D_GLIBCXX_USE_CXX11_ABI=1")
target_include_directories(wholegraph_torch_cxx11abi
        PRIVATE
        "${wholegraph_ROOT}/include")
target_link_libraries(wholegraph_torch_cxx11abi "${TORCH_LIBRARIES}" "${WHOLEGRAPH_CPP_TARGET}")

add_library(wholegraph_torch_precxx11abi SHARED "")
target_sources(wholegraph_torch_precxx11abi PRIVATE ${WHOLEGRAPH_PYTORCH_SOURCES})
set_property(TARGET torch PROPERTY INTERFACE_COMPILE_OPTIONS "-D_GLIBCXX_USE_CXX11_ABI=0")
target_include_directories(wholegraph_torch_precxx11abi
        PRIVATE
        "${wholegraph_ROOT}/include")
target_link_libraries(wholegraph_torch_precxx11abi "${TORCH_LIBRARIES}" "${WHOLEGRAPH_CPP_TARGET}")

set_property(TARGET torch PROPERTY INTERFACE_COMPILE_OPTIONS ${TORCH_CXX_FLAGS})
