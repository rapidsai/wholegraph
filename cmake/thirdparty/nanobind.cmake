#=============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

if (NOT SKBUILD)
  message(WARNING "WHOLEGRAPH: This CMake file should be executed via scikit-build.")
endif()

if (SKBUILD)
  # Constrain FindPython to find the Python version used by scikit-build
  set(Python_VERSION "${PYTHON_VERSION_STRING}")
  set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
  set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
  set(Python_LIBRARIES "${PYTHON_LIBRARY}")
elseif (MSVC)
  # MSVC needs a little extra help finding the Python library
  find_package(PythonInterp)
  find_package(Python)
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(DEFAULT_CXX_FLAGS "")

# reset the default flags if we have DEBUG_CXXFLAGS
if(CMAKE_BUILD_TYPE MATCHES Debug)
  if (DEFINED ENV{DEBUG_CXXFLAGS})
    set(DEFAULT_CXX_FLAGS "$ENV{DEBUG_CXXFLAGS}")
    separate_arguments(DEFAULT_CXX_FLAGS)
    add_compile_options(${DEFAULT_CXX_FLAGS})
  endif()
endif()

execute_process(
  COMMAND
  "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE _tmp_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT
)
message(STATUS "WHOLEGRAPH: nanobind dir='${_tmp_dir}'")
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# Now import nanobind from CMake
find_package(nanobind CONFIG REQUIRED)
