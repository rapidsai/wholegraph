# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#

find_package(Doxygen 1.8.11)

function(add_doxygen_target)
  if(DOXYGEN_FOUND)
    if(NOT ${DOXYGEN_DOT_EXECUTABLE})
      set(DOXYGEN_DOT_EXECUTABLE "")
    endif()
    set(options "")
    set(oneValueArgs IN_DOXYFILE OUT_DOXYFILE OUT_DIR CWD)
    set(multiValueArgs DEP_TARGETS)
    cmake_parse_arguments(dox
      "${options}" "${oneValueArgs}" "${multiValueArgs}"
      ${ARGN})
    configure_file(${dox_IN_DOXYFILE} ${dox_OUT_DOXYFILE} @ONLY)

    file(MAKE_DIRECTORY ${dox_OUT_DIR})
    # add any extra files related to Doxygen documentation here!
    SET(doxy_extra_files)
    cmake_path(SET OUT_INDEXFILE "${dox_OUT_DIR}")
    cmake_path(APPEND OUT_INDEXFILE "index.html")

    add_custom_command(
      OUTPUT ${OUT_INDEXFILE}
      COMMAND ${DOXYGEN_EXECUTABLE} ${dox_OUT_DOXYFILE}
      # see https://451.sh/post/cmake-doxygen-improved/ for further explanation
      MAIN_DEPENDENCY ${dox_OUT_DOXYFILE} ${dox_IN_DOXYFILE}
      # add any extra files related to Doxygen documentation here!
      DEPENDS ${dox_DEP_TARGETS} ${doxy_extra_files}
      COMMENT "Generating doxygen docs"
      WORKING_DIRECTORY ${dox_CWD}
      VERBATIM
    )

    add_custom_target(doxygen ALL DEPENDS ${OUT_INDEXFILE}
      COMMENT "Generate doxygen docs")

  else()
    message("add_doxygen_target: doxygen exe not found")
  endif()
endfunction(add_doxygen_target)
