# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

function(add_code_checks)
  set(options "")
  set(oneValueArgs CWD CLANG_FORMAT CLANG_TIDY FLAKE8)
  set(multiValueArgs "")
  cmake_parse_arguments(code_checker
    "${options}" "${oneValueArgs}" "${multiValueArgs}"
    ${ARGN})

  # clang format checker
  add_custom_target(clang-format
    python scripts/run-clang-format.py
      -exe ${code_checker_CLANG_FORMAT}
    WORKING_DIRECTORY ${code_checker_CWD}
    VERBATIM
    COMMENT "Checks for code formatting using clang-format")

  # clang format inplace fixer
  add_custom_target(fix-clang-format
    python scripts/run-clang-format.py
      -inplace
      -exe ${code_checker_CLANG_FORMAT}
    WORKING_DIRECTORY ${code_checker_CWD}
    VERBATIM
    COMMENT "Fixes any code formatting issues using clang-format")

  # clang tidy checker
  add_custom_target(clang-tidy
    python scripts/run-clang-tidy.py
      -cdb ${PROJECT_BINARY_DIR}/compile_commands.json
      -exe ${code_checker_CLANG_TIDY}
    WORKING_DIRECTORY ${code_checker_CWD}
    VERBATIM
    COMMENT "Checks for coding conventions using clang-tidy")

  # flake8
  add_custom_target(flake8
    ${code_checker_FLAKE8} --exclude build*
    WORKING_DIRECTORY ${code_checker_CWD}
    VERBATIM
    COMMENT "Checks for python coding conventions using flake8")
endfunction(add_code_checks)
