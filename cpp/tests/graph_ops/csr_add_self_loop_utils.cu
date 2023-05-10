/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include "csr_add_self_loop_utils.hpp"
#include <gtest/gtest.h>

namespace graph_ops {
namespace testing {
void host_get_csr_add_self_loop(int* host_csr_row_ptr,
                                wholememory_array_description_t csr_row_ptr_array_desc,
                                int* host_csr_col_ptr,
                                wholememory_array_description_t csr_col_ptr_array_desc,
                                int* host_ref_output_csr_row_ptr,
                                wholememory_array_description_t output_csr_row_ptr_array_desc,
                                int* host_ref_output_csr_col_ptr,
                                wholememory_array_description_t output_csr_col_ptr_array_desc)
{
  for (int64_t row_id = 0; row_id < csr_row_ptr_array_desc.size - 1; row_id++) {
    int start                                   = host_csr_row_ptr[row_id];
    int end                                     = host_csr_col_ptr[row_id + 1];
    host_ref_output_csr_row_ptr[row_id]         = start + row_id;
    host_ref_output_csr_col_ptr[start + row_id] = row_id;
    for (int64_t j = start; j < end; j++) {
      host_ref_output_csr_col_ptr[j + row_id + 1] = host_csr_col_ptr[j];
    }
  }
  host_ref_output_csr_row_ptr[csr_row_ptr_array_desc.size - 1] =
    host_csr_row_ptr[csr_row_ptr_array_desc.size - 1] + csr_row_ptr_array_desc.size - 1;
}

void host_csr_add_self_loop(void* host_csr_row_ptr,
                            wholememory_array_description_t csr_row_ptr_array_desc,
                            void* host_csr_col_ptr,
                            wholememory_array_description_t csr_col_ptr_array_desc,
                            void* host_ref_output_csr_row_ptr,
                            wholememory_array_description_t output_csr_row_ptr_array_desc,
                            void* host_ref_output_csr_col_ptr,
                            wholememory_array_description_t output_csr_col_ptr_array_desc)
{
  EXPECT_EQ(csr_row_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(csr_col_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(output_csr_row_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(output_csr_col_ptr_array_desc.dtype, WHOLEMEMORY_DT_INT);
  EXPECT_EQ(csr_row_ptr_array_desc.size, output_csr_row_ptr_array_desc.size);
  EXPECT_EQ(csr_col_ptr_array_desc.size + csr_row_ptr_array_desc.size - 1,
            output_csr_col_ptr_array_desc.size);

  host_get_csr_add_self_loop(static_cast<int*>(host_csr_row_ptr),
                             csr_row_ptr_array_desc,
                             static_cast<int*>(host_csr_col_ptr),
                             csr_col_ptr_array_desc,
                             static_cast<int*>(host_ref_output_csr_row_ptr),
                             output_csr_row_ptr_array_desc,
                             static_cast<int*>(host_ref_output_csr_col_ptr),
                             output_csr_col_ptr_array_desc);
}

}  // namespace testing
}  // namespace graph_ops
