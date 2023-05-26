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
#include <gtest/gtest.h>

#include <cuda_runtime_api.h>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>
#include <wholememory/wholememory_tensor.h>

#include "parallel_utils.hpp"

struct MatrixTestParam {
  MatrixTestParam& set_row(int64_t r)
  {
    row = r;
    return *this;
  }
  MatrixTestParam& set_col(int64_t c)
  {
    col = c;
    return *this;
  }
  MatrixTestParam& set_dtype(wholememory_dtype_t dt)
  {
    dtype = dt;
    return *this;
  }
  int64_t row               = 256LL * 128LL;
  int64_t col               = 256LL;
  wholememory_dtype_t dtype = WHOLEMEMORY_DT_FLOAT;
};

class WholeMemoryMatrixTest : public ::testing::TestWithParam<MatrixTestParam> {};

TEST(WholeMemoryMatrixTest, SubTensorTest)
{
  MatrixTestParam params;
  params.set_row(256LL * 128LL).set_col(256LL).set_dtype(WHOLEMEMORY_DT_INT);
  MultiProcessRun(1, [&params](int world_rank, int world_size) {
    EXPECT_EQ(wholememory_init(0), WHOLEMEMORY_SUCCESS);
    EXPECT_EQ(cudaSetDevice(0), cudaSuccess);

    wholememory_unique_id_t unique_id;
    wholememory_comm_t wm_comm;
    EXPECT_EQ(wholememory_create_unique_id(&unique_id), WHOLEMEMORY_SUCCESS);
    EXPECT_EQ(wholememory_create_communicator(&wm_comm, unique_id, world_rank, world_size),
              WHOLEMEMORY_SUCCESS);

    int64_t sizes[2] = {params.row, params.col};
    wholememory_matrix_description_t mat_desc =
      wholememory_create_matrix_desc(sizes, params.col, 0, params.dtype);
    wholememory_tensor_description_t tensor_desc;
    wholememory_copy_matrix_desc_to_tensor(&tensor_desc, &mat_desc);

    wholememory_tensor_t wholememory_tensor;
    EXPECT_EQ(
      wholememory_create_tensor(
        &wholememory_tensor, &tensor_desc, wm_comm, WHOLEMEMORY_MT_CONTINUOUS, WHOLEMEMORY_ML_HOST),
      WHOLEMEMORY_SUCCESS);
    wholememory_handle_t wm_handle = wholememory_tensor_get_memory_handle(wholememory_tensor);
    int* ptr                       = nullptr;
    EXPECT_EQ(wholememory_get_global_pointer((void**)&ptr, wm_handle), WHOLEMEMORY_SUCCESS);
    for (int64_t i = 0; i < params.row * params.col; i++) {
      ptr[i] = i;
    }
    wholememory_tensor_t wholememory_sub_tensor_0, wholememory_sub_tensor_1;
    wholememory_tensor_description_t sub_desc_0, sub_desc_1;

    int64_t starts_0[2] = {1, 10};
    int64_t ends_0[2]   = {-1, 100};
    int64_t starts_1[2] = {2, -1};
    int64_t ends_1[2]   = {10000, 80};

    EXPECT_EQ(wholememory_tensor_get_subtensor(
                wholememory_tensor, starts_0, ends_0, &wholememory_sub_tensor_0),
              WHOLEMEMORY_SUCCESS);
    sub_desc_0 = *wholememory_tensor_get_tensor_description(wholememory_sub_tensor_0);
    EXPECT_EQ(sub_desc_0.dim, 2);
    EXPECT_EQ(sub_desc_0.dtype, WHOLEMEMORY_DT_INT);
    EXPECT_EQ(sub_desc_0.storage_offset, params.col * 1 + 10);
    EXPECT_EQ(sub_desc_0.sizes[0], params.row - 1);
    EXPECT_EQ(sub_desc_0.sizes[1], 90);
    EXPECT_EQ(sub_desc_0.strides[0], 256);
    EXPECT_EQ(sub_desc_0.strides[1], 1);
    EXPECT_EQ(wholememory_tensor_get_subtensor(
                wholememory_sub_tensor_0, starts_1, ends_1, &wholememory_sub_tensor_1),
              WHOLEMEMORY_SUCCESS);
    sub_desc_1 = *wholememory_tensor_get_tensor_description(wholememory_sub_tensor_1);
    EXPECT_EQ(sub_desc_1.dim, 2);
    EXPECT_EQ(sub_desc_1.dtype, WHOLEMEMORY_DT_INT);
    EXPECT_EQ(sub_desc_1.storage_offset, params.col * 3 + 10);
    EXPECT_EQ(sub_desc_1.sizes[0], 10000 - 2);
    EXPECT_EQ(sub_desc_1.sizes[1], 80);
    EXPECT_EQ(sub_desc_1.strides[0], 256);
    EXPECT_EQ(sub_desc_1.strides[1], 1);

    EXPECT_EQ(wholememory_destroy_tensor(wholememory_sub_tensor_0), WHOLEMEMORY_SUCCESS);
    EXPECT_EQ(wholememory_destroy_tensor(wholememory_sub_tensor_1), WHOLEMEMORY_SUCCESS);

    for (int64_t i = 0; i < params.row * params.col; i++) {
      EXPECT_EQ(ptr[i], i);
    }

    EXPECT_EQ(wholememory_destroy_tensor(wholememory_tensor), WHOLEMEMORY_SUCCESS);

    EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);
    WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
  });
}
