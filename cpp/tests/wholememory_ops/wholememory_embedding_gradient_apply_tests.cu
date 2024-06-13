/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <wholememory/embedding.h>

#include <map>
#include <string>

#include "../wholememory/wholememory_test_utils.hpp"
#include "embedding_test_utils.hpp"
#include "wholememory/env_func_ptrs.hpp"

struct EmbeddingBackwardTestParams {
  EmbeddingBackwardTestParams()
  {
    const int64_t kDefaultEmbeddingEntryCount = 400001;
    const int64_t kDefaultEmbeddingDim        = 127;
    const int64_t kDefaultGatherIndiceCount   = 100005;
    int64_t embedding_sizes[2]                = {kDefaultEmbeddingEntryCount, kDefaultEmbeddingDim};
    embedding_description                     = wholememory_create_matrix_desc(
      &embedding_sizes[0], kDefaultEmbeddingDim, 0, WHOLEMEMORY_DT_FLOAT);
    indice_description =
      wholememory_create_array_desc(kDefaultGatherIndiceCount, 0, WHOLEMEMORY_DT_INT64);
    int64_t output_sizes[2] = {kDefaultGatherIndiceCount, kDefaultEmbeddingDim};
    grad_description        = wholememory_create_matrix_desc(
      &output_sizes[0], kDefaultEmbeddingDim, 0, WHOLEMEMORY_DT_FLOAT);
  }
  bool is_large_test()
  {
    int64_t embedding_table_mem_size =
      wholememory_get_memory_element_count_from_matrix(&embedding_description) *
      wholememory_dtype_get_element_size(embedding_description.dtype);
    if (embedding_table_mem_size > 2LL * 1024 * 1024 * 1024) return true;
    return false;
  }
  EmbeddingBackwardTestParams& set_entry_count(int64_t entry_count)
  {
    embedding_description.sizes[0] = entry_count;
    return *this;
  }
  EmbeddingBackwardTestParams& set_embedding_dim(int embedding_dim)
  {
    embedding_description.sizes[1] = embedding_dim;
    grad_description.sizes[1]      = embedding_dim;
    embedding_description.stride   = embedding_dim;
    if (grad_description.stride < embedding_dim) grad_description.stride = embedding_dim;
    return *this;
  }
  EmbeddingBackwardTestParams& set_indice_count(int indice_count)
  {
    indice_description.size   = indice_count;
    grad_description.sizes[0] = indice_count;
    return *this;
  }
  EmbeddingBackwardTestParams& set_indice_dtype(wholememory_dtype_t dtype)
  {
    indice_description.dtype = dtype;
    return *this;
  }
  EmbeddingBackwardTestParams& set_grad_stride(int stride)
  {
    grad_description.stride = stride;
    return *this;
  }
  EmbeddingBackwardTestParams& set_memory_type(wholememory_memory_type_t mt)
  {
    memory_type = mt;
    return *this;
  }
  EmbeddingBackwardTestParams& set_memory_location(wholememory_memory_location_t ml)
  {
    memory_location = ml;
    return *this;
  }
  EmbeddingBackwardTestParams& set_cache_memory_type(wholememory_memory_type_t cmt)
  {
    cache_memory_type = cmt;
    return *this;
  }
  EmbeddingBackwardTestParams& set_cache_memory_location(wholememory_memory_location_t cml)
  {
    cache_memory_location = cml;
    return *this;
  }
  EmbeddingBackwardTestParams& set_cache_ratio(float ratio)
  {
    cache_ratio = ratio;
    return *this;
  }
  wholememory_embedding_cache_policy_t get_cache_policy(wholememory_comm_t comm)
  {
    wholememory_embedding_cache_policy_t cache_policy = nullptr;
    if (!use_cache) return nullptr;
    EXPECT_EQ(wholememory_create_embedding_cache_policy(&cache_policy,
                                                        comm,
                                                        cache_memory_type,
                                                        cache_memory_location,
                                                        WHOLEMEMORY_AT_READWRITE,
                                                        cache_ratio),
              WHOLEMEMORY_SUCCESS);
    return cache_policy;
  }
  EmbeddingBackwardTestParams& set_use_cache()
  {
    use_cache = true;
    return *this;
  }
  EmbeddingBackwardTestParams& set_run_count(int rc)
  {
    run_count = rc;
    return *this;
  }
  EmbeddingBackwardTestParams& set_optimizer_type(wholememory_optimizer_type_t opt_type)
  {
    optimizer_type = opt_type;
    return *this;
  }
  EmbeddingBackwardTestParams& set_optimizer_params(const std::string& param_name, float value)
  {
    optimizer_params[param_name] = value;
    return *this;
  }
  EmbeddingBackwardTestParams& set_lr(const std::string& param_name, float lr)
  {
    lr_ = lr;
    return *this;
  }
  wholememory_array_description_t indice_description;
  wholememory_matrix_description_t embedding_description;
  wholememory_matrix_description_t grad_description;
  wholememory_memory_type_t memory_type               = WHOLEMEMORY_MT_CHUNKED;
  wholememory_memory_location_t memory_location       = WHOLEMEMORY_ML_HOST;
  wholememory_memory_type_t cache_memory_type         = WHOLEMEMORY_MT_CHUNKED;
  wholememory_memory_location_t cache_memory_location = WHOLEMEMORY_ML_DEVICE;
  wholememory_optimizer_type_t optimizer_type         = WHOLEMEMORY_OPT_SGD;
  float cache_ratio                                   = 0.2;
  bool use_cache                                      = false;
  int run_count                                       = 3;

  float lr_ = 0.1;

  std::map<std::string, float> optimizer_params;
};

class WholeMemoryEmbeddingBackwardParameterTests
  : public ::testing::TestWithParam<EmbeddingBackwardTestParams> {};

class CPUOptimizer {
 public:
  CPUOptimizer(EmbeddingBackwardTestParams* params, int64_t start_entry, int64_t end_entry)
    : params_(params), start_entry_(start_entry), end_entry_(end_entry)
  {
    parse_params();
    create_optimizer_states();
  }
  ~CPUOptimizer() = default;
  void Apply(float lr,
             const std::vector<int64_t>& indices,
             const std::vector<std::vector<float>>& grads,
             std::vector<std::vector<float>>& embs)
  {
    for (int64_t i = 0; i < indices.size(); i++) {
      int64_t index       = indices[i];
      int64_t local_index = index - start_entry_;
      auto& grad_vec      = grads[i];
      auto& emb_vec       = embs[index];
      switch (params_->optimizer_type) {
        case WHOLEMEMORY_OPT_LAZY_ADAM: {
          ApplyLazyAdam(lr, local_index, grad_vec, emb_vec);
          break;
        }
        case WHOLEMEMORY_OPT_SGD: {
          ApplySGD(lr, local_index, grad_vec, emb_vec);
          break;
        }
        case WHOLEMEMORY_OPT_ADAGRAD: {
          ApplyAdaGrad(lr, local_index, grad_vec, emb_vec);
          break;
        }
        case WHOLEMEMORY_OPT_RMSPROP: {
          ApplyRMSProp(lr, local_index, grad_vec, emb_vec);
          break;
        }
        default: {
          FAIL();
        }
      }
    }
  }

 private:
  void ApplyLazyAdam(float lr,
                     int64_t local_index,
                     const std::vector<float>& grad_vec,
                     std::vector<float>& emb_vec)
  {
    auto& m_vec  = optimizer_states_[0][local_index];
    auto& v_vec  = optimizer_states_[1][local_index];
    float beta1t = per_embedding_states_[0][local_index];
    float beta2t = per_embedding_states_[1][local_index];
    beta1t *= beta1_;
    beta2t *= beta2_;
    per_embedding_states_[0][local_index] = beta1t;
    per_embedding_states_[1][local_index] = beta2t;
    for (int i = 0; i < embedding_dim_; i++) {
      float grad_value = grad_vec[i];
      float emb_value  = emb_vec[i];
      if (adam_w_) {
        emb_value -= lr * weight_decay_ * emb_value;
      } else {
        grad_value += weight_decay_ * emb_value;
      }
      float m          = m_vec[i];
      float v          = v_vec[i];
      m                = beta1_ * m + (1 - beta1_) * grad_value;
      v                = beta2_ * v + (1 - beta2_) * grad_value * grad_value;
      float const mhat = m / (1 - beta1t);
      float const vhat = v / (1 - beta2t);
      emb_value        = emb_value - lr * mhat / (sqrtf(vhat) + epsilon_);
      emb_vec[i]       = emb_value;
      m_vec[i]         = m;
      v_vec[i]         = v;
    }
  }
  void ApplyAdaGrad(float lr,
                    int64_t local_index,
                    const std::vector<float>& grad_vec,
                    std::vector<float>& emb_vec)
  {
    auto& state_sum_vec = optimizer_states_[0][local_index];
    for (int i = 0; i < embedding_dim_; i++) {
      float grad_value = grad_vec[i];
      float emb_value  = emb_vec[i];
      grad_value += weight_decay_ * emb_value;
      float state_sum = state_sum_vec[i];
      state_sum += grad_value * grad_value;
      emb_value        = emb_value - lr * grad_value / (sqrtf(state_sum) + epsilon_);
      emb_vec[i]       = emb_value;
      state_sum_vec[i] = state_sum;
    }
  }
  void ApplyRMSProp(float lr,
                    int64_t local_index,
                    const std::vector<float>& grad_vec,
                    std::vector<float>& emb_vec)
  {
    auto& v_vec = optimizer_states_[0][local_index];
    for (int i = 0; i < embedding_dim_; i++) {
      float grad_value = grad_vec[i];
      float emb_value  = emb_vec[i];
      grad_value += weight_decay_ * emb_value;
      auto v     = v_vec[i];
      v          = alpha_ * v + (1 - alpha_) * grad_value * grad_value;
      emb_value  = emb_value - lr * grad_value / (sqrtf(v) + epsilon_);
      emb_vec[i] = emb_value;
      v_vec[i]   = v;
    }
  }
  void ApplySGD(float lr,
                int64_t local_index,
                const std::vector<float>& grad_vec,
                std::vector<float>& emb_vec)
  {
    for (int i = 0; i < embedding_dim_; i++) {
      float grad_value = grad_vec[i];
      float emb_value  = emb_vec[i];
      grad_value += weight_decay_ * emb_value;
      emb_value -= lr * grad_value;
      emb_vec[i] = emb_value;
    }
  }
  void parse_params()
  {
    for (auto& optimizer_param : params_->optimizer_params) {
      auto name         = optimizer_param.first;
      float const value = optimizer_param.second;
      if (name == "weight_decay") {
        weight_decay_ = value;
      } else if (name == "epsilon") {
        epsilon_ = value;
      } else if (name == "alpha") {
        alpha_ = value;
      } else if (name == "beta1") {
        beta1_ = value;
      } else if (name == "beta2") {
        beta2_ = value;
      } else if (name == "adam_w") {
        adam_w_ = value > 0.5;
      } else {
        FAIL();
      }
    }
  }
  void create_optimizer_states()
  {
    switch (params_->optimizer_type) {
      case WHOLEMEMORY_OPT_LAZY_ADAM: {
        state_count_         = 2;
        per_embedding_count_ = 2;
        break;
      }
      case WHOLEMEMORY_OPT_SGD: {
        state_count_         = 0;
        per_embedding_count_ = 0;
        break;
      }
      case WHOLEMEMORY_OPT_ADAGRAD: {
        state_count_         = 1;
        per_embedding_count_ = 1;
        break;
      }
      case WHOLEMEMORY_OPT_RMSPROP: {
        state_count_         = 1;
        per_embedding_count_ = 1;
        break;
      }
      default: {
        FAIL();
      }
    }
    embedding_dim_ = params_->grad_description.sizes[1];
    optimizer_states_.resize(state_count_);
    per_embedding_states_.resize(per_embedding_count_);
    for (int i = 0; i < state_count_; i++) {
      optimizer_states_[i].resize(end_entry_ - start_entry_);
      for (int j = 0; j < end_entry_ - start_entry_; j++) {
        optimizer_states_[i][j].resize(embedding_dim_, 0.0f);
      }
    }
    for (int i = 0; i < per_embedding_count_; i++) {
      per_embedding_states_[i].resize(end_entry_ - start_entry_, 1.0f);
    }
  }
  EmbeddingBackwardTestParams* params_;
  int64_t start_entry_;
  int64_t end_entry_;

  float weight_decay_ = 0.0f;
  float epsilon_      = 1e-8f;
  float alpha_        = 0.99f;
  float beta1_        = 0.9f;
  float beta2_        = 0.999f;
  bool adam_w_        = false;

  int embedding_dim_       = 0;
  int state_count_         = 0;
  int per_embedding_count_ = 0;
  std::vector<std::vector<std::vector<float>>> optimizer_states_;
  std::vector<std::vector<float>> per_embedding_states_;
};

void prepare_data_and_reference(
  EmbeddingBackwardTestParams& params,
  int world_size,
  std::vector<std::vector<std::vector<int64_t>>>& step_rank_indices,
  std::vector<std::vector<std::vector<std::vector<float>>>>& step_rank_grads,
  std::vector<std::vector<float>>& start_embedding_table,
  std::vector<std::vector<float>>& end_embedding_table)
{
  step_rank_indices.resize(params.run_count);
  step_rank_grads.resize(params.run_count);
  for (int run = 0; run < params.run_count; run++) {
    step_rank_indices[run].resize(world_size);
    step_rank_grads[run].resize(world_size);
  }
  int cpu_count        = GetProcessorCount();
  int thread_count     = std::max(1, cpu_count - 1);  // reserve one core for other usage
  int run_thread_count = std::min(thread_count, params.run_count * world_size);
  MultiThreadRun(run_thread_count,
                 [&step_rank_indices, &step_rank_grads, &params, world_size](
                   int thread_rank, int thread_world_size) {
                   for (int idx = thread_rank; idx < params.run_count * world_size;
                        idx += thread_world_size) {
                     int run               = idx / world_size;
                     int world_rank        = idx % world_size;
                     auto& indice          = step_rank_indices[run][world_rank];
                     auto& grads           = step_rank_grads[run][world_rank];
                     int rank_indice_count = params.indice_description.size;
                     indice.resize(rank_indice_count);
                     grads.resize(rank_indice_count);
                     wholememory_array_description_t init_indice_desc = params.indice_description;
                     init_indice_desc.dtype                           = WHOLEMEMORY_DT_INT64;
                     int64_t entry_count = params.embedding_description.sizes[0];
                     wholememory_ops::testing::host_random_init_indices(
                       indice.data(), init_indice_desc, entry_count);
                     for (int i = 0; i < rank_indice_count; i++) {
                       grads[i].resize(params.grad_description.sizes[1]);
                       wholememory_ops::testing::host_random_init_float(
                         grads[i].data(), params.grad_description.sizes[1], -5.0, 10);
                     }
                   }
                 });
  start_embedding_table.resize(params.embedding_description.sizes[0]);
  end_embedding_table.resize(params.embedding_description.sizes[0]);
  MultiThreadRun(thread_count,
                 [&params, &start_embedding_table, &end_embedding_table](int thread_rank,
                                                                         int thread_world_size) {
                   int64_t total_entry_count = start_embedding_table.size();
                   int64_t start_entry       = thread_rank * total_entry_count / thread_world_size;
                   int64_t end_entry = (thread_rank + 1) * total_entry_count / thread_world_size;
                   int embedding_dim = params.grad_description.sizes[1];
                   for (int64_t entry = start_entry; entry < end_entry; entry++) {
                     start_embedding_table[entry].resize(embedding_dim);
                     wholememory_ops::testing::host_random_init_float(
                       start_embedding_table[entry].data(), embedding_dim, -10.0, 10);
                     end_embedding_table[entry] = start_embedding_table[entry];
                   }
                 });
  MultiThreadRun(run_thread_count,
                 [world_size, &params, &step_rank_indices, &step_rank_grads, &end_embedding_table](
                   int thread_rank, int thread_world_size) {
                   int64_t total_entry_count = end_embedding_table.size();
                   int64_t start_entry       = thread_rank * total_entry_count / thread_world_size;
                   int64_t end_entry = (thread_rank + 1) * total_entry_count / thread_world_size;
                   CPUOptimizer cpu_optimizer(&params, start_entry, end_entry);
                   int embedding_dim = params.grad_description.sizes[1];
                   for (int step = 0; step < params.run_count; step++) {
                     int step_id = std::min(step, params.run_count - 1);
                     std::vector<int64_t> indices;
                     std::vector<std::vector<float>> grads;
                     std::unordered_map<int64_t, int> indice_map;
                     for (int rank = 0; rank < world_size; rank++) {
                       auto& indices_vec         = step_rank_indices[step_id][rank];
                       auto& grad_vec            = step_rank_grads[step_id][rank];
                       int64_t rank_indice_count = indices_vec.size();
                       EXPECT_EQ(rank_indice_count, grad_vec.size());
                       for (int i = 0; i < rank_indice_count; i++) {
                         int64_t idx = indices_vec[i];
                         if (idx < start_entry || idx >= end_entry) continue;
                         auto& grad_data = grad_vec[i];
                         auto it         = indice_map.find(idx);
                         if (it == indice_map.end()) {
                           indice_map[idx] = indices.size();
                           indices.push_back(idx);
                           grads.resize(grads.size() + 1);
                           grads.back() = grad_data;
                         } else {
                           int64_t array_idx = it->second;
                           for (int d = 0; d < embedding_dim; d++) {
                             grads[array_idx][d] += grad_data[d];
                           }
                         }
                       }
                     }

                     float lr = params.lr_;
                     cpu_optimizer.Apply(lr, indices, grads, end_embedding_table);
                   }
                 });
}

template <typename IndiceT>
void copy_indices(const int64_t* src_indice, IndiceT* dst_indice, int64_t indice_count)
{
  for (int64_t i = 0; i < indice_count; i++) {
    dst_indice[i] = src_indice[i];
  }
}

static void host_expect_all_close(const float* data_ptr,
                                  const float* ref_ptr,
                                  const float* old_ptr,
                                  int64_t count,
                                  int64_t entry,
                                  float atol = 1e-5,
                                  float rtol = 1e-5)
{
  int diff_count = 0;
  for (int64_t i = 0; i < count && diff_count < 10; i++) {
    float data = data_ptr[i];
    float ref  = ref_ptr[i];
    float aerr = abs(data - ref);
    if (aerr < atol) continue;
    float rerr = aerr / std::max(abs(data), abs(ref));
    if (rerr < rtol) continue;
    diff_count++;
    EXPECT_LT(rerr, rtol) << "data[" << entry << "][" << i << "]=" << data << ", but ref is " << ref
                          << ", old is " << old_ptr[i];
  }
}

TEST_P(WholeMemoryEmbeddingBackwardParameterTests, EmbeddingGatherGradientApplyTest)
{
  auto params = GetParam();
  EXPECT_EQ(params.embedding_description.sizes[1], params.grad_description.sizes[1]);
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  if (dev_count == 1 && params.is_large_test()) {
    GTEST_SKIP() << "skipping large test on single gpu";
  }
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);

  std::vector<std::vector<std::vector<int64_t>>> step_rank_indices;
  std::vector<std::vector<std::vector<std::vector<float>>>> step_rank_grads;
  std::vector<std::vector<float>> start_embedding_table;
  std::vector<std::vector<float>> ref_end_embedding_table;

  prepare_data_and_reference(params,
                             dev_count,
                             step_rank_indices,
                             step_rank_grads,
                             start_embedding_table,
                             ref_end_embedding_table);

  MultiProcessRun(
    dev_count,
    [&params,
     &pipes,
     &step_rank_indices,
     &step_rank_grads,
     &start_embedding_table,
     &ref_end_embedding_table](int world_rank, int world_size) {
      EXPECT_EQ(wholememory_init(0), WHOLEMEMORY_SUCCESS);
      EXPECT_EQ(cudaSetDevice(world_rank), cudaSuccess);
      wholememory_comm_t wm_comm    = create_communicator_by_pipes(pipes, world_rank, world_size);
      wholememory_comm_t cache_comm = wm_comm;

      if (wholememory_communicator_support_type_location(
            wm_comm, params.memory_type, params.memory_location) != WHOLEMEMORY_SUCCESS ||
          (params.use_cache &&
           wholememory_communicator_support_type_location(
             cache_comm, params.cache_memory_type, params.cache_memory_location) !=
             WHOLEMEMORY_SUCCESS)) {
        EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);
        EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);
        WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
        if (world_rank == 0) GTEST_SKIP_("Skip due to not supported.");
        return;
      }

      int64_t embedding_dim = params.embedding_description.sizes[1];

      void *dev_indices = nullptr, *dev_grad_buffer = nullptr;
      void *host_indices = nullptr, *host_grad_buffer = nullptr;
      size_t grad_buffer_size = wholememory_get_memory_size_from_matrix(&params.grad_description);
      size_t indices_buffer_size =
        wholememory_get_memory_size_from_array(&params.indice_description);

      cudaStream_t stream;
      EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);

      EXPECT_EQ(cudaMallocHost(&host_indices, indices_buffer_size), cudaSuccess);
      EXPECT_EQ(cudaMalloc(&dev_indices, indices_buffer_size), cudaSuccess);
      EXPECT_EQ(cudaMalloc(&dev_grad_buffer, grad_buffer_size), cudaSuccess);
      EXPECT_EQ(cudaMallocHost(&host_grad_buffer, grad_buffer_size), cudaSuccess);

      wholememory_tensor_t indices_tensor, grad_tensor;
      wholememory_tensor_description_t indices_tensor_desc, grad_tensor_desc;
      wholememory_copy_array_desc_to_tensor(&indices_tensor_desc, &params.indice_description);
      wholememory_copy_matrix_desc_to_tensor(&grad_tensor_desc, &params.grad_description);
      EXPECT_EQ(
        wholememory_make_tensor_from_pointer(&indices_tensor, dev_indices, &indices_tensor_desc),
        WHOLEMEMORY_SUCCESS);
      EXPECT_EQ(
        wholememory_make_tensor_from_pointer(&grad_tensor, dev_grad_buffer, &grad_tensor_desc),
        WHOLEMEMORY_SUCCESS);

      wholememory_embedding_cache_policy_t cache_policy = params.get_cache_policy(cache_comm);

      wholememory_embedding_t wm_embedding;
      wholememory_tensor_description_t embedding_tensor_description;
      wholememory_copy_matrix_desc_to_tensor(&embedding_tensor_description,
                                             &params.embedding_description);

      wholememory_embedding_optimizer_t optimizer;
      EXPECT_EQ(wholememory_create_embedding_optimizer(&optimizer, params.optimizer_type),
                WHOLEMEMORY_SUCCESS);

      for (auto& param_name_value : params.optimizer_params) {
        EXPECT_EQ(wholememory_optimizer_set_parameter(
                    optimizer, param_name_value.first.c_str(), &param_name_value.second),
                  WHOLEMEMORY_SUCCESS);
      }

      EXPECT_EQ(wholememory_create_embedding(&wm_embedding,
                                             &embedding_tensor_description,
                                             wm_comm,
                                             params.memory_type,
                                             params.memory_location,
                                             cache_policy),
                WHOLEMEMORY_SUCCESS);
      EXPECT_EQ(wholememory_embedding_set_optimizer(wm_embedding, optimizer), WHOLEMEMORY_SUCCESS);
      wholememory_tensor_t embedding_tensor =
        wholememory_embedding_get_embedding_tensor(wm_embedding);
      wholememory_tensor_t local_embed_tensor;
      EXPECT_EQ(wholememory_tensor_map_local_tensor(embedding_tensor, &local_embed_tensor),
                WHOLEMEMORY_SUCCESS);
      wholememory_handle_t embedding_handle =
        wholememory_tensor_get_memory_handle(embedding_tensor);
      auto entry_per_partition  = wholememory_tensor_get_entry_per_partition(embedding_tensor);
      int64_t total_entry_count = params.embedding_description.sizes[0];
      int64_t rank_start_entry =
        std::min<int64_t>(world_rank * entry_per_partition, total_entry_count);
      int64_t rank_end_entry =
        std::min<int64_t>((world_rank + 1) * entry_per_partition, total_entry_count);
      int64_t rank_entry_count = rank_end_entry - rank_start_entry;

      auto* dst_base_ptr =
        static_cast<float*>(wholememory_tensor_get_data_pointer(local_embed_tensor));
      size_t dst_stride = wholememory_tensor_get_tensor_description(local_embed_tensor)->strides[0];
      size_t embedding_copy_size = embedding_dim * sizeof(float);

      for (int64_t i = 0; i < rank_entry_count; i++) {
        WM_CUDA_CHECK_NO_THROW(cudaMemcpy(dst_base_ptr + i * dst_stride,
                                          start_embedding_table[rank_start_entry + i].data(),
                                          embedding_copy_size,
                                          cudaMemcpyHostToDevice));
      }
      EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
      EXPECT_EQ(wholememory_communicator_barrier(wm_comm), WHOLEMEMORY_SUCCESS);

      for (int run = 0; run < params.run_count; run++) {
        int step_id            = std::min(run, params.run_count - 1);
        auto& rank_indices_vec = step_rank_indices[step_id][world_rank];
        auto& rank_grads_vec   = step_rank_grads[step_id][world_rank];
        int64_t indice_count   = rank_indices_vec.size();
        if (params.indice_description.dtype == WHOLEMEMORY_DT_INT64) {
          copy_indices(rank_indices_vec.data(), static_cast<int64_t*>(host_indices), indice_count);
        } else {
          copy_indices(rank_indices_vec.data(), static_cast<int*>(host_indices), indice_count);
        }
        float* host_grad_float_ptr = static_cast<float*>(host_grad_buffer);
        size_t grad_stride         = params.grad_description.stride;
        size_t grad_vec_size       = params.grad_description.sizes[1];
        for (int64_t i = 0; i < indice_count; i++) {
          memcpy(&host_grad_float_ptr[i * grad_stride],
                 rank_grads_vec[i].data(),
                 grad_vec_size * sizeof(float));
        }
        auto indice_dtype = params.indice_description.dtype;

        EXPECT_EQ(cudaMemcpy(dev_indices,
                             host_indices,
                             indice_count * wholememory_dtype_get_element_size(indice_dtype),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        EXPECT_EQ(cudaMemcpy(dev_grad_buffer,
                             host_grad_buffer,
                             indice_count * grad_stride * sizeof(float),
                             cudaMemcpyHostToDevice),
                  cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
        wholememory_embedding_gather_gradient_apply(wm_embedding,
                                                    indices_tensor,
                                                    grad_tensor,
                                                    true,
                                                    params.lr_,
                                                    wholememory::get_default_env_func(),
                                                    reinterpret_cast<int64_t>(stream));
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        EXPECT_EQ(wholememory_communicator_barrier(wm_comm), WHOLEMEMORY_SUCCESS);
      }
      EXPECT_EQ(
        wholememory_embedding_writeback_cache(wm_embedding, reinterpret_cast<int64_t>(stream)),
        WHOLEMEMORY_SUCCESS);
      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      EXPECT_EQ(wholememory_communicator_barrier(wm_comm), WHOLEMEMORY_SUCCESS);

      std::vector<std::vector<float>> local_end_embedding(rank_entry_count);
      for (int64_t i = 0; i < rank_entry_count; i++) {
        local_end_embedding[i].resize(embedding_dim);
        EXPECT_EQ(cudaMemcpy(local_end_embedding[i].data(),
                             dst_base_ptr + i * dst_stride,
                             embedding_copy_size,
                             cudaMemcpyDeviceToHost),
                  cudaSuccess);
      }
      EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
      for (int64_t i = 0; i < rank_entry_count; i++) {
        if (::testing::Test::HasFailure()) break;
        host_expect_all_close(local_end_embedding[i].data(),
                              ref_end_embedding_table[i + rank_start_entry].data(),
                              start_embedding_table[i + rank_start_entry].data(),
                              embedding_dim,
                              i);
      }

      EXPECT_EQ(wholememory_destroy_embedding_cache_policy(cache_policy), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(wholememory_destroy_tensor(indices_tensor), WHOLEMEMORY_SUCCESS);
      EXPECT_EQ(wholememory_destroy_tensor(grad_tensor), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(cudaFreeHost(host_indices), cudaSuccess);
      EXPECT_EQ(cudaFree(dev_indices), cudaSuccess);
      EXPECT_EQ(cudaFree(dev_grad_buffer), cudaSuccess);
      EXPECT_EQ(cudaFreeHost(host_grad_buffer), cudaSuccess);

      EXPECT_EQ(wholememory_destroy_embedding(wm_embedding), WHOLEMEMORY_SUCCESS);
      wholememory_destroy_embedding_optimizer(optimizer);

      EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);
      WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
    },
    true);
}

INSTANTIATE_TEST_SUITE_P(
  CachedEmbeddingGatherBackwardTest,
  WholeMemoryEmbeddingBackwardParameterTests,
  ::testing::Values(
#if 0
        EmbeddingBackwardTestParams(),
        EmbeddingBackwardTestParams().set_optimizer_type(WHOLEMEMORY_OPT_RMSPROP),
        EmbeddingBackwardTestParams().set_optimizer_type(WHOLEMEMORY_OPT_ADAGRAD),
        EmbeddingBackwardTestParams().set_optimizer_type(WHOLEMEMORY_OPT_LAZY_ADAM),
        EmbeddingBackwardTestParams().set_use_cache(),
        EmbeddingBackwardTestParams().set_use_cache().set_optimizer_type(WHOLEMEMORY_OPT_RMSPROP),
        EmbeddingBackwardTestParams().set_use_cache().set_optimizer_type(WHOLEMEMORY_OPT_ADAGRAD),
        EmbeddingBackwardTestParams().set_use_cache().set_optimizer_type(WHOLEMEMORY_OPT_LAZY_ADAM),
        EmbeddingBackwardTestParams().set_run_count(10),
        EmbeddingBackwardTestParams().set_run_count(10).set_optimizer_type(WHOLEMEMORY_OPT_RMSPROP),
        EmbeddingBackwardTestParams().set_run_count(10).set_optimizer_type(WHOLEMEMORY_OPT_ADAGRAD),
        EmbeddingBackwardTestParams().set_run_count(10).set_optimizer_type(WHOLEMEMORY_OPT_LAZY_ADAM),
        EmbeddingBackwardTestParams().set_run_count(10).set_use_cache(),
        EmbeddingBackwardTestParams().set_run_count(10).set_use_cache().set_optimizer_type(WHOLEMEMORY_OPT_RMSPROP),
        EmbeddingBackwardTestParams().set_run_count(10).set_use_cache().set_optimizer_type(WHOLEMEMORY_OPT_ADAGRAD),
        EmbeddingBackwardTestParams().set_run_count(10).set_use_cache().set_optimizer_type(WHOLEMEMORY_OPT_LAZY_ADAM),

        EmbeddingBackwardTestParams().set_use_cache().set_indice_count(10000127),
        EmbeddingBackwardTestParams().set_use_cache().set_indice_count(10000127).set_optimizer_type(WHOLEMEMORY_OPT_RMSPROP),
        EmbeddingBackwardTestParams().set_use_cache().set_indice_count(10000127).set_optimizer_type(WHOLEMEMORY_OPT_ADAGRAD),
        EmbeddingBackwardTestParams().set_use_cache().set_indice_count(10000127).set_optimizer_type(WHOLEMEMORY_OPT_LAZY_ADAM),
#endif
    EmbeddingBackwardTestParams().set_entry_count(500).set_indice_count(400).set_embedding_dim(4),
    EmbeddingBackwardTestParams().set_embedding_dim(3),
    EmbeddingBackwardTestParams().set_use_cache().set_grad_stride(131),
    EmbeddingBackwardTestParams().set_use_cache().set_grad_stride(131).set_optimizer_type(
      WHOLEMEMORY_OPT_RMSPROP),
    EmbeddingBackwardTestParams().set_use_cache().set_grad_stride(131).set_optimizer_type(
      WHOLEMEMORY_OPT_ADAGRAD),
    EmbeddingBackwardTestParams().set_use_cache().set_grad_stride(131).set_optimizer_type(
      WHOLEMEMORY_OPT_LAZY_ADAM),

    EmbeddingBackwardTestParams()
      .set_use_cache()
      .set_memory_type(WHOLEMEMORY_MT_CONTINUOUS)
      .set_optimizer_type(WHOLEMEMORY_OPT_LAZY_ADAM),
    EmbeddingBackwardTestParams()
      .set_use_cache()
      .set_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_cache_memory_type(WHOLEMEMORY_MT_DISTRIBUTED)
      .set_optimizer_type(WHOLEMEMORY_OPT_LAZY_ADAM),

    EmbeddingBackwardTestParams().set_use_cache().set_cache_ratio(0.07).set_optimizer_type(
      WHOLEMEMORY_OPT_LAZY_ADAM),
    EmbeddingBackwardTestParams().set_use_cache().set_cache_ratio(0.53).set_optimizer_type(
      WHOLEMEMORY_OPT_LAZY_ADAM),

    EmbeddingBackwardTestParams(),
    EmbeddingBackwardTestParams().set_indice_dtype(WHOLEMEMORY_DT_INT),
    EmbeddingBackwardTestParams()
      .set_indice_dtype(WHOLEMEMORY_DT_INT)
      .set_optimizer_type(WHOLEMEMORY_OPT_RMSPROP),
    EmbeddingBackwardTestParams()
      .set_indice_dtype(WHOLEMEMORY_DT_INT)
      .set_optimizer_type(WHOLEMEMORY_OPT_ADAGRAD),
    EmbeddingBackwardTestParams()
      .set_indice_dtype(WHOLEMEMORY_DT_INT)
      .set_optimizer_type(WHOLEMEMORY_OPT_LAZY_ADAM),
    EmbeddingBackwardTestParams().set_use_cache().set_indice_dtype(WHOLEMEMORY_DT_INT),
    EmbeddingBackwardTestParams()
      .set_use_cache()
      .set_indice_dtype(WHOLEMEMORY_DT_INT)
      .set_optimizer_type(WHOLEMEMORY_OPT_RMSPROP),
    EmbeddingBackwardTestParams()
      .set_use_cache()
      .set_indice_dtype(WHOLEMEMORY_DT_INT)
      .set_optimizer_type(WHOLEMEMORY_OPT_ADAGRAD),
    EmbeddingBackwardTestParams()
      .set_use_cache()
      .set_indice_dtype(WHOLEMEMORY_DT_INT)
      .set_optimizer_type(WHOLEMEMORY_OPT_LAZY_ADAM),
    EmbeddingBackwardTestParams().set_use_cache().set_embedding_dim(129),
    EmbeddingBackwardTestParams().set_use_cache().set_embedding_dim(129).set_optimizer_type(
      WHOLEMEMORY_OPT_RMSPROP),
    EmbeddingBackwardTestParams().set_use_cache().set_embedding_dim(129).set_optimizer_type(
      WHOLEMEMORY_OPT_ADAGRAD),
    EmbeddingBackwardTestParams().set_use_cache().set_embedding_dim(129).set_optimizer_type(
      WHOLEMEMORY_OPT_LAZY_ADAM),
    EmbeddingBackwardTestParams()
      .set_use_cache()
      .set_embedding_dim(129)
      .set_optimizer_type(WHOLEMEMORY_OPT_LAZY_ADAM)
      .set_run_count(10)
      .set_optimizer_params("beta1", 0.8),
    EmbeddingBackwardTestParams()
      .set_use_cache()
      .set_embedding_dim(129)
      .set_optimizer_type(WHOLEMEMORY_OPT_LAZY_ADAM)
      .set_run_count(10)
      .set_optimizer_params("beta2", 0.9),

    EmbeddingBackwardTestParams().set_use_cache().set_embedding_dim(392),
    EmbeddingBackwardTestParams().set_use_cache().set_embedding_dim(392).set_optimizer_type(
      WHOLEMEMORY_OPT_RMSPROP),
    EmbeddingBackwardTestParams().set_use_cache().set_embedding_dim(392).set_optimizer_type(
      WHOLEMEMORY_OPT_ADAGRAD),
    EmbeddingBackwardTestParams().set_use_cache().set_embedding_dim(392).set_optimizer_type(
      WHOLEMEMORY_OPT_LAZY_ADAM),

    EmbeddingBackwardTestParams()));
