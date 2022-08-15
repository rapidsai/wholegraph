/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include "whole_graph_optimizers.h"

#include "data_type.h"
#include "whole_chunked_memory.cuh"

namespace whole_graph {

void GetStateSizes(size_t *state_0_size,
                   size_t *state_1_size,
                   size_t *per_slot_size,
                   OptimizerType opt_type,
                   WMType state_type) {
  switch (opt_type) {
    case OPT_TYPE_SGD: {
      *state_0_size = *state_1_size = *per_slot_size = 0;
      break;
    }
    case OPT_TYPE_LAZY_ADAM: {
      *state_0_size = *state_1_size = GetWMTSize(state_type);
      *per_slot_size = sizeof(LazyAdamData);
      break;
    }
    case OPT_TYPE_RMSPROP: {
      *state_0_size = GetWMTSize(state_type);
      *state_1_size = *per_slot_size = 0;
    }
    case OPT_TYPE_ADAGRAD: {
      *state_0_size = GetWMTSize(state_type);
      *state_1_size = *per_slot_size = 0;
    }
    default: {
      abort();
    }
  }
}

template<typename EmbType, typename GradType, typename StateType, typename EmbHandle, typename GradHandle, typename StateHandle>
struct SGDEmbOptimizer {
  __device__ __forceinline__ SGDEmbOptimizer() = default;
  __device__ __forceinline__ void run(OptimizerInfo opt_info,
                                      int64_t embedding_entry_id,
                                      int64_t gradient_entry_id,
                                      EmbHandle *embedding,
                                      GradHandle *grad,
                                      StateHandle *per_element_state_0,
                                      StateHandle *per_element_state_1,
                                      StateHandle *per_embedding_state,
                                      size_t storage_offset,
                                      int embedding_dim,
                                      int embedding_stride,
                                      int grad_per_element_state_stride) {
    PtrGen<EmbHandle, EmbType> emb_ptr_gen(embedding, storage_offset);
    PtrGen<GradHandle, GradType> grad_ptr_gen(grad);
    for (int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
      float grad = (float) *grad_ptr_gen.At(gradient_entry_id * grad_per_element_state_stride + i);
      float param = (float) *emb_ptr_gen.At(embedding_entry_id * embedding_stride + i);
      grad = grad + opt_info.private_info.sgd_info.weight_decay * param;
      param = param - opt_info.lr * grad;
      *emb_ptr_gen.At(embedding_entry_id * embedding_stride + i) = (EmbType) param;
      *grad_ptr_gen.At(gradient_entry_id * grad_per_element_state_stride + i) = (GradType) 0;
    }
  }
};

template<typename EmbType, typename GradType, typename StateType, typename EmbHandle, typename GradHandle, typename StateHandle>
struct LazyAdamEmbOptimizer {
  __device__ __forceinline__ LazyAdamEmbOptimizer() = default;
  __device__ __forceinline__ void run(OptimizerInfo opt_info,
                                      int64_t embedding_entry_id,
                                      int64_t gradient_entry_id,
                                      EmbHandle *embedding,
                                      GradHandle *grad,
                                      StateHandle *per_element_state_0,
                                      StateHandle *per_element_state_1,
                                      StateHandle *per_embedding_state,
                                      size_t storage_offset,
                                      int embedding_dim,
                                      int embedding_stride,
                                      int grad_per_element_state_stride) {
    PtrGen<EmbHandle, EmbType> emb_ptr_gen(embedding, storage_offset);
    PtrGen<GradHandle, GradType> grad_ptr_gen(grad);
    PtrGen<StateHandle, StateType> m_ptr_gen(per_element_state_0);
    PtrGen<StateHandle, StateType> v_ptr_gen(per_element_state_1);
    PtrGen<StateHandle, LazyAdamData> adam_data(per_embedding_state);
    LazyAdamData lazy_adam_data = *adam_data.At(embedding_entry_id);
    float beta1 = opt_info.private_info.lazy_adam_info.beta1;
    float beta2 = opt_info.private_info.lazy_adam_info.beta2;
    lazy_adam_data.beta1t *= beta1;
    lazy_adam_data.beta2t *= beta2;
    for (int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
      float grad = (float) *grad_ptr_gen.At(gradient_entry_id * grad_per_element_state_stride + i);
      float param = (float) *emb_ptr_gen.At(embedding_entry_id * embedding_stride + i);
      float m = (float) *m_ptr_gen.At(embedding_entry_id * grad_per_element_state_stride + i);
      float v = (float) *v_ptr_gen.At(embedding_entry_id * grad_per_element_state_stride + i);
      grad = grad + opt_info.private_info.lazy_adam_info.weight_decay * param;
      m = beta1 * m + (1 - beta1) * grad;
      v = beta2 * v + (1 - beta2) * grad * grad;
      float mhat = m / (1 - lazy_adam_data.beta1t);
      float vhat = v / (1 - lazy_adam_data.beta2t);
      param = param - opt_info.lr * mhat / (sqrtf(vhat) + opt_info.private_info.lazy_adam_info.epsilon);
      *emb_ptr_gen.At(embedding_entry_id * embedding_stride + i) = (EmbType) param;
      *grad_ptr_gen.At(gradient_entry_id * grad_per_element_state_stride + i) = (GradType) 0;
      *m_ptr_gen.At(embedding_entry_id * grad_per_element_state_stride + i) = (StateType) m;
      *v_ptr_gen.At(embedding_entry_id * grad_per_element_state_stride + i) = (StateType) v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      *adam_data.At(embedding_entry_id) = lazy_adam_data;
    }
  }
};

template<typename EmbType, typename GradType, typename StateType, typename EmbHandle, typename GradHandle, typename StateHandle>
struct RMSPropEmbOptimizer {
  __device__ __forceinline__ RMSPropEmbOptimizer() = default;
  __device__ __forceinline__ void run(OptimizerInfo opt_info,
                                      int64_t embedding_entry_id,
                                      int64_t gradient_entry_id,
                                      EmbHandle *embedding,
                                      GradHandle *grad,
                                      StateHandle *per_element_state_0,
                                      StateHandle *per_element_state_1,
                                      StateHandle *per_embedding_state,
                                      size_t storage_offset,
                                      int embedding_dim,
                                      int embedding_stride,
                                      int grad_per_element_state_stride) {
    PtrGen<EmbHandle, EmbType> emb_ptr_gen(embedding, storage_offset);
    PtrGen<GradHandle, GradType> grad_ptr_gen(grad);
    PtrGen<StateHandle, StateType> v_ptr_gen(per_element_state_0);
    float alpha = opt_info.private_info.rms_prop_info.alpha;
    for (int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
      float grad = (float) *grad_ptr_gen.At(gradient_entry_id * grad_per_element_state_stride + i);
      float param = (float) *emb_ptr_gen.At(embedding_entry_id * embedding_stride + i);
      float v = (float) *v_ptr_gen.At(embedding_entry_id * grad_per_element_state_stride + i);
      grad = grad + opt_info.private_info.rms_prop_info.weight_decay * param;
      v = alpha * v + (1 - alpha) * grad * grad;
      float vhat = v;
      param = param - opt_info.lr * grad / (sqrtf(vhat) + opt_info.private_info.rms_prop_info.epsilon);
      *emb_ptr_gen.At(embedding_entry_id * embedding_stride + i) = (EmbType) param;
      *grad_ptr_gen.At(gradient_entry_id * grad_per_element_state_stride + i) = (GradType) 0;
      *v_ptr_gen.At(embedding_entry_id * grad_per_element_state_stride + i) = (StateType) v;
    }
  }
};

template<typename EmbType, typename GradType, typename StateType, typename EmbHandle, typename GradHandle, typename StateHandle>
struct AdaGradEmbOptimizer {
  __device__ __forceinline__ AdaGradEmbOptimizer() = default;
  __device__ __forceinline__ void run(OptimizerInfo opt_info,
                                      int64_t embedding_entry_id,
                                      int64_t gradient_entry_id,
                                      EmbHandle *embedding,
                                      GradHandle *grad,
                                      StateHandle *per_element_state_0,
                                      StateHandle *per_element_state_1,
                                      StateHandle *per_embedding_state,
                                      size_t storage_offset,
                                      int embedding_dim,
                                      int embedding_stride,
                                      int grad_per_element_state_stride) {
    PtrGen<EmbHandle, EmbType> emb_ptr_gen(embedding, storage_offset);
    PtrGen<GradHandle, GradType> grad_ptr_gen(grad);
    PtrGen<StateHandle, StateType> state_sum_ptr_gen(per_element_state_0);
    for (int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
      float grad = (float) *grad_ptr_gen.At(gradient_entry_id * grad_per_element_state_stride + i);
      float param = (float) *emb_ptr_gen.At(embedding_entry_id * embedding_stride + i);
      float state_sum = (float) *state_sum_ptr_gen.At(embedding_entry_id * grad_per_element_state_stride + i);
      grad = grad + opt_info.private_info.ada_grad_info.weight_decay * param;
      state_sum = state_sum + grad * grad;
      param = param - opt_info.lr * grad / (sqrtf(state_sum) + opt_info.private_info.ada_grad_info.epsilon);
      *emb_ptr_gen.At(embedding_entry_id * embedding_stride + i) = (EmbType) param;
      *grad_ptr_gen.At(gradient_entry_id * grad_per_element_state_stride + i) = (GradType) 0;
      *state_sum_ptr_gen.At(embedding_entry_id * grad_per_element_state_stride + i) = (StateType) state_sum;
    }
  }
};

template<typename EmbGradType, typename StateType>
struct SGDSparseEmbOptimizer {
  __device__ __forceinline__ SGDSparseEmbOptimizer() = default;
  __device__ __forceinline__ void run(OptimizerInfo opt_info,
                                      int64_t embedding_entry_id,
                                      int64_t gradient_entry_id,
                                      EmbGradType *embedding,
                                      const EmbGradType *grad_ptr,
                                      StateType *per_element_state_0,
                                      StateType *per_element_state_1,
                                      StateType *per_embedding_state,
                                      int embedding_dim,
                                      int embedding_stride,
                                      int grad_per_element_state_stride) {
    for (int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
      float grad = (float) grad_ptr[gradient_entry_id * grad_per_element_state_stride + i];
      float param = (float) embedding[embedding_entry_id * embedding_stride + i];
      grad = grad + opt_info.private_info.sgd_info.weight_decay * param;
      param = param - opt_info.lr * grad;
      embedding[embedding_entry_id * embedding_stride + i] = (EmbGradType) param;
    }
  }
};

template<typename EmbGradType, typename StateType>
struct LazyAdamSparseEmbOptimizer {
  __device__ __forceinline__ LazyAdamSparseEmbOptimizer() = default;
  __device__ __forceinline__ void run(OptimizerInfo opt_info,
                                      int64_t embedding_entry_id,
                                      int64_t gradient_entry_id,
                                      EmbGradType *embedding,
                                      const EmbGradType *grad_ptr,
                                      StateType *per_element_state_0,
                                      StateType *per_element_state_1,
                                      StateType *per_embedding_state,
                                      int embedding_dim,
                                      int embedding_stride,
                                      int grad_per_element_state_stride) {
    auto *m_ptr = (StateType *) per_element_state_0;
    auto *v_ptr = (StateType *) per_element_state_1;
    auto *adam_data = (LazyAdamData *) per_embedding_state;
    LazyAdamData lazy_adam_data = adam_data[embedding_entry_id];
    float beta1 = opt_info.private_info.lazy_adam_info.beta1;
    float beta2 = opt_info.private_info.lazy_adam_info.beta2;
    lazy_adam_data.beta1t *= beta1;
    lazy_adam_data.beta2t *= beta2;
    for (int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
      float grad = (float) grad_ptr[gradient_entry_id * grad_per_element_state_stride + i];
      float param = (float) embedding[embedding_entry_id * embedding_stride + i];
      float m = (float) m_ptr[embedding_entry_id * grad_per_element_state_stride + i];
      float v = (float) v_ptr[embedding_entry_id * grad_per_element_state_stride + i];
      grad = grad + opt_info.private_info.lazy_adam_info.weight_decay * param;
      m = beta1 * m + (1 - beta1) * grad;
      v = beta2 * v + (1 - beta2) * grad * grad;
      float mhat = m / (1 - lazy_adam_data.beta1t);
      float vhat = v / (1 - lazy_adam_data.beta2t);
      param = param - opt_info.lr * mhat / (sqrtf(vhat) + opt_info.private_info.lazy_adam_info.epsilon);
      embedding[embedding_entry_id * embedding_stride + i] = (EmbGradType) param;
      m_ptr[embedding_entry_id * grad_per_element_state_stride + i] = (StateType) m;
      v_ptr[embedding_entry_id * grad_per_element_state_stride + i] = (StateType) v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      adam_data[embedding_entry_id] = lazy_adam_data;
    }
  }
};

template<typename EmbGradType, typename StateType>
struct RMSPropSparseEmbOptimizer {
  __device__ __forceinline__ RMSPropSparseEmbOptimizer() = default;
  __device__ __forceinline__ void run(OptimizerInfo opt_info,
                                      int64_t embedding_entry_id,
                                      int64_t gradient_entry_id,
                                      EmbGradType *embedding,
                                      const EmbGradType *grad_ptr,
                                      StateType *per_element_state_0,
                                      StateType *per_element_state_1,
                                      StateType *per_embedding_state,
                                      int embedding_dim,
                                      int embedding_stride,
                                      int grad_per_element_state_stride) {
    StateType *v_ptr = per_element_state_0;
    float alpha = opt_info.private_info.rms_prop_info.alpha;
    for (int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
      float grad = (float) grad_ptr[gradient_entry_id * grad_per_element_state_stride + i];
      float param = (float) embedding[embedding_entry_id * embedding_stride + i];
      float v = (float) v_ptr[embedding_entry_id * grad_per_element_state_stride + i];
      grad = grad + opt_info.private_info.rms_prop_info.weight_decay * param;
      v = alpha * v + (1 - alpha) * grad * grad;
      float vhat = v;
      param = param - opt_info.lr * grad / (sqrtf(vhat) + opt_info.private_info.rms_prop_info.epsilon);
      embedding[embedding_entry_id * embedding_stride + i] = (EmbGradType) param;
      v_ptr[embedding_entry_id * grad_per_element_state_stride + i] = (StateType) v;
    }
  }
};

template<typename EmbGradType, typename StateType>
struct AdaGradSparseEmbOptimizer {
  __device__ __forceinline__ AdaGradSparseEmbOptimizer() = default;
  __device__ __forceinline__ void run(OptimizerInfo opt_info,
                                      int64_t embedding_entry_id,
                                      int64_t gradient_entry_id,
                                      EmbGradType *embedding,
                                      const EmbGradType *grad_ptr,
                                      StateType *per_element_state_0,
                                      StateType *per_element_state_1,
                                      StateType *per_embedding_state,
                                      int embedding_dim,
                                      int embedding_stride,
                                      int grad_per_element_state_stride) {
    StateType *state_sum_ptr = per_element_state_0;
    for (int i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
      float grad = (float) grad_ptr[gradient_entry_id * grad_per_element_state_stride + i];
      float param = (float) embedding[embedding_entry_id * embedding_stride + i];
      float state_sum = (float) state_sum_ptr[embedding_entry_id * grad_per_element_state_stride + i];
      grad = grad + opt_info.private_info.ada_grad_info.weight_decay * param;
      state_sum = state_sum + grad * grad;
      param = param - opt_info.lr * grad / (sqrtf(state_sum) + opt_info.private_info.ada_grad_info.epsilon);
      embedding[embedding_entry_id * embedding_stride + i] = (EmbGradType) param;
      state_sum_ptr[embedding_entry_id * grad_per_element_state_stride + i] = (StateType) state_sum;
    }
  }
};

}// namespace whole_graph