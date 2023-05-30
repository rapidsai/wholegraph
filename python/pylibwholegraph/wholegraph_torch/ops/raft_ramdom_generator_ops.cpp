#include <torch/script.h>

#include <c10/cuda/CUDAStream.h>

#include "test_raft_random.cuh"

namespace wholegraph_torch_test {

torch::Tensor raft_pcg_generator_random(int64_t random_seed,
                                        int64_t subsequence,
                                        int64_t generated_random_number)
{
  auto to = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64).requires_grad(false);
  torch::Tensor output = torch::empty({(long)(generated_random_number)}, to);

  TestPCGenerator rng((unsigned long long)random_seed, subsequence, 0);
  for (int64_t i = 0; i < generated_random_number; i++) {
    uint32_t random_num;
    rng.next(random_num);
    output[i].data_ptr<int64_t>()[0] = (int64_t)random_num;
  }

  return output;
}

torch::Tensor raft_pcg_generator_random_from_weight(int64_t random_seed,
                                                    int64_t subsequence,
                                                    torch::Tensor edge_weight,
                                                    int64_t generated_random_number)
{
  auto to =
    torch::TensorOptions().device(torch::kCPU).dtype(edge_weight.dtype()).requires_grad(false);
  torch::Tensor output = torch::empty({(long)(generated_random_number)}, to);
  TestPCGenerator rng((unsigned long long)random_seed, subsequence, 0);
  for (int64_t i = 0; i < generated_random_number; i++) {
    float u             = -rng.next_float(1.0f, 0.5f);
    int64_t random_num2 = 0;
    int seed_count      = -1;
    do {
      rng.next(random_num2);
      seed_count++;
    } while (!random_num2);
    auto count_one = [](unsigned long long num) {
      int c = 0;
      while (num) {
        num >>= 1;
        c++;
      }
      return 64 - c;
    };
    int one_bit = count_one(random_num2) + seed_count * 64;
    u *= pow(2, -one_bit);
    // float logk = (log1pf(u) / logf(2.0)) * (1.0f / (float)weight);
    if (edge_weight.dtype() == torch::kFloat32) {
      float weight                   = edge_weight[i].data_ptr<float>()[0];
      float logk                     = (1 / weight) * (log1p(u) / log(2.0));
      output[i].data_ptr<float>()[0] = logk;
    } else if (edge_weight.dtype() == torch::kFloat64) {
      double weight                   = edge_weight[i].data_ptr<double>()[0];
      double logk                     = (1 / weight) * (log1p(u) / log(2.0));
      output[i].data_ptr<double>()[0] = logk;
    }
  }

  return output;
}

}  // namespace wholegraph_torch_test

static auto registry = torch::RegisterOperators()
                         .op("wholegraph_test::raft_pcg_generator_random",
                             &wholegraph_torch_test::raft_pcg_generator_random)
                         .op("wholegraph_test::raft_pcg_generator_random_from_weight",
                             &wholegraph_torch_test::raft_pcg_generator_random_from_weight);
