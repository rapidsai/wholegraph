#pragma once

#include <cstdint>

/** PCG random number generator from raft */
struct TestPCGenerator {
  /**
   * @brief ctor. Initializes the state for RNG. This code is derived from PCG basic code
   * @param seed the seed (can be same across all threads). Same as PCG's initstate
   * @param subsequence is same as PCG's initseq
   * @param offset unused
   */
  __host__ __device__ __forceinline__ TestPCGenerator(uint64_t seed,
                                                      uint64_t subsequence,
                                                      uint64_t offset)
  {
    pcg_state = uint64_t(0);
    inc       = (subsequence << 1u) | 1u;
    uint32_t discard;
    next(discard);
    pcg_state += seed;
    next(discard);
    skipahead(offset);
  }

  // Based on "Random Number Generation with Arbitrary Strides" F. B. Brown
  // Link https://mcnp.lanl.gov/pdf_files/anl-rn-arb-stride.pdf
  __host__ __device__ __forceinline__ void skipahead(uint64_t offset)
  {
    uint64_t G = 1;
    uint64_t h = 6364136223846793005ULL;
    uint64_t C = 0;
    uint64_t f = inc;
    while (offset) {
      if (offset & 1) {
        G = G * h;
        C = C * h + f;
      }
      f = f * (h + 1);
      h = h * h;
      offset >>= 1;
    }
    pcg_state = pcg_state * G + C;
  }

  /**
   * @defgroup NextRand Generate the next random number
   * @brief This code is derived from PCG basic code
   * @{
   */
  __host__ __device__ __forceinline__ uint32_t next_u32()
  {
    uint32_t ret;
    uint64_t oldstate   = pcg_state;
    pcg_state           = oldstate * 6364136223846793005ULL + inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot        = oldstate >> 59u;
    ret                 = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    return ret;
  }
  __host__ __device__ __forceinline__ uint64_t next_u64()
  {
    uint64_t ret;
    uint32_t a, b;
    a   = next_u32();
    b   = next_u32();
    ret = uint64_t(a) | (uint64_t(b) << 32);
    return ret;
  }

  __host__ __device__ __forceinline__ int32_t next_i32()
  {
    int32_t ret;
    uint32_t val;
    val = next_u32();
    ret = int32_t(val & 0x7fffffff);
    return ret;
  }

  __host__ __device__ __forceinline__ int64_t next_i64()
  {
    int64_t ret;
    uint64_t val;
    val = next_u64();
    ret = int64_t(val & 0x7fffffffffffffff);
    return ret;
  }

  __host__ __device__ __forceinline__ float next_float()
  {
    float ret;
    uint32_t val = next_u32() >> 8;
    ret          = static_cast<float>(val) / (1U << 24);
    return ret;
  }

  __host__ __device__ __forceinline__ float next_float(float max, float min)
  {
    float ret;
    uint32_t val = next_u32() >> 8;
    ret          = static_cast<float>(val) / (1U << 24);
    ret *= (max - min);
    ret += min;
    return ret;
  }

  __host__ __device__ __forceinline__ double next_double()
  {
    double ret;
    uint64_t val = next_u64() >> 11;
    ret          = static_cast<double>(val) / (1LU << 53);
    return ret;
  }

  __host__ __device__ __forceinline__ void next(uint32_t& ret) { ret = next_u32(); }
  __host__ __device__ __forceinline__ void next(uint64_t& ret) { ret = next_u64(); }
  __host__ __device__ __forceinline__ void next(int32_t& ret) { ret = next_i32(); }
  __host__ __device__ __forceinline__ void next(int64_t& ret) { ret = next_i64(); }

  __host__ __device__ __forceinline__ void next(float& ret) { ret = next_float(); }
  __host__ __device__ __forceinline__ void next(double& ret) { ret = next_double(); }

  /** @} */

 private:
  uint64_t pcg_state;
  uint64_t inc;
};
