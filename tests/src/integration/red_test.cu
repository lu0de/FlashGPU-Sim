#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>

namespace {

constexpr int kThreads = 128;

// Each thread adds its own value (tid+1) to a single global counter.
__global__ void red_add_u32(uint32_t *counter) {
  uint32_t v = threadIdx.x + 1;
  asm volatile("red.global.add.u32 [%0], %1;" ::"l"(counter), "r"(v) : "memory");
}

// red.relaxed.gpu scope form on a signed counter that starts negative.
__global__ void red_relaxed_gpu_add_s32(int32_t *counter) {
  int32_t v = static_cast<int32_t>(threadIdx.x) - 32;
  asm volatile(
      "red.relaxed.gpu.global.add.s32 [%0], %1;" ::"l"(counter), "r"(v)
      : "memory");
}

__global__ void red_add_u64(uint64_t *counter) {
  uint64_t v = static_cast<uint64_t>(threadIdx.x) * 2 + 1;
  asm volatile("red.global.add.u64 [%0], %1;" ::"l"(counter), "l"(v) : "memory");
}

// f32 sums use exactly-representable values (powers of two).
__global__ void red_add_f32(float *counter) {
  float v = static_cast<float>(1 << (threadIdx.x % 10));
  asm volatile("red.global.add.f32 [%0], %1;" ::"l"(counter), "f"(v) : "memory");
}

__global__ void red_min_s32(int32_t *slot) {
  int32_t v = static_cast<int32_t>(threadIdx.x) - 100;
  asm volatile("red.global.min.s32 [%0], %1;" ::"l"(slot), "r"(v) : "memory");
}

__global__ void red_max_s32(int32_t *slot) {
  int32_t v = static_cast<int32_t>(threadIdx.x) - 100;
  asm volatile("red.global.max.s32 [%0], %1;" ::"l"(slot), "r"(v) : "memory");
}

__global__ void red_and_b32(uint32_t *slot) {
  uint32_t v = 0xFFFFFFFFu >> (threadIdx.x % 8);
  asm volatile("red.global.and.b32 [%0], %1;" ::"l"(slot), "r"(v) : "memory");
}

__global__ void red_or_b32(uint32_t *slot) {
  uint32_t v = 1u << (threadIdx.x % 16);
  asm volatile("red.global.or.b32 [%0], %1;" ::"l"(slot), "r"(v) : "memory");
}

__global__ void red_xor_b32(uint32_t *slot) {
  uint32_t v = 0xFFu << (threadIdx.x % 8);
  asm volatile("red.global.xor.b32 [%0], %1;" ::"l"(slot), "r"(v) : "memory");
}

// Two distinct addresses: even threads hit slot[0], odd threads hit slot[1].
__global__ void red_two_addresses(uint32_t *slots) {
  uint32_t *a = slots + (threadIdx.x & 1);
  asm volatile("red.global.add.u32 [%0], %1;" ::"l"(a), "r"(threadIdx.x)
               : "memory");
}

// Shared-memory red: all threads in the block add 1 to a shared counter.
__global__ void red_shared_add_u32(uint32_t *out) {
  __shared__ uint32_t shared_counter;
  if (threadIdx.x == 0) shared_counter = 0;
  __syncthreads();
  const uint32_t addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(&shared_counter));
  asm volatile("red.shared.add.u32 [%0], 1;" ::"r"(addr) : "memory");
  __syncthreads();
  if (threadIdx.x == 0) *out = shared_counter;
}

// Plain CUDA atomicAdd with the result unused: nvcc should emit
// red.global.add.f32 (verified in the generated PTX separately).
__global__ void unused_atomic_add(float *counter, float v) { atomicAdd(counter, v); }

template <typename T>
T run_and_readback(void (*launch)(T *), T init) {
  T *dev = nullptr;
  T host = init;
  EXPECT_EQ(cudaMalloc(&dev, sizeof(T)), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(dev, &host, sizeof(T), cudaMemcpyHostToDevice),
            cudaSuccess);
  launch(dev);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_EQ(cudaMemcpy(&host, dev, sizeof(T), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(cudaFree(dev), cudaSuccess);
  return host;
}

TEST(RedIntegrationTest, GlobalAddU32) {
  const uint64_t expected = 5ull + (kThreads * (kThreads + 1)) / 2;
  uint32_t got = run_and_readback<uint32_t>(
      [](uint32_t *p) { red_add_u32<<<1, kThreads>>>(p); }, 5u);
  EXPECT_EQ(got, static_cast<uint32_t>(expected));
}

TEST(RedIntegrationTest, RelaxedGpuAddS32) {
  // sum(tid-32, tid in [0,127]) = 128*(-32) + 127*128/2 = -4096 + 8128 = 4032
  int32_t got = run_and_readback<int32_t>(
      [](int32_t *p) { red_relaxed_gpu_add_s32<<<1, kThreads>>>(p); }, 0);
  EXPECT_EQ(got, 4032);
}

TEST(RedIntegrationTest, GlobalAddU64) {
  // sum(2t+1, t in [0,127]) = 128*128 = 16384
  uint64_t got = run_and_readback<uint64_t>(
      [](uint64_t *p) { red_add_u64<<<1, kThreads>>>(p); }, 0ull);
  EXPECT_EQ(got, 16384ull);
}

TEST(RedIntegrationTest, GlobalAddF32) {
  float sum = 3.0f;
  for (int i = 0; i < kThreads; ++i) sum += static_cast<float>(1 << (i % 10));
  float got = run_and_readback<float>(
      [](float *p) { red_add_f32<<<1, kThreads>>>(p); }, 3.0f);
  EXPECT_EQ(got, sum);
}

TEST(RedIntegrationTest, GlobalMinS32) {
  int32_t got = run_and_readback<int32_t>(
      [](int32_t *p) { red_min_s32<<<1, kThreads>>>(p); }, 0);
  EXPECT_EQ(got, -100);
}

TEST(RedIntegrationTest, GlobalMaxS32) {
  int32_t got = run_and_readback<int32_t>(
      [](int32_t *p) { red_max_s32<<<1, kThreads>>>(p); }, 0);
  EXPECT_EQ(got, 27);
}

TEST(RedIntegrationTest, GlobalAndB32) {
  uint32_t expected = 0xFFFFFFFFu;
  for (int i = 0; i < kThreads; ++i) expected &= (0xFFFFFFFFu >> (i % 8));
  uint32_t got = run_and_readback<uint32_t>(
      [](uint32_t *p) { red_and_b32<<<1, kThreads>>>(p); }, 0xFFFFFFFFu);
  EXPECT_EQ(got, expected);
}

TEST(RedIntegrationTest, GlobalOrB32) {
  uint32_t expected = 0;
  for (int i = 0; i < kThreads; ++i) expected |= (1u << (i % 16));
  uint32_t got = run_and_readback<uint32_t>(
      [](uint32_t *p) { red_or_b32<<<1, kThreads>>>(p); }, 0u);
  EXPECT_EQ(got, expected);
}

TEST(RedIntegrationTest, GlobalXorB32) {
  uint32_t expected = 7;
  for (int i = 0; i < kThreads; ++i) expected ^= (0xFFu << (i % 8));
  uint32_t got = run_and_readback<uint32_t>(
      [](uint32_t *p) { red_xor_b32<<<1, kThreads>>>(p); }, 7u);
  EXPECT_EQ(got, expected);
}

TEST(RedIntegrationTest, TwoAddresses) {
  uint32_t *slots = nullptr;
  uint32_t init[2] = {0, 0};
  ASSERT_EQ(cudaMalloc(&slots, sizeof(init)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(slots, init, sizeof(init), cudaMemcpyHostToDevice),
            cudaSuccess);
  red_two_addresses<<<1, kThreads>>>(slots);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  uint32_t got[2] = {0, 0};
  ASSERT_EQ(cudaMemcpy(got, slots, sizeof(got), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(cudaFree(slots), cudaSuccess);
  uint64_t even = 0, odd = 0;
  for (int i = 0; i < kThreads; ++i) (i & 1 ? odd : even) += i;
  EXPECT_EQ(got[0], static_cast<uint32_t>(even));
  EXPECT_EQ(got[1], static_cast<uint32_t>(odd));
}

TEST(RedIntegrationTest, SharedAddU32) {
  uint32_t got = run_and_readback<uint32_t>(
      [](uint32_t *p) { red_shared_add_u32<<<1, kThreads>>>(p); }, 0xDEADu);
  EXPECT_EQ(got, static_cast<uint32_t>(kThreads));
}

TEST(RedIntegrationTest, UnusedAtomicAddF32) {
  float *dev = nullptr;
  float host = 1.5f;
  ASSERT_EQ(cudaMalloc(&dev, sizeof(host)), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(dev, &host, sizeof(host), cudaMemcpyHostToDevice),
            cudaSuccess);
  unused_atomic_add<<<1, kThreads>>>(dev, 2.5f);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&host, dev, sizeof(host), cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(cudaFree(dev), cudaSuccess);
  EXPECT_FLOAT_EQ(host, 1.5f + kThreads * 2.5f);
}

}  // namespace
