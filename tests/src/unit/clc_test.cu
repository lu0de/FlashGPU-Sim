// clusterlaunchcontrol minimal semantics on sm_100a.
// (a) 1 CTA: try_cancel completes 16 tx bytes, response is all zero,
//     is_canceled == false, get_first_ctaid zeros. Both plain and
//     .multicast::cluster::all (implicit 1-CTA cluster).
// (b) persistent-scheduler shape: several CTAs, each loops
//     try_cancel -> wait -> if (!is_canceled) break; sim always reports
//     "not canceled", so every CTA exits after one iteration (no deadlock).
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>

__device__ __forceinline__ void clc_round(uint32_t resp_addr,
                                          uint32_t mbar_addr, int multicast,
                                          uint32_t &done,
                                          uint32_t &is_canceled) {
#if !defined(__CUDA_ARCH_FEAT_SM100_ALL)
  __trap();
#else
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 16;" ::"r"(
      mbar_addr));
  if (multicast) {
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_"
        "tx::bytes.multicast::cluster::all.b128 [%0], [%1];" ::"r"(resp_addr),
        "r"(mbar_addr));
  } else {
    asm volatile(
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_"
        "tx::bytes.b128 [%0], [%1];" ::"r"(resp_addr),
        "r"(mbar_addr));
  }
  asm volatile(
      "{\n.reg .pred p;\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n"
      "selp.u32 %0, 1, 0, p;\n}"
      : "=r"(done)
      : "r"(mbar_addr));
  asm volatile(
      "{\n"
      ".reg .pred p1;\n"
      ".reg .b128 clc_result;\n"
      "ld.shared.b128 clc_result, [%1];\n"
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
      "clc_result;\n"
      "selp.u32 %0, 1, 0, p1;\n}"
      : "=r"(is_canceled)
      : "r"(resp_addr));
#endif
}

__global__ void clc_single_kernel(uint32_t *out, int multicast) {
  if (threadIdx.x != 0) return;
  __shared__ __align__(8) uint64_t mbar;
  __shared__ __align__(16) uint32_t resp[4];
  uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar);
  uint32_t resp_addr = (uint32_t)__cvta_generic_to_shared(resp);
  for (int i = 0; i < 4; ++i) resp[i] = 0xDEADBEEFu;
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mbar_addr));
  uint32_t done, is_canceled;
  clc_round(resp_addr, mbar_addr, multicast, done, is_canceled);
  uint32_t gx = 1, gy = 1, gz = 1, gxs = 1;
#if defined(__CUDA_ARCH_FEAT_SM100_ALL)
  asm volatile(
      "{\n"
      ".reg .pred p1;\n"
      ".reg .b128 clc_result;\n"
      "ld.shared.b128 clc_result, [%4];\n"
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, "
      "clc_result;\n"
      "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 "
      "{%0, %1, %2, _}, clc_result;\n"
      "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %3, "
      "clc_result;\n}"
      : "+r"(gx), "+r"(gy), "+r"(gz), "+r"(gxs)
      : "r"(resp_addr));
#else
  __trap();
#endif
  out[0] = done;
  out[1] = is_canceled;
  out[2] = gx;
  out[3] = gy;
  out[4] = gz;
  out[5] = gxs;
  out[6] = resp[0];
  out[7] = resp[1];
  out[8] = resp[2];
  out[9] = resp[3];
}

// Persistent shape: each CTA loops try_cancel until it is told to stop.
// Sim always reports "not canceled", so every CTA finishes one iteration
// and stamps its marker.
__global__ void clc_persistent_kernel(uint32_t *out) {
  if (threadIdx.x != 0) return;
  __shared__ __align__(8) uint64_t mbar;
  __shared__ __align__(16) uint32_t resp[4];
  uint32_t mbar_addr = (uint32_t)__cvta_generic_to_shared(&mbar);
  uint32_t resp_addr = (uint32_t)__cvta_generic_to_shared(resp);
  for (int i = 0; i < 4; ++i) resp[i] = 0;
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mbar_addr));
  int iters = 0;
  uint32_t done = 0, is_canceled = 1;
  while (iters < 8) {
    ++iters;
    clc_round(resp_addr, mbar_addr, 0, done, is_canceled);
    if (!is_canceled) break;
  }
  out[blockIdx.x * 4 + 0] = 1;           // marker: this CTA ran
  out[blockIdx.x * 4 + 1] = iters;       // loop iterations taken
  out[blockIdx.x * 4 + 2] = done;        // mbarrier completed
  out[blockIdx.x * 4 + 3] = is_canceled; // must be 0 (false)
}

TEST(Clc, SingleCta) {
  uint32_t *dout;
  ASSERT_EQ(cudaMalloc(&dout, 10 * 4), cudaSuccess);
  for (int mc = 0; mc < 2; ++mc) {
    ASSERT_EQ(cudaMemset(dout, 0xEE, 10 * 4), cudaSuccess);
    clc_single_kernel<<<1, 32>>>(dout, mc);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    uint32_t h[10];
    cudaMemcpy(h, dout, 10 * 4, cudaMemcpyDeviceToHost);
    EXPECT_EQ(h[0], 1u) << "mc=" << mc << " mbarrier not completed";
    EXPECT_EQ(h[1], 0u) << "mc=" << mc << " is_canceled should be false";
    EXPECT_EQ(h[2], 0u);
    EXPECT_EQ(h[3], 0u);
    EXPECT_EQ(h[4], 0u);
    EXPECT_EQ(h[5], 0u);
    for (int i = 6; i < 10; ++i)
      EXPECT_EQ(h[i], 0u) << "mc=" << mc << " resp[" << (i - 6) << "]";
  }
  cudaFree(dout);
}

TEST(Clc, PersistentMultiCta) {
  constexpr int kCtas = 4;
  uint32_t *dout;
  ASSERT_EQ(cudaMalloc(&dout, kCtas * 4 * 4), cudaSuccess);
  ASSERT_EQ(cudaMemset(dout, 0, kCtas * 4 * 4), cudaSuccess);
  clc_persistent_kernel<<<kCtas, 32>>>(dout);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint32_t> h(kCtas * 4);
  cudaMemcpy(h.data(), dout, h.size() * 4, cudaMemcpyDeviceToHost);
  for (int c = 0; c < kCtas; ++c) {
    EXPECT_EQ(h[c * 4 + 0], 1u) << "CTA " << c << " never ran (deadlock?)";
    EXPECT_EQ(h[c * 4 + 1], 1u) << "CTA " << c << " did not exit after 1 iter";
    EXPECT_EQ(h[c * 4 + 2], 1u) << "CTA " << c << " mbarrier not completed";
    EXPECT_EQ(h[c * 4 + 3], 0u) << "CTA " << c << " is_canceled should be 0";
  }
  cudaFree(dout);
}
