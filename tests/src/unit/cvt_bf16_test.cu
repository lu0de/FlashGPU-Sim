// Instruction-level tests for bf16 / bf16x2 cvt instructions (sm_90a).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

__global__ void cvt_bf16_kernels(uint32_t *out, const float *a, const float *b,
                               int n) {
  int i = threadIdx.x;
  if (i >= n) return;
  float fa = a[i], fb = b[i];
  uint32_t packed_rn = 0, packed_rz = 0;
  asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
               : "=r"(packed_rn) : "f"(fa), "f"(fb));
  asm volatile("cvt.rz.bf16x2.f32 %0, %1, %2;"
               : "=r"(packed_rz) : "f"(fa), "f"(fb));
  unsigned short scalar_bf = 0;
  asm volatile("{ .reg .b16 t; cvt.rn.bf16.f32 t, %1; cvt.u32.u16 %0, t; }"
               : "=r"(*reinterpret_cast<uint32_t *>(&scalar_bf)) : "f"(fa));
  uint32_t lo16 = *reinterpret_cast<uint32_t *>(&scalar_bf) & 0xFFFFu;
  float back = 0.0f;
  asm volatile("{ .reg .b16 t; cvt.u16.u32 t, %1; cvt.f32.bf16 %0, t; }"
               : "=f"(back) : "r"(lo16));
  out[i] = packed_rn;
  out[n + i] = packed_rz;
  out[2 * n + i] = lo16;
  uint32_t back_bits;
  std::memcpy(&back_bits, &back, 4);
  out[3 * n + i] = back_bits;
}

// Host f32 -> bf16 reference: expected values measured on H100 (sm_90a),
// see bf16/cvt_native/native.txt. NaN -> canonical 0x7FFF.
uint16_t host_f32_to_bf16_rn(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if (std::isnan(f)) return 0x7FFFu;
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}
uint16_t host_f32_to_bf16_rz(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if (std::isnan(f)) return 0x7FFFu;
  return static_cast<uint16_t>(u >> 16);
}
float host_bf16_to_f32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

TEST(CvtBf16Test, PackedAndScalar) {
  std::vector<float> cases = {
      1.0f,
      1.00390625f,          // 1 + 2^-8: midpoint below next bf16 step -> tie
      1.01171875f,          // 1 + 3*2^-8: tie at odd mantissa -> round to even
      1.005f,               // above midpoint
      -2.5f,
      -0.0f,
      INFINITY, -INFINITY,
      NAN,
      FLT_MAX,              // RNE -> inf
      1e-40f,               // subnormal
      3.14159265f};
  int n = static_cast<int>(cases.size());
  float *da = nullptr, *db = nullptr;
  uint32_t *dout = nullptr;
  ASSERT_EQ(cudaMalloc(&da, n * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&db, n * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dout, 4 * n * 4), cudaSuccess);
  std::vector<float> a(n), b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = cases[i];
    b[i] = cases[(i + 3) % n];
  }
  ASSERT_EQ(cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(db, b.data(), n * 4, cudaMemcpyHostToDevice),
            cudaSuccess);
  cvt_bf16_kernels<<<1, 64>>>(dout, da, db, n);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint32_t> out(4 * n);
  ASSERT_EQ(cudaMemcpy(out.data(), dout, 4 * n * 4, cudaMemcpyDeviceToHost),
            cudaSuccess);

  for (int i = 0; i < n; ++i) {
    uint32_t exp_rn = (uint32_t(host_f32_to_bf16_rn(a[i])) << 16) |
                      host_f32_to_bf16_rn(b[i]);
    uint32_t exp_rz = (uint32_t(host_f32_to_bf16_rz(a[i])) << 16) |
                      host_f32_to_bf16_rz(b[i]);
    EXPECT_EQ(out[i], exp_rn)
        << "bf16x2 rn case " << i << " a=" << a[i] << " b=" << b[i];
    EXPECT_EQ(out[n + i], exp_rz)
        << "bf16x2 rz case " << i << " a=" << a[i] << " b=" << b[i];
    uint16_t exp_scalar = host_f32_to_bf16_rn(a[i]);
    EXPECT_EQ(out[2 * n + i], uint32_t(exp_scalar))
        << "scalar bf16 rn case " << i;
    uint32_t exp_back;
    float fb = host_bf16_to_f32(exp_scalar);
    std::memcpy(&exp_back, &fb, 4);
    EXPECT_EQ(out[3 * n + i], exp_back) << "f32.bf16 case " << i;
  }
  cudaFree(da); cudaFree(db); cudaFree(dout);
}

}  // namespace
