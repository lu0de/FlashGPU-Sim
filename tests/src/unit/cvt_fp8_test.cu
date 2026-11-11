// Instruction-level tests for fp8 packed cvt instructions (sm_90a).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

__global__ void cvt_fp8_kernels(uint16_t *out, const float *a, const float *b,
                                int n) {
  int i = threadIdx.x;
  if (i >= n) return;
  float fa = a[i], fb = b[i];
  uint16_t e43 = 0, e52 = 0, e43r = 0, e52r = 0;
  asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}"
               : "=h"(e43) : "f"(fa), "f"(fb));
  asm volatile("{cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;}"
               : "=h"(e52) : "f"(fa), "f"(fb));
  asm volatile("{cvt.rn.satfinite.relu.e4m3x2.f32 %0, %2, %1;}"
               : "=h"(e43r) : "f"(fa), "f"(fb));
  asm volatile("{cvt.rn.satfinite.relu.e5m2x2.f32 %0, %2, %1;}"
               : "=h"(e52r) : "f"(fa), "f"(fb));
  out[i] = e43;
  out[n + i] = e52;
  out[2 * n + i] = e43r;
  out[3 * n + i] = e52r;
}

__global__ void cvt_fp8_decode_kernel(uint32_t *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  uint16_t in = static_cast<uint16_t>(i);
  uint32_t e43 = 0, e52 = 0, e43r = 0, e52r = 0;
  asm("{cvt.rn.f16x2.e4m3x2 %0, %1;}" : "=r"(e43) : "h"(in));
  asm("{cvt.rn.f16x2.e5m2x2 %0, %1;}" : "=r"(e52) : "h"(in));
  asm("{cvt.rn.relu.f16x2.e4m3x2 %0, %1;}" : "=r"(e43r) : "h"(in));
  asm("{cvt.rn.relu.f16x2.e5m2x2 %0, %1;}" : "=r"(e52r) : "h"(in));
  out[i] = e43;
  out[65536 + i] = e52;
  out[2 * 65536 + i] = e43r;
  out[3 * 65536 + i] = e52r;
}

// Host references. Expected values were validated against H800 native output
// (see lowp probe REFMIS=0 lines); the encode/decode tables below mirror the
// simulator's bit-exact reference model.
float e4m3_to_f32(uint8_t bits) {
  int sign = (bits >> 7) & 1, exp = (bits >> 3) & 0xF, mant = bits & 0x7;
  float v = 0.0f;
  if (exp == 0) v = mant ? std::ldexp(mant / 8.0f, -6) : 0.0f;
  else if (exp == 0xF && mant == 0x7) v = NAN;
  else v = std::ldexp(1.0f + mant / 8.0f, exp - 7);
  return sign ? -v : v;
}
float e5m2_to_f32(uint8_t bits) {
  int sign = (bits >> 7) & 1, exp = (bits >> 2) & 0x1F, mant = bits & 0x3;
  float v = 0.0f;
  if (exp == 0) v = mant ? std::ldexp(mant / 4.0f, -14) : 0.0f;
  else if (exp == 0x1F) v = mant ? NAN : INFINITY;
  else v = std::ldexp(1.0f + mant / 4.0f, exp - 15);
  return sign ? -v : v;
}
int fp8_round_even(float v) {
  float base_f = std::floor(v);
  int base = (int)base_f;
  float frac = v - base_f;
  if (frac > 0.5f) return base + 1;
  if (frac < 0.5f) return base;
  return (base & 1) ? base + 1 : base;
}
uint8_t f32_to_fp8(float value, bool e4m3) {
  if (std::isnan(value)) return 0x7f;
  uint8_t sign = std::signbit(value) ? 0x80 : 0;
  float m = std::fabs(value);
  float max_norm = e4m3 ? 448.0f : 57344.0f;
  if (m > max_norm || std::isinf(m)) m = max_norm;
  if (m == 0.0f) return sign;
  int mb = e4m3 ? 3 : 2, bias = e4m3 ? 7 : 15, max_exp = e4m3 ? 15 : 30;
  float min_normal = std::ldexp(1.0f, 1 - bias);
  if (m < min_normal) {
    int mant = fp8_round_even(m / std::ldexp(1.0f, 1 - bias - mb));
    if (mant == 0) return sign;
    if (mant >= (1 << mb)) return sign | (1 << mb);
    return sign | mant;
  }
  int exponent = 0;
  float fr = std::frexp(m, &exponent) * 2.0f;
  int ee = exponent - 1 + bias;
  int mant = fp8_round_even((fr - 1.0f) * (1 << mb));
  if (mant == (1 << mb)) { mant = 0; ++ee; }
  if (ee > max_exp || (e4m3 && ee == 15 && mant > 6))
    return sign | (e4m3 ? 0x7e : 0x7b);
  return sign | ((ee << mb) | mant);
}
uint16_t f32_to_f16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  uint32_t sign = (bits >> 31) & 1;
  int32_t exp = (int32_t)((bits >> 23) & 0xFF) - 127;
  uint32_t frac = bits & 0x7FFFFF;
  if (exp == 128) {
    if (frac) return (sign << 15) | 0x7E00;
    return (sign << 15) | 0x7C00;
  }
  if (exp > 15) return (sign << 15) | 0x7C00;
  if (exp < -24) return sign << 15;
  if (exp < -14) {
    uint32_t m = 0x800000 | frac;
    int shift = -14 - exp;
    uint32_t half = m >> shift;
    uint32_t rem = m & ((1u << shift) - 1);
    if (rem > (1u << (shift - 1)) || (rem == (1u << (shift - 1)) && (half & 1)))
      ++half;
    return (sign << 15) | (half >> 13);
  }
  uint32_t half = (uint32_t)(exp + 15) << 10 | (frac >> 13);
  uint32_t rem = frac & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;
  return (sign << 15) | half;
}
uint16_t f16_relu(uint16_t h) {
  bool nan = ((h >> 10) & 0x1F) == 0x1F && (h & 0x3FF);
  if (nan) return h;
  if ((h >> 15) & 1) return 0x0000;
  return h;
}
float fp8_relu_in(float v) {
  return std::signbit(v) && !std::isnan(v) ? 0.0f : v;
}
uint16_t fp8_dec_lane(uint8_t byte, bool e4m3) {
  bool nan = e4m3 ? (byte & 0x7F) == 0x7F : (byte & 0x7F) >= 0x7D;
  if (nan) return 0x7FFF;
  return f32_to_f16(e4m3 ? e4m3_to_f32(byte) : e5m2_to_f32(byte));
}

TEST(CvtFp8Test, EncodePacked) {
  std::vector<float> cases = {
      1.0f, 1.015625f, 1.03125f,  // exact, midpoint-even, midpoint-odd (e4m3)
      448.0f, 464.0f, 480.0f, 896.0f,   // e4m3 max_norm boundary
      57344.0f, 60928.0f, 114688.0f,    // e5m2 max_norm boundary
      0.001953125f,                   // e4m3 min subnormal
      0.0009765625f,                  // e4m3 min subnormal / 2
      0.0000152587890625f,            // e5m2 min subnormal
      1e-40f,
      0.0f, -0.0f, -2.5f,
      INFINITY, -INFINITY, NAN,
      3.14159265f};
  int n = static_cast<int>(cases.size());
  float *da = nullptr, *db = nullptr;
  uint16_t *dout = nullptr;
  ASSERT_EQ(cudaMalloc(&da, n * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&db, n * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&dout, 4 * n * 2), cudaSuccess);
  std::vector<float> a(n), b(n);
  for (int i = 0; i < n; ++i) {
    a[i] = cases[i];
    b[i] = cases[(7 * i + 3) % n];
  }
  ASSERT_EQ(cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(db, b.data(), n * 4, cudaMemcpyHostToDevice),
            cudaSuccess);
  cvt_fp8_kernels<<<1, n>>>(dout, da, db, n);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint16_t> out(4 * n);
  ASSERT_EQ(cudaMemcpy(out.data(), dout, 4 * n * 2, cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int i = 0; i < n; ++i) {
    float fa = a[i], fb = b[i];
    // PTX packs the first source in the high byte.
    uint16_t exp_e43 =
        (uint16_t(f32_to_fp8(fb, true)) << 8) | f32_to_fp8(fa, true);
    uint16_t exp_e52 =
        (uint16_t(f32_to_fp8(fb, false)) << 8) | f32_to_fp8(fa, false);
    uint16_t exp_e43r =
        (uint16_t(f32_to_fp8(fp8_relu_in(fb), true)) << 8) |
        f32_to_fp8(fp8_relu_in(fa), true);
    uint16_t exp_e52r =
        (uint16_t(f32_to_fp8(fp8_relu_in(fb), false)) << 8) |
        f32_to_fp8(fp8_relu_in(fa), false);
    EXPECT_EQ(out[i], exp_e43) << "e4m3x2 case " << i << " a=" << fa;
    EXPECT_EQ(out[n + i], exp_e52) << "e5m2x2 case " << i << " a=" << fa;
    EXPECT_EQ(out[2 * n + i], exp_e43r) << "e4m3x2.relu case " << i;
    EXPECT_EQ(out[3 * n + i], exp_e52r) << "e5m2x2.relu case " << i;
  }
  cudaFree(da);
  cudaFree(db);
  cudaFree(dout);
}

TEST(CvtFp8Test, DecodeAll256) {
  uint32_t *dout = nullptr;
  ASSERT_EQ(cudaMalloc(&dout, (size_t)4 * 65536 * 4), cudaSuccess);
  cvt_fp8_decode_kernel<<<256, 256>>>(dout);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint32_t> out((size_t)4 * 65536);
  ASSERT_EQ(cudaMemcpy(out.data(), dout, (size_t)4 * 65536 * 4,
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int i = 0; i < 65536; ++i) {
    uint8_t lo = i & 0xFF, hi = (i >> 8) & 0xFF;
    uint32_t exp43 = (uint32_t(fp8_dec_lane(hi, true)) << 16) |
                     fp8_dec_lane(lo, true);
    uint32_t exp52 = (uint32_t(fp8_dec_lane(hi, false)) << 16) |
                     fp8_dec_lane(lo, false);
    uint32_t exp43r = (uint32_t(f16_relu(exp43 >> 16)) << 16) |
                      f16_relu(exp43 & 0xFFFF);
    uint32_t exp52r = (uint32_t(f16_relu(exp52 >> 16)) << 16) |
                      f16_relu(exp52 & 0xFFFF);
    EXPECT_EQ(out[i], exp43) << "e4m3x2 input " << i;
    EXPECT_EQ(out[65536 + i], exp52) << "e5m2x2 input " << i;
    EXPECT_EQ(out[2 * 65536 + i], exp43r) << "e4m3x2.relu input " << i;
    EXPECT_EQ(out[3 * 65536 + i], exp52r) << "e5m2x2.relu input " << i;
  }
  cudaFree(dout);
}

}  // namespace
