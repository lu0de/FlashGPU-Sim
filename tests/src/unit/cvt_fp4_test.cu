// cvt.rn.satfinite{.relu}.e2m1x2.f32 and cvt.rn{.relu}.f16x2.e2m1x2
// on sm_100a. Expectations verified against B300 (fp4_b300.out):
//   - encode: RNE over {0,.5,1,1.5,2,3,4,6}, satfinite to +-6, NaN -> 0x7,
//     src1 -> high nibble; relu clamps negatives incl -0 -> +0, keeps NaN.
//   - decode: low nibble -> low half; no NaN codes; relu clamps negatives.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

static __device__ __forceinline__ uint32_t enc_e2m1(float a, float b,
                                                  bool relu) {
  uint16_t r = 0;
#if !defined(__CUDA_ARCH_FEAT_SM100_ALL)
  __trap();
#else
  if (relu)
    asm("{ .reg .b8 __$temp1;                                \n"
        " cvt.rn.satfinite.relu.e2m1x2.f32 __$temp1, %2, %1; \n"
        " mov.b16 %0, {__$temp1, 0};                        }\n"
        : "=h"(r)
        : "f"(a), "f"(b));
  else
    asm("{ .reg .b8 __$temp1;                           \n"
        " cvt.rn.satfinite.e2m1x2.f32 __$temp1, %2, %1; \n"
        " mov.b16 %0, {__$temp1, 0};                   }\n"
        : "=h"(r)
        : "f"(a), "f"(b));
#endif
  return r;
}
static __device__ __forceinline__ uint32_t dec_e2m1(uint16_t v, bool relu) {
  uint32_t r = 0;
#if !defined(__CUDA_ARCH_FEAT_SM100_ALL)
  __trap();
#else
  if (relu)
    asm("{ .reg .b8 __$temp1, __$tempz;              \n"
        " mov.b16 {__$temp1, __$tempz}, %1;          \n"
        " cvt.rn.relu.f16x2.e2m1x2 %0, __$temp1;    }\n"
        : "=r"(r)
        : "h"(v));
  else
    asm("{ .reg .b8 __$temp1, __$tempz;              \n"
        " mov.b16 {__$temp1, __$tempz}, %1;          \n"
        " cvt.rn.f16x2.e2m1x2 %0, __$temp1;         }\n"
        : "=r"(r)
        : "h"(v));
#endif
  return r;
}

__global__ void kern_enc(const float *a, const float *b, uint8_t *out,
                         uint8_t *outr, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = (uint8_t)enc_e2m1(a[i], b[i], false);
    outr[i] = (uint8_t)enc_e2m1(a[i], b[i], true);
  }
}
__global__ void kern_dec(const uint8_t *in, uint32_t *out, uint32_t *outr,
                         int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    uint16_t v = in[i];
    out[i] = dec_e2m1(v, false);
    outr[i] = dec_e2m1(v, true);
  }
}

// --- host reference (verified bit-exact against B300 fp4_b300.out) ---
static float bits_f32(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint8_t f32_to_e2m1_h(float value) {
  if (std::isnan(value)) return 0x07;
  uint8_t sign = std::signbit(value) ? 0x08 : 0;
  float m = std::fabs(value);
  if (m > 6.0f || std::isinf(m)) m = 6.0f;
  static const float vals[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  int best = 0, tiec = 0;
  float bd = INFINITY;
  bool tie = false;
  for (int c = 0; c < 8; ++c) {
    float d = std::fabs(m - vals[c]);
    if (d < bd) { bd = d; best = c; tie = false; }
    else if (d == bd) { tiec = c; tie = true; }
  }
  if (tie)
    best = (std::min(best, tiec) & 1) ? std::max(best, tiec)
                                    : std::min(best, tiec);
  return sign | (uint8_t)best;
}
static float e2m1_f32(uint8_t n) {
  static const float lut[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float v = lut[n & 7];
  return (n & 8) ? -v : v;
}
static uint16_t f32_f16(float f) {
  uint32_t u; memcpy(&u, &f, 4);
  uint32_t sign = (u >> 31) & 1;
  int32_t e = (int32_t)((u >> 23) & 0xFF) - 127;
  uint32_t frac = u & 0x7FFFFF;
  if (e == 128) return (sign << 15) | (frac ? 0x7E00 : 0x7C00);
  if (e > 15) return (sign << 15) | 0x7C00;
  if (e < -24) return sign << 15;
  if (e < -14) {
    uint32_t m = 0x800000 | frac;
    int sh = -14 - e;
    uint32_t h = m >> sh, r = m & ((1u << sh) - 1);
    if (r > (1u << (sh - 1)) || (r == (1u << (sh - 1)) && (h & 1))) ++h;
    return (sign << 15) | (h >> 13);
  }
  uint32_t h = (uint32_t)(e + 15) << 10 | (frac >> 13);
  uint32_t r = frac & 0x1FFF;
  if (r > 0x1000 || (r == 0x1000 && (h & 1))) ++h;
  return (sign << 15) | h;
}
static uint16_t dec_lane(uint8_t n, bool relu) {
  float f = e2m1_f32(n);
  if (relu && std::signbit(f)) f = 0.0f;
  return f32_f16(f);
}

TEST(CvtFp4, Encode) {
  const float vals[] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  std::vector<float> L;
  for (float v : vals) { L.push_back(v); L.push_back(-v); }
  static const float mids[] = {.25f, .75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f};
  for (float v : mids) { L.push_back(v); L.push_back(-v); }
  L.insert(L.end(), {6.5f, 7.f, 1e30f, INFINITY, -INFINITY, NAN, -NAN,
                     0.f, -0.f, 1e-30f, -1e-30f, .249f, .251f, 4.99f, 5.01f});
  int n = L.size();
  std::vector<float> A(n), B(n);
  for (int i = 0; i < n; ++i) { A[i] = L[i]; B[i] = L[(7 * i + 3) % n]; }
  float *da, *db; uint8_t *d0, *d1;
  ASSERT_EQ(cudaMalloc(&da, n * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&db, n * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d0, n), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d1, n), cudaSuccess);
  cudaMemcpy(da, A.data(), n * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(db, B.data(), n * 4, cudaMemcpyHostToDevice);
  kern_enc<<<(n + 127) / 128, 128>>>(da, db, d0, d1, n);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint8_t> got(n), gotr(n);
  cudaMemcpy(got.data(), d0, n, cudaMemcpyDeviceToHost);
  cudaMemcpy(gotr.data(), d1, n, cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; ++i) {
    // asm operands are cvt d, b, a: b lands in the high nibble.
    uint8_t exp =
        (uint8_t)(f32_to_e2m1_h(B[i]) << 4) | f32_to_e2m1_h(A[i]);
    float ra = std::signbit(A[i]) && !std::isnan(A[i]) ? 0.0f : A[i];
    float rb = std::signbit(B[i]) && !std::isnan(B[i]) ? 0.0f : B[i];
    uint8_t expr =
        (uint8_t)(f32_to_e2m1_h(rb) << 4) | f32_to_e2m1_h(ra);
    EXPECT_EQ(got[i], exp) << "idx " << i;
    EXPECT_EQ(gotr[i], expr) << "idx " << i;
  }
  cudaFree(da); cudaFree(db); cudaFree(d0); cudaFree(d1);
}

TEST(CvtFp4, Decode) {
  const int n = 256;
  std::vector<uint8_t> in(n);
  for (int i = 0; i < n; ++i) in[i] = i;
  uint8_t *din; uint32_t *d0, *d1;
  ASSERT_EQ(cudaMalloc(&din, n), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d0, n * 4), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&d1, n * 4), cudaSuccess);
  cudaMemcpy(din, in.data(), n, cudaMemcpyHostToDevice);
  kern_dec<<<2, 128>>>(din, d0, d1, n);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint32_t> got(n), gotr(n);
  cudaMemcpy(got.data(), d0, n * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(gotr.data(), d1, n * 4, cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; ++i) {
    uint8_t lo = i & 0xF, hi = (i >> 4) & 0xF;
    EXPECT_EQ(got[i], (uint32_t)dec_lane(hi, false) << 16 | dec_lane(lo, false))
        << "idx " << i;
    EXPECT_EQ(gotr[i],
              (uint32_t)dec_lane(hi, true) << 16 | dec_lane(lo, true))
        << "idx " << i;
  }
  cudaFree(din); cudaFree(d0); cudaFree(d1);
}
