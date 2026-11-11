// CUDA integration tests for Hopper BF16 WGMMA.

#include <gtest/gtest.h>

#include "tensor_wgmma_test.cuh"

namespace wgmma_test {

class WgmmaBf16M64N8K16IntegrationTest : public ::testing::Test {};

TEST_F(WgmmaBf16M64N8K16IntegrationTest, RandomValuesTest) {
  run_random_wgmma_case<WgmmaKind::BF16>(
      /*seed=*/2026, /*scale_d=*/1, /*min_value=*/-2.0f,
      /*max_value=*/2.0f, /*absolute_tolerance=*/0.15f,
      /*relative_tolerance=*/0.03f);
}

TEST_F(WgmmaBf16M64N8K16IntegrationTest, RandomValuesScaleDZeroTest) {
  run_random_wgmma_case<WgmmaKind::BF16>(
      /*seed=*/2027, /*scale_d=*/0, /*min_value=*/-2.0f,
      /*max_value=*/2.0f, /*absolute_tolerance=*/0.15f,
      /*relative_tolerance=*/0.03f);
}

// ---- m64nXk16 ss (smem descriptor) coverage: N=16/64/128/256, A from smem.
// K-major A; B K-major, plus MN-major B at N=64. Mirrors the f16 FA3 harness.

__host__ __device__ __forceinline__ uint32_t
bf16_wgmma_apply_swizzle(uint32_t byte_offset) {
  uint32_t y_mask = 0x7u << 7;
  return byte_offset ^ ((byte_offset & y_mask) >> 3);
}
__host__ __device__ __forceinline__ int bf16_wgmma_k_major_smem_index(int row,
                                                                     int k) {
  uint32_t row_group = static_cast<uint32_t>(row / 8);
  uint32_t row_in_group = static_cast<uint32_t>(row % 8);
  uint32_t k_group = static_cast<uint32_t>(k / 8);
  uint32_t k_in_group = static_cast<uint32_t>(k % 8);
  uint32_t byte_offset = row_group * 1024 + row_in_group * 128 +
                         k_group * 16 + k_in_group * 2;
  return static_cast<int>(bf16_wgmma_apply_swizzle(byte_offset) / 2);
}
__host__ __device__ __forceinline__ int bf16_wgmma_mn_major_smem_index(
    int row, int k) {
  uint32_t row_group = static_cast<uint32_t>(row / 64);
  uint32_t row_in_group = static_cast<uint32_t>(row % 64);
  uint32_t k_group = static_cast<uint32_t>(k / 8);
  uint32_t k_in_group = static_cast<uint32_t>(k % 8);
  uint32_t byte_offset = row_group * 16 + row_in_group * 2 +
                         k_group * 1024 + k_in_group * 128;
  return static_cast<int>(bf16_wgmma_apply_swizzle(byte_offset) / 2);
}
__device__ __forceinline__ uint64_t bf16_gmma_swizzle_desc(
    void const *ptr, uint32_t lbo, uint32_t sbo, uint32_t swizzle) {
  uint32_t smem_addr = smem_ptr_to_uint(ptr);
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>((smem_addr >> 4) & 0x3FFF);
  desc |= static_cast<uint64_t>((lbo >> 4) & 0x3FFF) << 16;
  desc |= static_cast<uint64_t>((sbo >> 4) & 0x3FFF) << 32;
  desc |= static_cast<uint64_t>(swizzle & 0x3) << 62;
  return desc;
}

template <int BMajor>
__device__ __forceinline__ void wgmma_m64n16k16_f32_bf16_bf16_ss(
    uint64_t desc_a, uint64_t desc_b, float (&d)[8], int32_t scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, %11, %12, %13, %14;\n"
      "}\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(BMajor));
}

template <int BMajor>
__global__ void wgmma_m64n16k16_bf16_ss_kernel(float *out, const uint16_t *A,
                                              const uint16_t *B, const float *C,
                                              int32_t scale_d) {
  if (threadIdx.x >= kWarpgroupThreads) return;
  __shared__ uint16_t smem_a[4096];
  __shared__ uint16_t smem_b[2048];
  
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) smem_a[i] = 0;
  for (int i = threadIdx.x; i < 2048; i += blockDim.x) smem_b[i] = 0;
  for (int i = threadIdx.x; i < kM * kK; i += blockDim.x) {
    int row = i / kK, k = i % kK;
    smem_a[bf16_wgmma_k_major_smem_index(row, k)] = A[row * kK + k];
  }
  for (int i = threadIdx.x; i < 16 * kK; i += blockDim.x) {
    int col = i / kK, k = i % kK;
    int idx = BMajor == 0 ? bf16_wgmma_k_major_smem_index(col, k)
                          : bf16_wgmma_mn_major_smem_index(col, k);
    smem_b[idx] = B[col * kK + k];
  }
  __syncthreads();
  uint64_t desc_a = bf16_gmma_swizzle_desc(smem_a, 16, 1024, 1);
  uint64_t desc_b = bf16_gmma_swizzle_desc(smem_b, BMajor == 0 ? 16 : 0, 1024, 1);
  float d[8];
  for (int reg = 0; reg < 8; ++reg) {
    int row = 0, cig = 0;
    accumulator_matrix_coord(threadIdx.x, reg % 4, row, cig);
    d[reg] = C[row * 16 + (reg / 4) * 8 + cig];
  }
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  wgmma_m64n16k16_f32_bf16_bf16_ss<BMajor>(desc_a, desc_b, d, scale_d);
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  for (int reg = 0; reg < 8; ++reg) {
    int row = 0, cig = 0;
    accumulator_matrix_coord(threadIdx.x, reg % 4, row, cig);
    out[row * 16 + (reg / 4) * 8 + cig] = d[reg];
  }
}

template <int BMajor>
RunResult run_wgmma_m64n16k16_bf16_ss(const std::vector<uint16_t> &A,
                                     const std::vector<uint16_t> &B,
                                     const std::vector<float> &C, int32_t scale_d) {
  RunResult result;
  result.output.assign(kM * 16, 0.0f);
  if (A.size() != kM * kK || B.size() != 16 * kK || C.size() != kM * 16) {
    result.setup_error = cudaErrorInvalidValue;
    return result;
  }
  uint16_t *d_A = nullptr, *d_B = nullptr;
  float *d_C = nullptr, *d_out = nullptr;
  result.setup_error = cudaMalloc(&d_A, A.size() * 2);
  if (result.setup_error == cudaSuccess) result.setup_error = cudaMalloc(&d_B, B.size() * 2);
  if (result.setup_error == cudaSuccess) result.setup_error = cudaMalloc(&d_C, C.size() * 4);
  if (result.setup_error == cudaSuccess) result.setup_error = cudaMalloc(&d_out, result.output.size() * 4);
  if (result.setup_error != cudaSuccess) return result;
  result.input_copy_error = cudaMemcpy(d_A, A.data(), A.size() * 2, cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess)
    result.input_copy_error = cudaMemcpy(d_B, B.data(), B.size() * 2, cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess)
    result.input_copy_error = cudaMemcpy(d_C, C.data(), C.size() * 4, cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess) {
    wgmma_m64n16k16_bf16_ss_kernel<BMajor><<<1, kWarpgroupThreads>>>(d_out, d_A, d_B, d_C, scale_d);
    result.launch_error = cudaGetLastError();
    if (result.launch_error == cudaSuccess) result.sync_error = cudaDeviceSynchronize();
  }
  if (result.input_copy_error == cudaSuccess && result.launch_error == cudaSuccess &&
      result.sync_error == cudaSuccess)
    result.copy_error = cudaMemcpy(result.output.data(), d_out,
        result.output.size() * 4, cudaMemcpyDeviceToHost);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_out);
  return result;
}

template <int BMajor>
__device__ __forceinline__ void wgmma_m64n64k16_f32_bf16_bf16_ss(
    uint64_t desc_a, uint64_t desc_b, float (&d)[32], int32_t scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, %35, %36, %37, %38;\n"
      "}\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(BMajor));
}

template <int BMajor>
__global__ void wgmma_m64n64k16_bf16_ss_kernel(float *out, const uint16_t *A,
                                              const uint16_t *B, const float *C,
                                              int32_t scale_d) {
  if (threadIdx.x >= kWarpgroupThreads) return;
  __shared__ uint16_t smem_a[4096];
  __shared__ uint16_t smem_b[4096];
  
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) smem_a[i] = 0;
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) smem_b[i] = 0;
  for (int i = threadIdx.x; i < kM * kK; i += blockDim.x) {
    int row = i / kK, k = i % kK;
    smem_a[bf16_wgmma_k_major_smem_index(row, k)] = A[row * kK + k];
  }
  for (int i = threadIdx.x; i < 64 * kK; i += blockDim.x) {
    int col = i / kK, k = i % kK;
    int idx = BMajor == 0 ? bf16_wgmma_k_major_smem_index(col, k)
                          : bf16_wgmma_mn_major_smem_index(col, k);
    smem_b[idx] = B[col * kK + k];
  }
  __syncthreads();
  uint64_t desc_a = bf16_gmma_swizzle_desc(smem_a, 16, 1024, 1);
  uint64_t desc_b = bf16_gmma_swizzle_desc(smem_b, BMajor == 0 ? 16 : 0, 1024, 1);
  float d[32];
  for (int reg = 0; reg < 32; ++reg) {
    int row = 0, cig = 0;
    accumulator_matrix_coord(threadIdx.x, reg % 4, row, cig);
    d[reg] = C[row * 64 + (reg / 4) * 8 + cig];
  }
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  wgmma_m64n64k16_f32_bf16_bf16_ss<BMajor>(desc_a, desc_b, d, scale_d);
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  for (int reg = 0; reg < 32; ++reg) {
    int row = 0, cig = 0;
    accumulator_matrix_coord(threadIdx.x, reg % 4, row, cig);
    out[row * 64 + (reg / 4) * 8 + cig] = d[reg];
  }
}

template <int BMajor>
RunResult run_wgmma_m64n64k16_bf16_ss(const std::vector<uint16_t> &A,
                                     const std::vector<uint16_t> &B,
                                     const std::vector<float> &C, int32_t scale_d) {
  RunResult result;
  result.output.assign(kM * 64, 0.0f);
  if (A.size() != kM * kK || B.size() != 64 * kK || C.size() != kM * 64) {
    result.setup_error = cudaErrorInvalidValue;
    return result;
  }
  uint16_t *d_A = nullptr, *d_B = nullptr;
  float *d_C = nullptr, *d_out = nullptr;
  result.setup_error = cudaMalloc(&d_A, A.size() * 2);
  if (result.setup_error == cudaSuccess) result.setup_error = cudaMalloc(&d_B, B.size() * 2);
  if (result.setup_error == cudaSuccess) result.setup_error = cudaMalloc(&d_C, C.size() * 4);
  if (result.setup_error == cudaSuccess) result.setup_error = cudaMalloc(&d_out, result.output.size() * 4);
  if (result.setup_error != cudaSuccess) return result;
  result.input_copy_error = cudaMemcpy(d_A, A.data(), A.size() * 2, cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess)
    result.input_copy_error = cudaMemcpy(d_B, B.data(), B.size() * 2, cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess)
    result.input_copy_error = cudaMemcpy(d_C, C.data(), C.size() * 4, cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess) {
    wgmma_m64n64k16_bf16_ss_kernel<BMajor><<<1, kWarpgroupThreads>>>(d_out, d_A, d_B, d_C, scale_d);
    result.launch_error = cudaGetLastError();
    if (result.launch_error == cudaSuccess) result.sync_error = cudaDeviceSynchronize();
  }
  if (result.input_copy_error == cudaSuccess && result.launch_error == cudaSuccess &&
      result.sync_error == cudaSuccess)
    result.copy_error = cudaMemcpy(result.output.data(), d_out,
        result.output.size() * 4, cudaMemcpyDeviceToHost);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_out);
  return result;
}

template <int BMajor>
__device__ __forceinline__ void wgmma_m64n128k16_f32_bf16_bf16_ss(
    uint64_t desc_a, uint64_t desc_b, float (&d)[64], int32_t scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, %67, %68, %69, %70;\n"
      "}\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(BMajor));
}

template <int BMajor>
__global__ void wgmma_m64n128k16_bf16_ss_kernel(float *out, const uint16_t *A,
                                              const uint16_t *B, const float *C,
                                              int32_t scale_d) {
  if (threadIdx.x >= kWarpgroupThreads) return;
  __shared__ uint16_t smem_a[4096];
  __shared__ uint16_t smem_b[8192];
  
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) smem_a[i] = 0;
  for (int i = threadIdx.x; i < 8192; i += blockDim.x) smem_b[i] = 0;
  for (int i = threadIdx.x; i < kM * kK; i += blockDim.x) {
    int row = i / kK, k = i % kK;
    smem_a[bf16_wgmma_k_major_smem_index(row, k)] = A[row * kK + k];
  }
  for (int i = threadIdx.x; i < 128 * kK; i += blockDim.x) {
    int col = i / kK, k = i % kK;
    int idx = BMajor == 0 ? bf16_wgmma_k_major_smem_index(col, k)
                          : bf16_wgmma_mn_major_smem_index(col, k);
    smem_b[idx] = B[col * kK + k];
  }
  __syncthreads();
  uint64_t desc_a = bf16_gmma_swizzle_desc(smem_a, 16, 1024, 1);
  uint64_t desc_b = bf16_gmma_swizzle_desc(smem_b, BMajor == 0 ? 16 : 0, 1024, 1);
  float d[64];
  for (int reg = 0; reg < 64; ++reg) {
    int row = 0, cig = 0;
    accumulator_matrix_coord(threadIdx.x, reg % 4, row, cig);
    d[reg] = C[row * 128 + (reg / 4) * 8 + cig];
  }
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  wgmma_m64n128k16_f32_bf16_bf16_ss<BMajor>(desc_a, desc_b, d, scale_d);
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  for (int reg = 0; reg < 64; ++reg) {
    int row = 0, cig = 0;
    accumulator_matrix_coord(threadIdx.x, reg % 4, row, cig);
    out[row * 128 + (reg / 4) * 8 + cig] = d[reg];
  }
}

template <int BMajor>
RunResult run_wgmma_m64n128k16_bf16_ss(const std::vector<uint16_t> &A,
                                     const std::vector<uint16_t> &B,
                                     const std::vector<float> &C, int32_t scale_d) {
  RunResult result;
  result.output.assign(kM * 128, 0.0f);
  if (A.size() != kM * kK || B.size() != 128 * kK || C.size() != kM * 128) {
    result.setup_error = cudaErrorInvalidValue;
    return result;
  }
  uint16_t *d_A = nullptr, *d_B = nullptr;
  float *d_C = nullptr, *d_out = nullptr;
  result.setup_error = cudaMalloc(&d_A, A.size() * 2);
  if (result.setup_error == cudaSuccess) result.setup_error = cudaMalloc(&d_B, B.size() * 2);
  if (result.setup_error == cudaSuccess) result.setup_error = cudaMalloc(&d_C, C.size() * 4);
  if (result.setup_error == cudaSuccess) result.setup_error = cudaMalloc(&d_out, result.output.size() * 4);
  if (result.setup_error != cudaSuccess) return result;
  result.input_copy_error = cudaMemcpy(d_A, A.data(), A.size() * 2, cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess)
    result.input_copy_error = cudaMemcpy(d_B, B.data(), B.size() * 2, cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess)
    result.input_copy_error = cudaMemcpy(d_C, C.data(), C.size() * 4, cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess) {
    wgmma_m64n128k16_bf16_ss_kernel<BMajor><<<1, kWarpgroupThreads>>>(d_out, d_A, d_B, d_C, scale_d);
    result.launch_error = cudaGetLastError();
    if (result.launch_error == cudaSuccess) result.sync_error = cudaDeviceSynchronize();
  }
  if (result.input_copy_error == cudaSuccess && result.launch_error == cudaSuccess &&
      result.sync_error == cudaSuccess)
    result.copy_error = cudaMemcpy(result.output.data(), d_out,
        result.output.size() * 4, cudaMemcpyDeviceToHost);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_out);
  return result;
}

template <int BMajor>
__device__ __forceinline__ void wgmma_m64n256k16_f32_bf16_bf16_ss(
    uint64_t desc_a, uint64_t desc_b, float (&d)[128], int32_t scale_d) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %130, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, %129, p, %131, %132, %133, %134;\n"
      "}\n"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),"+f"(d[64]),"+f"(d[65]),"+f"(d[66]),"+f"(d[67]),"+f"(d[68]),"+f"(d[69]),"+f"(d[70]),"+f"(d[71]),"+f"(d[72]),"+f"(d[73]),"+f"(d[74]),"+f"(d[75]),"+f"(d[76]),"+f"(d[77]),"+f"(d[78]),"+f"(d[79]),"+f"(d[80]),"+f"(d[81]),"+f"(d[82]),"+f"(d[83]),"+f"(d[84]),"+f"(d[85]),"+f"(d[86]),"+f"(d[87]),"+f"(d[88]),"+f"(d[89]),"+f"(d[90]),"+f"(d[91]),"+f"(d[92]),"+f"(d[93]),"+f"(d[94]),"+f"(d[95]),"+f"(d[96]),"+f"(d[97]),"+f"(d[98]),"+f"(d[99]),"+f"(d[100]),"+f"(d[101]),"+f"(d[102]),"+f"(d[103]),"+f"(d[104]),"+f"(d[105]),"+f"(d[106]),"+f"(d[107]),"+f"(d[108]),"+f"(d[109]),"+f"(d[110]),"+f"(d[111]),"+f"(d[112]),"+f"(d[113]),"+f"(d[114]),"+f"(d[115]),"+f"(d[116]),"+f"(d[117]),"+f"(d[118]),"+f"(d[119]),"+f"(d[120]),"+f"(d[121]),"+f"(d[122]),"+f"(d[123]),"+f"(d[124]),"+f"(d[125]),"+f"(d[126]),"+f"(d[127])
      : "l"(desc_a), "l"(desc_b), "r"(scale_d), "n"(1), "n"(1), "n"(0), "n"(BMajor));
}

template <int BMajor>
__global__ void wgmma_m64n256k16_bf16_ss_kernel(float *out, const uint16_t *A,
                                              const uint16_t *B, const float *C,
                                              int32_t scale_d) {
  if (threadIdx.x >= kWarpgroupThreads) return;
  __shared__ uint16_t smem_a[4096];
  __shared__ uint16_t smem_b[16384];
  
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) smem_a[i] = 0;
  for (int i = threadIdx.x; i < 16384; i += blockDim.x) smem_b[i] = 0;
  for (int i = threadIdx.x; i < kM * kK; i += blockDim.x) {
    int row = i / kK, k = i % kK;
    smem_a[bf16_wgmma_k_major_smem_index(row, k)] = A[row * kK + k];
  }
  for (int i = threadIdx.x; i < 256 * kK; i += blockDim.x) {
    int col = i / kK, k = i % kK;
    int idx = BMajor == 0 ? bf16_wgmma_k_major_smem_index(col, k)
                          : bf16_wgmma_mn_major_smem_index(col, k);
    smem_b[idx] = B[col * kK + k];
  }
  __syncthreads();
  uint64_t desc_a = bf16_gmma_swizzle_desc(smem_a, 16, 1024, 1);
  uint64_t desc_b = bf16_gmma_swizzle_desc(smem_b, BMajor == 0 ? 16 : 0, 1024, 1);
  float d[128];
  for (int reg = 0; reg < 128; ++reg) {
    int row = 0, cig = 0;
    accumulator_matrix_coord(threadIdx.x, reg % 4, row, cig);
    d[reg] = C[row * 256 + (reg / 4) * 8 + cig];
  }
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  wgmma_m64n256k16_f32_bf16_bf16_ss<BMajor>(desc_a, desc_b, d, scale_d);
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  for (int reg = 0; reg < 128; ++reg) {
    int row = 0, cig = 0;
    accumulator_matrix_coord(threadIdx.x, reg % 4, row, cig);
    out[row * 256 + (reg / 4) * 8 + cig] = d[reg];
  }
}

template <int BMajor>
RunResult run_wgmma_m64n256k16_bf16_ss(const std::vector<uint16_t> &A,
                                     const std::vector<uint16_t> &B,
                                     const std::vector<float> &C, int32_t scale_d) {
  RunResult result;
  result.output.assign(kM * 256, 0.0f);
  if (A.size() != kM * kK || B.size() != 256 * kK || C.size() != kM * 256) {
    result.setup_error = cudaErrorInvalidValue;
    return result;
  }
  uint16_t *d_A = nullptr, *d_B = nullptr;
  float *d_C = nullptr, *d_out = nullptr;
  result.setup_error = cudaMalloc(&d_A, A.size() * 2);
  if (result.setup_error == cudaSuccess) result.setup_error = cudaMalloc(&d_B, B.size() * 2);
  if (result.setup_error == cudaSuccess) result.setup_error = cudaMalloc(&d_C, C.size() * 4);
  if (result.setup_error == cudaSuccess) result.setup_error = cudaMalloc(&d_out, result.output.size() * 4);
  if (result.setup_error != cudaSuccess) return result;
  result.input_copy_error = cudaMemcpy(d_A, A.data(), A.size() * 2, cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess)
    result.input_copy_error = cudaMemcpy(d_B, B.data(), B.size() * 2, cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess)
    result.input_copy_error = cudaMemcpy(d_C, C.data(), C.size() * 4, cudaMemcpyHostToDevice);
  if (result.input_copy_error == cudaSuccess) {
    wgmma_m64n256k16_bf16_ss_kernel<BMajor><<<1, kWarpgroupThreads>>>(d_out, d_A, d_B, d_C, scale_d);
    result.launch_error = cudaGetLastError();
    if (result.launch_error == cudaSuccess) result.sync_error = cudaDeviceSynchronize();
  }
  if (result.input_copy_error == cudaSuccess && result.launch_error == cudaSuccess &&
      result.sync_error == cudaSuccess)
    result.copy_error = cudaMemcpy(result.output.data(), d_out,
        result.output.size() * 4, cudaMemcpyDeviceToHost);
  cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_out);
  return result;
}

template <int N>
class WgmmaBf16SsTest : public ::testing::Test {
 protected:
  void run_case(int bmajor, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<uint16_t> A(kM * kK), B(N * kK);
    std::vector<float> C(kM * N);
    for (auto &x : A) x = f32_to_bf16(dist(gen));
    for (auto &x : B) x = f32_to_bf16(dist(gen));
    for (auto &x : C) x = dist(gen);
    std::vector<float> ref(kM * N, 0.0f);
    for (int row = 0; row < kM; ++row)
      for (int col = 0; col < N; ++col) {
        float sum = 0.0f;
        for (int k = 0; k < kK; ++k)
          sum += bf16_to_f32(A[row * kK + k]) * bf16_to_f32(B[col * kK + k]);
        ref[row * N + col] = sum + C[row * N + col];
      }
    RunResult result;
    if (N == 16) result = bmajor == 0 ? run_wgmma_m64n16k16_bf16_ss<0>(A, B, C, 1)
                                    : run_wgmma_m64n16k16_bf16_ss<1>(A, B, C, 1);
    else if (N == 64) result = bmajor == 0 ? run_wgmma_m64n64k16_bf16_ss<0>(A, B, C, 1)
                                    : run_wgmma_m64n64k16_bf16_ss<1>(A, B, C, 1);
    else if (N == 128) result = bmajor == 0 ? run_wgmma_m64n128k16_bf16_ss<0>(A, B, C, 1)
                                    : run_wgmma_m64n128k16_bf16_ss<1>(A, B, C, 1);
    else result = bmajor == 0 ? run_wgmma_m64n256k16_bf16_ss<0>(A, B, C, 1)
                              : run_wgmma_m64n256k16_bf16_ss<1>(A, B, C, 1);
    ASSERT_EQ(result.setup_error, cudaSuccess);
    ASSERT_EQ(result.input_copy_error, cudaSuccess);
    ASSERT_EQ(result.launch_error, cudaSuccess);
    ASSERT_EQ(result.sync_error, cudaSuccess);
    ASSERT_EQ(result.copy_error, cudaSuccess);
    for (int i = 0; i < kM * N; ++i) {
      float diff = std::fabs(result.output[i] - ref[i]);
      float tol = 0.15f + 0.03f * std::fabs(ref[i]);
      ASSERT_LE(diff, tol) << "index " << i;
    }
  }
};
class WgmmaBf16SsN16Test : public WgmmaBf16SsTest<16> {};
class WgmmaBf16SsN64Test : public WgmmaBf16SsTest<64> {};
class WgmmaBf16SsN128Test : public WgmmaBf16SsTest<128> {};
class WgmmaBf16SsN256Test : public WgmmaBf16SsTest<256> {};
TEST_F(WgmmaBf16SsN16Test, KMajorRandom) { run_case(0, 2101); }
TEST_F(WgmmaBf16SsN64Test, KMajorRandom) { run_case(0, 2102); }
TEST_F(WgmmaBf16SsN64Test, MnMajorBRandom) { run_case(1, 2103); }
TEST_F(WgmmaBf16SsN128Test, KMajorRandom) { run_case(0, 2104); }
TEST_F(WgmmaBf16SsN256Test, KMajorRandom) { run_case(0, 2105); }

}  // namespace wgmma_test
