#include "tensor_wgmma.h"

#include <cassert>
#include <climits>
#include <cstdint>
#include <cstring>

#include "../../../abstract_hardware_model.h"
#include "../../../cuda-sim/ptx_ir.h"
#include "../lowp_float.h"
#include "../mma/tensor_mma.h"

class ptx_recognizer;
typedef void *yyscan_t;
#include "../../../trace.h"
#include "ptx.tab.h"

namespace flash_gpgpu_sim {

namespace {

template <unsigned ESZ> uint32_t packed_elem(const ptx_reg_t &reg, int index) {
  return (reg.u32 >> (index * ESZ * 8)) & ((1u << (ESZ * 8)) - 1u);
}

template <> uint32_t packed_elem<4>(const ptx_reg_t &reg, int index) {
  (void)index;
  return reg.u32;
}

float bits_to_f32(uint32_t bits) {
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

int32_t saturate_s32(int64_t value) {
  if (value > INT_MAX)
    return INT_MAX;
  if (value < INT_MIN)
    return INT_MIN;
  return static_cast<int32_t>(value);
}

// Generic dense WGMMA implementation. ESZ is the operand element size in
// bytes, K the instruction K dimension, AConv/BConv convert one raw element
// (zero-extended to 32 bits) into the accumulation domain, and AccT is the
// accumulator type (float or int32_t).
template <unsigned ESZ, int K, typename AccT, bool IntAccum, typename AConv,
          typename BConv>
void wgmma_m64nXkY_generic_impl(const ptx_instruction *pI, core_t *core,
                                warp_inst_t &inst, AConv a_to_acc,
                                BConv b_to_acc, bool has_trans) {
  constexpr int M = 64;
  constexpr int EPR = 4 / ESZ; // elements per 32-bit A register

  const operand_info &dst = pI->operand_lookup(0);
  const operand_info &src_a = pI->operand_lookup(1);
  const operand_info &src_b_desc = pI->operand_lookup(2);
  const operand_info &scale_d_pred = pI->operand_lookup(3);
  int N = pI->get_wgmma_shape_n();
  if (!dst.is_vector() ||
      dst.get_vect_nelem() != static_cast<unsigned>(N / 2)) {
    return;
  }

  AccT A_mat[M * K];
  AccT B_mat[256 * K];
  for (int i = 0; i < M * K; ++i)
    A_mat[i] = AccT{};
  for (int i = 0; i < N * K; ++i)
    B_mat[i] = AccT{};

  unsigned thread_count = wgmma_thread_count(core, inst);
  ptx_thread_info *representative_thread = wgmma_thread(core, inst, 0);
  bool a_from_register = src_a.is_vector();
  int operand_count = static_cast<int>(pI->get_num_operands());
  int scale_a = 1;
  int scale_b = 1;
  int major_a = 0;
  int major_b = 0;

  if (operand_count > 4) {
    ptx_reg_t scale = representative_thread->get_operand_value(
        pI->operand_lookup(4), pI->operand_lookup(4), S32_TYPE,
        representative_thread, 0);
    scale_a = scale.s32 == -1 ? -1 : 1;
  }
  if (operand_count > 5) {
    ptx_reg_t scale = representative_thread->get_operand_value(
        pI->operand_lookup(5), pI->operand_lookup(5), S32_TYPE,
        representative_thread, 0);
    scale_b = scale.s32 == -1 ? -1 : 1;
  }
  if (has_trans) {
    if (a_from_register) {
      if (operand_count > 6) {
        ptx_reg_t major = representative_thread->get_operand_value(
            pI->operand_lookup(6), pI->operand_lookup(6), S32_TYPE,
            representative_thread, 0);
        major_b = major.s32;
      }
    } else {
      if (operand_count > 6) {
        ptx_reg_t major = representative_thread->get_operand_value(
            pI->operand_lookup(6), pI->operand_lookup(6), S32_TYPE,
            representative_thread, 0);
        major_a = major.s32;
      }
      if (operand_count > 7) {
        ptx_reg_t major = representative_thread->get_operand_value(
            pI->operand_lookup(7), pI->operand_lookup(7), S32_TYPE,
            representative_thread, 0);
        major_b = major.s32;
      }
    }
  }

  ptx_reg_t b_desc_reg = representative_thread->get_operand_value(
      src_b_desc, src_b_desc, U64_TYPE, representative_thread, 0);
  uint64_t b_desc = static_cast<uint64_t>(b_desc_reg.u64);

  for (int col = 0; col < N; ++col) {
    for (int k = 0; k < K; ++k) {
      uint32_t b_bits = 0;
      unsigned b_smem_addr =
          major_b == 0 ? wgmma_gmma_k_major_smem_addr(b_desc, col, k, ESZ, K)
                       : wgmma_gmma_mn_major_smem_addr(b_desc, col, k, ESZ);
      representative_thread->m_shared_mem->read(b_smem_addr, ESZ, &b_bits);
      B_mat[col * K + k] = static_cast<AccT>(scale_b * b_to_acc(b_bits));
    }
  }

  if (a_from_register) {
    if (src_a.get_vect_nelem() != 4)
      return;
    for (unsigned thrd = 0; thrd < thread_count; ++thrd) {
      ptx_thread_info *thread = wgmma_thread(core, inst, thrd);
      unsigned lane = wgmma_lane(thread);
      int tid_mma_col = lane % 4;
      int tid_row = (lane / 4) % 8;
      int tid_m_block = lane / 32;

      ptx_reg_t a_regs[4];
      thread->get_vector_operand_values(src_a, a_regs, 4);

      for (int reg = 0; reg < 4; ++reg) {
        int reg_row_block = reg & 0x1;
        int reg_k_block = (reg >> 1) & 0x1;
        int row = tid_row + 16 * tid_m_block + 8 * reg_row_block;
        int k0 = EPR * tid_mma_col + (K / 2) * reg_k_block;

        for (int e = 0; e < EPR; ++e)
          A_mat[row * K + k0 + e] = static_cast<AccT>(
              scale_a * a_to_acc(packed_elem<ESZ>(a_regs[reg], e)));
      }
    }
  } else {
    ptx_reg_t a_desc_reg = representative_thread->get_operand_value(
        src_a, src_a, U64_TYPE, representative_thread, 0);
    uint64_t a_desc = static_cast<uint64_t>(a_desc_reg.u64);
    for (int row = 0; row < M; ++row) {
      for (int k = 0; k < K; ++k) {
        uint32_t a_bits = 0;
        unsigned a_smem_addr =
            major_a == 0 ? wgmma_gmma_k_major_smem_addr(a_desc, row, k, ESZ, K)
                         : wgmma_gmma_mn_major_smem_addr(a_desc, row, k, ESZ);
        representative_thread->m_shared_mem->read(a_smem_addr, ESZ, &a_bits);
        A_mat[row * K + k] = static_cast<AccT>(scale_a * a_to_acc(a_bits));
      }
    }
  }

  for (unsigned thrd = 0; thrd < thread_count; ++thrd) {
    ptx_thread_info *thread = wgmma_thread(core, inst, thrd);
    unsigned lane = wgmma_lane(thread);
    unsigned dst_elements = dst.get_vect_nelem();

    ptx_reg_t pred = thread->get_operand_value(scale_d_pred, scale_d_pred,
                                               PRED_TYPE, thread, 0);
    bool include_c = (pred.pred & 0x1u) == 0;

    for (unsigned reg = 0; reg < dst_elements; ++reg) {
      ptx_reg_t c_reg = thread->get_reg(dst.vec_symbol(reg));
      int row = 0;
      int col_in_tile = 0;
      wgmma_m64n8_accumulator_coord(lane, reg % 4, row, col_in_tile);
      int col = static_cast<int>((reg / 4) * 8) + col_in_tile;

      ptx_reg_t d_reg;
      if (IntAccum) {
        int64_t product = 0;
        for (int k = 0; k < K; ++k)
          product += static_cast<int64_t>(A_mat[row * K + k]) *
                     static_cast<int64_t>(B_mat[col * K + k]);
        int64_t result = product + (include_c ? c_reg.s32 : 0);
        d_reg.s32 = pI->get_wgmma_saturate()
                        ? saturate_s32(result)
                        : static_cast<int32_t>(static_cast<uint32_t>(result));
      } else {
        float product = 0.0f;
        for (int k = 0; k < K; ++k)
          product += static_cast<float>(A_mat[row * K + k]) *
                     static_cast<float>(B_mat[col * K + k]);
        d_reg.f32 = product + (include_c ? c_reg.f32 : 0.0f);
      }
      thread->set_reg(dst.vec_symbol(reg), d_reg);
    }
  }
}

} // namespace

void wgmma_m64nXk16_f16_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst) {
  wgmma_m64nXkY_generic_impl<2, 16, float, false>(
      pI, core, inst,
      [](uint32_t bits) { return mma_f16_to_f32(static_cast<uint16_t>(bits)); },
      [](uint32_t bits) { return mma_f16_to_f32(static_cast<uint16_t>(bits)); },
      true);
}

void wgmma_m64nXk16_bf16_impl(const ptx_instruction *pI, core_t *core,
                              warp_inst_t &inst) {
  wgmma_m64nXkY_generic_impl<2, 16, float, false>(
      pI, core, inst,
      [](uint32_t bits) {
        return mma_bf16_to_f32(static_cast<uint16_t>(bits));
      },
      [](uint32_t bits) {
        return mma_bf16_to_f32(static_cast<uint16_t>(bits));
      },
      true);
}

void wgmma_m64nXk8_tf32_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst) {
  wgmma_m64nXkY_generic_impl<4, 8, float, false>(
      pI, core, inst,
      [](uint32_t bits) { return mma_tf32_round(bits_to_f32(bits)); },
      [](uint32_t bits) { return mma_tf32_round(bits_to_f32(bits)); }, false);
}

void wgmma_m64nXk32_fp8_impl(const ptx_instruction *pI, core_t *core,
                             warp_inst_t &inst, bool a_is_e4m3,
                             bool b_is_e4m3) {
  wgmma_m64nXkY_generic_impl<1, 32, float, false>(
      pI, core, inst,
      [a_is_e4m3](uint32_t bits) {
        return fp8_to_f32(static_cast<uint8_t>(bits), a_is_e4m3);
      },
      [b_is_e4m3](uint32_t bits) {
        return fp8_to_f32(static_cast<uint8_t>(bits), b_is_e4m3);
      },
      false);
}

void wgmma_m64nXk32_int8_impl(const ptx_instruction *pI, core_t *core,
                              warp_inst_t &inst, bool a_is_signed,
                              bool b_is_signed) {
  auto to_i32 = [](uint32_t bits, bool is_signed) {
    return is_signed ? static_cast<int32_t>(static_cast<int8_t>(bits))
                     : static_cast<int32_t>(static_cast<uint8_t>(bits));
  };
  wgmma_m64nXkY_generic_impl<1, 32, int32_t, true>(
      pI, core, inst,
      [&to_i32, a_is_signed](uint32_t bits) {
        return to_i32(bits, a_is_signed);
      },
      [&to_i32, b_is_signed](uint32_t bits) {
        return to_i32(bits, b_is_signed);
      },
      false);
}

} // namespace flash_gpgpu_sim
