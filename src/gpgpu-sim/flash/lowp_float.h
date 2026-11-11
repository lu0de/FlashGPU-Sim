// Shared low-precision element codecs for the functional WGMMA/cvt models.
// The fp8 decoders are bit-identical to the ones originally in wgmma_fp8.cc.

#ifndef FLASH_GPGPU_SIM_LOWP_FLOAT_H
#define FLASH_GPGPU_SIM_LOWP_FLOAT_H

#include <cmath>
#include <cstdint>

namespace flash_gpgpu_sim {

inline float e4m3_to_f32(uint8_t bits) {
  const int sign = (bits >> 7) & 0x1;
  const int exp = (bits >> 3) & 0xF;
  const int mant = bits & 0x7;

  float value = 0.0f;
  if (exp == 0) {
    value = mant == 0 ? 0.0f : std::ldexp(static_cast<float>(mant) / 8.0f, -6);
  } else if (exp == 0xF && mant == 0x7) {
    value = NAN;
  } else {
    value = std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, exp - 7);
  }

  return sign ? -value : value;
}

inline float e5m2_to_f32(uint8_t bits) {
  const int sign = (bits >> 7) & 0x1;
  const int exp = (bits >> 2) & 0x1F;
  const int mant = bits & 0x3;

  float value = 0.0f;
  if (exp == 0) {
    value = mant == 0 ? 0.0f : std::ldexp(static_cast<float>(mant) / 4.0f, -14);
  } else if (exp == 0x1F) {
    value = mant == 0 ? INFINITY : NAN;
  } else {
    value = std::ldexp(1.0f + static_cast<float>(mant) / 4.0f, exp - 15);
  }

  return sign ? -value : value;
}

inline float fp8_to_f32(uint8_t bits, bool is_e4m3) {
  return is_e4m3 ? e4m3_to_f32(bits) : e5m2_to_f32(bits);
}

inline int fp8_round_to_even(float value) {
  const float base_f = std::floor(value);
  const int base = static_cast<int>(base_f);
  const float fraction = value - base_f;
  if (fraction > 0.5f)
    return base + 1;
  if (fraction < 0.5f)
    return base;
  return (base & 1) ? base + 1 : base;
}

// f32 -> fp8 with round-to-nearest-even and satfinite (matches
// cvt.rn.satfinite.{e4m3x2,e5m2x2}.f32 lane semantics).
inline uint8_t f32_to_fp8(float value, bool is_e4m3) {
  if (std::isnan(value))
    return 0x7f;
  const uint8_t sign = std::signbit(value) ? 0x80u : 0u;
  float magnitude = std::fabs(value);
  const float max_norm = is_e4m3 ? 448.0f : 57344.0f;
  if (magnitude > max_norm || std::isinf(magnitude))
    magnitude = max_norm;
  if (magnitude == 0.0f)
    return sign;

  const int mantissa_bits = is_e4m3 ? 3 : 2;
  const int bias = is_e4m3 ? 7 : 15;
  const int max_exp = is_e4m3 ? 15 : 30;
  const float min_normal = std::ldexp(1.0f, 1 - bias);
  if (magnitude < min_normal) {
    const float subnormal_step = std::ldexp(1.0f, 1 - bias - mantissa_bits);
    int mantissa = fp8_round_to_even(magnitude / subnormal_step);
    if (mantissa == 0)
      return sign;
    const int mantissa_limit = 1 << mantissa_bits;
    if (mantissa >= mantissa_limit) {
      return sign | static_cast<uint8_t>(1 << mantissa_bits);
    }
    return sign | static_cast<uint8_t>(mantissa);
  }

  int exponent = 0;
  const float fraction = std::frexp(magnitude, &exponent) * 2.0f;
  int encoded_exp = exponent - 1 + bias;
  int mantissa = fp8_round_to_even((fraction - 1.0f) *
                                   static_cast<float>(1 << mantissa_bits));
  if (mantissa == (1 << mantissa_bits)) {
    mantissa = 0;
    ++encoded_exp;
  }
  if (encoded_exp > max_exp || (is_e4m3 && encoded_exp == 15 && mantissa > 6)) {
    return sign | static_cast<uint8_t>(is_e4m3 ? 0x7e : 0x7b);
  }
  return sign | static_cast<uint8_t>((encoded_exp << mantissa_bits) | mantissa);
}

} // namespace flash_gpgpu_sim

#endif // FLASH_GPGPU_SIM_LOWP_FLOAT_H
