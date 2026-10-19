/*
 * Round-to-nearest-even quantization between the two float widths.
 *
 * Storing a float into a narrow float image rounds the way the device chooses,
 * and on some devices that choice is truncation, while a word-packing route and
 * a host cast both round to the nearest representable value with ties to even.
 * So a value headed for a half lane is quantized here first: the bits below
 * give the nearest half, and the float rebuilt from them is exactly
 * representable, which leaves the store nothing to round and makes the result
 * independent of the device.
 */

uint float_bits_to_half_bits(const uint bits) {
  const uint sign = (bits >> 16u) & 0x8000u;
  const int exponent = int((bits >> 23u) & 0xFFu) - 127;
  const uint fraction = bits & 0x7FFFFFu;

  // A float without an exponent has infinite magnitude: keep the sign, set the
  // half exponent, and carry a quiet NaN when the source was one.
  if (exponent >= 128) {
    return sign | 0x7C00u | (fraction != 0u ? 0x200u : 0u);
  }
  if (exponent > 15) {
    return sign | 0x7C00u;
  }
  if (exponent >= -14) {
    uint kept = fraction >> 13u;
    const uint dropped = fraction & 0x1FFFu;
    if (dropped > 0x1000u || (dropped == 0x1000u && (kept & 1u) != 0u)) {
      ++kept; // a carry past the fraction field lands in the exponent, as it must
    }
    return sign | (uint(exponent + 15) << 10u) + kept;
  }
  if (exponent < -25) {
    return sign; // nearer to zero than to the smallest subnormal half
  }
  // The destination has no exponent left to give, so the implicit leading bit
  // joins the fraction and the fraction shifts the extra distance down.
  const uint significand = fraction | 0x800000u;
  const uint shift = uint(-14 - exponent) + 13u;
  const uint hidden = (1u << shift) - 1u;
  const uint halfway = 1u << (shift - 1u);
  uint kept = significand >> shift;
  const uint dropped = significand & hidden;
  if (dropped > halfway || (dropped == halfway && (kept & 1u) != 0u)) {
    ++kept; // a carry here reaches the exponent field, making the smallest normal
  }
  return sign | kept;
}

float half_bits_to_float(const uint half_bits) {
  const uint exponent = (half_bits >> 10u) & 0x1Fu;
  const uint fraction = half_bits & 0x3FFu;
  if (exponent == 0u) {
    if (fraction == 0u) {
      return uintBitsToFloat((half_bits & 0x8000u) << 16u);
    }
    // Every subnormal half is an integer multiple of 2^-24, so the product is
    // exact in single precision.
    const float magnitude = float(fraction) * 5.9604644775390625e-08;
    return (half_bits & 0x8000u) != 0u ? -magnitude : magnitude;
  }
  const uint sign = (half_bits & 0x8000u) << 16u;
  if (exponent == 31u) {
    // Infinite magnitude, or a quiet NaN with the fraction kept where it fits.
    return uintBitsToFloat(sign | 0x7F800000u | (fraction << 13u));
  }
  return uintBitsToFloat(sign | ((exponent + 112u) << 23u) | (fraction << 13u));
}

float round_to_half(const float value) {
  return half_bits_to_float(float_bits_to_half_bits(floatBitsToUint(value)));
}
