/*
 * `pow` leaves several questions unanswered here.  A base of nothing has no
 * answer for a zero exponent, although it still names one, and none for a
 * negative exponent, which names a magnitude no finite word holds.  A base of
 * one has no answer for an exponent past every word either, although nothing
 * multiplied by itself ever leaves one, and neither does its negative, which
 * turns back on itself every other time.  Every other case is the hardware
 * power, so a caller wraps the expression it already has instead of restating
 * it.
 *
 * A texel carries four lanes and each decides on its own.  The selects below
 * weigh no branch against another, so the not-a-number a corner leaves behind
 * cannot spread into a lane that asked for nothing.
 */

// The two masks of a corner as lanes of 0.0 or 1.0 -- a base of nothing, and an
// exponent below nothing.  Their product names where both hold at once.
vec4 zero_base_corner(const vec4 base, const vec4 exponent) {
  return step(abs(base), vec4(0.0)) * (vec4(1.0) - step(vec4(0.0), exponent));
}

// Float corners.  Inverting the power of the opposite exponent is how the
// magnitude past every finite value reads: a nothing base drives it there.
vec4 pow_guard(const vec4 base, const vec4 exponent, const vec4 value) {
  const vec4 past_reach = 1.0 / pow(base, -exponent);
  const vec4 lowered =
      mix(value, past_reach, notEqual(zero_base_corner(base, exponent),
                                      vec4(0.0)));
  const vec4 raised = mix(lowered, vec4(1.0), equal(exponent, vec4(0.0)));
  const vec4 pair = mix(raised, vec4(1.0), equal(abs(base), vec4(1.0)));
  // An exponent that leaves nothing of itself to divide one by is past every
  // word, and only the bases that are their own reciprocals settle there.
  return mix(raised, pair, equal(vec4(1.0) / abs(exponent), vec4(0.0)));
}

// The linear writer covers one exponent per element, so the corners are plain
// scalar decisions and a select needs no mask.
float pow_guard(const float base, const float exponent, const float value) {
  if (exponent == 0.0 || base == 1.0) {
    return 1.0;
  }
  if (base == -1.0 && 1.0 / abs(exponent) == 0.0) {
    return 1.0;
  }
  return (base == 0.0 && exponent < 0.0) ? 1.0 / pow(base, -exponent) : value;
}

/*
 * A count raised to a count, taken the way a word multiplies: the exponent is
 * spent one bit at a time, squaring the base along the way, so a whole power
 * lands on the number it names rather than on the single-precision estimate of
 * it -- an estimate is already half a count away from the truth by the time the
 * exponent reaches the top of a word.  Past the top the product wraps.  An
 * exponent of nothing spends nothing, which is how nothing raised to nothing
 * still names one.
 *
 * A fraction of a count keeps a word only where the base is its own reciprocal:
 * the one word is itself, the negative one turns with whether the exponent is
 * odd or even, and anything wider has nothing to give back.
 */
int pow_count(const int base, const int exponent) {
  if (exponent < 0) {
    if (base == 1) {
      return 1;
    }
    if (base == -1) {
      return mod(exponent, 2) != 0 ? -1 : 1;
    }
    return 0;
  }
  int result = 1;
  int value = base;
  int remaining = exponent;
  while (remaining != 0) {
    if ((remaining & 1) != 0) {
      result *= value;
    }
    value *= value;
    remaining >>= 1;
  }
  return result;
}

// One count power per lane of the texel.
ivec4 pow_count(const ivec4 base, const ivec4 exponent) {
  return ivec4(pow_count(base.x, exponent.x),
               pow_count(base.y, exponent.y),
               pow_count(base.z, exponent.z),
               pow_count(base.w, exponent.w));
}
