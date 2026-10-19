/*
 * Broadcasting by stride arithmetic over a linear payload.
 *
 * A row-major flat index of the output shape decomposes into per-dimension
 * coordinates; an operand that is narrower along a dimension repeats whatever
 * it holds there, which the zero stride of a singleton dimension expresses.
 * So the operand's own flat index is the coordinate sum over its strides, and
 * no operand has to be materialized at the output shape.
 *
 * `out_strides` is the row-major stride vector of the output shape, zero-filled
 * past its rank, and `sizes`/`strides` are the operand's extents right-aligned
 * to that rank with a zero stride on every singleton.  A missing leading
 * dimension carries a size of one, so it repeats the whole plane.
 */
uint remap_index(
    uint i,
    ivec4 out_strides,
    ivec4 sizes,
    ivec4 strides) {
  uint rem = i;
  uint acc = 0u;
  for (int d = 0; d < 4; ++d) {
    const uint s = uint(out_strides[d]);
    if (s == 0u) {
      break;
    }
    const uint coord = rem / s;
    rem = rem % s;
    if (uint(sizes[d]) > 1u) {
      acc += coord * uint(strides[d]);
    }
  }
  return acc;
}
