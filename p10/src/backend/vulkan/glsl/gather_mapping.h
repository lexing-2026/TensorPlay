/*
 * Gather addressing over a row-major payload
 */

// The source element an output element of a gather stands for.  The output
// element is decomposed over the strides of its own shape, the coordinate on
// the gathered axis is replaced by its code, and the coordinates are recomposed
// with the source strides - so an index tensor narrower than the source along
// another axis still reads the row that coordinate belongs to.  Both stride
// tables describe a four-axis shape left-padded with ones, which is what lets a
// 1-d to 4-d payload share this one mapping; a leading axis of size one has the
// stride of the axis below it, and every element is below the first of the two,
// so its coordinate comes out zero.  The strides are passed one per axis
// because a uniform block cannot hand an array to a function.  -1 marks a code
// that names no row of the source.
int gather_source_element(
    const int element,
    const int code,
    const int out_stride_0,
    const int out_stride_1,
    const int out_stride_2,
    const int out_stride_3,
    const int self_stride_0,
    const int self_stride_1,
    const int self_stride_2,
    const int self_stride_3,
    const int axis,
    const int self_rows) {
  if (code < 0 || code >= self_rows) {
    return -1;
  }
  int rest = element;
  int source = 0;
  for (int axis_i = 0; axis_i < 4; ++axis_i) {
    const int stride = axis_i == 0
        ? out_stride_0
        : (axis_i == 1 ? out_stride_1
                       : (axis_i == 2 ? out_stride_2 : out_stride_3));
    const int self_stride = axis_i == 0
        ? self_stride_0
        : (axis_i == 1 ? self_stride_1
                       : (axis_i == 2 ? self_stride_2 : self_stride_3));
    const int coord = rest / stride;
    rest -= coord * stride;
    source += (axis_i == axis ? code : coord) * self_stride;
  }
  return source;
}
