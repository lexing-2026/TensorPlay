import tensorplay as tp

def test_transpose_behavior():
    # Contiguous (128, 64) transposed must stay a view with swapped
    # shape and strides, no materialization.
    t = tp.randn([128, 64])

    t_t = t.t()

    assert t_t.shape == [64, 128], f"shape {t_t.shape} != [64, 128]"
    # stride() reports a tuple; a (64, 128) view over row-major (128, 64)
    # storage must read (1, 64).
    assert t_t.stride() == (1, 64), f"strides {t_t.stride()} != (1, 64)"

if __name__ == "__main__":
    test_transpose_behavior()
