import numpy as np
import tensorplay as tp

# The unary maps whose out= pass walks one flat loop over the operand and the
# destination.  Such a loop is only faithful when the two meet at the same
# storage place: a packed destination, or one permuted exactly the way the
# operand is.  A destination stepped differently has to be declined, so these
# tests hold the judgement in both directions -- the permutation the pass must
# take, and the one it must send to the hand-over.
MAPS = [
    ("exp", lambda x: tp.exp(x), lambda x, out: tp.exp(x, out=out)),
    ("log", lambda x: tp.log(x), lambda x, out: tp.log(x, out=out)),
    ("tanh", lambda x: tp.tanh(x), lambda x, out: tp.tanh(x, out=out)),
    ("sigmoid", lambda x: tp.sigmoid(x), lambda x, out: tp.sigmoid(x, out=out)),
    ("abs", lambda x: tp.abs(x), lambda x, out: tp.abs(x, out=out)),
    ("neg", lambda x: tp.neg(x), lambda x, out: tp.neg(x, out=out)),
    ("floor", lambda x: tp.floor(x), lambda x, out: tp.floor(x, out=out)),
    ("ceil", lambda x: tp.ceil(x), lambda x, out: tp.ceil(x, out=out)),
    ("trunc", lambda x: tp.trunc(x), lambda x, out: tp.trunc(x, out=out)),
    ("round", lambda x: tp.round(x), lambda x, out: tp.round(x, out=out)),
    ("clamp", lambda x: tp.clamp(x, -1.1, 0.9),
     lambda x, out: tp.clamp(x, -1.1, 0.9, out=out)),
]

WIDTHS = (tp.float32, tp.float64, tp.float16)

ROWS, COLS = 13, 19  # neither factor a vector width

# Positive values: log answers rather than producing a NaN to compare.
_rng = np.random.default_rng(7)
PLAIN = np.abs(_rng.standard_normal(ROWS * COLS)) + 0.25
# Both signs, so a bound or a slope actually binds in the in-place spellings.
SIGNED = _rng.standard_normal(ROWS * COLS) * 1.6


def matrix(dtype):
    return tp.from_numpy(np.array(PLAIN, dtype=np.float32)).to(dtype).reshape(
        ROWS, COLS)


def signed_matrix(dtype):
    return tp.from_numpy(np.array(SIGNED, dtype=np.float32)).to(dtype).reshape(
        ROWS, COLS)


def test_out_walks_a_destination_permuted_as_the_operand_is():
    for name, alloc_fn, out_fn in MAPS:
        for dtype in WIDTHS:
            operand = matrix(dtype).transpose(0, 1)
            want = alloc_fn(operand).tolist()
            dest = tp.zeros(ROWS, COLS, dtype=dtype).transpose(0, 1)
            result = out_fn(operand, out=dest)
            assert result.data_ptr() == dest.data_ptr(), (name, dtype)
            assert dest.tolist() == want, (name, dtype, "same permutation")


def test_out_declines_a_destination_permuted_otherwise():
    # The operand packed, the destination stepped: one flat loop would meet the
    # two in different orders, so the value has to arrive by the hand-over and
    # still land where the caller is looking.
    for name, alloc_fn, out_fn in MAPS:
        for dtype in WIDTHS:
            operand = matrix(dtype)
            want = alloc_fn(operand).tolist()
            dest = tp.zeros(COLS, ROWS, dtype=dtype).transpose(0, 1)
            out_fn(operand, out=dest)
            assert dest.shape == operand.shape, (name, dtype)
            assert dest.tolist() == want, (name, dtype, "other permutation")


def test_out_writes_every_destination_it_judges():
    # The layouts the judgement is made of, held to the allocating answer at
    # every width: packed, stepped, and a window over the operand's own bytes.
    for name, alloc_fn, out_fn in MAPS:
        for dtype in WIDTHS:
            flat = matrix(dtype).reshape(-1)
            want = alloc_fn(flat).tolist()

            packed = tp.zeros(ROWS * COLS, dtype=dtype)
            assert out_fn(flat, out=packed).tolist() == want, (name, dtype, "packed")

            stepped = tp.zeros(2 * ROWS * COLS, dtype=dtype)[::2]
            assert out_fn(flat, out=stepped).tolist() == want, (name, dtype, "stepped")

            as_operand = matrix(dtype).reshape(-1)
            out_fn(as_operand, out=as_operand)
            assert as_operand.tolist() == want, (name, dtype, "in place")


# The in-place spellings, which hand the tensor to the pass as both operand and
# destination.  A view that carries a permutation is read through that
# permutation, so the pass either honours it or has to decline the whole thing.
IN_PLACE = [
    ("clamp_min_", lambda t: t.clamp_min_(-1.1), lambda x: tp.clamp_min(x, -1.1)),
    ("clamp_max_", lambda t: t.clamp_max_(0.9), lambda x: tp.clamp_max(x, 0.9)),
    ("abs_", lambda t: t.abs_(), lambda x: tp.abs(x)),
    ("neg_", lambda t: t.neg_(), lambda x: tp.neg(x)),
    ("floor_", lambda t: t.floor_(), lambda x: tp.floor(x)),
    ("ceil_", lambda t: t.ceil_(), lambda x: tp.ceil(x)),
    ("trunc_", lambda t: t.trunc_(), lambda x: tp.trunc(x)),
    ("round_", lambda t: t.round_(), lambda x: tp.round(x)),
    ("relu_", lambda t: t.relu_(), lambda x: tp.relu(x)),
    ("exp_", lambda t: t.exp_(), lambda x: tp.exp(x)),
    ("tanh_", lambda t: t.tanh_(), lambda x: tp.tanh(x)),
    ("sigmoid_", lambda t: t.sigmoid_(), lambda x: tp.sigmoid(x)),
    ("sign_", lambda t: t.sign_(), lambda x: tp.sign(x)),
]


def test_in_place_answers_through_a_permutation_the_operand_carries():
    for name, mutate, functional in IN_PLACE:
        for dtype in WIDTHS:
            stepped = signed_matrix(dtype).transpose(0, 1)
            want = functional(signed_matrix(dtype).transpose(0, 1)).tolist()
            mutate(stepped)
            assert stepped.tolist() == want, (name, dtype, "permuted operand")

            packed = signed_matrix(dtype)
            want = functional(signed_matrix(dtype)).tolist()
            mutate(packed)
            assert packed.tolist() == want, (name, dtype, "packed operand")
