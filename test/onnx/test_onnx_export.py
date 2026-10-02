"""Tests for tensorplay.onnx export."""

import io

import numpy as np
import pytest

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    pytest.skip("onnx/onnxruntime not installed", allow_module_level=True)

import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay.export import Dim, export
from tensorplay.onnx import UnsupportedOperatorError, VerificationError
from tensorplay.onnx import export as onnx_export
from onnx import TensorProto, checker


def _run_onnx_model(model_proto, inputs):
    buf = io.BytesIO()
    onnx.save(model_proto, buf)
    buf.seek(0)
    sess = ort.InferenceSession(
        buf.getvalue(), providers=["CPUExecutionProvider"]
    )
    input_names = [i.name for i in sess.get_inputs()]
    out = sess.run(None, dict(zip(input_names, inputs)))
    return out[0] if len(out) == 1 else out


def _export(module, *args, **kwargs):
    """Export, validate and numerically verify a module in one step."""

    program = export(module, *args)
    model = onnx_export(program, verify=True, **kwargs)
    checker.check_model(model)
    return model


def _export_fn(fn, x, **kwargs):
    return _export(_as_module(fn), x, **kwargs)


def _as_module(fn):
    """Wrap a one-argument callable in a module the tracer can capture."""

    class M(tp.nn.Module):
        def forward(self, x):
            return fn(x)

    return M()


def _op_types(model):
    return [node.op_type for node in model.graph.node]


class TestLinear:
    def test_simple_linear(self):
        class M(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = tp.nn.Linear(4, 2)
            def forward(self, x):
                return self.l(x)

        model = M()
        x = tp.ones((2, 4))
        prog = export(model, x)
        onnx_model = onnx_export(prog)
        checker.check_model(onnx_model)
        assert onnx_model.opset_import[0].version == 18

        tp_out = model(x).detach().numpy()
        ort_out = _run_onnx_model(onnx_model, [x.numpy()])
        assert tp_out.shape == ort_out.shape
        assert np.allclose(tp_out, ort_out, atol=1e-5)

    def test_rank_2_input_uses_gemm(self):
        class M(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = tp.nn.Linear(4, 2)
            def forward(self, x):
                return self.l(x)

        model = _export(M(), tp.randn((2, 4)))
        assert "Gemm" in _op_types(model)

    def test_batched_input_uses_matmul(self):
        class M(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = tp.nn.Linear(4, 2)
            def forward(self, x):
                return self.l(x)

        model = _export(M(), tp.randn((2, 3, 4)))
        assert "MatMul" in _op_types(model)

    def test_two_layer_relu(self):
        class M(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = tp.nn.Linear(4, 8)
                self.l2 = tp.nn.Linear(8, 2)
            def forward(self, x):
                return self.l2(tp.relu(self.l1(x)))

        model = M()
        x = tp.ones((2, 4))
        prog = export(model, x)
        onnx_model = onnx_export(prog)
        checker.check_model(onnx_model)

        tp_out = model(x).detach().numpy()
        ort_out = _run_onnx_model(onnx_model, [x.numpy()])
        assert np.allclose(tp_out, ort_out, atol=1e-5)


class TestElementwise:
    def test_add_mul_sub_div(self):
        class M(tp.nn.Module):
            def forward(self, x, y):
                return x + y, x * y, x - y, x / y

        model = M()
        x = tp.ones((2, 3))
        y = tp.ones((2, 3)) * 2
        prog = export(model, x, y)
        onnx_model = onnx_export(prog)
        checker.check_model(onnx_model)

        tp_out = model(x, y)
        ort_out = _run_onnx_model(onnx_model, [x.numpy(), y.numpy()])
        assert len(tp_out) == 4
        for a, b in zip(tp_out, ort_out):
            assert np.allclose(a.detach().numpy(), b, atol=1e-5)

    @pytest.mark.parametrize(
        "fn",
        [
            lambda x: tp.abs(x),
            lambda x: tp.exp(x),
            lambda x: tp.sqrt(tp.abs(x)),
            lambda x: tp.sign(x),
            lambda x: -x,
            lambda x: x**2,
            lambda x: x // 2.0,
            lambda x: x != 0,
            lambda x: tp.maximum(x, x * 2),
            lambda x: tp.minimum(x, x * 2),
            lambda x: tp.clamp(x, -0.5, 0.5),
            lambda x: tp.where(x > 0, x, x * 2),
            lambda x: x.masked_fill(x > 0, 0.0),
        ],
    )
    def test_pointwise(self, fn):
        _export_fn(fn, tp.randn((3, 4)))

    @pytest.mark.parametrize(
        "fn",
        [
            lambda x: tp.remainder(x, 3.0),
            lambda x: tp.remainder(x, x.abs() + 0.5),
            lambda x: tp.fmod(x, 3.0),
            lambda x: tp.floor_divide(x, 3.0),
            lambda x: tp.multiply(x, 2.0),
            lambda x: tp.divide(x, 2.0),
            lambda x: tp.true_divide(x, 2.0),
            lambda x: tp.clamp_min(x, 0.0),
            lambda x: tp.clamp_max(x, 0.0),
        ],
    )
    def test_arithmetic_composites(self, fn):
        _export_fn(fn, tp.randn((3, 4)) * 5)

    @pytest.mark.parametrize(
        "fn",
        [
            lambda x: tp.remainder(x, 3),
            lambda x: tp.fmod(x, 3),
        ],
    )
    def test_integer_arithmetic_composites(self, fn):
        values = np.array([[-7, -3, 4, 9], [5, -2, 8, -11]], dtype=np.int64)
        _export_fn(fn, tp.tensor(values))

    @pytest.mark.parametrize(
        "fn",
        [
            lambda x: tp.add(x, x, alpha=3.0),
            lambda x: tp.sub(x, x, alpha=3.0),
            lambda x: tp.subtract(x, x, alpha=2.0),
        ],
    )
    def test_alpha_is_not_dropped(self, fn):
        """``alpha`` scales the second operand; ignoring it exports silently
        wrong arithmetic."""

        _export_fn(fn, tp.randn((3, 4)))


class TestActivation:
    def test_sigmoid(self):
        class M(tp.nn.Module):
            def forward(self, x):
                return tp.sigmoid(x)

        model = M()
        x = tp.randn((2, 3))
        prog = export(model, x)
        onnx_model = onnx_export(prog)
        checker.check_model(onnx_model)

        tp_out = model(x).detach().numpy()
        ort_out = _run_onnx_model(onnx_model, [x.numpy()])
        assert np.allclose(tp_out, ort_out, atol=1e-5)

    def test_softmax(self):
        class M(tp.nn.Module):
            def forward(self, x):
                return tp.softmax(x, dim=1)

        model = M()
        x = tp.randn((2, 3))
        prog = export(model, x)
        onnx_model = onnx_export(prog)
        checker.check_model(onnx_model)

        tp_out = model(x).detach().numpy()
        ort_out = _run_onnx_model(onnx_model, [x.numpy()])
        assert np.allclose(tp_out, ort_out, atol=1e-5)

    @pytest.mark.parametrize(
        "fn",
        [
            lambda x: F.relu(x),
            lambda x: F.gelu(x),
            lambda x: F.gelu(x, approximate="tanh"),
            lambda x: F.silu(x),
            lambda x: F.mish(x),
            lambda x: F.elu(x),
            lambda x: F.leaky_relu(x, 0.1),
            lambda x: F.hardtanh(x),
            lambda x: F.hardsigmoid(x),
            lambda x: F.hardswish(x),
            lambda x: F.softplus(x),
            lambda x: F.softplus(x, 2.0),
            lambda x: F.log_softmax(x, 1),
            lambda x: F.glu(x),
            lambda x: F.normalize(x),
            lambda x: F.dropout(x, 0.5, False),
        ],
    )
    def test_activations(self, fn):
        _export_fn(fn, tp.randn((2, 4)))

    def test_gelu_decomposes_below_opset_20(self):
        model = _export_fn(lambda x: F.gelu(x), tp.randn((2, 4)), opset_version=18)
        assert "Erf" in _op_types(model)
        assert "Gelu" not in _op_types(model)

    def test_gelu_is_native_from_opset_20(self):
        model = _export_fn(lambda x: F.gelu(x), tp.randn((2, 4)), opset_version=20)
        assert "Gelu" in _op_types(model)

    def test_prelu_with_channel_weight(self):
        _export_fn(lambda x: F.prelu(x, tp.randn((3,))), tp.randn((2, 3, 4, 4)))


class TestConvolution:
    def test_conv2d_module(self):
        class M(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.c = tp.nn.Conv2d(3, 4, 3, stride=1, padding=1)
            def forward(self, x):
                return self.c(x)

        model = _export(M(), tp.randn((2, 3, 8, 8)))
        assert "Conv" in _op_types(model)

    def test_conv1d(self):
        _export_fn(
            lambda x: F.conv1d(x, tp.randn((2, 3, 3))), tp.randn((1, 3, 8))
        )

    def test_conv3d(self):
        _export_fn(
            lambda x: F.conv3d(x, tp.randn((2, 3, 3, 3, 3))),
            tp.randn((1, 3, 4, 4, 4)),
        )

    def test_conv2d_strided_with_bias(self):
        _export_fn(
            lambda x: F.conv2d(x, tp.randn((2, 3, 3, 3)), tp.randn((2,)), 2, 1, 1, 1),
            tp.randn((2, 3, 8, 8)),
        )

    def test_conv_transpose2d(self):
        _export_fn(
            lambda x: F.conv_transpose2d(x, tp.randn((3, 2, 3, 3))),
            tp.randn((2, 3, 8, 8)),
        )


class TestPooling:
    @pytest.mark.parametrize(
        "fn,shape",
        [
            (lambda x: F.max_pool1d(x, 2), (1, 3, 8)),
            (lambda x: F.max_pool2d(x, 2), (2, 3, 8, 8)),
            (lambda x: F.max_pool3d(x, 2), (1, 3, 4, 4, 4)),
            (lambda x: F.avg_pool1d(x, 2), (1, 3, 8)),
            (lambda x: F.avg_pool2d(x, 2), (2, 3, 8, 8)),
            (lambda x: F.avg_pool3d(x, 2), (1, 3, 4, 4, 4)),
        ],
    )
    def test_pooling(self, fn, shape):
        _export_fn(fn, tp.randn(shape))

    def test_adaptive_avg_pool_to_one_is_global(self):
        model = _export_fn(
            lambda x: F.adaptive_avg_pool2d(x, 1), tp.randn((2, 3, 8, 8))
        )
        assert "GlobalAveragePool" in _op_types(model)

    def test_adaptive_avg_pool_uses_average_pool(self):
        model = _export_fn(
            lambda x: F.adaptive_avg_pool2d(x, 2), tp.randn((2, 3, 8, 8))
        )
        assert "AveragePool" in _op_types(model)


class TestNormalization:
    def test_batch_norm_eval(self):
        class M(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.b = tp.nn.BatchNorm2d(3)
            def forward(self, x):
                return self.b(x)

        module = M()
        module.eval()
        model = _export(module, tp.randn((2, 3, 8, 8)))
        assert "BatchNormalization" in _op_types(model)

    def test_layer_norm(self):
        model = _export_fn(lambda x: F.layer_norm(x, (4,)), tp.randn((2, 4)))
        assert "LayerNormalization" in _op_types(model)

    def test_layer_norm_affine(self):
        _export_fn(
            lambda x: F.layer_norm(x, (4,), tp.randn((4,)), tp.randn((4,))),
            tp.randn((2, 4)),
        )

    def test_group_norm(self):
        _export_fn(lambda x: F.group_norm(x, 3), tp.randn((2, 3, 4, 4)))

    def test_group_norm_affine(self):
        _export_fn(
            lambda x: F.group_norm(x, 3, tp.randn((3,)), tp.randn((3,))),
            tp.randn((2, 3, 4, 4)),
        )

    def test_local_response_norm(self):
        model = _export_fn(
            lambda x: F.local_response_norm(x, 3), tp.randn((2, 5, 4, 4))
        )
        assert "LRN" in _op_types(model)

    def test_local_response_norm_even_window_is_rejected(self):
        program = export(
            _as_module(lambda x: F.local_response_norm(x, 2)),
            tp.randn((2, 5, 4, 4)),
        )
        with pytest.raises(UnsupportedOperatorError, match="odd window"):
            onnx_export(program)

    def test_instance_norm(self):
        model = _export_fn(lambda x: F.instance_norm(x), tp.randn((2, 3, 4, 4)))
        assert "InstanceNormalization" in _op_types(model)


class TestEmbedding:
    def test_embedding_functional(self):
        model = _export_fn(
            lambda x: F.embedding(x, tp.randn((10, 4))),
            tp.zeros((2, 3), dtype=tp.int64),
        )
        assert "Gather" in _op_types(model)

    def test_embedding_module(self):
        class M(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.e = tp.nn.Embedding(10, 4)
            def forward(self, x):
                return self.e(x)

        _export(M(), tp.zeros((2, 3), dtype=tp.int64))


class TestTensorOps:
    def test_cat(self):
        class M(tp.nn.Module):
            def forward(self, x, y):
                return tp.cat([x, y], dim=1)

        model = M()
        x = tp.randn((2, 3))
        y = tp.randn((2, 2))
        prog = export(model, x, y)
        onnx_model = onnx_export(prog)
        checker.check_model(onnx_model)

        tp_out = model(x, y).detach().numpy()
        ort_out = _run_onnx_model(onnx_model, [x.numpy(), y.numpy()])
        assert tp_out.shape == (2, 5)
        assert np.allclose(tp_out, ort_out, atol=1e-5)

    def test_reshape(self):
        class M(tp.nn.Module):
            def forward(self, x):
                return x.reshape(6)

        model = M()
        x = tp.ones((2, 3))
        prog = export(model, x)
        onnx_model = onnx_export(prog)
        checker.check_model(onnx_model)

        tp_out = model(x).detach().numpy()
        ort_out = _run_onnx_model(onnx_model, [x.numpy()])
        assert tp_out.shape == (6,)
        assert np.allclose(tp_out, ort_out, atol=1e-5)

    def test_transpose(self):
        class M(tp.nn.Module):
            def forward(self, x):
                return x.transpose(0, 1)

        model = M()
        x = tp.ones((2, 3))
        prog = export(model, x)
        onnx_model = onnx_export(prog)
        checker.check_model(onnx_model)

        tp_out = model(x).detach().numpy()
        ort_out = _run_onnx_model(onnx_model, [x.numpy()])
        assert tp_out.shape == (3, 2)
        assert np.allclose(tp_out, ort_out, atol=1e-5)

    def test_sum_keepdim(self):
        class M(tp.nn.Module):
            def forward(self, x):
                return x.sum(dim=1, keepdim=True)

        model = M()
        x = tp.ones((2, 3))
        prog = export(model, x)
        onnx_model = onnx_export(prog)
        checker.check_model(onnx_model)

        tp_out = model(x).detach().numpy()
        ort_out = _run_onnx_model(onnx_model, [x.numpy()])
        assert tp_out.shape == (2, 1)
        assert np.allclose(tp_out, ort_out, atol=1e-5)

    def test_expand(self):
        class M(tp.nn.Module):
            def forward(self, x):
                return x.expand(4, 3)

        model = M()
        x = tp.ones((1, 3))
        prog = export(model, x)
        onnx_model = onnx_export(prog)
        checker.check_model(onnx_model)

        tp_out = model(x).detach().numpy()
        ort_out = _run_onnx_model(onnx_model, [x.numpy()])
        assert tp_out.shape == (4, 3)
        assert np.allclose(tp_out, ort_out, atol=1e-5)

    @pytest.mark.parametrize(
        "fn,shape",
        [
            (lambda x: tp.flip(x, [0]), (2, 4)),
            (lambda x: tp.bmm(x, x), (2, 3, 3)),
            (lambda x: tp.cumsum(x, 0), (2, 4)),
            (lambda x: tp.tril(x), (4, 4)),
            (lambda x: x.t(), (2, 4)),
            (lambda x: x.repeat(2, 2), (2, 4)),
            (lambda x: x.permute([0, 3, 1, 2]), (2, 3, 4, 5)),
            (lambda x: x.flatten(), (2, 3, 4, 5)),
            (lambda x: x.flatten(1), (2, 3, 4, 5)),
            (lambda x: x.flatten(2), (2, 3, 4, 5)),
            (lambda x: x.flatten(1, 2), (2, 3, 4, 5)),
            (lambda x: x.unsqueeze(1), (2, 4)),
            (lambda x: x.unsqueeze(1).squeeze(1), (2, 4)),
            (lambda x: x.view(4, 2), (2, 4)),
            (lambda x: tp.stack([x, x], 0), (2, 4)),
            (lambda x: tp.split(x, 1, 0), (2, 4)),
            (lambda x: x.chunk(2, 0), (2, 4)),
            (lambda x: x.chunk(1, 0)[0], (2, 4)),
            (lambda x: tp.split(x, 2, 0)[0], (2, 4)),
            (lambda x: tp.narrow(x, 1, 1, 2), (2, 4)),
            (lambda x: x[0:1], (2, 4)),
            (lambda x: x[0], (2, 4)),
            (lambda x: x[-1], (2, 4)),
            (lambda x: x[:, ::2], (2, 4)),
            (lambda x: x.to(tp.float64), (2, 4)),
        ],
    )
    def test_shape_ops(self, fn, shape):
        _export_fn(fn, tp.randn(shape))

    def test_gather(self):
        _export_fn(
            lambda x: tp.gather(x, 1, tp.zeros((2, 1), dtype=tp.int64)),
            tp.randn((2, 4)),
        )

    def test_index_select(self):
        _export_fn(
            lambda x: tp.index_select(x, 0, tp.zeros((1,), dtype=tp.int64)),
            tp.randn((2, 4)),
        )

    def test_pixel_shuffle_roundtrip(self):
        _export_fn(lambda x: F.pixel_shuffle(x, 2), tp.randn((1, 4, 4, 4)))
        _export_fn(lambda x: F.pixel_unshuffle(x, 2), tp.randn((1, 4, 4, 4)))

    def test_one_hot(self):
        model = _export_fn(
            lambda x: F.one_hot(x, 10), tp.zeros((2, 3), dtype=tp.int64)
        )
        assert "OneHot" in _op_types(model)

    def test_one_hot_without_num_classes_is_rejected(self):
        program = export(
            _as_module(lambda x: F.one_hot(x)), tp.zeros((2,), dtype=tp.int64)
        )
        with pytest.raises(UnsupportedOperatorError, match="num_classes"):
            onnx_export(program)

    def test_pad(self):
        _export_fn(lambda x: F.pad(x, (1, 1, 1, 1)), tp.randn((2, 3, 4, 4)))

    def test_pad_reflect(self):
        _export_fn(
            lambda x: F.pad(x, (1, 1, 1, 1), mode="reflect"), tp.randn((2, 3, 4, 4))
        )

    def test_interpolate_scale_factor(self):
        model = _export_fn(
            lambda x: F.interpolate(x, scale_factor=2.0), tp.randn((2, 3, 4, 4))
        )
        assert "Resize" in _op_types(model)

    def test_interpolate_size_bilinear(self):
        _export_fn(
            lambda x: F.interpolate(x, size=(8, 8), mode="bilinear"),
            tp.randn((2, 3, 4, 4)),
        )


class TestReductions:
    @pytest.mark.parametrize(
        "fn",
        [
            lambda x: x.mean(),
            lambda x: tp.min(x),
            lambda x: tp.max(x, dim=1),
            lambda x: tp.prod(x, 1),
            lambda x: tp.argmax(x, 1),
            lambda x: tp.logsumexp(x, 1),
            lambda x: tp.norm(x),
            lambda x: tp.norm(x, 2.0, 1, True),
            lambda x: tp.norm(x, 1.0, 1, True),
            lambda x: tp.var(x, 1, 1, True),
            lambda x: tp.std(x, 1, 1, True),
            lambda x: (x > 0).all(),
            lambda x: (x > 0).any(),
            lambda x: tp.topk(x, 2),
            lambda x: tp.sort(x),
        ],
    )
    def test_reductions(self, fn):
        _export_fn(fn, tp.randn((3, 4)))

    def test_max_dim_returns_values_and_indices(self):
        model = _export_fn(lambda x: tp.max(x, dim=1), tp.randn((3, 4)))
        assert "ReduceMax" in _op_types(model)
        assert "ArgMax" in _op_types(model)
        assert len(model.graph.output) == 2

    def test_topk_indices_are_int64(self):
        model = _export_fn(lambda x: tp.topk(x, 2), tp.randn((3, 4)))
        assert model.graph.output[1].type.tensor_type.elem_type == TensorProto.INT64


class TestLosses:
    def test_mse_loss(self):
        _export_fn(lambda x: F.mse_loss(x, x * 2), tp.randn((2, 4)))

    def test_l1_loss(self):
        _export_fn(lambda x: F.l1_loss(x, x * 2), tp.randn((2, 4)))

    def test_cross_entropy(self):
        model = _export_fn(
            lambda x: F.cross_entropy(x, tp.zeros((2,), dtype=tp.int64)),
            tp.randn((2, 4)),
        )
        assert "SoftmaxCrossEntropyLoss" in _op_types(model)

    def test_nll_loss(self):
        model = _export_fn(
            lambda x: F.nll_loss(
                F.log_softmax(x, 1), tp.zeros((2,), dtype=tp.int64)
            ),
            tp.randn((2, 4)),
        )
        assert "NegativeLogLikelihoodLoss" in _op_types(model)

    def test_cross_entropy_label_smoothing_is_rejected(self):
        program = export(
            _as_module(
                lambda x: F.cross_entropy(
                    x, tp.zeros((2,), dtype=tp.int64), label_smoothing=0.1
                )
            ),
            tp.randn((2, 4)),
        )
        with pytest.raises(UnsupportedOperatorError, match="label_smoothing"):
            onnx_export(program)


class TestModels:
    def test_mlp(self):
        class MLP(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.l1 = tp.nn.Linear(8, 16)
                self.l2 = tp.nn.Linear(16, 4)
            def forward(self, x):
                return F.log_softmax(self.l2(F.relu(self.l1(x))), 1)

        _export(MLP(), tp.randn((3, 8)))

    def test_convnet(self):
        class Net(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.c1 = tp.nn.Conv2d(3, 8, 3, padding=1)
                self.b1 = tp.nn.BatchNorm2d(8)
                self.fc = tp.nn.Linear(8 * 4 * 4, 10)
            def forward(self, x):
                x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)
                return self.fc(x.flatten(1))

        module = Net()
        module.eval()
        _export(module, tp.randn((2, 3, 8, 8)))


class TestPasses:
    def test_constant_folding_folds_transposed_weight(self):
        class M(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = tp.nn.Linear(4, 2)
            def forward(self, x):
                return self.l(x)

        program = export(M(), tp.randn((2, 3, 4)))
        folded = onnx_export(program, do_constant_folding=True)
        unfolded = onnx_export(program, do_constant_folding=False)
        assert "Transpose" in _op_types(unfolded)
        assert "Transpose" not in _op_types(folded)

    def test_dead_nodes_are_removed(self):
        class M(tp.nn.Module):
            def forward(self, x):
                unused = x * 3
                return x + 1

        model = _export(M(), tp.randn((2, 3)))
        assert "Mul" not in _op_types(model)

    def test_output_names_do_not_leave_identity_nodes(self):
        class M(tp.nn.Module):
            def forward(self, x):
                return x + 1

        program = export(M(), tp.randn((2, 3)))
        model = onnx_export(program, input_names=["data"], output_names=["result"])
        assert [value.name for value in model.graph.input] == ["data"]
        assert [value.name for value in model.graph.output] == ["result"]
        assert "Identity" not in _op_types(model)

    def test_returning_an_input_unchanged_stays_valid(self):
        class M(tp.nn.Module):
            def forward(self, x):
                return x

        model = _export(M(), tp.randn((2, 3)))
        assert len(model.graph.output) == 1


class TestTypes:
    @pytest.mark.parametrize(
        "dtype,onnx_type",
        [
            (tp.float32, TensorProto.FLOAT),
            (tp.float64, TensorProto.DOUBLE),
            (tp.int32, TensorProto.INT32),
            (tp.int64, TensorProto.INT64),
            (tp.bool, TensorProto.BOOL),
        ],
    )
    def test_input_dtypes(self, dtype, onnx_type):
        class M(tp.nn.Module):
            def forward(self, x):
                return x.to(tp.float32) * 2

        x = tp.zeros((2, 3), dtype=dtype)
        program = export(M(), x)
        model = onnx_export(program)
        checker.check_model(model)
        assert model.graph.input[0].type.tensor_type.elem_type == onnx_type


class TestVerification:
    def test_verify_passes_for_a_correct_model(self):
        class M(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = tp.nn.Linear(4, 2)
            def forward(self, x):
                return self.l(x)

        program = export(M(), tp.randn((2, 4)))
        onnx_export(program, verify=True)

    def test_verify_reports_a_mismatch(self):
        from tensorplay.onnx._verify import verify_model

        class M(tp.nn.Module):
            def forward(self, x):
                return x + 1

        x = tp.randn((2, 3))
        program = export(M(), x)
        model = onnx_export(program)
        with pytest.raises(VerificationError):
            verify_model(
                model,
                expected=x,  # deliberately the wrong reference
                input_names=[value.name for value in model.graph.input],
                example_inputs=program.example_inputs,
            )


class TestOpsetAndDynamicShapes:
    def test_opset_version_is_honored(self):
        class M(tp.nn.Module):
            def forward(self, x):
                return x.sum(dim=1)

        program = export(M(), tp.randn((2, 3)))
        model = onnx_export(program, opset_version=17)
        checker.check_model(model)
        assert model.opset_import[0].version == 17

    def test_opset_below_minimum_is_rejected(self):
        class M(tp.nn.Module):
            def forward(self, x):
                return x + 1

        program = export(M(), tp.randn((2, 3)))
        with pytest.raises(ValueError, match="opset_version"):
            onnx_export(program, opset_version=9)

    def test_dynamic_axes_mark_the_batch_dimension(self):
        class M(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = tp.nn.Linear(4, 2)
            def forward(self, x):
                return self.l(x)

        program = export(M(), tp.randn((2, 4)))
        model = onnx_export(
            program, input_names=["x"], dynamic_axes={"x": {0: "batch"}}
        )
        checker.check_model(model)
        dim = model.graph.input[0].type.tensor_type.shape.dim[0]
        assert dim.dim_param == "batch"

        session = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        for batch in (1, 5):
            data = np.zeros((batch, 4), dtype=np.float32)
            assert session.run(None, {"x": data})[0].shape[0] == batch

    def test_dynamic_shapes_from_the_exported_program(self):
        class M(tp.nn.Module):
            def forward(self, x):
                return x + 1

        batch = Dim("batch")
        program = export(M(), tp.randn((2, 4)), dynamic_shapes={"x": {0: batch}})
        model = onnx_export(program)
        assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_param == "batch"


class TestUnsupported:
    def test_unknown_operator_names_the_target(self):
        def unsupported(value):
            return value

        class M(tp.nn.Module):
            def forward(self, x):
                return x.some_operator_that_does_not_exist()

        program = export(M(), tp.randn((2, 3)))
        with pytest.raises(UnsupportedOperatorError, match="some_operator"):
            onnx_export(program)


class TestSaveLoad:
    def test_save_to_file(self, tmp_path):
        class M(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = tp.nn.Linear(4, 2)
            def forward(self, x):
                return self.l(x)

        model = M()
        x = tp.ones((2, 4))
        prog = export(model, x)
        path = tmp_path / "model.onnx"
        onnx_export(prog, f=str(path))
        assert path.exists()
        assert path.stat().st_size > 0

        loaded = onnx.load(str(path))
        checker.check_model(loaded)

    def test_save_to_buffer(self):
        class M(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = tp.nn.Linear(4, 2)
            def forward(self, x):
                return self.l(x)

        model = M()
        x = tp.ones((2, 4))
        prog = export(model, x)
        buf = io.BytesIO()
        result = onnx_export(prog, f=buf)
        assert result is None
        assert len(buf.getvalue()) > 0

    def test_external_data_writes_a_side_car_file(self, tmp_path):
        class M(tp.nn.Module):
            def __init__(self):
                super().__init__()
                self.l = tp.nn.Linear(64, 64)
            def forward(self, x):
                return self.l(x)

        program = export(M(), tp.randn((2, 64)))
        path = tmp_path / "model.onnx"
        onnx_export(program, f=str(path), external_data=True)

        assert (tmp_path / "model.onnx.data").exists()
        assert path.stat().st_size < (tmp_path / "model.onnx.data").stat().st_size

        loaded = onnx.load(str(path))
        checker.check_model(loaded)
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        data = np.zeros((2, 64), dtype=np.float32)
        assert session.run(None, {session.get_inputs()[0].name: data})[0].shape == (2, 64)

    def test_external_data_requires_a_path(self):
        class M(tp.nn.Module):
            def forward(self, x):
                return x + 1

        program = export(M(), tp.randn((2, 3)))
        with pytest.raises(ValueError, match="file path"):
            onnx_export(program, f=io.BytesIO(), external_data=True)
