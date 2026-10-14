"""Weight extraction helpers for the FX numeric suite."""

import tensorplay as tp
from tensorplay import nn

__all__ = [
    "mod_weight_detach",
    "get_conv_mod_weight",
    "get_linear_mod_weight",
    "get_op_to_type_to_weight_extraction_fn",
    "extract_weight_from_node",
]


def mod_weight_detach(mod: nn.Module) -> tp.Tensor:
    return mod.weight.detach()


def mod_0_weight_detach(mod: nn.Module) -> tp.Tensor:
    return mod[0].weight.detach()


def mod_weight_bias_0(mod: nn.Module) -> tp.Tensor:
    return mod[0].weight.detach()


def get_lstm_weight(mod: nn.Module):
    return [mod.weight_ih_l0.detach(), mod.weight_hh_l0.detach()]


def get_qlstm_weight(mod: nn.Module):
    return [mod.weight_ih_l0.dequantize(), mod.weight_hh_l0.dequantize()]


def get_conv_mod_weight(mod: nn.Module) -> tp.Tensor:
    return mod_weight_detach(mod)


def get_linear_mod_weight(mod: nn.Module) -> tp.Tensor:
    return mod_weight_detach(mod)


def get_lstm_mod_weights(mod: nn.Module):
    return get_lstm_weight(mod)


def get_conv_fun_weight(node, gm) -> tp.Tensor:
    return node.args[1].detach()


def get_qconv_fun_weight(node, gm) -> tp.Tensor:
    return node.args[1].dequantize()


def get_linear_fun_weight(node, gm) -> tp.Tensor:
    return node.args[1].detach()


def get_qlinear_fun_weight(node, gm) -> tp.Tensor:
    return node.args[1].dequantize()


def get_op_to_type_to_weight_extraction_fn():
    """Return a mapping from op to a dict of target type to weight extractor."""
    return {
        "call_module": {
            nn.Conv1d: get_conv_mod_weight,
            nn.Conv2d: get_conv_mod_weight,
            nn.Conv3d: get_conv_mod_weight,
            nn.Linear: get_linear_mod_weight,
            nn.LSTM: get_lstm_mod_weights,
        },
        "call_function": {},
    }


def extract_weight_from_node(node, gm):
    """Extract the weight tensor(s) of a node."""
    if node.op == "call_module":
        module = gm.get_submodule(node.target)
        if hasattr(module, "weight"):
            weight = module.weight
            if weight.is_quantized():
                return weight.dequantize()
            return weight.detach()
        if hasattr(module, "weight_ih_l0"):
            return [module.weight_ih_l0.detach(), module.weight_hh_l0.detach()]
    if node.op == "call_function":
        for arg in node.args:
            if hasattr(arg, "detach") and getattr(arg, "dim", lambda: None)() is not None:
                return arg.detach()
    return None