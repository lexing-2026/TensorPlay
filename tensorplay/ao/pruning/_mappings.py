"""Sparse module mappings."""

import tensorplay as tp
import tensorplay.ao.nn.sparse


def get_static_sparse_quantized_mapping():
    return {tp.nn.Linear: tensorplay.ao.nn.sparse.quantized.Linear}


def get_dynamic_sparse_quantized_mapping():
    return {tp.nn.Linear: tensorplay.ao.nn.sparse.quantized.dynamic.Linear}