"""Utility functions for APoT non-uniform quantization."""

import math


def float_to_apot(x, levels, indices, alpha):
    """Map a float to the nearest APoT level index, clipping to ``alpha``."""
    if x < -alpha:
        return -alpha
    elif x > alpha:
        return alpha

    min_delta = math.inf
    best_idx = 0
    for level, idx in zip(levels.tolist(), indices.tolist()):
        cur_delta = abs(level - x)
        if cur_delta < min_delta:
            min_delta = cur_delta
            best_idx = idx
    return best_idx


def quant_dequant_util(x, levels):
    """Return the nearest APoT level in float."""
    min_delta = math.inf
    best_fp = 0.0
    for level in levels.tolist():
        cur_delta = abs(level - x)
        if cur_delta < min_delta:
            min_delta = cur_delta
            best_fp = level
    return best_fp


def apot_to_float(x_apot, levels, indices):
    """Map an APoT index back to its float level."""
    idx = indices.tolist().index(int(x_apot))
    return levels[idx]