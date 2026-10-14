#!/usr/bin/env python3

import tensorplay.nn as nn
from tensorplay import Tensor


class Addition_Module(nn.Module):
    """Custom addition module that uses multiple inputs to assure correct relevance
    propagation. Any addition in a forward function needs to be replaced with the
    module before using LRP."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return x1 + x2
