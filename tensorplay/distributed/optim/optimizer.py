# Ported from torch/distributed/optim/optimizer.py.
#
# DistributedOptimizer schedules local optimizers on the workers that own
# the parameters via RPC/RRef. tp ships no RPC runtime, so construction
# requires an initialized RPC context and reports its absence, matching
# torch's behavior when ``rpc.init_rpc`` has not been called.
import warnings

import tensorplay.distributed.rpc as rpc


__all__ = ["DistributedOptimizer"]


def _not_implemented():
    raise RuntimeError(
        "DistributedOptimizer requires tensorplay.distributed.rpc, which is "
        "not available in this build. Initialize RPC via rpc.init_rpc before "
        "using DistributedOptimizer."
    )


class DistributedOptimizer:
    r"""
    DistributedOptimizer takes remote references to parameters and runs the
    optimizer locally on the workers where the parameters live (torch parity).

    This class requires the RPC framework.
    """

    def __init__(self, optimizer_class, params_rref, *args, **kwargs):
        _not_implemented()

    def step(self, *args, **kwargs):
        _not_implemented()

    def get_optim_rrefs(self):
        _not_implemented()
