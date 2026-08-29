#
# DistributedOptimizer schedules local optimizers on the workers that own
# the parameters via RPC/RRef. tp ships no RPC runtime, so construction
# requires an initialized RPC context and reports its absence, matching
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

    This class requires the RPC framework.
    """

    def __init__(self, optimizer_class, params_rref, *args, **kwargs):
        _not_implemented()

    def step(self, *args, **kwargs):
        _not_implemented()

    def get_optim_rrefs(self):
        _not_implemented()
