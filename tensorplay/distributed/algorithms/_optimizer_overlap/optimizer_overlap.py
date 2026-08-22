# Ported from torch/distributed/algorithms/_optimizer_overlap/optimizer_overlap.py.
#
# torch implements this over fused C++ optimizers registered with the DDP
# reducer; tp's reducer runs Python hooks, so overlap degrades to hook-time
# execution via register_comm_hook.
from typing import Any


__all__ = ["OptimizerOverlap", "as_overlapped_optim"]


class OptimizerOverlap:
    def register_ddp(self, ddp) -> None:
        raise NotImplementedError(
            f"{type(self)} does not support overlapped DDP in tp."
        )


class _OverlappedOptim(OptimizerOverlap):
    def __init__(self, optim_cls, optim_params, *args, **kwargs) -> None:
        self.optim_cls = optim_cls
        self.optim_params = optim_params
        self.args = args
        self.kwargs = kwargs

    def register_ddp(self, ddp) -> None:
        def hook(_state, bucket):
            # Run the optimizer step for this bucket's parameters right after
            # its gradient reduction completes.
            params = [p for p in bucket.parameters() if p.grad is not None]
            opt = getattr(ddp, "_overlap_optim", None)
            if opt is None:
                pset = set(id(p) for p in (
                    self.optim_params or ddp.module.parameters()))
                params = [p for p in params if id(p) in pset] or params
                opt = self.optim_cls(params, *self.args, **self.kwargs)
                ddp._overlap_optim = opt
            else:
                opt.param_groups[0]["params"] = [
                    p for p in opt.param_groups[0]["params"] if p.grad is not None
                ]
            opt.step()

        ddp.register_comm_hook(None, hook)


def as_overlapped_optim(optim: Any, optim_params: Any, *args: Any, **kwargs: Any):
    return _OverlappedOptim(optim, optim_params, *args, **kwargs)
