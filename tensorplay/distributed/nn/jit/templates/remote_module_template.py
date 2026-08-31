from __future__ import annotations

_TEMPLATE_PREFIX = """from tensorplay.distributed import rpc
from tensorplay.distributed.nn.api.remote_module import _remote_forward


def forward_async(self, {arg_types}):
    call_args = {args}
    call_kwargs = {kwargs}
    return rpc.rpc_async(
        self.module_rref.owner(),
        _remote_forward,
        args=(self.module_rref, self.device, self.is_device_map_set, call_args, call_kwargs),
    )


def forward(self, {arg_types}):
    call_args = {args}
    call_kwargs = {kwargs}
    return forward_async(self, *call_args, **call_kwargs).wait()


_generated_methods = [forward_async, forward]
"""


def get_remote_module_template(enable_moving_cpu_tensors_to_cuda: bool = False) -> str:
    del enable_moving_cpu_tensors_to_cuda
    return _TEMPLATE_PREFIX
