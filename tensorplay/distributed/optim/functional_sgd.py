import tensorplay as tp
from tensorplay.optim import sgd as F
from tensorplay import Tensor

from tensorplay.distributed.optim._deprecation_warning import (
    _scripted_functional_optimizer_deprecation_warning,
)


__all__: list[str] = []


# Define a Functional SGD Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
class _FunctionalSGD:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        maximize: bool = False,
        foreach: bool = False,
        fused: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        self.defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
        }
        self.nesterov = nesterov
        self.maximize = maximize
        self.foreach = foreach
        self.fused = fused
        self.state = {}

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

    def step_param(self, param: Tensor, grad: Tensor | None):
        """Similar to self.step, but operates on a single parameter and
        its gradient.
        """
        params_with_grad = []
        grads = []
        momentum_buffer_list: list[Tensor | None] = []
        if grad is not None:
            params_with_grad.append(param)
            grads.append(grad)
            if self.defaults["momentum"] != 0:
                if param not in self.state:
                    self.state[param] = {"momentum_buffer": None}
                momentum_buffer_list.append(self.state[param]["momentum_buffer"])

        with tp.no_grad():
            F.sgd(
                params_with_grad,
                grads,
                momentum_buffer_list,
                weight_decay=self.defaults["weight_decay"],
                momentum=self.defaults["momentum"],
                lr=self.defaults["lr"],
                dampening=self.defaults["dampening"],
                nesterov=self.nesterov,
                maximize=self.maximize,
                has_sparse_grad=False,
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )
        # update momentum_buffer in state
        state = self.state.get(param)
        if state is not None and self.defaults["momentum"] != 0:
            momentum_buffer = momentum_buffer_list[0]
            if momentum_buffer is not None:
                state["momentum_buffer"] = momentum_buffer

    def step(self, gradients: list[Tensor | None]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        momentum_buffer_list: list[Tensor | None] = []
        lr = self.defaults["lr"]
        weight_decay = self.defaults["weight_decay"]
        momentum = self.defaults["momentum"]
        dampening = self.defaults["dampening"]

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_sparse_grad = False
        for param, gradient in zip(params, gradients):
            if gradient is not None:
                params_with_grad.append(param)
                grads.append(gradient)
                if gradient.is_sparse:
                    has_sparse_grad = True

                if param not in self.state:
                    self.state[param] = {}

                state = self.state[param]
                if "momentum_buffer" not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state["momentum_buffer"])

        with tp.no_grad():
            F.sgd(
                params_with_grad,
                grads,
                momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=self.nesterov,
                maximize=self.maximize,
                has_sparse_grad=has_sparse_grad,
                foreach=self.foreach,
                fused=self.fused,
                grad_scale=None,
                found_inf=None,
            )

        # update momentum_buffers in state
        for i, p in enumerate(params_with_grad):
            state = self.state[p]
            state["momentum_buffer"] = momentum_buffer_list[i]


def _test_sgd():
    pass
