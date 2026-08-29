import tensorplay as tp
from tensorplay.optim import rmsprop as F
from tensorplay import Tensor

from tensorplay.distributed.optim._deprecation_warning import (
    _scripted_functional_optimizer_deprecation_warning,
)


__all__: list[str] = []


# Define a Functional RMSprop Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
class _FunctionalRMSprop:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        foreach: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        self.defaults = {
            "lr": lr,
            "alpha": alpha,
            "eps": eps,
            "weight_decay": weight_decay,
            "momentum": momentum,
        }
        self.centered = centered
        self.foreach = foreach
        self.maximize = maximize
        self.state = {}

        if len(params) == 0 and not _allow_empty_param_list:
            raise ValueError("optimizer got an empty parameter list")

        # NOTE: we only have one param_group and don't allow user to add additional
        # param group as it's not a common use case.
        self.param_group = {"params": params}

    def step(self, gradients: list[Tensor | None]):
        params = self.param_group["params"]
        params_with_grad = []
        grads = []
        square_avgs = []
        grad_avgs = []
        momentum_buffer_list = []
        state_steps: list[Tensor] = []
        lr = self.defaults["lr"]
        alpha = self.defaults["alpha"]
        eps = self.defaults["eps"]

        if len(params) != len(gradients):
            raise ValueError(
                "the gradients passed in does not equal to the size of the parameters!"
                + f"Params length: {len(params)}. "
                + f"Gradients length: {len(gradients)}"
            )

        has_complex = False
        for param, gradient in zip(params, gradients):
            if gradient is not None:
                has_complex |= param.is_complex()
                params_with_grad.append(param)
                grads.append(gradient)
                if param not in self.state:
                    self.state[param] = {}
                    state = self.state[param]
                    state["step"] = tp.tensor(0.0)
                    state["square_avg"] = tp.zeros_like(param)
                    if self.defaults["momentum"] > 0:
                        state["momentum_buffer"] = tp.zeros_like(param)
                    if self.centered:
                        state["grad_avg"] = tp.zeros_like(param)

                state = self.state[param]
                square_avgs.append(state["square_avg"])
                if self.defaults["momentum"] > 0:
                    momentum_buffer_list.append(state["momentum_buffer"])
                if self.centered:
                    grad_avgs.append(state["grad_avg"])
                state_steps.append(state["step"])

        with tp.no_grad():
            F.rmsprop(
                params_with_grad,
                grads,
                square_avgs,
                grad_avgs,
                momentum_buffer_list,
                state_steps,
                lr=lr,
                alpha=alpha,
                eps=eps,
                weight_decay=self.defaults["weight_decay"],
                momentum=self.defaults["momentum"],
                centered=self.centered,
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
            )

    def step_param(self, param: Tensor, grad: Tensor | None):
        """
        Similar to step, but operates on a single parameter and optionally a
        gradient tensor.
        """
        params_with_grad = []
        grads = []
        square_avgs = []
        grad_avgs = []
        momentum_buffer_list = []
        state_steps: list[Tensor] = []
        has_complex = param.is_complex()
        if grad is not None:
            params_with_grad.append(param)
            grads.append(grad)
            if param not in self.state:
                self.state[param] = {}
                state = self.state[param]
                state["step"] = tp.tensor(0.0)
                state["square_avg"] = tp.zeros_like(param)
                if self.defaults["momentum"] > 0:
                    state["momentum_buffer"] = tp.zeros_like(param)
                if self.centered:
                    state["grad_avg"] = tp.zeros_like(param)

            state = self.state[param]
            square_avgs.append(state["square_avg"])
            if self.defaults["momentum"] > 0:
                momentum_buffer_list.append(state["momentum_buffer"])
            if self.centered:
                grad_avgs.append(state["grad_avg"])
            state_steps.append(state["step"])

        with tp.no_grad():
            F.rmsprop(
                params_with_grad,
                grads,
                square_avgs,
                grad_avgs,
                momentum_buffer_list,
                state_steps,
                lr=self.defaults["lr"],
                alpha=self.defaults["alpha"],
                eps=self.defaults["eps"],
                weight_decay=self.defaults["weight_decay"],
                momentum=self.defaults["momentum"],
                centered=self.centered,
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
            )
