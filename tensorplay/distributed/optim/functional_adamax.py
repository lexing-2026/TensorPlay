import tensorplay as tp
from tensorplay.optim import adamax as F
from tensorplay import Tensor

from tensorplay.distributed.optim._deprecation_warning import (
    _scripted_functional_optimizer_deprecation_warning,
)


__all__: list[str] = []


# Define a Functional Adamax Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
class _FunctionalAdamax:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        foreach: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        _scripted_functional_optimizer_deprecation_warning(stacklevel=2)
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.defaults = {
            "lr": lr,
            "eps": eps,
            "beta1": betas[0],
            "beta2": betas[1],
            "weight_decay": weight_decay,
        }
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
        exp_avgs = []
        exp_infs = []
        state_steps: list[Tensor] = []

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
                    state["exp_avg"] = tp.zeros_like(param)
                    state["exp_inf"] = tp.zeros_like(param)

                state = self.state[param]
                exp_avgs.append(state["exp_avg"])
                exp_infs.append(state["exp_inf"])
                state_steps.append(state["step"])

        with tp.no_grad():
            F.adamax(
                params_with_grad,
                grads,
                exp_avgs,
                exp_infs,
                state_steps,
                eps=self.defaults["eps"],
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
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
        exp_avgs = []
        exp_infs = []
        state_steps: list[Tensor] = []
        has_complex = param.is_complex()
        if grad is not None:
            params_with_grad.append(param)
            grads.append(grad)
        if param not in self.state:
            self.state[param] = {}
            state = self.state[param]
            state["step"] = tp.tensor(0.0)
            state["exp_avg"] = tp.zeros_like(param)
            state["exp_inf"] = tp.zeros_like(param)

        state = self.state[param]
        exp_avgs.append(state["exp_avg"])
        exp_infs.append(state["exp_inf"])
        state_steps.append(state["step"])
        with tp.no_grad():
            F.adamax(
                params_with_grad,
                grads,
                exp_avgs,
                exp_infs,
                state_steps,
                eps=self.defaults["eps"],
                beta1=self.defaults["beta1"],
                beta2=self.defaults["beta2"],
                lr=self.defaults["lr"],
                weight_decay=self.defaults["weight_decay"],
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
            )
