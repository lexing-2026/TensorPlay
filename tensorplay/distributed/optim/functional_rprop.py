import tensorplay as tp
from tensorplay.optim import rprop as F
from tensorplay import Tensor

__all__: list[str] = []


# Define a Functional Rprop Optimizer
# where we use these optimizer in a functional way.
# Instead of using the `param.grad` when updating parameters,
# we explicitly allow the distributed optimizer pass gradients to
# the `step` function. In this way, we could separate the gradients
# and parameters and allow multithreaded trainer to update the
# parameters without data traces on accumulating to the same .grad.
# NOTE: This should be only used by distributed optimizer internals
# and not meant to expose to the user.
class _FunctionalRprop:
    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-2,
        etas: tuple[float, float] = (0.5, 1.2),
        step_sizes: tuple[float, float] = (1e-6, 50),
        foreach: bool = False,
        maximize: bool = False,
        _allow_empty_param_list: bool = False,
    ):
        self.defaults = {
            "lr": lr,
            "step_size_min": step_sizes[0],
            "step_size_max": step_sizes[1],
            "etaminus": etas[0],
            "etaplus": etas[1],
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
        prevs = []
        step_sizes_local = []
        state_steps: list[Tensor] = []
        lr = self.defaults["lr"]

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
                    state["prev"] = tp.zeros_like(param)
                    state["step_size"] = tp.full_like(gradient, lr)

                state = self.state[param]
                prevs.append(state["prev"])
                step_sizes_local.append(state["step_size"])
                state_steps.append(state["step"])

        with tp.no_grad():
            F.rprop(
                params_with_grad,
                grads,
                prevs,
                step_sizes_local,
                state_steps,
                step_size_min=self.defaults["step_size_min"],
                step_size_max=self.defaults["step_size_max"],
                etaminus=self.defaults["etaminus"],
                etaplus=self.defaults["etaplus"],
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
        prevs = []
        step_sizes_local = []
        state_steps: list[Tensor] = []
        has_complex = param.is_complex()
        if grad is not None:
            params_with_grad.append(param)
            grads.append(grad)
            if param not in self.state:
                self.state[param] = {}
                state = self.state[param]
                state["step"] = tp.tensor(0.0)
                state["prev"] = tp.zeros_like(param)
                state["step_size"] = tp.full_like(gradient, self.defaults["lr"])

            state = self.state[param]
            prevs.append(state["prev"])
            step_sizes_local.append(state["step_size"])
            state_steps.append(state["step"])

        with tp.no_grad():
            F.rprop(
                params_with_grad,
                grads,
                prevs,
                step_sizes_local,
                state_steps,
                step_size_min=self.defaults["step_size_min"],
                step_size_max=self.defaults["step_size_max"],
                etaminus=self.defaults["etaminus"],
                etaplus=self.defaults["etaplus"],
                foreach=self.foreach,
                maximize=self.maximize,
                has_complex=has_complex,
            )
