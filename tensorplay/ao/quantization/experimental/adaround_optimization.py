"""Adaptive rounding optimization loop."""

import tensorplay as tp
from tensorplay import nn

from .adaround_fake_quantize import AdaroundFakeQuantizer
from .adaround_loss import AdaptiveRoundingLoss


class AdaptiveRoundingOptimization:
    """Optimize the soft-rounding parameters of an AdaRound fake quantizer."""

    def __init__(
        self,
        model: nn.Module,
        max_iter: int = 10000,
        warm_start: float = 0.2,
        beta_range: tuple[int, int] = (20, 2),
        reg_param: float = 0.001,
        lr: float = 0.01,
    ) -> None:
        self.model = model
        self.max_iter = max_iter
        self.lr = lr
        self.loss = AdaptiveRoundingLoss(max_iter, warm_start, beta_range, reg_param)

    def _find_adaround_modules(self):
        modules = []
        for module in self.model.modules():
            if isinstance(module, AdaroundFakeQuantizer):
                modules.append(module)
        return modules

    def optimize(self, data_loader, num_batches: int = 10):
        """Run the adaptive rounding optimization over the given data."""
        adaround_modules = self._find_adaround_modules()
        if not adaround_modules:
            raise ValueError("No AdaroundFakeQuantizer modules found in the model")
        params = [module.V for module in adaround_modules]
        optimizer = tp.optim.Adam(params, lr=self.lr)
        iter_count = 0
        for module in adaround_modules:
            module.enable_weight_fake_quant()
        for batch_idx, (input, _target) in enumerate(data_loader):
            if batch_idx >= num_batches:
                break
            for _inner in range(max(1, self.max_iter // num_batches)):
                optimizer.zero_grad()
                output = self.model(input)
                recon_loss = tp.tensor(0.0)
                for module in adaround_modules:
                    h_v = module.get_rectified_sigmoid_func()
                    x = input
                    x_hat = module(x)
                    recon_loss = recon_loss + self.loss.reconstruction_loss(
                        x.detach(), x_hat
                    )
                loss = recon_loss
                loss.backward()
                optimizer.step()
                iter_count += 1
        return self.model