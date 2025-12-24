"""
TensorPlay - 一个用于深度学习验证的工具包

版本: 0.1.2
作者: Welog
日期: 2025年9月3日

功能特点:
- 提供多阶自动微分处理能力
- 提供计算图可视化功能
- 支持多维度的模型组件管理
- 支持JSON格式保存和加载
- 支持模型结构打印
- 支持钩子调试
"""
__version__ = "0.1.1"
__author__ = "Welog"
__email__ = "2095774200@shu.edu.cn"
__description__ = "一个用于深度学习验证的工具包"
__url__ = "https://github.com/bluemoon-o2/TensorPlay"
__license__ = "MIT"


# =============================================================================
# 全局接口
# =============================================================================
from .core import (config, no_grad, to_data, Tensor, Layer, Operator, Optimizer)
from .layer import (Dense, BatchNorm, LayerNorm, Conv2D)
from .module import (Module, Sequential)
from .optimizer import (SGD, Adam, Momentum, AdamW, Nadam, Lookahead, RMSprop)
from .operator import (concatenate, load_operator)
from .func import (mse, sse, nll, cross_entropy, sphere, goldstein)
from .initializer import (he_init, xavier_init, uniform_init, my_init)
from .utils import (accuracy)
from .data import (DataLoader, load_moons, load_iris)
from .scheduler import (StepLR, MultiStepLR, ExponentialLR, EarlyStopping)


__all__ = [
    __version__,
    Dense, BatchNorm, LayerNorm, Conv2D,
    Module, Sequential,
    SGD, Adam, Momentum, AdamW, Nadam, Lookahead, RMSprop,
    concatenate,
    mse, sse, nll, cross_entropy, sphere, goldstein,
    he_init, xavier_init, uniform_init, my_init,
    accuracy,
    DataLoader, load_moons, load_iris,
    StepLR, MultiStepLR, ExponentialLR, EarlyStopping,
]


load_operator()