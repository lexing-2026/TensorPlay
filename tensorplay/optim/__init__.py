from .optimizer import Optimizer
from .sgd import SGD
from .adam import Adam
from .adamw import AdamW
from .rmsprop import RMSprop
from .adagrad import Adagrad
from .adadelta import Adadelta
from .adamax import Adamax
from .asgd import ASGD
from ._adafactor import Adafactor
from .lbfgs import LBFGS
from ._muon import Muon
from .nadam import NAdam
from .radam import RAdam
from .rprop import Rprop
from .sparse_adam import SparseAdam
from . import lr_scheduler
from . import swa_utils
from ._stateless import swap_in_optimizer_params_and_state

Adafactor.__module__ = "tensorplay.optim"
Muon.__module__ = "tensorplay.optim"
swap_in_optimizer_params_and_state.__module__ = "tensorplay.optim"

__all__ = [
    'Optimizer',
    'SGD',
    'Adam',
    'AdamW',
    'RMSprop',
    'Adagrad',
    'Adadelta',
    'Adamax',
    'ASGD',
    'Adafactor',
    'LBFGS',
    'Muon',
    'NAdam',
    'RAdam',
    'Rprop',
    'SparseAdam',
    'lr_scheduler',
    'swa_utils',
    'swap_in_optimizer_params_and_state',
]
