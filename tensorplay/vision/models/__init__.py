"""Classification models, mirroring tensorplay.vision==0.28.0 models/__init__.py.

The detection / optical_flow / quantization / segmentation / video
subpackages are not ported; everything else follows tensorplay.vision.
"""

from .alexnet import *
from .convnext import *
from .densenet import *
from .efficientnet import *
from .googlenet import *
from .inception import *
from .mnasnet import *
from .mobilenetv2 import *
from .mobilenetv3 import *
from .regnet import *
from .resnet import *
from .shufflenetv2 import *
from .squeezenet import *
from .vgg import *
from .vision_transformer import *
from .swin_transformer import *
from .maxvit import *

# The Weights and WeightsEnum are developer-facing utils that we make public
# for downstream libs (torchvision/models/__init__.py).
from ._api import (
    get_model,
    get_model_builder,
    get_model_weights,
    get_weight,
    list_models,
    Weights,
    WeightsEnum,
)
