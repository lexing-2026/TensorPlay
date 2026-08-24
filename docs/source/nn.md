```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# tensorplay.nn

## Containers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.module.Module
    tensorplay.nn.modules.container.Sequential
    tensorplay.nn.modules.container.ModuleList
    tensorplay.nn.modules.container.ModuleDict
    tensorplay.nn.modules.container.ParameterList
    tensorplay.nn.modules.container.ParameterDict
    tensorplay.nn.modules.module.register_module_forward_pre_hook
    tensorplay.nn.modules.module.register_module_forward_hook
    tensorplay.nn.modules.module.register_module_backward_hook
    tensorplay.nn.modules.module.register_module_full_backward_pre_hook
    tensorplay.nn.modules.module.register_module_full_backward_hook
    tensorplay.nn.modules.module.register_module_buffer_registration_hook
    tensorplay.nn.modules.module.register_module_module_registration_hook
    tensorplay.nn.modules.module.register_module_parameter_registration_hook
```

## Convolution Layers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.conv.Conv1d
    tensorplay.nn.modules.conv.Conv2d
    tensorplay.nn.modules.conv.Conv3d
    tensorplay.nn.modules.conv.ConvTranspose1d
    tensorplay.nn.modules.conv.ConvTranspose2d
    tensorplay.nn.modules.conv.ConvTranspose3d
    tensorplay.nn.modules.conv.LazyConv1d
    tensorplay.nn.modules.conv.LazyConv2d
    tensorplay.nn.modules.conv.LazyConv3d
    tensorplay.nn.modules.conv.LazyConvTranspose1d
    tensorplay.nn.modules.conv.LazyConvTranspose2d
    tensorplay.nn.modules.conv.LazyConvTranspose3d
    tensorplay.nn.modules.folding.Unfold
    tensorplay.nn.modules.folding.Fold
```

## Pooling layers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.pooling.MaxPool1d
    tensorplay.nn.modules.pooling.MaxPool2d
    tensorplay.nn.modules.pooling.MaxPool3d
    tensorplay.nn.modules.pooling.MaxUnpool1d
    tensorplay.nn.modules.pooling.MaxUnpool2d
    tensorplay.nn.modules.pooling.MaxUnpool3d
    tensorplay.nn.modules.pooling.AvgPool1d
    tensorplay.nn.modules.pooling.AvgPool2d
    tensorplay.nn.modules.pooling.AvgPool3d
    tensorplay.nn.modules.pooling.FractionalMaxPool2d
    tensorplay.nn.modules.pooling.FractionalMaxPool3d
    tensorplay.nn.modules.pooling.LPPool1d
    tensorplay.nn.modules.pooling.LPPool2d
    tensorplay.nn.modules.pooling.LPPool3d
    tensorplay.nn.modules.pooling.AdaptiveMaxPool1d
    tensorplay.nn.modules.pooling.AdaptiveMaxPool2d
    tensorplay.nn.modules.pooling.AdaptiveMaxPool3d
    tensorplay.nn.modules.pooling.AdaptiveAvgPool1d
    tensorplay.nn.modules.pooling.AdaptiveAvgPool2d
    tensorplay.nn.modules.pooling.AdaptiveAvgPool3d
```

## Padding Layers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.padding.ReflectionPad1d
    tensorplay.nn.modules.padding.ReflectionPad2d
    tensorplay.nn.modules.padding.ReflectionPad3d
    tensorplay.nn.modules.padding.ReplicationPad1d
    tensorplay.nn.modules.padding.ReplicationPad2d
    tensorplay.nn.modules.padding.ReplicationPad3d
    tensorplay.nn.modules.padding.ZeroPad1d
    tensorplay.nn.modules.padding.ZeroPad2d
    tensorplay.nn.modules.padding.ZeroPad3d
    tensorplay.nn.modules.padding.ConstantPad1d
    tensorplay.nn.modules.padding.ConstantPad2d
    tensorplay.nn.modules.padding.ConstantPad3d
    tensorplay.nn.modules.padding.CircularPad1d
    tensorplay.nn.modules.padding.CircularPad2d
    tensorplay.nn.modules.padding.CircularPad3d
```

## Non-linear Activations (weighted sum, nonlinearity)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.activation.ELU
    tensorplay.nn.modules.activation.Hardshrink
    tensorplay.nn.modules.activation.Hardsigmoid
    tensorplay.nn.modules.activation.Hardtanh
    tensorplay.nn.modules.activation.Hardswish
    tensorplay.nn.modules.activation.LeakyReLU
    tensorplay.nn.modules.activation.LogSigmoid
    tensorplay.nn.modules.multihead_attention.MultiheadAttention
    tensorplay.nn.modules.activation.PReLU
    tensorplay.nn.modules.activation.ReLU
    tensorplay.nn.modules.activation.ReLU6
    tensorplay.nn.modules.activation.RReLU
    tensorplay.nn.modules.activation.SELU
    tensorplay.nn.modules.activation.CELU
    tensorplay.nn.modules.activation.Sigmoid
    tensorplay.nn.modules.activation.SiLU
    tensorplay.nn.modules.activation.Mish
    tensorplay.nn.modules.activation.Softplus
    tensorplay.nn.modules.activation.Softshrink
    tensorplay.nn.modules.activation.Softsign
    tensorplay.nn.modules.activation.Tanh
    tensorplay.nn.modules.activation.Tanhshrink
    tensorplay.nn.modules.activation.Threshold
    tensorplay.nn.modules.activation.GLU
```

## Non-linear Activations (other)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.activation.Softmin
    tensorplay.nn.modules.activation.Softmax
    tensorplay.nn.modules.activation.LogSoftmax
    tensorplay.nn.modules.adaptive.AdaptiveLogSoftmaxWithLoss
```

## Normalization Layers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.batchnorm.BatchNorm1d
    tensorplay.nn.modules.batchnorm.BatchNorm2d
    tensorplay.nn.modules.batchnorm.BatchNorm3d
    tensorplay.nn.modules.batchnorm.LazyBatchNorm1d
    tensorplay.nn.modules.batchnorm.LazyBatchNorm2d
    tensorplay.nn.modules.batchnorm.LazyBatchNorm3d
    tensorplay.nn.modules.normalization.GroupNorm
    tensorplay.nn.modules.batchnorm.SyncBatchNorm
    tensorplay.nn.modules.instancenorm.InstanceNorm1d
    tensorplay.nn.modules.instancenorm.InstanceNorm2d
    tensorplay.nn.modules.instancenorm.InstanceNorm3d
    tensorplay.nn.modules.instancenorm.LazyInstanceNorm1d
    tensorplay.nn.modules.instancenorm.LazyInstanceNorm2d
    tensorplay.nn.modules.instancenorm.LazyInstanceNorm3d
    tensorplay.nn.modules.normalization.LayerNorm
    tensorplay.nn.modules.normalization.LocalResponseNorm
    tensorplay.nn.modules.normalization.RMSNorm
```

## Recurrent Layers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.rnn.RNNBase
    tensorplay.nn.modules.rnn.RNN
    tensorplay.nn.modules.rnn.LSTM
    tensorplay.nn.modules.rnn.GRU
    tensorplay.nn.modules.rnn.RNNCell
    tensorplay.nn.modules.rnn.LSTMCell
    tensorplay.nn.modules.rnn.GRUCell
```

## Transformer Layers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.transformer.Transformer
    tensorplay.nn.modules.transformer.TransformerEncoder
    tensorplay.nn.modules.transformer.TransformerDecoder
    tensorplay.nn.modules.transformer.TransformerEncoderLayer
    tensorplay.nn.modules.transformer.TransformerDecoderLayer
```

## Linear Layers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.linear.Identity
    tensorplay.nn.modules.linear.Linear
    tensorplay.nn.modules.linear.Bilinear
    tensorplay.nn.modules.lazy.LazyLinear
```

## Dropout Layers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.dropout.Dropout
    tensorplay.nn.modules.dropout.Dropout1d
    tensorplay.nn.modules.dropout.Dropout2d
    tensorplay.nn.modules.dropout.Dropout3d
    tensorplay.nn.modules.dropout.AlphaDropout
    tensorplay.nn.modules.dropout.FeatureAlphaDropout
```

## Sparse Layers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.sparse.Embedding
    tensorplay.nn.modules.sparse.EmbeddingBag
```

## Distance Functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.distance.CosineSimilarity
    tensorplay.nn.modules.distance.PairwiseDistance
```

## Loss Functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.loss.L1Loss
    tensorplay.nn.modules.loss.MSELoss
    tensorplay.nn.modules.loss.CrossEntropyLoss
    tensorplay.nn.modules.loss.CTCLoss
    tensorplay.nn.modules.loss.NLLLoss
    tensorplay.nn.modules.loss.PoissonNLLLoss
    tensorplay.nn.modules.loss.GaussianNLLLoss
    tensorplay.nn.modules.loss.KLDivLoss
    tensorplay.nn.modules.loss.BCELoss
    tensorplay.nn.modules.loss.BCEWithLogitsLoss
    tensorplay.nn.modules.loss.MarginRankingLoss
    tensorplay.nn.modules.loss.HingeEmbeddingLoss
    tensorplay.nn.modules.loss.MultiLabelMarginLoss
    tensorplay.nn.modules.loss.HuberLoss
    tensorplay.nn.modules.loss.SmoothL1Loss
    tensorplay.nn.modules.loss.SoftMarginLoss
    tensorplay.nn.modules.loss.MultiLabelSoftMarginLoss
    tensorplay.nn.modules.loss.CosineEmbeddingLoss
    tensorplay.nn.modules.loss.MultiMarginLoss
    tensorplay.nn.modules.loss.TripletMarginLoss
    tensorplay.nn.modules.loss.TripletMarginWithDistanceLoss
```

## Vision Layers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.pixelshuffle.PixelShuffle
    tensorplay.nn.modules.pixelshuffle.PixelUnshuffle
    tensorplay.nn.modules.upsampling.Upsample
    tensorplay.nn.modules.upsampling.UpsamplingNearest2d
    tensorplay.nn.modules.upsampling.UpsamplingBilinear2d
```

## Shuffle Layers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.channelshuffle.ChannelShuffle
```

## DataParallel Layers (multi-GPU, distributed)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.parallel.data_parallel.DataParallel
    tensorplay.nn.parallel.distributed.DistributedDataParallel
```

## Utilities

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.utils.rnn.PackedSequence
    tensorplay.nn.utils.rnn.pack_padded_sequence
    tensorplay.nn.utils.rnn.pad_packed_sequence
    tensorplay.nn.utils.rnn.pad_sequence
    tensorplay.nn.utils.rnn.pack_sequence
    tensorplay.nn.utils.rnn.unpack_sequence
    tensorplay.nn.utils.rnn.unpad_sequence
    tensorplay.nn.utils.rnn.invert_permutation
    tensorplay.nn.parameter.is_lazy
    tensorplay.nn.factory_kwargs
    tensorplay.nn.modules.flatten.Flatten
    tensorplay.nn.modules.flatten.Unflatten
```

## Lazy Modules Initialization

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.lazy.LazyModuleMixin
```

## TensorPlay-specific additions

```{eval-rst}
.. currentmodule:: tensorplay.nn
```
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    Buffer
    DepthwiseConv2d
    NonDynamicallyQuantizableLinear
    Parameter
    RNNCellBase
    UninitializedBuffer
    UninitializedParameter
```

