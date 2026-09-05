```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# tensorplay.nn.functional

## Convolution functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.functional.conv1d
    tensorplay.nn.functional.conv2d
    tensorplay.nn.functional.conv3d
    tensorplay.nn.functional.conv_transpose1d
    tensorplay.nn.functional.conv_transpose2d
    tensorplay.nn.functional.conv_transpose3d
    tensorplay.nn.functional.unfold
    tensorplay.nn.functional.fold
```

## Pooling functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.functional.avg_pool1d
    tensorplay.nn.functional.avg_pool2d
    tensorplay.nn.functional.avg_pool3d
    tensorplay.nn.functional.max_pool1d
    tensorplay.nn.functional.max_pool2d
    tensorplay.nn.functional.max_pool3d
    tensorplay.nn.functional.max_unpool1d
    tensorplay.nn.functional.max_unpool2d
    tensorplay.nn.functional.max_unpool3d
    tensorplay.nn.functional.lp_pool1d
    tensorplay.nn.functional.lp_pool2d
    tensorplay.nn.functional.lp_pool3d
    tensorplay.nn.functional.adaptive_max_pool1d
    tensorplay.nn.functional.adaptive_max_pool2d
    tensorplay.nn.functional.adaptive_max_pool3d
    tensorplay.nn.functional.adaptive_avg_pool1d
    tensorplay.nn.functional.adaptive_avg_pool2d
    tensorplay.nn.functional.adaptive_avg_pool3d
    tensorplay.nn.functional.fractional_max_pool2d
    tensorplay.nn.functional.fractional_max_pool3d
```

## Attention Mechanisms

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.functional.scaled_dot_product_attention
```

## Non-linear activation functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.functional.threshold
    tensorplay.threshold_
    tensorplay.nn.functional.relu
    tensorplay.relu_
    tensorplay.nn.functional.hardtanh
    tensorplay.hardtanh_
    tensorplay.nn.functional.hardswish
    tensorplay.nn.functional.relu6
    tensorplay.nn.functional.elu
    tensorplay.elu_
    tensorplay.nn.functional.selu
    tensorplay.nn.functional.celu
    tensorplay.nn.functional.leaky_relu
    tensorplay.leaky_relu_
    tensorplay.nn.functional.prelu
    tensorplay.nn.functional.rrelu
    tensorplay.nn.functional.rrelu_
    tensorplay.nn.functional.glu
    tensorplay.nn.functional.gelu
    tensorplay.nn.functional.logsigmoid
    tensorplay.nn.functional.hardshrink
    tensorplay.nn.functional.tanhshrink
    tensorplay.nn.functional.softsign
    tensorplay.nn.functional.softplus
    tensorplay.nn.functional.softmin
    tensorplay.nn.functional.softmax
    tensorplay.nn.functional.softshrink
    tensorplay.nn.functional.gumbel_softmax
    tensorplay.nn.functional.log_softmax
    tensorplay.nn.functional.tanh
    tensorplay.nn.functional.sigmoid
    tensorplay.nn.functional.hardsigmoid
    tensorplay.nn.functional.silu
    tensorplay.nn.functional.mish
    tensorplay.nn.functional.batch_norm
    tensorplay.nn.functional.group_norm
    tensorplay.nn.functional.instance_norm
    tensorplay.nn.functional.layer_norm
    tensorplay.nn.functional.local_response_norm
    tensorplay.nn.functional.rms_norm
    tensorplay.nn.functional.normalize
```

## Linear functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.functional.linear
    tensorplay.nn.functional.bilinear
```

## Dropout functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.functional.dropout
    tensorplay.nn.functional.alpha_dropout
    tensorplay.nn.functional.feature_alpha_dropout
    tensorplay.nn.functional.dropout1d
    tensorplay.nn.functional.dropout2d
    tensorplay.nn.functional.dropout3d
```

## Sparse functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.functional.embedding
    tensorplay.nn.functional.embedding_bag
    tensorplay.nn.functional.one_hot
```

## Distance functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.functional.pairwise_distance
    tensorplay.nn.functional.cosine_similarity
    tensorplay.nn.functional.pdist
```

## Loss functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.functional.binary_cross_entropy
    tensorplay.nn.functional.binary_cross_entropy_with_logits
    tensorplay.nn.functional.poisson_nll_loss
    tensorplay.nn.functional.cosine_embedding_loss
    tensorplay.nn.functional.cross_entropy
    tensorplay.nn.functional.ctc_loss
    tensorplay.nn.functional.gaussian_nll_loss
    tensorplay.nn.functional.hinge_embedding_loss
    tensorplay.nn.functional.kl_div
    tensorplay.nn.functional.l1_loss
    tensorplay.nn.functional.linear_cross_entropy
    tensorplay.nn.functional.mse_loss
    tensorplay.nn.functional.margin_ranking_loss
    tensorplay.nn.functional.multilabel_margin_loss
    tensorplay.nn.functional.multilabel_soft_margin_loss
    tensorplay.nn.functional.multi_margin_loss
    tensorplay.nn.functional.nll_loss
    tensorplay.nn.functional.huber_loss
    tensorplay.nn.functional.smooth_l1_loss
    tensorplay.nn.functional.soft_margin_loss
    tensorplay.nn.functional.triplet_margin_loss
    tensorplay.nn.functional.triplet_margin_with_distance_loss
```

## Vision functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.functional.pixel_shuffle
    tensorplay.nn.functional.pixel_unshuffle
    tensorplay.nn.functional.pad
    tensorplay.nn.functional.interpolate
    tensorplay.nn.functional.grid_sample
    tensorplay.nn.functional.affine_grid
```

### {hidden}`data_parallel`

## Low-Precision functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.functional.grouped_mm
    tensorplay.nn.functional.scaled_mm
    tensorplay.nn.functional.scaled_grouped_mm
```

## TensorPlay-specific additions

```{eval-rst}
.. currentmodule:: tensorplay.nn.functional
```
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    DType
    Tensor
    adaptive_max_pool1d_with_indices
    adaptive_max_pool2d_with_indices
    adaptive_max_pool3d_with_indices
    celu_
    channel_shuffle
    conv_tbc
    dropout_
    feature_dropout
    feature_dropout_
    flatten
    fractional_max_pool2d_with_indices
    fractional_max_pool3d_with_indices
    gru_cell
    lstm_cell
    max_pool1d_with_indices
    max_pool2d_with_indices
    max_pool3d_with_indices
    multi_head_attention_forward
    native_channel_shuffle
    rnn_relu_cell
    rnn_tanh_cell
    selu_
```

