```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# tensorplay

## Tensors

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.is_tensor
    tensorplay.functional.is_complex
    tensorplay._composite_funcs.is_conj
    tensorplay._composite_funcs.is_neg
    tensorplay._composite_funcs.is_nonzero
    tensorplay._composite_funcs.is_same_size
    tensorplay.set_default_dtype
    tensorplay.get_default_dtype
    tensorplay.set_default_device
    tensorplay.get_default_device
    tensorplay._composite_funcs.numel
    tensorplay.set_printoptions
```

### Creation Ops

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.tensor
    tensorplay.functional.sparse_coo_tensor
    tensorplay.as_tensor
    tensorplay.from_dlpack
    tensorplay.functional.zeros
    tensorplay.functional.zeros_like
    tensorplay.functional.ones
    tensorplay.functional.ones_like
    tensorplay.functional.arange
    tensorplay.functional.linspace
    tensorplay.functional.logspace
    tensorplay.functional.eye
    tensorplay.functional.empty
    tensorplay.functional.empty_like
    tensorplay.functional.full
    tensorplay.functional.full_like
    tensorplay.functional.quantize_per_tensor
    tensorplay.functional.quantize_per_channel
    tensorplay.functional.complex
    tensorplay.functional.polar
    tensorplay._composite_funcs.scalar_tensor
    tensorplay.functional.heaviside
```

### Indexing, Slicing, Joining, Mutating Ops

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay._composite_funcs.adjoint
    tensorplay._composite_funcs.alias_copy
    tensorplay.functional.argwhere
    tensorplay.functional.cat
    tensorplay._composite_funcs.concat
    tensorplay._composite_funcs.concatenate
    tensorplay.functional.conj
    tensorplay.functional.chunk
    tensorplay._composite_funcs.detach
    tensorplay._composite_funcs.diagonal_copy
    tensorplay.functional.dsplit
    tensorplay.functional.column_stack
    tensorplay.functional.dstack
    tensorplay._composite_funcs.expand_copy
    tensorplay.functional.fill
    tensorplay.functional.gather
    tensorplay.functional.hsplit
    tensorplay.functional.hstack
    tensorplay.functional.index_add
    tensorplay.functional.index_copy
    tensorplay.functional.index_put_
    tensorplay.functional.index_select
    tensorplay.functional.masked_fill
    tensorplay.functional.masked_select
    tensorplay.functional.movedim
    tensorplay.functional.moveaxis
    tensorplay.functional.narrow
    tensorplay._composite_funcs.narrow_copy
    tensorplay.functional.nonzero
    tensorplay.functional.permute
    tensorplay._composite_funcs.permute_copy
    tensorplay._composite_funcs.put
    tensorplay.functional.reshape
    tensorplay.functional.row_stack
    tensorplay.functional.select
    tensorplay._composite_funcs.select_copy
    tensorplay.functional.scatter
    tensorplay.functional.diagonal_scatter
    tensorplay.functional.select_scatter
    tensorplay._composite_funcs.slice_copy
    tensorplay.functional.slice_scatter
    tensorplay.functional.scatter_add
    tensorplay.functional.split
    tensorplay._composite_funcs.split_copy
    tensorplay.functional.squeeze
    tensorplay._composite_funcs.squeeze_copy
    tensorplay.functional.stack
    tensorplay.functional.swapaxes
    tensorplay.functional.swapdims
    tensorplay.functional.t
    tensorplay._composite_funcs.t_copy
    tensorplay.functional.take
    tensorplay.functional.take_along_dim
    tensorplay.functional.tensor_split
    tensorplay.functional.tile
    tensorplay.functional.transpose
    tensorplay._composite_funcs.transpose_copy
    tensorplay.functional.unbind
    tensorplay._composite_funcs.unbind_copy
    tensorplay._composite_funcs.unfold_copy
    tensorplay._shape_funcs.unravel_index
    tensorplay.functional.unsqueeze
    tensorplay._composite_funcs.unsqueeze_copy
    tensorplay._composite_funcs.view_copy
    tensorplay.functional.vsplit
    tensorplay.functional.vstack
    tensorplay.functional.where
```

## Generators

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.Generator
```

## Random sampling

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.seed
    tensorplay.manual_seed
    tensorplay.initial_seed
    tensorplay.get_rng_state
    tensorplay.set_rng_state
    tensorplay.functional.bernoulli
    tensorplay.functional.multinomial
    tensorplay.functional.normal
    tensorplay.functional.poisson
    tensorplay.functional.rand
    tensorplay.functional.rand_like
    tensorplay.functional.randint
    tensorplay.functional.randint_like
    tensorplay.functional.randn
    tensorplay.functional.randn_like
    tensorplay.functional.randperm
```

## Serialization

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.serialization.save
    tensorplay.serialization.load
```

## Parallelism

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.get_num_threads
    tensorplay.set_num_threads
```

## Locally disabling gradient computation

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.autograd.grad_mode.no_grad
    tensorplay.autograd.grad_mode.enable_grad
    tensorplay.autograd.grad_mode.set_grad_enabled
    tensorplay.autograd.grad_mode.is_grad_enabled
    tensorplay.autograd.grad_mode.inference_mode
```

### Pointwise Ops

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.functional.abs
    tensorplay.functional.abs_
    tensorplay._composite_funcs.absolute
    tensorplay.functional.acos
    tensorplay._composite_funcs.acos_
    tensorplay._composite_funcs.arccos
    tensorplay.functional.acosh
    tensorplay._composite_funcs.acosh_
    tensorplay._composite_funcs.arccosh
    tensorplay.functional.add
    tensorplay.functional.addcdiv
    tensorplay.functional.addcmul
    tensorplay.functional.angle
    tensorplay.functional.asin
    tensorplay._composite_funcs.asin_
    tensorplay._composite_funcs.arcsin
    tensorplay.functional.asinh
    tensorplay._composite_funcs.asinh_
    tensorplay._composite_funcs.arcsinh
    tensorplay.functional.atan
    tensorplay._composite_funcs.atan_
    tensorplay._composite_funcs.arctan
    tensorplay.functional.atanh
    tensorplay._composite_funcs.atanh_
    tensorplay._composite_funcs.arctanh
    tensorplay.functional.atan2
    tensorplay._composite_funcs.arctan2
    tensorplay.functional.bitwise_not
    tensorplay.functional.bitwise_and
    tensorplay.functional.bitwise_or
    tensorplay.functional.bitwise_xor
    tensorplay.functional.bitwise_left_shift
    tensorplay.functional.bitwise_right_shift
    tensorplay.functional.ceil
    tensorplay.functional.clamp
    tensorplay.functional.clamp_
    tensorplay.functional.clip
    tensorplay._composite_funcs.copysign
    tensorplay.functional.cos
    tensorplay.functional.cosh
    tensorplay.functional.deg2rad
    tensorplay.functional.div
    tensorplay._composite_funcs.divide
    tensorplay.functional.digamma
    tensorplay.functional.erf
    tensorplay.functional.erfc
    tensorplay.functional.erfinv
    tensorplay.functional.exp
    tensorplay.functional.exp2
    tensorplay.functional.expm1
    tensorplay.functional.fill_
    tensorplay.functional.fix
    tensorplay.functional.floor
    tensorplay._composite_funcs.floor_divide
    tensorplay._composite_funcs.fmod
    tensorplay.functional.frac
    tensorplay._composite_funcs.gradient
    tensorplay.functional.imag
    tensorplay.functional.lerp
    tensorplay.functional.lgamma
    tensorplay.functional.log
    tensorplay.functional.log10
    tensorplay.functional.log1p
    tensorplay.functional.log2
    tensorplay.functional.logaddexp
    tensorplay.functional.logaddexp2
    tensorplay.functional.logical_and
    tensorplay.functional.logical_not
    tensorplay.functional.logical_or
    tensorplay.functional.logical_xor
    tensorplay.functional.logit
    tensorplay.functional.hypot
    tensorplay.functional.i0
    tensorplay.functional.mul
    tensorplay._composite_funcs.multiply
    tensorplay.functional.nan_to_num
    tensorplay.functional.neg
    tensorplay.functional.neg_
    tensorplay.functional.negative
    tensorplay.functional.nextafter
    tensorplay.functional.polygamma
    tensorplay.functional.positive
    tensorplay.functional.pow
    tensorplay.functional.rad2deg
    tensorplay.functional.real
    tensorplay.functional.reciprocal
    tensorplay._composite_funcs.remainder
    tensorplay.functional.round
    tensorplay.functional.rsqrt
    tensorplay.functional.rsqrt_
    tensorplay.functional.sigmoid
    tensorplay.functional.sign
    tensorplay.functional.sgn
    tensorplay.functional.signbit
    tensorplay.functional.sin
    tensorplay.functional.sinc
    tensorplay.functional.sinh
    tensorplay.functional.softmax
    tensorplay.functional.sqrt
    tensorplay.functional.sqrt_
    tensorplay.functional.square
    tensorplay.functional.sub
    tensorplay._composite_funcs.subtract
    tensorplay.functional.tan
    tensorplay.functional.tanh
    tensorplay._composite_funcs.true_divide
    tensorplay.functional.trunc
    tensorplay.functional.zero_
```

### Reduction Ops

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.functional.argmax
    tensorplay.functional.argmin
    tensorplay.functional.amax
    tensorplay.functional.amin
    tensorplay.functional.aminmax
    tensorplay.functional.all
    tensorplay.functional.any
    tensorplay.functional.max
    tensorplay.functional.min
    tensorplay.functional.dist
    tensorplay.functional.logsumexp
    tensorplay.functional.mean
    tensorplay.functional.nanmean
    tensorplay.functional.median
    tensorplay.functional.nanmedian
    tensorplay.functional.mode
    tensorplay.functional.norm
    tensorplay.functional.nansum
    tensorplay.functional.prod
    tensorplay._composite_funcs.quantile
    tensorplay._composite_funcs.nanquantile
    tensorplay.functional.std
    tensorplay.functional.std_mean
    tensorplay.functional.sum
    tensorplay.functional.unique
    tensorplay._composite_funcs.unique_consecutive
    tensorplay.functional.var
    tensorplay.functional.var_mean
    tensorplay.functional.count_nonzero
```

### Comparison Ops

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.functional.allclose
    tensorplay.functional.argsort
    tensorplay.functional.eq
    tensorplay.functional.equal
    tensorplay.functional.ge
    tensorplay.functional.greater_equal
    tensorplay.functional.gt
    tensorplay.functional.greater
    tensorplay.functional.isclose
    tensorplay.functional.isfinite
    tensorplay._composite_funcs.isin
    tensorplay.functional.isinf
    tensorplay.functional.isposinf
    tensorplay.functional.isneginf
    tensorplay.functional.isnan
    tensorplay.functional.isreal
    tensorplay.functional.kthvalue
    tensorplay.functional.le
    tensorplay.functional.less_equal
    tensorplay.functional.lt
    tensorplay.functional.less
    tensorplay.functional.maximum
    tensorplay.functional.minimum
    tensorplay.functional.ne
    tensorplay.functional.not_equal
    tensorplay.functional.sort
    tensorplay.functional.topk
    tensorplay.functional.msort
```

### Spectral Ops

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.functional.stft
    tensorplay.functional.istft
    tensorplay.functional.bartlett_window
    tensorplay.functional.blackman_window
    tensorplay.functional.hamming_window
    tensorplay.functional.hann_window
    tensorplay._composite_funcs.kaiser_window
```

### Other Operations

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay._composite_funcs.adaptive_avg_pool1d
    tensorplay._composite_funcs.adaptive_max_pool1d
    tensorplay.functional.atleast_1d
    tensorplay.functional.atleast_2d
    tensorplay.functional.atleast_3d
    tensorplay._composite_funcs.avg_pool1d
    tensorplay.functional.bincount
    tensorplay.functional.block_diag
    tensorplay.functional.broadcast_tensors
    tensorplay.functional.broadcast_to
    tensorplay._shape_funcs.broadcast_shapes
    tensorplay.functional.bucketize
    tensorplay._composite_funcs.cartesian_prod
    tensorplay.functional.channel_shuffle
    tensorplay.functional.clone
    tensorplay._composite_funcs.combinations
    tensorplay.functional.conv1d
    tensorplay.functional.conv3d
    tensorplay.functional.conv_transpose1d
    tensorplay.functional.conv_transpose2d
    tensorplay.functional.conv_transpose3d
    tensorplay._composite_funcs.corrcoef
    tensorplay.functional.cosine_embedding_loss
    tensorplay._composite_funcs.cosine_similarity
    tensorplay._composite_funcs.cov
    tensorplay.functional.cummax
    tensorplay.functional.cummin
    tensorplay.functional.cumprod
    tensorplay.functional.cumsum
    tensorplay.functional.diag
    tensorplay.functional.diag_embed
    tensorplay._composite_funcs.diagflat
    tensorplay.functional.diagonal
    tensorplay.functional.diff
    tensorplay.functional.einsum
    tensorplay.functional.embedding
    tensorplay.functional.flatten
    tensorplay.functional.flip
    tensorplay.functional.gcd
    tensorplay.functional.group_norm
    tensorplay.functional.gru
    tensorplay.functional.hardshrink
    tensorplay.functional.hinge_embedding_loss
    tensorplay._composite_funcs.histc
    tensorplay._composite_funcs.histogram
    tensorplay.functional.instance_norm
    tensorplay.functional.kl_div
    tensorplay._composite_funcs.kron
    tensorplay.functional.lcm
    tensorplay.functional.logcumsumexp
    tensorplay.functional.lstm
    tensorplay._composite_funcs.lstm_cell
    tensorplay.functional.margin_ranking_loss
    tensorplay._composite_funcs.max_pool1d
    tensorplay.functional.meshgrid
    tensorplay.functional.pairwise_distance
    tensorplay.functional.pdist
    tensorplay.functional.pixel_unshuffle
    tensorplay.functional.poisson_nll_loss
    tensorplay.functional.prelu
    tensorplay.functional.ravel
    tensorplay.functional.relu_
    tensorplay.functional.renorm
    tensorplay._composite_funcs.repeat_interleave
    tensorplay._composite_funcs.rms_norm
    tensorplay.functional.rnn_relu
    tensorplay._composite_funcs.rnn_relu_cell
    tensorplay.functional.rnn_tanh
    tensorplay._composite_funcs.rnn_tanh_cell
    tensorplay.functional.roll
    tensorplay.functional.rot90
    tensorplay._composite_funcs.rsub
    tensorplay.functional.searchsorted
    tensorplay.functional.selu
    tensorplay._shape_funcs.tensordot
    tensorplay.functional.threshold
    tensorplay.functional.trace
    tensorplay.functional.tril
    tensorplay._composite_funcs.tril_indices
    tensorplay.functional.triu
    tensorplay._composite_funcs.triu_indices
    tensorplay.functional.triplet_margin_loss
    tensorplay.functional.unflatten
    tensorplay._composite_funcs.vander
    tensorplay.functional.view_as_real
    tensorplay.functional.view_as_complex
    tensorplay._composite_funcs.resolve_conj
    tensorplay._composite_funcs.resolve_neg
```

### BLAS and LAPACK Operations

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.functional.addbmm
    tensorplay.functional.addmm
    tensorplay.functional.addmv
    tensorplay.functional.addr
    tensorplay.functional.baddbmm
    tensorplay.functional.bmm
    tensorplay._composite_funcs.chain_matmul
    tensorplay.functional.cholesky_inverse
    tensorplay.functional.cholesky_solve
    tensorplay.functional.dot
    tensorplay._composite_funcs.ger
    tensorplay.functional.inner
    tensorplay.functional.matmul
    tensorplay._composite_funcs.matrix_power
    tensorplay.functional.mm
    tensorplay.functional.mv
    tensorplay.functional.outer
    tensorplay.functional.svd
    tensorplay._composite_funcs.trapz
    tensorplay._composite_funcs.trapezoid
    tensorplay._composite_funcs.cumulative_trapezoid
    tensorplay.functional.triangular_solve
    tensorplay.functional.vdot
```

## Utilities

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.autocast_decrement_nesting
    tensorplay.autocast_increment_nesting
    tensorplay.clear_autocast_cache
    tensorplay.amp.autocast_mode.get_autocast_cpu_dtype
    tensorplay.get_autocast_dtype
    tensorplay.amp.autocast_mode.get_autocast_gpu_dtype
    tensorplay._composite_funcs.get_device
    tensorplay.is_autocast_cache_enabled
    tensorplay.is_autocast_enabled
    tensorplay._composite_funcs.result_type
    tensorplay._composite_funcs.can_cast
    tensorplay._composite_funcs.promote_types
    tensorplay.set_autocast_cache_enabled
    tensorplay.set_autocast_dtype
    tensorplay.set_autocast_enabled
    tensorplay.use_deterministic_algorithms
    tensorplay.are_deterministic_algorithms_enabled
    tensorplay.is_deterministic_algorithms_warn_only_enabled
    tensorplay.set_deterministic_debug_mode
    tensorplay.get_deterministic_debug_mode
    tensorplay.set_float32_matmul_precision
    tensorplay.get_float32_matmul_precision
    tensorplay.typename
```

## Optimizations

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay._stax.api.compile
```

## TensorPlay-specific additions

```{eval-rst}
.. currentmodule:: tensorplay
```
```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    DType
    Device
    DeviceType
    GradScaler
    Layout
    MemoryFormat
    Scalar
    Size
    Tensor
    adaptive_avg_pool2d
    adaptive_avg_pool3d
    adaptive_max_pool2d
    add_
    add_relu
    addcdiv_
    addcmul_
    airy_ai
    autocast
    avg_pool2d
    avg_pool3d
    batch_norm
    bernoulli_
    bessel_j0
    bessel_j1
    bessel_y0
    bessel_y1
    binary_cross_entropy
    binary_cross_entropy_with_logits
    cauchy_
    celu
    chebyshev_polynomial_t
    chebyshev_polynomial_u
    chebyshev_polynomial_v
    chebyshev_polynomial_w
    cholesky
    circular_pad_nd
    clamp_max
    clamp_min
    col2im
    constant_pad_nd
    contiguous
    conv1d_grad_bias
    conv1d_grad_input
    conv1d_grad_weight
    conv2d
    conv2d_grad_bias
    conv2d_grad_input
    conv2d_grad_weight
    conv2d_relu
    conv3d_grad_bias
    conv3d_grad_input
    conv3d_grad_weight
    conv_transpose1d_grad_bias
    conv_transpose1d_grad_input
    conv_transpose1d_grad_weight
    conv_transpose2d_grad_bias
    conv_transpose2d_grad_input
    conv_transpose2d_grad_weight
    conv_transpose3d_grad_bias
    conv_transpose3d_grad_input
    conv_transpose3d_grad_weight
    copy_
    custom_bwd
    custom_fwd
    dequantize_per_channel
    dequantize_per_tensor
    device
    div_
    dtype
    elu
    expand
    expand_as
    exponential_
    fft_fft
    fft_ifft
    fft_irfft
    fft_rfft
    fork_rng
    forward_add
    forward_cos
    forward_div
    forward_exp
    forward_log
    forward_mm
    forward_mul
    forward_neg
    forward_pow
    forward_relu
    forward_sigmoid
    forward_sin
    forward_sqrt
    forward_sub
    forward_tanh
    fused_mul_add
    gammainc
    gammaincc
    gelu
    geometric_
    get_parallel_info
    get_thread_num
    glu
    hardsigmoid
    hardswish
    hardtanh
    hermite_polynomial_h
    hermite_polynomial_he
    huber_loss
    i0e
    i1
    i1e
    im2col
    in_parallel_region
    index_fill
    index_fill_
    index_put
    inner_backward_other
    inner_backward_self
    inspect_checkpoint
    item
    l1_loss
    laguerre_polynomial_l
    layer_norm
    leaky_relu
    legendre_polynomial_p
    lerp_
    log_normal_
    log_softmax
    masked_fill_
    masked_scatter
    matmul_backward_other
    matmul_backward_self
    max_pool2d
    mish
    modified_bessel_i1
    modified_bessel_k0
    modified_bessel_k1
    mse_loss
    mul_
    multi_margin_loss
    multilabel_margin_loss
    multilabel_soft_margin_loss
    native_dropout
    nll_loss
    normal_
    one_hot
    pixel_shuffle
    quantized_linear
    random_
    reflection_pad_nd
    relu
    relu6
    repeat
    replication_pad_nd
    resize_
    sample
    scaled_dot_product_attention
    scaled_modified_bessel_k0
    scaled_modified_bessel_k1
    scatter_
    scatter_add_
    shifted_chebyshev_polynomial_t
    shifted_chebyshev_polynomial_u
    shifted_chebyshev_polynomial_v
    shifted_chebyshev_polynomial_w
    silu
    slice
    smooth_l1_loss
    soft_margin_loss
    softplus
    softshrink
    sparse_add
    sparse_mm
    sparse_mul
    sparse_sum
    spdiags
    spherical_bessel_j0
    split_with_sizes
    sub_
    to_dense
    to_sparse
    to_sparse_csr
    tp_binary_cross_entropy
    tp_cosine_embedding_loss
    tp_hinge_embedding_loss
    tp_huber_loss
    tp_kl_div
    tp_l1_loss
    tp_margin_ranking_loss
    tp_poisson_nll_loss
    tp_smooth_l1_loss
    tp_soft_margin_loss
    unfold
    uniform_
    unsafe_chunk
    unsafe_split
    upsample_bicubic2d
    upsample_bilinear2d
    upsample_linear1d
    upsample_nearest1d
    upsample_nearest2d
    upsample_nearest3d
    upsample_trilinear3d
    view
    zeta
```

