```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# tensorplay.nn

These are the basic building blocks for graphs:
```{contents}
:depth: 2
:local:
:backlinks: top
```

## Containers

Global Hooks For Module

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.module.register_module_forward_pre_hook
    tensorplay.nn.modules.module.register_module_forward_hook
    tensorplay.nn.modules.module.register_module_full_backward_pre_hook
    tensorplay.nn.modules.module.register_module_full_backward_hook
    tensorplay.nn.modules.module.register_module_buffer_registration_hook
    tensorplay.nn.modules.module.register_module_module_registration_hook
    tensorplay.nn.modules.module.register_module_parameter_registration_hook
```

## Utilities

From the `tensorplay.nn.utils` module:
Utility functions to clip parameter gradients.
Utility functions to flatten and unflatten Module parameters to and from a single vector.
Utility functions to fuse Modules with BatchNorm modules.
Utility functions to convert Module parameter memory formats.
Utility functions to apply and remove weight normalization from Module parameters.
Utility functions for initializing Module parameters.
Utility classes and functions for pruning Module parameters.
Parametrizations implemented using the new parametrization functionality
in {func}`tensorplay.nn.utils.parameterize.register_parametrization`.
Utility functions to parametrize Tensors on existing Modules.
Note that these functions can be used to parametrize a given Parameter
or Buffer given a specific function that maps from an input space to the
parametrized space. They are not parameterizations that would transform
an object into a parameter. See the
[Parametrizations tutorial](https://tensorplay.org/tutorials/intermediate/parametrizations.html)
for more information on how to implement your own parametrizations.
Utility functions to call a given Module in a stateless manner.
Utility functions in other modules

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.utils.stateless.functional_call
    tensorplay.nn.utils.rnn.PackedSequence
    tensorplay.nn.utils.rnn.pack_padded_sequence
    tensorplay.nn.utils.rnn.pad_packed_sequence
    tensorplay.nn.utils.rnn.pad_sequence
    tensorplay.nn.utils.rnn.pack_sequence
    tensorplay.nn.utils.rnn.unpack_sequence
    tensorplay.nn.utils.rnn.unpad_sequence
    tensorplay.nn.utils.rnn.invert_permutation
    tensorplay.nn.parameter.is_lazy
    tensorplay.nn.modules.flatten.Flatten
    tensorplay.nn.modules.flatten.Unflatten
```

## Quantized Functions

Quantization refers to techniques for performing computations and storing tensors at lower bitwidths than
floating point precision. TensorPlay supports both per tensor and per channel asymmetric linear quantization. To learn more how to use quantized functions in TensorPlay, please refer to the {ref}`quantization-doc` documentation.

## Lazy Modules Initialization

% This module needs to be documented. Adding here in the meantime
% for tracking purposes
```{eval-rst}
.. py:module:: tensorplay.nn.backends
.. py:module:: tensorplay.nn.utils.stateless
.. py:module:: tensorplay.nn.backends.thnn
.. py:module:: tensorplay.nn.common_types
.. py:module:: tensorplay.nn.cpp
.. py:module:: tensorplay.nn.functional
.. py:module:: tensorplay.nn.grad
.. py:module:: tensorplay.nn.init
.. py:module:: tensorplay.nn.modules.activation
.. py:module:: tensorplay.nn.modules.adaptive
.. py:module:: tensorplay.nn.modules.batchnorm
.. py:module:: tensorplay.nn.modules.channelshuffle
.. py:module:: tensorplay.nn.modules.container
.. py:module:: tensorplay.nn.modules.conv
.. py:module:: tensorplay.nn.modules.distance
.. py:module:: tensorplay.nn.modules.dropout
.. py:module:: tensorplay.nn.modules.flatten
.. py:module:: tensorplay.nn.modules.folding
.. py:module:: tensorplay.nn.modules.instancenorm
.. py:module:: tensorplay.nn.modules.lazy
.. py:module:: tensorplay.nn.modules.linear
.. py:module:: tensorplay.nn.modules.loss
.. py:module:: tensorplay.nn.modules.module
.. py:module:: tensorplay.nn.modules.normalization
.. py:module:: tensorplay.nn.modules.padding
.. py:module:: tensorplay.nn.modules.pixelshuffle
.. py:module:: tensorplay.nn.modules.pooling
.. py:module:: tensorplay.nn.modules.rnn
.. py:module:: tensorplay.nn.modules.sparse
.. py:module:: tensorplay.nn.modules.transformer
.. py:module:: tensorplay.nn.modules.upsampling
.. py:module:: tensorplay.nn.modules.utils
.. py:module:: tensorplay.nn.parallel.comm
.. py:module:: tensorplay.nn.parallel.distributed
.. py:module:: tensorplay.nn.parallel.parallel_apply
.. py:module:: tensorplay.nn.parallel.replicate
.. py:module:: tensorplay.nn.parallel.scatter_gather
.. py:module:: tensorplay.nn.parameter
.. py:module:: tensorplay.nn.utils.clip_grad
.. py:module:: tensorplay.nn.utils.convert_parameters
.. py:module:: tensorplay.nn.utils.fusion
.. py:module:: tensorplay.nn.utils.init
.. py:module:: tensorplay.nn.utils.memory_format
.. py:module:: tensorplay.nn.utils.parametrizations
.. py:module:: tensorplay.nn.utils.parametrize
.. py:module:: tensorplay.nn.utils.prune
.. py:module:: tensorplay.nn.utils.rnn
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.nn.modules.lazy.LazyModuleMixin
```
