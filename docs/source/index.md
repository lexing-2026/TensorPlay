% TensorPlay documentation master file.
% You can adapt this file completely to your liking, but it should at least
% contain the root `toctree` directive.

TensorPlay documentation
===================================

TensorPlay is a tensor library for deep learning using GPUs and CPUs.

Features described in this documentation are classified by release status:

**Stable (API-Stable):**
These features will be maintained long-term and there should generally be no major performance limitations or gaps in documentation. We also expect to maintain backwards compatibility (although breaking changes can happen and notice will be given one release ahead of time).

**Unstable (API-Unstable):**
Encompasses all features that are under active development where APIs may change based on user feedback, requisite performance improvements or because coverage across operators is not yet complete.
The APIs and performance characteristics of these features may change.

```{toctree}
:maxdepth: 2

guide/index
tensorplay
tensor_attributes
tensor_view
type_info
size
deterministic
dlpack
complex_numbers
autograd
nn
nn.functional
nn.init
optim
cuda
accelerator
amp
linalg
fft
special
signal
sparse
random
data
checkpoint
futures
hub
multiprocessing
library
distributed
distributed.device_mesh
distributed.tensor
distributed.tensor.parallel
distributed.fsdp
distributed.checkpoint
distributed.elastic
distributed.rpc
distributed.pipelining
distributed.optim
distributed.nn
distributed.autograd
distributions
func
profiler
monitor
nested
masked
testing
serialization
onnx
package
export
compiler
quantization
notes/index
_stax
stax
vision
audio
viz
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
