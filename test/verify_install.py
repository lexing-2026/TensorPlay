import torch
import tensorplay


print(f"TensorPlay version: {tensorplay.__version__}")
print(f"TensorPlay CUDA available: {tensorplay.cuda.is_available()}")
print(f"TensorPlay CUDA version: {tensorplay.cuda.get_device_capability()}")

if tensorplay.cuda.is_available():
    print(f"Device count: {tensorplay.cuda.device_count()}")
    try:
        x = tensorplay.ones((2, 3)).cuda()
        print(f"CUDA Tensor x:\n{x}")
        y = tensorplay.zeros((2, 3)).cuda()
        z = x + y
        print(f"CUDA Add Result z:\n{z}")
    except Exception as e:
        print(f"CUDA Operation failed: {e}")
