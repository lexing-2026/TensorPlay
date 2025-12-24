import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import torch
import tensorplay as tp
import tensorplay.nn as nn

# Set environment variable for OpenMP to ensure fair comparison if needed
# os.environ["OMP_NUM_THREADS"] = "8" 

def benchmark_op(name, torch_func, tp_func, iterations=20):
    # Warmup
    print(f"Warming up {name}...")
    try:
        for _ in range(5):
            torch_func()
            tp_func()
    except Exception as e:
        print(f"Warmup failed: {e}")
        return

    # Torch Benchmark
    start = time.time()
    for _ in range(iterations):
        torch_func()
    torch_time = (time.time() - start) / iterations
    
    # TensorPlay Benchmark
    start = time.time()
    for _ in range(iterations):
        tp_func()
    tp_time = (time.time() - start) / iterations
    
    print(f"{name}:")
    print(f"  PyTorch:    {torch_time*1000:.3f} ms")
    print(f"  TensorPlay: {tp_time*1000:.3f} ms")
    ratio = torch_time/tp_time if tp_time > 0 else 0
    print(f"  Efficiency (TP/Torch): {ratio:.2f}x")
    print("-" * 40)

def run_conv_benchmark(N, C, H, W, OutC, K, name):
    print(f"Benchmarking {name}: Input({N},{C},{H},{W}) -> Out({OutC}), Kernel({K})")
    
    # Torch
    t_input = torch.randn(N, C, H, W, requires_grad=True)
    t_conv = torch.nn.Conv2d(C, OutC, K, padding=K//2)
    
    # TensorPlay
    tp_input = tp.tensor(t_input.detach(), requires_grad=True)
    tp_conv = nn.Conv2d(C, OutC, K, padding=K//2)
    tp_conv.weight.data = tp.tensor(t_conv.weight.detach())
    tp_conv.bias.data = tp.tensor(t_conv.bias.detach())
    
    def run_torch():
        out = t_conv(t_input)
        loss = out.sum()
        loss.backward()
        
    def run_tp():
        out = tp_conv(tp_input)
        loss = out.sum()
        loss.backward()
        
    benchmark_op(name, run_torch, run_tp, iterations=10)

if __name__ == "__main__":
    print("Running Comprehensive Conv2d Benchmarks...\n")
    
    # 1. Standard (Current Target)
    run_conv_benchmark(32, 64, 64, 64, 128, 3, "Standard (ResNet Block)")
    
    # 2. Small (CIFAR-like)
    run_conv_benchmark(16, 3, 32, 32, 64, 3, "Small (CIFAR Input)")
    
    # 3. Large (ImageNet-like Input)
    # run_conv_benchmark(8, 3, 224, 224, 64, 7, "Large (ImageNet Input)") # Might be slow/OOM on some machines, keep conservative
    run_conv_benchmark(4, 3, 224, 224, 64, 7, "Large (ImageNet Input)")

    # 4. Deep Layer (Small spatial, large channels)
    run_conv_benchmark(32, 256, 14, 14, 512, 3, "Deep Layer")
    
    # 5. 1x1 Conv (Projection)
    run_conv_benchmark(32, 64, 64, 64, 256, 1, "1x1 Projection")
