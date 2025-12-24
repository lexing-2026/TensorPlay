import time
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import tensorplay as tp

def benchmark_mm():
    print(f"GPU Name: {tp.cuda.get_device_name(0)}")
    
    sizes = [32, 64, 128, 512, 1024, 2048, 4096]
    
    print(f"{'Size':<10} | {'TP GFLOPS':<12} | {'Torch GFLOPS':<12} | {'Ratio (TP/Torch)':<15}")
    print("-" * 60)
    
    device_tp = tp.device("cuda")
    device_torch = torch.device("cuda")
    
    for N in sizes:
        # TFLOPS formula: 2 * N^3 / time / 1e9
        
        # TensorPlay
        a_tp = tp.rand([N, N], device=device_tp)
        b_tp = tp.rand([N, N], device=device_tp)
        
        # Warmup
        for _ in range(5):
            _ = tp.mm(a_tp, b_tp)
        tp.cuda.synchronize()
        
        start = time.time()
        iters = 100 if N < 1024 else 10
        for _ in range(iters):
            c_tp = tp.mm(a_tp, b_tp)
        # Force sync
        tp.cuda.synchronize()
        end = time.time()
        
        time_tp = (end - start) / iters
        gflops_tp = (2 * N**3) / time_tp / 1e9
        
        # PyTorch
        a_torch = torch.rand((N, N), device=device_torch)
        b_torch = torch.rand((N, N), device=device_torch)
        
        # Warmup
        for _ in range(5):
            _ = torch.mm(a_torch, b_torch)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(iters):
            c_torch = torch.mm(a_torch, b_torch)
        torch.cuda.synchronize()
        end = time.time()
        
        time_torch = (end - start) / iters
        gflops_torch = (2 * N**3) / time_torch / 1e9
        
        print(f"{N:<10} | {gflops_tp:<12.2f} | {gflops_torch:<12.2f} | {gflops_tp/gflops_torch:<15.2f}")

if __name__ == "__main__":
    try:
        benchmark_mm()
    except Exception as e:
        print(f"Benchmark failed: {e}")
