import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import tensorplay as tp
from tensorplay.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):
    def __init__(self, size, shape):
        self.size = size
        self.shape = shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Simulate data loading/processing
        # Create a random tensor
        # Add some math to simulate heavier transform
        t = tp.randn(self.shape)
        # Simulate some CPU work (e.g. image decoding, augmentation)
        # 1ms of sleep simulates minimal IO/processing
        # For a real speedup demo, we need the work to outweigh IPC overhead
        # time.sleep(0.001) 
        for _ in range(10):
            t = t.add(1.0)
        return t

def benchmark(num_workers, batch_size=64, dataset_size=2000, device=None):
    print(f"Benchmarking with num_workers={num_workers}, device={device}...")
    dataset = RandomDataset(dataset_size, (3, 224, 224))
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, device=device)
    
    start = time.time()
    count = 0
    # Warmup the workers
    if num_workers > 0:
        iter(dataloader)
    
    # Actually measure
    start = time.time()
    for batch in dataloader:
        if count == 0:
             # Check device once
             print(f"Batch device: {batch.device}")
        count += batch.shape[0]
    end = time.time()
    
    duration = end - start
    throughput = count / duration
    print(f"Processed {count} items in {duration:.2f}s. Throughput: {throughput:.2f} items/s")
    return throughput

if __name__ == '__main__':
    # Set OMP threads to 1 to avoid contention if any
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Warmup
    print("Warming up...")
    benchmark(0, batch_size=16, dataset_size=100)
    
    print("\n--- Starting Benchmark (10000 items) ---")
    dataset_size = 10000
    batch_size = 64 # Reduce batch size slightly to increase number of batches/interactions
    
    serial_throughput = benchmark(0, batch_size=batch_size, dataset_size=dataset_size, device='cpu')
    
    # Use 4 workers to have a chance at 3x speedup
    parallel_throughput = benchmark(4, batch_size=batch_size, dataset_size=dataset_size, device='cpu')
    
    speedup = parallel_throughput / serial_throughput
    print(f"\nSpeedup: {speedup:.2f}x")
    
    if speedup > 3.0:
        print("SUCCESS: Speedup > 3x.")
    elif speedup > 2.0:
        print("ACCEPTABLE: Speedup > 2x.")
    else:
        print("WARNING: Low speedup.")
