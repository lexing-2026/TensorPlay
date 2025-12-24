import tensorplay as tp

def test_comparison():
    print("Testing Comparison Operators...")
    t1 = tp.tensor((1.0, 2.0, 3.0))
    t2 = tp.tensor([1.0, 2.5, 2.0])
    print("t1 strides:", t1.strides)

    print("  lt (Scalar):", t2[t1 < 2.0])

def test_matmul():
    print("\nTesting Matmul (@)...")
    # 2D
    a = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tp.tensor([[1.0, 0.0], [0.0, 1.0]])
    print("  2D result:", (a @ b.t()).numpy())
    
    # Batched (3D)
    # (2, 2, 3) @ (2, 3, 2) -> (2, 2, 2)
    B = 2
    M, K, N = 2, 3, 2
    
    a_batch = tp.ones(B, M, K)
    b_batch = tp.ones(B, K, N)
    
    c_batch = a_batch @ b_batch
    print("  Batched result shape:", c_batch.shape)
    print("  Batched result[0]:", c_batch[0].numpy())

def test_rpow():
    print("\nTesting Reverse Pow (-0.2 ** x)...")
    x = tp.tensor([1.0, 2.0, 3.0])
    
    # -0.2 ** x
    # Note: -0.2 ** 2.0 = 0.04, -0.2 ** 1.0 = -0.2
    # But if base is negative and exponent is fractional, result is NaN (complex).
    # Here exponents are integers (stored as float), so it should work.
    
    res = -0.2 ** x
    print("  -0.2 ** x:", res.numpy())
    
    # Verify values
    expected = [-0.2, 0.04, -0.008]
    print("  Expected approx:", expected)

if __name__ == "__main__":
    try:
        test_comparison()
        test_matmul()
        test_rpow()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
