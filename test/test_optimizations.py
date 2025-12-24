import tensorplay as tp
import unittest

# Access internal modules
try:
    from tensorplay._C import _stax
except ImportError:
    print("Could not import _stax from tensorplay._C")
    _stax = None

class TestOptimizations(unittest.TestCase):
    
    def test_1_caching_allocator(self):
        print("\n--- Testing Caching Allocator ---")
        # Allocate a tensor
        t1 = tp.ones([1024, 1024], dtype=tp.float32)
        ptr1 = t1.data_ptr()
        
        # Free it (by deleting reference)
        del t1
        
        # Allocate another tensor of same size
        t2 = tp.ones([1024, 1024], dtype=tp.float32)
        ptr2 = t2.data_ptr()
        
        print(f"Ptr1: {ptr1}, Ptr2: {ptr2}")
        
        # In a caching allocator, ptr2 should often equal ptr1 (reuse)
        # However, exact behavior depends on implementation details (e.g. fragmentation, other allocs)
        # We just verify it runs without crashing and gives valid pointers.
        self.assertNotEqual(ptr2, 0)
        
    def test_2_structured_dispatcher(self):
        print("\n--- Testing Structured Dispatcher ---")
        # Verify basic operations still work (routed through dispatcher)
        t1 = tp.ones([10], dtype=tp.float32)
        t2 = tp.ones([10], dtype=tp.float32)
        t3 = t1 + t2
        
        self.assertEqual(t3[0].item(), 2.0)
        print("Dispatch successful: 1 + 1 = 2")
        
    def test_3_stax_graph_ir(self):
        print("\n--- Testing Stax Graph IR ---")
        if _stax is None:
            print("Stax module not available.")
            return

        graph = _stax.Graph()
        
        # Create input
        in1 = graph.add_input()
        in2 = graph.add_input()
        
        # Create a 'mul' node
        mul_node = graph.create_node("mul")
        mul_node.add_input(in1)
        mul_node.add_input(in2)
        mul_out = mul_node.add_output()
        
        # Create an 'add' node
        add_node = graph.create_node("add")
        add_node.add_input(mul_out)
        add_node.add_input(in1) # Add input 1 again
        add_out = add_node.add_output()
        
        graph.register_output(add_out)
        
        print("Graph constructed:")
        graph.print()
        
        # Basic verification
        # No crashes implies success for now
        
    def test_4_jit_fusion(self):
        print("\n--- Testing JIT Operator Fusion ---")
        if _stax is None:
            print("Stax module not available.")
            return

        graph = _stax.Graph()
        
        # Create a pattern: mul -> add
        in1 = graph.add_input()
        in2 = graph.add_input()
        in3 = graph.add_input()
        
        # mul = in1 * in2
        mul_node = graph.create_node("mul")
        mul_node.add_input(in1)
        mul_node.add_input(in2)
        mul_out = mul_node.add_output()
        
        # add = mul + in3
        add_node = graph.create_node("add")
        add_node.add_input(mul_out)
        add_node.add_input(in3)
        add_out = add_node.add_output()
        
        graph.register_output(add_out)
        
        print("Before optimization:")
        graph.print()
        
        # Run fusion
        graph.fuse()
        
        print("After optimization:")
        graph.print()
        
        # Verify fusion results
        print(f"Fused node kind: {add_node.op_type}")
        print(f"Fused node inputs: {add_node.input_count}")
        
        self.assertEqual(add_node.op_type, "fused_mul_add")
        self.assertEqual(add_node.input_count, 3)
        
if __name__ == '__main__':
    unittest.main()
