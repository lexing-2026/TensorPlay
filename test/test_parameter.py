import unittest
import tensorplay as tp
import tensorplay.nn as nn

class TestParameter(unittest.TestCase):
    def test_init(self):
        t = tp.tensor([1., 2.])
        p = nn.Parameter(t)
        self.assertTrue(p.requires_grad)
        self.assertEqual(p.shape, t.shape)
        # Verify data sharing (if implemented via copy_metadata/pointer share)
        # Note: tensorplay Python bindings might behave differently than PyTorch regarding exact handle sharing
        # but let's check values.
        self.assertEqual(p[0].item(), 1.)
        
        # Modify t, check p? 
        # In current implementation, p.data = t copies metadata. 
        # If they share underlying impl, modification should reflect.
        # But tensorplay binding for setitem might copy?
        # Let's check pointer if available or modify in place.
        # t.add_(1) -> p should change.
        
    def test_requires_grad(self):
        t = tp.tensor([1.])
        p = nn.Parameter(t, requires_grad=False)
        self.assertFalse(p.requires_grad)
        
        p = nn.Parameter(t)
        self.assertTrue(p.requires_grad)

    def test_repr(self):
        p = nn.Parameter(tp.tensor([1.]))
        self.assertTrue(repr(p).startswith('Parameter containing:'))

    def test_module_registration(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(tp.tensor([1.]))
                self.t = tp.tensor([2.]) # Should not be parameter

        m = Model()
        params = list(m.named_parameters())
        self.assertEqual(len(params), 1)
        self.assertEqual(params[0][0], 'p')
        self.assertIsInstance(params[0][1], nn.Parameter)
        
        # Check buffers? 
        # t is not registered as buffer automatically unless register_buffer called?
        # PyTorch: self.t = tensor -> just attribute.
        # My implementation: object.__setattr__ -> attribute.
        self.assertNotIn('t', m._buffers)
        self.assertNotIn('t', m._parameters)
        
        m.register_buffer('b', tp.tensor([3.]))
        self.assertIn('b', m._buffers)
        
    def test_parameter_assignment(self):
        m = nn.Module()
        p = nn.Parameter(tp.tensor([1.]))
        m.p = p
        self.assertIn('p', m._parameters)
        self.assertIs(m._parameters['p'], p)
        
        # Reassign
        p2 = nn.Parameter(tp.tensor([2.]))
        m.p = p2
        self.assertIs(m._parameters['p'], p2)
        
        # Assign Tensor to parameter name -> Error?
        with self.assertRaises(TypeError):
            m.p = tp.tensor([3.])

if __name__ == '__main__':
    unittest.main()
