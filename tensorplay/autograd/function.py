import tensorplay
import tensorplay._C._autograd as _autograd
import types


class _Context:
    """
    Records information needed for computing gradients.
    """
    def __init__(self):
        self.saved_tensors = ()
        self.backward_fn = None
        
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


class Function:
    """
    Records operation history and defines formulas for differentiating ops.
    """
    
    @staticmethod
    def forward(ctx, *args, **kwargs):
        """
        Performs the operation.
        """
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Defines a formula for differentiating the operation.
        """
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        # Check if grad is needed
        grad_enabled = _autograd.is_grad_enabled()
        
        # Check if any input requires grad
        needs_grad = False
        if grad_enabled:
            for arg in args:
                if isinstance(arg, tensorplay.Tensor) and arg.requires_grad:
                    needs_grad = True
                    break
        
        # Create context
        ctx = _Context()
        
        # Run forward with grad disabled
        _autograd.set_grad_enabled(False)
        try:
            output = cls.forward(ctx, *args, **kwargs)
        finally:
            _autograd.set_grad_enabled(grad_enabled)
            
        if not grad_enabled or not needs_grad:
            return output
            
        # Create PyNode
        ctx.backward_fn = cls.backward
        
        def backward_wrapper(self, *grads):
            return self.backward_fn(self, *grads)
        
        ctx.backward = types.MethodType(backward_wrapper, ctx)
        
        fn = _autograd.PyNode(ctx)
        
        # Connect inputs
        for arg in args:
            if isinstance(arg, tensorplay.Tensor):
                if arg.requires_grad:
                    edges = _autograd.collect_next_edges(arg)
                    if edges:
                        for edge in edges:
                             # edge is (Node, int) pair
                             fn.add_next_edge(edge[0], edge[1])
                    else:
                        fn.add_next_edge(None)
                else:
                    fn.add_next_edge(None)
            else:
                # Non-tensor arg
                fn.add_next_edge(None)
                    
        # Set grad_fn for outputs
        if isinstance(output, tensorplay.Tensor):
            output.requires_grad = True
            output._set_grad_fn(fn, 0)
        elif isinstance(output, (list, tuple)):
            for i, out in enumerate(output):
                if isinstance(out, tensorplay.Tensor):
                    out.requires_grad = True
                    out._set_grad_fn(fn, i)
        
        return output
