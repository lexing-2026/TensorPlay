
import tensorplay
from tensorplay import _C
import tensorplay.nn as nn
import functools
import inspect
import builtins
import operator
import types
from typing import Any, List, Dict, Tuple, Optional, Union, Callable, Set, Iterator
from .backends.triton import TritonBackend, HAS_TRITON

# Try to import C++ Stax if available
_stax_cpp = getattr(_C, "_stax", None)

# --- Core FX Components (Python Layer) ---

class Node:
    """
    Represents an operation in the computational graph.
    Wraps C++ Node if available for high-performance manipulation.
    """
    def __init__(self, graph: 'Graph', name: str, op: str, target: Any, args: Tuple[Any, ...] = (), kwargs: Dict[str, Any] = {}, cpp_node: Optional[Any] = None):
        self.graph = graph
        self.name = name
        self.op = op 
        self.target = target
        self.args = args
        self.kwargs = kwargs
        
        # C++ Node Binding
        self._cpp_node = cpp_node
        
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None
        
        self.users: Dict['Node', None] = {}
        self.meta: Dict[str, Any] = {}

    def __repr__(self):
        return f"{self.name} = {self.op}(target={self.target}, args={self.args}, kwargs={self.kwargs})"

    def append(self, node: 'Node'):
        self.users[node] = None
        # if self._cpp_node and node._cpp_node:
        #     # C++ graph maintains its own topology via add_input/add_output
        #     pass

    def replace_all_uses_with(self, replace_with: 'Node'):
        """Replace all uses of this node with another node."""
        for user in list(self.users.keys()):
            new_args = tuple(replace_with if arg is self else arg for arg in user.args)
            new_kwargs = {k: (replace_with if v is self else v) for k, v in user.kwargs.items()}
            user.args = new_args
            user.kwargs = new_kwargs
            
            del self.users[user]
            replace_with.users[user] = None

class Graph:
    """
    Container for the computational graph.
    Syncs with C++ Graph for heavy-lifting optimization passes.
    """
    def __init__(self):
        self.nodes: List[Node] = []
        self._node_name_counter = 0
        self._root: Optional[Node] = None
        
        # C++ Graph Binding
        self._cpp_graph = _stax_cpp.Graph() if _stax_cpp else None
        # Map Python Node -> C++ Value (for wiring inputs)
        self._py_to_cpp_map = {}

    def create_node(self, op: str, target: Any, args: Tuple[Any, ...] = (), kwargs: Dict[str, Any] = {}, name: Optional[str] = None) -> Node:
        # Normalize args
        args = tuple(arg.node if isinstance(arg, Proxy) else arg for arg in args)
        kwargs = {k: (v.node if isinstance(v, Proxy) else v) for k, v in kwargs.items()}

        if name is None:
            name = f"_{self._node_name_counter}"
            self._node_name_counter += 1
            
        # Create C++ Node if backend is active
        cpp_node = None
        if self._cpp_graph:
            if op == 'placeholder':
                # Input to graph
                cpp_value = self._cpp_graph.add_input()
                self._py_to_cpp_map[name] = cpp_value # Use name as key temporarily
                # Placeholder doesn't need a Node in C++ IR usually, it's a Value
                # But for consistency, maybe create a Node? 
                # Our C++ IR seems to treat Inputs as Values owned by Graph
                pass
            elif op == 'call_function':
                # Create generic node
                kind = target.__name__ if hasattr(target, '__name__') else str(target)
                cpp_node = self._cpp_graph.create_node(kind)
                
                # Wire inputs
                for arg in args:
                    if isinstance(arg, Node) and arg.name in self._py_to_cpp_map:
                         cpp_node.add_input(self._py_to_cpp_map[arg.name])
                
                # Create output value
                out_val = cpp_node.add_output()
                self._py_to_cpp_map[name] = out_val
            elif op == 'output':
                # Mark outputs
                for arg in args:
                    if isinstance(arg, Node) and arg.name in self._py_to_cpp_map:
                        self._cpp_graph.register_output(self._py_to_cpp_map[arg.name])

        node = Node(self, name, op, target, args, kwargs, cpp_node)
        self.nodes.append(node)
        
        # Maintain Py topology
        for arg in args:
            if isinstance(arg, Node):
                arg.append(node)
        for arg in kwargs.values():
            if isinstance(arg, Node):
                arg.append(node)
                
        return node
    
    def run_cpp_fusion(self):
        """Run C++ fusion pass and reflect changes back to Python?"""
        if self._cpp_graph:
            self._cpp_graph.fuse()
            # Note: Reflecting C++ topology changes back to Python FX graph 
            # is complex (requires bidirectional binding). 
            # Usually we lower to C++ and stay there for codegen.


    def placeholder(self, name: str) -> Node:
        return self.create_node('placeholder', name, name=name)

    def call_function(self, target: Any, args: Tuple[Any, ...] = (), kwargs: Dict[str, Any] = {}) -> Node:
        return self.create_node('call_function', target, args, kwargs)

    def call_module(self, target: str, args: Tuple[Any, ...] = (), kwargs: Dict[str, Any] = {}) -> Node:
        return self.create_node('call_module', target, args, kwargs)

    def get_attr(self, target: str) -> Node:
        return self.create_node('get_attr', target)

    def output(self, result: Any) -> Node:
        return self.create_node('output', 'output', args=(result,))

    def python_code(self, root_module: str = 'self') -> str:
        """
        Generate valid Python code from the graph.
        """
        lines = []
        lines.append(f"def forward({root_module}, {', '.join([n.target for n in self.nodes if n.op == 'placeholder'])}):")
        
        for node in self.nodes:
            if node.op == 'placeholder':
                continue
            elif node.op == 'output':
                # Handle return
                ret_val = self._format_arg(node.args[0])
                lines.append(f"    return {ret_val}")
                continue
            
            # RHS
            if node.op == 'call_function':
                func_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
                if hasattr(node.target, '__module__') and node.target.__module__ != 'builtins':
                     # Simplify for stax ops
                     if node.target in OP_MAP.values():
                         # Reverse lookup op map? or just use operator
                         pass 
                
                args_str = ", ".join([self._format_arg(a) for a in node.args])
                line = f"{node.name} = {func_name}({args_str})"
            elif node.op == 'call_module':
                args_str = ", ".join([self._format_arg(a) for a in node.args])
                line = f"{node.name} = {root_module}.{node.target}({args_str})"
            elif node.op == 'get_attr':
                line = f"{node.name} = {root_module}.{node.target}"
            else:
                line = f"{node.name} = # Unknown op {node.op}"
            
            lines.append(f"    {line}")
            
        return "\n".join(lines)

    def _format_arg(self, arg):
        if isinstance(arg, Node):
            return arg.name
        elif isinstance(arg, (tuple, list)):
            return f"({', '.join([self._format_arg(x) for x in arg])})"
        else:
            return str(arg)

class Proxy:
    def __init__(self, node: Node, tracer: 'Tracer'):
        self.node = node
        self.tracer = tracer

    def __repr__(self):
        return f"Proxy({self.node.name})"

    # --- Operator Overloading ---
    def _binary_op(self, other, op_name):
        return self.tracer.create_proxy('call_function', OP_MAP[op_name], (self, other), {})

    def __add__(self, other): return self._binary_op(other, 'add')
    def __sub__(self, other): return self._binary_op(other, 'sub')
    def __mul__(self, other): return self._binary_op(other, 'mul')
    def __truediv__(self, other): return self._binary_op(other, 'div')
    
    def __radd__(self, other): return self.tracer.create_proxy('call_function', OP_MAP['add'], (other, self), {})
    def __rsub__(self, other): return self.tracer.create_proxy('call_function', OP_MAP['sub'], (other, self), {})
    def __rmul__(self, other): return self.tracer.create_proxy('call_function', OP_MAP['mul'], (other, self), {})
    
    def __neg__(self): return self.tracer.create_proxy('call_function', OP_MAP['neg'], (self,), {})

    def __getitem__(self, key):
        return self.tracer.create_proxy('call_function', operator.getitem, (self, key), {})

    def __pow__(self, other): return self._binary_op(other, 'pow')

    def __getattr__(self, name):
        # Support for tensor methods not explicitly defined
        # We assume accessed attributes are methods to be called
        def method(*args, **kwargs):
            return self.tracer.create_proxy('call_method', name, (self,) + args, kwargs)
        return method

    def __call__(self, *args, **kwargs):
        # Support for calling the result of a node (if it's a callable)
        return self.tracer.create_proxy('call_function', self, args, kwargs)

    # Tensor Methods
    def sin(self): return self.tracer.create_proxy('call_function', tensorplay.sin, (self,), {})
    def cos(self): return self.tracer.create_proxy('call_function', tensorplay.cos, (self,), {})
    def exp(self): return self.tracer.create_proxy('call_function', tensorplay.exp, (self,), {})
    def relu(self): return self.tracer.create_proxy('call_function', tensorplay.relu, (self,), {})

class Tracer:
    def __init__(self):
        self.graph = Graph()
        self.root = None

    def create_proxy(self, kind, target, args, kwargs):
        node = self.graph.create_node(kind, target, args, kwargs)
        return Proxy(node, self)

    def trace(self, root: Union[Callable, nn.Module], concrete_args=None):
        self.root = root
        
        if isinstance(root, nn.Module):
            return self._trace_module(root)
        else:
            return self._trace_function(root)

    def _trace_function(self, func):
        sig = inspect.signature(func)
        proxies = []
        for param in sig.parameters.values():
            node = self.graph.placeholder(param.name)
            proxies.append(Proxy(node, self))
            
        output = func(*proxies)
        
        if isinstance(output, (tuple, list)):
            self.graph.output(tuple(p.node if isinstance(p, Proxy) else p for p in output))
        else:
            node = output.node if isinstance(output, Proxy) else output
            self.graph.output(node)
            
        return self.graph

    def _trace_module(self, module):
        # Simple symbolic tracing for Module
        # We wrap module's forward
        
        # 1. Placeholders
        forward = module.forward
        sig = inspect.signature(forward)
        proxies = []
        for param in sig.parameters.values():
            if param.name == 'self': continue
            node = self.graph.placeholder(param.name)
            proxies.append(Proxy(node, self))
            
        # 2. Patch Submodules (Mocking)
        # In a real FX implementation, we would recursively patch or use __torch_function__
        # Here we do a shallow patch of submodules to return Proxies
        
        original_modules = {}
        # We need to bypass __setattr__ check of nn.Module which enforces type checking
        # So we use __dict__ directly or object.__setattr__
        
        for name, submod in module.named_children():
            original_modules[name] = submod
            # Create a mock callable that emits call_module
            def make_proxy_caller(name):
                def proxy_caller(*args, **kwargs):
                    return self.create_proxy('call_module', name, args, kwargs)
                return proxy_caller
            
            # Use object.__setattr__ to bypass Module.__setattr__ type check
            object.__setattr__(module, name, make_proxy_caller(name))

        # 3. Run Forward
        try:
            output = module.forward(*proxies)
        finally:
            # Restore
            for name, submod in original_modules.items():
                object.__setattr__(module, name, submod)
        
        # 4. Output
        if isinstance(output, (tuple, list)):
             self.graph.output(tuple(p.node if isinstance(p, Proxy) else p for p in output))
        else:
            node = output.node if isinstance(output, Proxy) else output
            self.graph.output(node)
            
        return self.graph

class GraphModule(nn.Module):
    """
    Generated Module that holds a Graph and executes it.
    """
    def __init__(self, root: Union[nn.Module, Dict[str, Any]], graph: Graph):
        super().__init__()
        self.graph = graph
        
        # Copy attributes from root if it's a module
        if isinstance(root, nn.Module):
            # We need to handle parameters and buffers carefully.
            # GraphModule usually flattens the hierarchy or references the root.
            # PyTorch FX GraphModule does:
            # 1. Copies params/buffers/submodules to self
            # 2. But we need to handle nested names like 'layer1.weight'
            # register_parameter doesn't allow '.', so we need to recreate hierarchy or flatten names?
            # FX GraphModule actually maintains the module hierarchy if possible OR copies them.
            # For simplicity in this PoC, we will just alias the root's components 
            # by iterating and setting attributes, effectively flattening for access 
            # OR we respect the hierarchy by adding submodules.
            
            # Simple approach: Replicate submodule structure
            # But wait, trace() captured 'linear' as a call_module target.
            # So self.linear must exist and be a Module.
            
            for name, child in root.named_children():
                self.add_module(name, child)
                
            # For parameters of the root itself (not in children)
            for name, param in root.named_parameters(recurse=False):
                self.register_parameter(name, param)
                
            for name, buf in root.named_buffers(recurse=False):
                self.register_buffer(name, buf)
        
        # Generate Python Code (for debugging/interpreting)
        self.code = graph.python_code()
        
        # Compile the graph to a python callable (Interpreter)
        # For performance, we should compile this to a backend.
        # But GraphModule by default acts as an interpreter or runs generated python code.
        self._recompile()

    def _recompile(self):
        # Create a Python function from the graph code
        # This is the "Python Backend"
        exec_globals = globals().copy()
        exec_globals.update(OP_MAP) # Ensure ops are available
        exec_globals['tensorplay'] = tensorplay
        
        # We need to bind 'self' to access modules/params
        # But exec creates a function. We can bind it later.
        
        # NOTE: This naive exec works for simple cases. 
        # Real FX uses more robust loading.
        
        # For now, we'll use a simple Interpreter instead of exec-ing string code 
        # because resolving globals/closure is tricky in this snippet.
        pass

    def forward(self, *args):
        return Interpreter(self).run(*args)

class Interpreter:
    """
    Executes a Graph node-by-node.
    """
    def __init__(self, module: GraphModule):
        self.module = module
        self.graph = module.graph
        self.env = {}

    def run(self, *args):
        # Bind args to placeholders
        placeholders = [n for n in self.graph.nodes if n.op == 'placeholder']
        if len(args) != len(placeholders):
            raise RuntimeError(f"Expected {len(placeholders)} args, got {len(args)}")
        
        for node, arg in zip(placeholders, args):
            self.env[node] = arg
            
        # Execute
        for node in self.graph.nodes:
            if node.op == 'placeholder':
                continue
            
            if node.op == 'call_function':
                func = node.target
                args = self._fetch_args(node.args)
                kwargs = self._fetch_kwargs(node.kwargs)
                self.env[node] = func(*args, **kwargs)
                
            elif node.op == 'call_module':
                mod_name = node.target
                mod = self.module.get_submodule(mod_name)
                args = self._fetch_args(node.args)
                kwargs = self._fetch_kwargs(node.kwargs)
                self.env[node] = mod(*args, **kwargs)
                
            elif node.op == 'get_attr':
                attr_name = node.target
                # Look up in module
                # Naive lookup
                val = self.module
                for atom in attr_name.split('.'):
                    val = getattr(val, atom)
                self.env[node] = val
                
            elif node.op == 'output':
                return self._fetch_args(node.args)[0]
                
        return None

    def _fetch_args(self, args):
        return tuple(self.env[arg] if isinstance(arg, Node) else arg for arg in args)

    def _fetch_kwargs(self, kwargs):
        return {k: (self.env[v] if isinstance(v, Node) else v) for k, v in kwargs.items()}

# --- Optimization Passes ---

def dead_code_elimination(graph: Graph):
    """
    Remove nodes that have no users and are not side-effecting.
    """
    # Compute users
    # We maintain users in Node, but let's refresh to be safe or just use it.
    # We iterate backwards
    changed = False
    for node in reversed(graph.nodes):
        if node.op == 'output':
            continue
        if not node.users and node.op != 'placeholder': # Placeholders are args
            # Remove
            # Also remove from inputs' users
            for arg in node.args:
                if isinstance(arg, Node):
                    del arg.users[node]
            graph.nodes.remove(node)
            changed = True
    return changed

# --- Public API ---

def trace(root: Union[Callable, nn.Module]) -> GraphModule:
    tracer = Tracer()
    graph = tracer.trace(root)
    return GraphModule(root, graph)

from .backends.cpu import CPUBackend

def _compile_impl(func, backend):
    compiled_fn = None
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal compiled_fn
        if compiled_fn is None:
            # 1. Trace -> GraphModule
            gm = trace(func)
            
            # 2. Optimization Passes
            dead_code_elimination(gm.graph)
            
            # 3. Backend Compilation
            if backend == "triton":
                be = TritonBackend()
                compiled_fn = be.compile(gm.graph, args)
            elif backend == "cpu":
                be = CPUBackend()
                # CPU backend can execute Graph or GraphModule
                compiled_fn = be.compile(gm.graph, args)
            else:
                # Fallback to Interpreter
                compiled_fn = gm.forward
        
        return compiled_fn(*args, **kwargs)
    return wrapper

def compile(func_or_backend=None, backend=None):
    if backend is None:
        backend = "triton" if HAS_TRITON else "cpu"
        
    if func_or_backend is None:
        # @compile()
        def decorator(func):
            return _compile_impl(func, backend)
        return decorator
    
    if callable(func_or_backend):
        # @compile
        return _compile_impl(func_or_backend, backend)
    
    # @compile("cpu")
    backend = func_or_backend
    def decorator(func):
        return _compile_impl(func, backend)
    return decorator

# --- Robust Ops Mapping ---
OP_MAP = {
    'add': operator.add,
    'sub': operator.sub,
    'mul': operator.mul,
    'div': operator.truediv,
    'neg': operator.neg,
    'sin': tensorplay.sin,
    'cos': tensorplay.cos,
    'exp': tensorplay.exp,
    'relu': tensorplay.relu,
    'pow': operator.pow,
}
