import tensorplay as tp
import tensorplay.stax as stax
try:
    import onnx
    from onnx import helper, TensorProto
    from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
    import numpy as np
except ImportError:
    onnx = None

def export(model, args, f, input_names=None, output_names=None, opset_version=13):
    """
    Export a model to ONNX format.
    
    Args:
        model: callable (nn.Module or function)
        args: tuple of input tensors
        f: file path or file-like object
        input_names: list of input names (optional)
        output_names: list of output names (optional)
        opset_version: int (default 13)
    """
    if onnx is None:
        raise ImportError("onnx package is required for export")

    tracer = stax.Tracer()
    
    # 1. Prepare inputs
    proxy_args = []
    onnx_inputs = []
    
    # Ensure args is a tuple
    if isinstance(args, tp.Tensor):
        args = (args,)
    
    for i, arg in enumerate(args):
        if isinstance(arg, tp.Tensor):
            proxy = tracer.create_input(arg)
            proxy_args.append(proxy)
            
            # Create ONNX ValueInfo for input
            name = input_names[i] if input_names and i < len(input_names) else f"input_{i}"
            
            # Map dtype
            # We use numpy dtype mapping
            np_dtype = arg.numpy().dtype
            onnx_dtype = NP_TYPE_TO_TENSOR_TYPE[np_dtype]
            
            shape = list(arg.shape)
            
            vi = helper.make_tensor_value_info(name, onnx_dtype, shape)
            onnx_inputs.append(vi)
            
            # Map tracer input value id to name
            # We assume tracer.inputs order matches args order
            # But we need to map the internal Value ID to ONNX name
            # proxy.node is the Value object (from stax)
            # We'll use a mapping dict
        else:
            proxy_args.append(arg)
            
    # Map Value ID -> Name
    val_id_to_name = {}
    for i, proxy in enumerate(proxy_args):
        if isinstance(proxy, stax.ProxyTensor):
            name = input_names[i] if input_names and i < len(input_names) else f"input_{i}"
            val_id_to_name[proxy.node.id] = name

    # 2. Trace
    with stax.patch_tensorplay(tracer):
        out = model(*proxy_args)
        
    # 3. Build ONNX Nodes
    onnx_nodes = []
    
    # Helper to get name for a value
    def get_name(val):
        if val.id not in val_id_to_name:
            val_id_to_name[val.id] = f"val_{val.id}"
        return val_id_to_name[val.id]

    # Handle output
    onnx_outputs = []
    if isinstance(out, stax.ProxyTensor):
        outs = [out]
    elif isinstance(out, (tuple, list)):
        outs = out
    else:
        outs = []
        
    for i, o in enumerate(outs):
        if isinstance(o, stax.ProxyTensor):
            name = output_names[i] if output_names and i < len(output_names) else f"output_{i}"
            val_id_to_name[o.node.id] = name
            
            # ValueInfo
            # We don't easily know shape/dtype of intermediate nodes from stax tracer unless we propagate them.
            # Stax tracer doesn't propagate shape/dtype in Python currently (ProxyTensor has shape=None usually).
            # But for ONNX output, we assume shape inference or just use Any.
            # Ideally we run the model with real data to get shapes, but stax tracer is symbolic.
            # We can run the model normally to get shapes if needed, but for now let's skip shape info for output or infer it.
            # ONNX requires output type info.
            
            # Workaround: Run the model with real data to get output shapes/dtypes?
            # Or just set to unknown?
            # Let's try to set to Float/Unknown shape for now.
            onnx_outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None)) 
        
    # Iterate nodes
    # stax.Tracer.graph.nodes is iterable
    for node in tracer.graph.nodes:
        kind = node.kind
        inputs = [get_name(v) for v in node.inputs]
        outputs = [get_name(v) for v in node.outputs]
        
        # Attributes
        # stax nodes might have attributes like arg_0, arg_1 for scalars.
        # We need to extract them.
        # Currently stax python bindings don't expose iterating attributes easily, 
        # but we know what attributes we set in `stax.py`.
        
        onnx_attrs = {}
        
        # Mapping stax ops to ONNX ops
        op_type = kind
        if kind == "matmul" or kind == "mm":
            op_type = "MatMul"
        elif kind == "add":
            op_type = "Add"
        elif kind == "sub":
            op_type = "Sub"
        elif kind == "mul":
            op_type = "Mul"
        elif kind == "div":
            op_type = "Div"
        elif kind == "pow":
            op_type = "Pow"
        elif kind == "exp":
            op_type = "Exp"
        elif kind == "sqrt":
            op_type = "Sqrt"
        elif kind == "cos":
            op_type = "Cos"
        elif kind == "sin":
            op_type = "Sin"
        elif kind == "relu":
            op_type = "Relu"
        # Add more mappings as needed
        
        onnx_node = helper.make_node(
            op_type,
            inputs,
            outputs,
            **onnx_attrs
        )
        onnx_nodes.append(onnx_node)

    # 4. Create Graph
    graph_def = helper.make_graph(
        onnx_nodes,
        "tensorplay_model",
        onnx_inputs,
        onnx_outputs,
    )
    
    # 5. Create Model
    model_def = helper.make_model(graph_def, producer_name="tensorplay", opset_imports=[helper.make_opsetid("", opset_version)])
    
    # 6. Save
    if isinstance(f, str):
        onnx.save(model_def, f)
    else:
        f.write(model_def.SerializeToString())

