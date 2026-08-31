import tensorplay

try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

def make_dot(var, params=None):
    """
    If a node is a Variable (requires_grad=True), it will be blue.
    If a node is an operation (grad_fn), it will be gray.
    
    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: implement param naming)
    """
    if params is not None:
        assert isinstance(params.values().__iter__().__next__(), tensorplay.Tensor)
        # Use _impl_id for mapping to handle Python wrapper differences
        param_map = {v._impl_id: k for k, v in params.items()}
    else:
        param_map = {}

    if HAS_GRAPHVIZ:
        return _make_dot_graphviz(var, param_map)
    elif HAS_NETWORKX:
        print("Graphviz not found. Falling back to NetworkX + Matplotlib.")
        return _make_dot_networkx(var, param_map)
    else:
        raise RuntimeError("Neither graphviz nor networkx+matplotlib found. Cannot visualize graph.")

def _make_dot_graphviz(var, param_map):
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = graphviz.Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, tensorplay.Tensor):
                # This is a leaf variable or output
                node_id = str(id(var))
                if var.requires_grad:
                    # Leaf node (parameter)
                    name = param_map.get(var._impl_id, f"Tensor\n{size_to_str(var.shape)}")
                    dot.node(node_id, name, fillcolor='lightblue')
                else:
                    # Just a tensor (maybe output)
                    dot.node(node_id, f"Output\n{size_to_str(var.shape)}", fillcolor='lightgreen')
                
                if var.grad_fn:
                    add_nodes(var.grad_fn)
                    dot.edge(str(id(var.grad_fn)), node_id)
                
                seen.add(var)
            elif hasattr(var, 'next_functions'):
                # This is a Node (grad_fn)
                
                # Check if it is an AccumulateGrad node wrapping a leaf tensor
                leaf_tensor = getattr(var, 'variable', None)
                if leaf_tensor is not None:
                    # Treat as leaf tensor
                    node_id = str(id(var)) # Use Node ID, but label as Tensor
                    name = param_map.get(leaf_tensor._impl_id, f"Tensor\n{size_to_str(leaf_tensor.shape)}")
                    dot.node(node_id, name, fillcolor='lightblue')
                    seen.add(var)
                    return # AccumulateGrad has no next edges we care about for viz usually

                node_id = str(id(var))
                name = var.name
                # Clean up name
                if "::" in name:
                    name = name.split("::")[-1]
                if name.startswith("struct "):
                    name = name[7:]
                
                dot.node(node_id, name, fillcolor='white')
                seen.add(var)
                
                for fn, _ in var.next_functions:
                    if fn is not None:
                        add_nodes(fn)
                        dot.edge(str(id(fn)), node_id)

    add_nodes(var)
    return dot

def _make_dot_networkx(var, param_map):
    G = nx.DiGraph()
    labels = {}
    colors = []
    
    seen = set()
    
    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if isinstance(var, tensorplay.Tensor):
                node_id = id(var)
                if var.requires_grad:
                    name = param_map.get(var._impl_id, f"Tensor\n{size_to_str(var.shape)}")
                    labels[node_id] = name
                    G.add_node(node_id, color='lightblue', style='filled')
                else:
                    labels[node_id] = f"Output\n{size_to_str(var.shape)}"
                    G.add_node(node_id, color='lightgreen', style='filled') # Use distinct color for output
                
                if var.grad_fn:
                    add_nodes(var.grad_fn)
                    G.add_edge(id(var.grad_fn), node_id)
                
                seen.add(var)
            elif hasattr(var, 'next_functions'):
                # Check if it is an AccumulateGrad node wrapping a leaf tensor
                leaf_tensor = getattr(var, 'variable', None)
                if leaf_tensor is not None:
                    # Treat as leaf tensor
                    node_id = id(var)
                    name = param_map.get(leaf_tensor._impl_id, f"Tensor\n{size_to_str(leaf_tensor.shape)}")
                    labels[node_id] = name
                    G.add_node(node_id, color='lightblue', style='filled')
                    seen.add(var)
                    return # AccumulateGrad has no next edges we care about for viz usually

                node_id = id(var)
                name = var.name
                if "::" in name:
                    name = name.split("::")[-1]
                if name.startswith("struct "):
                    name = name[7:]
                
                labels[node_id] = name
                G.add_node(node_id, color='lightgray', style='filled')
                seen.add(var)
                
                for fn, _ in var.next_functions:
                    if fn is not None:
                        add_nodes(fn)
                        G.add_edge(id(fn), node_id)

    add_nodes(var)
    
    # Compute layout
    try:
        # Use topological generations for hierarchical layout
        generations = list(nx.topological_generations(G))
        for i, gen in enumerate(generations):
            for node in gen:
                G.nodes[node]['subset'] = i
        
        # Initial layout: left to right
        pos = nx.multipartite_layout(G, subset_key='subset', align='horizontal')
        
        # Rotate to Top-Down (optional, but standard for trees)
        # Swap x and y so that subset 0 is at Top (y=max)
        # multipartite_layout assigns x based on subset.
        # So x is increasing with depth.
        # We want y to be decreasing with depth.
        for node in pos:
            x, y = pos[node]
            pos[node] = (y, -x)
            
    except Exception as e:
        print(f"Hierarchical layout failed, falling back to spring layout: {e}")
        pos = nx.spring_layout(G)

    node_colors = [G.nodes[n].get('color', 'white') for n in G.nodes]
    
    plt.figure(figsize=(15, 15)) # Increase figure size
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrowsize=20, arrowstyle='-|>', width=1.5)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=4000, alpha=0.9, edgecolors='black', linewidths=1.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    plt.title("Computation Graph", fontsize=20)
    plt.axis('off')
    
    class Result:
        def render(self, filename, format='png', dpi=300):
            plt.savefig(f"{filename}.{format}", bbox_inches='tight', dpi=dpi)
            print(f"Graph saved to {filename}.{format}")
            plt.close() # Close figure to free memory
            
    return Result()
