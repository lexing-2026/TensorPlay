"""Attribution graphs over transformer MLP features.

An attribution graph shows which features at each layer positively contribute
to a model output, and how those features connect across layers. For every
selected layer we record the post-activation MLP features, compute their
attribution to the target logit as ``gradient * activation``, keep the most
attributing features, and add edges between features in adjacent layers whose
linearized path contribution (through the layer's MLP output projection and the
next layer's MLP input projection) is positive.

The edge weights are a first-order linearized approximation of the feature-to-
feature interaction (residual connections and attention are treated implicitly
through the measured activations). This is the same spirit as the attribution
graph methodology, kept intentionally lightweight and framework-native.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorplay as tensorplay

from tensorplay.lumen.llm.patching import resolve_layer_modules


@dataclass
class GraphNode:
    """A feature node in the attribution graph.

    Attributes:
        layer: Layer index the feature belongs to.
        index: Feature index within the layer.
        attribution: ``gradient * activation`` of the feature to the target.
        value: Post-activation feature value.
    """

    layer: int
    index: int
    attribution: float
    value: float


@dataclass
class GraphEdge:
    """A directed edge between two feature nodes.

    Attributes:
        source: Node index into ``nodes``.
        target: Node index into ``nodes``.
        weight: Positive linearized contribution of the source feature to the
            target feature.
    """

    source: int
    target: int
    weight: float


@dataclass
class AttributionGraphData:
    """The built attribution graph.

    Attributes:
        nodes: List of :class:`GraphNode`.
        edges: List of :class:`GraphEdge`.
        layers: Layer indices present in the graph.
        target: Target token id the graph was built for.
    """

    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    layers: List[int] = field(default_factory=list)
    target: Optional[int] = None


def _default_feature_module(block: Any) -> Any:
    """Return the post-activation module of a transformer block's MLP."""
    mlp = getattr(block, "mlp", None)
    if mlp is None:
        raise ValueError(
            "Transformer block has no `mlp` attribute; provide a custom "
            "feature_selector."
        )
    # mlp is expected to be Sequential(Linear, activation, Linear)
    if len(mlp) >= 2:
        return mlp[1]
    return mlp[0]


def _default_input_projection(block: Any) -> Any:
    mlp = getattr(block, "mlp", None)
    if mlp is None:
        raise ValueError("Transformer block has no `mlp` attribute")
    return mlp[0].weight


def _default_output_projection(block: Any) -> Any:
    mlp = getattr(block, "mlp", None)
    if mlp is None:
        raise ValueError("Transformer block has no `mlp` attribute")
    return mlp[2].weight


class AttributionGraph:
    """Build attribution graphs over transformer MLP features.

    Args:
        model: A tensorplay transformer whose blocks expose ``mlp`` as
            ``Sequential(Linear, activation, Linear)`` (like the test model).
        feature_selector: Optional callable ``(block) -> module`` returning the
            module whose output is the feature vector (default: the activation
            inside ``mlp``).
        input_projection: Optional callable ``(block) -> weight`` returning the
            next-layer MLP input projection weight matrix.
        output_projection: Optional callable ``(block) -> weight`` returning
            the layer's MLP output projection weight matrix.
    """

    def __init__(
        self,
        model: Any,
        feature_selector: Optional[Callable[[Any], Any]] = None,
        input_projection: Optional[Callable[[Any], Any]] = None,
        output_projection: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.model = model
        self.feature_selector = feature_selector or _default_feature_module
        self.input_projection = input_projection or _default_input_projection
        self.output_projection = output_projection or _default_output_projection

    def build(
        self,
        input_ids: Any,
        target_position: int = -1,
        target_token: Optional[int] = None,
        topk: int = 10,
    ) -> AttributionGraphData:
        """Build the attribution graph for one forward pass.

        Args:
            input_ids: Token ids (tensor, list, or numpy array).
            target_position: Token position whose logit is explained.
            target_token: Token id to attribute; ``None`` uses the argmax
                prediction at ``target_position``.
            topk: Number of top features kept per layer.

        Returns:
            An :class:`AttributionGraphData`.
        """
        if isinstance(input_ids, tensorplay.Tensor):
            ids = input_ids
        else:
            ids = tensorplay.as_tensor(np.asarray(input_ids), dtype=tensorplay.int64)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        ids = ids.to(next(self.model.parameters()).device)

        layer_dict = resolve_layer_modules(self.model)
        layer_list = sorted(layer_dict)
        if not layer_list:
            raise ValueError("No transformer layers found on the model")

        # capture post-activation features per layer
        features: Dict[int, tensorplay.Tensor] = {}
        handles = []

        def make_hook(layer_idx: int) -> Any:
            module = self.feature_selector(layer_dict[layer_idx])

            def hook(mod: Any, args: Any, output: Any) -> None:
                features[layer_idx] = output

            return module.register_forward_hook(hook)

        for layer_idx in layer_list:
            handles.append(make_hook(layer_idx))

        logits: tensorplay.Tensor
        try:
            out = self.model(input_ids=ids)
            logits = out.logits
        finally:
            for handle in handles:
                handle.remove()

        seq = ids.shape[1]
        if target_position < 0:
            target_position = seq + target_position
        if not (0 <= target_position < seq):
            raise IndexError("target_position out of range")
        if target_token is None:
            target_token = int(logits[0, target_position].argmax())
        target_logit = logits[0, target_position, target_token]

        # gradient * activation attribution per layer, keep top-k positive
        node_by_layer: Dict[int, Dict[int, int]] = {}
        nodes: List[GraphNode] = []
        for layer_idx in layer_list:
            f = features[layer_idx]
            grad = tensorplay.autograd.grad(target_logit, f, retain_graph=True)[0]
            attr = (grad * f)[0, target_position]
            vals = f[0, target_position]
            order = tensorplay.argsort(attr, descending=True)
            node_by_layer[layer_idx] = {}
            kept = 0
            for idx in order.tolist():
                a = float(attr[idx])
                if a <= 0:
                    continue
                node_by_layer[layer_idx][idx] = len(nodes)
                nodes.append(
                    GraphNode(
                        layer=layer_idx,
                        index=idx,
                        attribution=a,
                        value=float(vals[idx]),
                    )
                )
                kept += 1
                if kept >= topk:
                    break

        # edges between adjacent layers (linearized path contribution)
        edges: List[GraphEdge] = []
        for l, layer_idx in enumerate(layer_list[:-1]):
            next_layer = layer_list[l + 1]
            w_out = self.output_projection(layer_dict[layer_idx]).detach()
            w_in = self.input_projection(layer_dict[next_layer]).detach()
            for src_idx, src in node_by_layer[layer_idx].items():
                src_node = nodes[src]
                src_feature_value = src_node.value
                for dst_idx, dst in node_by_layer[next_layer].items():
                    path = float(
                        tensorplay.dot(w_in[dst_idx, :], w_out[:, src_idx])
                    )
                    weight = path * src_feature_value
                    if weight > 1e-6:
                        edges.append(GraphEdge(source=src, target=dst, weight=weight))

        edges.sort(key=lambda e: e.weight, reverse=True)
        return AttributionGraphData(
            nodes=nodes,
            edges=edges,
            layers=layer_list,
            target=target_token,
        )

    def visualize(
        self,
        graph: AttributionGraphData,
        figsize: Optional[Sequence[float]] = None,
        max_edges: int = 60,
        use_pyplot: bool = False,
    ) -> Any:
        """Render the attribution graph as a layered DAG.

        Nodes are placed by layer (x axis) and sorted by attribution within each
        layer. Node size and color encode attribution; edges are drawn with
        width proportional to their weight.

        Args:
            graph: The graph to render.
            figsize: Optional ``(width, height)``.
            max_edges: Maximum number of edges to draw (keeps the plot readable).
            use_pyplot: If ``True`` calls ``plt.show()``.

        Returns:
            ``(fig, ax)`` matplotlib objects.
        """
        import matplotlib.pyplot as plt

        if not graph.nodes:
            raise ValueError("The graph is empty; build it before visualizing.")

        layers = graph.layers
        x_pos = {layer: i for i, layer in enumerate(layers)}
        # y position: rank within layer by attribution
        per_layer: Dict[int, List[int]] = {layer: [] for layer in layers}
        for n, node in enumerate(graph.nodes):
            per_layer[node.layer].append(n)
        y_pos: Dict[int, float] = {}
        for layer, node_indices in per_layer.items():
            node_indices.sort(
                key=lambda n: graph.nodes[n].attribution, reverse=True
            )
            for rank, n in enumerate(node_indices):
                y_pos[n] = len(node_indices) - rank - 1

        fig, ax = plt.subplots(figsize=figsize if figsize is not None else (10, 7))

        max_attr = max((n.attribution for n in graph.nodes), default=1.0)
        # draw edges first (under nodes)
        for edge in graph.edges[:max_edges]:
            sx, sy = x_pos[graph.nodes[edge.source].layer], y_pos[edge.source]
            tx, ty = x_pos[graph.nodes[edge.target].layer], y_pos[edge.target]
            ax.plot(
                [sx, tx],
                [sy, ty],
                color="tab:blue",
                alpha=min(1.0, 0.25 + edge.weight / max_attr),
                linewidth=0.5 + 3.0 * edge.weight / max_attr,
                zorder=1,
            )

        # draw nodes
        for n, node in enumerate(graph.nodes):
            size = 80 + 300 * node.attribution / max_attr
            color = "tab:red" if node.attribution > 0 else "tab:gray"
            ax.scatter(
                x_pos[node.layer],
                y_pos[n],
                s=size,
                c=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=2,
            )
            ax.annotate(
                str(node.index),
                (x_pos[node.layer], y_pos[n]),
                textcoords="offset points",
                xytext=(0, 0),
                ha="center",
                va="center",
                fontsize=6,
                zorder=3,
            )

        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels(["L%d" % l for l in layers])
        ax.set_yticks([])
        ax.set_xlabel("layer")
        ax.set_title(
            "Attribution graph (target token id %s)"
            % ("?" if graph.target is None else str(graph.target))
        )
        fig.tight_layout()
        if use_pyplot:
            plt.show()
        return fig, ax