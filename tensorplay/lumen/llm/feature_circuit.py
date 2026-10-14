"""Feature circuits: causal feature-level circuit discovery.

Given a model, a target behavior, and a set of candidate feature nodes (MLP
features and attention heads), this module measures the direct effect of every
node on the target by zero-ablating it, and estimates pairwise edge weights
between the most influential nodes using joint-ablation mediation (the overlap
between two nodes' direct effects). The result is a sparse circuit graph: the
nodes that matter for the behavior and the interactions between them.

This is a lightweight, framework-native implementation of the feature-circuit
methodology. Edge weights are symmetric overlap scores from co-ablation, which
is a first-order approximation of edge attribution patching.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorplay as tensorplay

from tensorplay.lumen.llm._attention_utils import (
    compute_head_outputs,
    head_output_contribution,
)
from tensorplay.lumen.llm.patching import resolve_layer_modules


@dataclass
class CircuitNode:
    """A feature node in the circuit.

    Attributes:
        layer: Layer index.
        kind: ``"mlp"`` or ``"attn"``.
        index: Feature index (MLP) or head index (attention).
        direct_effect: Measured effect of zero-ablating the node on the target.
        name: Human-readable name.
    """

    layer: int
    kind: str
    index: int
    direct_effect: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = "L%d.%s.%d" % (self.layer, self.kind, self.index)


@dataclass
class CircuitEdge:
    """A symmetric interaction edge between two circuit nodes.

    Attributes:
        source: Node index into ``nodes``.
        target: Node index into ``nodes``.
        weight: Joint-ablation overlap of the two nodes' direct effects.
    """

    source: int
    target: int
    weight: float


@dataclass
class FeatureCircuitData:
    """The discovered feature circuit.

    Attributes:
        nodes: List of :class:`CircuitNode`.
        edges: List of :class:`CircuitEdge`.
        clean_score: Target value on the unperturbed forward pass.
        target: Optional target token id the circuit explains.
    """

    nodes: List[CircuitNode] = field(default_factory=list)
    edges: List[CircuitEdge] = field(default_factory=list)
    clean_score: float = 0.0
    target: Optional[int] = None


def _mlp_feature_module(block: Any) -> Any:
    mlp = getattr(block, "mlp", None)
    if mlp is None:
        raise ValueError(
            "Transformer block has no `mlp` attribute; provide custom node hooks."
        )
    return mlp[1] if len(mlp) >= 2 else mlp[0]


def top_mlp_features(
    model: Any,
    input_ids: Any,
    target_position: int = -1,
    topk: int = 5,
) -> List[Tuple[int, str, int]]:
    """Select the most attributing MLP features per layer.

    Uses ``gradient * activation`` of the post-activation MLP features with
    respect to the argmax logit at ``target_position``.

    Returns:
        List of ``(layer, "mlp", feature_index)`` node specs.
    """
    from tensorplay.lumen.llm.attribution_graph import AttributionGraph

    ag = AttributionGraph(model)
    graph = ag.build(input_ids, target_position=target_position, topk=topk)
    result: List[Tuple[int, str, int]] = []
    for node in graph.nodes:
        result.append((node.layer, "mlp", node.index))
    return result


def top_attention_heads(
    model: Any,
    input_ids: Any,
    target_position: int = -1,
    topk: int = 5,
) -> List[Tuple[int, str, int]]:
    """Select the most attributing attention heads per layer.

    The attribution of a head is the sum of ``gradient * head_output`` over its
    head dimension at ``target_position``.

    Returns:
        List of ``(layer, "attn", head_index)`` node specs.
    """
    if isinstance(input_ids, tensorplay.Tensor):
        ids = input_ids
    else:
        ids = tensorplay.as_tensor(np.asarray(input_ids), dtype=tensorplay.int64)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    ids = ids.to(next(model.parameters()).device)

    layer_dict = resolve_layer_modules(model)
    layer_list = sorted(layer_dict)
    seq = ids.shape[1]
    if target_position < 0:
        target_position = seq + target_position

    captured: Dict[int, Tuple[Any, Any, Any]] = {}  # layer -> (h_in, weights, attn_out)
    handles = []

    def make_hooks(layer_idx: int, block: Any) -> List[Any]:
        mha = block.attn
        store: Dict[str, Any] = {}

        def pre(module: Any, args: Any) -> None:
            store["in"] = args[0]

        def post(module: Any, args: Any, out: Any) -> None:
            captured[layer_idx] = (store["in"], out[1], out[0])

        return [
            mha.register_forward_pre_hook(pre),
            mha.register_forward_hook(post),
        ]

    for layer_idx in layer_list:
        handles.extend(make_hooks(layer_idx, layer_dict[layer_idx]))

    out = model(input_ids=ids)
    best_id = int(out.logits[0, target_position].argmax())
    target_logit = out.logits[0, target_position, best_id]

    result: List[Tuple[int, str, int]] = []
    try:
        for layer_idx in layer_list:
            h_in, weights, attn_out = captured[layer_idx]
            mha = layer_dict[layer_idx].attn
            grad_o = tensorplay.autograd.grad(
                target_logit, attn_out, retain_graph=True
            )[0]
            head_outputs = compute_head_outputs(mha, h_in, weights)
            n_heads = head_outputs.shape[1]
            attributions = []
            for h in range(n_heads):
                contribution = head_output_contribution(mha, head_outputs, h)
                # the full attention output is the sum of per-head
                # contributions plus a shared bias, so the per-head attribution
                # is exactly (d y / d o) . contribution_h.
                attr = float(
                    (grad_o * contribution)[0, target_position].sum()
                )
                attributions.append(attr)
            order = np.argsort(attributions)[::-1]
            kept = 0
            for h in order:
                if attributions[h] <= 0:
                    continue
                result.append((layer_idx, "attn", int(h)))
                kept += 1
                if kept >= topk:
                    break
    finally:
        for handle in handles:
            handle.remove()
    return result


class FeatureCircuit:
    """Discover a feature circuit for a target behavior.

    Args:
        model: A tensorplay transformer with ``transformer.h`` blocks exposing
            ``mlp`` (``Sequential(Linear, activation, Linear)``) and ``attn``
            (``nn.MultiheadAttention``).
        target_fn: Callable ``(model_output) -> scalar`` returning the target value
            (e.g., ``lambda out: out.logits[0, -1, token_id]``).
        layers: Optional layer indices to consider; ``None`` uses all.
    """

    def __init__(self, model: Any, target_fn: Callable[[Any], float], layers=None):
        self.model = model
        self.target_fn = target_fn
        self.layers = resolve_layer_modules(model, layers)

    # -- forward helpers -------------------------------------------------

    def _clean_score(self, input_ids: tensorplay.Tensor) -> float:
        with tensorplay.no_grad():
            out = self.model(input_ids=input_ids)
        return float(self.target_fn(out))

    def _run_with_ablations(
        self, input_ids: tensorplay.Tensor, nodes: Sequence[Tuple[int, str, int]]
    ) -> float:
        """Run one forward pass with the given feature nodes zero-ablated."""
        handles: List[Any] = []
        for layer, kind, index in nodes:
            block = self.layers[layer]
            if kind == "mlp":
                mod = _mlp_feature_module(block)

                def make_mlp_hook(idx: int) -> Callable:
                    def hook(module: Any, args: Any, out: Any) -> Any:
                        o = out.clone()
                        o[..., idx] = 0.0
                        return o

                    return hook

                handles.append(mod.register_forward_hook(make_mlp_hook(index)))
            elif kind == "attn":
                mha = block.attn
                head_dim = mha.out_proj.weight.shape[0] // mha.num_heads
                captured: Dict[str, Any] = {}

                def make_attn_hooks(
                    idx: int, store: Dict[str, Any], mha: Any, head_dim: int
                ) -> Tuple[Callable, Callable]:
                    def pre(module: Any, args: Any) -> None:
                        store["in"] = args[0]

                    def post(module: Any, args: Any, out: Any) -> Any:
                        attn_out, weights = out
                        head_outputs = compute_head_outputs(
                            mha, store["in"], weights
                        )
                        contribution = head_output_contribution(
                            mha, head_outputs, idx
                        )
                        return (attn_out - contribution, weights)

                    return pre, post

                pre, post = make_attn_hooks(index, captured, mha, head_dim)
                handles.append(mha.register_forward_pre_hook(pre))
                handles.append(mha.register_forward_hook(post))
            else:
                raise ValueError("Unknown node kind: %r" % kind)

        try:
            with tensorplay.no_grad():
                out = self.model(input_ids=input_ids)
            return float(self.target_fn(out))
        finally:
            for handle in handles:
                handle.remove()

    # -- circuit discovery -------------------------------------------------

    def direct_effects(
        self, input_ids: Any, nodes: Sequence[Tuple[int, str, int]]
    ) -> Tuple[float, Dict[Tuple[int, str, int], float]]:
        """Measure each node's direct effect on the target by zero-ablation.

        Returns:
            ``(clean_score, effects)`` where ``effects[node] = clean - ablated``.
        """
        ids = self._as_input_ids(input_ids)
        clean = self._clean_score(ids)
        effects: Dict[Tuple[int, str, int], float] = {}
        for node in nodes:
            ablated = self._run_with_ablations(ids, [node])
            effects[node] = clean - ablated
        return clean, effects

    def edge_weights(
        self,
        input_ids: Any,
        nodes: Sequence[Tuple[int, str, int]],
        clean_score: float,
        direct_effects: Dict[Tuple[int, str, int], float],
    ) -> List[Tuple[Tuple[int, str, int], Tuple[int, str, int], float]]:
        """Estimate pairwise interaction weights via joint-ablation mediation.

        For a pair ``(A, B)`` the weight is
        ``DE(A) + DE(B) - DE(A, B)`` where ``DE(A, B)`` is the joint effect of
        ablating both. This is the overlap between the two nodes' effects, a
        symmetric first-order approximation of edge attribution.
        """
        ids = self._as_input_ids(input_ids)
        edges: List[Tuple[Tuple[int, str, int], Tuple[int, str, int], float]] = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                joint_score = self._run_with_ablations(ids, [a, b])
                joint_effect = clean_score - joint_score
                overlap = direct_effects[a] + direct_effects[b] - joint_effect
                if overlap > 1e-8:
                    edges.append((a, b, overlap))
        return edges

    def build(
        self,
        input_ids: Any,
        nodes: Sequence[Tuple[int, str, int]],
        topk: int = 8,
        target: Optional[int] = None,
    ) -> FeatureCircuitData:
        """Discover the circuit for the given candidate nodes.

        Args:
            input_ids: Token ids.
            nodes: Candidate ``(layer, kind, index)`` node specs (see
                :func:`top_mlp_features` and :func:`top_attention_heads`).
            topk: Maximum number of nodes kept (by direct effect).
            target: Optional target token id (used only for bookkeeping).

        Returns:
            A :class:`FeatureCircuitData`.
        """
        ids = self._as_input_ids(input_ids)
        clean, effects = self.direct_effects(ids, nodes)
        ranked = sorted(effects.items(), key=lambda kv: kv[1], reverse=True)
        top_nodes = [node for node, effect in ranked if effect > 0][:topk]
        if not top_nodes:
            return FeatureCircuitData(clean_score=clean, target=target)

        node_list = [
            CircuitNode(
                layer=node[0],
                kind=node[1],
                index=node[2],
                direct_effect=effects[node],
            )
            for node in top_nodes
        ]
        node_id = {node: i for i, node in enumerate(top_nodes)}
        edge_list = [
            CircuitEdge(
                source=node_id[a], target=node_id[b], weight=weight
            )
            for a, b, weight in self.edge_weights(ids, top_nodes, clean, effects)
        ]
        edge_list.sort(key=lambda e: e.weight, reverse=True)
        return FeatureCircuitData(
            nodes=node_list,
            edges=edge_list,
            clean_score=clean,
            target=target,
        )

    def visualize(
        self,
        circuit: FeatureCircuitData,
        figsize: Optional[Sequence[float]] = None,
        max_edges: int = 40,
        use_pyplot: bool = False,
    ) -> Any:
        """Render the feature circuit as a layered graph.

        Nodes are colored by kind (MLP features red, attention heads blue) and
        sized by direct effect; edges are drawn with width proportional to
        their interaction weight.

        Returns:
            ``(fig, ax)`` matplotlib objects.
        """
        import matplotlib.pyplot as plt

        if not circuit.nodes:
            raise ValueError("The circuit is empty; build it before visualizing.")

        layers = sorted({n.layer for n in circuit.nodes})
        x_pos = {layer: i for i, layer in enumerate(layers)}
        per_layer: Dict[int, List[int]] = {layer: [] for layer in layers}
        for i, node in enumerate(circuit.nodes):
            per_layer[node.layer].append(i)
        y_pos: Dict[int, float] = {}
        max_de = max((n.direct_effect for n in circuit.nodes), default=1.0)
        for layer, indices in per_layer.items():
            indices.sort(key=lambda i: circuit.nodes[i].direct_effect, reverse=True)
            for rank, i in enumerate(indices):
                y_pos[i] = len(indices) - rank - 1

        fig, ax = plt.subplots(figsize=figsize if figsize is not None else (9, 6))
        for edge in circuit.edges[:max_edges]:
            sx, sy = x_pos[circuit.nodes[edge.source].layer], y_pos[edge.source]
            tx, ty = x_pos[circuit.nodes[edge.target].layer], y_pos[edge.target]
            ax.plot(
                [sx, tx],
                [sy, ty],
                color="tab:gray",
                alpha=min(1.0, 0.3 + edge.weight / max_de),
                linewidth=0.5 + 3.0 * edge.weight / max_de,
                zorder=1,
            )

        for i, node in enumerate(circuit.nodes):
            size = 80 + 300 * node.direct_effect / max_de
            color = "tab:red" if node.kind == "mlp" else "tab:blue"
            ax.scatter(
                x_pos[node.layer],
                y_pos[i],
                s=size,
                c=color,
                edgecolors="black",
                linewidths=0.5,
                zorder=2,
            )
            ax.annotate(
                str(node.index),
                (x_pos[node.layer], y_pos[i]),
                ha="center",
                va="center",
                fontsize=6,
                zorder=3,
            )

        ax.set_xticks(list(x_pos.values()))
        ax.set_xticklabels(["L%d" % l for l in layers])
        ax.set_yticks([])
        ax.set_xlabel("layer")
        ax.set_title("Feature circuit")
        fig.tight_layout()
        if use_pyplot:
            plt.show()
        return fig, ax

    @staticmethod
    def _as_input_ids(input_ids: Any) -> tensorplay.Tensor:
        if isinstance(input_ids, tensorplay.Tensor):
            ids = input_ids
        else:
            ids = tensorplay.as_tensor(np.asarray(input_ids), dtype=tensorplay.int64)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        return ids