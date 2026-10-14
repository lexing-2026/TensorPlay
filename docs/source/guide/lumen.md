# Interpretability with lumen

`tensorplay.lumen` is TensorPlay's native explainability suite. It covers the classic
attribution algorithms (integrated gradients, deep lift, LRP, SHAP, TCAV), layer and
neuron attribution, visualization and statistics, and a set of research-grade LLM
tools (sparse autoencoders, cross-layer transcoders, attribution graphs, feature
circuits, logit lens, causal tracing). Everything runs on tensorplay primitives — no
third-party runtime is required.

```python
import tensorplay.lumen as lumen
```

## What you can reproduce

The suite implements methods from the interpretability literature, so you can rerun
their core experiments on your own models:

| Paper / technique | lumen API |
|---|---|
| Integrated Gradients — Sundararajan et al., ICML 2017 | `IntegratedGradients` |
| DeepLIFT — Shrikumar et al., 2017 | `DeepLift`, `DeepLiftShap` |
| Layer-wise Relevance Propagation — Bach et al., PLOS ONE 2015 | `LRP` |
| SHAP — Lundberg & Lee, NeurIPS 2017 | `KernelShap` |
| TCAV — Kim et al., ICML 2018 | `tensorplay.lumen.concept.TCAV` |
| Causal tracing / knowledge localization — Vig et al., 2020; Meng et al., ICLR 2022 | `lumen.llm.ActivationPatching.causal_trace` |
| Logit lens — nostalgebraist, 2020 | `lumen.llm.LogitLens` |
| Sparse autoencoder features — Anthropic, 2023–2024 | `lumen.llm.SparseAutoencoder` |
| Cross-layer transcoders — Anthropic, 2025 | `lumen.llm.CrossLayerTranscoder` |
| Attribution graphs — Anthropic, 2023 | `lumen.llm.AttributionGraph` |
| Feature circuits — Anthropic, 2024 | `lumen.llm.FeatureCircuit` |

## Feature attribution on a small model

Start with a tiny network and ask "which input features drive the output?"

```python
import tensorplay as tp
from tensorplay import nn
from tensorplay.lumen import IntegratedGradients, DeepLift, LRP

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = MLP()
x = tp.tensor([[1.0, 2.0, -1.0, 0.5]])

ig = IntegratedGradients(model).attribute(x, target=1, n_steps=50)
dl = DeepLift(model).attribute(x, target=1, baselines=0.0)
lrp = LRP(model).attribute(x, target=1)

print("IG   :", ig)
print("DeepLift:", dl)
print("LRP     :", lrp)
```

- `IntegratedGradients` satisfies completeness: the attributions sum to the model's
  output difference between the input and a baseline.
- `DeepLift` and `LRP` are backpropagation-based and give fine-grained per-feature
  relevance. All three return tensors with the same shape as the input.

### Layer and neuron attribution

When you want to know what a *layer* or a *single neuron* is responding to, use the
layer and neuron series:

```python
from tensorplay.lumen import LayerActivation, LayerIntegratedGradients
from tensorplay.lumen import NeuronConductance, NeuronIntegratedGradients

layer = model.fc2
layer_act = LayerActivation(model, layer).attribute(x)          # (1, 3)
layer_ig = LayerIntegratedGradients(model, layer).attribute(x, target=1, baselines=0.0)

neuron_ig = NeuronIntegratedGradients(model, layer).attribute(
    x, neuron_selector=(0,), baselines=0.0
)  # gradient of neuron (0) w.r.t. the input
```

### Concept-based attribution (TCAV)

TCAV answers "does the model use concept *X* (e.g. stripes) when making a decision?"
It trains a linear classifier between concept activation vectors and measures the
sensitivity of the prediction to that concept direction:

```python
from tensorplay.lumen.concept import TCAV, Concept
from tensorplay.lumen.concept._utils.data_iterator import dataset_to_dataloader
from tensorplay.utils.data import TensorDataset

concept_data = dataset_to_dataloader(TensorDataset(tp.randn(32, 4)))
random_data = dataset_to_dataloader(TensorDataset(tp.randn(32, 4)))
tcav = TCAV(model, layers=["fc1"])
tcav.interpret(
    inputs=x,
    experimental_sets=[
        [Concept(0, "concept", concept_data)],
        [Concept(1, "random", random_data)],
    ],
)
```

TCAV trains its default classifier with `sklearn`; install it to run this example.
Without `sklearn`, `TCAV` raises a friendly `ValueError`.

## LLM interpretability

The LLM tools operate on any transformer built with tensorplay primitives. The
examples below use a tiny native GPT so they run offline, end to end.

### A tiny native transformer

```python
from types import SimpleNamespace

def causal_mask(seq):
    return tp.zeros(seq, seq, dtype=tp.float32).masked_fill(
        tp.triu(tp.ones(seq, seq, dtype=tp.bool), diagonal=1), float("-inf")
    )

class Block(nn.Module):
    def __init__(self, d=16, heads=4, ff=32):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        o, w = self.attn(h, h, h, attn_mask=mask, need_weights=True,
                        average_attn_weights=False)
        x = x + o
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x, w

class TinyGPT(nn.Module):
    def __init__(self, vocab=11, d=16, heads=4, layers=2, ff=32):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(vocab, d)
        self.transformer.wpe = nn.Embedding(16, d)
        self.transformer.h = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.transformer.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids, output_attentions=False):
        seq = input_ids.shape[1]
        x = self.transformer.wte(input_ids) + self.transformer.wpe(
            tp.arange(0, seq, dtype=tp.int64).unsqueeze(0)
        )
        mask = causal_mask(seq)
        attentions = []
        for blk in self.transformer.h:
            x, w = blk(x, mask)
            attentions.append(w)
        x = self.transformer.ln_f(x)
        return SimpleNamespace(
            logits=self.lm_head(x),
            attentions=tuple(attentions) if output_attentions else None,
        )

VOCAB = ["<pad>", "<unk>", "The", "capital", "of", "France", "England",
         "is", "Paris", "London", "<eos>"]

class TinyTokenizer:
    def __init__(self):
        self.id2tok = {i: t for i, t in enumerate(VOCAB)}
        self.tok2id = {t: i for i, t in enumerate(VOCAB)}

    def encode(self, text, return_tensors="pt", add_special_tokens=False):
        ids = [self.tok2id.get(w, 1) for w in text.split()]
        return tp.tensor([ids], dtype=tp.int64) if return_tensors == "pt" else ids

    def convert_ids_to_tokens(self, ids):
        return [self.id2tok[int(i)] for i in ids]

    def decode(self, ids):
        return " ".join(self.convert_ids_to_tokens(ids))

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        words = text.split()
        ids = [self.tok2id.get(w, 1) for w in words]
        res = {"input_ids": ids}
        if return_offsets_mapping:
            offsets, pos = [], 0
            for w in words:
                start = text.index(w, pos)
                offsets.append((start, start + len(w)))
                pos = start + len(w)
            res["offset_mapping"] = offsets
        return res

model = TinyGPT()
tokenizer = TinyTokenizer()
prompt = "The capital of France is"
ids = tokenizer.encode(prompt)
```

### Attention heatmaps

Extract per-head attention weights and render them with the visualization helper.
Highlight important tokens (e.g. a subject) to make the attended-to pattern legible:

```python
from tensorplay.lumen.llm import (
    extract_attention,
    visualize_attention_multi_layer,
    token_attention,
    visualize_token_attention,
)

result = extract_attention(model, tokenizer, prompt, layers=[0, 1])

# paper-style grid: layers x heads, what "France" attends to outlined
fig, axes = visualize_attention_multi_layer(result, query_token="France")

# single-token view: France's attention at every (layer, head)
ta = token_attention(result, query_token="France")
fig, ax = visualize_token_attention(ta)
```

### Causal tracing: where is a fact stored?

The causal-tracing protocol from knowledge-editing research corrupts a prompt,
then restores the clean hidden state at every `(layer, position)` and measures how
much of the clean prediction is recovered:

```python
from tensorplay.lumen.llm import ActivationPatching

clean = tokenizer.encode("The capital of France is")[0]
corrupted = tokenizer.encode("The capital of England is")[0]
trace = ActivationPatching(model).causal_trace(clean, corrupted, target_position=-1)

for layer, row in zip(trace["layers"], trace["scores"]):
    print("L%d:" % layer, ["%.2f" % v for v in row])
```

High recovery at the subject position ("France") and the final layer localizes
where the model stores the fact. This is the diagnostic used before applying
knowledge-editing weight updates.

### Logit lens

The logit lens decodes every layer's hidden state through the LM head, showing
which token the model "would have predicted" at each depth:

```python
from tensorplay.lumen.llm import LogitLens

result = LogitLens(model, tokenizer).lens(ids, position=-1, topk=3)
for layer, preds in result["predictions"].items():
    print("L%d:" % layer, ["%s=%.2f" % (t, p) for t, p in preds])
```

### Sparse autoencoder features

Sparse autoencoders learn a dictionary of sparse, interpretable features from
hidden states — the dictionary-learning step used in feature research. Train one on
activations collected from a layer:

```python
from tensorplay.utils.data import DataLoader, TensorDataset
from tensorplay.lumen.llm import (
    SparseAutoencoder,
    collect_hidden_states,
    train_sae,
    feature_frequency,
)

corpus = ["The capital of France is", "The capital of England is",
          "The capital of France is Paris", "The capital of England is London"]
seqs = [tokenizer.encode(s)[0] for s in corpus]
hidden = collect_hidden_states(model, layer=0, input_ids_list=seqs, batch_size=2)

sae = SparseAutoencoder(input_dim=16, hidden_dim=32)
loader = DataLoader(TensorDataset(hidden), batch_size=8)
train_sae(sae, loader, n_epochs=20, lr=1e-2, l1_coef=1e-3)

features = sae.encode(hidden)
print("active fraction per feature:", feature_frequency(features))
```

Each learned feature is a direction in activation space; `top_activating_indices`
lists the corpus examples that most activate each feature, which is how you would
label a feature (e.g. "the token right before a country name").

### Cross-layer transcoders

A cross-layer transcoder predicts the *update* a transformer component adds to the
residual stream from the incoming residual stream, decomposing the component's
computation into sparse features:

```python
from tensorplay.lumen.llm import (
    CrossLayerTranscoder,
    collect_component_updates,
    train_clt,
)

residuals, deltas = collect_component_updates(model, layer=0, input_ids_list=seqs, batch_size=2)
clt = CrossLayerTranscoder(input_dim=16, hidden_dim=32)
loader = DataLoader(TensorDataset(residuals, deltas), batch_size=8)
train_clt(clt, loader, n_epochs=20, lr=1e-2, l1_coef=1e-3)
```

### Attribution graphs

Attribution graphs show which MLP features at each layer positively contribute to
an output and how they connect across layers. The nodes are features weighted by
``gradient * activation``; the edges are positive linearized feature-to-feature
paths:

```python
from tensorplay.lumen.llm import AttributionGraph

graph = AttributionGraph(model).build(ids, target_position=-1, topk=5)
print("nodes:", [(n.layer, n.kind, n.index) for n in graph.nodes])
print("edges:", [(e.source, e.target, round(e.weight, 4)) for e in graph.edges[:5]])
```

### Feature circuits

Feature circuits measure the *direct effect* of every candidate feature node (MLP
features and attention heads) by zero-ablating it, then estimate pairwise
interaction edges via joint ablation:

```python
from tensorplay.lumen.llm import (
    FeatureCircuit,
    top_mlp_features,
    top_attention_heads,
)

clean_logits = model(ids).logits
target_id = int(clean_logits[0, -1].argmax())
target = lambda out: float(out.logits[0, -1, target_id])

nodes = top_mlp_features(model, ids, target_position=-1, topk=4)
nodes += top_attention_heads(model, ids, target_position=-1, topk=2)
circuit = FeatureCircuit(model, target).build(ids, nodes=nodes, topk=6)
print("circuit nodes:", [n.name for n in circuit.nodes])
```

## Visualization and statistics

Attribution results can be rendered with the visualization helpers (images, text,
timeseries, attention heatmaps) and aggregated with `Summarizer` / `ClassSummarizer`:

```python
from tensorplay.lumen import Summarizer, Mean, StdDev, Count

stats = Summarizer([Mean(), StdDev(), Count()])
for _ in range(10):
    stats.update(tp.randn(4))
print(stats.summary)
```

## Optional dependencies

- `sklearn` is used only by `Lime`, `KernelShap`, and the `TCAV` default classifier.
  Without it, those classes raise a friendly `ValueError` when you run them.
- `matplotlib` is used for visualization; the helpers accept either numpy arrays or
  tensorplay tensors directly.

## Where to go next

- `tensorplay.lumen` reference — the full public API (80+ symbols) is re-exported
  from `tensorplay/lumen/__init__.py`.
- [Models](models.md) — build the `nn.Module` models the attributions run on.
- [Autograd](autograd.md) — how the gradient computations behind the attributions work.