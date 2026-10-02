import numpy as np

import tensorplay as tp
import tensorplay.nn.functional as functional


def test_multi_head_attention_returns_causal_weights_without_explicit_mask():
    query = tp.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]], dtype=tp.float32)
    identity = np.eye(2, dtype=np.float32)
    in_proj_weight = tp.tensor(np.concatenate([identity, identity, identity]))
    out_proj_weight = tp.tensor(identity)

    _, weights = functional.multi_head_attention_forward(
        query=query,
        key=query,
        value=query,
        embed_dim_to_check=2,
        num_heads=1,
        in_proj_weight=in_proj_weight,
        out_proj_weight=out_proj_weight,
        need_weights=True,
        is_causal=True,
    )

    weight_array = weights.numpy()
    np.testing.assert_allclose(np.triu(weight_array[0], k=1), 0.0)
    np.testing.assert_allclose(weight_array.sum(axis=-1), 1.0)
