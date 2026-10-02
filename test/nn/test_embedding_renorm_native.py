import numpy as np

import tensorplay as tp
import tensorplay.nn.functional as functional


def test_embedding_max_norm_updates_referenced_rows_in_place():
    weight = tp.tensor([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])
    indices = tp.tensor([[0, 0], [2, 0]], dtype=tp.int64)

    result = functional.embedding(indices, weight, max_norm=2.0)

    np.testing.assert_allclose(weight.numpy(), [[1.2, 1.6], [1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(result.numpy(), [
        [[1.2, 1.6], [1.2, 1.6]],
        [[0.0, 2.0], [1.2, 1.6]],
    ])


def test_embedding_max_norm_supports_infinity_norm():
    weight = tp.tensor([[3.0, -4.0], [2.0, 1.0]])
    indices = tp.tensor([0], dtype=tp.int64)

    functional.embedding(indices, weight, max_norm=2.0, norm_type=float("inf"))

    np.testing.assert_allclose(weight.numpy(), [[1.5, -2.0], [2.0, 1.0]])
