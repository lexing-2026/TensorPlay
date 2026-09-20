import os

import tensorplay as tp
from tensorplay.utils.viz import make_dot

def test_viz():
    x = tp.randn(5, 5, requires_grad=True)
    y = tp.randn(5, 5, requires_grad=True)
    w = tp.randn(5, 5, requires_grad=True)

    h = (x + y).relu()
    z = h.matmul(w)
    loss = z.sum()

    dot = make_dot(loss, params={"x": x, "y": y, "w": w})
    output_file = "viz_test"
    try:
        dot.render(output_file, format="png")
        assert os.path.exists(f"{output_file}.png"), "render produced no image"
    finally:
        if os.path.exists(f"{output_file}.png"):
            os.remove(f"{output_file}.png")

if __name__ == "__main__":
    test_viz()
