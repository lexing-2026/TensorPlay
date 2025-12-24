import tensorplay as tp
from tensorplay.utils.viz import make_dot
import os

def test_viz():
    print("Creating graph...")
    x = tp.randn(5, 5, requires_grad=True)
    y = tp.randn(5, 5, requires_grad=True)
    w = tp.randn(5, 5, requires_grad=True)
    
    h = (x + y).relu()
    z = h.matmul(w)
    loss = z.sum()
    
    print("Generating visualization...")
    try:
        dot = make_dot(loss, params={"x": x, "y": y, "w": w})
        output_file = "viz_test"
        dot.render(output_file, format="png")
        print("Success!")
        os.remove(f"{output_file}.png")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_viz()
