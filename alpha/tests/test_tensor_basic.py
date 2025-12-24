import numpy as np
from alpha.tests import Tensor, sphere, goldstein


def test_tensor_creation():
    # 测试基本的Tensor创建
    data1 = np.array([1.0, 2.0, 3.0])
    t1 = Tensor(data1)
    data2 = [1.0, 2.0, 3.0]
    t2 = Tensor(data2)
    data3 = 1.0
    t3 = Tensor(data3)

    # 检查数据是否正确存储
    assert np.array_equal(t1.data, data1)
    assert np.array_equal(t2.data, data2)
    assert np.array_equal(t3.data, data3)
    assert t1.grad is None
    assert t1.op is None
    assert t2.grad is None
    assert t2.op is None
    assert t3.grad is None
    assert t3.op is None
    


def test_tensor_arithmetic():
    # 测试基本算术操作
    a = Tensor(np.array([1.0, 2.0, 3.0]))
    b = Tensor(np.array([4.0, 5.0, 6.0]))
    z = Tensor(np.array([[7.0], [8.0], [9.0]]))
    
    # 测试加法
    c = a + b
    assert np.array_equal(c.data, np.array([5.0, 7.0, 9.0], dtype=a.dtype))

    # 测试减法
    d = a - b
    assert np.array_equal(d.data, np.array([-3.0, -3.0, -3.0], dtype=a.dtype))
    
    # 测试乘法
    f = a * b
    assert np.array_equal(f.data, np.array([4.0, 10.0, 18.0], dtype=a.dtype))

    # 测试矩阵乘法
    g = a @ z
    assert np.array_equal(g.data, np.array([50.0], dtype=a.dtype))

    # 测试除法
    h = a / b
    assert np.array_equal(h.data, np.array([0.25, 0.4, 0.5], dtype=a.dtype))
    
    # 测试标量操作
    e = a + 2.0
    assert np.array_equal(e.data, np.array([3.0, 4.0, 5.0], dtype=a.dtype))

    i = a ** 2.0
    assert np.array_equal(i.data, np.array([1.0, 4.0, 9.0], dtype=a.dtype))
    


def test_tensor_functions():
    # 测试内置函数
    x = Tensor(np.array([-1.0, 0.0, 1.0]))
    y = Tensor(np.array([1.0, 0.0, -1.0]))

    z = goldstein(x, y)
    assert np.array_equal(z.data, np.array([87100.0, 600.0, 7100.0]))

    z = x.relu()
    assert np.array_equal(z.data, np.array([0.0, 0.0, 1.0]))
    


def test_simple_backward():
    # 测试简单的反向传播
    x = Tensor(np.array([2.0]))
    y = sphere(x, x)
    
    # 手动设置梯度并反向传播
    y.backward(higher_grad=True)
    gx = x.grad
    x.zero_grad()
    gx.backward()
    assert np.array_equal(x.grad.data, np.array([4.0]))
    

if __name__ == "__main__":
    test_tensor_creation()
    test_tensor_arithmetic()
    test_tensor_functions()
    test_simple_backward()
    print("All tests passed!")