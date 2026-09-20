"""Gradient-descent smoke tests on classic optimization benchmarks.

Each benchmark is minimized with the packaged AdamW optimizer over a
two-parameter start; the assertions pin convergence near the known
optimum, so a broken backward graph or optimizer step fails loudly.
"""
import numpy as np

import tensorplay as tp

# ----------------------
# 1. 经典优化测试函数（tensorplay实现，支持自动求导）
# ----------------------
# 1. 二次函数 (Quadratic Function)
# f(x, y) = x^2 + 2y^2
# Global minimum at (0, 0)
@tp.compile
def quadratic_function(x):
    return x[0]**2 + 2 * x[1]**2

# 2. Rosenbrock Function
# f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2)
   # Global minimum at (1, 1, ..., 1)
def rosenbrock_function(x):
    sum = 0
    for i in range(len(x) - 1):
        sum = sum + 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return sum

# 3. Ackley Function
# Global minimum at (0, 0, ..., 0)
def ackley_function(x):
    d = len(x)
    sum_sq = 0
    sum_cos = 0
    for i in range(d):
        sum_sq = sum_sq + x[i]**2
        sum_cos = sum_cos + tp.cos(2 * np.pi * x[i])

    a = 20
    b = 0.2
    c = 2 * np.pi

    term1 = -a * tp.exp(-b * tp.sqrt(sum_sq / d))
    term2 = -tp.exp(sum_cos / d)

    return term1 + term2 + a + np.e

# 4. Rastrigin Function
# Global minimum at (0, 0, ..., 0)
def rastrigin_function(x):
    d = len(x)
    sum = 0
    for i in range(d):
        sum = sum + x[i]**2 - 10 * tp.cos(2 * np.pi * x[i])
    return 10 * d + sum

# ----------------------
# 2. 手动实现梯度下降优化器
# ----------------------
def gradient_descent(
    func,          # 优化目标函数（输入tensor列表，输出标量tensor）
    init_params,   # 初始参数（list of tensor，每个tensor需requires_grad=True）
    lr=0.001,      # 学习率
    max_iter=10000,# 最大迭代次数
    tol=1e-6,      # 收敛阈值（函数值变化<tol则停止）
    clip_grad=1.0, # 梯度裁剪阈值（避免梯度爆炸）
    momentum=0.0   # 动量
):
    """
    使用SGD优化器
    返回：参数历史、函数值历史
    """
    params = init_params.copy()

    # Create Optimizer
    optimizer = tp.optim.AdamW(params, lr=lr)

    param_history = [np.array([p.item() for p in params])]  # 记录参数轨迹
    loss_history = []                                       # 记录函数值轨迹

    prev_loss = float('inf')
    for i in range(max_iter):
        # 1. Zero grad
        optimizer.zero_grad()

        # 2. Forward
        loss = func(params)
        loss_val = loss.item()
        loss_history.append(loss_val)

        # 3. Convergence Check
        if abs(loss_val - prev_loss) < tol or loss_val < 1e-5:
            break
        prev_loss = loss_val

        # 4. Backward
        loss.backward()

        # 5. Gradient Clipping (manual, before step)
        if clip_grad > 0:
            for p in params:
                if p.grad is not None:
                    p.grad.data = tp.clamp(p.grad.data, -clip_grad, clip_grad)

        # 6. Step
        optimizer.step()

        # 7. Record history
        param_history.append(np.array([p.item() for p in params]))

    return np.array(param_history), np.array(loss_history)

# ----------------------
# 3. 测试函数
# ----------------------
def test_quadratic():
    """测试二次函数（单峰凸函数，梯度下降易收敛）"""
    # 初始参数（远离最优解(0,0)）
    init_x1 = tp.Tensor([3.0], requires_grad=True)
    init_x2 = tp.Tensor([-2.0], requires_grad=True)
    init_params = [init_x1, init_x2]

    param_history, loss_history = gradient_descent(
        func=quadratic_function,
        init_params=init_params,
        lr=0.1,
        max_iter=100,
        tol=1e-8
    )

    final_x1, final_x2 = param_history[-1]
    assert abs(final_x1) < 0.05 and abs(final_x2) < 0.05, (
        f"final params ({final_x1}, {final_x2}) not near (0, 0)")
    assert loss_history[-1] < 1e-3, f"final loss {loss_history[-1]}"

def test_rosenbrock():
    """测试Rosenbrock函数（非凸，梯度下降需小学习率）"""
    init_x1 = tp.Tensor([2.0], requires_grad=True)
    init_x2 = tp.Tensor([3.0], requires_grad=True)
    init_params = [init_x1, init_x2]

    param_history, loss_history = gradient_descent(
        func=rosenbrock_function,
        init_params=init_params,
        lr=0.0003,
        max_iter=50000,
        tol=1e-7,
        clip_grad=5.0,
        momentum=0.9
    )

    final_x1, final_x2 = param_history[-1]
    assert abs(final_x1 - 1.0) < 0.1 and abs(final_x2 - 1.0) < 0.1, (
        f"final params ({final_x1}, {final_x2}) not near (1, 1)")
    assert loss_history[-1] < 1e-2, f"final loss {loss_history[-1]}"

def test_ackley():
    """测试Ackley函数（多峰，梯度下降可能陷入局部最优）"""
    init_x1 = tp.Tensor([0.5], requires_grad=True)
    init_x2 = tp.Tensor([-0.5], requires_grad=True)
    init_params = [init_x1, init_x2]

    param_history, loss_history = gradient_descent(
        func=ackley_function,
        init_params=init_params,
        lr=0.01,
        max_iter=10000,
        tol=1e-7,
        clip_grad=2.0,
        momentum=0.9
    )

    final_x1, final_x2 = param_history[-1]
    assert abs(final_x1) < 0.01 and abs(final_x2) < 0.01, (
        f"final params ({final_x1}, {final_x2}) not near (0, 0)")
    assert loss_history[-1] < 1e-2, f"final loss {loss_history[-1]}"

def test_rastrigin():
    """测试Rastrigin函数（多峰+噪声，梯度下降挑战最大）"""
    init_x1 = tp.Tensor([0.4], requires_grad=True)
    init_x2 = tp.Tensor([-0.4], requires_grad=True)
    init_params = [init_x1, init_x2]

    param_history, loss_history = gradient_descent(
        func=rastrigin_function,
        init_params=init_params,
        lr=0.001,
        max_iter=20000,
        tol=1e-7,
        clip_grad=3.0,
        momentum=0.9
    )

    final_x1, final_x2 = param_history[-1]
    assert abs(final_x1) < 0.01 and abs(final_x2) < 0.01, (
        f"final params ({final_x1}, {final_x2}) not near (0, 0)")
    assert loss_history[-1] < 1e-2, f"final loss {loss_history[-1]}"


if __name__ == "__main__":
    test_quadratic()
    test_rosenbrock()
    test_ackley()
    test_rastrigin()
