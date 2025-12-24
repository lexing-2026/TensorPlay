import os
import subprocess
import numpy as np
from typing import Tuple
from .core import Tensor, Operator

# =============================================================================
# Training Utils
# =============================================================================
def accuracy(output: Tensor, target: Tensor) -> float:
    if output.shape[1] == 2:
        result = np.argmax(output.data, axis=1, keepdims=True)
    elif output.shape[1] == 1:
        result = np.where(output.data > 0.5, 1, 0)
    result = (result == target.data).astype(output.dtype).mean()
    return result

# =============================================================================
# Graph Utils
# =============================================================================
def get_layer_info(layer):
    layer_type = str(layer).split(".")[-1]
    if layer_type == "Dense":
        return None, len(layer.w)
    elif layer_type == "Conv2D":
        return "Conv2D", len(layer.w)
    elif layer_type == "AveragePooling2D":
        return "AveragePooling2D"
    elif layer_type == "Flatten":
        return "Flatten"
    else:
        return (None,)

# =============================================================================
# Conv Utils
# =============================================================================
def recov_outsize(out_size: int, kernel_size: int, strides: int, padding: int) -> int:
    return strides * (out_size - 1) + kernel_size - padding


def conv_outsize(input_size: int, kernel_size: int, strides: int, padding: int) -> int:
    return (input_size + padding - kernel_size) // strides + 1


def _same_padding(input_height: int, input_width: int, kernel_size: Tuple[int, int],
                  stride: Tuple[int, int]) -> Tuple[int, int]:
    KH, KW = kernel_size
    SH, SW = stride

    # 步骤1：先计算目标输出尺寸（SAME模式核心：输出尺寸 = 输入尺寸/步长，向上取整）
    output_height = (input_height + SH - 1) // SH  # 等价于 ceil(input_height / SH)
    output_width = (input_width + SW - 1) // SW

    # 步骤2：根据输出尺寸反推总padding
    total_pad_h = max(0, (output_height - 1) * SH + KH - input_height)
    total_pad_w = max(0, (output_width - 1) * SW + KW - input_width)

    return total_pad_h, total_pad_w


def im2col_array(img: np.ndarray, kernel_size: Tuple[int, int], strides: Tuple[int, int],
                 padding: Tuple[int, int]) -> np.ndarray:
    """(B, OH, OW, C, KH, KW)"""
    B, H, W, C = img.shape
    KH, KW = kernel_size
    SH, SW = strides
    PH, PW = padding
    OH = conv_outsize(H, KH, SH, PH)
    OW = conv_outsize(W, KW, SW, PW)
    img = np.pad(img, pad_width=((0, 0), ((PH + 1) // 2, PH // 2 + SH - 1), ((PW + 1) // 2, PW // 2 + SW - 1), (0, 0)),
                 mode='constant')
    col = np.empty((B, C, KH, KW, OH, OW), dtype=img.dtype)
    for j in range(KH):
        j_end = j + SH * OH
        for i in range(KW):
            i_end = i + SW * OW
            # (B, OH, OW, C) -> (B, C, OH, OW) -> (B, C, j, i, OH, OW)
            col[:, :, j, i, :, :] = img[:, j:j_end:SH, i:i_end:SW, :].transpose(0, 3, 1, 2)
    # (B, C, KH, KW, OH, OW) -> (B, OH, OW, C, KH, KW)
    return col.transpose((0, 4, 5, 1, 2, 3))


def col2im_array(col: np.ndarray, img_shape: Tuple[int, int, int, int], kernel_size: Tuple[int, int],
                 strides: Tuple[int, int], padding: Tuple[int, int]) -> np.ndarray:
    """(B, H, W, C)"""
    B, OH, OW, OC = img_shape
    KH, KW = kernel_size
    SH, SW = strides
    PH, PW = padding
    H = recov_outsize(OH, KH, SH, PH)
    W = recov_outsize(OW, KW, SW, PW)
    img = np.zeros((B, H + PH + SH - 1, W + PW + SW - 1, col.shape[3]), dtype=col.dtype)
    for j in range(KH):
        j_end = j + SH * H
        for i in range(KW):
            i_end = i + SW * W
            img[:, j:j_end:SH, i:i_end:SW, :] += col[:, :, :, :, j, i]
    return img[:, (PH + 1) // 2 : PH // 2 + SH + H, (PW + 1) // 2 : PW // 2 + SW + W, :]

# =============================================================================
# Graph Utils
# =============================================================================
def _dot_ten(v: Tensor, verbose=False):
    """绘制张量节点"""
    name = '' if v.name is None else v.name
    if verbose:
        if v.name is not None:
            name += ': '
        if v.ndim > 1:
            name += f"{v.shape} "
        name += f"{v.dtype}"
    text = f'{id(v)} [label="{name}", color=orange, style=filled]\n'
    if v.op is not None:
        text += f'{id(v.op)} -> {id(v)}\n'
    return text


def _dot_op(op: Operator):
    """绘制算符节点"""
    text = f'{id(op)} [label="{op.__class__.__name__}", color=lightblue, style=filled, shape=box]\n'
    dot_edge = '{} -> {}\n'
    inp = op.inp if isinstance(op.inp, list) else [op.inp]
    for x in inp:
        text += dot_edge.format(id(x), id(op))
    return text


def _trace_dot_graph(out: Tensor, verbose=False):
    """获取计算图"""
    text = ''
    op_set = set()
    queue = []
    # 从输出张量开始，收集所有相关运算符
    if out.op is not None:
        queue.append(out.op)
        op_set.add(out.op)
    text += _dot_ten(out, verbose)
    # 广度优先搜索收集所有相关运算符
    while queue:
        current_op = queue.pop(0)  # 取出队首元素
        text += _dot_op(current_op)
        if current_op.inp is None:
            continue
        inputs = current_op.inp if isinstance(current_op.inp, list) else [current_op.inp]
        for inp_tensor in inputs:
            if inp_tensor.op is not None and inp_tensor.op not in op_set:
                queue.append(inp_tensor.op)
                op_set.add(inp_tensor.op)
            text += _dot_ten(inp_tensor, verbose)

    return 'digraph g {\n' + text + '}'


def _instill_dot_graph(book: list, verbose=False):
    """提取计算图"""
    text = ''
    for op in book:
        text += _dot_op(op)
        current = op.inp if isinstance(op.inp, list) else [op.inp]
        for inp_tensor in current:
            text += _dot_ten(inp_tensor, verbose)
    return 'digraph g {\n' + text + '}'


def plot_dot_graph(source, verbose=False, path='graph.png'):
    """绘制计算图"""
    if isinstance(source, Tensor):
        graph = _trace_dot_graph(source, verbose)
    elif isinstance(source, list):
        graph = _instill_dot_graph(source, verbose)
    else:
        raise TypeError("Graph source must be a Tensor or a list of Operators")
    tmp_dir = os.path.join(os.path.expanduser('~'), '.TensorPlay')
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')
    # 保存计算图
    with open(graph_path, 'w') as f:
        f.write(graph)
    # 调用dot命令生成图片
    extension = os.path.splitext(path)[1][1:]
    cmd = f'dot {graph_path} -T{extension} -o {path}'
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Graphviz Error: {e}")
