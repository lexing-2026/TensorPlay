from .core import Layer, Config
from .layer import ConstantTensor
from .utils import get_layer_info, plot_dot_graph
import json


class Module(Layer):
    """模块基类，支持子模块管理"""

    def __init__(self):
        super().__setattr__('modules', {})  # 子模块字典
        super().__setattr__('parameters', {})  # 模块参数字典
        super().__setattr__('layers', {})  # 参数层字典
        super().__init__()

    def __setattr__(self, name: str, value):
        """属性设置，自动注册子模块和参数层"""
        # 基类最后注册，避免覆盖子类
        if isinstance(value, ConstantTensor):
            self.parameters[name] = value
            super().__setattr__(name, value)
        elif isinstance(value, Module):
            self.modules[name] = value
            super().__setattr__(name, value)
        elif isinstance(value, Layer):
            self.layers[name] = value
            super().__setattr__(name, value)
        else:
            # 管理参数不注册
            super().__setattr__(name, value)

    def __repr__(self):
        prefix = '' if self.__class__.__name__ == 'Module' else 'Module.'
        return f"{prefix}{self.__class__.__name__}"

    def params(self):
        """递归返回所有可训练参数"""
        param = []
        for p in self.parameters.values():
            if p.param() is not None:
                param.extend(p.param())
        for l in self.layers.values():
            if l.param() is not None:
                param.extend(l.param())
        for m in self.modules.values():
            param.extend(m.params())
        return param

    def named_params(self, prefix: str = ''):
        """递归返回带名称的参数"""
        named_param = []
        prefix = prefix + ('.' if prefix else '')
        for name, l in self.layers.items():
            if l.param() is not None:
                named_param.append((prefix + name, l, sum([len(i.data) for i in l.param()])))
            else:
                named_param.append((prefix + name, l, 0))
        for name, p in self.parameters.items():
            if p.param() is not None:
                named_param.append((prefix + name, p, sum([len(i.data) for i in p.param()])))
            else:
                named_param.append((prefix + name, p, 0))
        for name, m in self.modules.items():
            named_param.extend(m.named_params(prefix + name))
        return named_param

    def named_modules(self, prefix: str = ''):
        """返回带名称的子模块（包括自身）"""
        if prefix == '':
            prefix = self.__class__.__name__
        for name, module in self.modules.items():
            current_prefix = f"{prefix}.{name}" if prefix else name
            yield current_prefix, module
            yield from module.named_modules(current_prefix)

    def train(self, mode: bool = True):
        """设置训练模式"""
        Config.training = mode
        return self

    def eval(self):
        """设置评估模式，复用train(False)"""
        return self.train(False)

    def save(self, path: str):
        """将模块数据保存为JSON格式文件"""
        params = []
        layer = self.named_params()
        for param in layer:
            data = {"type": str(param[1]), "params": param[1].save()}
            params.append(data)
        datas = {"model": self.__class__.__name__, "layers_num": len(params), "parameters": params}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(datas, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """从JSON格式文件加载模块数据"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.loads(f.read())
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found in: {path}!")
        if data.get("model") != self.__class__.__name__:
            raise ValueError(f"Model mismatched: expect {self.__class__.__name__}, got {data.get('model')}!")
        text = data.get("parameters")
        if text is None:
            raise ValueError(f"Got no parameters for {self.__class__.__name__}!")
        layer = self.named_params()
        l = len(layer)
        if data.get("layers_num") != l or len(text) != l:
            raise ValueError(f"Layers sizes mismatched: expect {l}, got {data.get('layers_num')}!")
        for i in range(l):
            if text[i]["type"] != str(layer[i][1]):
                raise ValueError(f"Layers type mismatched: expect {layer[i][1]}, got {text[i]['type']}!")
            layer[i][1].load(text[i]["params"])

    def summary(self):
        """打印模型摘要，包括各层类型、输出形状和参数"""
        # 收集层信息的列表
        layer = self.named_params()
        layers_info = []
        total_params = 0
        trainable_params = 0
        for p in layer:
            total_params += p[2]
            if p[1].training:
                trainable_params += p[2]
            out_shape = get_layer_info(p[1])
            layers_info.append({
                "name": p[0],
                "type": str(p[1]).split(".")[-1],
                "output_shape": out_shape,
                "params": p[2]
            })

        max_name_len = max(len(f"{info['name']} ({info['type']})") for info in layers_info) + 4  # 增加4以留出边距
        max_shape_len = max(len(str(info["output_shape"])) for info in layers_info) + 4
        max_param_len = max(len(str(info["params"])) for info in layers_info) + 4

        # tf表头的奇怪配比（21、12、8）
        header_name_len = len("Layer (type)") + 21
        header_shape_len = len("Output Shape") + 12
        header_param_len = len("Param #") + 8

        max_name_len = max(max_name_len, header_name_len)
        max_shape_len = max(max_shape_len, header_shape_len)
        max_param_len = max(max_param_len, header_param_len)

        # 打印表头
        print(f"Model: \"{self.__class__.__name__}\"")
        print(f"┌{'─' * max_name_len}┬{'─' * max_shape_len}┬{'─' * max_param_len}┐")

        header = (f"│ {'Layer (type)':<{max_name_len - 2}} "
                  f"│ {'Output Shape':<{max_shape_len - 2}} "
                  f"│ {'Param #':>{max_param_len - 2}} │")
        print(header)

        # 打印分隔线
        print(f"├{'─' * max_name_len}┼{'─' * max_shape_len}┼{'─' * max_param_len}┤")

        # 打印各层信息
        for i, info in enumerate(layers_info):
            layer = f"{info['name']} ({info['type']})"
            shape = str(info['output_shape'])
            param = f"{info['params']:,}"  # 层参数添加千位分隔符

            row = (f"│ {layer:<{max_name_len - 2}} "
                   f"│ {shape:<{max_shape_len - 2}} "
                   f"│ {param:>{max_param_len - 2}} │")  # 参数右对齐更易读
            print(row)
            if i < len(layers_info) - 1:
                print(f"├{'─' * max_name_len}┼{'─' * max_shape_len}┼{'─' * max_param_len}┤")

        # 打印底部边框
        line = f"└{'─' * max_name_len}┴{'─' * max_shape_len}┴{'─' * max_param_len}┘"
        print(line)

        # 计算参数占用空间（float64）
        byte = 8

        def format_size(params):
            """ 将参数数量转换为最合适的存储单位"""
            bytes_size = params * byte
            units = ['B', 'KB', 'MB', 'GB']
            unit_index = 0
            size = bytes_size
            while size >= 1024 and unit_index < len(units) - 1:
                size /= 1024
                unit_index += 1
            return f"{size:.2f} {units[unit_index]}"

        # 打印总计信息，保持和表格对齐的缩进
        non_trainable = total_params - trainable_params
        print(f" Total params: {total_params:,} ({format_size(total_params)})")
        print(f" Trainable params: {trainable_params:,} ({format_size(trainable_params)})")
        print(f" Non-trainable params: {non_trainable:,} ({format_size(non_trainable)})")


class Sequential(Module):
    """顺序层容器，集中管理多个层"""

    def __init__(self, *args):
        super().__init__()
        self.container = []  # 记录模块顺序
        for i, module in enumerate(args):
            if not isinstance(module, (Layer, Module)):
                raise TypeError(f"Sequential need Layer or Module, not {type(module).__name__}")
            self.container.append(module)
            # 自动注册模块
            setattr(self, f"{self.__class__.__name__}_{i + 1}", module)

    def forward(self, x):
        """按顺序执行各层的前向传播"""
        for layer in self.container:
            x = layer(x)
        return x

    def __getitem__(self, index):
        """支持通过索引访问内部层"""
        return self.container[index]

    def __len__(self):
        """返回层数"""
        return len(self.container)