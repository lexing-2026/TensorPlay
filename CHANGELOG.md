# 1. RNG 与 Torch 逐位对齐（2026-08-21）

seed 随机性全链路对齐 `third_party/pytorch`（基准 `893b6406`）：

- **引擎**：`std::mt19937` → 移植 `at::mt19937`（`p10/include/MT19937RNGEngine.h`），同种子同 uint32 流。
- **分布层**：新增 `p10/include/DistributionsHelper.h`，照搬 torch 变换公式与消耗模式
  （uniform mantissa 变换、Box-Muller + generator 内缓存第二样本、有偏 modulo 整数、
  Fisher-Yates randperm、Hoermann/Knuth poisson、double 精度 exponential/cauchy/geometric/log_normal）。
- **normal_fill**：移植 torch AVX2 向量化路径（avx_mathfun 的 log256_ps/sincos256_ps，
  `p10/include/avx_mathfun.h`），运行时按 `__builtin_cpu_supports("avx2")` 分发，大张量 randn 逐位一致。
- **Generator**：默认种子 67280421310721、seed=0 合法、默认 generator 非确定初始化；
  get/set_state 采用 torch 同款 5056 字节 POD 布局（字节兼容，含 normal 缓存）。
- **CUDA**：移除 host cuRAND，改为 torch 同款 curand 设备端 Philox4_32_10 + host (seed, offset)
  预留（`CUDAGenerator.h/.cpp`），state 为 16 字节 seed+offset。
- **Python**：新增 `get_rng_state/set_rng_state/fork_rng`（`tensorplay/random.py`）、
  `Generator.get_state/set_state`、无参 `seed()`。
- **dtype 补齐**：分布算子覆盖 Half/BFloat16（float 精度采样后转型）、random_/geometric_ 全整型谱 + Bool、
  randperm Int32；与 torch dispatch 范围一致。
- **验证**：`test/test_random.py` 新增 `TestTorchParity`（同种子逐值断言，含半精度与 state 互通）；
  编译验证待构建窗口执行。

## 8. CUDA 算子优化批次（2026-08-21）

参照 `third_party/pytorch`（基准 `893b6406`）补齐 CUDA 侧三个性能/覆盖缺口：

- **LayerNorm CUDA 前向/反向**：`p10/src/backend/cuda/NormalizationKernels.cu` 新增自定义
  kernel（不再依赖 cuDNN）。算法照 torch `layer_norm_kernel.cu`：每行一个 block、
  Welford 矩经 `__shfl_down_sync` warp 归约 + shared memory 跨 warp 合并、
  前向单次发射融合统计与归一化、`N % 4 == 0` 且指针对齐时走 4 宽向量化加载；
  半精度以 fp32 累加。反向三 kernel：行矩重算、grad_input（两 sum 块归约）、
  grad_weight/grad_bias 列并行确定性归约（无 atomic）。支持
  Float32/Float64/Float16/BFloat16；`native_functions.yaml` 注册
  `layer_norm`/`layer_norm_backward` 的 CUDA 分支，autograd 经
  `derivatives.yaml` 自动生效。
- **Loss 半精度**：`LossKernels.cu` 的 `nll_loss`/`nll_loss_backward`/`mse_loss_backward`
  从仅 Float32 扩展到 Float64/Half/BFloat16（fp32 数学；归约路径 fp32 atomic
  累加后单线程转回——BF16 atomicAdd 需 sm_90+，Ampere 不可用）。
- **二元算子向量化**：`ArithmeticKernels.cu` 的 add/sub/mul/div 增加同形状连续
  快路径：numel ≥ 4096、`n % 4 == 0` 且指针按 `sizeof(T)*4` 对齐时走 4 宽
  packed load/store（128 线程、4 元素/线程、grid 上限 4×SM，对齐 ATen
  elementwise 配置）；广播/非连续路径维持原 TensorDesc kernel。
- **半精度一元浮点向量化**：`PointwiseKernels.cu` 的 Half/BFloat16 一元浮点
  算子（exp/log/sqrt 等经 `unary_reduced_float` 路径）接入与 float 相同的
  4 宽向量化发射。
- CUDA caching allocator 与 Stream/Event 基础设施已在 `CUDAAllocator.cpp` 落地
  （大小池 best-fit、segment 切分合并、跨 stream 事件保护、OOM emptyCache 重试、
  memory stats API），本批次未再改动。
- 验证：编译与测试待构建窗口执行（本机 3090 当时被其他进程占用，性能对照
  延后到空闲窗口）。

## 9. tensorplay.graph 门面:FX 对齐 + 特征提取 + 图可视化(2026-08-22)

公共图 API 从占位 shim 迁移为真实实现;实现位置不变(`tensorplay/compiler/graph.py`),
新增门面 `tensorplay/graph.py` 作公共入口,删除无人引用的 `tensorplay/fx.py`。

- **Graph 原语**(`compiler/graph.py`):
  - 节点命名从 `_{counter}` 升级为 fx 风格语义名(`add`/`conv1`/`view`),按目标推导 +
    `_0/_1` 唯一化;显式命名优先。名字在 erase 后保留不复用,保证 recompile 生成代码的
    变量名安全;
  - 新增 `Node.erase_node` / `Node.replace_all_uses_with`(含拓扑守卫);
    `dead_code_elimination` 改为返回移除数量并同步清空被删节点的 graph 指针;
  - **修复双 output bug**:`Graph.output()` 现为单例替换语义(先擦旧 output 再建新),
    此前 `_interpret` 取首个 output 而 `recompile` 取末个,可能返回不同值;
  - `lint()` 增加跨图引用与多 output 校验。
- **Tracer**:
  - `is_leaf_module(module, qualified_name)` 钩子(默认 False 保持 Dynamo 式内联,
    `tp.compile` 行为不变);返回 True 的子模块产出 `call_module` 节点;
  - `Tracer.node_to_qualname`:每个 call_module 节点记录限定模块路径,共享模块多次执行
    得到 `path_0/path_1` 消歧(torchvision NodePathTracer 等价物);叶子子树参数不再产生
    悬空 `get_attr`;
  - 支持 `concrete_args={...}`:列出的参数特化 baked 进图,不生成 placeholder,
    结果 signature 同步收缩。
- **特征提取**(对齐 torchvision `models.feature_extraction`):
  - `get_graph_node_names(model)` → `(train_names, eval_names)`,合并语义节点名与叶子
    模块路径,排序去重;
  - `create_feature_extractor(model, return_nodes=..., train_return_nodes=/eval_return_nodes=...)`:
    双模式各剪枝一张图(output 先删后建 → DCE → lint),返回 `DualFeatureExtractor`
    (`nn.Module`,`.train()/.eval()` 切换活动图);叶子子模块 deepcopy 按原限定名注册,
    `state_dict` 与原模型键兼容可直接载入预训练权重。
- **可视化**(参考 `third_party/torchviz`、`third_party/torchview`):
  - `Graph.to_dot()`:零依赖生成 DOT 文本,节点按 op 类型着色(placeholder 蓝 /
    get_attr 橙 / function·method 黄 / module 绿 / output 淡绿),kwargs 边带标签;
  - `Graph.draw("model.png")`:检测 `dot` 二进制渲染 PNG/SVG/PDF;无 Graphviz 时落盘
    `.gv` 并给出可读指引。
- **测试**:`test/test_graph.py` 覆盖命名唯一化、output 单例、erase/replace、concrete_args、
  call_module 产出与 qualname 消歧、特征提取正确性/state_dict 兼容/双模式切换、DOT 导出。
- **附带最小修复**:`p10/src/backend/cuda/IndexingKernels.cu` scatter 分发——`if (Add)`
  改 `if constexpr`,assign 模式全 dtype 实例化不再要求 atomic 重载;add 模式分发收窄至
  Float32/Float64/Int32/Int64(与既有运行时契约一致)。

### P0:L2 pass 体系落地(compiler/passes.py)

- `PassManager`(对齐 `torch.fx.passes.infra`):按序执行至不动点(`run_passes_once`
  可退化为单轮),每轮后 `lint()`;`PassBase/PassResult` 契约同 fx。
- 内置 pass:`DeadCodeElimination`(包装既有 DCE)、`ConstFold`
  (operator 白名单 + 常量参数才折叠;张量操作数与运行期非法常量——如除零——
  保持原语义不折叠)、`ShapeProp`(按占位符名绑定示例输入解释执行,写
  `meta["val"]/meta["tensor_shape"]`,供 to_dot tooltip 与后续形状感知 pass 使用;
  Node/Proxy 符号目标的图跳过标注防污染)。
- 接线:`compiler/api.py::_compile_region` 默认管线改为
  ConstFold → DeadCodeElimination,绑定 backend_inputs 后追加 ShapeProp
  (失败仅放弃 meta,不阻断编译)。
- 导出:`tensorplay.compiler.*` 与 `tensorplay.graph.*` 双命名空间。
- 验证:HEAD 快照 worktree 中 test_passes/test_graph/test_compile 共 46/46 通过;
  更广套件的既有失败(einsum/serialization 等)经对照实验确认为快照缺失未提交
  修复所致,与本批改动无关。

### P1:L1-D1 控制流静态特化(元数据具体化)

- `Tracer().trace(model, sample_inputs={名: 值})`:捕获期为占位符绑定示例值。
  `api._compile_region` 自动把本次调用的实参经签名绑定后传入,`GraphModule.meta
  ["sample_inputs"]` 留档供后端/调试。
- **语义边界 = torch 同款**:元数据(shape/dtype/device/ndim/len)在捕获期解析为
  具体值——它们本就是编译特化签名的键,不引入新的重编译条件;张量数据仍严格符号,
  数据依赖分支(`bool((x>0).all())`)继续 `GraphCaptureError`(fullgraph 下传播)。
- 效果:`if x.shape[0] > 2`、`range(x.ndim)` 循环、`len(x)` 切片等 Python 控制
  流现在可 fullgraph 静态特化;不同 shape 触发各自特化,与 torch.compile 一致。
- 附带:`Proxy` 的 data-dependent 拒绝消息统一(torch.export strict 措辞);
  `compiler.registry.unregister_backend(name)` 公共化(测试/工具注销后端)。
- 测试:`test/test_control_flow.py`(8 例):静态分支、图内容断言、循环展开、
  len 切片、数据依赖拒绝、样本留档、无样本回退符号路径;原
  `test_fullgraph_rejects_python_control_flow` 拆分为"数据依赖拒绝"+"元数据特化
  正向"两例。四套件合计 54/54。

### P2:L3 守卫版符号形状(dynamic 缓存安全化)

- **问题**:dynamic 模式下缓存键泛化为 ("dynamic", rank),但若捕获期发生
  `x.shape[...]`/`len(x)` 读取并驱动了 Python 分支,分支结果被 bake 进图——
  其他尺寸复用该特化会静默算错。
- **机制**:`Tracer.metadata_touches` 记录占位符级元数据读取(shape/len/ndim/
  dtype/device/requires_grad);`api._extract_shape_guard_params` 提取 shape 类
  触碰参数;首次捕获后提升为 `guard_param_names`,缓存键追加
  ("shape-guards", 精确签名) 分量并清空旧键重存。dtype/device/requires_grad
  本就在签名内,无需额外守卫。
- **效果**:被读形状的参数按尺寸精确重特化(正确性),未触碰参数保持通配
  (dynamic 收益保留);静态模式行为不变。
- 测试:`test/test_symbolic.py` 5 例(守卫重特化、通配复用、参数级作用域、
  meta 留档、静态模式回归)。五套件合计 59/59。
- 备注:完整 sympy 符号表达式系统(size 变量约束传播)仍属 L3 后续,需配合
  native lowering(stax M 系列)才有落地价值;本批先堵正确性缺口。

## 图编译系统 L4(P3)—— AOT 双 partitioner

- `partition_default`:joint-graph 标签切分(结构判据:有 backward 消费者即保存),
  torch 契约 `(fwd..., bwd...)` + `num_fwd_outputs` 边界;
- `partition_min_cut`(P3-L4b 设计落地):自研 Edmonds-Karp 最大流/最小割,
  容量模型(候选保存=字节权重、必存=get_attr/fusible 链内部/双侧消费节点 ∞)、
  backward 重算闭包递归克隆;`memory_budget` 权重化;
- `build_aot(partitioner="default"|"min_cut")` 分发;sweep 阶段 tagged 链加产出
  叶子梯度,role 标签(tangent/leaf/saved)按名绑定彻底替代位置假设;
- test_aot:梯度用例按 policy×partitioner 参数化 + 3 个结构性验收
  (role 一致性 / saved 单调性 / budget 行为);
- 数值级对拍与 SIGSEGV 排查待原生层就绪后统一执行。
