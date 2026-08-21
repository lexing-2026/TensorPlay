结论：TensorPlay 已经是一个“真实可运行的轻量计算框架”，不是 NumPy 包装；P10 eager 内核、TPX 自动微分、CPU oneDNN/OpenMP、CUDA/cuBLAS/cuDNN 路径都存在。但目前更接近“教学/研究型 alpha”，距离 Torch 的主要差距不在单个 GEMM/Conv 内核，而在正确性契约、运行时、编译器、CUDA 内存/流和工程门禁。

## 1. 当前架构

```text
Python Tensor / nn / optim
          │
          ▼
     nanobind _C
          │
          ▼
生成的 TPX Autograd Wrapper  ← derivatives.yaml
          │
          ▼
       P10 Tensor
          │
          ▼
字符串 Dispatcher
     ┌────┴────┐
 CPU Kernels   CUDA Kernels
OpenMP/oneDNN  custom/cuBLAS/cuDNN

Stax Tracer ──► CPU Python Interpreter / Triton pointwise 原型
               （尚未融入主 eager 执行路径）
```

四层划分本身清晰，见 [README.zh.md](/home/bluemoon/projects/TensorPlay/README.zh.md:128)。TensorImpl 已有 Storage、offset、shape/stride、dtype、device、version counter 和 oneDNN 缓存元数据，见 [TensorImpl.h](/home/bluemoon/projects/TensorPlay/p10/include/TensorImpl.h:21)。

## 2. 与 Torch 的核心差距

| 领域 | TensorPlay 当前状态 | Torch 参照 | 判断 |
|---|---|---|---|
| 算子规模 | 187 个 schema、53 条反向规则；约 186 个 CPU、134 个 CUDA 注册项 | PyTorch 源码约 2,585 个 native schema、688 条反向规则；本机运行时 3,599 个 operator name | 约一个数量级差距，CUDA 覆盖更不完整 |
| Dispatcher | 每次按字符串查两层 `unordered_map`，热路径带全局 mutex，只按 CPU/CUDA 选 key，[Dispatcher.h](/home/bluemoon/projects/TensorPlay/p10/include/Dispatcher.h:15)、[Dispatcher.cpp](/home/bluemoon/projects/TensorPlay/p10/src/Dispatcher.cpp:11) | DispatchKeySet、Meta/Autograd/AMP/Sparse/Functionalize 等 key，固定小型 dispatch table，[PyTorch DispatchKey.h](/home/bluemoon/projects/TensorPlay/third_party/pytorch/c10/core/DispatchKey.h:36)、[OperatorEntry.h](/home/bluemoon/projects/TensorPlay/third_party/pytorch/aten/src/ATen/core/dispatch/OperatorEntry.h:237) | 扩展性、组合语义和并发热路径差距大 |
| Autograd | 单线程 priority queue；`create_graph` 参数未实际控制 GradMode，[Engine.h](/home/bluemoon/projects/TensorPlay/tpx/include/Engine.h:33) | 设备队列、线程池、reentrant backward、stream 同步、异常与 hook，[PyTorch engine.h](/home/bluemoon/projects/TensorPlay/third_party/pytorch/torch/csrc/autograd/engine.h:87) | 当前首先是语义安全问题，其次才是性能 |
| 原地修改安全 | 有 version counter 类，但没有找到任何 `bump()` 调用或 SavedVariable 检查，[VariableVersion.h](/home/bluemoon/projects/TensorPlay/p10/include/VariableVersion.h:8) | 保存版本并在 backward 时检测原地修改 | 已实测产生静默错误梯度 |
| CPU 核心内核 | FP32 GEMM/Conv 已有 oneDNN/OpenMP，特定形状接近 Torch；CPU exact-size caching allocator 是亮点 | MKL、oneDNN、TensorIterator、统一线程池、算子级粒度策略 | 大矩阵不错，reduction、小算子并行和通用 stride 较弱 |
| CUDA 运行时 | GEMM 支持 TF32，但只支持 FP32/FP64；输出和 cuDNN workspace 频繁 `cudaMalloc/free`，[LinearAlgebraKernels.cu](/home/bluemoon/projects/TensorPlay/p10/src/backend/cuda/LinearAlgebraKernels.cu:73)、[ConvKernels.cu](/home/bluemoon/projects/TensorPlay/p10/src/backend/cuda/ConvKernels.cu:121) | stream-aware caching allocator、cuBLASLt、cuDNN plan cache、CUDA Graph、AMP | 这是 GPU 性能的首要结构性瓶颈 |
| CUDA Stream/Event | Python Stream/Event 是 no-op，elapsed time 恒为 0，[cuda.py](/home/bluemoon/projects/TensorPlay/tensorplay/cuda.py:113)；handle 是进程级单例，[CUDAContext.cpp](/home/bluemoon/projects/TensorPlay/p10/src/backend/cuda/CUDAContext.cpp:37) | 真正的多流、事件、per-device/thread handle、recordStream | 无法可靠并发、计时或做异步生命周期管理 |
| dtype/AMP | 没有 float16、bfloat16、float8；`is_autocast_available()` 直接 `pass`，[DType.h](/home/bluemoon/projects/TensorPlay/p10/include/DType.h:48)、[autocast_mode.py](/home/bluemoon/projects/TensorPlay/tensorplay/amp/autocast_mode.py:26) | AMP、Tensor Core dtype、量化与复杂类型 | 深度学习训练吞吐差距会非常大 |
| Stax 编译器 | CPU backend 是逐节点 Python 解释器；Triton 假定整图为平坦 1D pointwise，[cpu.py](/home/bluemoon/projects/TensorPlay/tensorplay/backends/cpu.py:12)、[triton.py](/home/bluemoon/projects/TensorPlay/tensorplay/backends/triton.py:52) | Dynamo + AOTAutograd + Inductor、guards、dynamic shapes、functionalization | 当前是编译器原型，不是可替代 `torch.compile` 的执行栈 |
| 生态能力 | DataLoader 包缺失；AMP 是桩；ONNX 使用了 Stax 中不存在的 API；distributed 不存在 | DataLoader、DDP/FSDP、ONNX/export、profiler、distributed collectives | Python API 外观明显领先于实际实现 |
| 测试/发布 | 175 个测试，但部分 parity 失败只打印并返回，不会 fail，[test_op_parity.py](/home/bluemoon/projects/TensorPlay/test/test_op_parity.py:28) | 大规模多平台 CI、OpInfo/gradcheck/sanitizer | 当前测试不能作为兼容性证明 |

两个已复现的 autograd 错误：

- `x=2; y=x*x; x.fill_(10); y.backward()` 没有报错，返回错误梯度 `20`；Torch 会因版本不一致拒绝 backward。
- `create_graph=False` 后，`x.grad.requires_grad` 仍为 `True`，导致无谓构图和内存开销。

## 3. CPU 实测

环境：Intel Core Ultra 7 265K、Python 3.13；TensorPlay 为本地 CPU+oneDNN 3.4.1 构建，实际 BLAS 未启用；Torch 2.13.0+cu130，MKL+oneDNN 3.12。由于顶层包不能直接导入，TensorPlay 数据是绕过 Python 包装层、直接调用 `_C` 得到的核心性能。

下表中倍率为 `TensorPlay / Torch`，大于 1 表示 TensorPlay 更慢。

| 测试 | 单线程：TP / Torch | 8 线程：TP / Torch | 结论 |
|---|---:|---:|---|
| 1 元素 add | 0.47 / 0.52 µs，0.90× | 6.69 / 0.54 µs，12.4× | OpenMP 缺少最小粒度阈值 |
| 1M 元素 pointwise 三段链 | 855 / 3568 µs，0.24× | 437 / 1094 µs，0.40× | 热缓存 eager 表现很好；推测主要受益于 exact-size CPU allocator，不是 Stax 融合 |
| 1M 元素 sum | 406 / 63 µs，6.4× | 405 / 15.5 µs，26.0× | reduction 几乎没有多线程扩展 |
| 1024² GEMM | 14.69 / 14.76 ms，1.00× | 3.02 / 3.18 ms，0.95× | GEMM 已接近 Torch |
| 1×64×56² Conv3d-like 2D shape | 1.52 / 1.56 ms，0.98× | 0.447 / 0.371 ms，1.21× | Conv 主路径基本合格 |
| 512² pointwise 前反向 | 446 / 237 µs，1.89× | 661 / 101 µs，6.52× | TPX engine、额外构图和并行策略是瓶颈 |
| Stax CPU vs `torch.compile` pointwise | 947 / 379 µs，2.5× | 517 / 37.8 µs，13.7× | Stax CPU 当前反而比 TensorPlay eager 慢 |

因此性能结论不是“TensorPlay 全面慢”：FP32 GEMM/Conv 与部分 eager pointwise 已经不错；最突出的问题是 reduction、多线程小算子、autograd 和编译器。

CUDA 没有给出数字：截至 2026-08-17，你提供的 [cu130 索引](https://download.tensorplay.cn/whl/cu130/tensorplay/)只列出 `win_amd64` wheel，本机 Linux 执行安装命令无匹配版本；强制下载列出的 Win313 文件又返回 404。发布包确实存在，[PyPI 元数据](https://pypi.org/pypi/tensorplay/json)也包含 1.0.0rc0，但当前无法用于这台 Linux 机器的 CUDA 对照。

## 4. 设计优化表

| 优先级 | 设计项 | 建议实现 | 验收门槛 |
|---|---|---|---|
| P0 | 正确性与发布门禁 | 先修 Conv3d 内存破坏、wheel RPATH、OpenMP 链接、缺失 `utils.data`；增加 Linux/Windows、CPU/CUDA wheel smoke CI | 干净 venv 一条命令安装导入；ASan/UBSan 无崩溃；完整 pytest 可收集运行 |
| P0 | Autograd 安全模型 | SavedVariable 保存版本；所有原地/视图写入 bump；`create_graph=False` 下关闭 GradMode；正确处理 retain_graph、hooks、异常 | gradcheck/gradgradcheck；原地修改必须报错；二阶梯度和 Torch 对齐 |
| P0 | 统一 operator runtime | schema 作为唯一事实源；引入 DispatchKeySet、Meta、Autograd、Autocast、Composite key；kernel handle 缓存，热路径移除字符串与 mutex | schema、注册、Python stub 自动一致；并发 dispatcher 测试通过 |
| P0 | CUDA 执行契约 | 真正的 DeviceGuard、Stream/Event、异步 copy、per-device/thread handles、stream-aware allocator | 双流可重叠；事件时间非零且正确；多 GPU 不再硬编码 device 0 |
| P1 | TensorIterator/类型提升 | 统一 broadcasting、stride、dtype promotion、contiguous fast path，避免每个算子重复 clone | 非连续、broadcast、mixed dtype OpInfo 对齐 |
| P1 | Stax 编译契约 | 增加 Meta tensor、shape/dtype/device guards、graph break、functionalization、partition、真实 lowering 与缓存 | shape/dtype 改变会安全重编译；编译 CPU 不再解释执行 |
| P1 | API 能力真实性 | 未实现 API 明确抛 `NotImplementedError`；补齐或暂时移除 AMP、ONNX、DataLoader 的假兼容入口 | 文档示例全部在 CI 中执行 |
| P2 | 分布式与可观测性 | 在单机语义稳定后再做 NCCL/Gloo、DDP、profiler、memory snapshot | 多卡一致性、故障传播和 trace 可验证 |

## 5. 性能优化表

| 优先级 | 瓶颈证据 | 优化方向 | 性能验收 |
|---|---|---|---|
| P0 | 8 线程 1 元素 add 慢 12.4× | 建立统一 `parallel_for(begin,end,grain)`；小于阈值串行；线程池替代每算子裸 OpenMP | 小算子 8 线程开销不超过单线程 1.5× |
| P0 | sum 8 线程慢 26×且没有扩展 | 分块树形 reduction、SIMD 累积、线程局部 partial buffer；按维度和连续性专门化 | 1M sum 在 8 线程至少获得 4×单线程加速 |
| P0 | backward 8 线程慢 6.5× | 修复 `create_graph=False`；用 vector/small-map 替代每节点 unordered_map；CPU/device ready queue 并行执行 | 代表性前反向控制在 Torch eager 1.5×以内 |
| P0 | CUDA 每次 `cudaMalloc/free`，cuDNN workspace 也现申请 | 分尺寸、分 stream caching allocator；workspace pool；`recordStream`；OOM 回收与 per-device stats | warmup 后稳态零 `cudaMalloc/free`；无跨流 use-after-free |
| P1 | 无 FP16/BF16 | 增加 Half/BFloat16、autocast dispatch、GradScaler；使用 cuBLASLt epilogue 和 cuDNN frontend plan cache | Ampere 上 FP16/BF16 GEMM/Conv 使用 Tensor Core，吞吐达到 Torch 同配置的 80%+ |
| P1 | Stax CPU 比 eager 慢，Torch compile 快 13.7× | pointwise/reduction fusion、CPU LLVM/C++ codegen或 oneDNN Graph、Triton 分区；加入内存规划和 buffer reuse | 三段 pointwise 生成一个 kernel；性能接近 `torch.compile` 1.5×以内 |
| P1 | Conv/GEMM 每次重建 descriptor/primitive | 缓存 oneDNN primitive、reordered weight、cuDNN plan；支持 channels-last 和预打包 | 稳态不重复建 plan/reorder；Conv 8 线程追平 Torch |
| P2 | Python/dispatcher/optimizer 小操作多 | op handle 内联缓存、无锁热表、foreach/fused optimizer、批量参数更新 | Adam/SGD kernel launch 数和参数数解耦 |

## 6. 建议落地顺序

1. 先做 release/import、Conv3d 崩溃、autograd 版本安全和真实断言测试。
2. 再做 CPU parallel/reduction 与 CUDA allocator/stream。
3. 然后重构 Dispatcher + TensorIterator，给 AMP/Meta/Stax 提供正确基础。
4. 最后投入编译器、distributed、ONNX 等生态能力。

本次定向核心测试有 33 项通过；完整套件先被未声明的 `matplotlib` 阻断，排除后在 Conv3d 稳定触发 exit 139。源码跟踪文件未修改；PyTorch 已以稀疏浅克隆保留在 `third_party/pytorch`，基准提交为 `893b6406afc1a6384ab6fae8a2247d03cc230d87`，约 146 MB，目前为未跟踪目录。构建还留下了被 Git 忽略的 `_C`/`lib` 二进制产物。
