# 1. 版本号规则对齐 torch + nightly 发布通道 + release notes 机制（2026-08-28）

- **版本号规则对齐 torch**：`version.txt` 由 `1.0.0rc0` 改为 `1.0.0a0`——torch 规则为
  main 分支恒为 `<目标版本>a0`（dev 构建显示 `1.0.0a0+git<sha7>`），rc 版本只存在于
  release 分支且从 rc1 起编；正式发布经 `TENSORPLAY_BUILD_VERSION/NUMBER` 注入，
  CUDA 轮子带 `+cuXXX` local label（同 torch `2.8.0+cu128` 的
  `binary_populate_env.sh` 规则）。
- **新增 nightly 手动发布通道**（`.github/workflows/nightly.yml`，workflow_dispatch，
  无 cron）：双来源——`build-from-source`（版本默认按 torch nightly 规则
  `X.Y.0.dev<UTC日期>`，由新增的 `tools/generate_tensorplay_version.py::compute_nightly_version()`
  计算，可手动指定版本/`.postN` 同日重发/按平台与 CUDA 变体过滤矩阵）或
  `upload-wheels`（发布前上传名为 `wheels` 的 artifact，版本从轮子文件名读取并校验
  一致性）。变体后缀规则同 torch：CUDA 加 `+cuXXX`、CPU 加 `+cpu`、macOS 无后缀。
  轮子进滚动 `nightly` Release（prerelease），裁剪仅保留最新一天（带护栏：新版本
  轮子不存在时拒绝误删旧版）；索引发布到 pypi-pages 的 `whl/nightly/<variant>/`
  （含 cpu 伪变体），覆盖式更新、stable 索引不受影响。
- **`tools/build_simple_index.py` 增加 nightly 模式**：`--nightly` 只索引滚动 nightly
  Release 的资产；stable 索引跳过 nightly Release；修复 `+cpu`/macOS 无后缀轮子经
  cu124 遗留回退误入 CUDA 索引的问题。
- **release notes 机制原生对齐 torch**：torch 按 PR 的 `release notes: *` 标签聚合
  人工策展，正文直接进 GitHub Release。本仓库：7 个 `release notes: *` 标签
  （frontend/autograd/compiler/kernels/cuda/build/docs）已建，`.github/labeler.yml`
  按路径自动打标签（actions/labeler 工作流，torch 用 pytorch-probot 的等价物）；
  发布文案存 `docs/release-notes/`（`TEMPLATE.md` 为 torch 章节骨架：Highlights/
  BC 变更/弃用/新特性/改进/修复/性能/文档/开发者），`v1.0.0.md` 首版已从
  CHANGELOG 提炼成稿；`publish.yml` 打 tag 时自动 `--notes-file`，缺文件回退
  `--generate-notes`。
- README（中英）新增 nightly 安装说明（`pip install --pre tensorplay
  --index-url https://download.tensorplay.cn/whl/nightly/<variant>/`）。

# 2. 编译器减归约/逐元素链收口（L5-PERF 轮四 + 收口轮）（2026-08-27 ～ 08-28）

## 概要

`tensorplay.compile` CUDA 减归约/逐元素链 vs torch/inductor 收口轮:
geomean **1.03x → 1.15x**(4090D, iters=200),TP ≥ torch 8/11;
pw 链 0.84x → 0.98x,sum full 16M 0.38x → 0.58x(残差定位为结构性
Python 启动路径,内核已 parity),dims/epilogue 链 2.25-2.51x。
原生层收口轮:`CUDAReduce.cuh` 每调用 `cudaGetDeviceProperties`
(固定 ~1.5ms/call)消除,原生 eager sum 1M/4M 降至 8.5/23.1µs
(1.3-1.4x),详见下文原生层收口轮;其后的 global-reduce 带宽轮
逐行对齐 torch Reduce.cuh CTA 公式 + (32×8) 高块 + 单内核 tag-init
完成,**sum full 16M tp_eager=0.0236ms=torch(1.00x),记分板
9/11、geomean 1.21x**,见下文 global-reduce 带宽轮。

## 轮四(2026-08-27): evict/.cg + packed argmax + 静默窗口复核

### pw 链内核 parity 确认
- profiler 拆解:`tp_stax` 与 `torch_compile`(inductor) 纯内核 GPU 时间
  **145.0 vs 145.3 µs, 完全持平**。0.90x 差距全在 Python 启动器:TP
  wrapper 闭包链(~19 µs/call) vs inductor 静态 launcher(~0)。
- 改动:① `_load_lines` 参考布局无掩码加载加 `cache_modifier='.cg'`;② dims r-loop
  与 split-persistent 内循环加载加 `eviction_policy='evict_first'`;③ 固定配置
  launch 注入 literal grid + `XBLOCK`/`num_warps` constexpr kwarg。
- 实测(shape 4096×4096, warmup=30, iters=200,min-of-window):
  - pw gelu-ish tanh/exp chain: **0.90x**
  - epilogue chain sum(dim=1)*3+1: **1.47x**
  - full-sum sigmoid: **1.44x**
  - sum full 16M: 0.37x(噪声,需独占机器复核)

### 原生 argmax packed warp-shuffle
- `CUDAReduce.cuh` 新增 `PackedArgMaxOps`:将(float value, int64 index)
  打包为单个 u64 `[key(32)|~index(32)]`,warp shuffle 层只做一次整数
  max(5次 shuffles 替代原来 10次 + 分支 comparator),NaN/±0/首现 tie
  保持与原 ArgOps 位等价。
- `ReductionKernels.cu` argmax_same_dtype:float-family(num_inputs ≤ INT32_MAX)
  自动走 packed 路径,half/BFloat16 通过 float 提升兼容。
- 测试:`test/test_cuda_reductions.py` 8 例全绿(含 NaN 首现、±0、
  ±inf、tie、4096-long 行 vec4 跨路径)。

### 决策缓存盐
- `stax_autotune.decision_key` 加入 `TUNING_VERSION` 盐,消除 codegen 升级后
  旧决策长驻问题;CANDIDATE_CONFIGS 新增 (2048,4) 16-elem/thread 探子。

## 收口轮(2026-08-28): 静态 fast-launch + tuner 抗噪 + persistent split 修复

- **runtime/fastlaunch.py(新)**:静态 fast-launch。首次 triton dispatch 后
  记录 CompiledKernel `(run, function, packed_metadata)`,后续直接
  `kernel.run(...)`,跳过 binder/specialization/cache 查找。守卫:tensor
  实参 divisibility-16 对齐(OR 位测试)、标量等于记录值、无 profiling
  hooks;失配回退常规 dispatch;快路径异常自愈(_rec=None 重放)。
  对齐 torch/_inductor triton_heuristics CachingAutotuner;native
  static_triton_launcher 不采用(需 _C 重建)。
- **codegen/triton.py**:四类 launcher(single/dims/split/pw)生成 fast
  path;persistent split 修复(误发 classic tail、死 preamble、stride
  整除越界判定、neutral 填充、输出重掩码);dims 四元组 RBLOCK 覆盖生效;
  `_dims_decision_key` 固化;`_SPLIT_CANDIDATES`/`_DIM_REDUCTION_CANDIDATES`
  更新(classic+persistent、Inductor INNER 带);静态 `_ws` workspace;
  `_supports_runtime_inputs` 每调用守卫微优化。
  `_CODEGEN_VERSION = "m8-2026-08-28-fastlaunch"`。
- **runtime/stax_autotune.py**:`bench_launch` 首次 launch 不计时(吃掉懒
  JIT 编译),`warmup_ms` 成为真实稳态预热;新增 `bench_candidates`
  跨候选交错轮询(2 轮取 min,后轮失败保留先前结果),消除同候选
  30.7↔60.4µs 进程间方差与坏决策永久缓存;`pick_config`/dims/split 三个
  调参点接入。`TUNING_VERSION = "t9-fastlaunch"`。
- **benchmark/benchmark_vs_torch.py**:修正 docstring(--matsize 从未实现)。

## 实测(cc RTX 4090 D)

- kernel_launch CPU 45 → 9.7 µs/call;compiled 全链 51.3 → 20.2 µs/call;
- sum full 16M:tp_stax 62.5 → 41.0µs(0.38x → 0.58x);chain full-sum
  sigmoid 1.48x → 2.25x;chain sum(dim=1) 1.73x → 2.51x;pw 0.84x → 0.98x。
- sum full 16M closure:launch-only 事件 23.6-24.6µs ≈ torch eager 全链
  23.6µs(内核 parity,同为 L2 驻留条件);残差 ~17.4µs = 前端 10.5µs +
  提交延迟,对应 torch 的 C++ 静态 launcher/guard manager;原生方案受
  OptimizerMTA.cuh 构建冲突阻塞,Python 侧后续选项为接通 CudaGraphManager。

## 测试

- test_triton_reduction:+4 fast-launch 运行时回归 + fast-path 结构断言;
- test_stax_autotune:+bench_candidates 两例;pick_config 更新为轮询协议;
- cc:focused 129 passed + 1 xfailed;编译器套件 142 passed。

## 复核轮

- **静默窗口复核(2026-08-28 cc RTX 4090 D)**:测试套件
  `test_triton_reduction + test_stax_autotune + test_cuda_reductions`
  **65 passed, 1 xfailed**(`test_triton_reduction` — triton autotune 在
  共享机器上偶发,属已知噪声)。argmax packed parity vs torch eager
  (shape 4096×4096 last-dim):基本/tie(all-same)/NaN-first(首现)/±inf edge
  全部 **match**;bench(tp min-of-200 vs torch min-of-200) ~1.03x(torch
  0.03µs vs tp 0.02µs,量级为 kernel launch 噪声,实际计算时间在 µs 级)。
  sum-full 16M 噪声读数 0.37x 未复核(共享机器仍有其他进程占用)。
- **同日第二跑**:全表复现(geomean 1.15x, 8/11,sum 16M 0.55-0.58x);
  套件 218 passed + 1 xfailed。

## 遗留

- pw 链仍 0.90x:核心瓶颈是 TP `compiled()` 闭包链 + triton JITFunction
  dispatch(~19µs/call),inductor 静态 launcher 走 c_wrapper 直连。
  需复现 Inductor StaticTritonCompileResult 路径,范围超出本轮。
- pw 链 end-to-end 超越 inductor 仍需静态 launcher 路径复现,本轮未达成。

## 原生层收口轮(2026-08-28): eager 归约固定 ~1.5ms/call 开销消除

- **根因**:`CUDAReduce.cuh` global-reduce 分支每次 launch 调
  `cudaGetDeviceProperties`(同步全量查询,固定 ~1.5ms/call,与数据量
  无关);先例:`ReductionKernels.cu` Muon 路径同款查询 ~0.9ms/step。
- **修复**:新增 `DeviceReduceProps`/`query_reduce_device_props`/
  `reduce_device_props`,按设备缓存 `cudaDeviceGetAttribute` 查询的
  `multiProcessorCount`/`maxThreadsPerMultiProcessor`
  (`kMaxCachedReduceDevices = 64`);`kReductionEngineRevision` 2 → 3,
  `ReductionKernels.cu` 静态断言同步。
- **最小构建修复(他人工作流文件,仅前向声明)**:`PointwiseKernels.cu`
  缺 `binary_float_op_kernel_v2` 声明,补 template 前向声明,不重构该文件。
- **实测(cc 4090D,缩放探针 L2 驻留 min-of-window,ratio=tp/torch)**:
  1M 1561.6 → 8.5µs(torch 6.4µs,1.3x);4M 1564.9 → 23.1µs
  (torch 16.6µs,1.4x);16M 1573.4 → 165.4µs(torch 27.6µs,6.0x);
  64M 1819.4 → 769.6µs(torch 285.3µs,2.7x)。
- **harness**:sum full 16M tp_eager ~1.57ms → 0.085ms;best_tp 仍为
  tp_stax 0.041ms,记分板不变(TP ≥ torch 8/11,geomean 1.14x)。
- **测试**:focused 套件(test_cuda_reductions + test_triton_reduction +
  test_stax_autotune + test_compile)97 passed + 1 xfailed(缓存全清)。
- **遗留**:16M/64M 大尺寸归约带宽 406/349GB/s vs torch 2433/941GB/s
  ——global-reduce launch 配置的大输入扩展性问题,与本轮固定开销无关。
  → 已由下节 global-reduce 带宽轮解决。

## 原生 global-reduce 带宽轮(2026-08-28): sum full 16M 达 torch 持平,记分板 9/11

- **根因**(逐行对照 torch Reduce.cuh setReduceConfig):torch CTA 数 =
  clamp(target_grid, values/256, values/16)(元素制)→ 16M 全和
  8192 CTAs 全机占满;tp 旧公式 + 256 CTA 硬顶 → 8192 线程(2 warp/SM)
  ——406/349GB/s vs 2433/941GB/s 的全部来源;且 tp 的 values_per_thread
  按向量化 unit 计(vec4 低报 4 倍),torch 按元素计。
- **修复一**:CTA 公式逐行对齐 torch clamp(元素制 values,删 256 硬顶,
  加 gridDim.y 护栏)。
- **修复二**:global-reduce 高块 (32×8)(预测触发时,dim1==1 且
  num_inputs≥16384):同地址原子次数与末 CTA 折叠长度降 8 倍,总线程
  不变。
- **修复三**:单内核完成替代双内核 finalize 与 torch 的
  memset+semaphore:scratch=[counters][flags][partials],每启动唯一
  64-bit tag,(x,y==0) CTA 内核开头置零 counter 并发布 tag,各 CTA
  leader 原子加前单次 flag 检查,末 CTA 以 8 独立累加器折叠 —— 免
  per-launch cudaMemsetAsync(~1.3µs)与第二内核启动(~2µs)。
  `kReductionEngineRevision` 3 → 4。
- **实测(cc 4090D,同数据 fp64 对照,ratio=tp/torch)**:1M 5.1µs
  (torch 7.0,0.73x);4M 9.6(9.2,1.04x);16M 21.4(20.6,1.04x);
  64M 284.7(285.3,1.00x)——几何平均 ≈0.95x,总体快于 torch。
- **harness**:sum full 16M tp_eager **0.085 → 0.0236ms = torch 持平
  (1.00x)**,tp_eager 首次低于 tp_stax 成为 best_tp;记分板
  **8/11 → 9/11,geomean 1.14x → 1.21x**;argmax 1.04x → 1.14x。
- **测试**:focused 套件 97 passed + 1 xfailed(缓存全清)。
- **遗留**:4M/16M 残差 1.04x;进一步需 6 blocks/SM 常驻(49→42 寄存器,
  __launch_bounds__ 按 dtype 分档,影响 Welford 等高寄存器实例)。

# 3. autograd.Function 全面对齐 torch + 引擎原生增强 + forward-mode AD（2026-08-26）

以 third_party/pytorch 源码为蓝本逐文件对比（THPFunction_apply/unpack_input/
_wrap_outputs/PyNode::apply/engine.cpp），追平并超越：

- **Py 层 `Function` 对齐**：ctx-first backward（实证 torch 约定）；grad_fn 与
  ctx 双侧 hooks（torch 签名 `(grad_inputs, grad_outputs)` / prehook
  `(grad_outputs,)`）；`to_save`(tuple 契约)/`metadata`/`next_functions`/
  `requires_grad`/`mark_dirty`(版本号 bump)/`once_differentiable`/
  `InplaceFunction`/`NestedIOFunction`；梯度数校验含"多余 None 截断"规则；
  `FunctionMeta.name`、`generate_vmap_rule`。
- **单入口融合**：新增 `custom_function_apply` —— 节点创建/unpack_input/
  AutoGradMode(false) forward/setup_context/wrap_outputs 一次 C 穿越完成，
  缓存型 `fast_is_tensor` 类型检查，needs 以 tuple 直返。
- **引擎 InputMetadata 原生化**：Edge 记录 shape/dtype/device 三元组；
  Node 增 `OutputSlotMeta`（attach 时采集输出元数据）；
  Engine 在 apply 前 C++ 零填充缺失梯度槽（`set_materialize_grads(False)`
  透传 None），Python 热路径剔除物化分支。PyNode backward 输入槽语义修正为
  前向输出数（对齐 CustomFunctionNode），缺失槽自动补 None。
- **PyNode::apply** 梯度转换改 PyTuple 直构。
- **saved_tensors_hooks**：新模块 `tensorplay/autograd/graph.py`，pack 在
  save 时、unpack 随 ctx 快照走。
- **forward-mode AD 原生核心**：`tensorplay/autograd/forward_ad.py`
  （level 栈/DualTensor/种子算子 JVP），反向模式与有限差分双重验证。
- **C++ 编译期自定义算子端到端打通**：修复生成器 GIL 作用域 bug（wrap/error
  必须持 GIL）；桥接层重构为单拷贝 `libtp_python.so`（libtorch_python 模型）；
  新增 test/test_cpp_custom_op.py 端到端测试与 benchmark/autograd_function_overhead.py。

性能（静默机器）：custom Function 端到端 fwd+bwd **40-45µs vs torch 62-80µs**
（快 ~1.7×）；backward 腿 13.5µs vs torch 58.7µs（快 ~4×）；apply 层
16.9µs vs torch 18.8µs。

# 4. 自定义算子全通道对齐 torch：三层打底 + Py 层补齐 + 调度层性能超越 + tile-lang/tvm-ffi（2026-08-24 ～ 08-25）

08-24 对照 `third_party/pytorch` 搭建用户自定义 TVM/Triton 算子集成三层；
08-25 对照 torch 2.13 `torch.library` 全量公开面逐方法审计补齐，四条通道全覆盖。

## 算子注册（torch.library 对齐）

- 新增 `tensorplay/library.py`（`torch.library` 对标）：`custom_op/triton_op`
  装饰器（被装饰函数即默认 kernel，`device_types` 声明覆盖面）、
  `register_kernel/register_fake/register_autograd`（autograd 经既有 PyNode 引擎，
  `backward(ctx, *grads)` + `setup_context` 签名与 torch 一致）、`Library`
  （DEF/IMPL/FRAGMENT，schema 字符串仅取限定名）、`get_op/has_op`、顶层
  `tensorplay.ops.<ns>.<op>` 包命名空间（`_ops.py`，同时承接 `load_library`
  委托，修复 `_classes.py` 悬空引用）。
- **Py 层签名对齐**：`custom_op/triton_op/register_kernel/register_fake/register_vmap`
  支持 fn 位置参数；新增 `schema=` 参数附着与 `.schema` 内省；
  `register_autograd` backward 改位置传参。修复 eager 调用丢弃 kwargs 的对齐 bug。
- **补齐缺失 API**：`CustomOpDef.set_kernel_enabled`（上下文管理器，禁用具体设备
  内核回退 composite 槽）、`get_kernel`（严格按键查找，缺失抛 LookupError）、
  `register_autocast`（autocast 启用时浮点入参按规则转换后再进内核）、
  `register_vmap`（接入 autograd.Function.vmap 钩子）、顶层
  `define/impl/impl_abstract`、`infer_schema`（类型注解→schema 字符串，
  mutates_args 输出 `Tensor(a!)` 标记）、`opcheck` 四项检测
  （test_schema 未声明变更/输出别名输入、test_faketensor 元数据一致性、
  test_autograd_registration 断图探测——TP 内核为隐式复合微分故缺公式合法、
  test_aot_dispatch_dynamic 捕获重放一致性）。

## Triton 集成

- `wrap_triton` 幂等包装 `@triton.jit` kernel；eager 直通启动，
  被 `tensorplay.compile` 捕获到裸 launch（代理参数）时按 torch 语义报
  GraphCaptureError，引导用户走 `triton_op`。捕获后 op 以原生 `custom_op`
  节点进入 Stax 原生图——与 Inductor 对 triton op 的黑盒契约一致
  （不透明=融合屏障，但执行仍是原生的，绝不回退 Python 解释器）。
- **测试升级为真实 JITFunction**：远端 P4 安装 triton 3.7.1（清华镜像；
  sm_61 低于 Triton 的 sm_70 启动下限，真实发射不可行），`test_library.py` 新增
  `RealTritonJITFunctionTest`（无 triton 环境整体跳过、mock 用例保底）：真实
  `@triton.jit` 对象过 `wrap_triton` 类型检查/幂等；`triton_op` 捕获契约——
  单个不透明融合屏障节点、函数体在追踪期零执行（哨兵断言）、内部 add 不泄漏为
  独立图节点。启动级验证受限于硬件（本地无 GPU、P4 为 Pascal），已在用例
  docstring 注明。

## TVM 后端与 tvm-ffi

- 新增 `tensorplay/backends/tvm.py`（`backend="tvm"`），结构对齐
  `torch/_dynamo/backends/tvm.py`（薄委托、缺依赖时报 actionable 错误、
  `has_tvm()`）。点白名单（复用 POINTWISE_FUSED_OP_NAMES 单一事实源）逐节点
  下沉为纯 TIR 内建 te.compute，`create_prim_func` 内联成单 kernel（对应
  Inductor scheduler 的融合语义）；DLPack 零拷贝进出。刻意绕开 topi：unity
  线轮子的 topi 超越函数经 WorkspacePool，与 p10 已加载的 OpenMP 同进程段错误。
  strict_native/回退契约与 stax.triton 一致；训练区保持原生路径。
- **tvm-ffi 直通（C++/CUDA AOT+JIT）**：TP Tensor 实现 DLPack 协议，
  `tvm_ffi.cpp.load_inline` JIT 与 `build_inline→load_module` AOT 两路的
  `tvm::ffi::TensorView` 参数零拷贝直通 TP 张量（实测验证，零适配代码）；
  新增 `test/test_tvm_ffi.py` 覆盖 JIT/AOT 往返 + ffi 内核包进 custom_op 后的
  autograd/opcheck 全过/编译屏障语义保持；docs/source/library.md 增补用法章节。

## tile-lang 一等支持

- `tile_lang_op` + `wrap_tilelang`（已克隆 third_party/tilelang 对齐源码：
  JITKernel `torch_function` 直通、JITImpl 懒工厂 `compile()` 绑定、
  鸭子类型识别 `adapter/torch_function/get_tir`）；raw launch 误捕获给出指路
  GraphCaptureError；编译期单节点不透明融合屏障契约同 triton_op。torch 无此 API
  （Triton 系其自家栈），系按同一契约的超集扩展。

## 原生执行（非解释器）

- Stax 原生图新增 `custom_op` 节点类型（`stax/{include,src}/Graph.h/.cpp`
  执行器钩子 + `set_str_attr` 绑定）。`_lower_native` 遇 `CustomOpDef` 节点
  直接下沉原生图，经 `Ops.cpp` 安装的 executor 回调重入
  `library._native_invoke`——设备分发与 `register_autograd` 全语义保留，
  autograd 可穿透编译图求梯度。`_call_native_op/_has_native_kernel` 供
  `run_native`/测试走真实 findHandle+kernel 表路径（规范 unboxed 约定
  tensors-in/out，composite 槽双注册 CPU/CUDA）。

## Composite 分发键（torch CompositeExplicitAutograd 对齐）

- 形状/视图组合算子批（expand/broadcast_to/tile/stack 族/tensor_split 族/
  atleast/flatten/ravel/moveaxis/swapaxes/argwhere/equal/allclose/fill）
  此前只注册 CPU 键，CUDA 张量直接 `Kernel not found`。按 c10 机制补齐：
  `DispatchKey.h` 新增 `Composite` 键（对应
  `getRuntimeDispatchKeySet(CompositeExplicitAutograd)==backend_dispatch_keyset`
  与 `DefaultBackend` 别名），后端查找未命中时落到该键、显式后端内核可覆盖；
  注册集中于 `p10/src/RegisterComposites.cpp`（对标生成的
  `RegisterCompositeExplicitAutograd.cpp`），声明收口
  `p10/include/ShapeAlignKernels.h`，cpu/cuda 的 ShapeAlign fragment 只留真
  设备内核 `repeat`（上游 MPS: repeat_mps 同款覆盖模式）。顺带修正
  `is_autocast_key` 恒假 / `is_autograd_key` 上界吞新键两个区间谓词。
  远端 P4 实测：equal/allclose/expand/isclose(内含 expand)/argwhere/
  tensor_split 等全链路 GPU 通过，test_compile 既有 CUDA allclose 断言用例
  由失败转通过。

## 调度层热路径优化

- 捕获守卫接 `compiler.graph.capturing()`（无 trace 时完全跳过 proxy 扫描，
  与生成式 functional 包装层同款模式）；内核选择无锁化（GIL 原子读，锁只护写）；
  设备键免 str() 包装；`is_grad_enabled` 模块级别名绑定。
  基准 `benchmark/custom_ops_overhead.py`：调度层净开销 2.3µs vs torch 16.0µs，
  **约 7× 低开销**，端到端调用快 ~2.9×。

## dlpack 修复

- `Tensor.cpp to_dlpack_device` 将 CPU/-1 归一为 device_id=0（DLPack 规范、
  torch 同款）。此前 -1 使 tvm_ffi 判定 device 不匹配走入 workspace 复制路径
  → 段错误。

## 验证

- `test/test_library.py`（29 用例：注册/分发/autograd/Library/包/捕获屏障/
  **原生图下沉断言 `_stax_native_graph` 非空**/**autograd 穿透原生图**）、
  `test/test_backend_tvm.py`（数值 parity、alpha 形态、自定义算子边界回退、
  训练区、shape 变更重编译、CUDA target）。本地全绿（含重建后）；远端 5 套件
  108 passed / 2 skipped；远端 GPU 用例待共享树构建窗口。

# 5. CUDA graphs 原生对齐 torch：原生重写 + 能力补齐 + 回放路径性能超越（2026-08-25）

## 原生重写（对照 at::cuda::CUDAGraph）

原生层重写为 `tensorplay::cuda::graph::CUDAGraph` 类，删除旧的整型句柄自由
函数接口与 Python 侧流切换编排：

- **捕获语义对齐 torch**：`capture_end()` 即完成 `cudaGraphInstantiateWithFlags`
  （原实现推迟到首次 replay 且每次 replay 空转一次 instantiate 检查）；
  `replay()` 直接持有 exec 指针发射，无注册表互斥锁/查表；支持
  `capture_error_mode`（global/thread_local/relaxed → `cudaStreamBeginCapture`
  模式）、自定义捕获流、`enable_debug_mode()/debug_dump()`
  （`cudaGraphDebugDotPrint`，debug 时保留模板）。
- **共享内存池**：新增 `graph_pool_handle()`；allocator 的 `beginGraphCapture`
  支持复用已存在 pool id（惰性创建），多个 graph 经 `graph(pool=...)` 共享
  一个私有池；GraphState 按 pool 引用计数释放段，先 reset 的 graph 不会释放
  仍被其他可执行文件引用的地址。
- **回放快路径（超越点）**：新增 `stage_and_launch(static_inputs, inputs)` ——
  全部输入在单次 Python→C++ 调用内用裸 `cudaMemcpyAsync` D2D 拷贝到静态缓冲
  （绕过 dispatcher/版本号/autograd 记账），并对源 tensor 做 recordStream 防
  提前回收，随后同调用内发射图；非连续/跨设备/类型漂移回退完整 `copy_` 语义。
  `CudaGraphManager.replay` 默认走该路径，torch 无等价单调用接口。
- **CUDAStream 热路径优化**：设备数进程级缓存（原先每次取流都调
  `cudaGetDeviceCount`）；每线程当前流从 unordered_map 换成按 device 下标的
  扁平数组（消除每次 kernel launch 的哈希查找）；`priority()/query()/
  synchronize()` 去掉多余设备 guard（stream 句柄自带 context，c10 同款做法）。
- **Python 层**：`tensorplay/cuda/graphs.py` 重写为原生类薄封装并补齐
  pool/stream/error_mode/debug_dump/export_dot；`compiler/cudagraphs.py` 删除
  Python 流切换与逐输入 copy_ 编排；旧 `_C.cuda_graph_*` 自由函数绑定全部移除。

## 补齐最后三块能力差距

- **条件节点（if/while 图）**：`CUDAGraph.begin_capture_to_if_node /
  begin_capture_to_while_node / set_conditional_handle_for_current_node /
  end_capture_to_conditional_node`，CUDA ≥ 12.4（`cudaGraphConditionalHandle`
  + 捕获内核 `cudaGraphSetConditional` + `cudaStreamBeginCaptureToGraph`，
  版本分支与 torch 相同）。条件体在专属子流上捕获，分配路由经
  `routeStreamToPool/unrouteStreamFromPool` 复用父图私有池；
  嵌套条件以 handle 栈管理。Python 层同名方法 + `_C.conditional_nodes_supported()`。
- **多设备并发捕获**：allocator 路由从单一 `capture_` 插槽改为
  `active_captures_` 列表（线性扫描），不同设备/线程可同时各持一个捕获窗口；
  GraphState 按 `std::thread::id` 键控捕获槽——同线程嵌套仍拒绝，跨线程并行放行。
  `memory_stats` 新增 `active_captures` 字段。
- **make_graphed_callables / make_graphed_autograd_function**：自 torch 完整移植。
  共享池上先 warmup、再按 fwd1..fwdN → bwdN..bwd1 顺序捕获前后向图，
  返回 forward/backward 均为图回放的 autograd Function（once_differentiable）；
  附带最小 pytree（flatten/unflatten 往返含 tuple/list/dict 嵌套）替代
  torch.utils._pytree。nn.Module 传入时替换 forward 并保留 training 状态切换回退。

## 测试/基准

- `test/test_cudagraphs.py` 迁移到新注入面；新增 GPU 门控的
  `test/test_cuda_graph_gpu.py`（eager 一致性、RNG 跨回放新鲜、共享池引用计数、
  批量暂存、DOT 导出、嵌套捕获拒绝、多流回放）；新增
  `benchmark/bench_cudagraph_replay.py` 对比 eager/manual/bulk 与 torch 的
  每迭代主机开销。
- if 条件节点真伪两路验证、双 GPU 双线程并发捕获一致性、graphed callables
  前向输出与 x/W 梯度对齐 eager 三组 GPU 门控测试。

# 6. CUDA 流/图极致优化 + 碎片管理（2026-08-25，同日第二波）

- **当前设备线程级缓存**：所有 `cudaSetDevice` 收口到 `setDeviceCached`/
  `CUDAGuard`（全仓库仅 CUDARuntime.cpp 一处 set 点），`currentDevice()` 与
  `getCurrentCUDAStream(-1)` 从每次一次驱动调用降为一次 TLS 读——kernel launch
  热路径上 torch 仍需 `cudaGetDevice`，此点严格低于 torch 开销。
- **图回放微优化**：`replay()` 仅在设备不匹配时构造 guard（TLS 比较）；
  新增 `replay(stream=...)` 显式流重载，钉在单流的回放循环连 current-stream
  查询都省掉；RNG prologue 保持无 RNG 图零开销。
- **图池碎片整理**：GraphPool 新增地址有序索引与相邻空闲块合并
  （同 segment+stream），混合尺寸捕获共用一个池时不再碎裂成不可用小片；
  `take/insert` 全部走合并路径。
- **OOM 防御梯队**：分配失败时先同步排空该设备的跨流事件 pending 块
  （字节立即可复用），再整体 flush 缓存后重试；最终 OOM 报文附带
  reserved/allocated/free 统计（对齐 torch 的诊断信息量）。
- **碎片可观测性**：新增原生 `memory_stats(device)`（segments、free_blocks、
  free_bytes、largest_free_block、pending_blocks/bytes、graph_pools、capturing），
  绑定到 `_C._cuda.memory_stats` 并接入 `tp.cuda.memory_stats*` 的嵌套字典
  （torch 形状键保留，新增 `allocator` 小节）；`inactive_split*` 键从占位 0
  改为真实值。

# 7. 构建 / CI / 发布对齐 torch，删除自建构建脚本（2026-08-24）

对照 `third_party/pytorch` 的 pyproject.toml 与 `.github/workflows/` 重排构建、CI 与发布，
删除自建的 Python 构建编排脚本：

- **删除自建脚本**：`rebuild.py`（Windows 本地构建编排）、`release.py`（cibuildwheel 发布编排）
  一并移除。构建统一走 scikit-build-core 的 PEP 517 接口（`pip install .` /
  `python -m build --wheel`），与 torch 一致；`CMakeLists.txt` 中引用 rebuild.py 的过时注释同步清理，
  `CONTRIBUTING.md` 的 `python setup.py install` 过时说明改为 PEP 517 命令。
- **pyproject.toml 对齐 torch**：`[tool.scikit-build.env]` 把 `MAX_JOBS` 映射到
  `CMAKE_BUILD_PARALLEL_LEVEL`（torch 同款伞形并行度旋钮）；`build-dir` 固定 `build/`；
  不再显式设 `cmake.build-type`（默认 Release，且允许环境 `CMAKE_BUILD_TYPE` 覆盖，torch 同款）；
  去掉冗余的全局 `-GNinja`（scikit-build-core 默认即 Ninja）；editable 用 redirect 模式且
  关闭 import 时自动重编（torch 同款，避免每次 import 触发 cmake/ninja）；
  sdist 排除 `.github/`、`build/` 及仅作参考的 `third_party/pytorch`（1.7G gitlink）、`third_party/audio`；
  build-system 下限提升到 `scikit-build-core>=1.0`（env 表与 dynamic-metadata 所需）。
- **新增**：`requirements-build.txt` 与 `[dependency-groups] dev`（与 torch 的
  requirements-build.txt ↔ dependency-groups 同步机制一致）。
- **版本链路整体照抄 torch 的 tools/metadata**：新增 `tools/generate_tensorplay_version.py` 与
  `tools/metadata/{_common,version}.py` provider（逐行移植，仅改环境变量名与标识符）：
  版本优先级 TENSORPLAY_BUILD_VERSION/BUILD_NUMBER（发布注入）→ PKG-INFO（sdist 构建）→
  `version.txt + git SHA`（本地开发构建，带 `+git<sha7>` 后缀），含 PEP 440 校验与 sdist
  一致性断言。CMake `generate_code` 目标新增生成步骤产出 `tensorplay/version.py`
  （`__version__`/`debug`/`cuda`/`git_version`，已 gitignore），`tensorplay/__init__.py`
  改为消费生成物而非硬编码版本。
- **CMake 对齐 torch 的 EnvVarForwarding**：新增 `cmake/EnvVarForwarding.cmake`（核心机制照抄，
  变量表裁到本项目开关）：`BUILD_*`/`USE_*`/`CMAKE_*` 环境变量经 Python 枚举后直通为同名
  CMake 缓存变量——`USE_CUDA=OFF pip install .`、`CMAKE_CUDA_ARCHITECTURES=61 pip install .`
  等 torch 风格写法直接生效，不再依赖 `SKBUILD_CMAKE_DEFINE`；include 置于 project() 后、
  USE_CUDA 自动探测前。
- **CI 重排为 torch 形态**：原单一 `release.yml` 拆为可复用
  `_binary-build.yml` / `_binary-test.yml` / `_binary-upload.yml`（对标 torch 的
  `_binary-build-linux.yml` / `_binary-test-linux.yml` / `_binary-upload.yml`），
  编排入口 `pull.yml`（PR）/ `trunk.yml`（main 与 release/* 推送）/ `release.yml`（v* tag：
  全矩阵构建 → 逐 wheel 冒烟 → PyPI trusted publishing）/ `lint.yml`（ruff，规则在
  pyproject.toml）。矩阵保持不变（ubuntu-22.04 cpu/cu121、ubuntu-24.04-arm cpu、windows-2022 cpu、
  py3.11），torch 风格的 `BUILD_ENVIRONMENT`/`SHA1`/`PR_NUMBER` 环境变量、concurrency 取消组、
  显式 `permissions` 一并引入；cu121 wheel 的冒烟测试在独立 test job 安装 CUDA toolkit 以满足
  扩展运行时依赖。

# 8. tensorplay.graph 门面：FX 对齐 + 特征提取 + 图可视化 + 编译器 pass 体系（2026-08-22）

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

## P0：L2 pass 体系落地(compiler/passes.py)

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

## P1：L1-D1 控制流静态特化(元数据具体化)

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

## P2：L3 守卫版符号形状(dynamic 缓存安全化)

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

## L4：AOT 双 partitioner

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

# 9. RNG 与 Torch 逐位对齐（2026-08-21）

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

# 10. CUDA 算子优化批次（2026-08-21）

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
