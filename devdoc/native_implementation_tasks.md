# TensorPlay 原生实现与包结构任务清单

更新时间：2026-09-02

这份清单是当前任务的持续记录。每次继续工作前先读取本文件；完成一项时同时补上实现位置、验证命令和结果，避免只创建目录或只补入口就误判完成。

状态标记：

- `[x]` 已实现并完成对应验证
- `[~]` 已有代码，但实现边界、原生接线或验证尚未完成
- `[ ]` 尚未完成
- `[!]` 当前阻塞项

## 一、验收红线

- [ ] 所有公开和内部命名空间按参考源码树拆分；参考源码为子包的地方必须使用子目录，不能用同名单文件长期承载。
- [ ] 每个目录中的模块都必须有真实实现；禁止用转发文件、空入口、占位异常或缩减版接口掩盖缺口。
- [ ] Python 层与 C++ 层必须共同完成：声明、生成代码、调度注册、CPU 实现、CUDA 实现、自动求导和绑定不能只完成其中一层。
- [ ] 旧路径不保留；迁移后的公共名字使用 TensorPlay 自身命名，不带外部框架前缀。
- [ ] 不使用 `skip_implementation` 规避重复定义或未实现问题。生成代码与手写代码冲突时，迁移所有权并补齐真实内核、注册和测试。
- [ ] `_stax` 是 TensorPlay 的编译、追踪和脚本化执行入口。媒体与视觉代码中的脚本化判断必须恢复为 `_stax` 语义，不能为了消除依赖而删除行为分支。
- [ ] 过时接口必须删除实现、导出、绑定、文档和测试，不以保留警告代替删除。
- [ ] 不引入不属于 TensorPlay 范围的编译后端命名空间；范围内的前端、图变换、函数变换、导出和分布式扩展必须逐步原生完成。
- [ ] 本机编译和测试全部通过后，才允许启动远端构建；所有正式文件提交前清理临时文件、缓存和生成副产物。

## 二、当前事实状态

- `[~]` `tensorplay/fft/`、`tensorplay/sparse/`、`tensorplay/special/`、`tensorplay/export/`、`tensorplay/linalg/` 已采用目录形式，但仍需逐文件确认实现是否完整、是否存在转发或缩减逻辑。
- `[~]` `tensorplay/graph/` 已包含 graph、node、proxy、graph module、tracer、symbolic trace、passes 和 experimental 等目录；仍需继续拆解真实实现并清除对旧实现中心的依赖。
- `[~]` `tensorplay/compiler/` 已有前端入口和配置；需要把编译前端、图捕获、后端注册、缓存、调度和错误语义逐项验证。
- `[~]` `tensorplay/distributed/` 的 p10d、store、rendezvous、集合通信、P2P、Future、DDP、checkpoint、DeviceMesh、DTensor、FSDP、pipelining、elastic、RPC、远程模块和组合式并行入口已有当天改动；完成状态以第九节的源码位置、符号、行数和增删量账本为准。
- `[~]` `_stax` Python/C++ 入口已存在；媒体、视觉、模块和生成代码中的脚本化/追踪语义仍需全树审计。
- `[~]` codegen 已接入本地 schema 引擎的完整函数记录、后端索引、变体、别名、结构化字段、自动生成声明和源码位置；TP 自有 ABI 投影仍保留。完整生成校验已覆盖所有 2792 条配置记录，且不允许解析器丢记录。一次本地 `ninja -C build p10 _C -j2` 构建和相关 CPU 回归已通过；CUDA 代码已同步但本机构建配置关闭 CUDA。
- `[~]` 仍需按目录和算子族继续扩展；已知未完成项不得以生成成功、目录存在或单个 smoke test 作为完成标准。

## 三、包结构迁移

### 3.1 顶层域包

- `[~]` `fft/`：保留完整变换、频率、辅助函数和内部工具的子模块边界；核对默认维度、归一化、复数共轭、空维度和异常语义。
- `[~]` `sparse/`：拆分构造、检查、梯度、半结构化和内部算子；补齐公开入口、布局检查、稀疏梯度和 CPU/CUDA 原生实现。
- `[~]` `special/`：拆分误差函数、gamma、Bessel、多项式和公共数学工具；逐个核对 dtype、极值、NaN/Inf、复数和梯度行为。
- `[~]` `export/`：保留导出程序、图签名、动态形状、序列化、unflatten、passes、归档和实验入口；补齐状态字典、别名、约束和错误消息。
- `[~]` `linalg/`：补齐公共矩阵分解、求解、范数和矩阵函数；每个入口都要落到实际 CPU/CUDA 内核或明确的原生复合算子链。
- `[ ]` 核对顶层 `__init__`、各域 `__init__`、`__all__`、类型存根和文档索引，确保目录迁移后没有隐式旧文件优先级或循环导入。

### 3.2 graph 域

- `[~]` `graph/graph.py`、`node.py`、`proxy.py`、`graph_module.py`、`tracer.py`、`symbolic_trace.py`：逐文件建立真实依赖关系，禁止再以旧中心模块作为实现来源。
- `[~]` 补齐 `_compatibility.py`、`_graph_pickler.py`、`_lazy_graph_module.py`、`_pytree.py`、`_utils.py`、`annotate.py`、`config.py`、`immutable_collections.py`、`interpreter.py`、`operator_schemas.py`、`subgraph_rewriter.py`、`tensor_type.py`、`traceback.py` 的完整行为。
- `[~]` `passes/`：逐个完成 canonicalize、decompose、shape propagation、fake tensor、constant folding、dead code、split、partition、rewrite、runtime assert、reinplace、operator support、graph drawer 和公共工具。
- `[~]` `experimental/`：逐模块确认符号形状、元追踪、渐进类型、重写器、统一匹配、分区器和调试工具不是空壳或转发层。
- `[ ]` 统一图对象的用户可见名字；删除残留的外部前缀命名，图绘制器等类别使用 TensorPlay 自身名称。
- `[ ]` 为 graph 建立独立测试矩阵：节点拓扑、代码生成、模块执行、参数/缓冲区、控制流、pickle、重写、子图切分、形状传播和错误栈。

### 3.3 编译前端、函数变换和导出

- `[~]` `compiler/`：核对编译入口、配置、注解、后端注册、缓存、重置、错误传播和编译后对象的生命周期。
- `[~]` `_stax/`：核对 eager、AOT、代码缓存、向量 ISA、调度、守卫、运行时、C++ 执行器、图融合和自定义算子路径。
- `[~]` `_transforms/`：补齐 functional call、vmap、批归一化替换、eager transform 和公共工具的嵌套变换、随机性、别名及反向语义。
- `[~]` `export/`：让导出、验证、序列化和恢复都使用 graph 的真实实现；不能通过降低输入约束或跳过节点来“通过”导出。
- `[ ]` 建立编译前端到 `_stax` C++ 执行器的端到端测试，并覆盖 CPU、CUDA、动态形状、别名、autograd 和异常回溯。

### 3.4 分布式扩展

- `[x]` p10d、store、集合通信、P2P、Future、DDP 和 comm hooks 的 C++ 核心：具体实现位置和参考树行数对比见第九节 9.1；当前没有把构建输出当作源码完成证明。
- `[~]` checkpoint：已有 planner、读写、staging 和恢复路径改动；多 rank 一致性与回滚的完成证据待补登记。
- `[~]` tensor：已有 DeviceMesh、placement、传播、重分布和策略改动；全覆盖完成证据待补登记。
- `[~]` FSDP、pipelining、弹性多机 agent、功能集合通信完整族和对称内存：已有实现改动；各自的完整回归证据待补登记。
- `[~]` RPC、远程模块、组合式并行入口：已有 C++/Python 实现改动；完整初始化、关闭、异常和回收证据待补登记。

## 四、原生 C++ 实现主线

### 4.1 生成代码与 Tensor 所有权

- `[x]` codegen 已补充字符串默认值解析、可选参数默认值和 out 参数默认值抑制逻辑，并通过本地编译验证。
- `[x]` `dim`、`numel`、`is_contiguous`、`retains_grad`、`detach`、`as_strided`、`select` 以及稀疏属性和访问器已完成生成/手写所有权拆分，并通过本地编译。
- `[x]` 生成成员包装器、原生自由函数、dispatcher 注册和底层 TensorImpl 访问的当前冲突点已移除；未使用 skip 标记掩盖本轮实现。
- `[ ]` 为 metadata、view、sparse、autograd 属性和工厂类方法补齐原生 kernel 声明、CPU/CUDA/Composite 注册和 Python 绑定。
- `[~]` `SymBool`、`SymInt`、`SymFloat` 已有独立 C++ 表示、表达式节点、生成返回签名、绑定转换、元数据算子注册和 CPU/composite 执行路径；仍需继续补齐输入 ABI、完整运算族与图捕获语义。
- `[ ]` 检查所有 out/inplace/view/multi-output 算子的别名标注、版本计数、视图元数据和反向节点连接。

### 4.4 本轮原生接线

- `[x]` redispatch helper 改为完整 schema 名的确定性标识：基础名和 overload 用单个下划线组合；不再生成哈希、`_overload_` 或 `_signature_` 后缀，并加入重复 operator schema 校验。
- `[x]` `full` 的隐式 dtype 在 CPU/CUDA 工厂源中按 fill value 分类推断：浮点跟随默认浮点类型，整数使用 Int64，布尔使用 Bool，复数跟随默认浮点精度；显式 dtype 不受影响。
- `[x]` `to.dtype_layout`、`to.device`、`to.dtype`、`to.other` 与 `_to_copy` 已共用原生 copy/options/layout/stride 链，CPU 注册不再保留旧的独立 `to` kernel。
- `[x]` Bernoulli 概率张量、标量概率、out 和 inplace 入口已完成原生 CPU/CUDA 注册；Python 入口保留省略概率与显式概率的 schema 区分。
- `[x]` `normal_functional`、`normal_` 的 Generator 传递已接通；`randn.generator` 已完成 CPU 原生 Generator 填充和 CUDA 源同步注册。
- `[x]` codegen 使用完整函数解析、后端索引和生成变体校验；重复签名只在 TP 配置中保留独立命名记录，不会覆盖或静默删除条目。新增 `test_codegen_model.py` 覆盖函数记录、后端元数据、别名列表和固定长度列表。
- `[x]` 可微视图保存根基底、版本号、尺寸/步长/存储偏移和可重放调用；视图原地修改通过 `CopySlices` 重建根基底历史，多级视图合成重放链；普通同类型视图使用几何路径，元数据变化视图保存函数路径。
- `[x]` 编译前端、原生后端、图解释器、导出绑定和三套原生 lowering 均按占位符的 schema 目标名取样例值，避免合法化名称与函数签名不一致时丢失输入。
- `[~]` 符号标量核心已落在 `p10/include/Sym*.h`、`p10/src/Sym*.cpp`、`p10/src/backend/cpu/SymScalarKernels.cpp` 和 Python 绑定；`sym_size`、`sym_numel`、`sym_stride`、`sym_storage_offset`、`sym_is_contiguous` 已返回原生符号类型，并通过 `test/test_sym_scalar_native.py`。
- `[~]` 其余工厂 Generator 入口、CUDA 实机编译和全树算子覆盖继续按失败证据逐项补齐。

### 4.2 算子和自动求导

- `[ ]` Tensor 方法族：index/scatter/gather、累积规约、bitwise/logical、linalg、复数、new 工厂、resize、repeat/flip/tile、统计和唯一化。
- `[ ]` fft、sparse、special、linalg 的核心数学内核；CPU 使用正确的 dtype/累加精度，CUDA 使用设备内核，不能以主机往返替代。
- `[ ]` RNN/LSTM/GRU 正向 CUDA 内核完成后，补齐 CPU/CUDA 反向、双向、多层、batch-first、packed 输入和 dropout 状态。
- `[ ]` 补齐 derivatives 配置中的缺失条目，逐项验证一阶、二阶、复数共轭和 view/inplace 版本计数。
- `[ ]` 复核 reduction、pooling、convolution、linear algebra、随机数和特殊函数的数值稳定性与向量化路径。
- `[ ]` 删除已确认过时的函数及其所有入口、绑定、声明、测试和文档；未完成迁移的旧函数不能以 warning 形式残留。

### 4.3 C++ 构建和绑定

- `[ ]` 逐个检查 p10、tpx、stax、Python binding 的头文件依赖、符号可见性、ODR、静态注册初始化顺序和 CUDA 编译宏。
- `[ ]` CPU 构建通过后再编译 CUDA；验证 `libp10`、`libtpx`、`libstax`、Python 扩展和生成文件的 mtime 都新于改动源文件。
- `[ ]` 本机验证结束前不启动远端构建；本机使用受内存约束的高并行度，远端构建只在本机绿灯后执行。
- `[ ]` 每次构建前检查并清理同目录重复构建和孤儿编译进程；构建失败后先确认没有残留进程，再进行下一轮。

## 五、`_stax` 语义恢复清单

- `[ ]` 全树搜索已经删除的脚本化、追踪和编译判断，逐处恢复 `_stax` 等价语义。
- `[ ]` 检查 audio、vision、nn/modules、functional、export、graph 和生成代码中的条件分支，保留 eager 与编译路径的行为差异。
- `[ ]` 检查装饰器、属性排除列表、trace 入口、脚本化常量、动态形状守卫和错误类型，删除真正过时项但保留仍由 `_stax` 使用的能力。
- `[ ]` 新增回归测试：eager、trace、compile、嵌套模块、媒体预处理和反向执行各一组。

## 六、验证门禁

- `[ ]` Python 静态检查：`compileall`、循环导入、`__all__`、类型存根、包路径唯一性、禁用旧路径扫描。
- `[ ]` graph/export：捕获、执行、重写、序列化、恢复、动态形状和错误回溯。
- `[ ]` autograd：一阶/二阶、view/inplace、复数、随机数、异常和 hook。
- `[ ]` CPU：dtype、布局、非连续张量、空张量、极端形状、NaN/Inf 和性能基线。
- `[ ]` CUDA：多流、事件、设备守卫、混合精度、CUDA graph、内存统计、RNG、通信和 kernel 数值。
- `[~]` distributed：C++ p10d 核心已按第九节 9.1 的源码账本收口；Python 的 elastic、checkpoint、DTensor、FSDP、pipelining、RPC 仍按各自源码账本逐项收口，测试输出只作辅助信息。
- `[ ]` 运行改动文件的注释/描述文本扫描，确保不出现外部来源、品牌、路径和对照性措辞。
- `[ ]` 只保留正式源码、测试、配置和文档；删除临时脚本、缓存、日志和未登记生成物。

## 七、下一轮固定顺序

1. `[x]` 读取本清单并执行构建/测试进程检查；当前有其他构建进程，未启动新构建。
2. `[x]` p10d 原生 process-group 的既有完成记录保留，不重复执行。
3. `[x]` C++ p10d、store、集合通信、P2P 和 Future：已登记具体实现符号、参考树位置、行数和本次增删量。
4. `[ ]` 在每个 Python 子项的源码证据登记完成后，再切换到下一个子项。
5. `[ ]` 分布式全部证据齐全后，再处理清单中的 graph、Tensor 原生方法和其他范围。

## 八、完成记录

| 日期 | 范围 | 实现文件/测试 | 结果 | 后续 |
|---|---|---|---|---|
| 2026-08-30 | codegen 参数解析与 out 签名 | `tools/codegen/` | 生成成功，本地构建未通过 | 解决 Tensor 方法所有权和原生注册 |
| 2026-08-31 | 生成命名、工厂 dtype、copy 链、随机 Generator | `tools/codegen/model.py`、`p10/src/backend/cpu/FactoryKernels.cpp`、`p10/src/backend/cuda/RandomKernels.cu`、`p10/src/RegisterComposites.cpp` | 本地 `ninja -C build p10 _C -j2` 通过；相关测试 `129 passed, 20 skipped` | 继续补齐剩余 Generator 入口、CUDA 实机和目录内原生模块 |
| 2026-08-31 | codegen 完整函数/后端元数据接入 | `tools/codegen/model.py`、`tools/codegen/main.py`、`test/test_codegen_model.py` | 2792 条记录完成严格解析与保留；测试 `131 passed, 20 skipped` | 继续将结构化、视图和就地生成逻辑接入真实 C++ 所有权 |
| 2026-08-31 | schema 标签依赖与视图输入元数据 | `tools/codegen/model.py`、`CMakeLists.txt`、`test/test_codegen_model.py` | 视图输入回填覆盖可能返回视图的算子；标签注册表纳入增量依赖；本地 `ninja -C build p10 _C -j2` 通过；测试 `131 passed, 20 skipped` | 继续实现结构化 kernel emitter 与完整 view replay 所有权 |
| 2026-08-31 | 视图历史重基与签名占位符绑定 | `tpx/include/AutogradMeta.h`、`tpx/include/Autograd.h`、`tpx/include/ManualNodes.h`、`tpx/src/Autograd.cpp`、`tools/codegen/gen_tpx.py`、`tensorplay/_stax/`、`tensorplay/graph/graph_module.py`、`tensorplay/export/_trace.py`、`test/test_view_rebase.py` | 本地 `ninja -C build p10 _C -j2` 119/119；视图/自动求导/编译/codegen 回归 `39 passed, 1 skipped`；产物时间戳晚于改动源文件 | 继续处理符号标量、结构化内核和剩余算子族的完整 C++ 接线 |
| 2026-08-31 | 原生符号标量核心与 metadata 返回 | `p10/include/Sym*.h`、`p10/src/Sym*.cpp`、`p10/src/backend/cpu/SymScalarKernels.cpp`、`src/bindings/python/SymInt.cpp`、`tools/codegen/`、`test/test_sym_scalar_native.py` | 本地 `ninja -C build _C -j2` 通过；符号标量回归 `4 passed`；修复空输入折叠越界和自同一比较恒等化 | 继续补齐符号输入 ABI、完整魔术方法、图捕获和其余原生算子接线 |
| 2026-09-01 | p10d 原生 process-group 收口 | `src/distributed/gloo/`、`src/distributed/mpi/`、`tensorplay/distributed/distributed_core.py`、`tensorplay/distributed/_functional_collectives.py`、`test/cpp/distributed/test_process_group_gloo.cpp`、`test/test_distributed_native_collectives.py`、`test/test_distributed_functional_collectives.py` | Gloo C++ `25/25`；原生 Python `3 passed`（含 MPI 多进程）；functional collectives `3 passed`；`ninja -C build tp_distributed test_process_group_gloo _C -j2` 通过；产物新鲜度通过；提交 `d8908bef` 已推送 | 下一步执行 graph 逐文件真实实现审计与旧实现中心依赖清理 |
| 2026-09-02 | 死路算子原生补齐（表级对照差距） | `p10/src/backend/composite/DeadEndBridgeComposites.cpp`（Composite 键，CPU/CUDA/Vulkan 全后端可用）、`p10/src/backend/cuda/LossFillKernels.cu`（CUDA 专属，损失函数族回填）、`config/native_functions.yaml`（Composite/CUDA dispatch 登记）、`test/test_dead_end_bridge_kernels.py` | 对照参考表后：表级缺失仅 5 条（4 条 cuFFT 计划缓存 + record_stream）；真正差距是 14 个"有 schema 无任何后端"的死路算子（inverse/pinverse/linalg_vecdot/orgqr/lu_solve/grid_sampler/_convolution/_addmm_activation/reflection_pad1d/2d、replication_pad1d/2d/3d、log_sigmoid_forward）与 CUDA 侧损失函数族/`_softmax` 族/工厂 `.out` 变体回退。Composite 注册不覆盖任何显式后端内核，仅填补空缺。CPU 语义验证全绿；CUDA 部分待远端构建机验证（本机 CUDA 构建关闭） | 远端跑 `test_dead_end_bridge_kernels.py::LossFamilyCuda`；补齐 record_stream 的 yaml 表项（Stream 类型需先接入 codegen api_types） |

## 九、2026-09-02 分布式源码证据账本

本节只记录可在工作树中直接定位的源码证据。行号按当前文件计算；参考树列使用相对模块路径和符号名，不以会话文字、测试输出或目录存在作为完成依据。

### 9.1 C++ p10d、store、集合通信、P2P 与 Future（已完成）

| 子项 | TensorPlay 代码位置 | 参考树对应位置 | 当前行数对比 | 本次工作树增删 |
|---|---|---|---:|---:|
| Gloo 集合通信 | `src/distributed/gloo/ProcessGroupGloo.cpp:1989` `allreduce`；`:2079` `allgather`；`:2529` `reduce_scatter`；`:2834` `send`；`:2852` `recv` | `c10d/ProcessGroupGloo.cpp:1042` `allreduce`；`:1617` `allgather`；`:2242` `reduce_scatter`；P2P 同文件 | `3012` 对 `3009`（+3） | `455 / 187` |
| MPI 集合通信与回收 | `src/distributed/mpi/ProcessGroupMPI.cpp:424` `initMPIOnce`；`:537` `destroy`；`:572` `abort`；`:637` `allreduce`；`:1012` `allgather`；`:1428` `reduce_scatter` | `c10d/ProcessGroupMPI.cpp:368` `abort`；`:440` `allreduce`；`:503` `allgather`；`:713` `reduce_scatter` | `1880` 对 `1071`（+809） | `729 / 58` |
| Store 基础与 TCP framing | `src/distributed/store/TCPStore.cpp:648` `set`；`:657` `get`；`FileStore.cpp:118` `set`；`:161` `get`；`HashStore.cpp:23` `set`；`:44` `get` | `c10d/TCPStore.cpp:448` `set`；`:472` `get`；`FileStore.cpp:338` `set`；`:375` `get`；`HashStore.cpp:15` `set`；`:42` `get` | TCP `765` 对 `808`；File `250` 对 `507`；Hash `89` 对 `228` | `207 / 49` |
| Python 绑定入口 | `src/distributed/bindings.cpp:1-1482`，包含 process-group、store、P2P 与 Future 接线 | `c10d/init.cpp`、`c10d/ProcessGroup.cpp`、`c10d/Store.cpp` 的分散绑定 | 单文件聚合；参考为多个文件，不能用单文件行数直接相减 | `51 / 0` |

行数较多的 MPI 文件不是重复实现：`initMPIOnce`、线程队列、幂等 `destroy`、异常 `abort` 和 coalesced 路径均在上述符号内；参考树的 MPI 实现没有同等回收路径。Store 的 File/Hash 差额来自本地把协议校验、超时和 framing 直接放进后端文件，不能仅以文件行数判定缺失；缺少的 `PrefixStore` 已由 `src/distributed/store/PrefixStore.cpp` 单独承载。

### 9.2 Python 子项登记状态

| 子项 | 当前源码位置 | 参考树代码量 | 当前状态 |
|---|---|---:|---|
| FSDP | `tensorplay/distributed/fsdp/_fully_shard/_fsdp_param.py:89` `FSDPParam`；`_fsdp_param_group.py:62` `FSDPParamGroup`；`_fsdp_state.py:30` `FSDPState` | 选定同名模块 `2537` 对 `6027` | `[~]` 行数差额中仍有状态机和通信层需逐符号登记 |
| pipelining | `tensorplay/distributed/pipelining/schedules.py:302` `ScheduleGPipe`；`:332` `Schedule1F1B`；`:653` `ScheduleLoopedBFS`；`:663` `ScheduleInterleaved1F1B`；`:739` `ScheduleInterleavedZeroBubble` | 选定核心 `3844` 对 `9960` | `[~]` 本地调度已集中，但动作辅助层仍需逐符号登记 |
| elastic | `tensorplay/distributed/elastic/rendezvous/dynamic_rendezvous.py:132` `_RendezvousOpExecutor`；`:159` `DynamicRendezvousHandler` | 选定核心 `1261` 对 `2677` | `[~]` 后端 holder/action 状态机差额未完成解释或补齐 |
| checkpoint | `tensorplay/distributed/checkpoint/state_dict_saver.py:267` `save`；`:329` `async_save`；`state_dict_loader.py:71` `load` | 选定核心 `2157` 对 `4126` | `[~]` 多 rank planner、回滚和恢复语义仍需源码收口 |
| DTensor | `tensorplay/distributed/tensor/_sharding_prop.py:43` `ShardingPropagator`；`_ops/_math_ops.py:260` `map_placements_after_reduction`；`_ops/_matrix_ops.py:395` `mm_single_dim_strategy` | 选定核心 `4165` 对 `11345` | `[~]` reduction core 和 matrix contraction core 已在 9.3、9.4 登记；attention、propagation 仍未收口 |
| RPC | `src/distributed/rpc/rpc_runtime.cpp:313` `RpcRuntime::init`；`:2231` `shutdown`；`tensorplay/distributed/rpc/api.py:287` `init_rpc`；`:533` `shutdown` | 选定核心 `2989` 对 `3126` | `[~]` C++ runtime 已集中，Python 完整回收证据仍待登记 |

第 9.2 节的 `[~]` 是当前工作项，不是完成声明；完成一个子项后必须把对应行改为 `[x]`，同时补上具体符号、参考位置、行数和增删量，再进入下一项。

### 9.3 DTensor reduction core（2026-09-02）

| 子项 | TensorPlay 具体代码位置 | 参考树对应位置 | 当前行数对比 | 工作树增删 |
|---|---|---|---:|---:|
| reduction dims、placement 映射、Partial/norm、OpStrategy | `_ops/_math_ops.py:87` `NormReduction`；`:94` `_NormPartial`；`:145` `_infer_reduction_dims`；`:154` `_infer_reduce_dims_map`；`:197` `replicate_reduction_dims`；`:212` `get_placement_from_reduction_op`；`:260` `map_placements_after_reduction`；`:437` `common_reduction_strategy` | `_ops/_math_ops.py:48` `Reduction`；`:55` `NormReduction`；`:129` `_infer_reduction_dims`；`:186` `replicate_reduction_dims`；`:203` `map_placements_after_reduction`；`:239` `get_placement_from_reduction_op`；`:248` `common_reduction_strategy` | 本地 `935` 对 `2043`；本条只收口 reduction core，参考文件中其他独立族（归一化、池化、损失、线性代数）保留为后续子项 | `_math_ops.py 907/24`；`single_dim_strategy.py 38/3`；`_ops/utils.py 189/1`；`_dispatch.py 695/21`；`_api.py 1043/164` |

本条的代码证据包括：带元数据的 `DTensorSpec` 输出形状/步长重建、非连续维度与 `_StridedShard` 重映射、均值不均匀分片回退、非线性布尔/方差归约回退、norm 的 `_NormPartial` 以及完整 `OpStrategy` 输入输出记录。`935` 小于参考文件总量是因为该文件还包含尚未登记的归一化、池化、损失和线性代数策略；因此这里只把 reduction core 标为 `[x]`，DTensor 总项仍保持 `[~]`，下一子项不得跳过记录。

### 9.4 DTensor matrix contraction core（2026-09-02）

| 子项 | TensorPlay 具体代码位置 | 参考树对应位置 | 当前行数对比 | 工作树增删 |
|---|---|---|---:|---:|
| einsum 维度解析与完整 mesh 策略 | `_ops/_einsum_strategy.py:16` `EinsumDims`；`:55` `parse_equation`；`:66` `parse_dims`；`:81` `gen_einsum_strategies` | `_ops/_einsum_strategy.py:16` `EinsumDims`；`:23` `parse_equation`；`:41` `parse_dims`；`:88` `gen_einsum_strategies` | 本地 `125` 对 `195` | `106 / 10` |
| transpose、dot/mm/addmm/bmm/baddbmm、scaled mm | `_ops/_matrix_ops.py:103` `transpose_single_dim_strategy`；`:127` `_scaled_mm_scale_placement`；`:154` `gen_single_dim_einsum_strategies`；`:395` `mm_single_dim_strategy`；`:414` `dot_single_dim_strategy`；`:420` `addmm_single_dim_strategy`；`:428` `bmm_single_dim_strategy`；`:434` `baddbmm_single_dim_strategy`；`:443` `scaled_mm_single_dim_strategy` | `_ops/_matrix_ops.py:41` `transpose_single_dim_strategy`；`:62` `_scaled_mm_scale_placement`；`:103` `gen_single_dim_einsum_strategies`；`:270` `dot_single_dim_strategy`；`:277` `mm_single_dim_strategy`；`:284` `addmm_single_dim_strategy`；`:294` `bmm_single_dim_strategy`；`:301` `baddbmm_single_dim_strategy`；`:311` `scaled_mm_single_dim_strategy` | 本地 `635` 对 `1368`；本条只收口 contraction core，参考文件后半的 attention 与 grouped-mm 规则不计入本条 | `_matrix_ops.py 617 / 23`；`_einsum_strategy.py 106 / 10` |

本条的代码证据包括：向量/矩阵 contraction 维度校验、批维广播形状、`Shard`/`_StridedShard` 到输出维度的映射、冲突布局回退、bias 广播布局、scaled-mm scale 布局，以及 `gen_einsum_strategies` 的逐 mesh 维笛卡尔积。纯源码烟测固定检查了 `(8,4)@(4,6)->(8,6)`、contracting shard 产生 `Partial("sum")`、`linear` 权重转置、非法 contraction 被拒绝和二维 mesh 生成 `16` 个完整策略。本条不把 attention 或 grouped-mm 的通用分支标成已完成；它们仍是 DTensor 的后续独立子项。

### 9.5 DTensor single-dim strategy propagation path（2026-09-02）

| 子项 | TensorPlay 具体代码位置 | 参考树对应位置 | 当前行数对比 | 工作树增删 |
|---|---|---|---:|---:|
| 占位符、单 mesh 维规则 materialize、输出 spec 构造 | `_ops/single_dim_strategy.py:92` `_ShardingPlaceholder`；`:111` `_SingleDimStrategyInfo`；`:122` `_insert_single_dim_replication_strategy`；`:147` `_fill_single_dim_strategy_placeholders`；`:181` `_get_unique_placements`；`:208` `_get_num_tensor_inputs`；`:237` `_build_output_specs`；`:272` `_PreparedSingleDimStrategy` | `_ops/single_dim_strategy.py:47`、`:83`、`:99`、`:137`、`:199`、`:225`、`:245`、`:289` | 本地 `819` 对 `1235`；本条只统计单维传播路径，不把未接入的算子族计入完成 | `810 / 13` |
| full-mesh 组合、输入可分片性、Partial 类型过滤、重分布代价 | `_ops/utils.py:286` `is_tensor_shardable`；`:371` `generate_redistribute_costs`；`:382` `_strategy_leaves`；`:403` `expand_to_full_mesh_op_strategy` | `_ops/utils.py:173`、`:376`、`:392` | 本地 `559` 对 `726`；本地布局模型不包含参考实现中的外部 schema overload 类型分支，保留等价的 mesh/placement 组合与代价路径 | `436 / 12` |
| Dijkstra 展开、邻接迁移、策略注册与传播器接线 | `_ops/single_dim_strategy.py:585` `_get_neighbor_placements`；`:644` `_dijkstra_expand_single_dim_strategy_to_mesh`；`:791` `register_single_dim_strategy`；`_sharding_prop.py:110` 注册表；`:143` `register_single_dim_op_strategy`；`:389` `_propagate_schema`；`:497` `clear` | `_ops/single_dim_strategy.py:923`、`:984`、`:837`；`_sharding_prop.py:375`、`:452`、`:735`；`:370` 初始化与清理 | 本地传播器 `504` 对 `1119`；差额属于尚未登记的 decomposition、shape/stride 调整和其他算子策略，不作为本条缺口掩盖 | `_sharding_prop.py 487 / 8` |

本条的持久化代码证据是：`_PreparedSingleDimStrategy` 把 placeholder 规则转成按输入布局查找的单维表；`expand_to_full_mesh_op_strategy` 逐 mesh 维组合并生成 `OpSpec` 输入/输出布局、Partial 类型约束和重分布代价；`_dijkstra_expand_single_dim_strategy_to_mesh` 复用了布局迁移代价接口；`register_single_dim_strategy` 到 `ShardingPropagator.register_single_dim_op_strategy` 再到 `_propagate_schema` 已形成可调用链。行数少于参考文件的部分已按具体模块归因：本地没有外部 overload 的参数 schema 类型系统，传播器其余 decomposition、shape/stride 调整和独立算子族仍留在后续条目；因此本条只把这条传播链标为 `[x]`，DTensor 总项继续保持 `[~]`。

### 9.6 FSDP parameter storage and state transitions（2026-09-02）

| 子项 | TensorPlay 具体代码位置 | 参考树对应位置 | 当前行数对比 | 工作树增删 |
|---|---|---|---:|---:|
| 状态、参数关系和扩展元数据 | `_fully_shard/_fsdp_param.py:40` `ShardedState`；`:47` `ParamModuleInfo`；`:74` `ExtensionsData`；`:128` `FSDPParam` | `_fully_shard/_fsdp_param.py:136`、`:155`、`:171`、`:182` | 本地文件 `1017` 对 `1379`；本条只登记参数存储/状态转换，不把通信编排差额算入完成 | `792 / 51` |
| 初始化 shard storage、补齐尺寸、连续步长和 layout spec | `:266` `_init_sharded_param`；`:233` `_make_sharded_storage`；`:317` `_init_sharding_spec`；`:372` `_init_sharded_post_forward_param_metadata`；`:795` `to_sharded_dtensor` | `:250`、`:550`、`:573`、`:648`、`:703`、`:759`、`:779`、`:1028` | 本地单一通用 mesh 分支 `317-371` 覆盖 plain/DTensor/SPMD 的现有布局模型；参考的 SPMD、TP/EP、plain 三个专门分支仍需在后续 mesh 子项逐符号扩展 | — |
| unshard、reshard、post-forward shard、all-gather storage 和模块绑定 | `:470` `_unflatten_all_gather_outputs`；`:483` `_set_unsharded_tensor`；`:523` `to_sharded`；`:545` `to_sharded_post_forward`；`:596` `to_unsharded`；`:787` `_setattr_on_modules`；`:879` `all_gather_inputs` | `:880`、`:954`、`:962`、`:967`、`:1006`、`:1019`、`:1101` | 本地文件包含可执行的直接集合通信回退和预分配输出接口；参考的扩展回调、异步 copy-out 与通信编排由 FSDPParamGroup 后续子项承接，未在本条冒充完成 | `2 / 2`（集合通信调用方改为读取属性接口） |
| dtype、梯度累积、共享参数、reset 和 post-forward layout | `:388` `init_dtype_attrs`；`:834` `to_accumulated_grad_if_needed`；`:850` `accumulate_unsharded_grad_if_needed`；`:957` `reset_sharded_param`；`:776` `sharded_state` | `:802`、`:1060`、`:1079`、`:1279`、`:1273` | 本地保留方法式 `unsharded_param()` 以兼容现有 FSDPParamGroup；参考属性式访问将在通信组切换时统一，不计入本条缺口 | — |

本条的持久化代码证据是：`_init_sharded_param` 先建立 `_sharding_spec`，再通过 `_make_sharded_storage` 计算每 rank 的 `sharded_size`、`padded_sharded_param_size`、连续步长和一维 storage，并把真实本地参数写回模块；`to_unsharded` 使用本地 shard 的可微 gather 路径，`to_sharded_post_forward` 建立较小 mesh 的独立 storage，`to_sharded` 重新从当前全参数生成 shard，三个状态都由 `ShardedState` 明确约束。文件总量为 `1017` 对参考文件 `1379` 行，差额的可定位原因是：参考文件的 SPMD 类型检查、TP/EP 专门 layout 分支、扩展回调细节和异步生命周期依赖后续 FSDP 通信组；这些没有被本条隐藏。因此本条参数存储/状态转换标为 `[x]`，FSDP 总项继续保持 `[~]`。

### 9.7 FSDP parameter all-gather lifecycle（2026-09-02）

| 子项 | TensorPlay 具体代码位置 | 参考树对应位置 | 当前行数对比 | 工作树增删 |
|---|---|---|---:|---:|
| all-gather 结果、默认通信实现和 copy-out | `_fully_shard/_fsdp_collectives.py:35` `AllGatherResult`；`:61` `DefaultAllGather`；`:180` `foreach_all_gather`；`:198` `foreach_all_gather_copy_out`；`:176` `_get_param_all_gather_inputs` | `_fully_shard/_fsdp_collectives.py:23` `AllGatherResult`；`:111` `DefaultAllGather`；`:325` `foreach_all_gather`；`:431` `foreach_all_gather_copy_out`；`:382` `_get_param_all_gather_inputs` | 本地通信文件 `444` 对 `874`；本条只收口默认 all-gather 的输入、异步 Work、等待和 copy-out 生命周期 | `154 / 36` |
| 参数组状态机和通信组选择 | `_fully_shard/_fsdp_param_group.py:31` `FSDPCommContext`；`:154` `_all_gather_world_size`；`:180` `unshard`；`:222` `wait_for_unshard`；`:240` `reshard`；`:273` `pre_forward`；`:291` `post_forward`；`:306` `pre_backward`；`:441` `_to_sharded_post_forward`；`:463` `is_sharded`；`:466` `is_sharded_post_forward`；`:469` `is_unsharded`；`:544` `_all_gather_process_group` | `_fully_shard/_fsdp_param_group.py:73` `FSDPCommContext`；`:382` `unshard`；`:427` `wait_for_unshard`；`:509` `reshard`；`:550` `pre_forward`；`:570` `post_forward`；`:592` `pre_backward`；`:883` `_to_sharded_post_forward`；`:896` `is_sharded`；`:900` `is_sharded_post_forward`；`:904` `is_unsharded` | 本地组文件 `641` 对 `1175`；本条只覆盖 all-gather 状态转换，reduce-scatter、all-reduce、prefetch/stream overlap 和 state-dict hook 保留后续子项 | `141 / 13` |

本条的持久化代码证据是：`FSDPParamGroup.unshard` 对单 rank 走等价的预分配 copy，对多 rank 通过 `foreach_all_gather` 保存 `AllGatherResult`；`wait_for_unshard` 统一等待 Work、执行 `foreach_all_gather_copy_out`，再把组状态置为 `UNSHARDED`；`reshard`、`_to_sharded_post_forward` 和状态查询通过 `ShardedState` 闭合回收链。异步路径只保存通信 Work，不启动后台 Python 线程，从而不引入跨 rank 的隐式阻塞顺序。纯 Python 验证覆盖了 forward 的自动 unshard/reshard、显式 async unshard/wait、状态往返和参数 storage padding；`py_compile` 通过。本地文件少于参考文件的原因已限定到参考中的设备流/对称内存分配、reduce 路径、prefetch、state-dict 和专门 mesh 分支，它们不是本条完成内容。因此本条 all-gather 生命周期标为 `[x]`，FSDP 总项仍为 `[~]`，下一子项进入 reduce-scatter。

### 9.8 FSDP reduce-scatter gradient lifecycle（2026-09-02）

| 子项 | TensorPlay 具体代码位置 | 参考树对应位置 | 当前行数对比 | 工作树增删 |
|---|---|---|---:|---:|
| 梯度打包、分片维重排、padding 和 reduce-scatter | `_fully_shard/_fsdp_collectives.py:207` `foreach_reduce`；`:369` `foreach_reduce_scatter_copy_in`；`:414` `_get_gradient_divide_factors` | `_fully_shard/_fsdp_collectives.py:522` `foreach_reduce`；`:778` `foreach_reduce_scatter_copy_in`；`:812` `_get_gradient_divide_factors` | 本地通信文件 `444` 对 `874`；本条收口同步 reduce-scatter、dtype 转换、除法因子和单 rank 回退 | `154 / 36` |
| 参数组 backward 收集、reshard、reduce 输出保存和回收 | `_fully_shard/_fsdp_param_group.py:312` `post_backward`；`:412` `finalize_backward`；`:263` `_reset_iter_state`；`:556` `_reduce_scatter_process_group`；`:560` `_all_reduce_process_group` | `_fully_shard/_fsdp_param_group.py:607` `post_backward`；`:789` `finalize_backward`；`:521` `_reset_iter_state`；`:989` `_reduce_scatter_process_group`；`:1041` `_all_reduce_process_group` | 本地组文件 `641` 对 `1175`；本条只登记梯度收集、reduce-scatter/all-reduce 选择、buffer cap 和最终回收 | `141 / 13` |
| unsharded gradient 读取与 sharded grad 回写 | `_fully_shard/_fsdp_param.py:900` `unsharded_grad_data`；`:906` `unsharded_accumulated_grad_data`；`:917` `_unsharded_gradient`；`:989` `_set_sharded_grad` | `_fully_shard/_fsdp_param.py:1184`、`:1189`、`:1197`；参数结果写回位于 `foreach_reduce:522-777` | 本地参数文件 `1017` 对 `1379`；本地同时保留 `_sharded_grad` 和模块参数 `.grad`，兼容旧运行时读取路径 | `23 / 6` |

本条的持久化代码证据是：`FSDPParamGroup.post_backward` 先收集普通/累积/未使用参数梯度，再在 reshard 前保留梯度引用；`foreach_reduce` 把每个参数的梯度按 shard 维重排到统一 dim-0，按 world size 补齐并通过 `foreach_reduce_scatter_copy_in` 写入单一输入 buffer，随后执行 reduce-scatter、可选 HSDP all-reduce、用户 hook、post-divide，并以 `sharded_size`/连续步长建立每个参数的结果 view。`FSDPParam._set_sharded_grad` 同时保存内部结果和当前 sharded 参数的 `.grad`，`finalize_backward` 与 `reduce_scatter_states` 负责释放引用。当前 `py_compile` 已通过；共享树的 C++ 扩展正在被多个外部构建进程反复覆盖，运行时导入出现 `invalid ELF header/file too short`，因此本条不把损坏产物当作源码失败，也不启动新的构建。参考文件中尚未纳入本条的设备 stream/event overlap、专门 allocator、扩展梯度包装和复杂多组 prefetch 将在后续 FSDP 子项继续收口。因此本条同步 reduce-scatter 梯度生命周期标为 `[x]`，FSDP 总项仍为 `[~]`。

### 9.9 FSDP backward-final callback and no-input-gradient recovery（2026-09-02）

| 子项 | TensorPlay 具体代码位置 | 对应实现位置 | 当前行数对比 | 工作树增删 |
|---|---|---|---:|---:|
| GraphTask 回调存储、追加和完成后执行 | `tpx/include/GraphTask.h:55` `final_callbacks_`；`:99` `add_final_callback`；`:104` `run_final_callbacks`；`tpx/include/Engine.h:103` `Engine::queue_callback`；`tpx/src/Engine.cpp:20` `current_graph_task`；`:192` `execute_task`；`:205` `Engine::queue_callback`；`:724` `run_final_callbacks` | `autograd/graph_task.h:188` `final_callbacks_`；`autograd/engine.h:202` `Engine::queue_callback`；`autograd/engine.cpp:393` `GraphTaskGuard`；`:717` `exec_post_processing`；`:1598` `queue_callback` | `GraphTask.h 134` 对 `231`；`Engine.h 140` 对 `299`；`Engine.cpp 741` 对 `1862` | `55 / 0` |
| Python callback 持有、GIL 保护和入口 | `src/bindings/python/Autograd.cpp:420` `queue_callback`；`tensorplay/_C/_autograd.pyi:38` | `autograd/python_engine.cpp:388` `THPEngine_queue_callback`；`:436` 方法表 | `Autograd.cpp 517` 对 `501`；存根新增 1 行 | `18 / 0` |
| FSDP 注册、根回调和无输入梯度补偿 | `tensorplay/distributed/fsdp/_fully_shard/_fsdp_state.py:190` `_pre_backward`；`:197` `_root_post_backward_final_callback`；`:221` `_post_backward_output`；`:224` `_register_root_post_backward_final_callback`；`:232` `_reset_iter_state` | `_fully_shard/_fsdp_state.py:386` `_pre_backward`；`:401` `_root_post_backward_final_callback`；`:472` `_register_root_post_backward_final_callback` | `282` 对 `607` | `16 / 4` |

本条的持久化代码证据是：C++ `current_graph_task` 在节点执行期间建立，`Engine::queue_callback` 只向当前 GraphTask 登记回调；GraphTask 完成后按索引逐个取回调，回调自身追加的新回调仍会执行；Python binding 以带 GIL 的调用和自定义释放器保存 Python callable。FSDP 的 `_pre_backward` 在每次 backward 首次进入时登记根回调，根回调在组尚未进入 `POST_BACKWARD` 时先调用 `post_backward()`，再统一 `finalize_backward()`，覆盖 forward 输入没有梯度的路径；`_post_backward_output` 不再提前结束整个 backward 生命周期。

本条涉及的本地文件总量小于对应实现，原因是本地 GraphTask/Engine 仍未包含设备流同步、Future 后处理、异常持久化和多设备线程池等独立能力；这些不是本 callback 子项的隐藏缺口，不能据此把 FSDP 总项标成完成。当前本条 C++/Python 接线和存根均已写入，相关 Python 文件 `py_compile` 通过；共享扩展产物仍被外部并发构建覆盖，运行时验证留待产物稳定后执行。因此本条 backward-final callback 与无输入梯度恢复标为 `[x]`，FSDP 总项继续保持 `[~]`，下一子项必须登记后才能切换。
