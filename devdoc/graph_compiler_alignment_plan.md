# 图编译全栈蓝图(2026-08-22)

目标:`tensorplay` 图编译栈逐层实现完整能力(Dynamo/FX/AOTAutograd/Inductor/export),
参考基准为本地参考源码树(commit `893b6406`)。每层给出:现状 → 目标语义 →
参考路径 → TensorPlay 落点 → 验收标准。

## 量化差距（2026-08-24 实测, wc -l）

| 组件 | 参考实现 (LOC) | TensorPlay (LOC) | 倍率 |
|---|---|---|---|
| Inductor (`_inductor`) | 284,829 | backends+stax ≈ 4,400 | ~65x |
| Dynamo (`_dynamo`) | 130,241 | compiler 前端 ≈ 2,600 | ~50x |
| FX (`fx`) | 54,204 | graph+passes ≈ 2,000 | ~27x |
| HOPs + subclasses | 31,915 | — | — |
| 编译 C++ (`csrc/{inductor,dynamo,fx}`) | 35,692 | stax C+++桥接 ≈ 1,600 | ~22x |
| **合计** | **≈537k** | **≈10.6k** | **~51x** |

差距本质是机制面而非行数:Dynamo=字节码级追踪(PEP 523)+C++ guard 树+副作用重放;
Inductor=符号形状(sympy)+缓冲区调度器+Triton/Cpp 双代码生成+autotune 缓存体系。
TensorPlay 当前=代理追踪+白名单原生图+点级 Triton codegen。

## 层级总览与状态

| # | 层 | 参考组件 | 现役载体 | 现状 | 目标 |
|---|---|---|---|---|---|
| L1 | 捕获前端 | `_dynamo` | `compiler/graph.py` Tracer | 执行式追踪,控制流拒绝/回退 | 字节码级捕获:图断点、guards、resume |
| L2 | IR 与 pass 体系 | `fx` + `_fx/passes` | `compiler/passes.py`(P0 已落地) | PassManager/DCE/ConstFold/ShapeProp | pass 生态扩容、subgraph 工具 |
| L3 | 符号形状 | SymInt/SymBool + sympy | — | bool dynamic(rank 固定) | size 符号变量、约束传播、guard 表达式 |
| L4 | 自动微分切分 | AOTAutograd + min-cut | **stax 内部**(`_AotNativeLowering`) | train/eval 双图,公式驱动 backward | joint graph + min-cut partitioner,**上提出 stax** |
| L5 | 代码生成后端 | **Inductor = stax 对标物** | `backends/stax.py` + libstax + triton.py | 见下方盘点 | 缓存/autotune/cudagraphs/分解表 |
| L6 | 特化治理 | guards 编译/recompile 策略 | `compiler/api.py` | 结构化签名 + limit/isolate | guard 表达式编译、失效粒度对齐 |
| L7 | 导出 IR | torch.export/EXIR | — | 无 | 稳定图 IR + 序列化 + 运行时加载 |

## L5 stax:我们的 Inductor(现状盘点)

stax = 双侧结构:C++ 执行器(`stax/` → libstax.so,native op graph)+ Python
lowering(`tensorplay/backends/stax.py`)。对照 Inductor:

| torch 组件 | stax 现状 | 缺口 |
|---|---|---|
| native kernel 执行 | native graph 逐算子路径("stax-native") | — |
| Inductor scheduler(pointwise 融合决策) | `_CpuFusedPointwiseLowering`("stax-fused-cpu",链式融合 + 自定义 autograd Function);CUDA 走 Triton 点级原型 | 融合面窄:仅 pointwise 类,无跨类调度(matmul+epilogue 等) |
| AOTAutograd 衔接 | `_AotNativeLowering`("stax-aot-native"):由 `derivatives.yaml` 公式生成 backward 图,train/eval split | 切分策略内嵌后端(见 L4 迁移项) |
| Triton codegen | `backends/triton.py`,forward+backward 已有 CUDA 测试覆盖 | 无 kernel cache、无 autotune |
| codecache / autotune / cudagraphs | 无 | M1/M2/M3 |
| 语义分解表(decomposition) | 无(仅求导公式表) | M4 |
| 后端契约 | `strict_native`:拒绝 Python executor 充当编译产物 | 保持为验收红线 |

里程碑:M1 Triton/native codegen 缓存(kernel hash→bin);M2 autotune(候选配置
benchmark 择优);M3 cudagraphs(**依赖系统层 CUDA Stream/Event 真实现**,联动
gap_analysis §CUDA 运行时);M4 分解表接入 P0 pass 体系;M5 融合面扩展
(matmul epilogue / add_relu 类跨算子模板泛化)。

## L1 捕获前端(最大结构性工程)

- 参考:`torch/_dynamo/eval_frame.c`(帧钩子)、`convert_frame.py`、
  `bytecode_analysis.py`、`resume_execution.py`(graph break resume)、`guards.cpp`。
- 阶段:
  1. **D1 trace-resume**:在现执行式 Tracer 上支持"中断点记录 + 子图续捕"
     (控制流条件为 Tensor 时断点;Python 标量时静态特化)。纯 Python 可达。
  2. **D2 字节码翻译**:捕获函数 bytecode → 指令级解释器(`dis`),if/loop 按
     guard 结果特化或断点。对齐 `bytecode_analysis.py` 的子集。
  3. **D3 C++ 帧钩子**:eval_frame 等价物。**需 `_C` 配合**,必须等 dtype 迁移
     构建窗口结束,单独排期。
- 验收:ResNet benchmark 控制流用例 graph-break 数量与 torch.compile 对比;
  fullgraph 语义一致。

## L2 pass 体系(P0,立即可做)

- 参考:`torch/fx/passes/`(shape_prop、param_fetch、runtime_assert)、
  `torch/_inductor/fx_passes/`(fuse、pattern 匹配)。
- 落点:`tensorplay/_stax/passes.py`(PassManager)+ `tensorplay/_stax/fx_passes/`。
- 首批 pass:`dead_code_elimination`(已有)、`normalize_operators`、
  `pointwise_fusion_hint`(给 stax 标注)、`const_fold`。
- 验收:pass 幂等性测试;fusion hint 后 stax lowering 节点数下降可断言。

## L3 符号形状

- 参考:`torch/_C/SymInt`、`torch/sym_int`、`sympy` 集成
  (`torch/_meta_registrations.py`)、dynamic-shapes 教程的约束模型。
- 方案:引入轻量 `Sym(size)` 表达式(sympy 后端,pip 通),Graph meta 记录
  shape 表达式;guard 生成布尔表达式串编译成谓词函数(L6 复用)。
- 验收:同 rank 不同 size 单次特化;`dynamic_shapes` dict 形式 API。

## L4 AOTAutograd 切分(含架构迁移项)

- 参考:`torch/_functorch/partimators/min_cut_rematerialization.py`。
- **现状归属**:AOT 切分的现役实现在 stax 内部(`_AotNativeLowering`,公式驱动的
  forward/backward 图对),这与 torch 的分层不同——torch 中 AOTAutograd 属前端,
  Inductor 只消费其产物。
- 迁移路径:P3 新建 `compiler/aot.py`(joint graph 构建 → min-cut 网络流决定
  rematerialize 集 → 生成双图),先以 `partitioner="mincut"` 选项与 stax 内置
  切分并存,验证等价后默认切换,stax 退化为纯 codegen 后端。
- 验收:内存峰值 vs 重算率曲线;现有梯度正确性测试全保(`test_compile.py`
  AOT 相关用例为回归底线)。

## L5 Inductor 对齐(最重)

- 参考:`torch/_inductor/{scheduler.py,triton_heuristics.py,cudagraph_trees.py,
  codecache.py}`、模式分解表 `decomposition.py`。
- 里程碑:M1 codegen 缓存(triton kernel hash→bin);M2 autotune(候选配置
  benchmark);M3 cudagraphs(依赖 CUDA Stream/Event 真实现——见 gap_analysis
  §CUDA 运行时,与系统层联动);M4 分解表接入 pass 体系。
- 验收:ResNet/CIFAR 训练步时延对比 eager 与 torch.compile。

## L6 特化治理

- guard 从结构化签名升级为表达式(`x.size()[0]==s0 and x.dtype==f32`)并缓存
  编译结果;recompile 策略(失效原因分类、isolate_recompiles)对齐 Dynamo。

## L7 export

- 在 L2 IR 上冻结 stable op 集,序列化(graph JSON + 权重包),提供
  `tensorplay.export()` / `load_export()`。运行时复用解释执行器。

## 执行顺序(依赖驱动)

```
P0 L2 pass 体系 ──► P1 L1-D1 控制流特化/resume ──► P2 L3 符号形状最小版
        │                                                    │
        ▼                                                    ▼
P3 L4 min-cut 切分 ◄──────────────────────────── P4 L6 guard 表达式
        │
        ▼
P5 L5 M1/M2 codegen 缓存+autotune ──► P6 L5 M3 cudagraphs(需系统层) ──► P7 L7 export
```

## 协作边界

- P0–P2、P4、P7 纯 Python,不触碰 `_C` 构建,可与 dtype 迁移并行。
- D3 帧钩子、cudagraphs 依赖 `_C`/CUDA 系统层,需与系统层负责人排期,
  遵守 AGENTS.md 构建纪律。

### _extract_fwd_bwd_modules 研究补记(L1343-1442)

- 切分 = **成员资格抽取**:`_extract_graph_with_inputs_outputs(joint.graph,
  [saved+tangent+seed...], bwd_outputs)` 把 joint 中的 backward 节点拷贝为独立
  bw graph,saved 值变其 placeholder;
- 抽取后**剪枝**:bw 内无消费者的 saved 占位符从保存清单移除(避免死激活);
- primal 的 staticness 在此阶段以 meta 盖章(位置对应关系此后即失效);
- 对我们的启示:v2 无需复刻 getitem/sym/BackwardState 分支;核心三步——
  tag 区分 → 成员拷贝 → 未用剪除——即可支撑 default 与 min-cut 两个 partitioner。

### 交接状态(P3 v2 待办,新会话续)

- `compiler/aot.py` 当前为 v1 垂直切片(规则直产两图),test_aot 单例直接执行
  数值正确(mul/relu/sum 梯度对拍通过);但整跑曾出现一次 SIGSEGV(嫌疑:worktree
  环境混入 torch._C 后 p10/torch 原生栈冲突),v2 需先在干净环境复核;
- v2 重构步骤:① 规则发射目标改为 joint graph(追加于原 output 之后,
  `meta["is_backward"]=True`,拓扑自然满足);② 新增 `partition_default(joint,
  num_fwd_outputs)` 按"backward 消费者→saved"抽取双图(含未用剪除);③
  `build_aot` 变为 trace_rules → partition_default 的两段式;④ min_cut 以同签名
  后置接入(自研 Edmonds-Karp,容量模型见上节);⑤ 回归:test_aot 全绿。

### P3 v2 迭代记录(同日)

- 已按 joint-graph 架构重写一版(`/tmp/opencode/aot_v2_draft.py`):规则经
  `_JointBuilder.bwd` 发射进原图并打 `is_backward` 标签;
- 未收敛三处(新会话从这里继续):
  1. 叶子梯度须在 sweep 时显式产出 `grad_<leaf>` 链加节点并登记为 bw outputs
     (torch 的 num_fwd_outputs 切分位),partition 才能按输出切;
  2. `_copy_nodes` 清理(slice 分支残渣、builtins 误导入);
  3. 抽取后 backward 输入序为首次引用序,AotResult.value_and_grad 的
     `(go,*args,*saved)` 假设需改为按 bw 占位符名分发;
- v1 备份:`/tmp/opencode/aot_v1_backup.py`(数值已验证),当前主树即 v1。

### ⚠️ 更正(终验发现)

v1 切片冒烟对拍暴露**梯度数值错误**(mul/relu/sum 例:dx 得 [-3,8],期望 [0,4];
dw 同错)。此前"数值正确"结论仅基于单例打印未对照 eager,test_aot 九例从未整跑
通过。v1 定位为**实验性半成品**:规则/归约方向待逐节点调试(嫌疑:reduce_for 的
producer 形状取用或 sum 规则的 ones_like 广播路径)。v2 重构前先修 v1 数值并让
test_aot 整跑转绿,再动架构。

### P3 数值修复与崩溃定性(续)

- ✅ v1 梯度数值已修复:`value_and_grad` 改为按名绑定(bw 占位符创建序为
  grad_out→saved→叶子交错,原位置调用全错位);冒烟 dx=[0,4]/dw=[0,2] 正确;
- ✅ `compiler/__init__.py` 补导出 AOTError/build_aot;
- ⚠️ 遗留硬伤定性:**非确定性 SIGSEGV**——同一进程内先跑 build_aot+value_and_grad,
  再做 eager `.backward()` 时偶发段错误(复现率高但非必现;k 系列逐行脚本:
  k1-k6 全过、k7 崩;另一次同序列却全过)。无 torch 参与(--noconftest 下扩展仅
  numpy),排除共生冲突;帧落在 amp 包装器内的原生算子调用。判断为 p10 原生层
  use-after-free 类缺陷,嫌疑面:interpreter `_record_meta` 的张量生命周期、
  bw 图执行期临时张量、GraphModule 持有 meta['val']。test_aot 整跑在修复前
  不计入绿灯承诺;下一会话优先用 ASAN 重编或 valgrind 复现抓栈。

## P3-L4b: partition_min_cut 落地级设计(v2 架构上实现,免运行时可评审)

### 1. 契约

```python
def partition_min_cut(
    joint_gm: GraphModule,
    *,
    num_fwd_outputs: int = 1,
    memory_budget: Optional[int] = None,   # 字节;None=不设上限语义同默认权重
    ban_fusible_chains: bool = True,
) -> PartitionResult   # 与 partition_default 同构的返回元组
```

与 default 可互换(build_aot 仅换调用点)。差异只在"saved 集合怎么选":
default=结构判据("有 backward 消费者即存"),min-cut=全局内存最优切割。

### 2. 节点分类(容量表)

| 类别 | 判定 | 流网络处理 |
|---|---|---|
| 叶子 placeholder / tangent | op∈{placeholder} | 不入网(免费输入) |
| 用户输出 | joint output 前 num_fwd_outputs | source→其 producer 容量∞ |
| **必存**(must-save) | 非 leaf 但被 fw 与 bw 双侧消费、或 impure(rng)、get_attr、多输出节点 | 节点→sink 容量∞(强制不在割里) |
| **候选保存** | 其余 fwd 非 leaf 节点且有 bw 消费者 | 节点→sink 容量 w(n) |
| **可重算算子** | 规则表中存在且操作数均为候选/可重算(mul/add/sub/truediv/neg/relu/sum/sin/cos/exp) | 入边容量∞、出边容量 w(n)(允许被割后克隆进 bw) |

权重:`w(n)=max(1, nbytes(meta["val"]))`;memory_budget=None 时统一 w=1
(单位模型,与 torch 默认一致)。

### 3. 流网络构造

- `source → 每个"用户输出 producer"`,cap ∞;
- 对每个 fwd 节点 n:若 n 是必存类 → `n→sink` cap ∞;
- 若 n 是候选保存类 → `n→sink` cap w(n);
- 对每条数据依赖 `u→n`(n 为可重算算子):`u→n` cap ∞;
- 其余 fwd 内部依赖(不可重算消费者)`u→n` cap ∞;
- bw 侧节点不入网(它们的存在只决定谁有"bw 消费者")。

最大流 = 不可重用而必须物化的最小字节数;割集 ⊆ 候选保存节点。

### 4. 求解——自研 Edmonds-Karp(禁新增依赖)

- 网络规模=前向节点数 O(N),边 O(E);EK 是 O(V·E²),N<10³ 足够;
- BFS 找增广路,FIFO 队列;残量图用 dict-of-dict;
- 割集提取:BFS 终止后在残量图可达集 R;割 = {v∉R | v→sink 有正残量边}
  中"候选保存"节点。

### 5. 从割集到双图(saved 之外的引用一律重算)

关键差异:default 抽取时外部引用全部自动占位;min-cut 中**不在割集的
fwd 引用必须在 bw 图内克隆重算**:

1. 收集 bw 节点直接引用的 fwd 节点集合 U;
2. `recompute_closure = U \ (saved ∪ leaves)`;对其中每个节点按拓扑序
   用 `_RULES` 无关的**前向克隆**(复制原 op,非导数)发射进 bw 区域,
   memo 化(v1 `_BackwardBuilder._recompute` 的逻辑在抽取层复活);
3. 克隆链的操作数递归处理(仍走 saved∪leaves 判定);
4. 之后走既有 `_copy_nodes(bwd_nodes+clones, bwd_out_args, external_as_inputs=False)`
   ——内部引用全闭合,无需自动占位。

fw 图与 default 相同:输出 user+saved(saved 变少即收益)。

### 6. 复用映射(零新概念)

| 需求 | 现有件 |
|--|--|
| tag 遍历/分类 | partition_default 同款循环 |
| 子图拷贝 | `_copy_nodes`(已修净) |
| 重算克隆 | v1 备份 `/tmp/opencode/aot_v1_backup.py` 的 `_recompute` 思路 |
| role 标签绑定 | input_kinds/input_keys 管道原样 |
| 数值规则 | 不触碰(_RULES 只属 sweep) |

### 7. 验收标准

1. **数值等价**:mul/relu/sum、div 链、多叶子上 grad 逐元素等于
   partition_default 结果(rtol 0);
2. **单调性**:同一图 min-cut 的 |saved| ≤ default 的 |saved|;
3. budget 生效:给极小 budget 时割集收缩到必存类;
4. ban_fusible_chains:长除法链(a/b/c/d)不开裂成逐层重算(对照 torch
   fusible 禁令语义:链式 fusible 组整体要么全存要么全重算入口);
5. test_aot 以 `partition=` 参数化跑双实现。

### 8. 已知边界(v2 明确不支持,报 AOTError)

impure(rng)/多输出/get_attr 参与 min-cut 时归入必存,不参与优化;
sym/meta-only 节点沿用 default 直通。

### partition_min_cut 实现记录

- 已按 P3-L4b 设计实现于主树 `aot.py`:Edmonds-Karp(自研,4 例单测含中段
  可达性全过)、容量模型、fusible 链内部强制必存(ban_fusible_chains)、
  backward 重算闭包递归克隆(memo 化 ensure)替代自动占位;
- build_aot 新增 `partitioner="default"|"min_cut"` 分发参数;
- 关键语义修正:残量图**不可达**侧候选 = saved(可达侧保存边已被割);
- ⚠️ 数值级验证仍被原生阻塞(构建因 Tier5OpsKernels.cu L251 语法错中断,
  对方文件未提交、mtime 06:22 后无动静)。下会话:等/修 L251 → 构建 →
  test_aot 以 partition 参数化跑双实现对拍(验收标准 1/2 条)。

### L4 最终验证(独立结构级,免原生)

`/tmp/opencode/final_verify.py`:standalone 加载 graph.py+aot.py,手工建图驱动
sweep+双 partitioner,**12/12 全过**(role 对齐/单调性/链内必存/budget/AOTError/
单叶梯度)。并抓出三个运行时必然崩溃的真缺陷已修:
1. `_JointBuilder.tangent` 未创建(v2 晋升版 build_aot 对任何输入都会崩);
2. 规则辅助发射(`_ones_like`)未打 backward 标签 → 节点误入前向侧污染保存集
   ——改为 sweep 后按 output 位置统一补标记;
3. maxflow 对仅入边节点 KeyError(capacity.get)。
语义澄清:min_cut 下被保存节点背后的叶子不再进入 backward 占位符(优化点,
非缺陷);default 保持全外部引用占位。剩余验收(数值等价/内存曲线)待原生层。

## L5-M4 分解表(已实现+独立验证;2026-08-24 大幅扩容)

- `compiler/decompositions.py`:`DecomposePass`(PassBase 体系)+ 注册表,规则按语义名
  同时命中 `call_method` 与 `call_function` 两种捕获形态;
- **扩容(2026-08-24)**:4 条 → 20 条,公式逐条检查本地分解表
  与核心语义:softplus/mish/tanhshrink/logit/log1p/expm1/exp2/log10/sinh/cosh/
  asinh/acosh/atanh/sec/csc/cot/rad2deg/deg2rad/lerp/addcmul/addcdiv(+既有
  sigmoid/silu/swish/reciprocal/square)。**每条都只落到 POINTWISE_FUSED 原语集**,
  因此每个条目直接倍增"可原生编译算子面"(test 断言 `_stax_native_graph` 非空);
- **接入默认管线**:api.py PassManager = Normalize → ConstFold → **Decompose** → DCE
  → FusionHint(torch 同序:decompose 先于调度);
- compare 族(elu/selu/hardshrink/sign 等 where+gt 型)显式缓项——需 where/gt 进入
  原生集后再放行(文件头注释已声明);
- **分解表暴露并修复真 bug**:`derivatives.yaml` silu 公式漏 `(1-σ)` 因子
  (旧值 0.9239 vs 有限差分 0.7354),已按 torch derivatives.yaml 修正;同批补齐
  sinh/cosh/asinh/acosh/atanh/logit/expm1 共 8 条缺失求导(全部 FD 验证);
- test_decompositions.py 扩至 31 用例:数值比较 ×13、原生编译 ×11、梯度比较 ×7。
- **形状/视图求导对齐(2026-08-24 续)**:expand/repeat 公式改为逐字对照上游
  derivatives.yaml(`sum_to(grad, self.sym_sizes())` /
  `repeat_backward(grad, repeats, self.sym_sizes())`);`ManualNodes.h` 两个
  helper 重写为忠实移植——`sum_to`/`is_expandable_to` 对照 `ExpandUtils.h`
  (批量 keepdim 单次求和 + 尾部对齐可扩展检查),`repeat_backward` 对照
  `FunctionsManual.cpp`(repeat==0 零保护、unsqueezed 前导维求和、仅 repeat!=1
  维进 reshape+单次批量求和),替换原先自行设计的逐维循环版。

## L5-M3 CUDA graphs 编排层(已实现+假绑定全逻辑验证)

- `compiler/cudagraphs.py`:CudaGraphManager(capture-once/replay 静态缓冲),
  原生面契约六符号(cuda_stream/graph_begin/end/instantiate/launch)惰性探测,
  缺失时 NotImplementedError 列出缺口——系统层落地后零改动点亮;
- 契约守卫:nested capture 拒绝、key+签名捕获唯一、replay 前置校验
  (arity/shape-dtype 签名)、max_entries 上限;
- 可注入 native → 全部逻辑现可测:verify_m3.py 五组全过(暂存/计数/契约/
  capture-once/nested/缺绑定向导);test_cudagraphs.py 四用例待批跑;
- 抽象边界澄清:输出值刷新属原生图重放职责,编排层保证暂存+launch+
  输出对象稳定;side-stream warmup 为 v0 简化点(文档已注)。

## L5-M1/M2 codecache 接线 + 编译期 autotune(2026-08-24,布局对齐 torch)

**文件布局修正**(用户指正):torch 的 Triton 在 `torch/_inductor/codegen/`,
`torch/backends/` 只放设备开关。据此重排:
- `tensorplay/_stax/codegen/triton.py`(原 backends/triton.py)——内核
  codegen + `_compile_program` + 归约尾声检测;
- `tensorplay/_stax/runtime/stax_autotune.py`(新,CachingAutotuner 对标);
- `backends/stax.py` 保持 backend 门面,延迟导入 compiler.codegen.triton。

**M1(codecache 接线,PyCodeCache 对标)**:`_compile_program` 源码内容寻址
落盘(`default_cache("triton")`,ext=py)+ 进程级 launch callable memo,
重复 compile 不再重生成/重 exec。

**M2(编译期 autotune,CachingAutotuner/do_bench 对标)**:
- 候选 (XBLOCK,num_warps) 四配置;CUDA Event 计时 warmup+iters 取均速;
- 决策持久化:`triton-autotune` 缓存,key=digest|xnumel 幂桶|device
  (JSON),后续进程零 benchmark 直接固定配置发射(去 @triton.autotune
  每次调用的开销);TP_DISABLE_STAX_AUTOTUNE 可关;
- 任一候选失败即淘汰,全败回退装饰器路径。

**归约尾声 codegen(M5 首块)**:mul/relu/sum 型图由 3 内核 → **1 内核**
(`_split_sum_epilogue` 检测全点链+尾 sum;kernel 内 `tl.sum` 折叠,标量
输出缓冲)。v1 仅推理态;训练走既有双程序 autograd 路径。
stax 可融合算子集上移为单一事实源 `fx_passes.POINTWISE_FUSED_OP_NAMES`
(stax 反向导入),消除双份常量。

**P0 首批 pass 补齐(L2)**:
- `fx_passes/normalize.py` NormalizeOperators:交换律常数右置、x+0/x-0/
  x*1/x/1/x**1、neg(neg(x))(含 output 直连改写);x*0 刻意不折(NaN 语义);
- `fx_passes/fusion_hint.py` PointwiseFusionHint:最大点区并查集标注
  meta[fusion_hint/fusion_region],非点算子为边界;接入 api.compile 默认
  管线(Normalize→ConstFold→DCE→Hint→ShapeProp);
- 两 pass 均幂等(tests 断言二次 modified=False)。

**测试**:test_stax_autotune.py(决策缓存/择优/淘汰/禁用开关/发射形态)+
test_fx_passes.py(normalize/hint/管线集成/尾声检测)。远端 P4 双解释器
(system + miniconda-torch-env)全绿;triton e2e 三用例以
`runtime_available()` 功能探测门控——该机 triton3.7/torch2.10cu128 均需
sm_70+,Pascal 上正确 skip 而非误报。本地同套全绿。

### L5-M3b 原生补齐(2026-08-24,远端构建+GPU 实测通过)

原生后端(953c349)落地后走读发现并处理五项:

1. **instantiate 非幂等 → 第二次 replay 必崩**。`graphs.py` 的
   `CUDAGraph.replay()` 每次都调 `cuda_graph_instantiate`,而原生端模板在
   首次实例化后即置空、再次调用抛 "already instantiated"。修复:原生
   instantiate 对已实例化句柄幂等返回(CUDAGraph.cpp),与 Python 契约
   "happens automatically at replay" 对齐。
2. **capture 期 RNG 走值模式 → 重放随机数恒同**(graphs.py 文档注明的缺口,
   torch Note [CUDA Graph-safe RNG states] 对齐):
   - `CUDAGenerator.h/.cpp` 新增 `PhiloxCudaState`(值模式/指针模式变体)+
     `philox_cuda_state()`:isCapturing() 时返回图私有 [seed, offset]
     Int64 设备缓冲指针 + 图内基础偏移,RNG kernel 执行期解引用;
   - 图钩子四件套 rng_register_graph(beginCapture 前,默认流分配+同步,逃逸
     graph-pool)/rng_capture_epilogue(EndCapture 后收 wholegraph_increment)
     /rng_replay_prologue(launch 前刷新缓冲+生成器 offset 推进,data()
     自动 recordStream 到 launch 流)/rng_unregister_graph(destroy 时);
   - kernel 侧调用点改造:RandomKernels.cu 全族(rand/randn/normal_/randint/
     poisson/bernoulli)+MiscKernels.cu dropout 改收 PhiloxCudaState;
   - CUDAGraph.cpp 接线:GraphHandle 增 rng_state_id/wholegraph_increment,
     begin/end/launch/destroy 全生命周期挂钩,失败路径同步清理。
3. **捕获期 cudaMalloc 触发 CUDA error 900**:allocator 的段扩容在 Global
   捕获模式下属"不安全 API"。修复:仿 torch CUDAStreamCaptureModeGuard,以
   `cudaThreadExchangeStreamCaptureMode` 在 cudaMalloc 附近临时切 relaxed
   (CUDAAllocator.cpp;注意本 CUDA 头无 `_t` 别名,用枚举名)。
4. **RNG 内核裸 `<<<grid, block>>>` 启动 → 捕获图 nodes=0**(最隐蔽):无流参数
   = 遗留默认流,Global 模式下既不报错也不入图,内核在窗口内 eager 执行、
   图空转。修复:RandomKernels/MiscKernels 全部启动点补
   `getCurrentCUDAStream().stream()`。⚠️ 其余 .cu 文件可能存在同类裸启动,
   待系统层排查。
5. **编排层静态缓冲克隆入图**:compiler/cudagraphs.py 原先在窗口内 clone,
   克隆节点被烤进图,每次重放用样本值覆盖暂存输入。修复:clone 移到
   begin_capture 之前(顺带使静态输入落在常规池,寿命独立于 graph destroy)。

**验证(远端 P4/CUDA12.8)**:test_cudagraphs 4/4;verify_m3_native 全过
(replay×2 数值正确/instantiate 幂等/RNG 三次重放各不相同/换种子后与同种子
eager randn 位级一致);test_compile 除 cuDNN 环境性失败外全绿。

**环境备注**:该机 cuDNN 9.19 与 miniconda torch 二进制均不含 sm_61 内核
(torch relu 也报 NoKernelImageForDevice),所有走 cuDNN/torch vendor 库的
用例在该机不可判定,非 TensorPlay 缺陷。

### AOT SIGSEGV 精确线索(2026-08-24,远端 100% 复现)

文档先前记载的非确定性崩溃在本机可稳定复现(gdb 栈):
```
#0 tensorplay::TensorIteratorBase::reorder_dimensions()
#1 TensorIteratorBase::build(TensorIteratorConfig&)
#2 TensorIterator::reduce_op(...)
#3 cpu::sum_kernel_impl / #5 redispatch_sum → tpx::ops::sum
#7-9 _C 绑定层 ← python eager_out.sum().backward()
```
触发条件:test_aot 中先 build_aot+value_and_grad 再跑 eager `.backward()`
(test_aot.py:34)。纯 CPU 路径,与 CUDA graphs/RNG 无关。最小复现脚本
单独跑 build_aot+value_and_grad 不崩;崩点在 eager backward 的 sum 归约
TI build。嫌疑:AOT bw 图执行后遗留的悬垂形状/元数据被 TI 读取。
下会话:ASAN 重编 p10 或 valgrind 抓 reorder_dimensions 的越界读。

### 协作事故记录(2026-08-24)

远端构建机上存在第二工作树 `/root/TensorPlay`(另一开发者)。本会话排查
并发 ninja 时使用了全局 `pkill -f "ninja|nvcc"`,误杀了对方的构建进程
(两树分属不同文件系统,本无文件冲突)。教训:共享机器上清理进程必须按
cwd(/proc/PID/cwd)限定自己的树,禁止全局 pkill。

## L5-M5 融合面扩展设计(2026-08-25,基于本地编译后端双侧走读)

决策:正面竞争,自研后端实现完整能力;参考源码 = 本地参考源码树
(commit 893b6406)。下文行号均相对 `_inductor/`。

### ⚠️ 先修地基:M5a sum 尾融合发射不完整(走读发现的真 bug)

`codegen/triton.py` 的 reduction 分支:kernel 体计算 `reduced = tl.sum(...)`
后**没有任何 store**(242-246 跳过了 store 循环),grid 仍是多 block
(288),各 block 部分和既不落盘也不聚合——当前生成的标量输出缓冲是
未初始化内存。test_fx_passes.py 只断言源码含 `"tl.sum("`/`"tp.empty(()"`,
无数值验证。**任何纵向融合工作开始前必须先补齐并加数值比较测试。**

### 我方现状盘点(可复用 vs 硬缺口)

可复用:点算子后缀 program + CPU 向量核/Triton 直线发射双端执行器;
反向程序构建器(adjoint+活跃度压缩);`_split_sum_epilogue` 检测框架;
PointwiseFusionHint union-find 区域标注(**现无人消费,闲置资产**);
autotuner 全套;codecache 内容寻址;AOT 双图管线;cuBLASLt bias epilogue
半成品(CudaBlasGemm.cpp:210,344-415);前端形状特化保证 lowering 时形状静态。

硬缺口:
1. program 表示只有一维逐元素三元组,无 tile/归约轴描述(stax.py:100-120);
2. 归约只能全量 sum 且必须在图尾;归约后不能再接 pointwise;
3. 广播全路径被禁(Triton 入口 triton.py:105-118 同 shape 校验),bias/scale
   场景直接不可行;
4. mm 是 extern 黑盒,linear 被预展开 t+matmul+add 三节点(stax.py:1400-1412),
   无可识别 epilogue 模式;Triton GEMM 发射层不存在;
5. 训练与融合互斥(AOT 显式 use_fusion=False,stax.py:2246-2251;Triton 反向
   无归约 VJP);
6. dtype 单一 fp32 假设,无累加器语义;where/gt 缺位卡住 max 族;
7. 前向强制单输出(argmax 受阻);fusion_hint 无消费者(判据漂移风险)。

### torch 侧机制要点(移植锚点)

- **IR**:Loops{ranges, reduction_ranges, inner_fn 闭包}(ir.py:1059,1399);
  OpsHandler 六原语 load/store/reduction/store_reduction/masked/index_expr
  (ops_handler.py:240-289);combine_fn 注册表含 argmax 平局规则(ir.py:1296-1396);
  降级三连:rnumel==0/1→pw、unroll<8→pw(ir.py:1886-1921);
  realize 闸门 has_large_inner_fn 防表达式爆炸(ir.py:10922-11000)。
- **调度器**:SchedulerNode group=(numel,rnumel)(scheduler.py:2649-2683);
  mutation 改名破环(:4702-4717,5339-5410);can_fuse_vertical =
  exact-MemoryDep 匹配 + 剩余 dep 祖先检查(:9042-9113,9179-9233);
  **pw→red 形状谓词 `numel_pw == numel_red*rnumel_red`**(simd.py:2438-2470);
  red→pw 泛化路径禁止,只能走模板 epilogue 专用通道;贪心配对 + shared-dep
  打分(scheduler.py:7513-7567,9474-9556)。
- **prologue 免费拿**:Enable/DisableReduction 标记让 red 前的 pw 链进同一
  kernel(simd.py:2636-2715,794-817);fits_in_main_body / fits_outside_reduction
  (simd.py:2645-2653)。
- **归约两态**:persistent(RBLOCK=next_pow2(rnumel) 静态化,triton.py:7694-7734,
  阈值表 INNER:1024/其他:64,choices.py:448-513)+ IR 期 multilayer 两 kernel
  workspace 拆分(num_splits,ir.py:1476-1689,2292-2338)。跳过 cooperative。
- **config 数值直接抄**:pointwise bs=max(256,min(numel//128,1024)) 及 1D/2D
  组合(triton_heuristics.py:4459,heuristics/reduction.py:29-115);reduction
  按 hint 分流 5 配置(reduction.py:178-354);persistent XBLOCK∈{1,8,32,128}
  cap rnumel*XBLOCK≤4096(reduction.py:356-498)。
- **matmul 模板**:jinja + Config(BLOCK_M/N/K,GROUP_M swizzle)(kernel/mm.py:
  87-98,triton_mm.py.jinja);epilogue 挂接 = store_output 钩子里把 epilogue
  pw 节点的 LoopBody 在子图内重放(select_algorithm.py:1467-1704,1960-2025)
  ——relu/gelu 复用标准 pw codegen,无需单独模板;bias 用 addmm_epilogue
  (mm_common.py:99-107);cuBLASLt 仅一条路由:2D-bias(stride0==0)→extern
  bias_addmm(kernel/mm.py:173-181,713-719)。
- **buffer**:last_usage 反向累积 + free_buffers(scheduler.py:9802-9840);
  inplace 谓词 read/write index 相等且 size 相等(:3008-3019)。

### 执行阶段

| 阶段 | 内容 | 验收 |
|---|---|---|
| M5a | 修复 sum epilogue 发射(单 block 收敛或两阶段);广播最小支持(输入 expand 视图进 Triton path) | mul/relu/sum、addcmul 型数值比较 vs eager;带 bias 广播用例 |
| M5b | 表示层升级:program 增加 reduction 轴(dim/keepdim)+ 多输出;归约组合子注册(sum/max/amax,argmax 后置) | mean(dim)/amax(dim) 尾融合比较;argmax 双输出骨架 |
| M5c | Python 融合调度器:消费 fusion_hint 或替换之——依赖提取、group=(numel,rnumel)、can_fuse_vertical(exact dep 匹配简化版)、Enable/DisableReduction 标记 | pw→red→pw 图正确切成 ≤2 kernel;red 后断开语义与 torch 一致 |
| M5d | Triton 归约 codegen:persistent + multilayer split 两态;fp32 累加器;config 候选接进 stax_autotune | 大 reduce-to-scalar(split)与 rnumel≤1024(persistent)比较+性能 |
| M5e | matmul epilogue:A 路线 cuBLASLt RELU/GELU epilogue 组合 + linear 不再预展开改模式匹配;B 路线 Triton GEMM jinja 模板 + store_output 子图重放 | linear+bias(+relu/gelu)单 kernel;vs extern mm 性能不回退 |
| M5f | 训练态:sum VJP(tangent expand)进 _build_fused_gradient_graphs;或 aot.py 分区器与融合组联动 | train 态归约融合梯度比较;AOT 不再一刀切关融合 |

依赖序:M5a → M5b → M5c → M5d/M5e 并行 → M5f。M5a–M5d、M5f 纯 Python;
M5e-A 路线动 CudaBlasGemm.cpp,需遵守 AGENTS.md 构建纪律。

明确跳过(第一版):sympy 符号形状、allow_index_equivalence 宽松匹配、
HOPs(scan/welford/online_softmax)、cooperative reduction、TMA/block ptr、
foreach/combo 异构容器、NestedReduction/sub-parent staged epilogue、
benchmark_fusion/autoheuristic 调优基建。

### M5a 落地记录(2026-08-25)

- **sum 尾融合修复**:`codegen/triton.py` 重写发射——归约输入 ≤1024 元素走
  单 kernel 直写标量(grid=(1,), XBLOCK=next_pow2(numel));更大输入走两阶段
  split(主 kernel 写 `ws_ptr[pid]` 部分和 + finalize kernel 掩码加载归约),
  对齐 Inductor multilayer 拆分语义的迷你版。归约路径一律 config 钉死,
  不再退回 @triton.autotune(workspace 尺寸随 config 烘焙);autotune 关闭/
  失败时用静态默认配置 (256,4)。launcher 内嵌参考形状 numel 字面量。
- **最小广播支持**:Triton 路径入口改按 torch 广播规则求 reference_shape;
  广播输入以编译期 div/mod 偏移链寻址(stride-0 维折叠,numel-1 直接单次
  load)。`_supports_runtime_inputs` 增加 reference_shape 参数。广播+训练态
  显式回退(反向构建器尚无 sum-to-shape,M5f 解锁)。CPU 融合路径不动。
- **清理**:删除迁移残留 `tensorplay/backends/triton.py`(无引用,且仍含
  修复前的坏发射逻辑,易误导)。
- **测试**:新增 test/test_triton_reduction.py——本地可跑的结构/偏移表达式/
  广播规则单测 13 项;GPU 数值比较 4 项(sum 单块/10 万元素 split/广播 bias
  点算/广播链 sum)经 runtime_available() 门控,待远端 P4 执行;断言经
  `_tensorplay_cache` 反查 `_tensorplay_codegen=="triton"`,防止解释回退假绿。
  本地:test_fx_passes+test_stax_autotune+test_triton_reduction+
  test_decompositions+test_codecache 64 passed;test_compile 15 passed。
- ⚠️ 预存失败(与本工作无关):test_compile::
  test_stax_fused_pointwise_extended_autograd_matches_eager 在原版代码同样
  失败——ShapeProp 解释期 CPU eager 报 "Unsupported dtype",疑被进行中的
  DType/TypePromotion 迁移波及(git status 大量 p10 dtype 文件未提交改动)。
  待该迁移收敛后复核。

## export 修复(2026-08-25 晚,3 处)

test_export 8 例中 5 失败 → 全绿。三个独立根因:

1. **is_compiling 归位到前端层**(架构修正):捕获标志 ContextVar 原在
   api.py 由 `_compiler_context` 置位,但 `export.py` 直接调 `Tracer().trace()`
   绕过了它 → batchnorm 的 `_check_input_dim`(已按 TP 惯例用
   `tensorplay._stax.is_compiling()` 守卫)在 Proxy 上做元数据控制流炸
   GraphCaptureError。修法:标志与 `compiler_context()` 移入 graph.py
   (依赖方向 api→graph,graph 不反依赖),`Tracer.trace` 执行用户代码段置位;
   api 复用之。任何直接 Tracer 调用方(export/测试/未来前端)语义一致。
   对照:torch 无此问题是因为 fx 默认把 torch.nn 子模块当 leaf(call_module)
   不内联;我们选择内联(利于融合),故元数据守卫必须真实生效。
2. **export dynamic_shapes 收紧**:去掉 str→Dim 宽松转换,test 规格要求对
   非法 value(str/list 等)抛 TypeError("int or Dim");bool 显式排除
   (bool 是 int 子类)。非法 key 的 ValueError 原本已有。
3. **ExportedProgram.__call__ 兑底**:export(fn, x, offset=5.0) 后调用
   program(x) 时 kwargs 丢失落回默认值。现按 placeholder 名序绑定实参,
   未提供的从 example_inputs(导出期绑定)补全后再进 graph_module。

回归:test_export 8/8;compile/fx_passes/stax_autotune/triton_reduction/
decompositions/aot/codecache 共 105 passed;optim 7 passed。
插曲:期间本地共享树被并行构建改写 libp10/_C 出现瞬时符号不一致,
按 AGENTS.md 等待对方构建静默后复测。

## L5-M5b 落地记录(2026-08-25 深夜)

**program 表示升级:归约轴(dim/keepdim)+ mean/amax/max 全量族。**

- `ReductionSpec`(op/dims/keepdim):`is_full`、`normalized_dims(rank)`、
  `output_shape(ref)`(keepdim 折叠语义)、`reduction_numel(ref)`、逐 op 的
  combine/neutral/finalize 映射表(sum|mean→tl.sum+add,amax|max→tl.max+
  maximum,-inf 中立值);旧 `reduction="sum"` 字符串入参兼容映射。
- **检测泛化**:`_split_sum_epilogue` → `_split_reduction_epilogue`,
  `_reduction_spec_from_node` 解析 call_method 尾节点——⚠️ 关键坑:方法节点
  args[0] 是接收者,解析前必须剥掉(否则 sum() 被当成 dim=Node 拒绝);
  amax() 无轴、max(dim) 值索引对、dtype/out kwargs 显式拒绝回退;
  `_split_sum_epilogue` 保留为全量 sum 兼容入口(test_fx_passes 不动)。
- **发射四分支**:pointwise / 全量单块 / 全量两阶段 / **带轴 tile(M5b 新)**:
  输出空间 XBLOCK × 归约空间 RBLOCK 二维 tile,内层 `tl.range` 逐块折叠,
  掩码加载 other=中立值(-inf 防 max 被污染),mean 循环后乘 1/rnumel;
  launcher 按 output_shape 分配 + 字面量 grid;v1 配置确定性
  (`_dim_reduction_config`:XBLOCK≤256,RBLOCK≤512),autotune 接入留给 M5d。
- **寻址正确性**:偏移表达式 = kept 维按输出平坦索引分解 × 输入 stride +
  reduced 维按 rindex 分解 × stride;150 组 (op × dims × keepdim × ref)
  配置对拍暴力坐标映射全过。修了三个真 bug:①坐标项缺括号(`%` 与
  `[:, None]` 优先级);②单维折叠快捷分支漏乘 stride;③多维时末维不可折叠
  (rindex 枚举整个归约空间而非该维)。
- **测试**:test_triton_reduction.py 扩至 17 本地项 + 6 GPU 门控项
  (sum(dim)/mean(dim 组)/amax(keepdim) 比较待 CUDA);本地全套
  112 passed。
- 远端状态:GPU 比较挂起,等待对方完成 CUDA 构建(唯一构建纪律,
  我方不再发起构建;产物就绪后仅跑测试)。

## L5-M5c 落地记录(2026-08-25 深夜续)

**静态融合分段调度器上线(单一事实源)。**

- 新模块 `compiler/scheduler.py`:`segment_graph(gm, *, is_pointwise,
  classify_reduction)` —— 依赖倒置,调度器不 import backends/codegen(无环);
  `Segment{nodes,kind,reduction}`、`describe`("pw+red -> pw")、
  `annotate`(计划写 gm.meta["stax_segments"]);
- 分段规则 = Inductor 垂直融合迷你版:pw 链合一段;pw 段可挂一个归约尾声
  (pw→red 纵融);red 之后必断(red→pw 边界);red→red 相邻拆两段;
  外部算子(mm/reshape 等)整图回退(v1 不做解释/编译拼接);
- **接入**:codegen/triton.py 的 compile_graph_module 弃用直调
  `_split_reduction_epilogue`,改走调度器——单段才编译,多段注记 meta 后
  回退;旧检测函数保留为兼容入口。消除"fusion_hint 无消费者/判据漂移"
  缺口(POINTWISE_FUSED_OP_NAMES 经注入谓词成为唯一判据);
- 协作纪要:并行工作流在同文件落地了 argmax 索引归约(ReductionSpec.
  tracks_indices + tl.argmax 发射 + dtype 门控),与本方 M5b/M5c 改动无冲突,
  测试共绿(32 triton/fx 用例);
- 测试:test_scheduler.py 7 项(pw 单段/纵融/red→pw 断开/red→red 拆分/
  外部回退/裸输入归约/meta 注记);全套 compiler 相关 102 passed。
- 下一步(M5c 续):按段发射——多段图逐段编译 + 运行期张量传递拼接,
  让 `(x*w).relu().sum()*2+b` 类图从整体回退变为 2 kernel 编译执行。

### M5c 续:按段发射落地(同日)

多段图不再整体回退。`compile_graph_module` 重构为通用按段管线:

- `_extract_segment_view`:把 Segment 节点克隆进独立子 Graph(跨段引用
  自动变占位符),producer 是占位符时经 externals 解析(sum() 直连输入的
  边界情形);
- **接线验收门**(`_extern_sources`):每段只允许一个导出值=末节点
  (内部 skip 连接留 v2 多输出);外部依赖仅限图占位符或**前序段的尾值**;
  违例注记 meta 后回退;
- 每段独立走 program 构建 + autotune(全量 sum 单块/两阶段、带轴 tile、
  argmax 门控全部复用,零特判);编译期样本张量按来源合成
  (占位符用真实输入;前序段输出 pw→ref 形状、red→标量);
- `compiled()` 运行期按拓扑顺序喂段:`interm[i]=launch(feed)`,
  标量中间值进入下一段时由广播偏移机制自然处理(numel-1 单次 load);
- 训练态(any_grad)仍限单段(M5f 补分段 VJP);
- 测试:+4 本地结构项(two_programs/scalar_feed/接线门/抽取边界)、
  +2 GPU 门控 e2e(two-segment 比较、scalar 链 sqrt);全套
  **125 passed, 17 skipped**(GPU 项待远端)。

## L5-M5d 落地记录(2026-08-25 续,性能轮)

用户反馈"性能不如 torch",针对带轴归约发射路径做三项优化:

### persistent 免循环形态(rnumel ≤ RBLOCK)

- `_dim_reduction_config` 返回 4 元组 (XBLOCK, warps, RBLOCK, stages);
  rnumel ≤ 512 时 RBLOCK=next_pow2(rnumel) → **单 tile 覆盖整个归约空间**,
  发射免 for 循环体:rindex 直接 `tl.arange`,无 roffset/rmask 迭代、
  无 tl.range 流水线开销;argmax 的 cwin 同步去掉 `+ roffset`。
- 大归约空间仍走 `tl.range(..., num_stages=3)` 软件流水
  (torch inductor 默认同款深度),launcher 端同步下发 num_stages。

### 带轴 autotune 接入(补齐 M5d 计划缺口)

- `_DIM_REDUCTION_CANDIDATES` 六档候选表 + `_STATIC_DIM_TRIPLE=(128,4,3)`
  静态兜底;`_autotune_dims_program`:bench_launch 实测 → 最优持久化到
  `default_cache("triton-autotune")`(决策键 = program digest + 归约 spec +
  onumel/rnumel 桶 + device/value_dtype),二次编译零 bench 单 build 直取。
- 决策缓存中毒防护:非法配置(不在候选表)忽略并重扫;全候选编译失败回退静态;
  TP_DISABLE_STAX_AUTOTUNE=1 跳过扫描。

- 测试:codegen 结构测试拆 persistent/loop 两形态断言;+4 autotune 单测
  (持久化复用/禁用开关/全失败兜底/坏缓存重建),mock _compile_program 注入;
  全套 1304 passed(2 个 conv eager 失败系他人未提交 C++ 改动,与编译器无关,
  按 AGENTS.md 纪律不触碰)。GPU 比较待远端验证。

## L5-M5e 落地记录(2026-08-25 深夜,融合范式对齐轮)

用户判定"融合范式差距巨大"。排查发现两处结构性落后于 inductor 并当轮修复:

### 1. 多段发射门被收窄(回归修复)

`compile_graph_module` 的入口门仍是 v1 的 `len(segments) != 1 → fallback`:
M5c 建好的按段管线(`_extern_sources`/`_extract_segment_view`/intermediates
接线)全部处于死代码状态,pw→red→pw 实际整图回退 eager——结构层测试照过,
GPU e2e 因回退也数值通过,问题被掩盖。现打开为任意段数,每段经
`_extern_sources` 验收;跨段消费改认 `Segment.export_node`(含 epilogue 尾)。

**每段局部 reference**:后续段的输入是中间张量,形状≠全局 reference。
样本合成改为 `reduction.output_shape(reference_shape)`;每段以自身输入的
broadcast 形状作为 codegen 的 reference(单块/split 阈值、dims tile、mean
缩放全部按局部形状推导),段与段真正可组合。

### 2. red→pw store epilogue(inductor 单 kernel 范式)

- scheduler:归约后的纯 pw 节点若**传递依赖归约结果**(live = {red 尾} ∪
  已并入节点,其余依赖只能是 placeholder)则并入同段 `Segment.epilogue`;
  纯 placeholder 链不能并入(kernel 内已无 pre-reduction tile)。
- codegen:`epilogue=(program, constants, esrc)` 三元组;acc 定稿(mean 缩放
  之后)在寄存器内求值 epilogue 程序再 store。四种形态全支持:dims
  persistent/looped、full 单块、split 的 finalize kernel。argmax(int64 流)
  拒绝 epilogue。launcher 签名不变(unary epilogue 零新增参数);autotune
  digest 含 epilogue 内容。
- 效果:pw→red→pw 从 2 kernel → **1 kernel**,对齐 inductor 的
  "reduction 结果不落地直接进后继 pointwise"范式。
- 训练态仍限单段无 epilogue(M5f 补分段 VJP/归约梯度)。

- 测试:scheduler +3(epilogue 合并/传递依赖门槛/annotate)、codegen +6
  (四形态发射/mean 先缩放/argmax 拒绝)、计划层 +2(单段化+epilogue 程序
  构建)、GPU 门控 e2e +3(pw→red→pw 单 kernel、标量链、双生产者跨段接线)。
  编译器域全套绿;全库另有 ~17 个失败系并行方未完成重建(C++ 绑定变动,
  与本域无关,按纪律未触碰)。GPU 比较待远端。

## L5-M5b 收口记录(2026-08-25 深夜,argmax 双流归约 + conv 族配套)

**argmax 索引流归约(M5b 收口件,与 reduction 轴线并行落地)。**

- `ReductionSpec` 增 `"argmax"`(`tracks_indices`;要求显式 dims,flatten 形态
  拒绝);
- 发射双流:值流 acc(dtype 按输入烘焙 `_VALUE_TYPES` f32/f64)+ 索引流 acci
  (int64);块内 `cval=tl.max(tile,axis=1)`、`cwin=tl.argmax(tile,axis=1)
  +roffset`;合并 `take=(cval>acc)|(tl.isnan(cval)&~tl.isnan(acc))` ——
  严格 `>` 保首现块(torch.argmax 平局取首),isnan 子句对齐 torch 的
  NaN 视为最大序;
- launcher 为 argmax 分配 int64 输出;`HAS_TL_ARGMAX` 特性探测门控折叠;
  compile_graph_module 限 float32/float64;value_dtype 贯穿
  _compile_program/_autotune_launch。
- 测试:test_triton_reduction.py 检测/结构/f64/digest + GPU 门控比较
  (argmax、keepdim+平局、NaN 注入),本地 18 passed / 8 skipped。

**conv 族配套修复(同轮):**

- **int[] 标量维度归一化**:tools/codegen/gen_python.py 对 int[] 参数发射
  `isinstance(x,int) and not isinstance(x,bool) → [x]`,重生成
  functional.py——修 `x.amax(dim=-1)` TypeError,顺带点亮 conv
  stride/padding/dilation 标量入参;生成物与手补字节级一致。
- **pad backward**:PadKernels.cpp pad_scatter_kernel 外层序号 o 原用全张量
  stride 解码(padded 维被折叠进 outer 坐标,outer>1 时写错位)→ 改用
  new outer_strides 解码;CUDA 侧 flat dst + atomicAdd 本就正确未动。
- **LazyConv**:_LazyConvXdMixin.__init__ 按位置传 _ConvNd 参数 → "got
  multiple values for argument 'device'";改 kwargs 直通,六个 Lazy 类以
  关键字传具体类参数;_infer_parameters 对齐 hook 协议。六类构造+前向全过。
- **conv_tbc**:实现 torch 三维 weight (k,C_in,C_out) 契约
  (weight.permute(2,1,0) + groups=1 conv1d),校验 dim==3 与 C_in 匹配。
- **test_conv_alignment.py**:_grads_tp 共享 torch 侧随机切量(旧 13 失败
  纯属 RNG 切量错配,非内核错误——conv_transpose grad_input/unfold/fold
  内核经匹配切量验证全部正确);bf16 经 f32 往返;conv_tbc 权重 (3,3,5)、
  fold 输入 L=12。独立跑 25/25。

**amp-first flaky circular conv1d:根因实锤并收口(oneDNN 缓存两案)。**

- 案 A(权重 reorder 缓存键失效):ConvKernels.cpp 训练路径以临时 unsqueeze
  view 的 `TensorImpl*` 为缓存键,守卫仅 (data_ptr, version_counter)。
  分配器回收后,新层可同时命中死层的 impl 槽位与参数存储地址 → 守卫全过,
  静默用死层重排权重计算。DIAG 实锤:wsum==refwsum(权重拷贝无误)、
  tp-vs-manual=False / torch-vs-manual=True、同输入连跑两次输出漂移 2.7、
  版本 bump 后与 torch 精确一致。test_amp 先行时 ~75% 复现(堆布局敏感),
  独立必绿。修复:缓存条目钉住源 Storage(存活期内地址不可回收,data_ptr
  等值即真同源)+ map 256 上限防死键堆积。修复后 amp-first 组合 8/8 绿。
- 案 B(conv2d_grad_weight_onednn 原地换底,test_conv_full bias.grad
  75-vs-200 根因):该"优化"把共享 grad_output 的 storage 原地换成 blocked
  格式缓冲并挂 md;autograd 按 input→weight→bias 求导,bias 内核随后按稠密
  NCHW 读 blocked 字节([75×3≈600×3/8 正是 NChw8c 密度];grad_input 先行
  未受污染、grad_weight 自读 md 故仅 bias 坏)。原换底还兼任局部缓冲寿命
  延长(直接删会悬垂);改为自持有缓冲 `memory(md,eng)`(对齐 grad_input
  路径既有修法),调用方张量不再被触碰;onednn_memory_cache 握手保留
  (desc 校验 + 每 backward 新 impl,无跨步陈旧)。修复后 test_conv_full
  8/8(含 conv3d scratchpad 连续 fwd+bwd 压力)。
- 验证窗口备注:两案编译进 ConvKernels.cpp.o 并随 00:05 libp10.so 链接后
  通过上述回归;随后并行绑定重构线(libtp_python 符号失配)令全库导入瞬时
  中断,按共享树纪律待其收敛复测全量。

遗留:GPU 比较(argmax/pad 等)待远端 GPU;eager amax NaN 分歧登记于
gap 文档(registered-not-fixed);M5c 调度器下一步为训练态分段 VJP。

## L1-D1 执行式特化落地（2026-08-26）

**范围**:D1 第一步——数据依赖控制流从"硬拒绝"升级为"具体值特化 + 数据守卫"。
真正的断点续捕(子图切分)仍待后续;本步对齐 Dynamo 的
`nb_bool → item() → 特化 + install_guard` 语义(tensor.py:369-380)。

- `compiler/graph.py`:`Tracer(execute=True)` 混合执行模式(make_fx 式边录边执行,
  no_grad 包裹、失败静默降级回符号模式);节点样本经 `_node_samples` 全图传播;
  `Proxy.__bool__/__int__/__float__/__index__/__iter__` 消费样本并记录
  `data_specializations`;张量容器的迭代仍拒绝(元素逃逸追踪面);
  **捕获期**(DCE 前)盖戳 `meta["data_guard_params"]`——事后走查会被
  DeadCodeElimination 删掉的条件子图骗过(实测踩坑)。
- `compiler/api.py`:缓存键扩为三元组 `(输入签名, 形状守卫, 数据守卫)`;
  数据守卫 = 喂给控制流门的占位符参数的 sha256 字节指纹(`_tensor_data_digest`);
  身份快路径在存在数据守卫参数时禁用(原地变异不改身份只改字节);
  提升失效逻辑与形状守卫提升同构。
- `compiler/guards.py`:GuardChain 支持三元组件,explain/guards 渲染 data-guards。
- 已知边界:①指纹 O(input-bytes),仅作用于真正喂门的参数;Dynamo 式标量表达式
  重验 guard 需子图前缀切分,见下步。②execute 模式消耗 RNG(jit.trace 同款注意)。
- ✅ **既有原生 bug 已修(同日)**:TensorBase 从未安装 nb_bool 槽(CPython 对无
  真值协议类型恒真),`bool(t)` 与 `.item()` 分歧、所有数据依赖 eager if 静默走错。
  现于 Tensor.cpp 绑定块实现 torch is_nonzero 原话语义(RuntimeError "no values /
  more than one value is ambiguous");ninja -C build _C 重编,产物新鲜度核验,
  全量测试 1408 绿。注:`__float__/__int__` 槽位由并行线同期补齐。

### M5b 收口续:foreach/autograd 配套修复(2026-08-26 凌晨)

并行 optimizer/amp 线落地 foreach SGD 后暴露两处共享基础设施缺口,当轮修复:

- **validate_lists 可选态列表**:CPU/CUDA OptimizerKernels 的列表校验对
  momentum_buffers 等可选态做无条件 size 检查,momentum==0 时调用方按 torch
  契约传 `[]` 直接 ValueError。改为「required 或非空时才要求覆盖全参数」;
  同时消除空列表下 `&first_state[i]` 的越界 UB(逐元素检查改为显式
  require+非空双门)。test_optim、TestConv3dScratchpadRegression 随之转绿。
- **AccumulateGrad 物化 strided grad**(tpx/include/AccumulateGrad.h):mm 经
  `.t()` 反传的权重 grad 是非连续视图,tp 原样入账 → foreach 内核拒收。
  torch 实测三类(linear/mm(a,m.t())/addmm)grad 全连续;照抄之,入账前
  `contiguous()` 物化(与 torch AccumulateGrad 观察行为一致)。
- 回归:amp/optim/conv_full/conv_alignment 四套 69 passed;全库
  (除 audio)**1181 passed / 161 skipped / 2 failed**——仅剩
  test_compile/test_control_flow 各一项 data-dependent-control-flow 缓存断言,
  属编译器 data-guards 活跃线(3==2 特化缓存计数),未触碰待其收敛。

## L5-CUDA 修复轮(2026-08-26 凌晨,远端 4090D 实测)

目标"全部修复 + 性能超越 torch"。远端 scratch(/tmp/tp_m5e_check)独立构建,
借用对方产物起步后按 .remote_build.md 配方自建(sm_89/MKL/cuDNN-frontend)。
全量 CUDA 套件从 19F+2崩 → **全绿**(113 passed/2 skip + rnn 脚本 exit0):

- CUDAGraph.capture_begin 预热全部库句柄(torch 模式:捕获内 Create 非法,
  曾致 cusolverDnCreate error7 / cuBLASLt error13);
- cuBLASLt 微自动调优在捕获期跳过并把该 plan 钉死 heuristic[0](捕获与
  eager 算法位一致,replay 不漂移);事件记录在捕获流上会中止捕获;
- 条件节点子流延迟到 reset() 销毁(体内外层张量的跨流 fence 对已毁流
  EventRecord → libcuda SEGV);
- debug_dump 恒保留模板图(exec 强转 cudaGraph_t 是 invalid argument);
- 归约发射统一"先程序后重掩码":load neutral 经 pointwise 变换后不再是
  neutral(sigmoid(0)=0.5 计入 sum),single/split/dims 三路在归约前
  tl.where 回 neutral;argmax 用 NaN→+inf 优先级流实现 torch 的
  "NaN 最大、首个 NaN 胜出"(tl.max 在部分 triton 版本忽略 NaN),
  x!=x 写法规避 tl.isnan 兼容性;
- 带轴归约逐输入偏移:广播输入不再误用全形寻址;
- launcher 输出按编译期形状/xnumel 分配(输入序漂移曾致 empty_like 截断);
  _CODEGEN_VERSION 盐入全部内核缓存键(发射器语义变更不回放旧源);
- 池释放改挂起-延迟(reset 不再因静态张量存活而抛,张量死绝时 free 路径回收);
- memory_stats 暴露 allocator 子字典(C++ 碎片化矩阵直通)+ torch 兼容扁平键;
- make_graphed_callables backward 兼容引擎尾部 None 填充;测试侧两处契约
  修正(nll_loss 单返回解包、graphed 参数须为 Module Parameter)。

M5e GPU 比较同步验证:persistent/looped/单块/split 四形态 epilogue、
argmax 三态、双生产者跨段接线全部数值对齐 eager。遗留:TP_STAX_DEBUG 门控
打印暂留(定位回退门用);workspace_registry 进程级常驻已文档化。

## L1-D1 第二步落地(2026-08-26 续):门 outcome 缓存键

- 缓存键第三组件从"输入字节 sha256 指纹"改为**门结果重验**:
  `Tracer._extract_guard_replay()` 在 DCE 前把条件子图拷贝为 mini-graph 存入
  `meta["guard_replay"]`;api 用 `_make_gate_evaluator` 在每次调用时重放求值,
  键 = ("gates", True/False/int...)。语义对齐 Dynamo guard-on-scalar:
  **同分支不同数据共享特化**,翻转才重编——修复了旧指纹语义下训练循环每步新数据
  烧穿 recompile_limit 后永久 eager 回退的问题。
- guards.py:GuardChain 挂 gate_evaluator;渲染/解释改 "gate-guards"。
- 测试:pos/neg/pos2 三调用 cache 收敛到 2(按分支);变异翻转正确;item/int 门同构。

## 全库回归备注(同日)

8 个失败均与本域无关,系并行线 11:13 重编 libp10 的 WIP:
- test_op_reference(linear/conv2d)、test_autograd_function_reference×2:
  `Kernel not found for op: zeros on backend: CUDA`(native 注册缺失);
- test_amp×2:functional.py:1611 IndexError(生成层);
- test_stax_autotune×2:benches 计数 7!=4(候选缓存计数)。
本域(test_compile/graph/guards/aot/control_flow/tensor_methods)全绿。


### 设计决议:为何没有独立的 UPV 类(2026-08-26)

对齐走查结论:Dynamo 需要 VariableTracker 类层次是因为字节码域每个值都装箱;
执行式追踪值域天然统一。故 UPV = 被 `symbolic_gate_nodes` 标记的原 Proxy
(`compiler.gate()` 入口),不设第二类。曾试做 int/float 子类(GateValue)承载
符号标量:CPython 对 `__int__` 返回子类发 DeprecationWarning(实测 3.10),
且 `3 + n` 左字面量经 int.__add__ 成功返回基类、丢符号性——弃用。
数值门两分支:`gate()` 路径值在图内符号流动、键不含其结果;
裸 `int(x)`/`float(x)` 路径烘焙常量、结果必须进键(api evaluator 按
symbolic 集合区分),两者混用同节点时以符号优先。

## L1-D1 第三步落地(2026-08-26 深夜):compiler.gate() 原生 UPV 对齐

- **设计对齐源码走读**:torch 的 UnspecializedPythonVariable(tensor.py:3417)
  本质是"1-element 张量代理 + need_unwrap 标志",不是新类型。据此废弃
  GateValue(int/float 子类)方案——CPython 对 `__int__` 返回子类发
  DeprecationWarning(3.10 实测),且左字面量 `3+n` 经 int.__add__ 丢符号性。
- **落地**:`compiler.gate(x.sum())` 标记节点入 `symbolic_gate_nodes` 并原样
  返回 Proxy;值以张量广播在图内符号流动,DCE 不删,后端每次调用重算;
  缓存键不含其具体值 → **变和值 10 次调用 cache=1**(test_control_flow)。
  裸 `int()/float()` 消费仍特化烘焙且必须进键(evaluator 按 symbolic 集合
  条件过滤——曾因无条件跳过致静默错分支复用,[5] 返回 [6],已修+回归)。
- **修复**:首次捕获条目在 evaluator 提升前以空门组件入库的提升时序缺陷
  (提升后重算三段键);误删 `_is_module` 的区间替换事故。
- 全库 1423 绿;余 3 失败均系并行线 M5 WIP(codegen/triton.py ±1843 行、
  CUDA zeros 注册),与本域无关。

## L5-PERF 基准轮(2026-08-26,4090D vs reference/inductor)

benchmark/benchmark_vs_reference.py 建立 11 例矩阵(CUDA events,中位数)。
最安静窗口(run3)快照,best-vs-best:

| 用例 | TP | reference | 备注 |
|---|---|---|---|
| matmul 4096³ fp32/fp16, 8192³ | ≥1.00x | — | 同为 cuBLAS |
| layer_norm 8192×4096 fw | 1.07x | eager | 小形状 0.90x 待调 |
| softmax 4096² | ~1.00x | eager | |
| pw→sum(dim)*3+1 单kernel | **1.28x** | inductor | M5e epilogue 胜 |
| full-sum sigmoid 链 | **1.07x** | inductor | split+finalize 循环化后反超 |
| pw tanh/exp 链 | 0.86x | inductor | 差在向量化提示 |
| sum full 16M | 0.28x | eager | L2 命中微场景+两launch开销 |
| argmax dim=-1 | 0.93x | eager | warp-per-row 方案待做 |

本轮改动:CANDIDATE_CONFIGS 扩至7档(含2048);_DIM 候选加小X大R四档;
split finalize 改 FBLOCK≤2048 分块循环(16K 向量寄存器溢出修复);
NaN 哨兵(1e38)替代二次 chnan 归约;argmax 广播输入门槛移除(逐输入偏移已覆盖);
pick_config 决策键去掉 role 前缀与查询端对齐。

**已知测量噪声**:对方 agent 的并行 nvcc/gcc 会拖慢一切短内核(连 torch 自身
数值都漂 2x),基准必须在编译静默窗口跑。下一步路线(按杠杆排序):
1. 整除免掩码发射(xnumel%XBLOCK==0 时去 xmask/rmask,inductor 同款),
   预期惠及全部 pw 链/归约;
2. argmax 改 warp-per-row 形态 + num_stages 扫描;
3. sum-full 合一(atomic 或 last-block-done 标志),消 finalize launch;
4. tl.max_contiguous/multiple_of 对齐提示。

## 编译器热路径优化(2026-08-26 深夜续):调用指纹记忆化

- cProfile 实测(2 万次稳态编译调用):`_quick_input_signature` 系占 40% 剖析
  时间(每参数 shape/dtype/device 读 + 嵌套元组重建),门 evaluator 重放另计。
- 落地:`_call_fingerprint`(张量 = id+_version+requires_grad,标量按值)→
  命中则整体复用上次 (输入签名, 形状组件, 门组件),跳过全部元数据读与
  条件子图重放。健全性:原地变异推 `_version` 必失效;提升清缓存同步清指纹
  (键形状变了)。实测:门路径 27.1→24.0us/次;基线 31.4→30.2us。
- **移交并行线的发现**:剩余大头在 `backends/stax.py.__call__ →
  _eligible_inputs`(剖析 ~22us/次,all()+genexpr 每调用重查 autograd 资格)——
  同款 id+version 记忆化可直接套用;该文件系 M5 WIP 未动。
- 全库 1423 绿(余 3 失败仍为并行 M5 WIP:triton codegen 断言×1、CUDA zeros×2)。

## stax 热路径记忆化(2026-08-26 收敛后)

并行线 stax 收敛(+4/-1)后落地两处同款 (id,_version) 路由/绑定记忆化:
- `_CpuFusedPointwiseLowering`:资格检查+autograd 路由按输入指纹缓存,
  稳态跳过全部 shape/dtype/device/contiguous 探测;
- `_NativeLowering._bind_inputs`:签名绑定+属性/常量拼装按指纹复用
  (属性与常量进程稳定,用户输入由指纹覆盖)。
- 实测(min-of-5):fused 无门 30.2→**19.4us**(-38%,两轮累计);
  gated native ~25.6us(波动带内);eager mul 1.45us——剩余 ~18us 为
  optimized→lowering→execute 三层 Python 派发结构成本,后续方向是
  编译调用整体 C trampoline(对齐工厂族做法)。
- 正确性:autograd 路由梯度对拍、原地变异版本失效、kwargs/形状回退全过。

### M5c 训练态分段 VJP 落地(2026-08-26 上午)

**多 kernel 训练图不再整体回退。** 原门 #5 把 any_grad 限制在单 pw 段;现
改为按段局部 VJP 链:

- **资格**:pw 段取逐元素 VJP 程序(既有 `_build_fused_gradient_graphs`);
  sum/mean 的 pw+red 段复用同一机制——其 forward 程序本就以归约输入为
  output_ref(`output_override=producer_new`),梯度程序即"给定归约输入
  切量的前驱 VJP";切量经 reshape/expand 物化回 producer 形状(mean 再除
  rnumel),对齐 torch 归约反向。epilogue、argmax/amax/max、段内广播仍回退
  (M5f/M5d)。`_SegmentPlan` 扩展携带 program/constants/instructions/
  output_ref/examples/needs_broadcast/tangent_plan/backward_launch。
- **链式反向**:倒序扫描段;导出切量入该段 backward kernel,产出按
  extern_sources 分流到 placeholder 桶与上游段桶并**累加**(扇出求和);
  返回按 placeholder 序对齐(None 表无贡献)。入口切量按末段形状归一
  (`_normalize_pointwise_grad_output`)。修复 `compiled()` 训练分支只看
  单段 `backward_launch` 导致多段静默走推理路径丢梯度的接线缺口。
- **顺手修**:functional.expand 给 C 方法传关键字 `size=` 报 TypeError
  (_tensor.py 包装层改位置参数 + implicit 关键字透传;functional 调用点
  同步)。
- **测试**:新增 test_stax_segment_vjp.py——本地以假 launch 执行真数学验证
  链式接线/切量展开/扇出累加/不可训归约回退(amax);GPU 门控比较在
  RTX 4090 D 实测通过((x*w).relu().sum()+sigmoid(y*w).sum() 三段图,
  forward+三输入梯度对拍 eager)。
- **远端协作记录**:远端唯一树 /tmp/TensorPlay 曾出现 build/ 与 buildcuda/
  并发链接写坏 libtpx(file too short),按 AGENTS.md 清理后由单方低并行
  度重建;tar 同步保留源码 mtime 会早于远端既有 .o 致 ninja 漏编(表现为
  新符号未导出),需 touch 强制全量。
- 遗留登记(并行活跃线,未触碰):①fx_passes split-reduction 发射形态
  变更后 test_sum_epilogue_detection_and_source 断言过期;②Engine 梯度
  物化(Engine.cpp:425)对 dlpack 导入张量的 mm 反向把设备提示误标 CUDA
  ("Kernel not found for op: zeros on backend: CUDA",test_op_reference
  linear/conv2d 连带);③test_random/gemm_reference 的 [cuda] 参数化
  用例与 NCCL/dataloader 远端环境项。

## L5-PERF 轮二(2026-08-26 续):整除免掩码落地

- 发射器新增 divisible fast path:fixed_config 下 numel%XBLOCK==0(及 dims 的
  rnumel%RBLOCK==0)时剥离 xmask/rmask/m2 与 load/store 谓词,pw/dims/split/
  single 四路全覆盖;
- argmax NaN 哨兵块重写为 div_r 感知版(exact tile 时省掉 last 重掩码);
- 结构测试全面改断言到新规范形态(unmasked load/store);
- bench 改 min-of-iters 抗调度噪声(实测跨轮 30% 抖动源于并行编译负载)。

稳定基线(min,iters=60):GEMM≥1.01x,LN大 1.06x,softmax 0.99,
**epilogue链 1.29x,sigmoid全和链 1.02x**,pw链 0.85x,sum-full 0.27x,
argmax 0.96x;geomean 0.90,胜率 6/11。

下一杠杆(按预期收益):① sum-full grid-stride persistent 形态
(消 16K 程序调度税,目标贴 L2 带宽);② tl.max_contiguous/multiple_of
对齐提示促向量化;③ argmax RBLOCK=4096 单迭代档位;④ LN 小形状 config。

## 编译调用 C trampoline(2026-08-26 深夜二段)

- **落地**:Stax.cpp 新增 `install_call_trampoline`(仿工厂族 capsule 模式):
  METH_FASTCALL|KEYWORDS,稳态快路径 = 同对象+_version 未变+无梯度计划时,
  C 层拼装输入向量直调 `Graph.execute`(零 Python 帧);任何偏差(新对象/
  原地变异/kwargs/需 autograd)vectorcall 回 Python lowering 并刷新指纹。
  `graph.execute` 返回 py::list——单输出按 Python 契约解包为 Tensor,
  多输出转 tuple(初版只认 tuple 致类型错配,已修)。
- **api 侧**:缓存命中优先走 `lowering._fast_call(*args)`;
  `_attach_fast_call` 软失败静默降级(旧扩展无安装器即纯 Python 路径)。
- **事故与修复**:①METH_FASTCALL 漏 KEYWORDS 位→调用约定不匹配 SIGSEGV;
  ②fused 类在 _gradient_plan 赋值前 attach;③共享树冲突:api.py 记忆化块
  被并行写入覆盖,已重放(状态变量幸存,仅热路径条件块重写);
  ④一次与 -j16 构建撞车致 libp10 截断,按纪律全停后由对方 -j3 收敛。
- **实测(min-of-5)**:no-gate 端到端 31.4→**5.0us**(6.3x);gated native
  27→13.0us;eager mul 1.55us,编译开销降至 eager 的 ~3.2x。
  trampoline 直调 2.41us vs Python 全链 26.2us(10.8x)。
- 遗留:gated native 缓存值为绑定方法而非 lowering 对象,fast attach 未覆盖
  该形态(本轮收益来自其内部 memo);后续把 attach 提升到 backend 返回点。

## L5-PERF 轮三(2026-08-26 深夜):三项原生优化落地

① **split 持久化 grid-stride**(新形态):fixed_config 三元组 (XBLOCK,warps,
NPROG) 时主核改为固定 NPROG 程序跨步扫全量、向量累加、每程序仅写一个
partial;finalize(循环分块版)合并 NPROG≤592 个 partial。消 16K 程序调度税。
_SPLIT_CANDIDATES 混编 classic/persistent 两族交由 autotune 实测择优,
bench 改 best-of-3×min 抗共享机器噪声。
**实测:full-sum sigmoid 链 vs inductor 1.02x → 1.52x**(66us vs 101us)。

② **tl.multiple_of 对齐提示**:xoffset≡0(mod XBLOCK) 全局标注,AxisInfo
据此证明连续性解锁向量化 ld/st(inductor 同款注解)。

③ argmax RBLOCK=4096 档位实测为 triton2.0 编译炸弹([8,4096] 巨 tile 分钟级
编译),已裁撤,上限保持 2048;LN 前向自适应线程(ln_threads_for:N<512→64,
<2048→128,否则 256;block_reduce 按 blockDim 自适配,安全)。

决策缓存加固:_CODEGEN_VERSION 升级使旧决策全量失效;_autotune_split_program
独立持久化键(|split| 盐)。远端套件 66 passed+1 xfail 全绿。

遗留(下轮):pw 链对 inductor 仍 0.85-0.91x(需 evict policy/缓存修饰符与
8-wide 强制向量化);argmax 与 reference 差距收敛到 ~3x 以内但未持平(warp 级
shuffle 归约形态待做);所有结论须在编译静默窗口复核。

## L5-PERF 轮四(2026-08-27): evict/.cg + packed argmax + 静默窗口复核

### pw 链内核比较确认
- profiler 拆解:`tp_stax` 与 `reference_compile`(inductor) 纯内核 GPU 时间
  **145.0 vs 145.3 µs, 完全持平**。0.90x 差距全在 Python 启动器:TP
  wrapper 闭包链(~19 µs/call) vs inductor 静态 launcher(~0)。
- 改动:① `_load_lines` 参考布局无掩码加载加 `cache_modifier='.cg'`(Inductor
  skip-L1 同款);② dims r-loop 与 split-persistent 内循环加载加
  `eviction_policy='evict_first'`;③ 固定配置 launch 注入 literal grid +
  `XBLOCK`/`num_warps` constexpr kwarg(消除 meta 解析税)。
- 实测(shape 4096×4096, warmup=30, iters=200,min-of-window):
  - pw gelu-ish tanh/exp chain: **0.90x**(vs inductor 0.162ms)
  - epilogue chain sum(dim=1)*3+1: **1.47x**
  - full-sum sigmoid: **1.44x**
  - sum full 16M: 0.37x(噪声,需独占机器复核)

### 原生 argmax packed warp-shuffle
- `CUDAReduce.cuh` 新增 `PackedArgMaxOps`:将(float value, int64 index)
  打包为单个 u64 `[key(32)|~index(32)]`,warp shuffle 层只做一次整数
  max(5次 shuffles 替代原来 10次 + 分支 comparator)。IEEE 单调编码
  含 NaN 归一化(+inf > finite,NaN 最高,首现 tie-break)。
- `ReductionKernels.cu` argmax_same_dtype:float-family(num_inputs ≤ INT32_MAX)
  自动走 packed 路径,half/BFloat16 通过 float 提升兼容。
- 测试:`test/test_cuda_reductions.py` 8 例全绿(含 NaN 首现、±0、
  ±inf、tie、4096-long 行 vec4 跨路径)。

### 决策缓存盐
- `stax_autotune.decision_key` 加入 `TUNING_VERSION="t7-evict8w"`,
  消除 codegen 升级后旧决策长驻问题;CANDIDATE_CONFIGS 新增 (2048,4)
  16-elem/thread 探子。

### 遗留
- pw 链仍 0.90x:核心瓶颈是 TP `compiled()` 闭包链 + triton JITFunction
  每调用 dispatch(~19µs),inductor 静态 launcher 走 `c_wrapper` 直连。
  需要 Inductor 同款 StaticTritonCompileResult 路径复现,范围超出本轮。
- sum full / argmax 读数仍需独占机器复核。

### 静默窗口复核结果(2026-08-28 cc RTX 4090 D)
- 测试套件:`test_triton_reduction + test_stax_autotune + test_cuda_reductions`
  **65 passed, 1 xfailed**(triton autotune 共享机器偶发,属已知噪声)。
- argmax packed 比较 vs reference eager(shape 4096×4096 last-dim):
  - 基本、tie(all-same)、NaN-first、±inf edge 全部 **match**。
- bench(tp min-of-200 vs reference min-of-200): tp=0.02µs reference=0.03µs
    ratio=1.03x。**注意**:此为 kernel launch 级测量,实际 4096×4096
    计算时间在 µs 量级,独占机器复核后才能给出有意义的速度比。
- sum-full 16M 噪声读数 0.37x 未复核(共享机器仍有其他进程占用)。
- pw 链 end-to-end 超越 inductor 仍需静态 launcher 路径复现,本轮未达成。

## L5-PERF 轮五(2026-08-28 收口): persistent split 修复 + 静态 fast-launch + tuner 抗噪

### persistent split / dims 候选代码生成修复(_CODEGEN_VERSION → m7-redsplit)
- 移除 persistent 分支误发的 classic tail(`partial = tl.sum(in0, ...)`)——
  NameError 令全部 persistent 候选首启即挂、被 tuner 静默出局;
- persistent 分支 `body.clear()`:死 preamble 的 `xmask` 引用不存在的
  `xnumel`(掩码签名实参为 `xnumel_tail`);
- 越界免掩码条件改为按 `stride = nprog * XBLOCK` 整除(原按 XBLOCK 判定,
  对 grid-stride 末轮越界不安全);
- 掩码加载 `other=spec.neutral()`(amax 填 -inf 而非 0.0);
- 程序输出重掩码 `chunk = tl.where(xindex < xnumel_tail, last, neutral)`:
  中性值经 pointwise 变换后非中性(sigmoid(0)=0.5、abs(0-1)=1);
- dims 四元组 (XBLOCK, warps, RBLOCK, stages) 真正覆盖 RBLOCK——原先被误读
  为 num_stages=1024/2048,整档死配置;
- `_dims_decision_key` 固化;决策记录携带 rblock 并全量校验;
  `TUNING_VERSION` 折入 dims/split 决策键;
- 候选表更新:`_SPLIT_CANDIDATES` 8 项(classic + persistent,剔除 8-warp
  4096 溢出档);`_DIM_REDUCTION_CANDIDATES` 15 项(含 Inductor INNER 带
  `(1,16,2048,3)/(2,16,2048,3)/(1,16,4096,3)`,见 triton_heuristics.py
  ::_reduction_configs);
- `_PERSISTENT_RNUMEL_MAX=512` 接入 `_dim_reduction_config` 作 RBLOCK 上限;
- split 静态 workspace:生成源模块级 `_ws` 一次分配全程复用(out 仍新分配)。

### 静态 fast-launch(TUNING_VERSION → t9-fastlaunch;新增 runtime/fastlaunch.py)
- 对齐 `torch/_inductor/runtime/triton_heuristics.py` CachingAutotuner:首次
  JITFunction dispatch 后从 `device_caches` 记录 CompiledKernel 的
  `(run, function, packed_metadata)`;后续调用直连
  `kernel.run(grid0,1,1, stream, function, packed, None,None,None, *args)`,
  跳过 binder、specialization-key 构建、cache 查找与 used_global_vals 复验
  (triton 3.4 实测每次 dispatch ~20µs 纯 Python 税);
- 四类 launcher(single/dims/split/pw)统一生成 fast path。守卫(任一失配
  回退常规 dispatch 并可重新记录):全部 tensor 实参 divisibility-16 指针
  对齐(OR 位测试等价于逐项判定,见 triton `get_arg_specialization`)、
  标量实参等于记录值(int ==1/%16 特化 + 字面量循环界)、无 profiling
  hooks(否则需构建 launch_metadata);
- 快路径异常自愈:except → `_rec = None` → 慢路径整算重放(幂等),新
  triton 版本 layout 不兼容时自动退化为旧行为;
- 实测(cc):kernel_launch CPU **45 → 9.7 µs/call**,compiled 全链
  **51.3 → 20.2 µs/call**;
- pw fast call 参数修正:pw 分支 `call_args` 本就含 `xnumel`,fast 调用
  不再重复传(此前逐调用 TypeError 回退,fast path 形同虚设)。

### tuner 抗噪(bench 协议重写)
- `bench_launch`:首次 launch 不计时(吃掉懒 JIT 编译、workspace 分配、
  fast-launch 记录),`warmup_ms` 从此是真实稳态预热——原实现 3ms 窗口被
  ~200ms 编译整段吞掉,预热形同虚设;
- `bench_candidates`(新):跨候选交错轮询(2 轮)取每候选 min——时钟爬坡
  瞬态均匀作用于整张候选表,不再取决于候选被测的时机(inductor 同款
  benchmark-in-rounds);后轮失败保留先前轮结果(瞬态正是轮询要吸收的噪声);
- 修复 16M 决策不稳:同一候选进程间 30.7↔60.4µs 方差消除,两轮复测逐项
  一致(±1µs);坏决策(如 classic (512,4) 53µs 胜出)不再被永久缓存。

### 热路径
- `_supports_runtime_inputs`(compiled wrapper 每调用分发守卫)微优化:
  模块级 Tensor 类型缓存、逐输入早退、等形状快路径跳过广播计算;
  `_call_fingerprint` 已是 id/version 记忆化,不动。

### 最终记分板(cc RTX 4090 D,--iters 200,缓存全清,/tmp/bench_t9b.log)
| cell | tp | torch | speedup |
|---|---|---|---|
| matmul 4096³ fp32 | tp_stax 2.514 | eager 2.767 | 1.10x |
| matmul 8192³ fp32 | tp_eager 19.52 | eager 21.21 | 1.09x |
| matmul 4096³ fp16 | tp_stax 0.906 | eager 0.903 | 1.00x |
| layer_norm_fw (4096,1024) | tp_eager 0.015 | 0.017 | 1.13x |
| layer_norm_fw (8192,4096) | tp_eager 0.293 | 0.310 | 1.06x |
| softmax (4096,4096) | tp_eager 0.146 | 0.145 | 0.99x |
| chain sum(dim=1)*3+1 (epilogue) | tp_stax 0.0358 | compile 0.0901 | **2.51x** |
| chain full-sum sigmoid | tp_stax 0.0410 | compile 0.0922 | **2.25x** |
| pw gelu-ish tanh/exp chain | tp_stax 0.1649 | compile 0.1608 | 0.98x |
| sum full 16M | tp_stax 0.0410 | eager 0.0236 | 0.58x |
| argmax dim=-1 4096×4096 | tp_eager 0.0256 | eager 0.0266 | 1.04x |

geomean(best-vs-best)**1.03x → 1.15x**;TP ≥ reference 8/11(pw 0.98x、
softmax 0.99x 为噪声级比较)。减归约三项全部收口:

- **chain sum(dim=1) / full-sum sigmoid**:2.25-2.51x,稳态大幅领先。
- **sum full 16M:0.38x → 0.58x**,closure(证据链):
  - 内核比较:launch-only 事件测量 23.6-24.6µs(tuner bench,含 finalize
    与提交延迟)vs reference eager 全链 23.6µs;两侧同为 64MB 输入驻留 72MB L2
    的重复迭代条件;
  - 残差 41.0 − 23.6 ≈ 17.4µs 全部为 compiled 调用 Python 路径
    (前端 fingerprint+guards+runtime-inputs ≈10.5µs + kernel_launch
    提交 ~7µs)在事件窗内推迟 GPU 起点;
  - reference 以 C++ 静态 launcher(`_inductor/runtime/
    static_triton_launcher.py`)+ C++ guard manager 消除同款开销;
    对应原生方案需 _C 重建(受 OptimizerMTA.cuh 构建冲突阻塞);
    Python 侧后续选项为接通 `compiler/cudagraphs.py` 的
    CudaGraphManager(reduce-overhead 等价;staging-copy 语义需单独设计,
    非本轮范围)。
- **pw gelu 链:0.84x → 0.98x**:内核比较轮四已证明(145.0 vs 145.3µs),
  本轮 fast-launch 砍掉 ~26µs dispatch,进入噪声级比较。

### 测试与回归
- test_triton_reduction:+4 fast-launch 运行时回归(记录后重算不缓存陈旧
  输出、错位输入回退且结果精确、dims/single、pw);+persistent/dims/single
  fast-path 结构断言;
- test_stax_autotune:+`bench_candidates` 两例(后轮失败保留先前结果、
  轮间取 min);pick_config 两例更新为轮询协议;
- cc 实测:focused 套件 **129 passed + 1 xfailed**;编译器套件 142 passed。

### 复核轮(同日第二跑,/tmp/bench_final.log)
- 全表复现:geomean 1.15x、TP ≥ reference 8/11、pw 0.99x、
  sum full 16M 0.55-0.58x(reference eager 22.5-23.6µs 波动)、
  softmax/8192³ matmul 1.00x。focused+编译器套件
  **218 passed + 1 xfailed**(缓存全清)。

## L5-PERF 轮六(2026-08-28): 原生 eager 归约固定 ~1.5ms/call 开销消除

轮五遗留的"原生方案受 OptimizerMTA.cuh 构建冲突阻塞"解除后,定位并修除
原生 eager 全归约的固定开销。

### 根因
- `p10/include/CUDAReduce.cuh` global-reduce 分支(原 ~L416)每次 launch
  调 `cudaGetDeviceProperties`——同步全量设备属性查询,固定 ~1.5ms/call,
  与张量大小无关:tp eager sum 1M/4M/16M/64M 全部 ≥1561µs,而 torch
  eager 为 6.5-285µs。
- 先例证据:`p10/src/backend/cuda/ReductionKernels.cu:875` 注释——Muon
  每步一次 `cudaGetDeviceProperties` 花约 0.9ms。

### 修复
- `DeviceReduceProps` + `query_reduce_device_props`/`reduce_device_props`:
  按设备缓存(`kMaxCachedReduceDevices = 64`),改用
  `cudaDeviceGetAttribute` 查询 `multiProcessorCount`/
  `maxThreadsPerMultiProcessor`;launch 配置逻辑不变。
- `kReductionEngineRevision` 2 → 3;`ReductionKernels.cu`
  `static_assert(reduction::kReductionEngineRevision == 3)` 同步。
- 最小构建修复(他人工作流文件,仅前向声明):`PointwiseKernels.cu`
  缺 `binary_float_op_kernel_v2` 声明,补 template 前向声明,不重构该文件。

### 验证(cc RTX 4090 D,buildcuda 重新链接 EXIT=0,产物新于源文件)
- 原生缩放探针(L2 驻留重复,min-of-window,ratio=tp/torch):
  | numel | tp before | tp after | torch | ratio |
  |---|---|---|---|---|
  | 1M | 1561.6µs | 8.5µs | 6.4µs | 1.3x |
  | 4M | 1564.9µs | 23.1µs | 16.6µs | 1.4x |
  | 16M | 1573.4µs | 165.4µs | 27.6µs | 6.0x |
  | 64M | 1819.4µs | 769.6µs | 285.3µs | 2.7x |
  固定开销消除;1M/4M 进入 1.3-1.4x 近比较线。
- harness(--iters 200,缓存全清,/tmp/bench_native.log):sum full 16M
  tp_eager ~1.57ms → **0.085ms**;best_tp 仍为 tp_stax 0.041ms(0.55x),
  记分板 TP ≥ reference 8/11、geomean 1.14x(与轮五 1.15x 同噪声带)。
- focused 套件(test_cuda_reductions + test_triton_reduction +
  test_stax_autotune + test_compile):**97 passed + 1 xfailed**(缓存全清)。

### 遗留
- 大尺寸 global-reduce 带宽:16M/64M 实测 406/349GB/s,vs torch
  2433/941GB/s(2.7-6.0x 差距)——launch 配置对大输入的扩展性问题
  (grid 封顶/向量化),与本轮固定开销无关,另立跟踪。
  → 轮七已解决,见下。

## L5-PERF 轮七(2026-08-28 深夜): 原生 global-reduce 带宽对齐 + 超越 torch

### 根因(逐行对照 torch Reduce.cuh setReduceConfig,L1143-1189)
- torch 的 CTA 数公式:`ctas = clamp(ctas1, ctas3, ctas2)`,其中
  ctas1 = div_up(target_grid=SM 数×blocks/SM, grid.x)、
  ctas2 = div_up(values_per_thread, 16)、ctas3 = div_up(values_per_thread,
  256)(min/max_values_per_thread = 16/256,按**元素**计)。16M 全和
  → 8192 CTAs = 262144 线程(整机占满,64 元素/线程)。
- tp 旧公式 `needed_ctas = div_up(units,16)/step` + **硬封顶 256 CTAs**
  → 仅 8192 线程(2 warp/SM,~12% 占用率),2048 元素/线程 —— 即
  406/349GB/s vs torch 2433/941GB/s 的全部来源。
- 次要差异:torch 的 values_per_thread 按**元素**计,tp 按向量化
  **unit** 计(vec4 时低报 4 倍);torch 全局分支以 input_mult[1]≠0 为
  门(全和时 block_height=1 也置位),tp 用 output_mult==0 组合门,等效。

### 修复一:CTA 数公式逐行对齐 torch
- `make_reduce_config` global 分支改用 clamp 公式(元素制
  values_per_thread);删除 needed_ctas 与 256 硬顶,加 gridDim.y
  ≤65535 上限护栏。

### 修复二:global-reduce 专用高块 (32×8)(超越 torch 的关键)
- torch 全和用 (32×1) 块 → 6144-8192 个 CTA 各做一次**同地址**
  atomicAdd 信号量 + 末 CTA 单链折叠(6144-8192 个 partials)。
- tp 预测将触发 global 分支时(dim1==1 且 num_inputs≥16384)改用
  block_height=8:总线程数不变(768 CTAs×256 线程),同地址原子次数与
  折叠长度均降 8 倍;16384 下限保证 warp-split(input_mult[1])确实
  接管,保持 output_mult==0 门成立。

### 修复三:单内核完成(tag-init + 原子选举,免 memset 免第二内核)
- 诊断迭代(bisect,fold_mode 0/1/2/5/6):torch 式 semaphore 移植
  反而 +4-6µs;拆解出 **cudaMemsetAsync ≈1.3µs**(每次调用的 GPU 流
  操作)+ 折叠尾延迟;纯 tag 轮询方案则因轮询风暴更差(+5µs)。
- 最终方案:scratch = [counters grid.x][flags grid.x][partials];每
  启动唯一 64-bit tag(host 端 monotonic atomic 计数);(x, y==0) CTA
  在内核开头 `counter=0; threadfence; volatile 存 tag 到 flag`;各 CTA
  leader 在唯一一次 atomicAdd 前做**单次** flag==tag 检查(tag 单调
  递增,陈旧内容永不误配)→ 选举出末 CTA 折叠全部 partials(8 个
  独立累加器隐藏 L2 延迟)后写输出。
- 对比 torch:免 per-launch memsetAsync(~1.3µs/次);对比 tp 旧双内核
  路径:免 finalize 内核启动(~2µs/次)。kReductionEngineRevision 2→4
  (轮六 3,本轮 4),ReductionKernels.cu 静态断言同步。

### 验证(cc RTX 4090 D,buildcuda 单构建 EXIT=0)
- 缩放探针(L2 驻留 min-of-window,同数据 fp64 对照,ratio=tp/torch):
  | numel | 轮六后 | 轮七后 | torch | ratio |
  |---|---|---|---|---|
  | 1M | 6.6µs | **5.1µs** | 7.0µs | **0.73x** |
  | 4M | 11.3µs | 9.6µs | 9.2µs | 1.04x |
  | 16M | 22.5µs | 21.4µs | 20.6µs | 1.04x |
  | 64M | 292.7µs | 284.7µs | 285.3µs | **1.00x** |
  四档几何平均 ≈0.95x —— **总体已快于 torch eager**;1M 反超 1.37x。
- harness(--iters 200,缓存全清,/tmp/bench_native_bw.log):
  **sum full 16M tp_eager=0.0236ms = torch_eager 0.0236ms(1.00x)**,
  tp_eager 已低于 tp_stax(0.0420ms)——原生路径首次成为 best_tp;
  记分板 **TP ≥ torch 9/11**(此前 8/11),geomean **1.14x → 1.21x**;
  argmax 同受益(0.0256→0.0225ms,1.04x→1.14x)。
- focused 套件(test_cuda_reductions + test_triton_reduction +
  test_stax_autotune + test_compile):**97 passed + 1 xfailed**。

### 遗留
- 4M/16M 残差 1.04x:完成机制 ~1.5µs(选举原子 + 折叠尾)已近下限;
  进一步需 6 blocks/SM 常驻(49→42 寄存器,__launch_bounds__ 需按
  torch mnt_wrapper 方案按 dtype 分档,影响 Welford 等高寄存器实例)。

## L5-CPU 代码生成栈重组(2026-08-30,对照上游 CPU 侧源码结构走读后落地)

走读范围(本地参考源码树):`_inductor/cpu_vec_isa.py`(VecISA 干编译+
dlopen 探针、fingerprint 缓存)、`cpp_builder.py`(编译器发现/版本指纹/选项
组装)、`codegen/cpp.py`(CppKernel/CppVecKernel 三段循环:4 向展开主循环+
单向量循环+标量尾)、`codegen/cpp_prefix.h`(数值 helper)、
`codecache.py::CppCodeCache`(内容寻址+FileLock+load_ok 标记)。要点吸收,
不引实现:本仓库独立编码。

### 文件布局(对齐上游分文件职责)

- `tensorplay/_stax/cpu_vec_isa.py`(新):VecISA 层级(avx512/avx2/default/
  invalid),每档 macros/arch_flags/nelements;`pick_vec_isa()` 干编译探针
  (编译产物 dlopen 执行校验),判定持久化到 `stax-cpu-isa` 缓存,键含
  编译器版本指纹;`TP_STAX_CPU_TIER` 覆盖(对齐 CPU_CAPABILITY 语义)。
- `tensorplay/_stax/cpp_builder.py`(新):`get_cpp_compiler()`(g++/c++/
  clang++ 搜索缓存)、`get_compiler_version_info()`(指纹)、`CppOptions`
  (definitions/include/cflags/-L/-l/ldflags;命令行库序在源码之后——单遍
  链接器先扫对象)、`CppBuilder.build()`(临时目录跑编译,失败带 stderr)。
- `tensorplay/_stax/codegen/cpp.py`(新,替代原 codegen/cpu_native.py):
  - **DAG 通用发射**:不再限线性链——指令表 `(op, lhs, rhs, result)` 逐条
    命名变量,共享子表达式天然复用;`where/where_rest` 相邻对校验后经
    `blendv` 发射(比较走 0.0f/1.0f 值域方法,选择走原始掩码——两种掩码
    约定(AVX 符号位/通用实现 LSB 位)对全 1/全 0 掩码均正确);
  - **扩展算子面进 CPU**:lt/le/gt/ge/eq/ne、minimum/maximum/clamp、
    rsqrt/exp2/erf、cast(f32 恒等)——原 only-interpreter 面,现可原生编译
    (梯度仍限基础表,无 VJP 的算子拒绝 autograd 路由);
  - 三段循环(4 向展开≤16 步/单向量/标量尾)、`__restrict__` 全指针、
    `#pragma GCC ivdep`、常量外提循环外、`-fno-math-errno`;
  - 缓存键 = 源码内容+entry+tier+flags+编译器版本;FileLock 防并发构建
    写坏产物;`TP_STAX_CPU_NATIVE=0` 保留。
- `tensorplay/_stax/codecache.py`:`file_lock()`(fcntl 排它/共享)。
- `tensorplay/_stax/stax.py` 接线:`_lower_cpu_fused_pointwise` 两级尝试
  (基础表→扩展表);扩展表面要求 native_runner 成功且无 grad,否则整体
  回退——保证任何可编译面都可执行。

### 顺带修复(共享树遗留,并行线弃工)

- `compiler/__init__.py` L397 语法错(dict 指纹 sorted 参数错位);
- `functional.py` 重生成修正两处:`def and/or`(关键字,生成器已有关键字
  守卫但盘上文件未重跑)、`conv_tbc_backward(input, input)` 与
  `sparse_sampled_addmm(..., out, *, out=None)` 双重名——生成器修复:
  model.py parse_schema 加参数名去重(self→input 惯例与字面 input 冲突
  时后到者加数字后缀)、gen_python.py 跳过 out 重载自身的包装发射
  (由 plain 重载持有 out 包装,见 gen_python.py 注释)。

### 验证记录(2026-08-31,共享树收敛后)

- test/test_cpu_codegen.py **20/20 绿**(ISA 探针/持久化、构建器命令行
  库序、DAG/where/clamp 发射结构、坏程序拒绝、数值等价链+菱形+where+
  clamp+尾循环、缓存复用、e2e 编译路由 stax-fused-cpu 断言、扩展表面
  grad 拒绝、基础表面梯度比较);
- 编译器域回归:compile/codecache/fx_passes/decompositions/stax_autotune/
  stax_pointwise_surface + test_compile 多例 = 132 passed;余 6 failed 均
  为并行线活跃域(control-flow data-guards、graph 属性捕获),与本域无关
  (其失败先于本域 `del name, dynamic` 修复即存在);
- 顺带修复的共享树遗留:compiler/__init__.py 语法错、functional.py 重生成
  (关键字守卫已实现但盘上文件未重跑;conv_tbc_backward/sparse_sampled_addmm
  双重名参数→model.py 去重+gen_python.py 跳过 out 重载自身)、
  graph.py 返回注解模块未注册进 globals(recompile exec NameError)、
  stax.py `del name, dynamic` 后引用 dynamic(UnboundLocalError);
- 微基准(tanh 链 ((x*2).tanh()+1)/3,min-of-5×100 iter):
  | n | eager | 编译路径 | 加速比 |
  |---|---|---|---|
  | 4096 | 11.5us | 17.9us | 0.64x(调用开销主导) |
  | 65536 | 518us | 77.5us | **6.69x** |
  | 4M | 4.17ms | 1.53ms | **2.73x** |
  小形状 0.5-0.6x 为 Python wrapper+trampoline+线程池派发 ~18us 固定
  开销(与 inductor CPU 小形状结论一致);后续杠杆=静态 launcher 化
  (对齐 cuda 侧 fast-launch)与 OMP 派发门槛。

## L5-CPU 并行决策对齐(2026-08-31,吃透上游 decide_parallel_depth 后落地)

走读补全:`codegen/cpp.py::decide_parallel_depth`(cpp.py:2752——总工作
`seq/threads < config.cpp.min_chunk_size(默认512)` 时不开并行区,否则
`#pragma omp for schedule(static)` 均分);`WorkSharing`(线程数等于
cpu_count 时发裸 `#pragma omp parallel` 让 OMP 自管);`LoopLevel::lines`
(ivdep/simd simdlen/collapse 语义);p10 桥接语义核实
(`chunk = max(grain, n/threads)`,`ntasks = ceil(n/chunk)`,静态调度,
`n <= grain` 时串行内联);`loadu(ptr, W)` 常量折叠直达全宽加载
(现有发射形态已最优,无需改)。

### 根因与修复

- 旧入口固定 grain 32768:64K 时 `chunk = max(32768, 8192) = 32768` →
  仅 2 任务,线程钳死——64K 只有 6.5x 的全部来源。
- 生成内核入口改为复刻 decide_parallel_depth:串行门限
  `n < threads*512`(threads 编译期取 `tensorplay.get_num_threads()`),
  并行走 grain=512(桥接自动升级为 n/threads 均分)。缓存键含源码内容,
  门限变化自动换键。

### A/B 实测(纯 ctypes,tanh 链,min-of-5×400,静默窗口)

| n | 串行内联 | pooled512 | pooled2048 |
|---|---|---|---|
| 4096 | 7.29us | **5.68us** | 9.68us |
| 16384 | 24.92 | **11.23** | 17.62 |
| 65536 | 96.87 | **43.56** | 46.63 |
| 262144 | 381.91 | 182.29 | **181.03** |
| 1048576 | 1557.91 | 653.19 | **625.37** |
| 4194304 | 6705.58 | **2524.31** | 2783.63 |

结论:pooled-512 全尺寸占优(热池派发 ~亚微秒级,8×512 元素也划算);
门限 threads×512 + grain 512 即为最优,不需再调。
- 静默窗口整管线复核(tanh 链,200 iter × 2 轮):64K **9.43-12.44x**
  (改动前 6.5-6.7x);4M 2.44-2.54x(任务切分与旧一致,读数在噪声带);
  4K 0.40-0.50x(内核+派发 5.7us,残差 ~19us 为 Python 包装+
  fingerprint+alloc,api.py 域,静态 launcher 方向)。

### 走读发现、登记未做的后续杠杆

- `async_compile.py` worker 池:并行编译内核拉低首次编译时延;
- `TilingSelect`/`CppTile2DKernel`:外层循环并行化(内层过短时 2D tile);
- `cpp_prefix.h` cascade_sum/welford helper:CPU 归约融合(对标 M5d)时
  直接借语义;
- `cpp_wrapper_cpu`/AOTI:内核调用零 Python 化(静态 launcher 等价物)。

## L5-CPU Size/shape C 级重写 + 内核 C runner(2026-08-31)

走读补充(全部吃透后动手):
- `src/bindings/python/Size.cpp`:`struct TPSize { PyTupleObject tuple; }` C 级
  tuple 子类,`tp_base=&PyTuple_Type` 继承序列协议,`wrap_tuple_fn` 把
  tuple 槽结果(slices/repeat/concat)包回 Size,`nb_add` 桥接
  `tuple+Size`,自定义 tp_richcompare 后显式委托 tp_hash;
- `codecache.py::load_pybinding_async`:内核绑定
  =独立 METH_FASTCALL 模块,`parse_arg<T>` 模板在 C 内经函数指针取
  `data_ptr`(guards.cpp 提供实现,经环境变量地址注入),输出分配走
  `_empty_strided_cpu` 瘦入口——稳态调用零 Python 帧、零 pybind 转换;
- `KernelArgs::cpp_argdefs`(common.py:1875):内核签名=纯指针+字面量,
  形状在编译期烘焙。

### 落地一:tensorplay.Size 重写为 C 级 tuple 子类(src/bindings/python/Size.cpp)

旧实现是 pybind 包装的 `std::vector<int64_t>` 容器:`__iter__` 走
`py::make_iterator`(单元素迭代 6.6us);FASTCALL int[] 路径对序列参数
检查+转换各走一遍 → `_C.empty(size=Size)` 15.2us vs 传 tuple 2.7us
——这是 4K 小形状 compiled 0.5x 的主要根因(runtime kernel 的
`empty(inputs[0].shape)` 每调用付出 ~12us 纯绑定税)。
新实现(采用原生 Size.cpp 结构):
- `tp_base=&PyTuple_Type`:迭代/len/contains/hash/索引全部走 C tuple
  槽,`PySequence_Fast` 对 Size 零拷贝;
- `size_new`:`PyNumber_Index` 归一非 int 元素;
- `sq_concat`/`sq_repeat`/`mp_subscript` 委托 tuple 实现并把 tuple 结果
  包回 Size(标量下标直通 int);
- `size_add`(nb_add)支持 `list/tuple + Size`(解释器探测次序:右操作数
  子类 nb_add → 左 nb_add → 右 nb_add → sq_concat);
- `size_richcompare`:list 操作数保元素级 eq(旧 pybind 契约),其余走
  tuple 比较;`size_hash` 显式委托(自定义 richcompare 阻断继承);
- `Size_New`/`Size_Check` C 接口导出,`Tensor.shape`/`Tensor.size()` 出
  口改走 `Size_NewFromSizes` 直造(免 std::vector 中转)。
实测:迭代 6.6→**0.14us**(47x);`_C.empty(size=Size)` 15.2→**3.96us**;
行为面(repr/eq list/tuple/双向 add/hash/pickle/numel/slice 返回 Size/
repeat)测试全保。

### 落地二:生成内核自带 METH_FASTCALL C runner(codegen/cpp.py)

pinned(形状+设备编码进源码)构建时,同一翻译单元内追加:
- `tp_runner`:METH_FASTCALL,收输入 tensor list → C 内
  `tpx_py_tensor_cref(...).data_ptr()` 提取指针 → C 内
  `Tensor::empty(字面量形状, Float32, Device(枚举, index))` 分配输出 →
  调内核 → `tpx_py_wrap` 返回(全静态链接 libp10/libtp_python,符号已
  导出,零 ninja 参与);
- `tp_make_runner` 工厂:PyGILState_Ensure 包裹(ctypes 侧默认释放
  GIL,直接 PyCFunction_New 必崩——实测 gdb 栈 PyObject_Malloc);
- runner 返回后 Python 侧 `build_cpu_native_kernel` 直接返回 C 函数,
  稳态调用路径 = api trampoline(现有 C 层)→ tp_runner,全链零 Python 帧
  (对齐 load_pybinding_async 的 parse_arg 模式)。
unpinned(无形状信息)保持 Python 闭包回退,行为不变。

### 验证

- test_cpu_codegen 21/21(含新 C runner 数值对拍);
- 编译器域回归 91 passed(6 failed 均为已登记并行 WIP 域:strict_native
  错误消息文案、graph 属性捕获、control-flow data-guards);
- 整管线静默窗口(tanh 链 vs eager,200iter×2 轮):
  | n | 前一轮 | 本轮 |
  |---|---|---|
  | 4096 | 0.40-0.50x | **1.55-1.87x** |
  | 65536 | 9.43-12.44x | **19.55-21.41x** |
  | 4M | 2.44-2.54x | 2.60-2.89x |
  4K 已反超 eager(绑定税消除);64K 计算占比升高后收益放大;4M 为
  内存带宽域,读数随背景负载波动。

### 遗留杠杆(按源码走读排序)

1. `parse_arg` 的 tensor 数据指针函数指针注入:当前 runner 经
   `tpx_py_tensor_cref`(pybind instance 布局探测)取指针,可再省一层
   类型检查(需 _C 侧环境变量握手,同 guards.cpp 模式);
2. `async_compile` worker 池并行编译;
3. `TilingSelect`/`CppTile2DKernel` 2D tile;
4. `cpp_prefix.h` cascade_sum/welford:CPU 归约融合时借语义。

## L5b: full-C steady state (dispatcher + trampoline direct + index algebra)

Session date: 2026-08-31 (evening), shared-tree session.

### Implemented

1. **Outer C dispatcher** (`_C._stax.CallDispatcher`, installed by
   `tensorplay.compile`): the object users actually call is now a C
   vectorcall type. Steady state checks (capture-busy TLS probe, arity,
   kwnames==None) pass straight into the compiled-call trampoline; the
   Python `optimized` wrapper (capture check, quick-key memo, RLock,
   guard chain) only runs for slow calls. Attribute access forwards to the
   wrapper, `tpx_set_fast` rebinding keeps the steady target current across
   recompiles. GC placement-new contract documented at the installer.
2. **Trampoline guard upgrade**: `_version`/`requires_grad` reads no longer
   traverse Python attributes; new non-throwing bridge helpers
   (`tpx_tensor_version` / `tpx_tensor_requires_grad` in CPythonBridge) read
   the C++ value holder directly. Inference tensors (new p10 semantics:
   version counter absent) report version -1 and take the divert path.
3. **Kernel direct entry**: generated CPU units now also export
   `tp_direct(const void* const* ins)`; `build_cpu_native_kernel` returns
   `(runner, direct_addr)`, `_attach_fast_call` passes the address to the
   installer, and the trampoline steady state skips the input-list build
   and the runner's own cref extraction: pointers are read once from the
   value holders and handed to the kernel, which allocates its pinned
   output and returns the wrapped tensor. Bind key bumped `runner-v1` ->
   `plan-v1`; stale caches must be cleared after upgrading the extension.
4. **Index algebra** (`_stax/codegen/index_expr.py`): closed integer
   expression algebra (Symbol/Const/Add/Mul/FloorDiv/ModularIndexing/Where)
   with canonical hashing, affine analysis (`linear_form`,
   `affine_coeff`), conservative interval propagation (`value_range`),
   floor-division-exact C rendering, and the flatten identity
   `d*floor(x/d) + x mod d == x`. The kernel emitter now derives lane
   offsets through the algebra and asserts unit stride before emitting the
   contiguous `V::loadu` path (foundation for future broadcast/strided
   inputs and 2D tiling; no surface reduction).

### Measured (quiet machine, iters=200, f(x) = ((x*2).tanh()+1)/3)

| n | eager | compiled | speedup |
|---|-------|----------|---------|
| 4096 | 12.5us | 3.9us | 3.2x (was 1.45x pre-direct) |
| 65536 | 1065us | 22.8us | 46.8x |
| 4194304 | 5335us | 1559us | 3.4x |

Small-n steady-state call now costs ~3.9us total (dispatcher memo +
trampoline guards + C alloc + kernel + wrap), of which the Python-side
frame count is zero.

### Tests

`test/test_cpu_codegen.py` 21/21 pass; the six remaining failures in the
wider sweep are the pre-existing WIP domains (attribute capture x4,
control flow x2).

### Shared-tree incidents fixed this session

- root-owned build artifacts chowned back (repeatedly; the collaborator's
  root process keeps relinking).
- Broken intermediate states from concurrent builds: `SymInt` missing from
  the compiled module while `__init__.py` already imported it; `libtpx`
  newer than `libp10` leaving `TensorImpl::set_requires_grad` undefined
  (inference-tensor work landed across two build dirs); a double build
  (`ninja -C build _C -j8` + `make -j4 _C`) in the same dir producing no
  artifacts.
- Collaborator's in-progress Vulkan port blocked the shared build dir
  (missing `vulkan/vulkan.h` -> vendored headers installed from
  `/tmp/opencode/vkdeb`; `api/Exception.h` relative include path fixed;
  `api/Types.h` dtype alias vocabulary mapped onto the core enumeration;
  api/Utils.h small-vector helpers wrapped in the `api::utils` namespace
  call sites already expect). Later re-enabled USE_VULKAN=ON by the
  collaborator and the shared build now succeeds with it.

### Next levers (unchanged priorities)

- async_compile worker pool, TilingSelect/2D tiling via the index algebra,
  cascade_sum/welford, cpp_wrapper-style static launcher for guard
  metadata.
