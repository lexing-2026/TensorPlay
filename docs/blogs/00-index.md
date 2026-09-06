# TensorPlay 核心设计深度解析 · 系列导读

> 一套面向教学与硬件实验的透明深度学习框架，如何在 7.5 万行 C++ 中构建清晰的核心骨架，又在何处做了刻意的裁剪与简化？本系列基于 TensorPlay 1.x 源码（ `p10 / tpx / stax / bindings` 四库）逐文件精读，给出可复现的源码级分析。
>
> **风格说明**：本系列为**知乎风格 · 文字理论优先**——用故事与类比讲透设计思想，代码仅作“证据”出现，每篇配 2-3 张图（AI 提示词 + 本地 Python 一键生成），适合碎片化阅读与收藏。

- 源码基准：TensorPlay `1.0.0rc0`（`version.txt:1`）
- 写作时间：2026-09-02 · 全部结论均可由 `grep -rn` / `read` 复现
- 适用读者：想从“调包”走向“造轮子”的学习者、内核/编译器贡献者
- 配图：每篇文末附 AI 绘图提示词 + `python docs/blogs/assets/gen_*.py` 本地生成（matplotlib，无需外网）

## 系列目录

| 篇章 | 标题 | 核心问题 | 关键源码锚点 |
|---|---|---|---|
| [01](01-architecture-four-pillars.md) | **总体架构与四支柱解耦** | 为什么要把张量、自动微分、编译器拆成三个独立动态库？ | `p10/CMakeLists.txt:28`、`tpx/CMakeLists.txt:26`、`CMakeLists.txt:938-945`、`tensorplay/__init__.py:1-1197` |
| [02](02-tensor-storage-impl-view.md) | **张量内存模型：Storage → TensorImpl → Tensor** | 视图如何零拷贝共享存储？版本号如何防“静默算错”？ | `p10/include/StorageImpl.h:13`、`TensorImpl.h:28`、`Tensor.h:80`、`p10/src/Tensor.cpp:645` |
| [03](03-dispatcher-codegen.md) | **分发器与代码生成：从 yaml 到 C++/Python 的全链路** | 478 条算子注册表条目如何一次生成 6 份产物？ | `config/native_functions.yaml:1`、`tools/codegen/gen_api.py:304`、`p10/include/Dispatcher.h:49` |
| [04](04-tensor-iterator-dtype.md) | **TensorIterator 与类型提升：逐元素算子的“执行模板”** | 广播、类型提升、维度重排如何在 1145 行内完成？ | `p10/include/TensorIterator.h:32`、`p10/src/TensorIterator.cpp:1077`、`p10/include/TypePromotion.h:12` |
| [05](05-autograd-engine.md) | **自动微分引擎：DAG/Engine/SavedVariable** | 为什么没有 `Tape`？`Node` 如何在多设备 ReadyQueue 间调度？ | `tpx/include/Node.h:26`、`tpx/include/Engine.h:23`、`tpx/include/SavedVariable.h:15` |
| [06](06-compiler-stax-graph.md) | **编译器栈：compiler / _stax / graph 三层捕获与 lowering** | FX 式捕获如何与 C++ 静态 IR 衔接？Pointwise 融合如何生成 C++/Triton？ | `tensorplay/compiler/__init__.py:1`、`tensorplay/_stax/api.py:1`、`tensorplay/graph/tracer.py:1`、`stax/include/Graph.h:47` |

## 阅读建议

- **顺序阅读**：01→02 建立“对象模型”心智，03→04 理解“算子如何落地”，05→06 进入“动态图与静态图”两条执行路径
- **结构阅读**：每篇均从数据结构、调用链和构建关系切入，并给出可复现的源码行号
- **动手验证**：文末均附 `grep` / `python -c` 可复现命令

## 约定

- 源码引用格式 `path:line`，如 `p10/include/Tensor.h:80`
- 文中的代码路径均指向本仓库；涉及外部互操作的名称仅保留在实际接口或工具配置中
- 涉及自动生成的产物（如 `build/generated/tensorplay/ops/*`）会同时标注生成器 `tools/codegen/*.py:line`

---

> 下一篇： [01 总体架构与四支柱解耦](01-architecture-four-pillars.md)
