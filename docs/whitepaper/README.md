# TensorPlay 白皮书 — LaTeX 可发表版

**业界最高标准，Windows `E:\texlive` 本地可编译，已全面中文化**

本目录包含 TensorPlay 白皮书的可发表 LaTeX 源码。与高层概述不同，本文档以 `path:line` 援引**具体**机制：13 槽 `DispatchTable`、`TENSORPLAY_LIBRARY_IMPL` 静态注册、7 步 `TensorIterator` 流水线、显式有向无环图（无 Tape）、按 `sequence_nr` 排序的 `ReadyQueue`、`SavedVariable` 版本守卫、偏移 9 的 `Vmap` 键，以及 `OpRecord / GpuTimerPair / Cupti / Nvtx` 剖析器栈。**全文已译为中文**，正文、标题、图表标题、术语表均为中文，代码清单与 `path:line` 援引保持原文。

## 文件结构

```
docs/whitepaper/
├── main.tex                 # 导言 + 摘要 + \input{sections/*.tex}（已配置 ctex 中文支持）
├── sections/
│   ├── 01-introduction.tex  # 问题、核心机制、范围（已中文化）
│   ├── 02-architecture.tex  # 四支柱、CMake 有向无环图、虚接口（已中文化）
│   ├── 03-redispatch.tex    # 分发键 13、分发器 214 行、宏、双绑定（已中文化）
│   ├── 04-profiler.tex      # OpRecord、GpuTimerPair、Cupti/Nvtx/Itt、开销（已中文化）
│   ├── 05-autograd.tex      # Node/Edge/GraphTask/InputBuffer/Engine、SavedVariable、Anomaly（已中文化）
│   ├── 06-vmap.tex          # 偏移 9 的向量化映射、TransformDispatch、BatchingKernels（已中文化）
│   ├── 07-memory.tex        # DataPtr、Storage、TensorImpl、Tensor、MemoryFormat（已中文化）
│   ├── 07b-tensoriterator.tex # 张量迭代器流水线（已中文化）
│   ├── 08-codegen.tex       # YAML 契约、7 个生成器、构建分段（已中文化）
│   ├── 09-evaluation.tex    # 可读性、正确性守卫、裁剪轴（已中文化）
│   └── 10-appendix.tex      # 可复现性、Windows 构建、文件地图（已中文化）
├── glossary.tex             # 术语与缩写（已中文化）
└── figures/                 # TikZ 内联插图，无需外部 PDF
```

## 在 Windows `E:\texlive` 上编译（中文）

TeX Live 预期位于 `E:\texlive`。所需集合：`collection-latexrecommended`、`collection-fontsrecommended`、`collection-latexextra`（`booktabs`、`listings`、`tikz`）及中文支持 `collection-langchinese`（`ctex`、`fandol` 字体）。

在 `E:\TensorPlay\docs\whitepaper\` 打开 **PowerShell** 或 **cmd**：

```powershell
# 推荐：xelatex（中文首选，ctex + fandol）
latexmk -xelatex main.tex
# 或
xelatex main.tex
xelatex main.tex

# 回退：pdflatex（需安装 CJK 支持，效果略逊于 xelatex）
latexmk -pdf main.tex

# 输出：main.pdf（A4，11pt，含目录、插图目录、表格目录）
```

`main.tex` 已配置 `\usepackage[UTF8, fontset=fandol]{ctex}`，封面、摘要、目录、图表标题、术语表均已设为中文；`fancyhdr` 页眉显示“TensorPlay 白皮书”。如在 Windows 上缺 `fandol`，可将 `fontset=fandol` 改为 `fontset=windows` 或 `fontset=founder`。

无需外部数据或联网；所有插图为 TikZ，所有代码清单来自所引 `path:line`。

## 源码验证

所有结构性主张均以 `path:line` 对 `version.txt:1`（1.0.0rc0）处的源码树援引。快速检查：

```powershell
grep -rn "tpx" p10/include/ | wc -l   # 期望 0 — 头文件隔离
grep -n "enum class DispatchKey" p10/include/DispatchKey.h -A 18
ls -lh build/generated/tensorplay/ops/  # 10 个产物
```

## 样式说明

- 文档类：`article`（A4，11pt），`ctex`（中文）+ `lmodern`/`microtype`/`booktabs`/`tikz`/`listings`/`hyperref`。
- 代码清单：`style=cppstyle`（7.5pt，行号，`codebg`），支持中文注释（`extendedchars=true`）。
- 插图：仅 TikZ，无外部 PDF，标准 TeX Live 即可编译。
- 参考文献：内联 `thebibliography`（无需 `.bib`）。
- 语言：**中文正文**（可发表），含中文摘要/关键词，`fancyhdr` 带版本页脚；中文化通过 `ctex` 实现，术语表与缩写表已译为中文。

## 何为具体（非高层）

- 剖析器：每算子 `OpRecord`/`GpuTimerPair`、`RingBuffer`、`cuptiSubscribe`、`nvtxRangePushA`，禁用开销约 2 周期。
- 自动微分：`Node:150 next_edges_`、`Edge:14 shape_hint`、`GraphTask:22 dependencies_`、`InputBuffer:16 accumulate`、`Engine:621 execute` 带 `local_queue` 重入、`SavedVariable:15` 版本守卫。
- 向量化映射：`VmapCPU=9` 于 `kVmapKeyOffset=9`、`TransformDispatch` 在 `Autograd` 前解包、按切片回退 vs. 批量内核。
- 重分发：`array<atomic<void*>,13>` + `Composite` 回退、`TP_CONCAT` 唯一命名、`Library::impl` 链式、`DispatchStub::call` 自动混合精度关口。
