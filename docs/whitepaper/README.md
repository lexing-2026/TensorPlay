# TensorPlay White Paper — LaTeX Publishable Edition

**Industry highest standard, locally compilable on Windows `E:\texlive`**

This directory contains the publishable LaTeX sources for the TensorPlay white paper. Unlike high-level overviews, it documents **concrete** mechanisms with `path:line` citations: 13-slot `DispatchTable`, `TENSORPLAY_LIBRARY_IMPL` static registration, 7-step `TensorIterator` pipeline, explicit DAG without a Tape, `ReadyQueue` by `sequence_nr`, `SavedVariable` version guards, `Vmap` at offset 9, and the `OpRecord / GpuTimerPair / Cupti / Nvtx` profiler stack.

## Files

```
docs/whitepaper/
├── main.tex                 # preamble + abstract + \input{sections/*.tex}
├── sections/
│   ├── 01-introduction.tex  # problem, contributions, scope
│   ├── 02-architecture.tex  # four pillars, CMake DAG, virtual interface
│   ├── 03-redispatch.tex    # DispatchKey 13, Dispatcher 214 lines, macros, dual binding
│   ├── 04-profiler.tex      # OpRecord, GpuTimerPair, Cupti/Nvtx/Itt, overhead
│   ├── 05-autograd.tex      # Node/Edge/GraphTask/InputBuffer/Engine, SavedVariable, Anomaly
│   ├── 06-vmap.tex          # Vmap offset 9, TransformDispatch, BatchingKernels
│   ├── 07-memory.tex        # DataPtr, Storage, TensorImpl, Tensor, MemoryFormat
│   ├── 08-codegen.tex       # YAML contracts, 7 generators, build staging
│   ├── 09-evaluation.tex    # readability, correctness guards, trimmed axes
│   └── 10-appendix.tex      # reproducibility, Windows build, file map
└── figures/                 # TikZ figures are inline; no external PDFs required
```

## Compile on Windows `E:\texlive`

TeX Live is expected at `E:\texlive` (per your environment). Required collections: `collection-latexrecommended`, `collection-fontsrecommended`, `collection-latexextra` (for `booktabs`, `listings`, `tikz`).

Open **PowerShell** or **cmd** in `E:\TensorPlay\docs\whitepaper\`:

```powershell
# Recommended: latexmk (handles reruns + bibtex)
latexmk -pdf main.tex

# Manual:
pdflatex main.tex
bibtex main    # no .bib needed — bibliography is inline \begin{thebibliography}
pdflatex main.tex
pdflatex main.tex

# Output: main.pdf (A4, 11pt, with ToC, LoF, LoT)
```

For `xelatex` (if you prefer):

```powershell
xelatex main.tex
xelatex main.tex
```

No external data or Internet is required; all figures are TikZ and all code listings are from the cited `path:line`.

## Verify Sources

Every structural claim cites `path:line` against the source tree at `version.txt:1` (1.0.0rc0). Quick checks:

```powershell
grep -rn "tpx" p10/include/ | wc -l   # expect 0 — header isolation
grep -n "enum class DispatchKey" p10/include/DispatchKey.h -A 18
ls -lh build/generated/tensorplay/ops/  # 10 artifacts
```

## Style Notes

- Class: `article` (A4, 11pt), `lmodern`, `microtype`, `booktabs`, `tikz`, `listings`, `hyperref`.
- Code listings: `style=cppstyle` (7.5pt, line numbers, `codebg`).
- Figures: TikZ only, no external PDFs, compiles on stock TeX Live.
- Bibliography: inline `thebibliography` (no `.bib` file needed).
- Language: English (publishable), abstract/keywords included, `fancyhdr` with version footer.

## What Is Concrete (Not High-Level)

- Profiler: `OpRecord`/`GpuTimerPair` per-op, `RingBuffer`, `cuptiSubscribe`, `nvtxRangePushA`, disabled overhead ≈2 cycles.
- Autograd: `Node:150 next_edges_`, `Edge:14 shape_hint`, `GraphTask:22 dependencies_`, `InputBuffer:16 accumulate`, `Engine:621 execute` with `local_queue` for reentrancy, `SavedVariable:15` version guard.
- Vmap: `VmapCPU=9` at `kVmapKeyOffset=9`, `TransformDispatch` unwrapping before `Autograd`, per-slice loop vs. batched kernels.
- Redispatch: `array<atomic<void*>,13>` + `Composite` fallback, `TP_CONCAT` unique names, `Library::impl` chaining, `DispatchStub::call` autocast choke point.
