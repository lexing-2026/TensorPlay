# 参考实现的第三方依赖梳理与本项目处置

对照基准:本仓库 `third_party/` 下 vendored 的参考实现源码树及其
`cmake/Dependencies.cmake`(仅作依赖清单调研输入,不在本文外引用)。本文回答三个问题:

1. 参考实现带了哪些第三方依赖、各自承担什么;
2. 本项目为什么可以不带这些;
3. 哪些位置存在真实缺口,决策是什么。

调研结论先行:**本项目对参考实现的依赖面覆盖率很高,绝大多数依赖要么由
libc/系统承担,要么已有原生替代,要么属于本项目不做的业务(移动端/ROCm/
Vulkan)。真正需要记录的缺口只有少数几项**,见文末决策记录。

## 一、处置原则

本项目对第三方能力采取七类处置,优先级从上到下:

| 处置 | 含义 | 例子 |
|---|---|---|
| libc/系统自带 | 能力属于平台 libc 或编译器,零依赖 | libmvec、OpenMP、`__builtin_cpu_supports` |
| 运行时 dlopen | 可选能力运行时探测加载,缺席时优雅降级 | CUPTI、NVTX、ITT、ILP64 LAPACK |
| 原生重写 | 语义清晰、体量可控的部分自己写 | 线程池、profiler、量化内核、RPC、分配器 |
| Python 层承接 | 能力搬到 Python 侧,省掉 C++ 依赖 | ONNX 导出(onnx 包)、序列化 |
| 单头 vendored | 单文件、许可清晰、直接放进 p10 | pocketfft_hdronly.h |
| 可选 vendored | 大型库,构建期可选开关 | oneDNN、NNPACK、NCCL、gloo |
| 不适用 | 对应能力本项目不实现 | 移动端、ROCm、Vulkan、TorchScript |

判断标准:**优先消除构建期强依赖**;同一能力若 libc 已提供(如 libmvec),
不重复造轮子;第三方库只在"没有它功能就缺一块"时才 vendor,且尽量做成
可选(缺席可编译、可运行,只是少了对应后端)。

## 二、逐项映射

### 2.1 数学与计算内核

| 参考实现依赖 | 用途(参考实现中) | 本项目处置 |
|---|---|---|
| sleef | 参考实现 CPU 矢量层的超越函数(exp/log/sin/erf…) | **已 vendor 并接线**(见决策记录 #1) |
| pocketfft | CPU FFT | 单头 vendored:`p10/include/pocketfft_hdronly.h`,SpectralKernels 已接入 |
| oneDNN (mkl-dnn) | x86 conv/RNN/GEMM 加速 | 同样可选 vendored(`third_party/oneDNN`,`FindMKLDNN.cmake`),`OneDNNContext.cpp` 直连其 C++ API |
| eigen | 无 BLAS 时的 linalg 回退 | 不需要:原生 `p10/src/backend/cpu/Lapack.cpp` 运行时 dlopen ILP64 OpenBLAS(优先 numpy 自带的 `libscipy_openblas64_`),另有构建期 vendored OpenBLAS tarball |
| fbgemm | 量化 GEMM 打包、FP8 | 原生:`QuantKernels.cpp` / `SemiStructuredKernels.cpp`(CPU+CUDA) |
| gemmlowp | 旧量化参考路径 | 不需要(同上) |
| XNNPACK | 移动端 CPU 推理 | 不适用(无移动端业务) |
| kleidiai / mslk | ARM 量化微核 / GPU 分组缩放 GEMM | 不适用(本项目量化走原生内核;分组 GEMM 未列为目标) |
| NNPACK(+FP16/FXdiv/psimd/pthreadpool/cpuinfo/python-peachpy/python-six) | conv/pooling 加速及其支撑库 | 可选 vendored(`cmake/External/nnpack.cmake`),`ConvKernels.cpp` 有接入点;整套子模块已就位 |
| cutlass / flash-attention / composable_kernel / aiter | GPU 注意力与 GEMM 后端(CUDA/ROCm) | 不 vendor。`AttentionKernels.cu` 原生实现;仅当检测到包含 flash/cutlass 头的 vendor 根时以头文件方式参与(`p10/CMakeLists.txt` 的 `TP_NATIVE_FLASH_SRC` glob),不链接 |
| cudnn_frontend | cuDNN graph API | 可选头文件;主路径走 cuDNN C API(`CUDNNUtils.h`) |
| ideep | oneDNN 的 C++ 封装 | 不需要(直接用 oneDNN 原生 API) |

### 2.2 CPU 检测与并行

| 参考实现依赖 | 用途 | 本项目处置 |
|---|---|---|
| cpuinfo | CPU ISA 运行时检测 | 原生:`__builtin_cpu_supports` / cpuid(`DispatchStub.cpp`、`tp_cpu_supports_*`) |
| pthreadpool | NNPACK/XNNPACK 线程池 | 原生:`Parallel.cpp`( intra-op 线程池 + parallel_for) |
| llvm-openmp | Windows/macOS 捆绑 OpenMP | 系统 OpenMP(`find_package(OpenMP)`) |

### 2.3 序列化与模型格式

| 参考实现依赖 | 用途 | 本项目处置 |
|---|---|---|
| miniz | torch.save 的 zip64 容器 | 不需要:原生 MEGA 格式(`tensorplay/serialization.py`)+ safetensors 互认 + mmap 零拷贝加载;zip 能力由 Python 标准库承担 |
| protobuf + onnx | ONNX 导出 | Python 层承接:`tensorplay/onnx/*.py` 直接使用 onnx 包,无 C++ protobuf |
| flatbuffers | 移动端 lite interpreter | 不适用 |
| nlohmann/json | c10d 控制面、AOTInductor 打包等 | 不需要(无对应子系统) |
| cpp-httplib | c10d 控制面 HTTP 服务 | 不需要(hub/工具走 urllib) |

### 2.4 分布式

| 参考实现依赖 | 用途 | 本项目处置 |
|---|---|---|
| gloo | CPU 集合通信后端 | 可选 vendored(`third_party/gloo`,集成进行中) |
| tensorpipe | RPC 传输 | 不需要:`src/bindings/python/Rpc.cpp` 原生 socket 实现 |
| MPI | c10d 可选传输 | 不适用 |

### 2.5 Profiler

| 参考实现依赖 | 用途 | 本项目处置 |
|---|---|---|
| kineto | profiler 后端(CUPTI/EVT) | 原生:`Profiler.cpp` / `ProfilerCupti.cpp`(dlopen CUPTI)/ `ProfilerGpu.cpp` |
| ittapi | VTune 桥 | 原生 dlopen 桥:`ProfilerItt.cpp` |
| NVTX 头 | nsight 桥 | 原生 dlopen 桥:`ProfilerNvtx.cpp` |
| perfetto / valgrind-headers | 系统追踪 / 内存检查 | 不适用 |
| concurrentqueue (moodycamel) | 新执行器 MPMC 队列 | 不需要(futures/调度原生实现) |

### 2.6 GPU 栈(CUDA)

参考实现通过 CUDA Toolkit 引入 cublas/cublasLt/cufft/cusolver/cusparse/curand/nvrtc/cupti,NCCL 可 vendored 可系统。本项目完全同构:`p10/CMakeLists.txt` 链接 `CUDA::cublas CUDA::cublasLt CUDA::cufft CUDA::cusolver`,`cmake/External/nccl.cmake` 提供 NCCL(系统优先、vendored 兜底),CUPTI 同样 dlopen。**无差异,不赘述。**

### 2.7 构建、绑定与杂项

| 参考实现依赖 | 用途 | 本项目处置 |
|---|---|---|
| pybind11 | Python 绑定 | 相同(系统 `find_package`);另有 nanobind/gitignore 通道 |
| googletest / benchmark | 测试 | 系统包/本地目录 |
| mimalloc | 可选 CPU 分配器(默认关) | 不需要:`Allocator.cpp` 原生对齐分配 |
| fmt | 少量格式化 | 不需要(std 设施) |
| Vulkan 全家(VulkanMemoryAllocator 等) | 移动 GPU | 不适用 |

## 三、为什么可以不带这些

1. **参考实现的依赖面比本项目宽,是因为它服务的平台面宽。** SLEEF 是为了
   Windows/macOS 没有 libmvec;flatbuffers/Vulkan/XNNPACK 是为了移动端;
   composable_kernel/aiter 是为了 ROCm。本项目当前目标平台是 Linux x86_64
   (+CUDA),这些依赖的前提条件不成立。
2. **libc 本身已经是矢量数学库。** glibc ≥ 2.35 把 libmvec 合并进 libm,
   `_ZGVdN8v_*f` / `_ZGVeN16v_*` 就是平台原生的 SIMD 超越函数,由发行版
   调优维护。自己再写一套多项式,在 glibc 上既不可能更快,还会引入"矢量
   路径与标量路径结果不一致"的新问题。参考实现 vendor SLEEF 的动机是
   可移植性与跨平台结果一致,不是性能。
3. **"可选、可降级"比"全带"更符合本项目定位。** oneDNN/NNPACK/NCCL/gloo
   全部缺席时,p10 仍然编译、运行,走原生内核——这是与参考实现最大的结构
   差异:参考实现大量依赖是强依赖,本项目是弱依赖。
4. **Python 层可以承接一大类 C++ 依赖。** ONNX(protobuf)和 zip 容器
   (miniz)是两个典型案例:能力不丢,依赖从 C++ 链接图里消失。

## 四、决策记录与真实缺口

### 决策 #1:矢量数学采用 vendored SLEEF(2026-08-30 实施)

* `third_party/sleef`(SLEEF 4.0,含其 TLFloat 子模块)已 vendored,
  `cmake/External/sleef.cmake` 按参考实现的配方构建静态 libsleef
  (关 DFT/quad/tests/OpenMP;TLFloat 产物装入构建树私有前缀)。
* p10 全部矢量超越函数(f32 f8/f16、f64 d4/d8)经
  `p10/include/cpu/vec/SleefShims.h` 直连 SLEEF 的运行时分发入口,
  替换了原先 glibc libmvec(`_ZGV*`)的全部用点——快速路径不再绑定
  glibc,任何 GCC/clang x86-64 平台(含 musl)一致可用。
* 精度档位沿用参考实现的选择:多数 u10,f32 sin/cos 用 u35,
  erfc 用 u15,hypot 用 u05。
* libmvec 不再被 p10 引用;`atan2`/`hypot`/`pow` 亦升级为矢量路径。

### 决策 #2:f64 矢量路径同样走 SLEEF(2026-08-30 实施)

原 f64 快速路径(libmvec)已随 #1 一并迁移;`vec256_double.h` 原先的
标量 `map()` 回退也升级为 SLEEF d4/d8 路径,f64 超越函数获得矢量加速,
实测对 float64 参考误差 ~1e-15(1 ulp 量级)。

### 缺口 #3:gloo 集成(进行中)

`third_party/gloo` 已就位,CMake 接入与 `ProcessGroup` 语义由分布式线
推进,本清单仅登记,不重复实现。

### 缺口 #4:CPU conv 无 NNPACK/oneDNN 时的深度

原生 `ConvKernels.cpp` 已含 Winograd/AVX2 路径;oneDNN 缺席时的极端
shape 覆盖度未做过系统性对标,列为后续验证项(验证任务,非实现任务)。

## 五、总结

| 参考实现依赖类别 | 数量(约) | 本项目对应 |
|---|---|---|
| 不适用(移动/ROCm/Vulkan/文档工具) | 12+ | 无需对应 |
| libc/编译器/系统承担 | 8+ | libmvec、OpenMP、cpuid、CUDA Toolkit |
| 运行时 dlopen 替代 | 5 | CUPTI/NVTX/ITT/LAPACK |
| 原生重写 | 8 | 线程池、profiler、量化、RPC、分配器、序列化 |
| Python 层承接 | 2 | onnx、zip 容器 |
| vendored(可选) | 6 | oneDNN、NNPACK 族、NCCL、gloo、pocketfft(单头) |
| 真实缺口 | 2 | gloo 集成(进行中)、跨平台矢量数学(决策:vendor SLEEF,暂缓) |
