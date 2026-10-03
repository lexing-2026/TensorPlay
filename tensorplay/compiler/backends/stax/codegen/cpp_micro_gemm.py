"""Small matrix products written into a kernel as it is generated.

A kernel that multiplies blocks of its operands -- an attention kernel taking
the product of a block of queries with a block of keys, say -- does not call
out for each block.  It carries a product of its own, written for the block
shape it uses, so the block stays in registers from the first multiply to the
last.  This module writes those products.

The product here computes in float, one vector register per row segment, and
reads float or a sixteen-bit float operand that it widens on the way in.  The
result is float; a kernel that wants it narrower narrows it itself.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import tensorplay as tp

from ..cpu_vec_isa import pick_vec_isa
from ..utils import IndentedBuffer
from .common import KernelTemplate
from .cpp_template_kernel import CppTemplateKernel
from .cpp_utils import DTYPE_TO_CPP, GemmBlocking, value_to_cpp


class LayoutType(Enum):
    """How the second operand is laid out in memory."""

    NORMAL = 0
    VNNI2 = 1
    VNNI4 = 2


def get_restrict_keyword() -> str:
    return "__restrict__"


class CppMicroGemm:
    """A product of two small blocks, C = alpha * A @ B (or C += it).

    A subclass writes the body; this class writes the declaration, the call and
    the options both are rendered from, so every product is called the same
    way whatever its body does.
    """

    DECLARE_KERNEL = r"""
template <bool accum, bool prefetch=false>
inline void {{kernel_name}}(
{%- if kernel_extra_args_declare %}
    {{kernel_extra_args_declare}}
{%- endif %}
    const {{input_t}}* {{restrict_keyword}} A,
    const {{input2_t}}* {{restrict_keyword}} B,
    {{output_t}}* {{restrict_keyword}} C,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
)
"""

    def __init__(
        self,
        name: str,
        input_dtype: Any,
        input2_dtype: Any,
        output_dtype: Any,
        compute_dtype: Any,
        register_blocking: GemmBlocking,
        alpha: Any = 1,
    ) -> None:
        if input2_dtype is None:
            raise AssertionError("expected input2_dtype to be set, got None")
        self.name = name
        self.input_dtype = input_dtype
        self.input2_dtype = input2_dtype
        self.output_dtype = output_dtype
        self.compute_dtype = compute_dtype
        self.register_blocking = register_blocking
        self.alpha = alpha
        self.pack_vnni_B_locally = False

    def get_common_options(self) -> dict[str, Any]:
        return {
            "tp": tp,
            "kernel_name": self.name,
            "input_dtype": self.input_dtype,
            "input2_dtype": self.input2_dtype,
            "output_dtype": self.output_dtype,
            "compute_dtype": self.compute_dtype,
            "input_t": DTYPE_TO_CPP[self.input_dtype],
            "input2_t": DTYPE_TO_CPP[self.input2_dtype],
            "output_t": DTYPE_TO_CPP[self.output_dtype],
            "compute_t": DTYPE_TO_CPP[self.compute_dtype],
            "alpha": self.alpha,
            "kernel_extra_args_declare": self.get_kernel_extra_args_declare(),
            "vnni_size": 2,
            "restrict_keyword": get_restrict_keyword(),
            "pack_vnni_B_locally": self.pack_vnni_B_locally,
            "template": self,
        }

    def get_kernel_declaration(self) -> str:
        options = self.get_common_options()
        return KernelTemplate._template_from_string(self.DECLARE_KERNEL).render(options)

    def get_kernel_extra_args_declare(self) -> str:
        return ""

    def get_kernel_extra_args(self, **kwargs: Any) -> list[str]:
        return []

    def codegen_define(self, kernel: CppTemplateKernel) -> str:
        raise NotImplementedError

    def codegen_call(
        self,
        kernel: CppTemplateKernel,
        A: Any,
        B: Any,
        C: Any,
        accum: bool,
        prefetch: bool = False,
        **kwargs_for_extra_args: Any,
    ) -> str:
        """The call computing C = alpha * A @ B, or C += it when ``accum``."""

        A_ptr = f"&({kernel.index(A, [0, 0])})"
        B_ptr = f"&({kernel.index(B, [0, 0])})"
        C_ptr = f"&({kernel.index(C, [0, 0])})"
        M = kernel.size(C, 0)
        N = kernel.size(C, 1)
        K = kernel.size(A, 1)
        lda = kernel.stride(A, 0)
        ldb = kernel.stride(B, 0)
        ldc = kernel.stride(C, 0)
        res = IndentedBuffer()
        res.writeline(
            f"{self.name}<{value_to_cpp(accum, 'bool')}, {value_to_cpp(prefetch, 'bool')}>("
        )
        with res.indent():
            kwargs_for_extra_args.update({"kernel": kernel})
            for arg in self.get_kernel_extra_args(**kwargs_for_extra_args):
                res.writeline(arg)
            res.writeline(f"{A_ptr},")
            res.writeline(f"{B_ptr},")
            res.writeline(f"{C_ptr},")
            res.writeline(f"{M},")
            res.writeline(f"{N},")
            res.writeline(f"{K},")
            res.writeline(f"{lda},")
            res.writeline(f"{ldb},")
            res.writeline(f"{ldc}")
        res.writeline(");")
        return res.getvalue()

    def use_local_vnni_blocking(self, should_block_weight: bool) -> None:
        self.pack_vnni_B_locally = should_block_weight

    def codegen_init(self, kernel: CppTemplateKernel) -> str:
        return ""

    def codegen_finalize(self, kernel: CppTemplateKernel) -> str:
        return ""

    def get_b_layout(self) -> LayoutType:
        return LayoutType.NORMAL


class CppMicroGemmFP32Vec(CppMicroGemm):
    """The product computed with float vector instructions.

    Reads float, bfloat16 or half operands and accumulates in float.  ``tail_n``
    also writes the variant for a block whose column count is not a whole
    number of registers, and ``trans_b`` the variants that read B transposed --
    which is how a product of queries with keys reads the keys.
    """

    TEMPLATE_ENTRY = r"""
{{declare_kernel}} {
    using Vectorized = tensorplay::vec::Vectorized<{{compute_t}}>;
    constexpr auto VLEN = Vectorized::size();
    {{kernel.assert_function}}({{block_n}} % VLEN == 0, "block_n dimension must be multiple of Vector size");
    {{kernel.assert_function}}(K % {{block_k}} == 0, "K dimension must be multiple of {{block_k}}");
    for (int64_t m = 0; m < M; m += {{block_m}}) {
        int64_t block_m = std::min<int64_t>(M - m, {{block_m}});
        for (int64_t n = 0; n < N; n += {{block_n}}) {
            int64_t block_n = std::min<int64_t>(N - n, {{block_n}});
            if (block_m == {{block_m}} && block_n == {{block_n}}) {
{%- if not trans_b %}
                {{kernel_name}}_kernel<{{block_m}}, {{block_n}}, accum, prefetch>(
{%- else %}
                {{kernel_name}}_transpose_b_kernel<{{block_m}}, {{block_n}}, accum, prefetch>(
{%- endif %}
                    A + m * lda,
{%- if not trans_b %}
                    B + n,
{%- else %}
                    B + n * ldb,
{%- endif %}
                    C + m * ldc + n,
                    K,
                    lda,
                    ldb,
                    ldc
                );
{%- if tail_n %}
            } else if (block_n == {{block_n}}){
{%- else %}
            } else {
{%- endif %}
                switch (block_m) {
{%- for b in range(block_m - 1, 0, -1) %}
                case {{b}}:
    {%- if not trans_b %}
                    {{kernel_name}}_kernel<{{b}}, {{block_n}}, accum, prefetch>(
    {%- else %}
                    {{kernel_name}}_transpose_b_kernel<{{b}}, {{block_n}}, accum, prefetch>(
    {%- endif %}
                        A + m * lda,
    {%- if not trans_b %}
                        B + n,
    {%- else %}
                        B + n * ldb,
    {%- endif %}
                        C + m * ldc + n,
                        K,
                        lda,
                        ldb,
                        ldc
                    );
                    break;
{%- endfor %}
                default:
                    {{kernel.assert_function}}(false, "Unsupported block_m: {{block_m}}");
                }

{%- if tail_n %}
            } else {
                switch (block_m) {
    {%- for b in range(block_m, 0, -1) %}
                case {{b}}:
        {%- if not trans_b %}
                    {{kernel_name}}_ntail_kernel<{{b}}, {{block_n}}, accum, prefetch>(
        {%- else %}
                    {{kernel_name}}_ntail_transpose_b_kernel<{{b}}, {{block_n}}, accum, prefetch>(
        {%- endif %}
                        A + m * lda,
        {%- if not trans_b %}
                        B + n,
        {%- else %}
                        B + n * ldb,
        {%- endif %}
                        C + m * ldc + n,
                        block_n,
                        K,
                        lda,
                        ldb,
                        ldc
                    );
                    break;
    {%- endfor %}
                default:
                    {{kernel.assert_function}}(false, "Unsupported block_m: {{block_m}}");
                }
            }
{%- else %}
            }
{%- endif %}
        }
    }
}
"""

    TEMPLATE_KERNEL = r"""

template <int64_t BLOCK_M, int64_t BLOCK_N, bool accum, bool prefetch=false>
{%- if not trans_b %}
    {%- if tail_n %}
inline void {{kernel_name}}_ntail_kernel(
    {%- else %}
inline void {{kernel_name}}_kernel(
    {%- endif %}
{%- else %}
    {%- if tail_n %}
inline void {{kernel_name}}_ntail_transpose_b_kernel(
    {%- else %}
inline void {{kernel_name}}_transpose_b_kernel(
    {%- endif %}
{%- endif %}
    const {{input_t}}* {{restrict_keyword}} A,
    const {{input2_t}}* {{restrict_keyword}} B,
    {{output_t}}* {{restrict_keyword}} C,
{%- if tail_n %}
    int64_t N,
{%- endif %}
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc
) {
    using Vectorized = tensorplay::vec::Vectorized<{{compute_t}}>;
{%- if input2_dtype in [tp.bfloat16, tp.float16] %}
    using VectorizedIn = tensorplay::vec::Vectorized<{{input_t}}>;
{%- endif %}

{%- if not trans_b %}
    constexpr auto VLEN = Vectorized::size();
    constexpr auto ROWS = BLOCK_M;
    constexpr auto COLS = BLOCK_N / VLEN;

    Vectorized va;
    tensorplay::vec::VectorizedN<{{compute_t}}, COLS> vb;
    tensorplay::vec::VectorizedN<{{compute_t}}, ROWS*COLS> vc;

    {%- if tail_n %}
    int64_t rCOLS = (N + VLEN - 1) / VLEN;
    int ntail = N % VLEN;
    {%- endif %}
    auto loadc = [&](auto i) {
        if constexpr (accum) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
    {%- if tail_n %}
            int load_size = (col == rCOLS - 1 && ntail != 0) ? ntail : VLEN;
            if (col < rCOLS) {
                vc[i] = Vectorized::loadu(C + row * ldc + col * VLEN, load_size);
            }
    {%- else %}
            vc[i] = Vectorized::loadu(C + row * ldc + col * VLEN);
    {%- endif %}
        } else {
            vc[i] = Vectorized(0.0f);
        }
    };
    tensorplay::ForcedUnroll<ROWS * COLS>{}(loadc);

    auto compute = [&, COLS](auto i, int k) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
    {%- if tail_n %}
        int load_size = (col == rCOLS - 1 && ntail != 0) ? ntail : VLEN;
    {%- endif %}
        if constexpr (col == 0) {
    {%- if alpha != 1 %}
            va = Vectorized(static_cast<{{compute_t}}>(A[row * lda + k]) * {{alpha}});
    {%- else %}
            va = Vectorized(static_cast<{{compute_t}}>(A[row * lda + k]));
    {%- endif %}
        }

        if constexpr (row == 0) {
    {%- if tail_n %}
            if (col < rCOLS) {
        {%- if input2_dtype in [tp.bfloat16, tp.float16] %}
                auto b = VectorizedIn::loadu(B + k * ldb + col * VLEN, load_size);
                vb[col] = tensorplay::vec::convert<{{compute_t}}>(b);
        {%- else %}
                vb[col] = Vectorized::loadu(B + k * ldb + col * VLEN, load_size);
        {%- endif %}
            } else {
                vb[col] = Vectorized(0.0f);
            }

    {%- else %}

        {%- if input2_dtype in [tp.bfloat16, tp.float16] %}
            auto b = VectorizedIn::loadu(B + k * ldb + col * VLEN, VLEN);
            vb[col] = tensorplay::vec::convert<{{compute_t}}>(b);
        {%- else %}
            vb[col] = Vectorized::loadu(B + k * ldb + col * VLEN);
        {%- endif %}
    {%- endif %}

        }

        constexpr int idx = row * COLS + col;
    {%- if tail_n %}
        if (col < rCOLS) {
            vc[idx] = tensorplay::vec::fmadd(va, vb[col], vc[idx]);
        }
    {%- else %}
        vc[idx] = tensorplay::vec::fmadd(va, vb[col], vc[idx]);
    {%- endif %}
    };

    for (int k = 0; k < K; ++k) {
        tensorplay::ForcedUnroll<ROWS * COLS>{}(compute, k);
    }

    // store to C
    auto storec = [&](auto i) {
        constexpr int row = i / COLS;
        constexpr int col = i % COLS;
    {%- if tail_n %}
        int store_size = (col == rCOLS - 1 && ntail != 0) ? ntail : VLEN;
        if (col < rCOLS) {
            vc[i].store(C + row * ldc + col * VLEN, store_size);
        }
    {%- else %}
        vc[i].store(C + row * ldc + col * VLEN);
    {%- endif %}
    };
    tensorplay::ForcedUnroll<ROWS * COLS>{}(storec);

{%- else %}
    // Use 2 implementations for the transposed B:
    // First implementation:
    //   Transpose first and then perform outer product calculation in sub-blocks,
    //   which introduces an additional transpose overhead of [K, N] compared to the non-transpose version.
    // Second implementation:
    //   Directly perform inner product calculation in sub-blocks,
    //   which introduces an additional vector reduction of [M, N] compared to the non-transpose version.
    // Therefore, when M * N / (K * N) is large, the first implementation has better performance.
    {%- if tail_n %}
    if (K % Vectorized::size() == 0 && N % Vectorized::size() == 0 && 24 * BLOCK_M > K) {
    {%- else %}
    if (K % Vectorized::size() == 0 && 24 * BLOCK_M > K) {
    {%- endif %}
        // First implementation:
        constexpr auto VLEN = Vectorized::size();
        constexpr auto ROWS = BLOCK_M;
        constexpr auto COLS = BLOCK_N / VLEN;
        int _K = K / VLEN;
        Vectorized va;
        tensorplay::vec::VectorizedN<{{compute_t}}, VLEN> vb;
        tensorplay::vec::VectorizedN<{{compute_t}}, ROWS*COLS> vc;
        auto loadc = [&](auto i) {
            if constexpr (accum) {
                constexpr int row = i / COLS;
                constexpr int col = i % COLS;
                vc[i] = Vectorized::loadu(C + row * ldc + col * VLEN);
            } else {
                vc[i] = Vectorized(0.0f);
            }
        };
        tensorplay::ForcedUnroll<ROWS * COLS>{}(loadc);
        auto unroll_loadB = [&](auto i, const {{input2_t}}* {{restrict_keyword}} src_ptr) {
    {%- if input2_dtype in [tp.bfloat16, tp.float16] %}
            auto b = VectorizedIn::loadu(src_ptr + i * ldb, VLEN);
            vb[i] = tensorplay::vec::convert<{{compute_t}}>(b);
    {%- else %}
            vb[i] = Vectorized::loadu(src_ptr + i * ldb, VLEN);
    {%- endif %}
        };
        auto compute_trans = [&, COLS](auto i, int k) {
            constexpr int row = i % ROWS;
            constexpr int col = i / ROWS;
            constexpr int e_col = col * VLEN;
            int idk = k * VLEN;
            if constexpr (row == 0) {
                tensorplay::ForcedUnroll<VLEN>{}(unroll_loadB, B + e_col * ldb + idk);
                tensorplay::vec::transpose_block(vb);
            }
            constexpr int idx = row * COLS + col;
            {{kernel.unroll_pragma(16)}}
            for (int j = 0; j < VLEN; j++) {
    {%- if alpha != 1 %}
                va = Vectorized(static_cast<{{compute_t}}>(A[row * lda + idk + j]) * {{alpha}});
    {%- else %}
                va = Vectorized(static_cast<{{compute_t}}>(A[row * lda + idk + j]));
    {%- endif %}
                vc[idx] = tensorplay::vec::fmadd(va, vb[j], vc[idx]);
            }
        };
        for (int k = 0; k < _K; ++k) {
            tensorplay::ForcedUnroll<ROWS * COLS>{}(compute_trans, k);
        }
        // store to C
        auto storec = [&](auto i) {
            constexpr int row = i / COLS;
            constexpr int col = i % COLS;
            vc[i].store(C + row * ldc + col * VLEN);
        };
        tensorplay::ForcedUnroll<ROWS * COLS>{}(storec);
    } else {
        // Second implementation
    {%- if input2_dtype in [tp.bfloat16, tp.float16] %}
        constexpr auto VLEN = VectorizedIn::size();
    {%- else %}
        constexpr auto VLEN = Vectorized::size();
    {%- endif %}
        int _K = (K + VLEN - 1) / VLEN;
        // sub-block size of BLOCK_N and BLOCK_M
        constexpr int sM = {{sub_block_m}};
        constexpr int sN = {{sub_block_n}};
    {%- if tail_n %}
        int bN = (N + sN - 1) / sN;
    {%- else %}
        constexpr int bN = (BLOCK_N + sN - 1) / sN;
    {%- endif %}
        constexpr int bM = (BLOCK_M + sM - 1) / sM;

    {%- if input2_dtype in [tp.bfloat16, tp.float16] %}
        tensorplay::vec::VectorizedN<{{compute_t}}, 2> va;
        tensorplay::vec::VectorizedN<{{compute_t}}, 2 * sN> vb;
    {%- else %}
        tensorplay::vec::Vectorized<{{compute_t}}> va;
        tensorplay::vec::VectorizedN<{{compute_t}}, sN> vb;
    {%- endif %}
        tensorplay::vec::VectorizedN<{{compute_t}}, sN * sM> vmid;

    {%- if tail_n %}
        int ntail = N % sN;
    {%- else %}
        constexpr int ntail = BLOCK_N % sN;
    {%- endif %}
        constexpr int mtail = BLOCK_M % sM;
        int ktail = K % VLEN;

        auto compute_trans = [&](int m, int n, int k) {
    {%- if tail_n %}
            int e_n = (n == bN - 1 && ntail != 0) ? (N - n * sN) : sN;
    {%- else %}
            int e_n = (n == bN - 1 && ntail != 0) ? (BLOCK_N - n * sN) : sN;
    {%- endif %}
            int e_m = (m == bM - 1 && mtail != 0) ? (BLOCK_M - m * sM) : sM;
            int e_k = (k == _K - 1 && ktail != 0) ? (K - k * VLEN) : VLEN;
            {{kernel.unroll_pragma(sub_block_n)}}
            for (int i = 0; i < e_n; i++) {
    {%- if input2_dtype in [tp.bfloat16, tp.float16] %}
                auto b = VectorizedIn::loadu(B + (sN * n + i) * ldb + k * VLEN, e_k);
                std::tie(vb[2 * i], vb[2 * i + 1]) = tensorplay::vec::convert_to_float<{{input_t}}>(b);
    {%- else %}
                vb[i] = Vectorized::loadu(B + (sN * n + i) * ldb + k * VLEN, e_k);
    {%- endif %}
            }

            {{kernel.unroll_pragma(sub_block_m)}}
            for (int s = 0; s < e_m; s++) {
    {%- if input2_dtype in [tp.bfloat16, tp.float16] %}
                auto a = VectorizedIn::loadu(A + (sM * m + s) * lda + k * VLEN, e_k);
                std::tie(va[0], va[1]) = tensorplay::vec::convert_to_float<{{input_t}}>(a);
    {%- else %}
                va = Vectorized::loadu(A + (sM * m + s) * lda + k * VLEN, e_k);
    {%- endif %}

    {%- if alpha != 1 %}
                va = va * Vectorized({{alpha}});
    {%- endif %}
                if (k == 0) {
                    {{kernel.unroll_pragma(sub_block_n)}}
                    for (int i = 0; i < e_n; i++) {
    {%- if input2_dtype in [tp.bfloat16, tp.float16] %}
                        vmid[sN * s + i] = tensorplay::vec::fmadd(va[0], vb[2 * i], Vectorized(0.0f));
                        vmid[sN * s + i] = tensorplay::vec::fmadd(va[1], vb[2 * i + 1], vmid[sN * s + i]);
    {%- else %}
                        vmid[sN * s + i] = tensorplay::vec::fmadd(va, vb[i], Vectorized(0.0f));
    {%- endif %}
                    }
                } else {
                    {{kernel.unroll_pragma(sub_block_n)}}
                    for (int i = 0; i < e_n; i++) {
    {%- if input2_dtype in [tp.bfloat16, tp.float16] %}
                        vmid[sN * s + i] = tensorplay::vec::fmadd(va[0], vb[2 * i], vmid[sN * s + i]);
                        vmid[sN * s + i] = tensorplay::vec::fmadd(va[1], vb[2 * i + 1], vmid[sN * s + i]);
    {%- else %}
                        vmid[sN * s + i] = tensorplay::vec::fmadd(va, vb[i], vmid[sN * s + i]);
    {%- endif %}
                    }
                }
            }

            // store to C
            if (k == _K - 1) {
                {{kernel.unroll_pragma(sub_block_m)}}
                for (int s = 0; s < e_m; s++) {
                    {{kernel.unroll_pragma(sub_block_n)}}
                    for (int i = 0; i < e_n; i++) {
                        auto v = tensorplay::vec::vec_reduce_all([](Vectorized& x, Vectorized& y) { return x + y; }, vmid[sN * s + i]);
                        if constexpr (accum) {
                            auto c = *(C + (sM * m + s) * ldc + sN * n + i);
                            *(C + (sM * m + s) * ldc + sN * n + i) = c + v;
                        } else {
                            *(C + (sM * m + s) * ldc + sN * n + i) = v;
                        }
                    }
                }
            }
        };

        for (int n = 0; n < bN; ++n) {
            for (int m = 0; m < bM; ++m) {
                for (int k = 0; k < _K; ++k) {
                    compute_trans(m, n, k);
                }
            }
        }
    }
{%- endif %}
}
"""

    def __init__(
        self,
        name: str,
        input_dtype: Any,
        input2_dtype: Any,
        output_dtype: Any,
        compute_dtype: Any,
        register_blocking: GemmBlocking,
        alpha: Any = 1,
        tail_n: bool = False,
        trans_b: bool = False,
    ) -> None:
        super().__init__(
            name,
            input_dtype,
            input2_dtype,
            output_dtype,
            compute_dtype,
            register_blocking,
            alpha,
        )
        self.tail_n = tail_n
        # Reading B transposed goes through a register transpose, which is
        # written for the two vector widths that have the shuffles for it.
        if trans_b:
            isa = pick_vec_isa().name
            if isa not in ("avx512", "avx2"):
                raise AssertionError(
                    f"a transposed second operand needs avx512 or avx2, got {isa}"
                )
        self.trans_b = trans_b

    def codegen_define(self, kernel: CppTemplateKernel) -> str:
        options = {
            "declare_kernel": self.get_kernel_declaration(),
            "kernel": kernel,
            "block_m": self.register_blocking.block_m,
            "block_n": self.register_blocking.block_n,
            "block_k": self.register_blocking.block_k,
            "trans_b": False,
            "tail_n": False,
            "restrict_keyword": get_restrict_keyword(),
            **self.get_common_options(),
        }
        if self.trans_b:
            options.update(
                {
                    "trans_b": self.trans_b,
                    "sub_block_m": min(1, self.register_blocking.block_m),
                    "sub_block_n": min(4, self.register_blocking.block_n),
                }
            )
        render = KernelTemplate._template_from_string
        result = render(self.TEMPLATE_KERNEL).render(options)
        if self.tail_n:
            options.update({"tail_n": self.tail_n})
            result += render(self.TEMPLATE_KERNEL).render(options)
        result += render(self.TEMPLATE_ENTRY).render(options)
        return result
