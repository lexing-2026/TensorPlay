#include "Tensor.h"
#include "TensorImpl.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "OneDNNContext.h"
#include "Allocator.h"
#include <memory>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <string>
#include <sstream>
#include <chrono>
#include <cstdlib>

#ifdef USE_MKL
#include <mkl.h>
#elif defined(USE_BLAS)
#include <cblas.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_ONEDNN
struct ConvKey {
    int64_t n, ic, ih, iw;
    int64_t oc, kh, kw;
    int64_t oh, ow;
    int64_t sh, sw;
    int64_t ph_t, ph_b, pw_l, pw_r;
    int64_t dh, dw;
    int64_t groups;
    bool has_bias;
    int type; // 0: fwd, 1: bwd_data, 2: bwd_weights

    bool operator==(const ConvKey& other) const {
        return n == other.n && ic == other.ic && ih == other.ih && iw == other.iw &&
               oc == other.oc && kh == other.kh && kw == other.kw &&
               oh == other.oh && ow == other.ow &&
               sh == other.sh && sw == other.sw &&
               ph_t == other.ph_t && ph_b == other.ph_b && pw_l == other.pw_l && pw_r == other.pw_r &&
               dh == other.dh && dw == other.dw &&
               groups == other.groups && has_bias == other.has_bias && type == other.type;
    }
};

namespace std {
    template<> struct hash<ConvKey> {
        size_t operator()(const ConvKey& k) const {
            size_t h = 0;
            auto hc = [&](size_t val) {
                h ^= val + 0x9e3779b9 + (h << 6) + (h >> 2);
            };
            hc(k.n); hc(k.ic); hc(k.ih); hc(k.iw);
            hc(k.oc); hc(k.kh); hc(k.kw);
            hc(k.oh); hc(k.ow);
            hc(k.sh); hc(k.sw);
            hc(k.ph_t); hc(k.ph_b); hc(k.pw_l); hc(k.pw_r);
            hc(k.dh); hc(k.dw);
            hc(k.groups); hc(k.has_bias); hc(k.type);
            return h;
        }
    };
}
#endif

namespace tensorplay {
namespace cpu {

using namespace dnnl;

// Forward declaration of mm_kernel from LinearAlgebraKernels.cpp
Tensor mm_kernel(const Tensor& self, const Tensor& mat2);

// Forward declaration from PadKernels.cpp
Tensor constant_pad_nd_cpu(const Tensor& self, const std::vector<int64_t>& pad, Scalar value);

// Helper for GEMM (reuse from LinearAlgebraKernels logic implicitly or duplicate for now)
// We need a direct gemm call to avoid tensor overhead
void gemm_direct(bool transA, bool transB, int64_t M, int64_t N, int64_t K, float alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float beta, float* C, int64_t ldc) {
    #if defined(USE_MKL) || defined(USE_BLAS)
        cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans, 
                    (int)M, (int)N, (int)K, alpha, A, (int)lda, B, (int)ldb, beta, C, (int)ldc);
    #elif defined(USE_ONEDNN)
        // OneDNN provides dnnl_sgemm which uses standard Fortran (Col-Major) layout
        // To compute C = A * B in Row-Major:
        // C^T (Col-Major) = B^T * A^T (Col-Major)
        // In memory: B^T (Col-Major) is B (Row-Major), A^T is A, C^T is C
        // So we call sgemm(N, M, K, B, A, C)
        // Adjust Transpose flags:
        // We want A * B.
        // If transA=false, A is RowMajor. A^T is ColMajor.
        // We pass A as 'matrix B' to sgemm.
        // If transA=false, we want A (RowMajor) -> A^T (ColMajor). So we pass 'N' for second matrix?
        // Wait. 
        // We want C (RowMajor).
        // sgemm computes C_col = alpha * op(A_col) * op(B_col) + beta * C_col
        // Map: C_col -> C_row (our result)
        // A_col -> B_row (our B input)
        // B_col -> A_row (our A input)
        // So: C_row = alpha * op(B_row) * op(A_row)
        // But we want C = A * B.
        // So we need op(B_row) * op(A_row) = A * B.
        // This implies op(B_row) should be B^T, and op(A_row) should be A^T? No.
        // (B^T * A^T)^T = A * B.
        // So yes, we calculate C^T = B^T * A^T.
        // In sgemm(ColMajor), passing pointer to B (RowMajor) means we passed B^T (ColMajor).
        // Passing pointer to A (RowMajor) means we passed A^T (ColMajor).
        // So if we want B^T * A^T, we just pass NoTrans for both?
        // Let's trace:
        // sgemm('N', 'N', ...) computes RefA * RefB.
        // RefA is matrix at pointer B. Memory matches B^T.
        // RefB is matrix at pointer A. Memory matches A^T.
        // Result is B^T * A^T.
        // Memory at C will contain (B^T * A^T) in ColMajor.
        // Which is (A * B)^T in ColMajor.
        // Which is A * B in RowMajor!
        // So: if transA=false, transB=false: call sgemm('N', 'N', N, M, K, B, A, C)
        
        char ta = transB ? 'T' : 'N'; // Swap A and B
        char tb = transA ? 'T' : 'N';
        dnnl_sgemm(ta, tb, (dnnl_dim_t)N, (dnnl_dim_t)M, (dnnl_dim_t)K, 
                   alpha, B, (dnnl_dim_t)ldb, A, (dnnl_dim_t)lda, 
                   beta, C, (dnnl_dim_t)ldc);
    #else
        // Naive implementation (Supports only NoTrans for now, fallback to mm_kernel if complex)
        if (!transA && !transB) {
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t n = 0; n < N; ++n) {
                    float sum = 0;
                    for (int64_t k = 0; k < K; ++k) {
                        sum += A[m * lda + k] * B[k * ldb + n];
                    }
                    C[m * ldc + n] = beta * C[m * ldc + n] + alpha * sum;
                }
            }
        } else {
             // Fallback for naive transpose not implemented here efficiently
             // Should not happen if we use MKL/BLAS which is required for perf
             TP_THROW(NotImplementedError, "gemm_direct naive transpose not implemented");
        }
    #endif
}

// Helper to expand parameters
std::vector<int64_t> expand_param(const std::vector<int64_t>& param, int64_t expected_dim, const char* name) {
    if (param.size() == 1) {
        return std::vector<int64_t>(expected_dim, param[0]);
    } else if (param.size() == expected_dim) {
        return param;
    } else if (param.size() == expected_dim * 2) {
        // Support asymmetric padding (e.g., 2D padding with 4 values: top, bottom, left, right)
        return param;
    } else {
        TP_THROW(RuntimeError, std::string(name) + " must have size 1 or " + std::to_string(expected_dim) + " or " + std::to_string(expected_dim * 2));
    }
}

// Winograd F(2,3) implementation details
// F(2,3) means: Output 2x2, Filter 3x3, Input 4x4
// B' = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]
// G  = [[1, 0, 0], [1/2, 1/2, 1/2], [1/2, -1/2, 1/2], [0, 0, 1]]
// A' = [[1, 1, 1, 0], [0, 1, -1, -1]]

static void winograd_f23_transform_weight(const float* weight, int64_t K, int64_t C, float* U) {
    // weight: (K, C, 3, 3)
    // U: (16, K, C) - Transformed weights packed for GEMM
    // U[xi][k][c]
    
    // G matrix logic
    // g: 3x3
    // T = G * g * G'
    
    // 16 components
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t c = 0; c < C; ++c) {
            const float* w_ptr = weight + (k * C + c) * 9;
            float g[3][3];
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    g[i][j] = w_ptr[i * 3 + j];

            // Compute G * g
            // Row 0: g0
            // Row 1: (g0 + g1 + g2) * 0.5
            // Row 2: (g0 - g1 + g2) * 0.5
            // Row 3: g2
            float M[4][3];
            for (int j = 0; j < 3; ++j) {
                M[0][j] = g[0][j];
                M[1][j] = (g[0][j] + g[1][j] + g[2][j]) * 0.5f;
                M[2][j] = (g[0][j] - g[1][j] + g[2][j]) * 0.5f;
                M[3][j] = g[2][j];
            }

            // Compute M * G'
            // Col 0: m0
            // Col 1: (m0 + m1 + m2) * 0.5
            // Col 2: (m0 - m1 + m2) * 0.5
            // Col 3: m2
            float UT[4][4];
            for (int i = 0; i < 4; ++i) {
                UT[i][0] = M[i][0];
                UT[i][1] = (M[i][0] + M[i][1] + M[i][2]) * 0.5f;
                UT[i][2] = (M[i][0] - M[i][1] + M[i][2]) * 0.5f;
                UT[i][3] = M[i][2];
            }

            // Scatter to U buffer (16, K, C)
            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    int64_t idx = i * 4 + j;
                    U[idx * K * C + k * C + c] = UT[i][j];
                }
            }
        }
    }
}

static void winograd_f23_transform_input(const float* input, int64_t C, int64_t H, int64_t W, 
                                        int64_t padding_h, int64_t padding_w,
                                        int64_t tile_h, int64_t tile_w, float* V) {
    // input: (C, H, W)
    // V: (16, C, tile_h * tile_w) - Transformed input packed for GEMM
    // V[xi][c][tile_idx]
    
    int64_t num_tiles = tile_h * tile_w;
    int64_t stride_v = C * num_tiles; // Stride between components in V
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t c = 0; c < C; ++c) {
        for (int64_t t = 0; t < num_tiles; ++t) {
            int64_t th = t / tile_w;
            int64_t tw = t % tile_w;
            
            int64_t h_start = th * 2 - padding_h;
            int64_t w_start = tw * 2 - padding_w;
            
            float d[4][4];
            
            // Optimization: Fast path for valid tiles (no boundary checks)
            if (h_start >= 0 && h_start + 3 < H && w_start >= 0 && w_start + 3 < W) {
                const float* in_ptr = input + c * H * W + h_start * W + w_start;
                // Unroll reads
                d[0][0] = in_ptr[0]; d[0][1] = in_ptr[1]; d[0][2] = in_ptr[2]; d[0][3] = in_ptr[3];
                d[1][0] = in_ptr[W]; d[1][1] = in_ptr[W+1]; d[1][2] = in_ptr[W+2]; d[1][3] = in_ptr[W+3];
                d[2][0] = in_ptr[2*W]; d[2][1] = in_ptr[2*W+1]; d[2][2] = in_ptr[2*W+2]; d[2][3] = in_ptr[2*W+3];
                d[3][0] = in_ptr[3*W]; d[3][1] = in_ptr[3*W+1]; d[3][2] = in_ptr[3*W+2]; d[3][3] = in_ptr[3*W+3];
            } else {
                // Slow path
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        int64_t h_idx = h_start + i;
                        int64_t w_idx = w_start + j;
                        if (h_idx >= 0 && h_idx < H && w_idx >= 0 && w_idx < W) {
                            d[i][j] = input[c * H * W + h_idx * W + w_idx];
                        } else {
                            d[i][j] = 0.0f;
                        }
                    }
                }
            }
            
            // Compute T = B' * d
            // B' = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]
            float T[4][4];
            // Manually unroll
            for (int j = 0; j < 4; ++j) {
                T[0][j] = d[0][j] - d[2][j];
                T[1][j] = d[1][j] + d[2][j];
                T[2][j] = d[2][j] - d[1][j];
                T[3][j] = d[1][j] - d[3][j];
            }
            
            // Compute VT = T * B
            // Col 0: t0 - t2 ...
            float VT[4][4];
            for (int i = 0; i < 4; ++i) {
                VT[i][0] = T[i][0] - T[i][2];
                VT[i][1] = T[i][1] + T[i][2];
                VT[i][2] = T[i][2] - T[i][1];
                VT[i][3] = T[i][1] - T[i][3];
            }
            
            // Scatter to V buffer
            int64_t base_idx = c * num_tiles + t;
            float* v_ptr = V + base_idx;
            
            // Unroll scatter
            v_ptr[0 * stride_v] = VT[0][0];
            v_ptr[1 * stride_v] = VT[0][1];
            v_ptr[2 * stride_v] = VT[0][2];
            v_ptr[3 * stride_v] = VT[0][3];
            v_ptr[4 * stride_v] = VT[1][0];
            v_ptr[5 * stride_v] = VT[1][1];
            v_ptr[6 * stride_v] = VT[1][2];
            v_ptr[7 * stride_v] = VT[1][3];
            v_ptr[8 * stride_v] = VT[2][0];
            v_ptr[9 * stride_v] = VT[2][1];
            v_ptr[10 * stride_v] = VT[2][2];
            v_ptr[11 * stride_v] = VT[2][3];
            v_ptr[12 * stride_v] = VT[3][0];
            v_ptr[13 * stride_v] = VT[3][1];
            v_ptr[14 * stride_v] = VT[3][2];
            v_ptr[15 * stride_v] = VT[3][3];
        }
    }
}

static void winograd_f23_transform_output(const float* M, int64_t K, int64_t H_out, int64_t W_out,
                                         int64_t tile_h, int64_t tile_w, float* output) {
    // M: (16, K, num_tiles)
    // output: (K, H_out, W_out)
    
    int64_t num_tiles = tile_h * tile_w;
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t t = 0; t < num_tiles; ++t) {
            // Gather 4x4 matrix MT from M
            // M is (16, K, num_tiles)
            // Stride for 16 components is K * num_tiles
            int64_t stride_m = K * num_tiles;
            int64_t base_idx = k * num_tiles + t;
            
            float MT[4][4];
            const float* m_ptr = M + base_idx;
            
            // Unroll gather
            MT[0][0] = m_ptr[0 * stride_m]; MT[0][1] = m_ptr[1 * stride_m]; MT[0][2] = m_ptr[2 * stride_m]; MT[0][3] = m_ptr[3 * stride_m];
            MT[1][0] = m_ptr[4 * stride_m]; MT[1][1] = m_ptr[5 * stride_m]; MT[1][2] = m_ptr[6 * stride_m]; MT[1][3] = m_ptr[7 * stride_m];
            MT[2][0] = m_ptr[8 * stride_m]; MT[2][1] = m_ptr[9 * stride_m]; MT[2][2] = m_ptr[10 * stride_m]; MT[2][3] = m_ptr[11 * stride_m];
            MT[3][0] = m_ptr[12 * stride_m]; MT[3][1] = m_ptr[13 * stride_m]; MT[3][2] = m_ptr[14 * stride_m]; MT[3][3] = m_ptr[15 * stride_m];
            
            // Y = A' * M * A
            // A' = [[1, 1, 1, 0], [0, 1, -1, -1]]
            
            // Compute T = A' * MT
            // Row 0: m0 + m1 + m2
            // Row 1: m1 - m2 - m3
            float T[2][4];
            // Unroll T computation
            T[0][0] = MT[0][0] + MT[1][0] + MT[2][0]; T[0][1] = MT[0][1] + MT[1][1] + MT[2][1]; T[0][2] = MT[0][2] + MT[1][2] + MT[2][2]; T[0][3] = MT[0][3] + MT[1][3] + MT[2][3];
            T[1][0] = MT[1][0] - MT[2][0] - MT[3][0]; T[1][1] = MT[1][1] - MT[2][1] - MT[3][1]; T[1][2] = MT[1][2] - MT[2][2] - MT[3][2]; T[1][3] = MT[1][3] - MT[2][3] - MT[3][3];
            
            // Compute Y = T * A
            // Col 0: t0 + t1 + t2
            // Col 1: t1 - t2 - t3
            float Y[2][2];
            // Unroll Y computation
            Y[0][0] = T[0][0] + T[0][1] + T[0][2]; Y[0][1] = T[0][1] - T[0][2] - T[0][3];
            Y[1][0] = T[1][0] + T[1][1] + T[1][2]; Y[1][1] = T[1][1] - T[1][2] - T[1][3];
            
            // Write to output
            int64_t th = t / tile_w;
            int64_t tw = t % tile_w;
            int64_t h_start = th * 2;
            int64_t w_start = tw * 2;
            
            // Fast path for valid tiles
            if (h_start + 1 < H_out && w_start + 1 < W_out) {
                float* out_ptr = output + k * H_out * W_out + h_start * W_out + w_start;
                out_ptr[0] = Y[0][0]; out_ptr[1] = Y[0][1];
                out_ptr[W_out] = Y[1][0]; out_ptr[W_out+1] = Y[1][1];
            } else {
                // Slow path
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        int64_t h_idx = h_start + i;
                        int64_t w_idx = w_start + j;
                        if (h_idx < H_out && w_idx < W_out) {
                            output[k * H_out * W_out + h_idx * W_out + w_idx] = Y[i][j];
                        }
                    }
                }
            }
        }
    }
}

static void conv2d_winograd_3x3(const Tensor& input, const Tensor& weight, const Tensor& bias,
                               int64_t pH, int64_t pW, Tensor& output) {
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t K = weight.size(0);
    int64_t H_out = output.size(2);
    int64_t W_out = output.size(3);
    
    int64_t tile_h = (H_out + 1) / 2;
    int64_t tile_w = (W_out + 1) / 2;
    int64_t num_tiles = tile_h * tile_w;
    
    // Allocate buffers
    // U: 16 * K * C
    // V: 16 * C * num_tiles (per image)
    // M: 16 * K * num_tiles (per image)
    
    std::unique_ptr<float[]> U_buf(new float[16 * K * C]);
    winograd_f23_transform_weight(weight.data_ptr<float>(), K, C, U_buf.get());
    
    // Per-image processing
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        std::unique_ptr<float[]> V_buf(new float[16 * C * num_tiles]);
        std::unique_ptr<float[]> M_buf(new float[16 * K * num_tiles]);
        
        #ifdef _OPENMP
        #pragma omp for
        #endif
        for (int64_t n = 0; n < N; ++n) {
            // 1. Transform Input
            const float* in_ptr = input.data_ptr<float>() + n * C * H * W;
            winograd_f23_transform_input(in_ptr, C, H, W, pH, pW, tile_h, tile_w, V_buf.get());
            
            // 2. Batched GEMM (16x)
            for (int i = 0; i < 16; ++i) {
                // M[i] = U[i] * V[i]
                // U[i]: K x C
                // V[i]: C x num_tiles
                // M[i]: K x num_tiles
                const float* u_ptr = U_buf.get() + i * K * C;
                const float* v_ptr = V_buf.get() + i * C * num_tiles;
                float* m_ptr = M_buf.get() + i * K * num_tiles;
                
                gemm_direct(false, false, K, num_tiles, C, 
                            1.0f, u_ptr, C, 
                            v_ptr, num_tiles, 
                            0.0f, m_ptr, num_tiles);
            }
            
            // 3. Transform Output
            float* out_ptr = output.data_ptr<float>() + n * K * H_out * W_out;
            winograd_f23_transform_output(M_buf.get(), K, H_out, W_out, tile_h, tile_w, out_ptr);
            
            // 4. Add Bias
            if (bias.defined() && bias.numel() > 0) {
                const float* b_ptr = bias.data_ptr<float>();
                for (int64_t k = 0; k < K; ++k) {
                    float b = b_ptr[k];
                    float* out_k = out_ptr + k * H_out * W_out;
                    for (int64_t idx = 0; idx < H_out * W_out; ++idx) {
                        out_k[idx] += b;
                    }
                }
            }
        }
    }
}

// Im2Col implementation
template <typename T>
void im2col(const T* data_im, int64_t channels, int64_t height, int64_t width,
            int64_t kernel_h, int64_t kernel_w, int64_t pad_h, int64_t pad_w,
            int64_t stride_h, int64_t stride_w, int64_t dilation_h, int64_t dilation_w,
            T* data_col) {
    int64_t height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int64_t width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int64_t channels_col = channels * kernel_h * kernel_w;
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t c = 0; c < channels_col; ++c) {
        int64_t w_offset = c % kernel_w;
        int64_t h_offset = (c / kernel_w) % kernel_h;
        int64_t c_im = c / kernel_h / kernel_w;
        
        for (int64_t h = 0; h < height_col; ++h) {
            for (int64_t w = 0; w < width_col; ++w) {
                int64_t h_pad = h * stride_h - pad_h + h_offset * dilation_h;
                int64_t w_pad = w * stride_w - pad_w + w_offset * dilation_w;
                
                if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                    data_col[(c * height_col + h) * width_col + w] =
                        data_im[(c_im * height + h_pad) * width + w_pad];
                else
                    data_col[(c * height_col + h) * width_col + w] = 0;
            }
        }
    }
}

// Col2Im implementation
template <typename T>
void col2im(const T* data_col, int64_t channels, int64_t height, int64_t width,
            int64_t kernel_h, int64_t kernel_w, int64_t pad_h, int64_t pad_w,
            int64_t stride_h, int64_t stride_w, int64_t dilation_h, int64_t dilation_w,
            T* data_im) {
    std::memset(data_im, 0, height * width * channels * sizeof(T));
    
    int64_t height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int64_t width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t c_im = 0; c_im < channels; ++c_im) {
        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                int64_t c = (c_im * kernel_h + kh) * kernel_w + kw;
                int64_t h_offset = kh;
                int64_t w_offset = kw;
                
                for (int64_t h = 0; h < height_col; ++h) {
                    for (int64_t w = 0; w < width_col; ++w) {
                        int64_t h_pad = h * stride_h - pad_h + h_offset * dilation_h;
                        int64_t w_pad = w * stride_w - pad_w + w_offset * dilation_w;
                        
                        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                            data_im[(c_im * height + h_pad) * width + w_pad] +=
                                data_col[(c * height_col + h) * width_col + w];
                    }
                }
            }
        }
    }
}

// Im2Col 3D implementation
template <typename T>
void im2col3d(const T* data_im, int64_t channels, int64_t depth, int64_t height, int64_t width,
            int64_t kernel_d, int64_t kernel_h, int64_t kernel_w,
            int64_t pad_d, int64_t pad_h, int64_t pad_w,
            int64_t stride_d, int64_t stride_h, int64_t stride_w,
            int64_t dilation_d, int64_t dilation_h, int64_t dilation_w,
            T* data_col) {
    int64_t depth_col = (depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1;
    int64_t height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int64_t width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    int64_t channels_col = channels * kernel_d * kernel_h * kernel_w;
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t c = 0; c < channels_col; ++c) {
        int64_t w_offset = c % kernel_w;
        int64_t h_offset = (c / kernel_w) % kernel_h;
        int64_t d_offset = (c / kernel_w / kernel_h) % kernel_d;
        int64_t c_im = c / kernel_w / kernel_h / kernel_d;
        
        for (int64_t d = 0; d < depth_col; ++d) {
            for (int64_t h = 0; h < height_col; ++h) {
                for (int64_t w = 0; w < width_col; ++w) {
                    int64_t d_pad = d * stride_d - pad_d + d_offset * dilation_d;
                    int64_t h_pad = h * stride_h - pad_h + h_offset * dilation_h;
                    int64_t w_pad = w * stride_w - pad_w + w_offset * dilation_w;
                    
                    if (d_pad >= 0 && d_pad < depth && h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                        data_col[((c * depth_col + d) * height_col + h) * width_col + w] =
                            data_im[((c_im * depth + d_pad) * height + h_pad) * width + w_pad];
                    else
                        data_col[((c * depth_col + d) * height_col + h) * width_col + w] = 0;
                }
            }
        }
    }
}

// Col2Im 3D implementation
template <typename T>
void col2im3d(const T* data_col, int64_t channels, int64_t depth, int64_t height, int64_t width,
            int64_t kernel_d, int64_t kernel_h, int64_t kernel_w,
            int64_t pad_d, int64_t pad_h, int64_t pad_w,
            int64_t stride_d, int64_t stride_h, int64_t stride_w,
            int64_t dilation_d, int64_t dilation_h, int64_t dilation_w,
            T* data_im) {
    std::memset(data_im, 0, depth * height * width * channels * sizeof(T));
    
    int64_t depth_col = (depth + 2 * pad_d - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1;
    int64_t height_col = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    int64_t width_col = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int64_t c_im = 0; c_im < channels; ++c_im) {
        for (int64_t kd = 0; kd < kernel_d; ++kd) {
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
                for (int64_t kw = 0; kw < kernel_w; ++kw) {
                    int64_t c = ((c_im * kernel_d + kd) * kernel_h + kh) * kernel_w + kw;
                    int64_t d_offset = kd;
                    int64_t h_offset = kh;
                    int64_t w_offset = kw;
                    
                    for (int64_t d = 0; d < depth_col; ++d) {
                        for (int64_t h = 0; h < height_col; ++h) {
                            for (int64_t w = 0; w < width_col; ++w) {
                                int64_t d_pad = d * stride_d - pad_d + d_offset * dilation_d;
                                int64_t h_pad = h * stride_h - pad_h + h_offset * dilation_h;
                                int64_t w_pad = w * stride_w - pad_w + w_offset * dilation_w;
                                
                                if (d_pad >= 0 && d_pad < depth && h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
                                    data_im[((c_im * depth + d_pad) * height + h_pad) * width + w_pad] +=
                                        data_col[((c * depth_col + d) * height_col + h) * width_col + w];
                            }
                        }
                    }
                }
            }
        }
    }
}

#ifdef USE_ONEDNN
#include <omp.h>
using namespace dnnl;

static dnnl::algorithm get_onednn_algo(int64_t kh, int64_t kw) {
    const char* env_p = std::getenv("TP_ONEDNN_ALGO");
    if (env_p) {
        std::string algo(env_p);
        if (algo == "direct") return algorithm::convolution_direct;
        if (algo == "winograd") return algorithm::convolution_winograd;
        if (algo == "auto") return algorithm::convolution_auto;
    }
    // Heuristic: Prefer Winograd for 3x3 convolutions
    if (kh == 3 && kw == 3) {
        return algorithm::convolution_winograd;
    }
    // Heuristic: Prefer MatMul for 1x1 convolutions (Projection)
    return algorithm::convolution_auto;
}

static bool conv2d_onednn(const Tensor& input, const Tensor& weight, const Tensor& bias,
                         const std::vector<int64_t>& stride, 
                         int64_t pH_top, int64_t pH_bottom, int64_t pW_left, int64_t pW_right,
                         const std::vector<int64_t>& dilation, int64_t groups,
                         Tensor& output) {
    
    if (!OneDNNContext::is_enabled()) {
        return false;
    }
    if (input.dtype() != DType::Float32) return false;
    
    try {
        auto& eng = OneDNNContext::get_engine();
        auto& s = OneDNNContext::get_stream();
        
        // Cache key generation
        ConvKey key;
        key.n = input.size(0); key.ic = input.size(1); key.ih = input.size(2); key.iw = input.size(3);
        key.oc = output.size(1); key.kh = weight.size(2); key.kw = weight.size(3);
        key.oh = output.size(2); key.ow = output.size(3);
        key.sh = stride[0]; key.sw = stride[1];
        key.ph_t = pH_top; key.ph_b = pH_bottom; key.pw_l = pW_left; key.pw_r = pW_right;
        key.dh = dilation[0]; key.dw = dilation[1];
        key.groups = groups;
        key.has_bias = (bias.defined() && bias.numel() > 0);
        key.type = 0; // Forward
        
        struct CachedConv {
            convolution_forward::primitive_desc pd;
            convolution_forward prim;
            std::vector<std::pair<memory::desc, reorder>> reorder_input_cache;
            std::vector<std::pair<memory::desc, reorder>> reorder_weights_cache;
        };

        static std::unordered_map<ConvKey, CachedConv> cache;
        static std::mutex mtx;
        
        bool found = false;
        CachedConv cached_entry; 
        
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto it = cache.find(key);
            if (it != cache.end()) {
                cached_entry = it->second;
                found = true;
            }
        }
        
        memory::dims src_dims = {input.size(0), input.size(1), input.size(2), input.size(3)};
        memory::dims dst_dims = {output.size(0), output.size(1), output.size(2), output.size(3)};
        
        memory::dims weights_dims;
        if (groups > 1) {
            weights_dims = {groups, weight.size(0)/groups, weight.size(1), weight.size(2), weight.size(3)};
        } else {
            weights_dims = {weight.size(0), weight.size(1), weight.size(2), weight.size(3)};
        }
        
            // Use ANY format to allow OneDNN to choose the best blocked layout
            // For 1x1 convolutions, we used to force NHWC, but this causes reorders if previous layer is blocked.
            // Let OneDNN decide globally.
            bool is_1x1 = (key.kh == 1 && key.kw == 1 && key.sh == 1 && key.sw == 1 && key.ph_t == 0 && key.ph_b == 0 && key.pw_l == 0 && key.pw_r == 0);
            auto src_tag = memory::format_tag::any;
            auto dst_tag = memory::format_tag::any;
            
            auto src_md = memory::desc(src_dims, memory::data_type::f32, src_tag);
            auto dst_md = memory::desc(dst_dims, memory::data_type::f32, dst_tag);
            
            auto weights_md = groups > 1 
                ? memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::any)
                : memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::any);
            
            memory::desc bias_md;
            if (bias.defined() && bias.numel() > 0) {
                 int64_t b_sz = (bias.dim() == 0) ? 1 : bias.size(0);
                 bias_md = memory::desc({b_sz}, memory::data_type::f32, memory::format_tag::any); 
            } else {
                 bias_md = memory::desc();
            }
            
            if (!found) {
                memory::dims strides_dims = {stride[0], stride[1]};
                memory::dims padding_l_dims = {pH_top, pW_left};
                memory::dims padding_r_dims = {pH_bottom, pW_right};
                memory::dims dilates_dims = {dilation[0] - 1, dilation[1] - 1};
                
                dnnl::algorithm algo = get_onednn_algo(key.kh, key.kw);
                convolution_forward::primitive_desc conv_pd;
                
                // Try selected algorithm first, fallback to auto if it fails
                try {
                    conv_pd = convolution_forward::primitive_desc(
                        eng,
                        prop_kind::forward_inference, algo,
                        src_md, weights_md, bias_md, dst_md,
                        strides_dims, dilates_dims, padding_l_dims, padding_r_dims);
                } catch (dnnl::error& e) {
                    if (algo != algorithm::convolution_auto) {
                        conv_pd = convolution_forward::primitive_desc(
                            eng,
                            prop_kind::forward_inference, algorithm::convolution_auto,
                            src_md, weights_md, bias_md, dst_md,
                            strides_dims, dilates_dims, padding_l_dims, padding_r_dims);
                    } else {
                        throw;
                    }
                }
            
            auto conv = convolution_forward(conv_pd);
            
            cached_entry = {conv_pd, conv};

            {
                std::lock_guard<std::mutex> lock(mtx);
                cache.insert({key, cached_entry});
            }
        }
        
        auto& conv = cached_entry.prim;
        auto& pd = cached_entry.pd;
        auto expected_src_md = pd.src_desc();
        auto expected_weights_md = pd.weights_desc();
        auto expected_dst_md = pd.dst_desc();
        
        // 1. Input Reorder
        memory src_mem;
        memory::desc user_src_md;
        
        if (input.unsafeGetTensorImpl()->has_onednn_md()) {
             auto stored_md = std::static_pointer_cast<memory::desc>(input.unsafeGetTensorImpl()->get_onednn_md());
             user_src_md = *stored_md;
        } else {
             user_src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
        }

        auto user_src_mem = memory(user_src_md, eng, input.data_ptr<float>());
        
        if (expected_src_md != user_src_md) {             
             size_t req_size = expected_src_md.get_size();
             Allocator* allocator = getAllocator(input.device().type());
             Storage new_storage(req_size, allocator);

             src_mem = memory(expected_src_md, eng, new_storage.data());
             
             reorder r_prim;
             bool r_found = false;
             for (const auto& item : cached_entry.reorder_input_cache) {
                 if (item.first == user_src_md) {
                     r_prim = item.second;
                     r_found = true;
                     break;
                 }
             }
             
             if (!r_found) {
                 r_prim = reorder(user_src_mem, src_mem);
                 cached_entry.reorder_input_cache.push_back({user_src_md, r_prim});
                 {
                     std::lock_guard<std::mutex> lock(mtx);
                     auto it = cache.find(key);
                     if (it != cache.end()) {
                         it->second.reorder_input_cache.push_back({user_src_md, r_prim});
                     }
                 }
             }
             r_prim.execute(s, user_src_mem, src_mem);

             input.unsafeGetTensorImpl()->set_storage(new_storage);
             input.unsafeGetTensorImpl()->set_onednn_md(std::make_shared<memory::desc>(expected_src_md));
        } else {
             src_mem = user_src_mem;
        }

        // 2. Weights
        memory weights_mem;
        
        if (weight.unsafeGetTensorImpl()->has_onednn_md()) {
             auto stored_md = std::static_pointer_cast<memory::desc>(weight.unsafeGetTensorImpl()->get_onednn_md());
             if (*stored_md == expected_weights_md) {
                 weights_mem = memory(*stored_md, eng, weight.data_ptr<float>());
             } else {
                 // Reorder from stored to expected
                 auto src_mem = memory(*stored_md, eng, weight.data_ptr<float>());
                 
                 size_t required_size = expected_weights_md.get_size();
                 if (weight.numel() * sizeof(float) < required_size) {
                      Allocator* allocator = getAllocator(weight.device().type());
                      Storage new_storage(required_size, allocator);
                      
                      // Use temporary buffer for new storage to avoid overwriting source if it overlaps (it doesn't here, but good practice)
                      // Actually we need to keep old storage alive for src_mem until reorder is done.
                      // src_mem holds pointer to old storage.
                      
                      weights_mem = memory(expected_weights_md, eng, new_storage.data());
                      reorder(src_mem, weights_mem).execute(s, src_mem, weights_mem);
                      
                      weight.unsafeGetTensorImpl()->set_storage(new_storage);
                 } else {
                      // Reuse storage? If layout is different, in-place reorder is risky/impossible if data is scrambled.
                      // Safer to always allocate new storage for simplicity when changing layout.
                      Allocator* allocator = getAllocator(weight.device().type());
                      Storage new_storage(required_size, allocator);
                      
                      weights_mem = memory(expected_weights_md, eng, new_storage.data());
                      reorder(src_mem, weights_mem).execute(s, src_mem, weights_mem);
                      
                      weight.unsafeGetTensorImpl()->set_storage(new_storage);
                 }
                 weight.unsafeGetTensorImpl()->set_onednn_md(std::make_shared<memory::desc>(expected_weights_md));
             }
        } else {
             auto user_weights_md = groups > 1 
                 ? memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::goihw)
                 : memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);
                 
             if (expected_weights_md != user_weights_md) {
                 auto user_mem = memory(user_weights_md, eng, weight.data_ptr<float>());
                 
                 size_t required_size = expected_weights_md.get_size();
                 Allocator* allocator = getAllocator(weight.device().type());
                 Storage new_storage(required_size, allocator);
                 
                 weights_mem = memory(expected_weights_md, eng, new_storage.data());
                 reorder(user_mem, weights_mem).execute(s, user_mem, weights_mem);
                 
                 weight.unsafeGetTensorImpl()->set_storage(new_storage);
                 weight.unsafeGetTensorImpl()->set_onednn_md(std::make_shared<memory::desc>(expected_weights_md));
             } else {
                 weights_mem = memory(user_weights_md, eng, weight.data_ptr<float>());
             }
        }
        
        // 3. Output
        memory dst_mem;
        auto user_dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);
        
        // If expected layout is different from NCHW, we need a temporary buffer
        bool need_reorder_dst = (expected_dst_md != user_dst_md);
        Storage blocked_storage_handle; // Keep alive

        if (need_reorder_dst) {
             size_t required_size = expected_dst_md.get_size();
             Allocator* allocator = getAllocator(output.device().type());
             blocked_storage_handle = Storage(required_size, allocator);
             dst_mem = memory(expected_dst_md, eng, blocked_storage_handle.data());
        } else {
             dst_mem = memory(expected_dst_md, eng, output.data_ptr<float>());
        }
        
        std::unordered_map<int, memory> args;
        args.insert({DNNL_ARG_SRC, src_mem});
        args.insert({DNNL_ARG_WEIGHTS, weights_mem});
        args.insert({DNNL_ARG_DST, dst_mem});
        
        if (bias.defined() && bias.numel() > 0) {
            auto user_bias_md = memory::desc({bias.size(0)}, memory::data_type::f32, memory::format_tag::x);
            auto expected_bias_md = pd.bias_desc();
            memory bias_mem;
            
            if (expected_bias_md != user_bias_md) {
                 auto user_bias_mem = memory(user_bias_md, eng, bias.data_ptr<float>());
                 bias_mem = memory(expected_bias_md, eng);
                 reorder(user_bias_mem, bias_mem).execute(s, user_bias_mem, bias_mem);
            } else {
                 bias_mem = memory(user_bias_md, eng, bias.data_ptr<float>());
            }
            args.insert({DNNL_ARG_BIAS, bias_mem});
        }
        
        conv.execute(s, args);
        
        if (need_reorder_dst) {
             auto user_dst_mem = memory(user_dst_md, eng, output.data_ptr<float>());
             reorder(dst_mem, user_dst_mem).execute(s, dst_mem, user_dst_mem);
        }
        
        s.wait();
        
        // Clear OneDNN MD if any (ensure it's treated as NCHW)
        if (output.unsafeGetTensorImpl()->has_onednn_md()) {
            output.unsafeGetTensorImpl()->set_onednn_md(nullptr);
        }
        
        return true;
    } catch (dnnl::error& e) {
        return false;
    }
    catch (...) {
        return false;
    }
}

static bool conv3d_onednn(const Tensor& input, const Tensor& weight, const Tensor& bias,
                         const std::vector<int64_t>& stride, 
                         int64_t pD_front, int64_t pD_back,
                         int64_t pH_top, int64_t pH_bottom, 
                         int64_t pW_left, int64_t pW_right,
                         const std::vector<int64_t>& dilation, int64_t groups,
                         Tensor& output) {
    
    if (!OneDNNContext::is_enabled()) return false;
    if (input.dtype() != DType::Float32) return false;
    
    try {
        auto& eng = OneDNNContext::get_engine();
        auto& s = OneDNNContext::get_stream();
        
        memory::dims src_dims = {input.size(0), input.size(1), input.size(2), input.size(3)};
        memory::dims dst_dims = {output.size(0), output.size(1), output.size(2), output.size(3)};
        
        memory::dims weights_dims;
        if (groups > 1) {
            weights_dims = {groups, weight.size(0)/groups, weight.size(1), weight.size(2), weight.size(3)};
        } else {
            weights_dims = {weight.size(0), weight.size(1), weight.size(2), weight.size(3)};
        }
        
        memory::dims strides_dims = {stride[0], stride[1]};
        memory::dims padding_l_dims = {pH_top, pW_left};
        memory::dims padding_r_dims = {pH_bottom, pW_right};
        memory::dims dilates_dims = {dilation[0] - 1, dilation[1] - 1};
        
        auto src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::ncdhw);
        auto dst_tag = memory::format_tag::any;
        auto dst_md = memory::desc(dst_dims, memory::data_type::f32, dst_tag);
        
        auto weights_md = groups > 1 
            ? memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::goidhw)
            : memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oidhw);
            
        auto bias_md = (bias.defined() && bias.numel() > 0)
            ? memory::desc({bias.size(0)}, memory::data_type::f32, memory::format_tag::x)
            : memory::desc();
            
        auto conv_pd = convolution_forward::primitive_desc(
            eng,
            prop_kind::forward_inference, algorithm::convolution_auto,
            src_md, weights_md, bias_md, dst_md,
            strides_dims, dilates_dims, padding_l_dims, padding_r_dims);
        
        auto expected_dst_md = conv_pd.dst_desc();
        auto user_dst_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::ncdhw);
        bool need_reorder_dst = (expected_dst_md != user_dst_md);
        
        memory dst_mem;
        Storage blocked_storage_handle;
        
        if (need_reorder_dst) {
             size_t required_size = expected_dst_md.get_size();
             Allocator* allocator = getAllocator(output.device().type());
             blocked_storage_handle = Storage(required_size, allocator);
             dst_mem = memory(expected_dst_md, eng, blocked_storage_handle.data());
        } else {
             dst_mem = memory(expected_dst_md, eng, output.data_ptr<float>());
        }

        auto src_mem = memory(src_md, eng, input.data_ptr<float>());
        auto weights_mem = memory(weights_md, eng, weight.data_ptr<float>());
        
        convolution_forward conv(conv_pd);
        
        std::unordered_map<int, memory> args;
        args.insert({DNNL_ARG_SRC, src_mem});
        args.insert({DNNL_ARG_WEIGHTS, weights_mem});
        args.insert({DNNL_ARG_DST, dst_mem});
        
        if (bias.defined() && bias.numel() > 0) {
            auto bias_mem = memory(bias_md, eng, bias.data_ptr<float>());
            args.insert({DNNL_ARG_BIAS, bias_mem});
        }
        
        conv.execute(s, args);
        
        if (need_reorder_dst) {
             auto user_dst_mem = memory(user_dst_md, eng, output.data_ptr<float>());
             reorder(dst_mem, user_dst_mem).execute(s, dst_mem, user_dst_mem);
        }
        
        s.wait();
        
        if (output.unsafeGetTensorImpl()->has_onednn_md()) {
            output.unsafeGetTensorImpl()->set_onednn_md(nullptr);
        }
        
        return true;
    } catch (dnnl::error& e) {
        return false;
    }
    catch (...) {
        return false;
    }
}

static bool conv2d_grad_input_onednn(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                                    const std::vector<int64_t>& stride,
                                    int64_t pH_top, int64_t pH_bottom, int64_t pW_left, int64_t pW_right,
                                    const std::vector<int64_t>& dilation, int64_t groups,
                                    Tensor& grad_input) {
    if (!OneDNNContext::is_enabled()) {
        return false;
    }
    // std::cout << "DEBUG: conv2d_grad_input_onednn called" << std::endl;
    if (input.dtype() != DType::Float32) return false;

    // Ensure contiguous or blocked
    Tensor input_c = (input.is_contiguous() || input.unsafeGetTensorImpl()->has_onednn_md()) ? input : input.clone();
    Tensor weight_c = (weight.is_contiguous() || weight.unsafeGetTensorImpl()->has_onednn_md()) ? weight : weight.clone();
    Tensor grad_output_c = (grad_output.is_contiguous() || grad_output.unsafeGetTensorImpl()->has_onednn_md()) ? grad_output : grad_output.clone();

    try {
        auto& eng = OneDNNContext::get_engine();
        auto& s = OneDNNContext::get_stream();

        // Cache Key
        ConvKey key;
        key.n = input_c.size(0); key.ic = input_c.size(1); key.ih = input_c.size(2); key.iw = input_c.size(3);
        key.oc = grad_output_c.size(1); key.kh = weight.size(2); key.kw = weight.size(3);
        key.oh = grad_output_c.size(2); key.ow = grad_output_c.size(3);
        key.sh = stride[0]; key.sw = stride[1];
        key.ph_t = pH_top; key.ph_b = pH_bottom; key.pw_l = pW_left; key.pw_r = pW_right;
        key.dh = dilation[0]; key.dw = dilation[1];
        key.groups = groups;
        key.has_bias = false;
        key.type = 1; // BwdData

        struct CachedConvBwdData {
            convolution_backward_data::primitive_desc pd;
            convolution_backward_data prim;
            std::vector<std::pair<memory::desc, reorder>> reorder_grad_output_cache;
            std::vector<std::pair<memory::desc, reorder>> reorder_weight_cache;
        };

        static std::unordered_map<ConvKey, CachedConvBwdData> cache;
        static std::mutex mtx;
        
        bool found = false;
        CachedConvBwdData cached_entry;
        
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto it = cache.find(key);
            if (it != cache.end()) {
                cached_entry = it->second;
                found = true;
            }
        }

        memory::dims src_dims = {input_c.size(0), input_c.size(1), input_c.size(2), input_c.size(3)};
        memory::dims dst_dims = {grad_output_c.size(0), grad_output_c.size(1), grad_output_c.size(2), grad_output_c.size(3)};
        
        memory::dims weights_dims;
        if (groups > 1) {
            weights_dims = {groups, weight.size(0)/groups, weight.size(1), weight.size(2), weight.size(3)};
        } else {
            weights_dims = {weight.size(0), weight.size(1), weight.size(2), weight.size(3)};
        }

        auto strides_dims = memory::dims{stride[0], stride[1]};
        auto padding_l_dims = memory::dims{pH_top, pW_left};
        auto padding_r_dims = memory::dims{pH_bottom, pW_right};
        auto dilates_dims = memory::dims{dilation[0] - 1, dilation[1] - 1};

        // For 1x1 convolutions, use NHWC to avoid reorders and match Forward
        bool is_1x1 = (key.kh == 1 && key.kw == 1 && key.sh == 1 && key.sw == 1 && key.ph_t == 0 && key.ph_b == 0 && key.pw_l == 0 && key.pw_r == 0);
        auto src_tag = is_1x1 ? memory::format_tag::nhwc : memory::format_tag::any;
        auto dst_tag = is_1x1 ? memory::format_tag::nhwc : memory::format_tag::any;

        auto src_md = memory::desc(src_dims, memory::data_type::f32, src_tag);
        auto dst_md = memory::desc(dst_dims, memory::data_type::f32, dst_tag);
        auto weights_md = groups > 1 
            ? memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::any)
            : memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::any);

        if (!found) {
             // Forward PD (Hint) - must use ANY too
             dnnl::algorithm algo = get_onednn_algo(key.kh, key.kw);
             
             convolution_forward::primitive_desc fwd_pd;
             try {
                fwd_pd = convolution_forward::primitive_desc(
                    eng, prop_kind::forward_inference, algo,
                    src_md, weights_md, memory::desc(), dst_md,
                    strides_dims, dilates_dims, padding_l_dims, padding_r_dims);
             } catch (...) {
                 fwd_pd = convolution_forward::primitive_desc(
                    eng, prop_kind::forward_inference, algorithm::convolution_auto,
                    src_md, weights_md, memory::desc(), dst_md,
                    strides_dims, dilates_dims, padding_l_dims, padding_r_dims);
                 algo = algorithm::convolution_auto; // Fallback algo for backward too
             }

             // Backward Data PD
             convolution_backward_data::primitive_desc bwd_d_pd;
             try {
                 bwd_d_pd = convolution_backward_data::primitive_desc(
                     eng, algo,
                     src_md, weights_md, dst_md,
                     strides_dims, dilates_dims, padding_l_dims, padding_r_dims,
                     fwd_pd);
             } catch (...) {
                 if (algo != algorithm::convolution_auto) {
                      bwd_d_pd = convolution_backward_data::primitive_desc(
                         eng, algorithm::convolution_auto,
                         src_md, weights_md, dst_md,
                         strides_dims, dilates_dims, padding_l_dims, padding_r_dims,
                         fwd_pd);
                 } else {
                     throw;
                 }
             }

             auto bwd_d = convolution_backward_data(bwd_d_pd);
             cached_entry = {bwd_d_pd, bwd_d};
             
             {
                 std::lock_guard<std::mutex> lock(mtx);
                 cache.insert({key, cached_entry});
             }
        }

        auto& bwd_d = cached_entry.prim;
        auto& pd = cached_entry.pd;
        
        auto expected_diff_dst_md = pd.diff_dst_desc();
        auto expected_weights_md = pd.weights_desc();
        auto expected_diff_src_md = pd.diff_src_desc();

        // Prepare Diff Dst (grad_output)
        memory diff_dst_mem;
        bool loaded_from_cache = false;
        
        // Try to use cached memory object from SharedState to avoid redundant reorder
        if (grad_output_c.unsafeGetTensorImpl()->has_onednn_memory_cache()) {
             auto cached_mem_ptr = std::static_pointer_cast<memory>(grad_output_c.unsafeGetTensorImpl()->get_onednn_memory_cache());
             // Check if the cached memory descriptor matches what we need
             if (cached_mem_ptr && cached_mem_ptr->get_desc() == expected_diff_dst_md) {
                 diff_dst_mem = *cached_mem_ptr;
                 loaded_from_cache = true;
             }
        }

        if (!loaded_from_cache) {
            if (grad_output_c.unsafeGetTensorImpl()->has_onednn_md()) {
                 auto stored_md = std::static_pointer_cast<memory::desc>(grad_output_c.unsafeGetTensorImpl()->get_onednn_md());
                 if (*stored_md == expected_diff_dst_md) {
                     diff_dst_mem = memory(*stored_md, eng, grad_output_c.data_ptr<float>());
                 } else {
                     // std::cout << "DEBUG: Reordering grad_output (stored) in conv2d_grad_input_onednn" << std::endl;
                     auto src = memory(*stored_md, eng, grad_output_c.data_ptr<float>());
                     diff_dst_mem = memory(expected_diff_dst_md, eng);
                     reorder(src, diff_dst_mem).execute(s, src, diff_dst_mem);
                 }
            } else {
                 auto user_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);
                 auto user_mem = memory(user_md, eng, grad_output_c.data_ptr<float>());
                 if (user_md != expected_diff_dst_md) {
                     // std::cout << "DEBUG: Reordering grad_output (NCHW) in conv2d_grad_input_onednn" << std::endl;
                     
                     // Optimize: Update grad_output storage to blocked format to avoid re-reordering in grad_weight
                    // DISABLED: Modifying input tensor storage breaks other consumers (e.g. grad_bias, Python view)
                    // We just use a temporary reordered buffer (owned by diff_dst_mem) and cache it.
                    
                    diff_dst_mem = memory(expected_diff_dst_md, eng);

                     reorder r_prim;
                     bool r_found = false;
                     for (const auto& item : cached_entry.reorder_grad_output_cache) {
                         if (item.first == user_md) {
                             r_prim = item.second;
                             r_found = true;
                             break;
                         }
                     }
                     
                     if (!r_found) {
                        r_prim = reorder(user_mem, diff_dst_mem);
                        cached_entry.reorder_grad_output_cache.push_back({user_md, r_prim});
                        {
                            std::lock_guard<std::mutex> lock(mtx);
                            auto it = cache.find(key);
                            if (it != cache.end()) {
                                it->second.reorder_grad_output_cache.push_back({user_md, r_prim});
                            }
                        }
                    }
                    auto start_r = std::chrono::high_resolution_clock::now();
                    r_prim.execute(s, user_mem, diff_dst_mem);
                    auto end_r = std::chrono::high_resolution_clock::now();
                    // std::cout << "DEBUG: GradOutput Reorder Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end_r - start_r).count() << " us" << std::endl;

                    // Update the tensor - DISABLED
                   // grad_output_c.unsafeGetTensorImpl()->set_storage(new_storage);
                   // grad_output_c.unsafeGetTensorImpl()->set_onednn_md(std::make_shared<memory::desc>(expected_diff_dst_md));
                 } else {
                     diff_dst_mem = user_mem;
                 }
            }
            // Cache the memory object for other backward functions (e.g. grad_weight)
            grad_output_c.unsafeGetTensorImpl()->set_onednn_memory_cache(std::make_shared<memory>(diff_dst_mem));
        }

        // Prepare Weights
        memory weights_mem;
        if (weight_c.unsafeGetTensorImpl()->has_onednn_md()) {
             auto stored_md = std::static_pointer_cast<memory::desc>(weight_c.unsafeGetTensorImpl()->get_onednn_md());
             if (*stored_md == expected_weights_md) {
                 weights_mem = memory(*stored_md, eng, weight_c.data_ptr<float>());
             } else {
                 auto src = memory(*stored_md, eng, weight_c.data_ptr<float>());
                 weights_mem = memory(expected_weights_md, eng);
                 reorder(src, weights_mem).execute(s, src, weights_mem);
             }
        } else {
             auto user_md = groups > 1 
                 ? memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::goihw)
                 : memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);
             auto user_mem = memory(user_md, eng, weight_c.data_ptr<float>());
             if (user_md != expected_weights_md) {
                 weights_mem = memory(expected_weights_md, eng);

                 reorder r_prim;
                 bool r_found = false;
                 for (const auto& item : cached_entry.reorder_weight_cache) {
                     if (item.first == user_md) {
                         r_prim = item.second;
                         r_found = true;
                         break;
                     }
                 }
                 
                 if (!r_found) {
                     r_prim = reorder(user_mem, weights_mem);
                     cached_entry.reorder_weight_cache.push_back({user_md, r_prim});
                     {
                         std::lock_guard<std::mutex> lock(mtx);
                         auto it = cache.find(key);
                         if (it != cache.end()) {
                             it->second.reorder_weight_cache.push_back({user_md, r_prim});
                         }
                     }
                 }
                 r_prim.execute(s, user_mem, weights_mem);
             } else {
                 weights_mem = user_mem;
             }
        }

        // Prepare Diff Src (grad_input)
        memory diff_src_mem;
        auto user_diff_src_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
        bool need_reorder_diff_src = (expected_diff_src_md != user_diff_src_md);
        Storage blocked_storage_handle;

        if (need_reorder_diff_src) {
             size_t required_size = expected_diff_src_md.get_size();
             Allocator* allocator = getAllocator(grad_input.device().type());
             blocked_storage_handle = Storage(required_size, allocator);
             diff_src_mem = memory(expected_diff_src_md, eng, blocked_storage_handle.data());
        } else {
             diff_src_mem = memory(expected_diff_src_md, eng, grad_input.data_ptr<float>());
        }

        auto start_conv = std::chrono::high_resolution_clock::now();
        bwd_d.execute(s, {
            {DNNL_ARG_DIFF_DST, diff_dst_mem},
            {DNNL_ARG_WEIGHTS, weights_mem},
            {DNNL_ARG_DIFF_SRC, diff_src_mem}
        });
        
        if (need_reorder_diff_src) {
             auto user_mem = memory(user_diff_src_md, eng, grad_input.data_ptr<float>());
             reorder(diff_src_mem, user_mem).execute(s, diff_src_mem, user_mem);
        }
        s.wait();
        
        if (grad_input.unsafeGetTensorImpl()->has_onednn_md()) {
            grad_input.unsafeGetTensorImpl()->set_onednn_md(nullptr);
        }
        auto end_conv = std::chrono::high_resolution_clock::now();
        // std::cout << "DEBUG: ConvBwdData Time: " << std::chrono::duration_cast<std::chrono::microseconds>(end_conv - start_conv).count() << " us" << std::endl;

        return true;
    } catch (...) {
        return false;
    }
}

static bool conv2d_grad_weight_onednn(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                                     const std::vector<int64_t>& stride,
                                     int64_t pH_top, int64_t pH_bottom, int64_t pW_left, int64_t pW_right,
                                     const std::vector<int64_t>& dilation, int64_t groups,
                                     Tensor& grad_weight) {
    if (!OneDNNContext::is_enabled()) {
        return false;
    }
    // std::cout << "DEBUG: conv2d_grad_weight_onednn called" << std::endl;
    if (input.dtype() != DType::Float32) return false;

    // Ensure contiguous or blocked
    Tensor input_c = (input.is_contiguous() || input.unsafeGetTensorImpl()->has_onednn_md()) ? input : input.clone();
    Tensor grad_output_c = (grad_output.is_contiguous() || grad_output.unsafeGetTensorImpl()->has_onednn_md()) ? grad_output : grad_output.clone();

    try {
        auto& eng = OneDNNContext::get_engine();
        auto& s = OneDNNContext::get_stream();

        // Cache Key
        ConvKey key;
        key.n = input_c.size(0); key.ic = input_c.size(1); key.ih = input_c.size(2); key.iw = input_c.size(3);
        key.oc = grad_output_c.size(1); key.kh = weight.size(2); key.kw = weight.size(3);
        key.oh = grad_output_c.size(2); key.ow = grad_output_c.size(3);
        key.sh = stride[0]; key.sw = stride[1];
        key.ph_t = pH_top; key.ph_b = pH_bottom; key.pw_l = pW_left; key.pw_r = pW_right;
        key.dh = dilation[0]; key.dw = dilation[1];
        key.groups = groups;
        key.has_bias = false;
        key.type = 2; // BwdWeights

        struct CachedConvBwdWeights {
            convolution_backward_weights::primitive_desc pd;
            convolution_backward_weights prim;
            std::vector<std::pair<memory::desc, reorder>> reorder_input_cache;
            std::vector<std::pair<memory::desc, reorder>> reorder_grad_output_cache;
        };

        static std::unordered_map<ConvKey, CachedConvBwdWeights> cache;
        static std::mutex mtx;
        
        bool found = false;
        CachedConvBwdWeights cached_entry;
        
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto it = cache.find(key);
            if (it != cache.end()) {
                cached_entry = it->second;
                found = true;
            }
        }

        memory::dims src_dims = {input_c.size(0), input_c.size(1), input_c.size(2), input_c.size(3)};
        memory::dims dst_dims = {grad_output_c.size(0), grad_output_c.size(1), grad_output_c.size(2), grad_output_c.size(3)};
        
        memory::dims weights_dims;
        if (groups > 1) {
            weights_dims = {groups, weight.size(0)/groups, weight.size(1), weight.size(2), weight.size(3)};
        } else {
            weights_dims = {weight.size(0), weight.size(1), weight.size(2), weight.size(3)};
        }

        memory::dims strides_dims = {stride[0], stride[1]};
        memory::dims padding_l_dims = {pH_top, pW_left};
        memory::dims padding_r_dims = {pH_bottom, pW_right};
        memory::dims dilates_dims = {dilation[0] - 1, dilation[1] - 1};

        // For 1x1 convolutions, use NHWC to avoid reorders and match Forward
        bool is_1x1 = (key.kh == 1 && key.kw == 1 && key.sh == 1 && key.sw == 1 && key.ph_t == 0 && key.ph_b == 0 && key.pw_l == 0 && key.pw_r == 0);
        auto src_tag = is_1x1 ? memory::format_tag::nhwc : memory::format_tag::any;
        auto dst_tag = is_1x1 ? memory::format_tag::nhwc : memory::format_tag::any;

        auto src_md = memory::desc(src_dims, memory::data_type::f32, src_tag);
        auto dst_md = memory::desc(dst_dims, memory::data_type::f32, dst_tag);
        auto weights_md = groups > 1 
            ? memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::any)
            : memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::any);
            
        auto bias_md = memory::desc();

        if (!found) {
            // Forward PD (Hint)
            dnnl::algorithm algo = get_onednn_algo(key.kh, key.kw);
            
            convolution_forward::primitive_desc fwd_pd;
            try {
                fwd_pd = convolution_forward::primitive_desc(
                    eng, prop_kind::forward_inference, algo,
                    src_md, weights_md, bias_md, dst_md,
                    strides_dims, dilates_dims, padding_l_dims, padding_r_dims);
            } catch (...) {
                fwd_pd = convolution_forward::primitive_desc(
                    eng, prop_kind::forward_inference, algorithm::convolution_auto,
                    src_md, weights_md, bias_md, dst_md,
                    strides_dims, dilates_dims, padding_l_dims, padding_r_dims);
                algo = algorithm::convolution_auto;
            }

            // Backward Weights PD
            convolution_backward_weights::primitive_desc bwd_w_pd;
            try {
                bwd_w_pd = convolution_backward_weights::primitive_desc(
                    eng, algo,
                    src_md, weights_md, bias_md, dst_md,
                    strides_dims, dilates_dims, padding_l_dims, padding_r_dims,
                    fwd_pd);
            } catch (...) {
                if (algo != algorithm::convolution_auto) {
                    bwd_w_pd = convolution_backward_weights::primitive_desc(
                        eng, algorithm::convolution_auto,
                        src_md, weights_md, bias_md, dst_md,
                        strides_dims, dilates_dims, padding_l_dims, padding_r_dims,
                        fwd_pd);
                } else {
                    throw;
                }
            }

            auto bwd_w = convolution_backward_weights(bwd_w_pd);
            cached_entry = {bwd_w_pd, bwd_w};
            
            {
                std::lock_guard<std::mutex> lock(mtx);
                cache.insert({key, cached_entry});
            }
        }

        auto& bwd_w = cached_entry.prim;
        auto& pd = cached_entry.pd;

        auto expected_src_md = pd.src_desc();
        auto expected_diff_dst_md = pd.diff_dst_desc();
        auto expected_diff_weights_md = pd.diff_weights_desc();

        // Prepare Src (Input)
        memory src_mem;
        if (input_c.unsafeGetTensorImpl()->has_onednn_md()) {
             auto stored_md = std::static_pointer_cast<memory::desc>(input_c.unsafeGetTensorImpl()->get_onednn_md());
             if (*stored_md == expected_src_md) {
                 src_mem = memory(*stored_md, eng, input_c.data_ptr<float>());
             } else {
                 // std::cout << "DEBUG: Reordering input (stored) in conv2d_grad_weight_onednn" << std::endl;
                 auto src = memory(*stored_md, eng, input_c.data_ptr<float>());
                 src_mem = memory(expected_src_md, eng);
                 reorder(src, src_mem).execute(s, src, src_mem);
             }
        } else {
             auto user_md = memory::desc(src_dims, memory::data_type::f32, memory::format_tag::nchw);
             auto user_mem = memory(user_md, eng, input_c.data_ptr<float>());
             if (user_md != expected_src_md) {
                 // std::cout << "DEBUG: Reordering input (NCHW) in conv2d_grad_weight_onednn" << std::endl;
                 src_mem = memory(expected_src_md, eng);
                 
                 reorder r_prim;
                 bool r_found = false;
                 for (const auto& item : cached_entry.reorder_input_cache) {
                     if (item.first == user_md) {
                         r_prim = item.second;
                         r_found = true;
                         break;
                     }
                 }
                 
                 if (!r_found) {
                     r_prim = reorder(user_mem, src_mem);
                     cached_entry.reorder_input_cache.push_back({user_md, r_prim});
                     {
                         std::lock_guard<std::mutex> lock(mtx);
                         auto it = cache.find(key);
                         if (it != cache.end()) {
                             it->second.reorder_input_cache.push_back({user_md, r_prim});
                         }
                     }
                 }
                 r_prim.execute(s, user_mem, src_mem);
             } else {
                 src_mem = user_mem;
             }
        }

        // Prepare Diff Dst (Grad Output)
        memory diff_dst_mem;
        bool loaded_from_cache = false;

        // Try to use cached memory object from SharedState
        if (grad_output_c.unsafeGetTensorImpl()->has_onednn_memory_cache()) {
             auto cached_mem_ptr = std::static_pointer_cast<memory>(grad_output_c.unsafeGetTensorImpl()->get_onednn_memory_cache());
             if (cached_mem_ptr && cached_mem_ptr->get_desc() == expected_diff_dst_md) {
                 diff_dst_mem = *cached_mem_ptr;
                 loaded_from_cache = true;
             }
        }
        
        if (!loaded_from_cache) {
             if (grad_output_c.unsafeGetTensorImpl()->has_onednn_md()) {
                  auto stored_md = std::static_pointer_cast<memory::desc>(grad_output_c.unsafeGetTensorImpl()->get_onednn_md());
                  if (*stored_md == expected_diff_dst_md) {
                      diff_dst_mem = memory(*stored_md, eng, grad_output_c.data_ptr<float>());
                  } else {
                      // std::cout << "DEBUG: Reordering grad_output (stored) in conv2d_grad_weight_onednn. Impl: " << grad_output_c.unsafeGetTensorImpl() << std::endl;
                      auto src = memory(*stored_md, eng, grad_output_c.data_ptr<float>());
                      diff_dst_mem = memory(expected_diff_dst_md, eng);
                      reorder(src, diff_dst_mem).execute(s, src, diff_dst_mem);
                  }
             } else {
                  auto user_md = memory::desc(dst_dims, memory::data_type::f32, memory::format_tag::nchw);
                  auto user_mem = memory(user_md, eng, grad_output_c.data_ptr<float>());
                  if (user_md != expected_diff_dst_md) {
                      // Optimize: Update grad_output storage to blocked format
                      size_t req_size = expected_diff_dst_md.get_size();
                      Allocator* allocator = getAllocator(grad_output_c.device().type());
                      Storage new_storage(req_size, allocator);
                      
                      diff_dst_mem = memory(expected_diff_dst_md, eng, new_storage.data());
     
                      reorder r_prim;
                      bool r_found = false;
                      for (const auto& item : cached_entry.reorder_grad_output_cache) {
                          if (item.first == user_md) {
                              r_prim = item.second;
                              r_found = true;
                              break;
                          }
                      }
                      
                      if (!r_found) {
                          r_prim = reorder(user_mem, diff_dst_mem);
                          cached_entry.reorder_grad_output_cache.push_back({user_md, r_prim});
                          {
                              std::lock_guard<std::mutex> lock(mtx);
                              auto it = cache.find(key);
                              if (it != cache.end()) {
                                  it->second.reorder_grad_output_cache.push_back({user_md, r_prim});
                              }
                          }
                      }
                      r_prim.execute(s, user_mem, diff_dst_mem);
                      
                      // Update the tensor
                      grad_output_c.unsafeGetTensorImpl()->set_storage(new_storage);
                      grad_output_c.unsafeGetTensorImpl()->set_onednn_md(std::make_shared<memory::desc>(expected_diff_dst_md));
                  } else {
                      diff_dst_mem = user_mem;
                  }
             }
             // Cache the memory object
             grad_output_c.unsafeGetTensorImpl()->set_onednn_memory_cache(std::make_shared<memory>(diff_dst_mem));
        }

        // Prepare Diff Weights (Grad Weight)
        memory diff_weights_mem;
        
        auto user_diff_weights_md = groups > 1 
             ? memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::goihw)
             : memory::desc(weights_dims, memory::data_type::f32, memory::format_tag::oihw);

        bool need_reorder_diff_weights = (expected_diff_weights_md != user_diff_weights_md);
        Storage blocked_storage_handle;

        if (need_reorder_diff_weights) {
             size_t required_size = expected_diff_weights_md.get_size();
             Allocator* allocator = getAllocator(grad_weight.device().type());
             blocked_storage_handle = Storage(required_size, allocator);
             diff_weights_mem = memory(expected_diff_weights_md, eng, blocked_storage_handle.data());
        } else {
             diff_weights_mem = memory(expected_diff_weights_md, eng, grad_weight.data_ptr<float>());
        }

#ifdef _OPENMP
        int original_threads = omp_get_max_threads();
        bool adjusted = false;
        // Heuristic: For small batches (N < threads*2) and small weights, reduce threads to avoid reduction overhead.
        if (input_c.size(0) < original_threads * 2 && weight.numel() < 20000) {
            omp_set_num_threads(1);
            adjusted = true;
        }
#endif

        bwd_w.execute(s, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DIFF_DST, diff_dst_mem},
            {DNNL_ARG_DIFF_WEIGHTS, diff_weights_mem}
        });

#ifdef _OPENMP
        if (adjusted) {
            omp_set_num_threads(original_threads);
        }
#endif
        
        if (need_reorder_diff_weights) {
             auto user_mem = memory(user_diff_weights_md, eng, grad_weight.data_ptr<float>());
             reorder(diff_weights_mem, user_mem).execute(s, diff_weights_mem, user_mem);
        }
        s.wait();
        
        if (grad_weight.unsafeGetTensorImpl()->has_onednn_md()) {
            grad_weight.unsafeGetTensorImpl()->set_onednn_md(nullptr);
        }

        return true;
    } catch (...) {
        return false;
    }
}
#endif

Tensor conv2d_cpu(const Tensor& input_arg, const Tensor& weight_arg, const Tensor& bias, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups) {
    Tensor input = input_arg.is_contiguous() ? input_arg : input_arg.clone();
    Tensor weight = weight_arg.is_contiguous() ? weight_arg : weight_arg.clone();
    
    if (input.dim() != 4 || weight.dim() != 4) TP_THROW(RuntimeError, "conv2d: Expected 4D input and weight");
    
    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    
    int64_t C_out = weight.size(0);
    int64_t C_in_group = weight.size(1); 
    int64_t kH = weight.size(2);
    int64_t kW = weight.size(3);
    
    if (C_in % groups != 0) TP_THROW(RuntimeError, "in_channels must be divisible by groups");
    if (C_out % groups != 0) TP_THROW(RuntimeError, "out_channels must be divisible by groups");
    if (C_in / groups != C_in_group) TP_THROW(RuntimeError, "Weight shape mismatch: expected " + std::to_string(C_in/groups) + " input channels per group, got " + std::to_string(C_in_group));
    
    auto stride = expand_param(stride_arg, 2, "stride");
    auto padding = expand_param(padding_arg, 2, "padding");
    auto dilation = expand_param(dilation_arg, 2, "dilation");

    int64_t sH = stride[0]; int64_t sW = stride[1];
    
    // Support asymmetric padding
    int64_t pH_top, pH_bottom, pW_left, pW_right;
    if (padding.size() == 2) {
        pH_top = pH_bottom = padding[0];
        pW_left = pW_right = padding[1];
    } else {
        // Assume order: top, bottom, left, right
        pH_top = padding[0]; pH_bottom = padding[1];
        pW_left = padding[2]; pW_right = padding[3];
    }

    int64_t dH = dilation[0]; int64_t dW = dilation[1];
    
    int64_t H_out = (H_in + pH_top + pH_bottom - dH * (kH - 1) - 1) / sH + 1;
    int64_t W_out = (W_in + pW_left + pW_right - dW * (kW - 1) - 1) / sW + 1;
    
    if (H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "conv2d: Calculated output size is too small");

    Tensor out = Tensor::empty({N, C_out, H_out, W_out}, input.dtype(), input.device());
    
    // Manual 1x1 Optimization Removed: Let OneDNN handle it for better integration and performance.


    // Optimization: Winograd Input Size Alignment (User Request: Break constraints)
    // Force padding to multiple of 8 for 3x3s1 convolutions to enable Winograd on AVX2
    // Only apply if input is NCHW (not blocked) to avoid messing up OneDNN formats
    if (groups == 1 && kH == 3 && kW == 3 && sH == 1 && sW == 1 && dH == 1 && dW == 1 &&
        input.dtype() == DType::Float32 && !input.unsafeGetTensorImpl()->has_onednn_md()) {
        
        int64_t h_padded_total = H_in + pH_top + pH_bottom;
        int64_t w_padded_total = W_in + pW_left + pW_right;
        
        int64_t pad_h_extra = (h_padded_total % 8 != 0) ? (8 - (h_padded_total % 8)) : 0;
        int64_t pad_w_extra = (w_padded_total % 8 != 0) ? (8 - (w_padded_total % 8)) : 0;
        
        // Check overhead to avoid performance regression
        // Winograd usually gives 2-2.5x speedup, but padding adds copy + larger compute.
        // We conservatively allow up to 15% overhead.
        // 64x64 (pad=1) -> 66x66. Aligned to 72x72. Overhead ~19%. Too high.
        double overhead = (double)((h_padded_total + pad_h_extra) * (w_padded_total + pad_w_extra)) / 
                          (double)(h_padded_total * w_padded_total);

        if ((pad_h_extra > 0 || pad_w_extra > 0) && overhead < 1.15) {
            // Apply padding: (left, right, top, bottom)
            // We need to preserve existing padding logic, so we pass 0 to conv2d_onednn
            // and apply ALL padding here.
            std::vector<int64_t> pads = {pW_left, pW_right + pad_w_extra, pH_top, pH_bottom + pad_h_extra};
            Tensor input_padded = constant_pad_nd_cpu(input, pads, 0.0f);
            
            // Output size will be larger
            int64_t H_out_padded = h_padded_total + pad_h_extra - 2; // 3x3 kernel: L - K + 1 = L - 3 + 1 = L - 2
            int64_t W_out_padded = w_padded_total + pad_w_extra - 2;
            
            Tensor out_padded = Tensor::empty({N, C_out, H_out_padded, W_out_padded}, input.dtype(), input.device());
            
            if (conv2d_onednn(input_padded, weight, bias, stride, 0, 0, 0, 0, dilation, groups, out_padded)) {
                 // Crop to original size
                 return out_padded.slice(2, 0, H_out).slice(3, 0, W_out).contiguous();
            }
        }
    }

    #ifdef USE_ONEDNN
    // Try oneDNN implementation first
    if (conv2d_onednn(input, weight, bias, stride, pH_top, pH_bottom, pW_left, pW_right, dilation, groups, out)) {
        return out;
    }
    #endif
    
    if (input.dtype() == DType::Float32) {
        // Winograd Check (F(2,3) for 3x3s1)
        // Only apply if groups=1 (Standard Conv)
        if (groups == 1 && kH == 3 && kW == 3 && sH == 1 && sW == 1 && dH == 1 && dW == 1) {
             conv2d_winograd_3x3(input, weight, bias, pH_top, pW_left, out);
             return out;
        }

        int64_t C_out_group = C_out / groups;
        int64_t col_size = C_in_group * kH * kW;
        int64_t out_spatial = H_out * W_out;
        
        const float* in_ptr = input.data_ptr<float>();
        const float* w_ptr = weight.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();
        
        // Parallelize over batch size N
        // Each thread needs its own 'col' buffer
        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            #ifdef USE_MKL
            mkl_set_num_threads_local(1);
            #endif

            // Optimization: 1x1 Conv check (Stride 1, Pad 0)
            bool is_1x1 = (kH == 1 && kW == 1 && sH == 1 && sW == 1 && dH == 1 && dW == 1 && 
                           pH_top == 0 && pH_bottom == 0 && pW_left == 0 && pW_right == 0);

            // Optimization: Avoid zero-initialization using unique_ptr + new[]
            std::unique_ptr<float[]> col_buffer;
            if (!is_1x1) {
                col_buffer.reset(new float[col_size * out_spatial]);
            }
            float* col_ptr = col_buffer.get();
            
            #ifdef _OPENMP
            #pragma omp for
            #endif
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t g = 0; g < groups; ++g) {
                    const float* in_n_g = in_ptr + (n * C_in + g * C_in_group) * H_in * W_in;
                    const float* gemm_col_ptr = nullptr;
                    
                    if (is_1x1) {
                        // 1x1 Fast Path: Direct pointer, skip im2col and allocation
                        gemm_col_ptr = in_n_g;
                    } else {
                        // Standard Im2Col Path
                        gemm_col_ptr = col_ptr;
                        
                        // Optimized im2col (Split Loop Strategy)
                        for (int64_t c = 0; c < col_size; ++c) {
                            int64_t w_offset = c % kW;
                            int64_t h_offset = (c / kW) % kH;
                            int64_t c_im = c / kH / kW;
                            
                            // Calculate valid range for h
                            int64_t h_start_num = pH_top - h_offset * dH;
                            int64_t h_start = 0;
                            if (h_start_num > 0) h_start = (h_start_num + sH - 1) / sH;
                            
                            int64_t h_end_num = H_in + pH_top - h_offset * dH;
                            int64_t h_end = H_out;
                            if (h_end_num > 0) h_end = std::min(H_out, (h_end_num + sH - 1) / sH);
                            else h_end = 0;

                            // Calculate valid range for w
                            int64_t w_start_num = pW_left - w_offset * dW;
                            int64_t w_start = 0;
                            if (w_start_num > 0) w_start = (w_start_num + sW - 1) / sW;

                            int64_t w_end_num = W_in + pW_left - w_offset * dW;
                            int64_t w_end = W_out;
                            if (w_end_num > 0) w_end = std::min(W_out, (w_end_num + sW - 1) / sW);
                            else w_end = 0;

                            int64_t row_in_base = c_im * H_in * W_in;
                            int64_t row_out_base = c * out_spatial;

                            // 1. Top padding
                            if (h_start > 0) {
                                std::memset(col_ptr + row_out_base, 0, h_start * W_out * sizeof(float));
                            }
                            
                            // 2. Middle rows
                            for (int64_t h = h_start; h < h_end; ++h) {
                                 int64_t h_pad = h * sH - pH_top + h_offset * dH;
                                 int64_t in_h_idx = row_in_base + h_pad * W_in;
                                 int64_t out_h_idx = row_out_base + h * W_out;
                                 
                                 // Left padding
                                 if (w_start > 0) {
                                     std::memset(col_ptr + out_h_idx, 0, w_start * sizeof(float));
                                 }
                                 
                                 // Center
                                 if (sW == 1 && dW == 1) {
                                     int64_t len = w_end - w_start;
                                     if (len > 0) {
                                         int64_t w_pad_start = w_start - pW_left + w_offset;
                                         std::memcpy(col_ptr + out_h_idx + w_start, 
                                                     in_n_g + in_h_idx + w_pad_start, 
                                                     len * sizeof(float));
                                     }
                                 } else {
                                     for (int64_t w = w_start; w < w_end; ++w) {
                                         int64_t w_pad = w * sW - pW_left + w_offset * dW;
                                         col_ptr[out_h_idx + w] = in_n_g[in_h_idx + w_pad];
                                     }
                                 }

                                 // Right padding
                                 if (w_end < W_out) {
                                     std::memset(col_ptr + out_h_idx + w_end, 0, (W_out - w_end) * sizeof(float));
                                 }
                            }
                            
                            // 3. Bottom padding
                            if (h_end < H_out) {
                                std::memset(col_ptr + row_out_base + h_end * W_out, 0, (H_out - h_end) * W_out * sizeof(float));
                            }
                        }
                    }
                    
                    // GEMM: Weight_g (C_out_g, col_size) * Col (col_size, out_spatial) -> Out (C_out_g, out_spatial)
                    // C = alpha * A * B + beta * C
                    // We want to write directly to out tensor
                    float* out_n_g = out_ptr + (n * C_out + g * C_out_group) * out_spatial;
                    const float* w_g = w_ptr + g * C_out_group * col_size;
                    
                    // M = C_out_group, N = out_spatial, K = col_size
                    // A = w_g (M x K), B = col (K x N), C = out_n_g (M x N)
                    gemm_direct(false, false, C_out_group, out_spatial, col_size, 
                                1.0f, w_g, col_size, 
                                gemm_col_ptr, out_spatial, 
                                0.0f, out_n_g, out_spatial);
                }
            }
        }
        
        // Add bias if present (Parallelized)
        if (bias.defined() && bias.numel() > 0) {
             const float* b_ptr = bias.data_ptr<float>();
             
             #ifdef _OPENMP
             #pragma omp parallel for
             #endif
             for (int64_t n = 0; n < N; ++n) {
                 for (int64_t c = 0; c < C_out; ++c) {
                     float b = b_ptr[c];
                     float* out_n_c = out_ptr + (n * C_out + c) * out_spatial;
                     
                     #ifdef _OPENMP
                     // #pragma omp simd
                     #endif
                     for (int64_t i = 0; i < out_spatial; ++i) {
                         out_n_c[i] += b;
                     }
                 }
             }
        }
        
    } else {
        TP_THROW(NotImplementedError, "conv2d only supports Float32");
    }
    
    return out;
}

Tensor conv1d_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, int64_t groups) {
    // Map 1D to 2D: (N, C, L) -> (N, C, 1, L)
    // Weight: (Out, In/G, K) -> (Out, In/G, 1, K)
    // Stride/Padding/Dilation: (S) -> (1, S)
    
    if (input.dim() != 3) TP_THROW(RuntimeError, "conv1d: Expected 3D input (N, C, L)");
    if (weight.dim() != 3) TP_THROW(RuntimeError, "conv1d: Expected 3D weight");
    
    Tensor in_2d = input.unsqueeze(2);
    Tensor w_2d = weight.unsqueeze(2);
    
    std::vector<int64_t> s_2d = {1, stride.empty() ? 1 : stride[0]};
    std::vector<int64_t> p_2d = {0, padding.empty() ? 0 : padding[0]};
    std::vector<int64_t> d_2d = {1, dilation.empty() ? 1 : dilation[0]};
    
    Tensor out_2d = conv2d_cpu(in_2d, w_2d, bias, s_2d, p_2d, d_2d, groups);
    
    return out_2d.squeeze(2);
}

Tensor conv3d_cpu(const Tensor& input_arg, const Tensor& weight_arg, const Tensor& bias, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups) {
    Tensor input = input_arg.is_contiguous() ? input_arg : input_arg.clone();
    Tensor weight = weight_arg.is_contiguous() ? weight_arg : weight_arg.clone();

    if (input.dim() != 5 || weight.dim() != 5) TP_THROW(RuntimeError, "conv3d: Expected 5D input and weight");
    
    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t D_in = input.size(2);
    int64_t H_in = input.size(3);
    int64_t W_in = input.size(4);
    
    int64_t C_out = weight.size(0);
    int64_t C_in_group = weight.size(1);
    int64_t kD = weight.size(2);
    int64_t kH = weight.size(3);
    int64_t kW = weight.size(4);
    
    auto stride = expand_param(stride_arg, 3, "stride");
    auto padding = expand_param(padding_arg, 3, "padding");
    auto dilation = expand_param(dilation_arg, 3, "dilation");
    
    int64_t sD = stride[0]; int64_t sH = stride[1]; int64_t sW = stride[2];
    int64_t pD = padding[0]; int64_t pH = padding[1]; int64_t pW = padding[2];
    int64_t dD = dilation[0]; int64_t dH = dilation[1]; int64_t dW = dilation[2];
    
    int64_t D_out = (D_in + 2 * pD - dD * (kD - 1) - 1) / sD + 1;
    int64_t H_out = (H_in + 2 * pH - dH * (kH - 1) - 1) / sH + 1;
    int64_t W_out = (W_in + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    
    if (D_out <= 0 || H_out <= 0 || W_out <= 0) TP_THROW(RuntimeError, "conv3d: Calculated output size is too small");

    Tensor out = Tensor::empty({N, C_out, D_out, H_out, W_out}, input.dtype(), input.device());
    
    #ifdef USE_ONEDNN
    // Try oneDNN implementation first
    if (conv3d_onednn(input, weight, bias, stride, pD, pD, pH, pH, pW, pW, dilation, groups, out)) {
        return out;
    }
    #endif
    
    if (input.dtype() == DType::Float32) {
        int64_t C_out_group = C_out / groups;
        int64_t col_size = C_in_group * kD * kH * kW;
        int64_t out_volume = D_out * H_out * W_out;
        
        const float* in_ptr = input.data_ptr<float>();
        const float* w_ptr = weight.data_ptr<float>();
        float* out_ptr = out.data_ptr<float>();

        #ifdef _OPENMP
        #pragma omp parallel
        #endif
        {
            #ifdef USE_MKL
            mkl_set_num_threads_local(1);
            #endif

            // Optimization: 1x1x1 Conv check
            bool is_1x1 = (kD == 1 && kH == 1 && kW == 1 && 
                           sD == 1 && sH == 1 && sW == 1 && 
                           dD == 1 && dH == 1 && dW == 1 && 
                           pD == 0 && pH == 0 && pW == 0);

            // Optimization: Avoid zero-initialization using unique_ptr + new[]
            std::unique_ptr<float[]> col_buffer;
            if (!is_1x1) {
                col_buffer.reset(new float[col_size * out_volume]);
            }
            float* col_ptr = col_buffer.get();
            
            #ifdef _OPENMP
            #pragma omp for
            #endif
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t g = 0; g < groups; ++g) {
                    const float* in_n_g = in_ptr + (n * C_in + g * C_in_group) * D_in * H_in * W_in;
                    const float* gemm_col_ptr = nullptr;

                    if (is_1x1) {
                        gemm_col_ptr = in_n_g;
                    } else {
                        gemm_col_ptr = col_ptr;
                        // Serial im2col3d
                        for (int64_t c = 0; c < col_size; ++c) {
                            int64_t w_offset = c % kW;
                            int64_t h_offset = (c / kW) % kH;
                            int64_t d_offset = (c / kW / kH) % kD;
                            int64_t c_im = c / kW / kH / kD;
                            
                            for (int64_t d = 0; d < D_out; ++d) {
                                for (int64_t h = 0; h < H_out; ++h) {
                                    for (int64_t w = 0; w < W_out; ++w) {
                                        int64_t d_pad = d * sD - pD + d_offset * dD;
                                        int64_t h_pad = h * sH - pH + h_offset * dH;
                                        int64_t w_pad = w * sW - pW + w_offset * dW;
                                        
                                        int64_t val_idx = (c * out_volume + (d * H_out + h) * W_out + w);
                                        
                                        if (d_pad >= 0 && d_pad < D_in && h_pad >= 0 && h_pad < H_in && w_pad >= 0 && w_pad < W_in) {
                                            col_ptr[val_idx] = in_n_g[((c_im * D_in + d_pad) * H_in + h_pad) * W_in + w_pad];
                                        } else {
                                            col_ptr[val_idx] = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    
                    float* out_n_g = out_ptr + (n * C_out + g * C_out_group) * out_volume;
                    const float* w_g = w_ptr + g * C_out_group * col_size;
                    
                    gemm_direct(false, false, C_out_group, out_volume, col_size, 
                                1.0f, w_g, col_size, 
                                gemm_col_ptr, out_volume, 
                                0.0f, out_n_g, out_volume);
                }
            }
        }
        
        // Bias addition
        if (bias.defined() && bias.numel() > 0) {
             const float* b_ptr = bias.data_ptr<float>();
             #ifdef _OPENMP
             #pragma omp parallel for
             #endif
             for (int64_t n = 0; n < N; ++n) {
                 for (int64_t c = 0; c < C_out; ++c) {
                     float b = b_ptr[c];
                     float* out_n_c = out_ptr + (n * C_out + c) * out_volume;
                     for (int64_t i = 0; i < out_volume; ++i) {
                         out_n_c[i] += b;
                     }
                 }
             }
        }
    } else {
        TP_THROW(NotImplementedError, "conv3d only supports Float32");
    }
    return out;
}

Tensor conv_transpose2d_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& output_padding_arg, int64_t groups, const std::vector<int64_t>& dilation_arg) {
    // Input: (N, C_in, H_in, W_in)
    // Weight: (C_in, C_out/groups, kH, kW) - NOTE: Inverted compared to conv2d!
    // Output: (N, C_out, H_out, W_out)
    
    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    
    int64_t C_in_weight = weight.size(0);
    int64_t C_out_group = weight.size(1);
    int64_t kH = weight.size(2);
    int64_t kW = weight.size(3);
    
    if (C_in != C_in_weight) TP_THROW(RuntimeError, "Input channels mismatch weight channels");
    int64_t C_out = C_out_group * groups;
    
    auto stride = expand_param(stride_arg, 2, "stride");
    auto padding = expand_param(padding_arg, 2, "padding");
    auto output_padding = expand_param(output_padding_arg, 2, "output_padding");
    auto dilation = expand_param(dilation_arg, 2, "dilation");
    
    int64_t sH = stride[0]; int64_t sW = stride[1];
    int64_t pH = padding[0]; int64_t pW = padding[1];
    int64_t opH = output_padding[0]; int64_t opW = output_padding[1];
    int64_t dH = dilation[0]; int64_t dW = dilation[1];
    
    int64_t H_out = (H_in - 1) * sH - 2 * pH + dH * (kH - 1) + opH + 1;
    int64_t W_out = (W_in - 1) * sW - 2 * pW + dW * (kW - 1) + opW + 1;
    
    Tensor out = Tensor::zeros({N, C_out, H_out, W_out}, input.dtype(), input.device()); // Initialize with zeros for accumulation
    
    if (input.dtype() == DType::Float32) {
        float* out_ptr = out.data_ptr<float>();
        const float* in_ptr = input.data_ptr<float>();
        const float* w_ptr = weight.data_ptr<float>();
        const float* b_ptr = (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;
        
        // Add bias first
        if (b_ptr) {
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C_out; ++c) {
                    float b = b_ptr[c];
                    for (int64_t h = 0; h < H_out; ++h) {
                        for (int64_t w = 0; w < W_out; ++w) {
                            out_ptr[((n * C_out + c) * H_out + h) * W_out + w] = b;
                        }
                    }
                }
            }
        }
        
        int64_t C_in_group_size = C_in / groups;
        
        // Transposed Conv Logic: Distribute input to output
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < groups; ++g) {
                for (int64_t c_in_i = 0; c_in_i < C_in_group_size; ++c_in_i) {
                    int64_t c_in = g * C_in_group_size + c_in_i;
                    
                    for (int64_t h_in = 0; h_in < H_in; ++h_in) {
                        for (int64_t w_in = 0; w_in < W_in; ++w_in) {
                            
                            // Input value
                            int64_t in_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                            float in_val = in_ptr[in_idx];
                            
                            for (int64_t c_out_i = 0; c_out_i < C_out_group; ++c_out_i) {
                                int64_t c_out = g * C_out_group + c_out_i;
                                
                                for (int64_t kh = 0; kh < kH; ++kh) {
                                    for (int64_t kw = 0; kw < kW; ++kw) {
                                        // Calculate output position
                                        // out_pos = in_pos * stride + kernel_pos * dilation - padding
                                        int64_t h_out_pos = h_in * sH + kh * dH - pH;
                                        int64_t w_out_pos = w_in * sW + kw * dW - pW;
                                        
                                        if (h_out_pos >= 0 && h_out_pos < H_out && w_out_pos >= 0 && w_out_pos < W_out) {
                                            // Weight: (c_in, c_out_group, kh, kw)
                                            int64_t w_idx = ((c_in * C_out_group + c_out_i) * kH + kh) * kW + kw;
                                            
                                            // Output Accumulate
                                            int64_t out_idx = ((n * C_out + c_out) * H_out + h_out_pos) * W_out + w_out_pos;
                                            out_ptr[out_idx] += in_val * w_ptr[w_idx];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
         TP_THROW(NotImplementedError, "conv_transpose2d only supports Float32");
    }
    
    return out;
}

Tensor conv_transpose3d_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& output_padding_arg, int64_t groups, const std::vector<int64_t>& dilation_arg) {
    if (input.dim() != 5 || weight.dim() != 5) TP_THROW(RuntimeError, "conv_transpose3d: Expected 5D input and weight");

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t D_in = input.size(2);
    int64_t H_in = input.size(3);
    int64_t W_in = input.size(4);

    int64_t C_in_weight = weight.size(0);
    int64_t C_out_group = weight.size(1);
    int64_t kD = weight.size(2);
    int64_t kH = weight.size(3);
    int64_t kW = weight.size(4);

    if (C_in != C_in_weight) TP_THROW(RuntimeError, "Input channels mismatch weight channels");
    int64_t C_out = C_out_group * groups;

    auto stride = expand_param(stride_arg, 3, "stride");
    auto padding = expand_param(padding_arg, 3, "padding");
    auto output_padding = expand_param(output_padding_arg, 3, "output_padding");
    auto dilation = expand_param(dilation_arg, 3, "dilation");

    int64_t sD = stride[0]; int64_t sH = stride[1]; int64_t sW = stride[2];
    int64_t pD = padding[0]; int64_t pH = padding[1]; int64_t pW = padding[2];
    int64_t opD = output_padding[0]; int64_t opH = output_padding[1]; int64_t opW = output_padding[2];
    int64_t dD = dilation[0]; int64_t dH = dilation[1]; int64_t dW = dilation[2];

    int64_t D_out = (D_in - 1) * sD - 2 * pD + dD * (kD - 1) + opD + 1;
    int64_t H_out = (H_in - 1) * sH - 2 * pH + dH * (kH - 1) + opH + 1;
    int64_t W_out = (W_in - 1) * sW - 2 * pW + dW * (kW - 1) + opW + 1;

    Tensor out = Tensor::zeros({N, C_out, D_out, H_out, W_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        float* out_ptr = out.data_ptr<float>();
        const float* in_ptr = input.data_ptr<float>();
        const float* w_ptr = weight.data_ptr<float>();
        const float* b_ptr = (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr;

        if (b_ptr) {
            for (int64_t n = 0; n < N; ++n) {
                for (int64_t c = 0; c < C_out; ++c) {
                    float b = b_ptr[c];
                    for (int64_t d = 0; d < D_out; ++d) {
                        for (int64_t h = 0; h < H_out; ++h) {
                            for (int64_t w = 0; w < W_out; ++w) {
                                out_ptr[(((n * C_out + c) * D_out + d) * H_out + h) * W_out + w] = b;
                            }
                        }
                    }
                }
            }
        }

        int64_t C_in_group_size = C_in / groups;

        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < groups; ++g) {
                for (int64_t c_in_i = 0; c_in_i < C_in_group_size; ++c_in_i) {
                    int64_t c_in = g * C_in_group_size + c_in_i;
                    
                    for (int64_t d_in = 0; d_in < D_in; ++d_in) {
                        for (int64_t h_in = 0; h_in < H_in; ++h_in) {
                            for (int64_t w_in = 0; w_in < W_in; ++w_in) {
                                
                                int64_t in_idx = (((n * C_in + c_in) * D_in + d_in) * H_in + h_in) * W_in + w_in;
                                float in_val = in_ptr[in_idx];
                                
                                for (int64_t c_out_i = 0; c_out_i < C_out_group; ++c_out_i) {
                                    int64_t c_out = g * C_out_group + c_out_i;
                                    
                                    for (int64_t kd = 0; kd < kD; ++kd) {
                                        for (int64_t kh = 0; kh < kH; ++kh) {
                                            for (int64_t kw = 0; kw < kW; ++kw) {
                                                
                                                int64_t d_out_pos = d_in * sD + kd * dD - pD;
                                                int64_t h_out_pos = h_in * sH + kh * dH - pH;
                                                int64_t w_out_pos = w_in * sW + kw * dW - pW;
                                                
                                                if (d_out_pos >= 0 && d_out_pos < D_out &&
                                                    h_out_pos >= 0 && h_out_pos < H_out &&
                                                    w_out_pos >= 0 && w_out_pos < W_out) {
                                                    
                                                    int64_t w_idx = ((((c_in * C_out_group + c_out_i) * kD + kd) * kH + kh) * kW + kw);
                                                    
                                                    int64_t out_idx = (((n * C_out + c_out) * D_out + d_out_pos) * H_out + h_out_pos) * W_out + w_out_pos;
                                                    out_ptr[out_idx] += in_val * w_ptr[w_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        TP_THROW(NotImplementedError, "conv_transpose3d only supports Float32");
    }
    return out;
}

// conv2d_grad_input implementation (using col2im)
Tensor conv2d_grad_input_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups) {
    // grad_output: (N, C_out, H_out, W_out)
    // weight: (C_out, C_in_group, kH, kW)
    // input: Used only for shape (N, C_in, H_in, W_in)
    
    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    
    int64_t C_out = weight.size(0);
    int64_t C_in_group = weight.size(1);
    int64_t kH = weight.size(2);
    int64_t kW = weight.size(3);
    
    auto stride = expand_param(stride_arg, 2, "stride");
    auto padding = expand_param(padding_arg, 2, "padding");
    auto dilation = expand_param(dilation_arg, 2, "dilation");

    int64_t sH = stride[0]; int64_t sW = stride[1];
    int64_t pH = padding[0]; int64_t pW = padding[1];
    int64_t dH = dilation[0]; int64_t dW = dilation[1];
    
    int64_t H_out = grad_output.size(2);
    int64_t W_out = grad_output.size(3);
    
    Tensor grad_input = Tensor::zeros({N, C_in, H_in, W_in}, input.dtype(), input.device());
    
    Tensor grad_output_contig = grad_output.contiguous();

    // Optimization: 1x1 NCHW MatMul (User Request: MatMul Algorithm)
    bool is_1x1_s1 = (groups == 1 && kH == 1 && kW == 1 && sH == 1 && sW == 1 && 
                      dH == 1 && dW == 1 && pH == 0 && pW == 0);

    if (is_1x1_s1 && input.is_contiguous() && weight.is_contiguous() && 
        input.dtype() == DType::Float32 && !input.unsafeGetTensorImpl()->has_onednn_md()) {
         
         // GradInput = Weight^T * GradOutput
         // Weight: (C_out, C_in)
         // GradOutput: (N, C_out, Pixels)
         // GradInput: (N, C_in, Pixels)
         // Per batch item:
         // GI_n = W^T * GO_n
         // M = C_in, N = Pixels, K = C_out
         
         const float* w_ptr = weight.data_ptr<float>();
         const float* go_ptr = grad_output_contig.data_ptr<float>();
         float* gi_ptr = grad_input.data_ptr<float>();
         
         int64_t M = C_in;
         int64_t K = C_out;
         int64_t N_pixels = H_in * W_in;
         
         #ifdef _OPENMP
         #pragma omp parallel for
         #endif
         for (int64_t n = 0; n < N; ++n) {
             const float* go_n = go_ptr + n * C_out * N_pixels;
             float* gi_n = gi_ptr + n * C_in * N_pixels;
             
             // W is (C_out, C_in) in memory (RowMajor)
             // We want W^T * GO
             // W^T is (C_in, C_out).
             // Calling gemm_direct with transA=true?
             // gemm_direct(transA, transB, M, N, K, ...)
             // C = op(A) * op(B)
             // We want C(M,N) = W^T(K, M)^T * GO(K, N)
             // Wait.
             // GI (C_in, Pixels) = W^T (C_in, C_out) * GO (C_out, Pixels)
             // A = W (C_out, C_in). op(A) = W^T. So transA = true.
             // B = GO (C_out, Pixels). op(B) = GO. So transB = false.
             // M = C_in, N = Pixels, K = C_out.
             
             gemm_direct(true, false, M, N_pixels, K,
                         1.0f, w_ptr, C_in, // lda = C_in (since A is C_out x C_in)
                         go_n, N_pixels,    // ldb = N_pixels
                         0.0f, gi_n, N_pixels); // ldc = N_pixels
         }
         return grad_input;
    }

    #ifdef USE_ONEDNN
    if (conv2d_grad_input_onednn(grad_output_contig, input, weight, stride, pH, pH, pW, pW, dilation, groups, grad_input)) {
        return grad_input;
    }
    #endif
    
    if (input.dtype() == DType::Float32) {
        int64_t C_out_group = C_out / groups;
        int64_t col_size = C_in_group * kH * kW;
        int64_t out_spatial = H_out * W_out;
        
        Tensor weight_reshaped = weight.reshape({groups, C_out_group, col_size});
        
        #pragma omp parallel
        {
            // Allocate per-thread buffer
            Tensor grad_col = Tensor::empty({col_size, out_spatial}, input.dtype(), input.device());
            
            #pragma omp for
            for (int64_t idx = 0; idx < N * groups; ++idx) {
                int64_t n = idx / groups;
                int64_t g = idx % groups;

                // grad_col = weight_g^T * grad_output_g
                
                Tensor w_g = weight_reshaped.select(0, g); // (C_out_group, col_size)
                
                // Use contig version for slicing to avoid overhead? 
                // slice creates view. copy if needed.
                Tensor grad_out_g = grad_output_contig.slice(1, g * C_out_group, (g + 1) * C_out_group);
                grad_out_g = grad_out_g.select(0, n).reshape({C_out_group, out_spatial});
                
                // Check contiguous for gemm
                if (!grad_out_g.is_contiguous()) {
                    grad_out_g = grad_out_g.contiguous();
                }

                gemm_direct(true, false, col_size, out_spatial, C_out_group,
                            1.0f, w_g.data_ptr<float>(), col_size, // lda = col_size (RowMajor)
                            grad_out_g.data_ptr<float>(), out_spatial, // ldb = out_spatial
                            0.0f, grad_col.data_ptr<float>(), out_spatial);
                
                // col2im
                col2im<float>(grad_col.data_ptr<float>(),
                              C_in_group, H_in, W_in, kH, kW, pH, pW, sH, sW, dH, dW,
                              grad_input.data_ptr<float>() + (n * C_in + g * C_in_group) * H_in * W_in);
            }
        }
    } else {
        TP_THROW(NotImplementedError, "conv2d_grad_input only supports Float32");
    }
    
    return grad_input;
}

Tensor conv2d_grad_weight_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups) {
    Tensor grad_output_contig = grad_output.contiguous();
    Tensor input_contig = input.contiguous();
    
    int64_t N = input_contig.size(0);
    int64_t C_in = input_contig.size(1);
    int64_t H_in = input_contig.size(2);
    int64_t W_in = input_contig.size(3);
    
    int64_t C_out = weight.size(0); 
    int64_t C_in_group = weight.size(1);
    int64_t kH = weight.size(2);
    int64_t kW = weight.size(3);
    
    auto stride = expand_param(stride_arg, 2, "stride");
    auto padding = expand_param(padding_arg, 2, "padding");
    auto dilation = expand_param(dilation_arg, 2, "dilation");

    int64_t sH = stride[0]; int64_t sW = stride[1];
    int64_t pH = padding[0]; int64_t pW = padding[1];
    int64_t dH = dilation[0]; int64_t dW = dilation[1];
    
    int64_t H_out = grad_output_contig.size(2);
    int64_t W_out = grad_output_contig.size(3);
    
    Tensor grad_weight = Tensor::zeros(static_cast<std::vector<int64_t>>(weight.shape()), weight.dtype(), weight.device());
    
    // Optimization: 1x1 NCHW MatMul (User Request: MatMul Algorithm)
    bool is_1x1_s1 = (groups == 1 && kH == 1 && kW == 1 && sH == 1 && sW == 1 && 
                      dH == 1 && dW == 1 && pH == 0 && pW == 0);

    if (is_1x1_s1 && input_contig.is_contiguous() && weight.is_contiguous() && 
        input_contig.dtype() == DType::Float32 && !input_contig.unsafeGetTensorImpl()->has_onednn_md()) {
         
         // GradWeight = GradOutput * Input^T
         // GradOutput: (N, C_out, Pixels) -> Treat as (C_out, N*Pixels) ?
         // Input: (N, C_in, Pixels) -> Treat as (C_in, N*Pixels) ?
         // GW (C_out, C_in) = GO (C_out, N*Pixels) * Input^T (N*Pixels, C_in)
         // M = C_out, N = C_in, K = N*Pixels
         
         // We can do one giant GEMM if we view (N, C, Pixels) as (C, N*Pixels).
         // BUT standard NCHW layout is (N, C, H, W).
         // Stride of C is H*W. Stride of N is C*H*W.
         // This is NOT (C, N*H*W). It is physically separated by N.
         // So we cannot do one single GEMM unless we permute/copy to (C, N, H, W) or similar.
         // OR we accumulate over N.
         
         const float* in_ptr = input_contig.data_ptr<float>();
         const float* go_ptr = grad_output_contig.data_ptr<float>();
         float* gw_ptr = grad_weight.data_ptr<float>();
         
         int64_t M = C_out;
         int64_t N_dim = C_in; // N in GEMM context
         int64_t K = H_in * W_in; // Pixels
         
         // Accumulate over batch
         for (int64_t n = 0; n < N; ++n) {
             const float* go_n = go_ptr + n * C_out * K;
             const float* in_n = in_ptr + n * C_in * K;
             
             // GW += GO_n * In_n^T
             // GO_n: (C_out, K)
             // In_n: (C_in, K). In_n^T: (K, C_in)
             // gemm_direct(false, true, ...)
             // alpha = 1.0, beta = 1.0 (accumulate)
             // Except for first n=0, beta=0.0? No, grad_weight init to zeros.
             // Wait, if we use beta=1.0 for all n, it works.
             
             gemm_direct(false, true, M, N_dim, K,
                         1.0f, go_n, K,
                         in_n, K,
                         1.0f, gw_ptr, N_dim); // ldc = C_in
         }
         return grad_weight;
    }

    #ifdef USE_ONEDNN
    if (conv2d_grad_weight_onednn(grad_output_contig, input_contig, weight, stride, pH, pH, pW, pW, dilation, groups, grad_weight)) {
        return grad_weight;
    }
    #endif

    if (input_contig.dtype() == DType::Float32) {
        int64_t C_out_group = C_out / groups;
        int64_t col_size = C_in_group * kH * kW;
        int64_t out_spatial = H_out * W_out;
        
        Tensor grad_weight_reshaped = grad_weight.reshape({groups, C_out_group, col_size});
        const float* in_ptr = input_contig.data_ptr<float>();
        
        if (groups > 1) {
            #pragma omp parallel
            {
                Tensor col = Tensor::empty({col_size, out_spatial}, input_contig.dtype(), input_contig.device());
                float* col_ptr = col.data_ptr<float>();
                
                #pragma omp for
                for (int64_t g = 0; g < groups; ++g) {
                    Tensor gw_g = grad_weight_reshaped.select(0, g);
                    float* gw_ptr = gw_g.data_ptr<float>();
                    
                    for (int64_t n = 0; n < N; ++n) {
                        im2col<float>(in_ptr + (n * C_in + g * C_in_group) * H_in * W_in,
                                      C_in_group, H_in, W_in, kH, kW, pH, pW, sH, sW, dH, dW,
                                      col_ptr);
                        
                        Tensor grad_out_g = grad_output_contig.slice(1, g * C_out_group, (g + 1) * C_out_group);
                        grad_out_g = grad_out_g.select(0, n).reshape({C_out_group, out_spatial});
                        
                        if (!grad_out_g.is_contiguous()) grad_out_g = grad_out_g.contiguous();
                        
                        gemm_direct(false, true, C_out_group, col_size, out_spatial,
                                    1.0f, grad_out_g.data_ptr<float>(), out_spatial,
                                    col_ptr, out_spatial,
                                    1.0f, gw_ptr, col_size);
                    }
                }
            }
        } else {
            // groups == 1, parallelize over N with reduction
            #pragma omp parallel
            {
                Tensor local_gw = Tensor::zeros_like(grad_weight);
                float* local_gw_ptr = local_gw.data_ptr<float>();
                
                Tensor col = Tensor::empty({col_size, out_spatial}, input_contig.dtype(), input_contig.device());
                float* col_ptr = col.data_ptr<float>();
                
                #pragma omp for
                for (int64_t n = 0; n < N; ++n) {
                    im2col<float>(in_ptr + n * C_in * H_in * W_in,
                                  C_in_group, H_in, W_in, kH, kW, pH, pW, sH, sW, dH, dW,
                                  col_ptr);
                    
                    Tensor grad_out_g = grad_output_contig.select(0, n).reshape({C_out_group, out_spatial});
                     // grad_output_contig is (N, C, H, W). select(0, n) is (C, H, W). It is contiguous.
                    
                    gemm_direct(false, true, C_out_group, col_size, out_spatial,
                                1.0f, grad_out_g.data_ptr<float>(), out_spatial,
                                col_ptr, out_spatial,
                                1.0f, local_gw_ptr, col_size);
                }
                
                #pragma omp critical
                {
                    float* dst = grad_weight.data_ptr<float>();
                    int64_t len = grad_weight.numel();
                    for(int64_t i=0; i<len; ++i) dst[i] += local_gw_ptr[i];
                }
            }
        }

    } else {
        TP_THROW(NotImplementedError, "conv2d_grad_weight only supports Float32");
    }
    
    return grad_weight;
}

#include <iostream>

Tensor conv2d_grad_bias_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups) {
    int64_t N = grad_output.size(0);
    int64_t C_out = grad_output.size(1);
    int64_t H_out = grad_output.size(2);
    int64_t W_out = grad_output.size(3);
    
    Tensor grad_bias = Tensor::zeros({C_out}, grad_output.dtype(), grad_output.device());
    
    Tensor grad_output_contig = grad_output.contiguous();
    
    if (grad_output_contig.dtype() == DType::Float32) {
        float* gb_ptr = grad_bias.data_ptr<float>();
        const float* go_ptr = grad_output_contig.data_ptr<float>();
        
        #pragma omp parallel for
        for (int64_t c = 0; c < C_out; ++c) {
            double sum = 0.0;
            for (int64_t n = 0; n < N; ++n) {
                // Offset: n * (C * H * W) + c * (H * W)
                int64_t offset = (n * C_out + c) * H_out * W_out;
                const float* ptr = go_ptr + offset;
                for (int64_t i = 0; i < H_out * W_out; ++i) {
                    sum += ptr[i];
                }
            }
            gb_ptr[c] = (float)sum;
        }
    } else {
         TP_THROW(NotImplementedError, "conv2d_grad_bias only supports Float32");
    }
    
    return grad_bias;
}

// Conv1d Backward (Reuse Conv2d)
Tensor conv1d_grad_input_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, int64_t groups) {
    Tensor grad_out_2d = grad_output.unsqueeze(2); // (N, C, 1, L)
    Tensor in_2d = input.unsqueeze(2);
    Tensor w_2d = weight.unsqueeze(2);
    std::vector<int64_t> s_2d = {1, stride.empty() ? 1 : stride[0]};
    std::vector<int64_t> p_2d = {0, padding.empty() ? 0 : padding[0]};
    std::vector<int64_t> d_2d = {1, dilation.empty() ? 1 : dilation[0]};
    
    Tensor grad_in_2d = conv2d_grad_input_cpu(grad_out_2d, in_2d, w_2d, s_2d, p_2d, d_2d, groups);
    return grad_in_2d.squeeze(2);
}

Tensor conv1d_grad_weight_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, int64_t groups) {
    Tensor grad_out_2d = grad_output.unsqueeze(2);
    Tensor in_2d = input.unsqueeze(2);
    Tensor w_2d = weight.unsqueeze(2);
    std::vector<int64_t> s_2d = {1, stride.empty() ? 1 : stride[0]};
    std::vector<int64_t> p_2d = {0, padding.empty() ? 0 : padding[0]};
    std::vector<int64_t> d_2d = {1, dilation.empty() ? 1 : dilation[0]};
    
    Tensor grad_w_2d = conv2d_grad_weight_cpu(grad_out_2d, in_2d, w_2d, s_2d, p_2d, d_2d, groups);
    return grad_w_2d.squeeze(2);
}

Tensor conv1d_grad_bias_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& dilation, int64_t groups) {
    Tensor grad_out_2d = grad_output.unsqueeze(2);
    Tensor in_2d = input.unsqueeze(2); // Not used but kept for API consistency
    Tensor w_2d = weight.unsqueeze(2); // Not used
    std::vector<int64_t> s_2d = {1, stride.empty() ? 1 : stride[0]};
    std::vector<int64_t> p_2d = {0, padding.empty() ? 0 : padding[0]};
    std::vector<int64_t> d_2d = {1, dilation.empty() ? 1 : dilation[0]};
    
    return conv2d_grad_bias_cpu(grad_out_2d, in_2d, w_2d, s_2d, p_2d, d_2d, groups);
}

// Conv3d Backward
Tensor conv3d_grad_input_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups) {
    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t D_in = input.size(2);
    int64_t H_in = input.size(3);
    int64_t W_in = input.size(4);
    
    int64_t C_out = weight.size(0);
    int64_t C_in_group = weight.size(1);
    int64_t kD = weight.size(2);
    int64_t kH = weight.size(3);
    int64_t kW = weight.size(4);
    
    auto stride = expand_param(stride_arg, 3, "stride");
    auto padding = expand_param(padding_arg, 3, "padding");
    auto dilation = expand_param(dilation_arg, 3, "dilation");
    
    int64_t sD = stride[0]; int64_t sH = stride[1]; int64_t sW = stride[2];
    int64_t pD = padding[0]; int64_t pH = padding[1]; int64_t pW = padding[2];
    int64_t dD = dilation[0]; int64_t dH = dilation[1]; int64_t dW = dilation[2];
    
    int64_t D_out = grad_output.size(2);
    int64_t H_out = grad_output.size(3);
    int64_t W_out = grad_output.size(4);
    
    Tensor grad_input = Tensor::zeros({N, C_in, D_in, H_in, W_in}, input.dtype(), input.device());
    
    if (input.dtype() == DType::Float32) {
        int64_t C_out_group = C_out / groups;
        int64_t col_size = C_in_group * kD * kH * kW;
        int64_t out_spatial = D_out * H_out * W_out;
        
        Tensor grad_output_contig = grad_output.is_contiguous() ? grad_output : grad_output.clone();
        const float* grad_out_ptr = grad_output_contig.data_ptr<float>();
        const float* w_ptr = weight.data_ptr<float>();
        
        // Buffer for col result
        std::vector<float> grad_col_vec(col_size * out_spatial);
        float* grad_col_ptr = grad_col_vec.data();
        
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < groups; ++g) {
                const float* w_g = w_ptr + g * C_out_group * col_size;
                const float* grad_out_g_ptr = grad_out_ptr + (n * C_out + g * C_out_group) * out_spatial;
                
                // GEMM: Weight^T * Grad_Out
                // Weight: (C_out_group, col_size) -> Transpose -> (col_size, C_out_group)
                // Grad_Out: (C_out_group, out_spatial)
                // Result: (col_size, out_spatial)
                
                gemm_direct(true, false, col_size, out_spatial, C_out_group,
                            1.0f, w_g, col_size,
                            grad_out_g_ptr, out_spatial,
                            0.0f, grad_col_ptr, out_spatial);
                
                col2im3d<float>(grad_col_ptr,
                              C_in_group, D_in, H_in, W_in, kD, kH, kW, pD, pH, pW, sD, sH, sW, dD, dH, dW,
                              grad_input.data_ptr<float>() + (n * C_in + g * C_in_group) * D_in * H_in * W_in);
            }
        }
    } else {
        TP_THROW(NotImplementedError, "conv3d_grad_input only supports Float32");
    }
    return grad_input;
}

Tensor conv3d_grad_weight_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups) {
    Tensor grad_output_contig = grad_output.is_contiguous() ? grad_output : grad_output.clone();
    Tensor input_contig = input.is_contiguous() ? input : input.clone();
    
    int64_t N = input_contig.size(0);
    int64_t C_in = input_contig.size(1);
    int64_t D_in = input_contig.size(2);
    int64_t H_in = input_contig.size(3);
    int64_t W_in = input_contig.size(4);
    
    int64_t C_out = weight.size(0);
    int64_t C_in_group = weight.size(1);
    int64_t kD = weight.size(2);
    int64_t kH = weight.size(3);
    int64_t kW = weight.size(4);
    
    auto stride = expand_param(stride_arg, 3, "stride");
    auto padding = expand_param(padding_arg, 3, "padding");
    auto dilation = expand_param(dilation_arg, 3, "dilation");

    int64_t sD = stride[0]; int64_t sH = stride[1]; int64_t sW = stride[2];
    int64_t pD = padding[0]; int64_t pH = padding[1]; int64_t pW = padding[2];
    int64_t dD = dilation[0]; int64_t dH = dilation[1]; int64_t dW = dilation[2];
    
    int64_t D_out = grad_output_contig.size(2);
    int64_t H_out = grad_output_contig.size(3);
    int64_t W_out = grad_output_contig.size(4);
    
    Tensor grad_weight = Tensor::zeros(static_cast<std::vector<int64_t>>(weight.shape()), weight.dtype(), weight.device());
    
    if (input_contig.dtype() == DType::Float32) {
        int64_t C_out_group = C_out / groups;
        int64_t col_size = C_in_group * kD * kH * kW;
        int64_t out_spatial = D_out * H_out * W_out;
        
        float* grad_weight_ptr = grad_weight.data_ptr<float>();
        
        // Buffer
        std::vector<float> col_vec(col_size * out_spatial);
        float* col_ptr = col_vec.data();
        
        const float* in_ptr = input_contig.data_ptr<float>();
        const float* grad_out_ptr = grad_output_contig.data_ptr<float>();
        
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t g = 0; g < groups; ++g) {
                im2col3d<float>(in_ptr + (n * C_in + g * C_in_group) * D_in * H_in * W_in,
                                C_in_group, D_in, H_in, W_in, kD, kH, kW, pD, pH, pW, sD, sH, sW, dD, dH, dW,
                                col_ptr);
                
                const float* grad_ptr = grad_out_ptr + (n * C_out + g * C_out_group) * out_spatial;
                float* gw_ptr = grad_weight_ptr + g * C_out_group * col_size;
                
                gemm_direct(false, true, C_out_group, col_size, out_spatial,
                            1.0f, grad_ptr, out_spatial, // lda
                            col_ptr, out_spatial,        // ldb
                            1.0f, gw_ptr, col_size);     // ldc
            }
        }
    } else {
        TP_THROW(NotImplementedError, "conv3d_grad_weight only supports Float32");
    }
    
    return grad_weight;
}

Tensor conv3d_grad_bias_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride_arg, const std::vector<int64_t>& padding_arg, const std::vector<int64_t>& dilation_arg, int64_t groups) {
    int64_t N = grad_output.size(0);
    int64_t C_out = grad_output.size(1);
    int64_t D_out = grad_output.size(2);
    int64_t H_out = grad_output.size(3);
    int64_t W_out = grad_output.size(4);
    
    Tensor grad_bias = Tensor::zeros({C_out}, grad_output.dtype(), grad_output.device());
    Tensor grad_output_contig = grad_output.is_contiguous() ? grad_output : grad_output.clone();
    
    if (grad_output_contig.dtype() == DType::Float32) {
        float* gb_ptr = grad_bias.data_ptr<float>();
        const float* go_ptr = grad_output_contig.data_ptr<float>();
        
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t c = 0; c < C_out; ++c) {
                float sum = 0.0f;
                for (int64_t i = 0; i < D_out * H_out * W_out; ++i) {
                    sum += go_ptr[(n * C_out + c) * D_out * H_out * W_out + i];
                }
                gb_ptr[c] += sum;
            }
        }
    } else {
         TP_THROW(NotImplementedError, "conv3d_grad_bias only supports Float32");
    }
    return grad_bias;
}

// ConvTranspose2d Backward (Reuse Conv2d)
Tensor conv_transpose2d_grad_input_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& output_padding, int64_t groups, const std::vector<int64_t>& dilation) {
    // grad_input = conv2d(grad_output, weight, stride, padding, dilation, groups)
    // input is unused for calculation, but can be used for checking shape if needed.
    // We assume standard behavior where output shape depends on parameters.
    return conv2d_cpu(grad_output, weight, Tensor(), stride, padding, dilation, groups);
}

Tensor conv_transpose2d_grad_weight_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& output_padding, int64_t groups, const std::vector<int64_t>& dilation) {
    // grad_weight = conv2d_grad_weight(grad_output=input, input=grad_output, weight=weight, ...)
    // Note: input and grad_output are swapped compared to conv2d_grad_weight
    // We treat grad_output (Large) as the input to the convolution (im2col), and input (Small) as the gradient output.
    // Wait, conv2d_grad_weight(grad_output, input, ...)
    // Arg1 (grad_output) is used for MM (Left matrix). Shape (Out, ...).
    // Arg2 (input) is used for Im2Col (Right matrix). Shape (In, ...).
    
    // We want Im2Col on grad_output (Large). So Arg2 = grad_output.
    // We want MM with input (Small). So Arg1 = input.
    
    return conv2d_grad_weight_cpu(input, grad_output, weight, stride, padding, dilation, groups);
}

Tensor conv_transpose2d_grad_bias_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& output_padding, int64_t groups, const std::vector<int64_t>& dilation) {
    return conv2d_grad_bias_cpu(grad_output, input, weight, stride, padding, dilation, groups);
}

// ConvTranspose3d Backward (Reuse Conv3d)
Tensor conv_transpose3d_grad_input_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& output_padding, int64_t groups, const std::vector<int64_t>& dilation) {
    return conv3d_cpu(grad_output, weight, Tensor(), stride, padding, dilation, groups);
}

Tensor conv_transpose3d_grad_weight_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& output_padding, int64_t groups, const std::vector<int64_t>& dilation) {
    return conv3d_grad_weight_cpu(input, grad_output, weight, stride, padding, dilation, groups);
}

Tensor conv_transpose3d_grad_bias_cpu(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const std::vector<int64_t>& stride, const std::vector<int64_t>& padding, const std::vector<int64_t>& output_padding, int64_t groups, const std::vector<int64_t>& dilation) {
    return conv3d_grad_bias_cpu(grad_output, input, weight, stride, padding, dilation, groups);
}

TENSORPLAY_LIBRARY_IMPL(CPU, ConvKernels) {
    m.impl("conv1d", conv1d_cpu);
    m.impl("conv2d", conv2d_cpu);
    m.impl("conv3d", conv3d_cpu);
    m.impl("conv_transpose2d", conv_transpose2d_cpu);
    m.impl("conv_transpose3d", conv_transpose3d_cpu);
    
    m.impl("conv2d_grad_input", conv2d_grad_input_cpu);
    m.impl("conv2d_grad_weight", conv2d_grad_weight_cpu);
    m.impl("conv2d_grad_bias", conv2d_grad_bias_cpu);
    
    m.impl("conv1d_grad_input", conv1d_grad_input_cpu);
    m.impl("conv1d_grad_weight", conv1d_grad_weight_cpu);
    m.impl("conv1d_grad_bias", conv1d_grad_bias_cpu);
    
    m.impl("conv3d_grad_input", conv3d_grad_input_cpu);
    m.impl("conv3d_grad_weight", conv3d_grad_weight_cpu);
    m.impl("conv3d_grad_bias", conv3d_grad_bias_cpu);
    
    m.impl("conv_transpose2d_grad_input", conv_transpose2d_grad_input_cpu);
    m.impl("conv_transpose2d_grad_weight", conv_transpose2d_grad_weight_cpu);
    m.impl("conv_transpose2d_grad_bias", conv_transpose2d_grad_bias_cpu);
    
    m.impl("conv_transpose3d_grad_input", conv_transpose3d_grad_input_cpu);
    m.impl("conv_transpose3d_grad_weight", conv_transpose3d_grad_weight_cpu);
    m.impl("conv_transpose3d_grad_bias", conv_transpose3d_grad_bias_cpu);
}

} // namespace cpu
} // namespace tensorplay
