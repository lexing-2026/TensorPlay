#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "TypePromotion.h"
#include "Utils.h"
#include "Exception.h"
#include "OneDNNContext.h"
#include "Allocator.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <immintrin.h>

#ifdef USE_ONEDNN
#include "dnnl.hpp"
#endif

#ifdef USE_MKL
#include <mkl.h>
#elif defined(USE_BLAS)
#include <cblas.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensorplay {
namespace cpu {

// --- Helper for Binary Ops ---

template<typename Op, typename MklOp>
Tensor binary_op_kernel_impl(const Tensor& self, const Tensor& other, Op op, MklOp mkl_op, bool use_mkl_op = false, bool force_float = false) {
    std::vector<int64_t> out_shape;
    try {
        out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    } catch (const std::exception& e) {
        std::cout << "DEBUG: broadcast_shapes failed in binary_op_kernel_impl: " << e.what() << std::endl;
        std::cout << "Self shape: ";
        for (auto s : self.shape()) std::cout << s << " ";
        std::cout << std::endl;
        std::cout << "Other shape: ";
        for (auto s : other.shape()) std::cout << s << " ";
        std::cout << std::endl;
        throw;
    }
    DType result_dtype = promoteTypes(self.dtype(), other.dtype());
    if (force_float && isIntegralType(result_dtype, true)) {
        result_dtype = DType::Float32;
    }

    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());
    
    bool optimized = false;
    if (result_dtype == DType::Float32 && 
        self.dtype() == DType::Float32 && 
        other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() && result.is_contiguous() &&
        self.shape() == other.shape()) {
        
        #ifdef USE_MKL
        if (use_mkl_op) {
            int64_t n = self.numel();
            mkl_op((int)n, self.data_ptr<float>(), other.data_ptr<float>(), result.data_ptr<float>());
            optimized = true;
        }
        #endif
    }
    
    if (!optimized) {
        Tensor self_casted = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
        Tensor other_casted = (other.dtype() == result_dtype) ? other : other.to(result_dtype);

        Tensor self_expanded = self_casted.expand(out_shape);
        Tensor other_expanded = other_casted.expand(out_shape);
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            apply_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                                     self_expanded, self_expanded.strides(), \
                                     other_expanded, other_expanded.strides(), \
                                     0, 0, 0, 0, out_shape, op); \
            break; \
        }

        switch (result_dtype) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(TypeError, "binary_op: unsupported dtype");
        }
        #undef OP_CASE
    }
    
    return result;
}

// --- Binary Kernels ---

Tensor add_kernel(const Tensor& self, const Tensor& other, Scalar alpha) {
    #ifdef USE_ONEDNN
    if (OneDNNContext::is_enabled()) {
        auto self_impl = self.unsafeGetTensorImpl();
        auto other_impl = other.unsafeGetTensorImpl();
        bool self_blocked = self_impl->has_onednn_md();
        bool other_blocked = other_impl->has_onednn_md();

        if (self_blocked || other_blocked) {
            bool match = false;
            if (self_blocked && other_blocked) {
                auto md1 = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                auto md2 = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                if (*md1 == *md2) match = true;
            }

            if (match) {
                 // Optimization: Treat as contiguous 1D array
                 auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                 
                 std::vector<int64_t> out_shape;
                 try {
                     out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
                 } catch (const std::exception& e) {
                     std::cout << "DEBUG: broadcast_shapes failed in add_kernel (OneDNN): " << e.what() << std::endl;
                     std::cout << "Self shape: ";
                     for (auto s : self.shape()) std::cout << s << " ";
                     std::cout << std::endl;
                     std::cout << "Other shape: ";
                     for (auto s : other.shape()) std::cout << s << " ";
                     std::cout << std::endl;
                     throw;
                 }

                 DType result_dtype = promoteTypes(self.dtype(), other.dtype());
                 if (alpha.isFloatingPoint()) result_dtype = promoteTypes(result_dtype, DType::Float32);
                 Tensor result = Tensor::empty(out_shape, result_dtype, self.device());

                 size_t req_size = md->get_size();
                 if (result.numel() * sizeof(float) < req_size) {
                      Allocator* allocator = getAllocator(result.device().type());
                      Storage new_storage(req_size, allocator);
                      result.unsafeGetTensorImpl()->set_storage(new_storage);
                 }
                 result.unsafeGetTensorImpl()->set_onednn_md(self_impl->get_onednn_md());

                 int64_t n = req_size / sizeof(float);
                 float alpha_val = alpha.to<float>();
                 float* r_ptr = result.data_ptr<float>();
                 const float* s_ptr = self.data_ptr<float>();
                 const float* o_ptr = other.data_ptr<float>();

                 // Reuse AVX logic
                 #if defined(__AVX512F__)
                 if (std::abs(alpha_val - 1.0f) < 1e-6) {
                      #ifdef _OPENMP
                      #pragma omp parallel for
                      #endif
                      for (int64_t i = 0; i < n; i += 16) {
                          if (i + 16 <= n) {
                              __m512 a = _mm512_loadu_ps(s_ptr + i);
                              __m512 b = _mm512_loadu_ps(o_ptr + i);
                              _mm512_storeu_ps(r_ptr + i, _mm512_add_ps(a, b));
                          } else {
                              for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] + o_ptr[j];
                          }
                      }
                 } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                      #ifdef _OPENMP
                      #pragma omp parallel for
                      #endif
                      for (int64_t i = 0; i < n; i += 16) {
                          if (i + 16 <= n) {
                              __m512 a = _mm512_loadu_ps(s_ptr + i);
                              __m512 b = _mm512_loadu_ps(o_ptr + i);
                              _mm512_storeu_ps(r_ptr + i, _mm512_sub_ps(a, b));
                          } else {
                              for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] - o_ptr[j];
                          }
                      }
                 } else {
                      __m512 valpha = _mm512_set1_ps(alpha_val);
                      #ifdef _OPENMP
                      #pragma omp parallel for
                      #endif
                      for (int64_t i = 0; i < n; i += 16) {
                          if (i + 16 <= n) {
                              __m512 a = _mm512_loadu_ps(s_ptr + i);
                              __m512 b = _mm512_loadu_ps(o_ptr + i);
                              _mm512_storeu_ps(r_ptr + i, _mm512_fmadd_ps(valpha, b, a));
                          } else {
                              for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] + alpha_val * o_ptr[j];
                          }
                      }
                 }
                 #elif defined(__AVX2__)
                 if (std::abs(alpha_val - 1.0f) < 1e-6) {
                      #ifdef _OPENMP
                      #pragma omp parallel for
                      #endif
                      for (int64_t i = 0; i < n; i += 8) {
                          if (i + 8 <= n) {
                              __m256 a = _mm256_loadu_ps(s_ptr + i);
                              __m256 b = _mm256_loadu_ps(o_ptr + i);
                              _mm256_storeu_ps(r_ptr + i, _mm256_add_ps(a, b));
                          } else {
                              for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] + o_ptr[j];
                          }
                      }
                 } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                      #ifdef _OPENMP
                      #pragma omp parallel for
                      #endif
                      for (int64_t i = 0; i < n; i += 8) {
                          if (i + 8 <= n) {
                              __m256 a = _mm256_loadu_ps(s_ptr + i);
                              __m256 b = _mm256_loadu_ps(o_ptr + i);
                              _mm256_storeu_ps(r_ptr + i, _mm256_sub_ps(a, b));
                          } else {
                              for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] - o_ptr[j];
                          }
                      }
                 } else {
                      __m256 valpha = _mm256_set1_ps(alpha_val);
                      #ifdef _OPENMP
                      #pragma omp parallel for
                      #endif
                      for (int64_t i = 0; i < n; i += 8) {
                          if (i + 8 <= n) {
                              __m256 a = _mm256_loadu_ps(s_ptr + i);
                              __m256 b = _mm256_loadu_ps(o_ptr + i);
                              _mm256_storeu_ps(r_ptr + i, _mm256_add_ps(a, _mm256_mul_ps(valpha, b)));
                          } else {
                              for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] + alpha_val * o_ptr[j];
                          }
                      }
                 }
                 #else
                 for(int64_t i=0; i<n; ++i) r_ptr[i] = s_ptr[i] + alpha_val * o_ptr[i];
                 #endif
                 
                 return result;
            } else {
                 // Reorder to NCHW
                 auto& eng = OneDNNContext::get_engine();
                 auto& s = OneDNNContext::get_stream();
                 
                 Tensor self_nchw = self;
                 if (self_blocked) {
                      auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                      dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(self.shape());
                      auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                      self_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
                      auto src_mem = dnnl::memory(*md, eng, self.data_ptr<float>());
                      auto dst_mem = dnnl::memory(nchw_md, eng, self_nchw.data_ptr<float>());
                      dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                 }
                 
                 Tensor other_nchw = other;
                 if (other_blocked) {
                      auto md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                      dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
                      auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                      other_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(other.shape()), other.dtype(), other.device());
                      auto src_mem = dnnl::memory(*md, eng, other.data_ptr<float>());
                      auto dst_mem = dnnl::memory(nchw_md, eng, other_nchw.data_ptr<float>());
                      dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                 }
                 if (self_blocked || other_blocked) s.wait();
                 
                 return add_kernel(self_nchw, other_nchw, alpha);
            }
        }
    }
    #endif

    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    DType result_dtype = promoteTypes(self.dtype(), other.dtype());
    if (alpha.isFloatingPoint()) {
        result_dtype = promoteTypes(result_dtype, DType::Float32);
    }
    Tensor result = Tensor::empty(out_shape, result_dtype, self.device());

    bool optimized = false;
    // Optimization for same-shape tensors (handles contiguous and non-contiguous via temporary copies)
    if (result_dtype == DType::Float32 && 
        self.dtype() == DType::Float32 && 
        other.dtype() == DType::Float32 &&
        self.shape() == other.shape()) {
        
        // Create contiguous accessors (might trigger copy)
        Tensor self_contig = self.is_contiguous() ? self : self.contiguous();
        Tensor other_contig = other.is_contiguous() ? other : other.contiguous();
        // Result is already contiguous if created via empty(), but check to be safe or if passed in
        Tensor result_contig = result.is_contiguous() ? result : result.contiguous();

        int64_t n = self_contig.numel();
        float alpha_val = alpha.to<float>();
        float* r_ptr = result_contig.data_ptr<float>();
        const float* s_ptr = self_contig.data_ptr<float>();
        const float* o_ptr = other_contig.data_ptr<float>();
        
        #ifdef USE_MKL
        if (std::abs(alpha_val - 1.0f) < 1e-6) {
            #ifdef _OPENMP
            #pragma omp parallel
            {
                int64_t num_threads = omp_get_num_threads();
                int64_t tid = omp_get_thread_num();
                int64_t chunk_size = (n + num_threads - 1) / num_threads;
                int64_t start = tid * chunk_size;
                int64_t end = std::min(start + chunk_size, n);
                if (start < end) {
                    vsAdd(end - start, s_ptr + start, o_ptr + start, r_ptr + start);
                }
            }
            #else
            vsAdd((int)n, s_ptr, o_ptr, r_ptr);
            #endif
            optimized = true;
        } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
            #ifdef _OPENMP
            #pragma omp parallel
            {
                int64_t num_threads = omp_get_num_threads();
                int64_t tid = omp_get_thread_num();
                int64_t chunk_size = (n + num_threads - 1) / num_threads;
                int64_t start = tid * chunk_size;
                int64_t end = std::min(start + chunk_size, n);
                if (start < end) {
                    vsSub(end - start, s_ptr + start, o_ptr + start, r_ptr + start);
                }
            }
            #else
            vsSub((int)n, s_ptr, o_ptr, r_ptr);
            #endif
            optimized = true;
        }
        #endif
        
        if (!optimized) {
            #if defined(__AVX512F__)
            if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 16) {
                     if (i + 16 <= n) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(r_ptr + i, _mm512_add_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] + o_ptr[j];
                     }
                 }
            } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 16) {
                     if (i + 16 <= n) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(r_ptr + i, _mm512_sub_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] - o_ptr[j];
                     }
                 }
            } else {
                 __m512 valpha = _mm512_set1_ps(alpha_val);
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 16) {
                     if (i + 16 <= n) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(r_ptr + i, _mm512_fmadd_ps(valpha, b, a));
                     } else {
                         for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] + alpha_val * o_ptr[j];
                     }
                 }
            }
            #elif defined(__AVX2__)
            if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 8) {
                     if (i + 8 <= n) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(r_ptr + i, _mm256_add_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] + o_ptr[j];
                     }
                 }
            } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 8) {
                     if (i + 8 <= n) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(r_ptr + i, _mm256_sub_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] - o_ptr[j];
                     }
                 }
            } else {
                 __m256 valpha = _mm256_set1_ps(alpha_val);
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 8) {
                     if (i + 8 <= n) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         // Use mul+add instead of fmadd to be safe with AVX2 but no FMA3
                         _mm256_storeu_ps(r_ptr + i, _mm256_add_ps(a, _mm256_mul_ps(valpha, b)));
                     } else {
                         for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] + alpha_val * o_ptr[j];
                     }
                 }
            }
            #else
            if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; ++i) {
                     r_ptr[i] = s_ptr[i] + o_ptr[i];
                 }
             } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; ++i) {
                     r_ptr[i] = s_ptr[i] - o_ptr[i];
                 }
             } else {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; ++i) {
                     r_ptr[i] = s_ptr[i] + alpha_val * o_ptr[i];
                 }
             }
            #endif
            optimized = true;
        }
    }
    
    if (!optimized) {
        Tensor self_casted = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
        Tensor other_casted = (other.dtype() == result_dtype) ? other : other.to(result_dtype);
        
        Tensor self_expanded = self_casted.expand(out_shape);
        Tensor other_expanded = other_casted.expand(out_shape);
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            auto op = [alpha](ctype a, ctype b) -> ctype { \
                if constexpr (std::is_floating_point_v<ctype>) { \
                    return a + alpha.to<ctype>() * b; \
                } else { \
                    if (alpha.isFloatingPoint()) { \
                        return static_cast<ctype>(a + alpha.toDouble() * b); \
                    } else { \
                        return static_cast<ctype>(a + alpha.to<int64_t>() * b); \
                    } \
                } \
            }; \
            apply_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                                     self_expanded, self_expanded.strides(), \
                                     other_expanded, other_expanded.strides(), \
                                     0, 0, 0, 0, out_shape, op); \
            break; \
        }
        switch (result_dtype) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(TypeError, "add: unsupported dtype");
        }
        #undef OP_CASE
    }
    return result;
}

Tensor sub_kernel(const Tensor& self, const Tensor& other, Scalar alpha) {
    if (alpha.isFloatingPoint()) {
        return add_kernel(self, other, Scalar(-alpha.toDouble()));
    } else {
        if (alpha.isIntegral()) {
             return add_kernel(self, other, Scalar(-alpha.to<int64_t>()));
        }
        return add_kernel(self, other, Scalar(-alpha.to<double>()));
    }
}

Tensor mul_kernel(const Tensor& self, const Tensor& other) {
    #ifdef USE_ONEDNN
    if (OneDNNContext::is_enabled()) {
        auto self_impl = self.unsafeGetTensorImpl();
        auto other_impl = other.unsafeGetTensorImpl();
        bool self_blocked = self_impl->has_onednn_md();
        bool other_blocked = other_impl->has_onednn_md();

        if (self_blocked || other_blocked) {
            bool match = false;
            if (self_blocked && other_blocked) {
                auto md1 = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                auto md2 = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                if (*md1 == *md2) match = true;
            }

            if (match) {
                 auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                 std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
                 DType result_dtype = promoteTypes(self.dtype(), other.dtype());
                 Tensor result = Tensor::empty(out_shape, result_dtype, self.device());

                 size_t req_size = md->get_size();
                 if (result.numel() * sizeof(float) < req_size) {
                      Allocator* allocator = getAllocator(result.device().type());
                      Storage new_storage(req_size, allocator);
                      result.unsafeGetTensorImpl()->set_storage(new_storage);
                 }
                 result.unsafeGetTensorImpl()->set_onednn_md(self_impl->get_onednn_md());

                 int64_t n = req_size / sizeof(float);
                 float* r_ptr = result.data_ptr<float>();
                 const float* s_ptr = self.data_ptr<float>();
                 const float* o_ptr = other.data_ptr<float>();

                 #if defined(__AVX512F__)
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 16) {
                     if (i + 16 <= n) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(r_ptr + i, _mm512_mul_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] * o_ptr[j];
                     }
                 }
                 #elif defined(__AVX2__)
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 8) {
                     if (i + 8 <= n) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(r_ptr + i, _mm256_mul_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] * o_ptr[j];
                     }
                 }
                 #else
                 for(int64_t i=0; i<n; ++i) r_ptr[i] = s_ptr[i] * o_ptr[i];
                 #endif
                 return result;
            } else {
                 // Reorder to NCHW
                 auto& eng = OneDNNContext::get_engine();
                 auto& s = OneDNNContext::get_stream();
                 
                 Tensor self_nchw = self;
                 if (self_blocked) {
                      auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                      dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(self.shape());
                      auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                      self_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
                      auto src_mem = dnnl::memory(*md, eng, self.data_ptr<float>());
                      auto dst_mem = dnnl::memory(nchw_md, eng, self_nchw.data_ptr<float>());
                      dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                 }
                 
                 Tensor other_nchw = other;
                 if (other_blocked) {
                      auto md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                      dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
                      auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                      other_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(other.shape()), other.dtype(), other.device());
                      auto src_mem = dnnl::memory(*md, eng, other.data_ptr<float>());
                      auto dst_mem = dnnl::memory(nchw_md, eng, other_nchw.data_ptr<float>());
                      dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                 }
                 if (self_blocked || other_blocked) s.wait();
                 return mul_kernel(self_nchw, other_nchw);
            }
        }
    }
    #endif

    auto op = [](auto a, auto b) { return a * b; };
    auto mkl_op = [](int n, float* a, float* b, float* y) {
        #ifdef USE_MKL
        vsMul(n, a, b, y);
        #endif
    };
    return binary_op_kernel_impl(self, other, op, mkl_op, true);
}

Tensor div_kernel(const Tensor& self, const Tensor& other) {
    #ifdef USE_ONEDNN
    if (OneDNNContext::is_enabled()) {
        auto self_impl = self.unsafeGetTensorImpl();
        auto other_impl = other.unsafeGetTensorImpl();
        bool self_blocked = self_impl->has_onednn_md();
        bool other_blocked = other_impl->has_onednn_md();

        if (self_blocked || other_blocked) {
            bool match = false;
            if (self_blocked && other_blocked) {
                auto md1 = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                auto md2 = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                if (*md1 == *md2) match = true;
            }

            if (match) {
                 auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                 std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
                 DType result_dtype = promoteTypes(self.dtype(), other.dtype());
                 if (result_dtype != DType::Float32) result_dtype = DType::Float32; // Div always produces float
                 Tensor result = Tensor::empty(out_shape, result_dtype, self.device());

                 size_t req_size = md->get_size();
                 if (result.numel() * sizeof(float) < req_size) {
                      Allocator* allocator = getAllocator(result.device().type());
                      Storage new_storage(req_size, allocator);
                      result.unsafeGetTensorImpl()->set_storage(new_storage);
                 }
                 result.unsafeGetTensorImpl()->set_onednn_md(self_impl->get_onednn_md());

                 int64_t n = req_size / sizeof(float);
                 float* r_ptr = result.data_ptr<float>();
                 const float* s_ptr = self.data_ptr<float>();
                 const float* o_ptr = other.data_ptr<float>();

                 #if defined(__AVX512F__)
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 16) {
                     if (i + 16 <= n) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(r_ptr + i, _mm512_div_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] / o_ptr[j];
                     }
                 }
                 #elif defined(__AVX2__)
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 8) {
                     if (i + 8 <= n) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(r_ptr + i, _mm256_div_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) r_ptr[j] = s_ptr[j] / o_ptr[j];
                     }
                 }
                 #else
                 for(int64_t i=0; i<n; ++i) r_ptr[i] = s_ptr[i] / o_ptr[i];
                 #endif
                 return result;
            } else {
                 // Reorder to NCHW
                 auto& eng = OneDNNContext::get_engine();
                 auto& s = OneDNNContext::get_stream();
                 
                 Tensor self_nchw = self;
                 if (self_blocked) {
                      auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                      dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(self.shape());
                      auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                      self_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
                      auto src_mem = dnnl::memory(*md, eng, self.data_ptr<float>());
                      auto dst_mem = dnnl::memory(nchw_md, eng, self_nchw.data_ptr<float>());
                      dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                 }
                 
                 Tensor other_nchw = other;
                 if (other_blocked) {
                      auto md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                      dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
                      auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                      other_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(other.shape()), other.dtype(), other.device());
                      auto src_mem = dnnl::memory(*md, eng, other.data_ptr<float>());
                      auto dst_mem = dnnl::memory(nchw_md, eng, other_nchw.data_ptr<float>());
                      dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                 }
                 if (self_blocked || other_blocked) s.wait();
                 return div_kernel(self_nchw, other_nchw);
            }
        }
    }
    #endif

    auto op = [](auto a, auto b) { 
        using T = std::decay_t<decltype(a)>;
        if constexpr (std::is_same_v<T, bool>) return static_cast<float>(a) / static_cast<float>(b);
        else return a / b;
    };
    auto mkl_op = [](int n, float* a, float* b, float* y) {
        #ifdef USE_MKL
        vsDiv(n, a, b, y);
        #endif
    };
    return binary_op_kernel_impl(self, other, op, mkl_op, true, true);
}

// --- Inplace Binary Kernels ---

Tensor& add_inplace_kernel(Tensor& self, const Tensor& other, Scalar alpha) {
    std::vector<int64_t> out_shape;
    try {
        out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    } catch (const std::exception& e) {
        std::cout << "DEBUG: broadcast_shapes failed in add_inplace_kernel: " << e.what() << std::endl;
        std::cout << "Self shape: ";
        for (auto s : self.shape()) std::cout << s << " ";
        std::cout << std::endl;
        std::cout << "Other shape: ";
        for (auto s : other.shape()) std::cout << s << " ";
        std::cout << std::endl;
        throw;
    }
    
    if (static_cast<std::vector<int64_t>>(self.shape()) != out_shape) {
        TP_THROW(RuntimeError, "output with shape " + self.shape().toString() + " doesn't match the broadcast shape " + Size(out_shape).toString());
    }

    if (self.shape() != other.shape()) {
    }

    #ifdef USE_ONEDNN
    if (OneDNNContext::is_enabled()) {
        auto self_impl = self.unsafeGetTensorImpl();
        auto other_impl = other.unsafeGetTensorImpl();
        bool self_blocked = self_impl->has_onednn_md();
        bool other_blocked = other_impl->has_onednn_md();
        
        if (self_blocked && self.shape() != other.shape()) {
            // Broadcasting required. Unblock self (convert to dense).
            auto& eng = OneDNNContext::get_engine();
            auto& s = OneDNNContext::get_stream();
            auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
            
            dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(self.shape());
            auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
            
            // Debug info
            // std::cout << "DEBUG: Broadcasting unblock. Src dims: " << md->get_ndims() << " Dst dims: " << nchw_md.get_ndims() << std::endl;
            // for(int i=0; i<md->get_ndims(); ++i) std::cout << md->get_dims()[i] << " "; std::cout << std::endl;
            
            Tensor self_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
            auto src_mem = dnnl::memory(*md, eng, self.data_ptr<float>());
            auto dst_mem = dnnl::memory(nchw_md, eng, self_nchw.data_ptr<float>());
            
            try {
                dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                s.wait();
            } catch (const dnnl::error& e) {
                TP_THROW(RuntimeError, "OneDNN reorder failed in add_inplace_kernel (unblock): " + std::string(e.message));
            }
            
            // Replace storage and clear MD
            self.unsafeGetTensorImpl()->set_storage(self_nchw.unsafeGetTensorImpl()->storage());
            self.unsafeGetTensorImpl()->set_onednn_md(nullptr);
            self_blocked = false;
        }

        if (self_blocked) {
             // Self is blocked, we MUST preserve it.
             Tensor other_matching = other;
             bool match = false;
             if (other_blocked) {
                 auto md1 = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                 auto md2 = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                 if (*md1 == *md2) match = true;
             }
             
             if (!match) {
                  auto& eng = OneDNNContext::get_engine();
                  auto& s = OneDNNContext::get_stream();
                  auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                  
                  size_t req_size = md->get_size();
                  
                  Tensor temp = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
                  if (temp.numel() * sizeof(float) < req_size) {
                      Allocator* allocator = getAllocator(self.device().type());
                      Storage new_storage(req_size, allocator);
                      temp.unsafeGetTensorImpl()->set_storage(new_storage);
                  }
                  temp.unsafeGetTensorImpl()->set_onednn_md(self_impl->get_onednn_md());
                  
                  auto dst_mem = dnnl::memory(*md, eng, temp.data_ptr<float>());
                  
                  try {
                       if (other_blocked) {
                            auto other_md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                            auto src_mem = dnnl::memory(*other_md, eng, other.data_ptr<float>());
                            dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                       } else {
                            dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
                            auto src_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                            auto src_mem = dnnl::memory(src_md, eng, other.data_ptr<float>());
                            dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                       }
                       s.wait();
                   } catch (const std::exception& e) {
                      TP_THROW(RuntimeError, "OneDNN reorder failed in add_inplace_kernel (match block): " + std::string(e.what()));
                  }
                  other_matching = temp;
             }
             
             float alpha_val = alpha.to<float>();
             auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
             int64_t n = md->get_size() / sizeof(float);
             
             float* s_ptr = self.data_ptr<float>();
             const float* o_ptr = other_matching.data_ptr<float>();
             
             #if defined(__AVX512F__)
             if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 16) {
                     if (i + 16 <= n) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(s_ptr + i, _mm512_add_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) s_ptr[j] += o_ptr[j];
                     }
                 }
             } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 16) {
                     if (i + 16 <= n) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(s_ptr + i, _mm512_sub_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) s_ptr[j] -= o_ptr[j];
                     }
                 }
             } else {
                 __m512 valpha = _mm512_set1_ps(alpha_val);
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 16) {
                     if (i + 16 <= n) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(s_ptr + i, _mm512_fmadd_ps(valpha, b, a));
                     } else {
                         for (int64_t j = i; j < n; ++j) s_ptr[j] += alpha_val * o_ptr[j];
                     }
                 }
             }
             #elif defined(__AVX2__)
             if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 8) {
                     if (i + 8 <= n) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(s_ptr + i, _mm256_add_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) s_ptr[j] += o_ptr[j];
                     }
                 }
             } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 8) {
                     if (i + 8 <= n) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(s_ptr + i, _mm256_sub_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) s_ptr[j] -= o_ptr[j];
                     }
                 }
             } else {
                 __m256 valpha = _mm256_set1_ps(alpha_val);
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 8) {
                     if (i + 8 <= n) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(s_ptr + i, _mm256_add_ps(a, _mm256_mul_ps(valpha, b)));
                     } else {
                         for (int64_t j = i; j < n; ++j) s_ptr[j] += alpha_val * o_ptr[j];
                     }
                 }
             }
             #else
             for (int64_t i = 0; i < n; ++i) s_ptr[i] += alpha_val * o_ptr[i];
             #endif
             return self;
        } else if (other_blocked) {
             auto& eng = OneDNNContext::get_engine();
             auto& s = OneDNNContext::get_stream();
             auto md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
             dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
             auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
             
             Tensor other_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(other.shape()), other.dtype(), other.device());
             try {
                 auto src_mem = dnnl::memory(*md, eng, other.data_ptr<float>());
                 auto dst_mem = dnnl::memory(nchw_md, eng, other_nchw.data_ptr<float>());
                 dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                 s.wait();
             } catch (const std::exception& e) {
                 TP_THROW(RuntimeError, "OneDNN reorder failed in add_inplace_kernel (other_blocked): " + std::string(e.what()));
             }
             
             return add_inplace_kernel(self, other_nchw, alpha);
        }
    }
    #endif

    bool optimized = false;
    if (self.dtype() == DType::Float32 && 
        other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() &&
        self.shape() == other.shape()) {
        
        float alpha_val = alpha.to<float>();
        int64_t n = self.numel();

        #if defined(USE_MKL) || defined(USE_BLAS)
        cblas_saxpy((int)n, alpha_val, other.data_ptr<float>(), 1, self.data_ptr<float>(), 1);
        optimized = true;
        #endif

        if (!optimized) {
             float* s_ptr = self.data_ptr<float>();
             const float* o_ptr = other.data_ptr<float>();
             
             #if defined(__AVX512F__)
             if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 16) {
                     if (i + 16 <= n) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(s_ptr + i, _mm512_add_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) s_ptr[j] += o_ptr[j];
                     }
                 }
             } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 16) {
                     if (i + 16 <= n) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(s_ptr + i, _mm512_sub_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) s_ptr[j] -= o_ptr[j];
                     }
                 }
             } else {
                 __m512 valpha = _mm512_set1_ps(alpha_val);
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 16) {
                     if (i + 16 <= n) {
                         __m512 a = _mm512_loadu_ps(s_ptr + i);
                         __m512 b = _mm512_loadu_ps(o_ptr + i);
                         _mm512_storeu_ps(s_ptr + i, _mm512_fmadd_ps(valpha, b, a));
                     } else {
                         for (int64_t j = i; j < n; ++j) s_ptr[j] += alpha_val * o_ptr[j];
                     }
                 }
             }
             #elif defined(__AVX2__)
             if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 8) {
                     if (i + 8 <= n) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(s_ptr + i, _mm256_add_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) s_ptr[j] += o_ptr[j];
                     }
                 }
             } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 8) {
                     if (i + 8 <= n) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(s_ptr + i, _mm256_sub_ps(a, b));
                     } else {
                         for (int64_t j = i; j < n; ++j) s_ptr[j] -= o_ptr[j];
                     }
                 }
             } else {
                 __m256 valpha = _mm256_set1_ps(alpha_val);
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; i += 8) {
                     if (i + 8 <= n) {
                         __m256 a = _mm256_loadu_ps(s_ptr + i);
                         __m256 b = _mm256_loadu_ps(o_ptr + i);
                         _mm256_storeu_ps(s_ptr + i, _mm256_add_ps(a, _mm256_mul_ps(valpha, b)));
                     } else {
                         for (int64_t j = i; j < n; ++j) s_ptr[j] += alpha_val * o_ptr[j];
                     }
                 }
             }
             #else
             if (std::abs(alpha_val - 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; ++i) {
                     s_ptr[i] += o_ptr[i];
                 }
             } else if (std::abs(alpha_val + 1.0f) < 1e-6) {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; ++i) {
                     s_ptr[i] -= o_ptr[i];
                 }
             } else {
                 #ifdef _OPENMP
                 #pragma omp parallel for
                 #endif
                 for (int64_t i = 0; i < n; ++i) {
                     s_ptr[i] += alpha_val * o_ptr[i];
                 }
             }
             #endif
             optimized = true;
        }
    }
    
    if (!optimized) {
        Tensor other_expanded = other.expand(out_shape);
        if (other_expanded.dtype() != self.dtype()) {
            other_expanded = other_expanded.to(self.dtype());
        }
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            auto op = [alpha](ctype a, ctype b) -> ctype { \
                if constexpr (std::is_floating_point_v<ctype>) { \
                    return a + alpha.to<ctype>() * b; \
                } else { \
                    if (alpha.isFloatingPoint()) { \
                        return static_cast<ctype>(a + alpha.toDouble() * b); \
                    } else { \
                        return static_cast<ctype>(a + alpha.to<int64_t>() * b); \
                    } \
                } \
            }; \
            apply_op_recursive<ctype>(self.data_ptr<ctype>(), self.strides(), \
                                     self, self.strides(), \
                                     other_expanded, other_expanded.strides(), \
                                     0, 0, 0, 0, out_shape, op); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(TypeError, "add_: unsupported dtype");
        }
        #undef OP_CASE
    }
    return self;
}

Tensor& sub_inplace_kernel(Tensor& self, const Tensor& other, Scalar alpha) {
    if (alpha.isFloatingPoint()) {
        return add_inplace_kernel(self, other, Scalar(-alpha.toDouble()));
    } else {
        if (alpha.isIntegral()) {
             return add_inplace_kernel(self, other, Scalar(-alpha.to<int64_t>()));
        }
        return add_inplace_kernel(self, other, Scalar(-alpha.to<double>()));
    }
}

Tensor& mul_inplace_kernel(Tensor& self, const Tensor& other) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    if (static_cast<std::vector<int64_t>>(self.shape()) != out_shape) TP_THROW(RuntimeError, "mul_: shape mismatch");

    #ifdef USE_ONEDNN
    if (OneDNNContext::is_enabled()) {
        auto self_impl = self.unsafeGetTensorImpl();
        auto other_impl = other.unsafeGetTensorImpl();
        bool self_blocked = self_impl->has_onednn_md();
        bool other_blocked = other_impl->has_onednn_md();
        
        if (self_blocked) {
             Tensor other_matching = other;
             bool match = false;
             if (other_blocked) {
                 auto md1 = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                 auto md2 = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                 if (*md1 == *md2) match = true;
             }
             
             if (!match) {
                  auto& eng = OneDNNContext::get_engine();
                  auto& s = OneDNNContext::get_stream();
                  auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                  
                  size_t req_size = md->get_size();
                  Tensor temp = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
                  if (temp.numel() * sizeof(float) < req_size) {
                      Allocator* allocator = getAllocator(self.device().type());
                      Storage new_storage(req_size, allocator);
                      temp.unsafeGetTensorImpl()->set_storage(new_storage);
                  }
                  temp.unsafeGetTensorImpl()->set_onednn_md(self_impl->get_onednn_md());
                  
                  auto dst_mem = dnnl::memory(*md, eng, temp.data_ptr<float>());
                  
                  if (other_blocked) {
                       auto other_md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                       auto src_mem = dnnl::memory(*other_md, eng, other.data_ptr<float>());
                       dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                  } else {
                       dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
                       auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                       auto src_mem = dnnl::memory(nchw_md, eng, other.data_ptr<float>());
                       dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                  }
                  s.wait();
                  other_matching = temp;
             }
             
             auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
             int64_t n = md->get_size() / sizeof(float);
             
             float* s_ptr = self.data_ptr<float>();
             const float* o_ptr = other_matching.data_ptr<float>();
             
             #if defined(__AVX512F__)
             #ifdef _OPENMP
             #pragma omp parallel for
             #endif
             for (int64_t i = 0; i < n; i += 16) {
                 if (i + 16 <= n) {
                     __m512 a = _mm512_loadu_ps(s_ptr + i);
                     __m512 b = _mm512_loadu_ps(o_ptr + i);
                     _mm512_storeu_ps(s_ptr + i, _mm512_mul_ps(a, b));
                 } else {
                     for (int64_t j = i; j < n; ++j) s_ptr[j] *= o_ptr[j];
                 }
             }
             #elif defined(__AVX2__)
             #ifdef _OPENMP
             #pragma omp parallel for
             #endif
             for (int64_t i = 0; i < n; i += 8) {
                 if (i + 8 <= n) {
                     __m256 a = _mm256_loadu_ps(s_ptr + i);
                     __m256 b = _mm256_loadu_ps(o_ptr + i);
                     _mm256_storeu_ps(s_ptr + i, _mm256_mul_ps(a, b));
                 } else {
                     for (int64_t j = i; j < n; ++j) s_ptr[j] *= o_ptr[j];
                 }
             }
             #else
             for (int64_t i = 0; i < n; ++i) s_ptr[i] *= o_ptr[i];
             #endif
             return self;
        } else if (other_blocked) {
             auto& eng = OneDNNContext::get_engine();
             auto& s = OneDNNContext::get_stream();
             auto md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
             dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
             auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
             
             Tensor other_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(other.shape()), other.dtype(), other.device());
             auto src_mem = dnnl::memory(*md, eng, other.data_ptr<float>());
             auto dst_mem = dnnl::memory(nchw_md, eng, other_nchw.data_ptr<float>());
             dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
             s.wait();
             
             return mul_inplace_kernel(self, other_nchw);
        }
    }
    #endif

    bool optimized = false;
    if (self.dtype() == DType::Float32 && other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() && self.shape() == other.shape()) {
        #ifdef USE_MKL
        int64_t n = self.numel();
        vsMul((int)n, self.data_ptr<float>(), other.data_ptr<float>(), self.data_ptr<float>());
        optimized = true;
        #endif
    }
    
    if (!optimized) {
        Tensor other_expanded = other.expand(out_shape);
        if (other_expanded.dtype() != self.dtype()) other_expanded = other_expanded.to(self.dtype());
        
        auto op = [](auto a, auto b) { return a * b; };
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            apply_op_recursive<ctype>(self.data_ptr<ctype>(), self.strides(), \
                                     self, self.strides(), \
                                     other_expanded, other_expanded.strides(), \
                                     0, 0, 0, 0, out_shape, op); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(TypeError, "mul_: unsupported dtype");
        }
        #undef OP_CASE
    }
    return self;
}

Tensor& div_inplace_kernel(Tensor& self, const Tensor& other) {
    std::vector<int64_t> out_shape = broadcast_shapes(static_cast<std::vector<int64_t>>(self.shape()), static_cast<std::vector<int64_t>>(other.shape()));
    if (static_cast<std::vector<int64_t>>(self.shape()) != out_shape) TP_THROW(RuntimeError, "div_: shape mismatch");

    #ifdef USE_ONEDNN
    if (OneDNNContext::is_enabled()) {
        auto self_impl = self.unsafeGetTensorImpl();
        auto other_impl = other.unsafeGetTensorImpl();
        bool self_blocked = self_impl->has_onednn_md();
        bool other_blocked = other_impl->has_onednn_md();
        
        if (self_blocked) {
             Tensor other_matching = other;
             bool match = false;
             if (other_blocked) {
                 auto md1 = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                 auto md2 = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                 if (*md1 == *md2) match = true;
             }
             
             if (!match) {
                  auto& eng = OneDNNContext::get_engine();
                  auto& s = OneDNNContext::get_stream();
                  auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
                  
                  size_t req_size = md->get_size();
                  Tensor temp = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), self.dtype(), self.device());
                  if (temp.numel() * sizeof(float) < req_size) {
                      Allocator* allocator = getAllocator(self.device().type());
                      Storage new_storage(req_size, allocator);
                      temp.unsafeGetTensorImpl()->set_storage(new_storage);
                  }
                  temp.unsafeGetTensorImpl()->set_onednn_md(self_impl->get_onednn_md());
                  
                  auto dst_mem = dnnl::memory(*md, eng, temp.data_ptr<float>());
                  
                  if (other_blocked) {
                       auto other_md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
                       auto src_mem = dnnl::memory(*other_md, eng, other.data_ptr<float>());
                       dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                  } else {
                       dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
                       auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
                       auto src_mem = dnnl::memory(nchw_md, eng, other.data_ptr<float>());
                       dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
                  }
                  s.wait();
                  other_matching = temp;
             }
             
             auto md = std::static_pointer_cast<dnnl::memory::desc>(self_impl->get_onednn_md());
             int64_t n = md->get_size() / sizeof(float);
             
             float* s_ptr = self.data_ptr<float>();
             const float* o_ptr = other_matching.data_ptr<float>();
             
             #if defined(__AVX512F__)
             #ifdef _OPENMP
             #pragma omp parallel for
             #endif
             for (int64_t i = 0; i < n; i += 16) {
                 if (i + 16 <= n) {
                     __m512 a = _mm512_loadu_ps(s_ptr + i);
                     __m512 b = _mm512_loadu_ps(o_ptr + i);
                     _mm512_storeu_ps(s_ptr + i, _mm512_div_ps(a, b));
                 } else {
                     for (int64_t j = i; j < n; ++j) s_ptr[j] /= o_ptr[j];
                 }
             }
             #elif defined(__AVX2__)
             #ifdef _OPENMP
             #pragma omp parallel for
             #endif
             for (int64_t i = 0; i < n; i += 8) {
                 if (i + 8 <= n) {
                     __m256 a = _mm256_loadu_ps(s_ptr + i);
                     __m256 b = _mm256_loadu_ps(o_ptr + i);
                     _mm256_storeu_ps(s_ptr + i, _mm256_div_ps(a, b));
                 } else {
                     for (int64_t j = i; j < n; ++j) s_ptr[j] /= o_ptr[j];
                 }
             }
             #else
             for (int64_t i = 0; i < n; ++i) s_ptr[i] /= o_ptr[i];
             #endif
             return self;
        } else if (other_blocked) {
             auto& eng = OneDNNContext::get_engine();
             auto& s = OneDNNContext::get_stream();
             auto md = std::static_pointer_cast<dnnl::memory::desc>(other_impl->get_onednn_md());
             dnnl::memory::dims dims = static_cast<std::vector<int64_t>>(other.shape());
             auto nchw_md = dnnl::memory::desc(dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
             
             Tensor other_nchw = Tensor::empty(static_cast<std::vector<int64_t>>(other.shape()), other.dtype(), other.device());
             auto src_mem = dnnl::memory(*md, eng, other.data_ptr<float>());
             auto dst_mem = dnnl::memory(nchw_md, eng, other_nchw.data_ptr<float>());
             dnnl::reorder(src_mem, dst_mem).execute(s, src_mem, dst_mem);
             s.wait();
             
             return div_inplace_kernel(self, other_nchw);
        }
    }
    #endif

    bool optimized = false;
    if (self.dtype() == DType::Float32 && other.dtype() == DType::Float32 &&
        self.is_contiguous() && other.is_contiguous() && self.shape() == other.shape()) {
        #ifdef USE_MKL
        int64_t n = self.numel();
        vsDiv((int)n, self.data_ptr<float>(), other.data_ptr<float>(), self.data_ptr<float>());
        optimized = true;
        #endif
    }
    
    if (!optimized) {
        Tensor other_expanded = other.expand(out_shape);
        if (other_expanded.dtype() != self.dtype()) other_expanded = other_expanded.to(self.dtype());
        
        auto op = [](auto a, auto b) {
            using T = std::decay_t<decltype(a)>;
            if constexpr (std::is_same_v<T, bool>) {
                return static_cast<bool>(static_cast<int>(a) / static_cast<int>(b));
            } else {
                return a / b;
            }
        };
        
        #define OP_CASE(ctype, name) \
        case DType::name: { \
            apply_op_recursive<ctype>(self.data_ptr<ctype>(), self.strides(), \
                                     self, self.strides(), \
                                     other_expanded, other_expanded.strides(), \
                                     0, 0, 0, 0, out_shape, op); \
            break; \
        }
        switch (self.dtype()) {
            TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
            default: TP_THROW(TypeError, "div_: unsupported dtype");
        }
        #undef OP_CASE
    }
    return self;
}



// --- Scalar Kernels ---

Tensor add_scalar_kernel(const Tensor& self, Scalar other, Scalar alpha) {
    DType result_dtype = self.dtype();
    if (other.isFloatingPoint() || alpha.isFloatingPoint()) {
        result_dtype = promoteTypes(result_dtype, DType::Float32);
    }
    
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), result_dtype, self.device());
    Tensor self_casted = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other, alpha](ctype a) -> ctype { \
            if constexpr (std::is_floating_point_v<ctype>) { \
                return a + alpha.to<ctype>() * other.to<ctype>(); \
            } else { \
                return static_cast<ctype>(a + alpha.toDouble() * other.toDouble()); \
            } \
        }; \
        apply_unary_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                                       self_casted, self_casted.strides(), \
                                       0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }
    
    switch (result_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "add_scalar: unsupported dtype");
    }
    #undef OP_CASE
    
    return result;
}

Tensor sub_scalar_kernel(const Tensor& self, Scalar other, Scalar alpha) {
    DType result_dtype = self.dtype();
    if (other.isFloatingPoint() || alpha.isFloatingPoint()) {
        result_dtype = promoteTypes(result_dtype, DType::Float32);
    }
    
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), result_dtype, self.device());
    Tensor self_casted = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other, alpha](ctype a) -> ctype { \
            if constexpr (std::is_floating_point_v<ctype>) { \
                return a - alpha.to<ctype>() * other.to<ctype>(); \
            } else { \
                return static_cast<ctype>(a - alpha.toDouble() * other.toDouble()); \
            } \
        }; \
        apply_unary_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                                       self_casted, self_casted.strides(), \
                                       0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }
    
    switch (result_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "sub_scalar: unsupported dtype");
    }
    #undef OP_CASE
    
    return result;
}

Tensor mul_scalar_kernel(const Tensor& self, Scalar other) {
    DType result_dtype = self.dtype();
    if (other.isFloatingPoint()) {
        result_dtype = promoteTypes(result_dtype, DType::Float32);
    }
    
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), result_dtype, self.device());
    Tensor self_casted = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other](ctype a) -> ctype { \
            if constexpr (std::is_floating_point_v<ctype>) { \
                return a * other.to<ctype>(); \
            } else { \
                return static_cast<ctype>(a * other.toDouble()); \
            } \
        }; \
        apply_unary_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                                       self_casted, self_casted.strides(), \
                                       0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }
    
    switch (result_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "mul_scalar: unsupported dtype");
    }
    #undef OP_CASE
    
    return result;
}

Tensor div_scalar_kernel(const Tensor& self, Scalar other) {
    DType result_dtype = self.dtype();
    // Div usually promotes to float
    result_dtype = promoteTypes(result_dtype, DType::Float32);
    
    Tensor result = Tensor::empty(static_cast<std::vector<int64_t>>(self.shape()), result_dtype, self.device());
    Tensor self_casted = (self.dtype() == result_dtype) ? self : self.to(result_dtype);
    
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other](ctype a) -> ctype { \
            if constexpr (std::is_same_v<ctype, bool>) { \
                return static_cast<ctype>(static_cast<double>(a) / other.to<double>()); \
            } else { \
                return static_cast<ctype>(a / other.to<double>()); \
            } \
        }; \
        apply_unary_op_recursive<ctype>(result.data_ptr<ctype>(), result.strides(), \
                                       self_casted, self_casted.strides(), \
                                       0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }
    
    switch (result_dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "div_scalar: unsupported dtype");
    }
    #undef OP_CASE
    
    return result;
}

// Inplace Scalar
Tensor& add_scalar_inplace_kernel(Tensor& self, Scalar other, Scalar alpha) {
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other, alpha](ctype a) -> ctype { \
            if constexpr (std::is_floating_point_v<ctype>) { \
                return a + alpha.to<ctype>() * other.to<ctype>(); \
            } else { \
                return static_cast<ctype>(a + alpha.toDouble() * other.toDouble()); \
            } \
        }; \
        apply_unary_op_recursive<ctype>(self.data_ptr<ctype>(), self.strides(), \
                                       self, self.strides(), \
                                       0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "add_scalar_: unsupported dtype");
    }
    #undef OP_CASE
    return self;
}

Tensor& sub_scalar_inplace_kernel(Tensor& self, Scalar other, Scalar alpha) {
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other, alpha](ctype a) -> ctype { \
            if constexpr (std::is_floating_point_v<ctype>) { \
                return a - alpha.to<ctype>() * other.to<ctype>(); \
            } else { \
                return static_cast<ctype>(a - alpha.toDouble() * other.toDouble()); \
            } \
        }; \
        apply_unary_op_recursive<ctype>(self.data_ptr<ctype>(), self.strides(), \
                                       self, self.strides(), \
                                       0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "sub_scalar_: unsupported dtype");
    }
    #undef OP_CASE
    return self;
}

Tensor& mul_scalar_inplace_kernel(Tensor& self, Scalar other) {
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other](ctype a) -> ctype { \
            if constexpr (std::is_floating_point_v<ctype>) { \
                return a * other.to<ctype>(); \
            } else { \
                return static_cast<ctype>(a * other.toDouble()); \
            } \
        }; \
        apply_unary_op_recursive<ctype>(self.data_ptr<ctype>(), self.strides(), \
                                       self, self.strides(), \
                                       0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "mul_scalar_: unsupported dtype");
    }
    #undef OP_CASE
    return self;
}

Tensor& div_scalar_inplace_kernel(Tensor& self, Scalar other) {
    #define OP_CASE(ctype, name) \
    case DType::name: { \
        auto op = [other](ctype a) -> ctype { \
            if constexpr (std::is_floating_point_v<ctype>) { \
                return a / other.to<ctype>(); \
            } else { \
                if constexpr (std::is_same_v<ctype, bool>) { \
                    return static_cast<ctype>(static_cast<double>(a) / other.to<double>()); \
                } else { \
                    return static_cast<ctype>(a / other.to<ctype>()); \
                } \
            } \
        }; \
        apply_unary_op_recursive<ctype>(self.data_ptr<ctype>(), self.strides(), \
                                       self, self.strides(), \
                                       0, 0, 0, static_cast<std::vector<int64_t>>(self.shape()), op); \
        break; \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(OP_CASE)
        default: TP_THROW(TypeError, "div_scalar_: unsupported dtype");
    }
    #undef OP_CASE
    return self;
}

// Registration
TENSORPLAY_LIBRARY_IMPL(CPU, ArithmeticKernels) {
    m.impl("add.Tensor", add_kernel);
    m.impl("sub.Tensor", sub_kernel);
    m.impl("mul.Tensor", mul_kernel);
    m.impl("div.Tensor", div_kernel);

    m.impl("add_.Tensor", add_inplace_kernel);
    m.impl("sub_.Tensor", sub_inplace_kernel);
    m.impl("mul_.Tensor", mul_inplace_kernel);
    m.impl("div_.Tensor", div_inplace_kernel);

    m.impl("add.Scalar", add_scalar_kernel);
    m.impl("sub.Scalar", sub_scalar_kernel);
    m.impl("mul.Scalar", mul_scalar_kernel);
    m.impl("div.Scalar", div_scalar_kernel);

    m.impl("add_.Scalar", add_scalar_inplace_kernel);
    m.impl("sub_.Scalar", sub_scalar_inplace_kernel);
    m.impl("mul_.Scalar", mul_scalar_inplace_kernel);
    m.impl("div_.Scalar", div_scalar_inplace_kernel);
}

} // namespace cpu
} // namespace tensorplay
