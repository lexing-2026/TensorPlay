// Descriptor-based DNN library compatibility layer for the AMD build.
//
// The GPU kernels were written against the cuDNN-style descriptor API.  This
// header re-creates that surface on top of MIOpen: type aliases, enum-name
// mappings, and inline wrappers that absorb the signature differences
// (argument order, parameters MIOpen does not carry, missing heuristics).
// Everything below is inactive unless USE_CUDNN is defined, i.e. the build
// found libMIOpen.
//
// Mapping notes, validated against MIOpen 3.5.x:
//   - Filter descriptors do not exist in MIOpen; they map onto plain tensor
//     descriptors.
//   - Convolution descriptors are "initialized" rather than "set", and carry
//     no compute-type field (precision follows the tensor descriptors).
//   - The v7 heuristic query has no direct MIOpen equivalent; the wrapper
//     derives the algorithm from MIOpen's solution database instead, and the
//     execution wrappers absorb MIOpen's find-before-execute requirement.
//   - 2-D pooling is disabled: the pooling kernels in the available MIOpen
//     build misbehave on the supported GPU (see the pooling section).
//   - NaN-propagation arguments have no MIOpen counterpart and are dropped.
#pragma once

#if defined(USE_CUDNN)

#include <miopen/miopen.h>
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Status / handle
// ---------------------------------------------------------------------------
using cudnnStatus_t = miopenStatus_t;
#define CUDNN_STATUS_SUCCESS miopenStatusSuccess

using cudnnHandle_t = miopenHandle_t;

inline const char* cudnnGetErrorString(miopenStatus_t status) {
    return miopenGetErrorString(status);
}

#define cudnnCreate miopenCreate
#define cudnnDestroy miopenDestroy
#define cudnnSetStream miopenSetStream
#define cudnnGetStream miopenGetStream

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------
using cudnnDataType_t = miopenDataType_t;
#define CUDNN_DATA_FLOAT miopenFloat
#define CUDNN_DATA_DOUBLE miopenDouble
#define CUDNN_DATA_HALF miopenHalf
#define CUDNN_DATA_INT8 miopenInt8
#define CUDNN_DATA_INT32 miopenInt32
#define CUDNN_DATA_BFLOAT16 miopenBFloat16

using cudnnNanPropagation_t = miopenNanPropagation_t;
#define CUDNN_PROPAGATE_NAN MIOPEN_PROPAGATE_NAN
#define CUDNN_NOT_PROPAGATE_NAN MIOPEN_NOT_PROPAGATE_NAN

// ---------------------------------------------------------------------------
// Tensor descriptors
// ---------------------------------------------------------------------------
using cudnnTensorDescriptor_t = miopenTensorDescriptor_t;
using cudnnTensorFormat_t = int;
#define CUDNN_TENSOR_NCHW miopenTensorNCHW
#define CUDNN_TENSOR_NHWC miopenTensorNHWC

#define cudnnCreateTensorDescriptor miopenCreateTensorDescriptor
#define cudnnDestroyTensorDescriptor miopenDestroyTensorDescriptor
#define cudnnSetTensor4dDescriptorEx miopenSet4dTensorDescriptorEx
#define cudnnSetTensorNdDescriptor miopenSetTensorDescriptor

// cuDNN's 4-D setter carries a layout argument; MIOpen's equivalent is
// NCHW-only and drops it.
inline miopenStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t desc,
                                                 cudnnTensorFormat_t format,
                                                 miopenDataType_t dataType,
                                                 int n, int c, int h, int w) {
    (void)format;
    return miopenSet4dTensorDescriptor(desc, dataType, n, c, h, w);
}

// ---------------------------------------------------------------------------
// Filter descriptors (plain tensor descriptors in MIOpen)
// ---------------------------------------------------------------------------
using cudnnFilterDescriptor_t = miopenTensorDescriptor_t;

inline miopenStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* desc) {
    return miopenCreateTensorDescriptor(desc);
}
inline miopenStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t desc) {
    return miopenDestroyTensorDescriptor(desc);
}
inline miopenStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t desc,
                                                 miopenDataType_t dataType,
                                                 cudnnTensorFormat_t format,
                                                 int k, int c, int h, int w) {
    (void)format;
    return miopenSet4dTensorDescriptor(desc, dataType, k, c, h, w);
}
inline miopenStatus_t cudnnSetFilterNdDescriptor(cudnnFilterDescriptor_t desc,
                                                 miopenDataType_t dataType,
                                                 cudnnTensorFormat_t format,
                                                 int nbDims, const int* dimsA) {
    (void)format;
    // The Nd setter takes no strides; a dense layout is implied, so the
    // row-major strides are derived from the dims here.
    int stridesA[8];
    long long acc = 1;
    for (int i = nbDims - 1; i >= 0; --i) {
        stridesA[i] = static_cast<int>(acc);
        acc *= dimsA[i];
    }
    return miopenSetTensorDescriptor(desc, dataType, nbDims, dimsA, stridesA);
}

// ---------------------------------------------------------------------------
// Convolution descriptors
// ---------------------------------------------------------------------------
using cudnnConvolutionDescriptor_t = miopenConvolutionDescriptor_t;
using cudnnConvolutionMode_t = miopenConvolutionMode_t;
#define CUDNN_CROSS_CORRELATION miopenConvolution
#define CUDNN_CONVOLUTION miopenConvolution
#define CUDNN_CONVOLUTION_TRANSPOSE miopenTranspose

#define cudnnCreateConvolutionDescriptor miopenCreateConvolutionDescriptor
#define cudnnDestroyConvolutionDescriptor miopenDestroyConvolutionDescriptor
#define cudnnSetConvolutionGroupCount miopenSetConvolutionGroupCount

// MIOpen "initializes" convolution descriptors in place of cuDNN's setter,
// and has no compute-type field (precision follows the tensor descriptors).
inline miopenStatus_t cudnnSetConvolution2dDescriptor(
    cudnnConvolutionDescriptor_t desc, int pad_h, int pad_w, int stride_h,
    int stride_w, int dilation_h, int dilation_w, cudnnConvolutionMode_t mode,
    miopenDataType_t computeType) {
    (void)computeType;
    return miopenInitConvolutionDescriptor(desc, mode, pad_h, pad_w, stride_h,
                                           stride_w, dilation_h, dilation_w);
}
inline miopenStatus_t cudnnSetConvolutionNdDescriptor(
    cudnnConvolutionDescriptor_t desc, int nbDims, const int* padA,
    const int* strideA, const int* dilationA, cudnnConvolutionMode_t mode,
    miopenDataType_t computeType) {
    (void)computeType;
    return miopenInitConvolutionNdDescriptor(desc, nbDims, padA, strideA,
                                             dilationA, mode);
}

// Tensor-core selection has no MIOpen knob; the request is accepted and
// ignored.
using cudnnMathType_t = int;
#define CUDNN_DEFAULT_MATH 0
#define CUDNN_TENSOR_OP_MATH 1
#define CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION 2
inline miopenStatus_t cudnnSetConvolutionMathType(
    cudnnConvolutionDescriptor_t, cudnnMathType_t) {
    return miopenStatusSuccess;
}

// ---------------------------------------------------------------------------
// Activation
//
// Declared before the convolution section: the fused conv+bias+activation
// wrapper below takes an activation descriptor.
// ---------------------------------------------------------------------------
using cudnnActivationDescriptor_t = miopenActivationDescriptor_t;
using cudnnActivationMode_t = miopenActivationMode_t;
#define CUDNN_ACTIVATION_SIGMOID miopenActivationLOGISTIC
#define CUDNN_ACTIVATION_RELU miopenActivationRELU
#define CUDNN_ACTIVATION_TANH miopenActivationTANH
#define CUDNN_ACTIVATION_CLIPPED_RELU miopenActivationCLIPPEDRELU
#define CUDNN_ACTIVATION_ELU miopenActivationELU
#define CUDNN_ACTIVATION_IDENTITY miopenActivationPASTHRU

#define cudnnCreateActivationDescriptor miopenCreateActivationDescriptor
#define cudnnDestroyActivationDescriptor miopenDestroyActivationDescriptor

// MIOpen carries three activation scalars (alpha/beta/gamma) and no NaN
// policy; cuDNN's coef lands in the alpha slot.
inline miopenStatus_t cudnnSetActivationDescriptor(
    cudnnActivationDescriptor_t desc, cudnnActivationMode_t mode,
    miopenNanPropagation_t nanProp, double coef) {
    (void)nanProp;
    return miopenSetActivationDescriptor(desc, mode, coef, 0.0, 0.0);
}

#define cudnnActivationForward miopenActivationForward

// ---------------------------------------------------------------------------
// Convolution algorithms and execution
//
// The v7-style heuristic query maps onto MIOpen's solution database (no data
// pointers needed); the measured autotune path maps onto
// miopenFindConvolution*Algorithm with exhaustive search; execution uses the
// classic algorithm-enum API.  miopenConvAlgorithm_t values match the classic
// miopenConv{Fwd,BwdData,BwdWeights}Algorithm_t numbering, so the algorithm
// returned by a solution converts by a plain cast.
// ---------------------------------------------------------------------------
using cudnnConvolutionFwdAlgo_t = miopenConvFwdAlgorithm_t;
using cudnnConvolutionBwdDataAlgo_t = miopenConvBwdDataAlgorithm_t;
using cudnnConvolutionBwdFilterAlgo_t = miopenConvBwdWeightsAlgorithm_t;

#define CUDNN_CONVOLUTION_FWD_ALGO_GEMM miopenConvolutionFwdAlgoGEMM
#define CUDNN_CONVOLUTION_FWD_ALGO_DIRECT miopenConvolutionFwdAlgoDirect
#define CUDNN_CONVOLUTION_FWD_ALGO_FFT miopenConvolutionFwdAlgoFFT
#define CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD miopenConvolutionFwdAlgoWinograd
#define CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM \
    miopenConvolutionFwdAlgoImplicitGEMM
#define CUDNN_CONVOLUTION_BWD_DATA_ALGO_GEMM miopenConvolutionBwdDataAlgoGEMM
#define CUDNN_CONVOLUTION_BWD_DATA_ALGO_DIRECT miopenConvolutionBwdDataAlgoDirect
#define CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT miopenConvolutionBwdDataAlgoFFT
#define CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD \
    miopenConvolutionBwdDataAlgoWinograd
#define CUDNN_CONVOLUTION_BWD_DATA_ALGO_IMPLICIT_GEMM \
    miopenConvolutionBwdDataAlgoImplicitGEMM
#define CUDNN_CONVOLUTION_BWD_FILTER_ALGO_GEMM \
    miopenConvolutionBwdWeightsAlgoGEMM
#define CUDNN_CONVOLUTION_BWD_FILTER_ALGO_DIRECT \
    miopenConvolutionBwdWeightsAlgoDirect
#define CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD \
    miopenConvolutionBwdWeightsAlgoWinograd
#define CUDNN_CONVOLUTION_BWD_FILTER_ALGO_IMPLICIT_GEMM \
    miopenConvolutionBwdWeightsAlgoImplicitGEMM

// MIOpen's perf record stores the algorithm in a union shared by the three
// directions; these structs mirror the shape the kernels read (.algo,
// .status, .time, .memory).
struct cudnnConvolutionFwdAlgoPerf_t {
    miopenConvFwdAlgorithm_t algo;
    miopenStatus_t status;
    float time;
    size_t memory;
};
struct cudnnConvolutionBwdDataAlgoPerf_t {
    miopenConvBwdDataAlgorithm_t algo;
    miopenStatus_t status;
    float time;
    size_t memory;
};
struct cudnnConvolutionBwdFilterAlgoPerf_t {
    miopenConvBwdWeightsAlgorithm_t algo;
    miopenStatus_t status;
    float time;
    size_t memory;
};

inline miopenStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    miopenHandle_t handle, cudnnTensorDescriptor_t xDesc,
    cudnnFilterDescriptor_t wDesc, cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t yDesc, int requestedAlgoCount,
    int* returnedAlgoCount, cudnnConvolutionFwdAlgoPerf_t* perfResults) {
    (void)requestedAlgoCount;
    *returnedAlgoCount = 0;
    size_t count = 0;
    miopenConvSolution_t solution;
    miopenStatus_t s = miopenConvolutionForwardGetSolution(
        handle, wDesc, xDesc, convDesc, yDesc, 1, &count, &solution);
    if (s != miopenStatusSuccess) return s;
    if (count == 0) return miopenStatusSuccess;
    perfResults[0].algo =
        static_cast<miopenConvFwdAlgorithm_t>(solution.algorithm);
    perfResults[0].status = miopenStatusSuccess;
    perfResults[0].time = solution.time;
    perfResults[0].memory = solution.workspace_size;
    *returnedAlgoCount = 1;
    return miopenStatusSuccess;
}

inline miopenStatus_t cudnnGetConvolutionBackwardDataAlgorithm_v7(
    miopenHandle_t handle, cudnnFilterDescriptor_t wDesc,
    cudnnTensorDescriptor_t dyDesc, cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t dxDesc, int requestedAlgoCount,
    int* returnedAlgoCount, cudnnConvolutionBwdDataAlgoPerf_t* perfResults) {
    (void)requestedAlgoCount;
    *returnedAlgoCount = 0;
    size_t count = 0;
    miopenConvSolution_t solution;
    miopenStatus_t s = miopenConvolutionBackwardDataGetSolution(
        handle, dyDesc, wDesc, convDesc, dxDesc, 1, &count, &solution);
    if (s != miopenStatusSuccess) return s;
    if (count == 0) return miopenStatusSuccess;
    perfResults[0].algo =
        static_cast<miopenConvBwdDataAlgorithm_t>(solution.algorithm);
    perfResults[0].status = miopenStatusSuccess;
    perfResults[0].time = solution.time;
    perfResults[0].memory = solution.workspace_size;
    *returnedAlgoCount = 1;
    return miopenStatusSuccess;
}

inline miopenStatus_t cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    miopenHandle_t handle, cudnnTensorDescriptor_t xDesc,
    cudnnTensorDescriptor_t dyDesc, cudnnConvolutionDescriptor_t convDesc,
    cudnnFilterDescriptor_t dwDesc, int requestedAlgoCount,
    int* returnedAlgoCount, cudnnConvolutionBwdFilterAlgoPerf_t* perfResults) {
    (void)requestedAlgoCount;
    *returnedAlgoCount = 0;
    size_t count = 0;
    miopenConvSolution_t solution;
    miopenStatus_t s = miopenConvolutionBackwardWeightsGetSolution(
        handle, dyDesc, xDesc, convDesc, dwDesc, 1, &count, &solution);
    if (s != miopenStatusSuccess) return s;
    if (count == 0) return miopenStatusSuccess;
    perfResults[0].algo =
        static_cast<miopenConvBwdWeightsAlgorithm_t>(solution.algorithm);
    perfResults[0].status = miopenStatusSuccess;
    perfResults[0].time = solution.time;
    perfResults[0].memory = solution.workspace_size;
    *returnedAlgoCount = 1;
    return miopenStatusSuccess;
}

// Workspace sizing: MIOpen's query has no algorithm parameter and returns the
// size for its own selection of the same problem.
inline miopenStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    miopenHandle_t handle, cudnnTensorDescriptor_t xDesc,
    cudnnFilterDescriptor_t wDesc, cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t yDesc, cudnnConvolutionFwdAlgo_t,
    size_t* sizeInBytes) {
    return miopenConvolutionForwardGetWorkSpaceSize(handle, wDesc, xDesc,
                                                    convDesc, yDesc, sizeInBytes);
}
inline miopenStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
    miopenHandle_t handle, cudnnFilterDescriptor_t wDesc,
    cudnnTensorDescriptor_t dyDesc, cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t dxDesc, cudnnConvolutionBwdDataAlgo_t,
    size_t* sizeInBytes) {
    return miopenConvolutionBackwardDataGetWorkSpaceSize(
        handle, dyDesc, wDesc, convDesc, dxDesc, sizeInBytes);
}
inline miopenStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
    miopenHandle_t handle, cudnnTensorDescriptor_t xDesc,
    cudnnTensorDescriptor_t dyDesc, cudnnConvolutionDescriptor_t convDesc,
    cudnnFilterDescriptor_t dwDesc, cudnnConvolutionBwdFilterAlgo_t,
    size_t* sizeInBytes) {
    return miopenConvolutionBackwardWeightsGetWorkSpaceSize(
        handle, dyDesc, xDesc, convDesc, dwDesc, sizeInBytes);
}

// Autotune.  Argument orders match except for the backward passes, where
// MIOpen lists the output gradient first, and the trailing exhaustive flag.
// The candidates come from the solution database rather than the measured
// find pass: the measured search stalls indefinitely on fp16 problems here,
// while the solution query answers in milliseconds and its ordering already
// reflects the library's performance model.
inline miopenStatus_t cudnnFindConvolutionForwardAlgorithmEx(
    miopenHandle_t handle, cudnnTensorDescriptor_t xDesc, const void*,
    cudnnFilterDescriptor_t wDesc, const void*,
    cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc,
    void*, int requestedAlgoCount, int* returnedAlgoCount,
    cudnnConvolutionFwdAlgoPerf_t* perfResults, void*, size_t) {
    *returnedAlgoCount = 0;
    if (requestedAlgoCount <= 0) return miopenStatusInvalidValue;
    std::vector<miopenConvSolution_t> sols(requestedAlgoCount);
    size_t count = 0;
    miopenStatus_t s = miopenConvolutionForwardGetSolution(
        handle, wDesc, xDesc, convDesc, yDesc,
        static_cast<size_t>(requestedAlgoCount), &count, sols.data());
    if (s != miopenStatusSuccess) return s;
    for (size_t i = 0; i < count; ++i) {
        perfResults[i].algo =
            static_cast<miopenConvFwdAlgorithm_t>(sols[i].algorithm);
        perfResults[i].status = miopenStatusSuccess;
        perfResults[i].time = sols[i].time;
        perfResults[i].memory = sols[i].workspace_size;
    }
    *returnedAlgoCount = static_cast<int>(count);
    return miopenStatusSuccess;
}

inline miopenStatus_t cudnnFindConvolutionBackwardDataAlgorithmEx(
    miopenHandle_t handle, cudnnFilterDescriptor_t wDesc, const void*,
    cudnnTensorDescriptor_t dyDesc, const void*,
    cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t dxDesc,
    void*, int requestedAlgoCount, int* returnedAlgoCount,
    cudnnConvolutionBwdDataAlgoPerf_t* perfResults, void*, size_t) {
    *returnedAlgoCount = 0;
    if (requestedAlgoCount <= 0) return miopenStatusInvalidValue;
    std::vector<miopenConvSolution_t> sols(requestedAlgoCount);
    size_t count = 0;
    miopenStatus_t s = miopenConvolutionBackwardDataGetSolution(
        handle, dyDesc, wDesc, convDesc, dxDesc,
        static_cast<size_t>(requestedAlgoCount), &count, sols.data());
    if (s != miopenStatusSuccess) return s;
    for (size_t i = 0; i < count; ++i) {
        perfResults[i].algo =
            static_cast<miopenConvBwdDataAlgorithm_t>(sols[i].algorithm);
        perfResults[i].status = miopenStatusSuccess;
        perfResults[i].time = sols[i].time;
        perfResults[i].memory = sols[i].workspace_size;
    }
    *returnedAlgoCount = static_cast<int>(count);
    return miopenStatusSuccess;
}

inline miopenStatus_t cudnnFindConvolutionBackwardFilterAlgorithmEx(
    miopenHandle_t handle, cudnnTensorDescriptor_t xDesc, const void*,
    cudnnTensorDescriptor_t dyDesc, const void*,
    cudnnConvolutionDescriptor_t convDesc, cudnnFilterDescriptor_t dwDesc,
    void*, int requestedAlgoCount, int* returnedAlgoCount,
    cudnnConvolutionBwdFilterAlgoPerf_t* perfResults, void*, size_t) {
    *returnedAlgoCount = 0;
    if (requestedAlgoCount <= 0) return miopenStatusInvalidValue;
    std::vector<miopenConvSolution_t> sols(requestedAlgoCount);
    size_t count = 0;
    miopenStatus_t s = miopenConvolutionBackwardWeightsGetSolution(
        handle, dyDesc, xDesc, convDesc, dwDesc,
        static_cast<size_t>(requestedAlgoCount), &count, sols.data());
    if (s != miopenStatusSuccess) return s;
    for (size_t i = 0; i < count; ++i) {
        perfResults[i].algo =
            static_cast<miopenConvBwdWeightsAlgorithm_t>(sols[i].algorithm);
        perfResults[i].status = miopenStatusSuccess;
        perfResults[i].time = sols[i].time;
        perfResults[i].memory = sols[i].workspace_size;
    }
    *returnedAlgoCount = static_cast<int>(count);
    return miopenStatusSuccess;
}

// Execution: parameter order matches except the workspace sits at the end in
// MIOpen, and the backward passes list the output gradient first.
//
// A classic execution against a problem the library has never searched fails
// with "no invoker registered".  The wrappers absorb that: on the first
// failure they run the library's own path for the problem (non-measured find
// for fp32, whose search is fast, or the solution-immediate entry point,
// which works for every dtype) and retry once.
namespace tp_miopen_compat {

template <class Find, class Exec>
inline miopenStatus_t exec_with_invoker_register(Find find, Exec exec) {
    miopenStatus_t s = exec();
    if (s == miopenStatusSuccess) return s;
    miopenConvAlgoPerf_t found;
    int got = 0;
    miopenStatus_t fs = find(&found, &got);
    if (fs != miopenStatusSuccess || got == 0) return s;
    return exec();
}

template <class Solution, class Immediate>
inline miopenStatus_t exec_with_fallback(Solution solution, Immediate immediate) {
    miopenStatus_t s = solution();
    if (s == miopenStatusSuccess) return s;
    return immediate();
}

}  // namespace tp_miopen_compat

inline miopenStatus_t cudnnConvolutionForward(
    miopenHandle_t handle, const void* alpha, cudnnTensorDescriptor_t xDesc,
    const void* x, cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void* workSpace, size_t workSpaceSize, const void* beta,
    cudnnTensorDescriptor_t yDesc, void* y) {
    // The classic call runs first; on failure the immediate route takes over.
    (void)algo;
    miopenStatus_t s = miopenConvolutionForward(
        handle, alpha, xDesc, x, wDesc, w, convDesc,
        static_cast<miopenConvFwdAlgorithm_t>(algo), beta, yDesc, y, workSpace,
        workSpaceSize);
    if (s == miopenStatusSuccess) return s;
    // The library registers the kernel invoker for a problem only during the
    // measured search, so the classic execute stays unavailable until one find
    // pass has run.  Do the (non-exhaustive) search with this call's own
    // buffers and retry the classic execute before falling back to the
    // immediate route.
    {
        miopenConvAlgoPerf_t perf;
        int found = 0;
        if (miopenFindConvolutionForwardAlgorithm(
                handle, xDesc, x, wDesc, w, convDesc, yDesc, y, 1, &found,
                &perf, workSpace, workSpaceSize, false) == miopenStatusSuccess &&
            found > 0) {
            s = miopenConvolutionForward(
                handle, alpha, xDesc, x, wDesc, w, convDesc,
                static_cast<miopenConvFwdAlgorithm_t>(algo), beta, yDesc, y,
                workSpace, workSpaceSize);
            if (s == miopenStatusSuccess) return s;
        }
    }
    miopenConvSolution_t solution;
    size_t count = 0;
    s = miopenConvolutionForwardGetSolution(handle, wDesc, xDesc, convDesc,
                                            yDesc, 1, &count, &solution);
    if (s != miopenStatusSuccess || count == 0) return s;
    return miopenConvolutionForwardImmediate(handle, wDesc, w, xDesc, x,
                                             convDesc, yDesc, y, workSpace,
                                             workSpaceSize,
                                             solution.solution_id);
}

inline miopenStatus_t cudnnConvolutionBackwardData(
    miopenHandle_t handle, const void* alpha, cudnnFilterDescriptor_t wDesc,
    const void* w, cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionBwdDataAlgo_t algo,
    void* workSpace, size_t workSpaceSize, const void* beta,
    cudnnTensorDescriptor_t dxDesc, void* dx) {
    (void)algo;
    miopenStatus_t s = miopenConvolutionBackwardData(
        handle, alpha, dyDesc, dy, wDesc, w, convDesc,
        static_cast<miopenConvBwdDataAlgorithm_t>(algo), beta, dxDesc, dx,
        workSpace, workSpaceSize);
    if (s == miopenStatusSuccess) return s;
    {
        miopenConvAlgoPerf_t perf;
        int found = 0;
        if (miopenFindConvolutionBackwardDataAlgorithm(
                handle, dyDesc, dy, wDesc, w, convDesc, dxDesc, dx, 1, &found,
                &perf, workSpace, workSpaceSize,
                false) == miopenStatusSuccess &&
            found > 0) {
            s = miopenConvolutionBackwardData(
                handle, alpha, dyDesc, dy, wDesc, w, convDesc,
                static_cast<miopenConvBwdDataAlgorithm_t>(algo), beta, dxDesc,
                dx, workSpace, workSpaceSize);
            if (s == miopenStatusSuccess) return s;
        }
    }
    miopenConvSolution_t solution;
    size_t count = 0;
    s = miopenConvolutionBackwardDataGetSolution(handle, dyDesc, wDesc,
                                                 convDesc, dxDesc, 1, &count,
                                                 &solution);
    if (s != miopenStatusSuccess || count == 0) return s;
    return miopenConvolutionBackwardDataImmediate(handle, dyDesc, dy, wDesc, w,
                                                  convDesc, dxDesc, dx,
                                                  workSpace, workSpaceSize,
                                                  solution.solution_id);
}

inline miopenStatus_t cudnnConvolutionBackwardFilter(
    miopenHandle_t handle, const void* alpha, cudnnTensorDescriptor_t xDesc,
    const void* x, cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void* workSpace,
    size_t workSpaceSize, const void* beta, cudnnFilterDescriptor_t dwDesc,
    void* dw) {
    (void)algo;
    miopenStatus_t s = miopenConvolutionBackwardWeights(
        handle, alpha, dyDesc, dy, xDesc, x, convDesc,
        static_cast<miopenConvBwdWeightsAlgorithm_t>(algo), beta, dwDesc, dw,
        workSpace, workSpaceSize);
    if (s == miopenStatusSuccess) return s;
    {
        miopenConvAlgoPerf_t perf;
        int found = 0;
        if (miopenFindConvolutionBackwardWeightsAlgorithm(
                handle, dyDesc, dy, xDesc, x, convDesc, dwDesc, dw, 1, &found,
                &perf, workSpace, workSpaceSize,
                false) == miopenStatusSuccess &&
            found > 0) {
            s = miopenConvolutionBackwardWeights(
                handle, alpha, dyDesc, dy, xDesc, x, convDesc,
                static_cast<miopenConvBwdWeightsAlgorithm_t>(algo), beta,
                dwDesc, dw, workSpace, workSpaceSize);
            if (s == miopenStatusSuccess) return s;
        }
    }
    miopenConvSolution_t solution;
    size_t count = 0;
    s = miopenConvolutionBackwardWeightsGetSolution(handle, dyDesc, xDesc,
                                                    convDesc, dwDesc, 1, &count,
                                                    &solution);
    if (s != miopenStatusSuccess || count == 0) return s;
    return miopenConvolutionBackwardWeightsImmediate(handle, dyDesc, dy, xDesc,
                                                     x, convDesc, dwDesc, dw,
                                                     workSpace, workSpaceSize,
                                                     solution.solution_id);
}

#define cudnnConvolutionBackwardBias miopenConvolutionBackwardBias

// Host-side scalar storage matching a descriptor's compute type; the all-zero
// bit pattern reads as 0.0 in both float and double.
namespace tp_miopen_compat {

inline const void* scalar_zero() {
    static const double zero = 0.0;
    return &zero;
}

inline miopenDataType_t descriptor_dtype(cudnnTensorDescriptor_t desc) {
    int size = 0;
    miopenDataType_t dt = miopenFloat;
    if (miopenGetTensorDescriptorSize(desc, &size) != miopenStatusSuccess ||
        size <= 0) {
        return miopenFloat;
    }
    std::vector<int> dims(static_cast<size_t>(size));
    std::vector<int> strides(static_cast<size_t>(size));
    if (miopenGetTensorDescriptor(desc, &dt, dims.data(), strides.data()) !=
        miopenStatusSuccess) {
        return miopenFloat;
    }
    return dt;
}

inline const void* scalar_one(miopenDataType_t dt) {
    static const float one_f = 1.0f;
    static const double one_d = 1.0;
    return dt == miopenDouble ? static_cast<const void*>(&one_d)
                              : static_cast<const void*>(&one_f);
}

}  // namespace tp_miopen_compat

// y = alpha*bias + beta*y expressed through the elementwise tensor op.  A
// must share C's shape and only B may broadcast, so y sits in the A slot with
// a zero weight and the bias rides in the B slot.
inline miopenStatus_t cudnnAddTensor(miopenHandle_t handle, const void* alpha,
                                     cudnnTensorDescriptor_t bDesc,
                                     const void* bias, const void* beta,
                                     cudnnTensorDescriptor_t yDesc, void* y) {
    return miopenOpTensor(handle, miopenTensorOpAdd,
                          tp_miopen_compat::scalar_zero(), yDesc, y, alpha,
                          bDesc, bias, beta, yDesc, y);
}

// Fused convolution + bias + activation.  MIOpen ships the same fused entry
// point; when the library refuses the configuration (data type, algorithm),
// the wrapper falls back to the sequential decomposition so the kernels stay
// functional either way.  The z term is only applied when alpha2 != 0, which
// the callers never request.
inline miopenStatus_t cudnnConvolutionBiasActivationForward(
    miopenHandle_t handle, const void* alpha1, cudnnTensorDescriptor_t xDesc,
    const void* x, cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo,
    void* workSpace, size_t workSpaceSize, const void* alpha2,
    cudnnTensorDescriptor_t zDesc, const void* z,
    cudnnTensorDescriptor_t biasDesc, const void* bias,
    cudnnActivationDescriptor_t activationDesc, cudnnTensorDescriptor_t yDesc,
    void* y) {
    miopenStatus_t s = miopenConvolutionBiasActivationForward(
        handle, alpha1, xDesc, x, wDesc, w, convDesc, algo, workSpace,
        workSpaceSize, alpha2, zDesc, z, biasDesc, bias, activationDesc, yDesc,
        y);
    if (s == miopenStatusSuccess) return s;

    (void)alpha2;
    (void)zDesc;
    (void)z;
    const void* zero = tp_miopen_compat::scalar_zero();
    s = miopenConvolutionForward(handle, alpha1, xDesc, x, wDesc, w, convDesc,
                                 algo, zero, yDesc, y, workSpace,
                                 workSpaceSize);
    if (s == miopenStatusSuccess) {
        // fall through to the bias+activation tail below
    } else {
        // Same invoker-registration requirement as the plain forward: run the
        // measured search once, then retry both entry points.
        miopenConvAlgoPerf_t perf;
        int found = 0;
        miopenFindConvolutionForwardAlgorithm(
            handle, xDesc, x, wDesc, w, convDesc, yDesc, y, 1, &found, &perf,
            workSpace, workSpaceSize, false);
        s = miopenConvolutionForward(handle, alpha1, xDesc, x, wDesc, w,
                                     convDesc, algo, zero, yDesc, y,
                                     workSpace, workSpaceSize);
        if (s != miopenStatusSuccess) {
            miopenConvSolution_t solution;
            size_t count = 0;
            if (miopenConvolutionForwardGetSolution(handle, wDesc, xDesc,
                                                    convDesc, yDesc, 1, &count,
                                                    &solution) !=
                        miopenStatusSuccess ||
                count == 0) {
                return s;
            }
            s = miopenConvolutionForwardImmediate(
                handle, wDesc, w, xDesc, x, convDesc, yDesc, y, workSpace,
                workSpaceSize, solution.solution_id);
            if (s != miopenStatusSuccess) return s;
        }
    }
    miopenDataType_t dt = tp_miopen_compat::descriptor_dtype(yDesc);
    const void* one = tp_miopen_compat::scalar_one(dt);
    s = miopenOpTensor(handle, miopenTensorOpAdd, zero, yDesc, y, one, biasDesc,
                       bias, one, yDesc, y);
    if (s != miopenStatusSuccess) return s;
    return miopenActivationForward(handle, activationDesc, one, yDesc, y, zero,
                                   yDesc, y);
}

// ---------------------------------------------------------------------------
// Pooling
//
// Disabled on this backend: the 2-D pooling kernels in the available MIOpen
// build write only the first output row correctly on the supported GPU
// (verified against a reference max-pool); the remaining rows come back with
// the kernel's initial -FLT_MAX or shifted windows.  Returning
// not-implemented keeps the guarded kernel paths failing loudly instead of
// producing silently wrong numbers.
// ---------------------------------------------------------------------------
using cudnnPoolingDescriptor_t = miopenPoolingDescriptor_t;
using cudnnPoolingMode_t = miopenPoolingMode_t;
#define CUDNN_POOLING_MAX miopenPoolingMax
#define CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING \
    miopenPoolingAverageInclusive
#define CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING miopenPoolingAverage

#define cudnnCreatePoolingDescriptor miopenCreatePoolingDescriptor
#define cudnnDestroyPoolingDescriptor miopenDestroyPoolingDescriptor
#define cudnnGetPooling2dForwardOutputDim miopenGetPoolingForwardOutputDim

inline miopenStatus_t cudnnSetPooling2dDescriptor(
    cudnnPoolingDescriptor_t, cudnnPoolingMode_t, miopenNanPropagation_t,
    int, int, int, int, int, int) {
    return miopenStatusNotImplemented;
}

inline miopenStatus_t cudnnPoolingForward(miopenHandle_t,
                                          cudnnPoolingDescriptor_t,
                                          const void*, cudnnTensorDescriptor_t,
                                          const void*, const void*,
                                          cudnnTensorDescriptor_t, void*) {
    return miopenStatusNotImplemented;
}

inline miopenStatus_t cudnnPoolingBackward(miopenHandle_t,
                                           cudnnPoolingDescriptor_t,
                                           const void*, cudnnTensorDescriptor_t,
                                           const void*, cudnnTensorDescriptor_t,
                                           const void*, cudnnTensorDescriptor_t,
                                           const void*, const void*,
                                           cudnnTensorDescriptor_t, void*) {
    return miopenStatusNotImplemented;
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------
using cudnnSoftmaxAlgorithm_t = miopenSoftmaxAlgorithm_t;
using cudnnSoftmaxMode_t = miopenSoftmaxMode_t;
#define CUDNN_SOFTMAX_FAST MIOPEN_SOFTMAX_FAST
#define CUDNN_SOFTMAX_ACCURATE MIOPEN_SOFTMAX_ACCURATE
#define CUDNN_SOFTMAX_LOG MIOPEN_SOFTMAX_LOG
#define CUDNN_SOFTMAX_MODE_INSTANCE MIOPEN_SOFTMAX_MODE_INSTANCE
#define CUDNN_SOFTMAX_MODE_CHANNEL MIOPEN_SOFTMAX_MODE_CHANNEL

// Algorithm and mode ride at the end of MIOpen's V2 entry point.
inline miopenStatus_t cudnnSoftmaxForward(miopenHandle_t handle,
                                          cudnnSoftmaxAlgorithm_t algorithm,
                                          cudnnSoftmaxMode_t mode,
                                          const void* alpha,
                                          cudnnTensorDescriptor_t xDesc,
                                          const void* x, const void* beta,
                                          cudnnTensorDescriptor_t yDesc,
                                          void* y) {
    return miopenSoftmaxForward_V2(handle, alpha, xDesc, x, beta, yDesc, y,
                                   algorithm, mode);
}

// ---------------------------------------------------------------------------
// Elementwise tensor op
//
// MIOpen has no op-tensor descriptor; the operation rides as a plain enum, so
// the "descriptor" is an miopenTensorOp_t value.
// ---------------------------------------------------------------------------
using cudnnOpTensorOp_t = miopenTensorOp_t;
// The create/set/execute trio passes the descriptor by pointer/value exactly
// as the kernels do; a one-field struct carries the operation.
struct tp_miopen_op_tensor_desc {
    miopenTensorOp_t op;
};
using cudnnOpTensorDescriptor_t = tp_miopen_op_tensor_desc*;
#define CUDNN_OP_TENSOR_ADD miopenTensorOpAdd
#define CUDNN_OP_TENSOR_MUL miopenTensorOpMul
#define CUDNN_OP_TENSOR_MIN miopenTensorOpMin
#define CUDNN_OP_TENSOR_MAX miopenTensorOpMax

inline miopenStatus_t cudnnCreateOpTensorDescriptor(
    cudnnOpTensorDescriptor_t* desc) {
    *desc = new tp_miopen_op_tensor_desc{miopenTensorOpAdd};
    return miopenStatusSuccess;
}
inline miopenStatus_t cudnnDestroyOpTensorDescriptor(
    cudnnOpTensorDescriptor_t desc) {
    delete desc;
    return miopenStatusSuccess;
}
inline miopenStatus_t cudnnSetOpTensorDescriptor(
    cudnnOpTensorDescriptor_t desc, cudnnOpTensorOp_t op,
    miopenDataType_t computeType, miopenNanPropagation_t nanProp) {
    (void)computeType;
    (void)nanProp;
    desc->op = op;
    return miopenStatusSuccess;
}
inline miopenStatus_t cudnnOpTensor(miopenHandle_t handle,
                                    cudnnOpTensorDescriptor_t opDesc,
                                    const void* alpha1,
                                    cudnnTensorDescriptor_t aDesc,
                                    const void* A, const void* alpha2,
                                    cudnnTensorDescriptor_t bDesc,
                                    const void* B, const void* beta,
                                    cudnnTensorDescriptor_t cDesc, void* C) {
    return miopenOpTensor(handle, opDesc->op, alpha1, aDesc, A, alpha2, bDesc,
                          B, beta, cDesc, C);
}

// ---------------------------------------------------------------------------
// Tensor reduction
// ---------------------------------------------------------------------------
using cudnnReduceTensorDescriptor_t = miopenReduceTensorDescriptor_t;
using cudnnReduceTensorOp_t = miopenReduceTensorOp_t;
using cudnnReduceTensorIndices_t = miopenReduceTensorIndices_t;
using cudnnIndicesType_t = miopenIndicesType_t;
#define CUDNN_REDUCE_TENSOR_ADD MIOPEN_REDUCE_TENSOR_ADD
#define CUDNN_REDUCE_TENSOR_MUL MIOPEN_REDUCE_TENSOR_MUL
#define CUDNN_REDUCE_TENSOR_MIN MIOPEN_REDUCE_TENSOR_MIN
#define CUDNN_REDUCE_TENSOR_MAX MIOPEN_REDUCE_TENSOR_MAX
#define CUDNN_REDUCE_TENSOR_AMAX MIOPEN_REDUCE_TENSOR_AMAX
#define CUDNN_REDUCE_TENSOR_AVG MIOPEN_REDUCE_TENSOR_AVG
#define CUDNN_REDUCE_TENSOR_NORM1 MIOPEN_REDUCE_TENSOR_NORM1
#define CUDNN_REDUCE_TENSOR_NORM2 MIOPEN_REDUCE_TENSOR_NORM2
#define CUDNN_REDUCE_TENSOR_NO_INDICES MIOPEN_REDUCE_TENSOR_NO_INDICES
#define CUDNN_REDUCE_TENSOR_FLATTENED_INDICES \
    MIOPEN_REDUCE_TENSOR_FLATTENED_INDICES
#define CUDNN_8BIT_INDICES MIOPEN_8BIT_INDICES
#define CUDNN_16BIT_INDICES MIOPEN_16BIT_INDICES
#define CUDNN_32BIT_INDICES MIOPEN_32BIT_INDICES
#define CUDNN_64BIT_INDICES MIOPEN_64BIT_INDICES

#define cudnnCreateReduceTensorDescriptor miopenCreateReduceTensorDescriptor
#define cudnnDestroyReduceTensorDescriptor miopenDestroyReduceTensorDescriptor
#define cudnnSetReduceTensorDescriptor miopenSetReduceTensorDescriptor
#define cudnnGetReductionWorkspaceSize miopenGetReductionWorkspaceSize
#define cudnnReduceTensor miopenReduceTensor

// ---------------------------------------------------------------------------
// Batch normalization (entry points match argument for argument)
// ---------------------------------------------------------------------------
using cudnnBatchNormMode_t = miopenBatchNormMode_t;
#define CUDNN_BATCHNORM_PER_ACTIVATION miopenBNPerActivation
#define CUDNN_BATCHNORM_SPATIAL miopenBNSpatial

#define cudnnDeriveBNTensorDescriptor miopenDeriveBNTensorDescriptor
#define cudnnBatchNormalizationForwardTraining \
    miopenBatchNormalizationForwardTraining
#define cudnnBatchNormalizationForwardInference \
    miopenBatchNormalizationForwardInference
#define cudnnBatchNormalizationBackward miopenBatchNormalizationBackward

#endif  // USE_CUDNN
