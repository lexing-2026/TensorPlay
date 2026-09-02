 #include <iomanip>
#include <type_traits>
#include "Tensor.h"
#include "tensorplay/ops/TPXOpsGenerated.h"
namespace ops = tensorplay::tpx::ops;
#include "TensorImpl.h"
#include "Storage.h"
#include "Quantizer.h"
#include "SparseKernels.h"
#include "BatchingKernels.h"
#include "Utils.h"
#include "ErrorReporting.h"
#include <iostream>
#include <cstring>
#include <sstream>

namespace tensorplay {

// Helper for DType output
std::ostream& operator<<(std::ostream& os, DType dt) {
    switch (dt) {
        case DType::UInt8: os << "uint8"; break;
        case DType::Int8: os << "int8"; break;
        case DType::Int16: os << "int16"; break;
        case DType::Float32: os << "float32"; break;
        case DType::Float64: os << "float64"; break;
        case DType::Int32: os << "int32"; break;
        case DType::Int64: os << "int64"; break;
        case DType::UInt16: os << "uint16"; break;
        case DType::UInt32: os << "uint32"; break;
        case DType::UInt64: os << "uint64"; break;
        case DType::Float16: os << "float16"; break;
        case DType::BFloat16: os << "bfloat16"; break;
        case DType::ComplexHalf: os << "complex32"; break;
        case DType::ComplexFloat: os << "complex64"; break;
        case DType::ComplexDouble: os << "complex128"; break;
        case DType::BComplex32: os << "bcomplex32"; break;
        case DType::Bool: os << "bool"; break;
        case DType::QInt8: os << "qint8"; break;
        case DType::QUInt8: os << "quint8"; break;
        case DType::QInt32: os << "qint32"; break;
        default: os << "dtype(" << static_cast<int>(dt) << ")"; break;
    }
    return os;
}

// Helper for Device output
std::ostream& operator<<(std::ostream& os, const Device& d) {
    os << (d.type() == DeviceType::CPU ? "cpu" : "cuda");
    if (d.index() != -1) os << ":" << d.index();
    return os;
}

// Size implementation
std::string Size::toString() const {
    std::stringstream ss;
    ss << "tensorplay.Size(";
    for (size_t i = 0; i < sizes_.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << sizes_[i];
    }
    ss << ")";
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Size& s) {
    os << s.toString();
    return os;
}

Tensor::Tensor(const std::vector<int64_t>& sizes, DType dtype, const Device& device) {
    impl_ = std::make_shared<TensorImpl>(sizes, dtype, device);
}

Tensor::Tensor(Storage storage, const std::vector<int64_t>& sizes, DType dtype) {
    impl_ = std::make_shared<TensorImpl>(storage, sizes, dtype);
}

Tensor::Tensor(Storage storage, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, size_t storage_offset) {
    impl_ = std::make_shared<TensorImpl>(storage, sizes, strides, dtype, storage_offset);
}

Tensor::Tensor(const std::vector<int64_t>& sizes, Scalar fill_value, const Device& device) {
    impl_ = std::make_shared<TensorImpl>(sizes, fill_value.dtype(), device);
    fill_(fill_value);
}

Tensor Tensor::make_sparse_coo_tensor(const Tensor& indices,
                                      const Tensor& values,
                                      const std::vector<int64_t>& size,
                                      bool is_coalesced) {
    if (!indices.defined() || !values.defined()) {
        TP_THROW(ValueError, "sparse_coo_tensor: indices and values must be defined");
    }
    if (indices.device() != values.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse_coo_tensor: indices and values must be on the same device");
    }
    if (indices.dim() != 2) {
        TP_THROW(ValueError, "sparse_coo_tensor: indices must be a 2-D tensor");
    }
    if (indices.dtype() != DType::Int64) {
        if (indices.dtype() != DType::Int32) {
            TP_THROW(TypeError, "sparse_coo_tensor: indices must be Int32 or Int64");
        }
    }
    if (size.size() < static_cast<size_t>(indices.size(0))) {
        TP_THROW(ValueError, "sparse_coo_tensor: size has fewer dimensions than indices");
    }
    for (int64_t dim_size : size) {
        if (dim_size < 0) {
            TP_THROW(ValueError, "sparse_coo_tensor: size entries must be non-negative");
        }
    }

    Tensor canonical_indices = indices.dtype() == DType::Int64
        ? indices
        : indices.to(DType::Int64);
    const int64_t sparse_dim = canonical_indices.size(0);
    const int64_t nnz = canonical_indices.size(1);
    if (values.dim() == 0 || values.size(0) != nnz) {
        TP_THROW(ValueError,
                 "sparse_coo_tensor: values first dimension must equal indices nnz");
    }
    if (values.dim() - 1 != static_cast<int64_t>(size.size()) - sparse_dim) {
        TP_THROW(ValueError,
                 "sparse_coo_tensor: values dense dimensions do not match size");
    }
    for (int64_t i = 0; i < values.dim() - 1; ++i) {
        if (values.size(i + 1) != size[static_cast<size_t>(sparse_dim + i)]) {
            TP_THROW(ValueError,
                     "sparse_coo_tensor: values shape does not match sparse size");
        }
    }

    // Construct only the logical TensorImpl.  A sparse tensor must not first
    // allocate storage for its full logical dense shape; large embeddings
    // would otherwise briefly materialize the entire parameter.
    Tensor result(std::make_shared<TensorImpl>(
        size, values.dtype(), values.device(), /*allocate_storage=*/false));
    result.unsafeGetTensorImpl()->set_sparse_state(
        canonical_indices.unsafeGetTensorImpl(),
        values.unsafeGetTensorImpl(),
        size,
        is_coalesced);
    return result;
}

Tensor Tensor::make_sparse_csr_tensor(const Tensor& crow,
                                      const Tensor& col,
                                      const Tensor& values,
                                      const std::vector<int64_t>& size) {
    if (!crow.defined() || !col.defined() || !values.defined()) {
        TP_THROW(ValueError, "sparse_csr_tensor: crow/col/values must be defined");
    }
    if (crow.device() != col.device() || crow.device() != values.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse_csr_tensor: crow/col/values must be on the same device");
    }
    if (crow.dim() != 1 || col.dim() != 1) {
        TP_THROW(ValueError, "sparse_csr_tensor: crow/col must be 1-D tensors");
    }
    if (size.size() != 2) {
        TP_THROW(ValueError, "sparse_csr_tensor: CSR layout supports exactly 2-D sizes");
    }
    if (crow.size(0) != size[0] + 1) {
        TP_THROW(ValueError,
                 "sparse_csr_tensor: crow must have rows+1 entries");
    }
    if (col.size(0) != values.size(0)) {
        TP_THROW(ValueError,
                 "sparse_csr_tensor: col and values must have the same nnz");
    }
    Tensor canonical_crow = crow.dtype() == DType::Int64 ? crow : crow.to(DType::Int64);
    Tensor canonical_col = col.dtype() == DType::Int64 ? col : col.to(DType::Int64);

    // Same rationale as the COO constructor: install logical metadata only.
    Tensor result(std::make_shared<TensorImpl>(
        size, values.dtype(), values.device(), /*allocate_storage=*/false));
    result.unsafeGetTensorImpl()->set_sparse_csr_state(
        canonical_crow.unsafeGetTensorImpl(),
        canonical_col.unsafeGetTensorImpl(),
        values.unsafeGetTensorImpl(),
        size);
    return result;
}

int64_t Tensor::dim() const { return impl_ ? impl_->dim() : 0; }
int64_t Tensor::numel() const { return impl_ ? impl_->numel() : 0; }
Size Tensor::shape() const { return impl_ ? Size(impl_->sizes()) : Size(); }
std::vector<int64_t> Tensor::strides() const {
    if (is_sparse()) return std::vector<int64_t>(static_cast<size_t>(dim()), 0);
    return impl_ ? impl_->strides().vec() : std::vector<int64_t>();
}
int64_t Tensor::size(int64_t dim) const {
    if (!impl_) return 0;
    if (dim < 0) dim += impl_->dim();
    return impl_->size(dim);
}
int64_t Tensor::stride(int64_t dim) const {
    if (!impl_) return 0;
    if (is_sparse()) return 0;
    if (dim < 0) dim += impl_->dim();
    return impl_->stride(dim);
}
DType Tensor::dtype() const { return impl_ ? impl_->dtype() : DType::Undefined; }
Device Tensor::device() const { return impl_ ? impl_->device() : Device(DeviceType::CPU); }
size_t Tensor::itemsize() const { return impl_ ? impl_->itemsize() : 0; }
bool Tensor::is_contiguous() const { return impl_ ? impl_->is_contiguous() : false; }

Tensor Tensor::transform_value() const {
    if (!impl_ || !impl_->is_batched()) return *this;
    return Tensor(impl_->transform_value_impl());
}

bool Tensor::requires_grad() const {
    return impl_ && impl_->autograd_meta() && impl_->autograd_meta()->requires_grad();
}

void Tensor::set_requires_grad(bool requires_grad) {
    if (!impl_) return;
    impl_->set_requires_grad(requires_grad);
}

Tensor Tensor::grad() const {
    if (impl_ && impl_->autograd_meta()) return impl_->autograd_meta()->grad();
    return Tensor();
}

void Tensor::set_grad(const Tensor& grad) {
    if (!impl_) return;
    if (auto* meta = impl_->autograd_meta()) meta->set_grad(grad);
}

bool Tensor::retains_grad() const {
    return impl_ && impl_->autograd_meta() && impl_->autograd_meta()->retains_grad();
}

void Tensor::set_retains_grad(bool retains_grad) {
    if (!impl_) return;
    if (auto* meta = impl_->autograd_meta()) meta->set_retains_grad(retains_grad);
}

Tensor Tensor::detach() const {
    if (!impl_) return Tensor();
    return Tensor(std::make_shared<TensorImpl>(*impl_));
}

bool Tensor::is_pinned() const {
#ifdef USE_CUDA
    return impl_ && device().is_cpu() && impl_->has_storage() &&
           impl_->storage().allocator() == getPinnedMemoryAllocator();
#else
    return false;
#endif
}

Tensor Tensor::pin_memory() const {
    if (!impl_) return Tensor();
    if (!device().is_cpu()) {
        TP_THROW(RuntimeError, "cannot pin a tensor on " + device().toString() +
                               "; only dense CPU tensors can be pinned");
    }
#ifdef USE_CUDA
    if (is_pinned()) return *this;
    const auto sizes = static_cast<std::vector<int64_t>>(shape());
    const size_t nbytes = static_cast<size_t>(numel()) * itemsize();
    Storage storage(nbytes, getPinnedMemoryAllocator(), Device(DeviceType::CPU));
    Tensor result(storage, sizes, dtype());
    result.copy_(*this);
    return result;
#else
    TP_THROW(RuntimeError, "pin_memory requires a CUDA-enabled TensorPlay build");
#endif
}

bool Tensor::is_sparse() const { return impl_ && impl_->is_sparse(); }

bool Tensor::is_coalesced() const {
    if (!is_sparse() || is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "is_coalesced expected sparse coordinate tensor layout");
    }
    return impl_->is_coalesced();
}

int64_t Tensor::sparse_dim() const {
    if (!is_sparse()) return 0;
    return _indices().dim() == 0 ? 0 : _indices().size(0);
}

int64_t Tensor::dense_dim() const {
    if (!is_sparse()) return dim();
    return _values().dim() == 0 ? 0 : _values().dim() - 1;
}

Tensor Tensor::_indices() const {
    if (!is_sparse() || is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "_indices() is only defined for sparse COO tensors");
    }
    return Tensor(impl_->sparse_indices_impl());
}

Tensor Tensor::_values() const {
    if (!is_sparse()) {
        TP_THROW(RuntimeError,
                 "_values() is only defined for sparse tensors");
    }
    return Tensor(impl_->sparse_values_impl());
}

bool Tensor::is_sparse_csr() const {
    return is_sparse() && impl_->sparse_layout() == TensorImpl::kSparseCSRLayout;
}

Tensor Tensor::_crow_indices() const {
    if (!is_sparse_csr()) {
        TP_THROW(RuntimeError, "_crow_indices() is only defined for sparse CSR tensors");
    }
    return Tensor(impl_->sparse_crow_impl());
}

Tensor Tensor::_col_indices() const {
    if (!is_sparse_csr()) {
        TP_THROW(RuntimeError, "_col_indices() is only defined for sparse CSR tensors");
    }
    return Tensor(impl_->sparse_col_impl());
}

Tensor Tensor::coalesce() const {
    if (!is_sparse() || is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "coalesce() is only defined for sparse COO tensors");
    }
    if (is_coalesced()) return *this;
    if (device().is_cpu()) return cpu::coalesce_sparse_cpu(*this);
#ifdef USE_CUDA
    if (device().is_cuda()) return cuda::coalesce_sparse_cuda(*this);
#endif
    TP_THROW(NotImplementedError, "coalesce() is not implemented for this device");
}

Tensor Tensor::sparse_mask(const Tensor& mask) const {
    if (is_sparse()) {
        TP_THROW(RuntimeError, "sparse_mask(): self must be dense");
    }
    if (!mask.is_sparse() || mask.is_sparse_csr()) {
        TP_THROW(RuntimeError, "sparse_mask(): mask must be sparse COO");
    }
    if (shape() != mask.shape()) {
        TP_THROW(RuntimeError,
                 "sparse_mask(): operands have incompatible sizes");
    }
    if (device() != mask.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse_mask(): operands must be on the same device");
    }
    if (device().is_cpu()) return cpu::sparse_mask_cpu(*this, mask);
#ifdef USE_CUDA
    if (device().is_cuda()) return cuda::sparse_mask_cuda(*this, mask);
#endif
    TP_THROW(NotImplementedError,
             "sparse_mask() is not implemented for this device");
}

void* Tensor::data_ptr() const {
    if (is_sparse()) {
        TP_THROW(RuntimeError, "data_ptr() is not supported for sparse COO tensors");
    }
    return impl_ ? impl_->data() : nullptr;
}

Scalar Tensor::item() const {
    if (is_sparse()) {
        TP_THROW(RuntimeError, "item() is not supported for sparse tensors");
    }
    if (numel() != 1) {
        TP_THROW(ValueError, "item() only supported for 1-element tensors");
    }

    if (device().type() != DeviceType::CPU) {
        return to(Device(DeviceType::CPU)).item();
    }

    switch (dtype()) {
        case DType::Float32: return Scalar(static_cast<double>(*data_ptr<float>()));
        case DType::Float64: return Scalar(*data_ptr<double>());
        case DType::Float16: return Scalar(static_cast<float>(*data_ptr<Half>()));
        case DType::BFloat16: return Scalar(static_cast<float>(*data_ptr<BFloat16>()));
        case DType::Float8_e4m3fn: return Scalar(static_cast<float>(*data_ptr<Float8_e4m3fn>()));
        case DType::Float8_e5m2: return Scalar(static_cast<float>(*data_ptr<Float8_e5m2>()));
        case DType::Int8: return Scalar(static_cast<int64_t>(*data_ptr<int8_t>()));
        case DType::Int16: return Scalar(static_cast<int64_t>(*data_ptr<int16_t>()));
        case DType::Int32: return Scalar(static_cast<int64_t>(*data_ptr<int32_t>()));
        case DType::Int64: return Scalar(*data_ptr<int64_t>());
        case DType::UInt8: return Scalar(static_cast<uint64_t>(*data_ptr<uint8_t>()));
        case DType::UInt16: return Scalar(static_cast<uint64_t>(*data_ptr<uint16_t>()));
        case DType::UInt32: return Scalar(static_cast<uint64_t>(*data_ptr<uint32_t>()));
        case DType::UInt64: return Scalar(*data_ptr<uint64_t>());
        case DType::Bool: return Scalar(static_cast<bool>(*data_ptr<bool>()));
        case DType::ComplexHalf: {
            const auto value = *data_ptr<std::complex<Half>>();
            return Scalar(std::complex<float>(static_cast<float>(value.real()),
                                              static_cast<float>(value.imag())));
        }
        case DType::ComplexFloat: return Scalar(*data_ptr<std::complex<float>>());
        case DType::ComplexDouble: return Scalar(*data_ptr<std::complex<double>>());
        case DType::BComplex32: {
            const auto value = *data_ptr<std::complex<BFloat16>>();
            return Scalar(std::complex<float>(static_cast<float>(value.real()),
                                              static_cast<float>(value.imag())));
        }
        default:
            TP_THROW(NotImplementedError, "item() not implemented for this dtype");
    }
}

namespace {
    struct PrintOptions {
        int64_t edge_items = 3;
        int64_t threshold = 1000;
        int64_t precision = 4;
        int64_t linewidth = 80;
    };

    static PrintOptions g_print_options;

    template <typename T>
    std::string format_float(T value, const PrintOptions& options) {
        std::stringstream ss;
        ss << std::fixed << std::setprecision(options.precision) << value;
        std::string s = ss.str();
        // Remove trailing zeros
        size_t last_not_zero = s.find_last_not_of('0');
        if (last_not_zero != std::string::npos) {
            size_t dot_pos = s.find('.');
            if (dot_pos != std::string::npos) {
                // If we stripped everything after dot, keep the dot
                // e.g. 1.0000 -> last_not_zero is at '1' (index < dot_pos) -> erase after dot
                // e.g. 10.0000 -> last_not_zero is at '0' (index < dot_pos) -> erase after dot
                // e.g. 0.0000 -> last_not_zero is at '.' (index == dot_pos) -> erase after dot
                // e.g. 1.2000 -> last_not_zero is at '2' (index > dot_pos) -> erase after '2'
                
                if (last_not_zero <= dot_pos) {
                    s.erase(dot_pos + 1);
                } else {
                    s.erase(last_not_zero + 1);
                }
            }
        }
        return s;
    }

    template <typename T>
    void print_data_recursive(std::ostream& os, const T* data, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, int64_t dim, int64_t indent, const PrintOptions& options, bool summarizing) {
        if (sizes.empty()) { // Scalar 0-dim
             if constexpr (std::is_floating_point_v<T>) {
                os << format_float(*data, options);
            } else {
                os << *data;
            }
            return;
        }

        if (dim == sizes.size()) {
             // Should not happen if sizes is not empty and logic is correct, but base case for recursion
             if constexpr (std::is_floating_point_v<T>) {
                os << format_float(*data, options);
            } else {
                os << *data;
            }
            return;
        }

        int64_t size = sizes[dim];
        int64_t stride = strides[dim];
        bool summarize_dim = summarizing && (size > 2 * options.edge_items);

        if (dim == sizes.size() - 1) { // Last dimension (row)
            os << "[";
            int64_t count = summarize_dim ? options.edge_items : size;
            
            for (int64_t i = 0; i < count; ++i) {
                if (i > 0) os << ", ";
                if constexpr (std::is_floating_point_v<T>) {
                    os << format_float(data[i * stride], options);
                } else {
                    os << (std::is_same_v<T, bool> ? (data[i * stride] ? "True" : "False") : std::to_string(data[i * stride]));
                }
            }
            
            if (summarize_dim) {
                os << ", ...";
                for (int64_t i = size - options.edge_items; i < size; ++i) {
                    os << ", ";
                    if constexpr (std::is_floating_point_v<T>) {
                        os << format_float(data[i * stride], options);
                    } else {
                        os << (std::is_same_v<T, bool> ? (data[i * stride] ? "True" : "False") : std::to_string(data[i * stride]));
                    }
                }
            }
            os << "]";
            return;
        }


        // Higher dimensions
        os << "[";
        int64_t count = summarize_dim ? options.edge_items : size;
        
        for (int64_t i = 0; i < count; ++i) {
            if (i > 0) {
                os << ",\n";
                for (int k = 0; k < indent + 1; ++k) os << " "; 
            }
            print_data_recursive(os, data + i * stride, sizes, strides, dim + 1, indent + 1, options, summarizing);
        }
        
        if (summarize_dim) {
            os << ",\n";
            for (int k = 0; k < indent + 1; ++k) os << " "; 
            os << "...";
            
            for (int64_t i = size - options.edge_items; i < size; ++i) {
                os << ",\n";
                for (int k = 0; k < indent + 1; ++k) os << " "; 
                print_data_recursive(os, data + i * stride, sizes, strides, dim + 1, indent + 1, options, summarizing);
            }
        }
        os << "]";
    }
}

void set_printoptions(int64_t edge_items, int64_t threshold, int64_t precision, int64_t linewidth) {
    if (edge_items >= 0) g_print_options.edge_items = edge_items;
    if (threshold >= 0) g_print_options.threshold = threshold;
    if (precision >= 0) g_print_options.precision = precision;
    if (linewidth > 0) g_print_options.linewidth = linewidth;
}

std::string Tensor::toString() const {
    if (!impl_) return "Tensor(Undefined)";

    if (is_sparse()) {
        std::stringstream sparse;
        sparse << "tensor(indices=" << _indices().toString()
               << ", values=" << _values().toString()
               << ", size=" << shape()
               << ", nnz=" << _indices().size(1)
               << ", layout=sparse_coo)";
        return sparse.str();
    }
    
    std::stringstream ss;

    // 为了支持非CPU张量的打印（如CUDA），我们需要将其拷贝到CPU
    Tensor tensor_to_print = *this;
    if (device().type() != DeviceType::CPU) {
        try {
            // 尝试拷贝到CPU
            tensor_to_print = this->to(Device(DeviceType::CPU));
        } catch (...) {
            // 如果拷贝失败（例如未编译CUDA支持但加载了CUDA张量），回退到仅显示元数据
            ss << "Tensor(shape=" << shape() << ", dtype=" << dtype() << ", device=" << device() << ")";
            return ss.str();
        }
    }

    PrintOptions options = g_print_options;
    bool summarizing = numel() > options.threshold;
    
    ss << "tensor(";
    std::vector<int64_t> current_sizes = static_cast<std::vector<int64_t>>(tensor_to_print.shape());
    std::vector<int64_t> current_strides = tensor_to_print.strides();
    
    switch (tensor_to_print.dtype()) {
        case DType::Float32:
            print_data_recursive(ss, tensor_to_print.data_ptr<float>(), current_sizes, current_strides, 0, 7, options, summarizing);
            break;
        case DType::Float64:
            print_data_recursive(ss, tensor_to_print.data_ptr<double>(), current_sizes, current_strides, 0, 7, options, summarizing);
            break;
        case DType::Int32:
            print_data_recursive(ss, tensor_to_print.data_ptr<int32_t>(), current_sizes, current_strides, 0, 7, options, summarizing);
            break;
        case DType::Int64:
            print_data_recursive(ss, tensor_to_print.data_ptr<int64_t>(), current_sizes, current_strides, 0, 7, options, summarizing);
            break;
        case DType::Bool:
            print_data_recursive(ss, tensor_to_print.data_ptr<bool>(), current_sizes, current_strides, 0, 7, options, summarizing);
            break;
        case DType::QInt8:
            print_data_recursive(ss, tensor_to_print.data_ptr<int8_t>(), current_sizes, current_strides, 0, 7, options, summarizing);
            break;
        case DType::QUInt8:
            print_data_recursive(ss, tensor_to_print.data_ptr<uint8_t>(), current_sizes, current_strides, 0, 7, options, summarizing);
            break;
        case DType::QInt32:
            print_data_recursive(ss, tensor_to_print.data_ptr<int32_t>(), current_sizes, current_strides, 0, 7, options, summarizing);
            break;
        default:
            ss << "Tensor(shape=" << shape() << ", dtype=" << dtype() << ", device=" << device() << ")";
            return ss.str();
    }
    
    if (dtype() != DType::Float32) {
        ss << ", dtype=" << dtype();
    }

    // Quantized tensors surface their scheme and affine parameters in the
    // representation, so the mapping is visible alongside the codes.
    if (impl_->has_quantizer()) {
        const auto q = impl_->quantizer();
        ss << ", quantization_scheme=tensorplay." << tensorplay::toString(q->qscheme());
        if (q->qscheme() == QScheme::PerTensorAffine) {
            ss << ", scale=" << q->scale()
               << ", zero_point=" << q->zero_point();
        } else {
            ss << ", scale=" << q->scales().toString()
               << ", zero_point=" << q->zero_points().toString()
               << ", axis=" << q->axis();
        }
    }

    if (device().type() != DeviceType::CPU) {
        ss << ", device='" << device() << "'";
    }

    ss << ")";
    
    return ss.str();
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << t.toString();
    return os;
}

Tensor operator-(const Tensor& t) {
    return t * Scalar(-1);
}

Tensor Tensor::as_strided(const std::vector<int64_t>& size,
                          const std::vector<int64_t>& stride,
                          std::optional<int64_t> storage_offset) const {
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");
    if (size.size() != stride.size()) {
        TP_THROW(ValueError,
                 "as_strided(): sizes and strides must have the same length");
    }
    for (int64_t value : size) {
        if (value < 0) {
            TP_THROW(ValueError, "as_strided(): sizes must be non-negative");
        }
    }
    const int64_t offset = storage_offset.value_or(
        static_cast<int64_t>(impl_->storage_offset()));
    if (offset < 0) {
        TP_THROW(ValueError, "as_strided(): storage_offset must be non-negative");
    }
    // The Vulkan payload has no stride addressing: a strided view over it
    // would read back in storage order, so the view materializes on the GPU
    // through the backend's as_strided kernel instead of aliasing storage.
    if (impl_->device().is_vulkan()) {
        static const OperatorHandle handle =
            Dispatcher::singleton().findHandle("as_strided");
        return DispatchStub<Tensor, const Tensor&, const std::vector<int64_t>&,
                            const std::vector<int64_t>&,
                            std::optional<int64_t>>::
            call(handle, DispatchKey::Vulkan, *this, size, stride,
                 std::optional<int64_t>(offset));
    }
    Tensor out = Tensor(impl_->storage(), size, stride, impl_->dtype(),
                        static_cast<size_t>(offset));
    out.unsafeGetTensorImpl()->share_version_counter(*impl_);
    // Every view op funnels through here: a quantized source's quantizer
    // rides along since the view aliases the same codes and mapping.
    if (impl_->has_quantizer()) {
        out.unsafeGetTensorImpl()->set_quantizer(impl_->quantizer());
    }
    return out;
}

Tensor Tensor::view(const std::vector<int64_t>& shape) const {
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");

    std::vector<int64_t> inferred = SizesAndStrides::infer_size(shape, numel());
    auto stride = SizesAndStrides::compute_view_strides(
        static_cast<std::vector<int64_t>>(this->shape()), strides(), inferred);
    if (!stride.has_value()) {
        TP_THROW(RuntimeError,
                 "view size is not compatible with input tensor's size and stride");
    }
    return as_strided(inferred, *stride);
}

// Reinterprets the element stream as `dtype` while aliasing the same storage.  Same-size
// dtypes keep shape/strides; otherwise only the last dimension may change
Tensor Tensor::view_dtype(DType dtype) const {
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");
    const DType self_dtype = impl_->dtype();
    if (dtype == self_dtype) {
        // aliasing the storage (metadata changes on it must not leak back to
        // the original tensor), carrying the base's version counter.  Note
        // x.view(dtype).detach_() is legal and _is_view() is False.
        Tensor out = Tensor(impl_->storage(),
                            static_cast<std::vector<int64_t>>(shape()),
                            static_cast<std::vector<int64_t>>(strides()),
                            dtype, impl_->storage_offset());
        out.unsafeGetTensorImpl()->share_version_counter(*impl_);
        if (impl_->has_quantizer()) {
            out.unsafeGetTensorImpl()->set_quantizer(impl_->quantizer());
        }
        return out;
    }
    // A reinterpretation crosses VkFormat boundaries on the Vulkan payload:
    // the texture was allocated for the source element type, so aliasing it
    // as another dtype cannot be addressed.  Same-dtype views stay safe.
    if (impl_->device().is_vulkan()) {
        TP_THROW(NotImplementedError,
                 "view(dtype) reinterpretation is not supported on Vulkan tensors");
    }

    const std::vector<int64_t> self_sizes = static_cast<std::vector<int64_t>>(shape());
    const std::vector<int64_t> self_strides = strides();
    const size_t src_esize = elementSize(self_dtype);
    const size_t dst_esize = elementSize(dtype);

    if (self_strides.empty()) {
        TP_THROW(RuntimeError,
                 "view(): cannot reinterpret a 0-dim tensor to a dtype of a different element size");
    }
    if (self_strides.back() != 1 && src_esize != dst_esize) {
        TP_THROW(RuntimeError,
                 "view(): view(dtype) requires the last dimension to be contiguous when "
                 "element sizes differ");
    }

    std::vector<int64_t> new_sizes = self_sizes;
    std::vector<int64_t> new_strides = self_strides;
    // Storage offset is measured in elements, so it must be rescaled by the
    // downsizing, offset / ratio when upsizing).
    size_t new_offset = impl_->storage_offset();

    if (dst_esize < src_esize) {
        const int64_t ratio = static_cast<int64_t>(src_esize / dst_esize);
        new_sizes.back() *= ratio;
        for (size_t i = 0; i + 1 < new_strides.size(); ++i) new_strides[i] *= ratio;
        new_offset = impl_->storage_offset() * static_cast<size_t>(ratio);
    } else if (dst_esize > src_esize) {
        const int64_t ratio = static_cast<int64_t>(dst_esize / src_esize);
        if (new_sizes.back() % ratio != 0) {
            TP_THROW(RuntimeError,
                     "view(): the last dimension must be divisible by the element size ratio");
        }
        for (size_t i = 0; i + 1 < new_strides.size(); ++i) {
            if (new_strides[i] % ratio != 0) {
                TP_THROW(RuntimeError,
                         "view(): strides must be divisible by the element size ratio");
            }
            new_strides[i] /= ratio;
        }
        new_sizes.back() /= ratio;
        if ((impl_->storage_offset() * static_cast<int64_t>(src_esize)) %
                static_cast<int64_t>(dst_esize) != 0) {
            TP_THROW(RuntimeError,
                     "view(): storage offset is not aligned to the target element size");
        }
        new_offset = impl_->storage_offset() / static_cast<size_t>(ratio);
    }

    Tensor out = Tensor(impl_->storage(), new_sizes, new_strides, dtype, new_offset);
    out.unsafeGetTensorImpl()->share_version_counter(*impl_);
    return out;
}

Tensor Tensor::select(int64_t dim, int64_t index) const {
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");
    const int64_t ndim = this->dim();
    if (ndim == 0) {
        TP_THROW(IndexError, "select() cannot be applied to a 0-dim tensor");
    }
    const int64_t original_dim = dim;
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(IndexError, format_dim_range(ndim, original_dim));
    }

    const int64_t size_dim = size(dim);
    if (size_dim <= -1 - index || size_dim <= index) {
        TP_THROW(IndexError, "select(): index out of range");
    }
    if (index < 0) index += size_dim;

    std::vector<int64_t> new_sizes = static_cast<std::vector<int64_t>>(shape());
    std::vector<int64_t> new_strides = strides();
    const size_t new_offset = impl_->storage_offset() +
        static_cast<size_t>(index * new_strides[static_cast<size_t>(dim)]);
    new_sizes.erase(new_sizes.begin() + dim);
    new_strides.erase(new_strides.begin() + dim);
    return as_strided(new_sizes, new_strides, static_cast<int64_t>(new_offset));
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");
    if (is_batched()) {
        return transform::batch::slice(
            *this, dim, std::optional<int64_t>(start),
            std::optional<int64_t>(end), step);
    }
    int64_t ndim = this->dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) TP_THROW(IndexError, format_dim_range(ndim, dim));

    int64_t size_dim = size(dim);
    if (start < 0) start += size_dim;
    if (end < 0) end += size_dim;
    if (start < 0) start = 0;
    if (start > size_dim) start = size_dim;
    if (end < start) end = start;
    if (end > size_dim) end = size_dim;
    if (step <= 0) TP_THROW(ValueError, "Step must be positive");
    
    int64_t new_len = (end - start + step - 1) / step;
    if (new_len < 0) new_len = 0;
    
    std::vector<int64_t> new_sizes = static_cast<std::vector<int64_t>>(shape());
    std::vector<int64_t> new_strides = strides();
    
    new_sizes[dim] = new_len;
    new_strides[dim] *= step;
    
    size_t new_offset = impl_->storage_offset() + start * stride(dim);
    
    return as_strided(new_sizes, new_strides, new_offset);
}

Tensor Tensor::operator+(const Tensor& other) const { return add(other); }
Tensor Tensor::operator-(const Tensor& other) const { return sub(other); }
Tensor Tensor::operator*(const Tensor& other) const { return mul(other); }
Tensor Tensor::operator/(const Tensor& other) const { return div(other); }

Tensor Tensor::operator+(Scalar other) const { 
    // Keep the scalar overload on the dispatcher.  Materializing a scalar as
    // a Float32 tensor silently promoted fp16/bf16 tensors and introduced a
    // device allocation in the hot path (notably mean/RMSNorm).
    return add(other, Scalar(1));
}
Tensor Tensor::operator-(Scalar other) const {
    return sub(other, Scalar(1));
}
Tensor Tensor::operator*(Scalar other) const {
    return mul(other);
}
Tensor Tensor::operator/(Scalar other) const {
    return div(other);
}

Tensor& Tensor::operator+=(const Tensor& other) { return add_(other); }
Tensor& Tensor::operator-=(const Tensor& other) { return sub_(other); }
Tensor& Tensor::operator*=(const Tensor& other) { return mul_(other); }
Tensor& Tensor::operator/=(const Tensor& other) { return div_(other); }

Tensor& Tensor::operator+=(Scalar other) {
    return add_(other, Scalar(1));
}
Tensor& Tensor::operator-=(Scalar other) {
    return sub_(other, Scalar(1));
}
Tensor& Tensor::operator*=(Scalar other) {
    return mul_(other);
}
Tensor& Tensor::operator/=(Scalar other) {
    return div_(other);
}

namespace detail {

// Dispatcher-level clone shared by the generated Tensor::clone member via
// backend kernels.  Kept free-standing so kernels never re-enter the
// dispatcher for a plain byte copy.
//
// with no/Preserve memory_format the result keeps the input's exact strides
// when they are non-overlapping and dense (e.g. transposed tensors), and
// falls back to contiguous otherwise (expanded/overlapping inputs).  An
// channels-last formats.  Sparse clones reject any memory_format like
Tensor clone_impl(const Tensor& self, std::optional<MemoryFormat> memory_format) {
    if (!self.defined()) return Tensor();
    if (self.is_sparse()) {
        if (memory_format.has_value()) {
            TP_THROW(RuntimeError, "unsupported memory format option ",
                     toString(*memory_format));
        }
        return Tensor::make_sparse_coo_tensor(self._indices().clone(), self._values().clone(),
                                              static_cast<std::vector<int64_t>>(self.shape()),
                                              self.is_coalesced());
    }
    const auto format = memory_format.value_or(MemoryFormat::Preserve);
    const auto sizes_v = static_cast<std::vector<int64_t>>(self.shape());
    std::vector<int64_t> strides;
    MemoryFormat result_format = MemoryFormat::Contiguous;
    if (format == MemoryFormat::Preserve) {
        const auto strides_v = static_cast<std::vector<int64_t>>(self.strides());
        if (SizesAndStrides::is_non_overlapping_and_dense(sizes_v, strides_v)) {
            strides = strides_v;
        } else {
            strides = SizesAndStrides::compute_contiguous_strides(sizes_v);
        }
    } else if (format == MemoryFormat::ChannelsLast) {
        TP_THROW_IF(self.dim() != 4, RuntimeError,
                    "required rank 4 tensor to use channels_last format");
        strides = get_channels_last_strides(sizes_v);
        result_format = MemoryFormat::ChannelsLast;
    } else if (format == MemoryFormat::ChannelsLast3d) {
        TP_THROW_IF(self.dim() != 5, RuntimeError,
                    "required rank 5 tensor to use channels_last_3d format");
        strides = get_channels_last_strides(sizes_v);
        result_format = MemoryFormat::ChannelsLast3d;
    } else {
        strides = SizesAndStrides::compute_contiguous_strides(sizes_v);
    }
    Storage storage(static_cast<size_t>(self.numel()) * self.itemsize(),
                    getAllocator(self.device().type()), self.device());
    auto out_impl = std::make_shared<TensorImpl>(storage, sizes_v, strides, self.dtype(), 0);
    if (format == MemoryFormat::Preserve) {
        // Apply TensorImpl::set_sizes_and_strides layout tagging so a
        // preserved channels-last input clones into a channels-last tensor.
        if (strides == get_channels_last_strides(sizes_v) &&
            strides != SizesAndStrides::compute_contiguous_strides(sizes_v)) {
            result_format = sizes_v.size() == 5 ? MemoryFormat::ChannelsLast3d
                                                : MemoryFormat::ChannelsLast;
        }
    }
    out_impl->set_memory_format(result_format);
    // A clone is a full-value copy: a quantized source's quantizer rides
    // along so the result stays a quantized tensor with the same mapping.
    if (self.unsafeGetTensorImpl()->has_quantizer()) {
        out_impl->set_quantizer(self.unsafeGetTensorImpl()->quantizer());
    }
    Tensor t(std::move(out_impl));
    // every same-dtype contiguous clone through the dispatcher.  Optimizer
    // momentum initialization creates one clone per parameter, so this
    // dispatch overhead is visible even though the operation is just a byte
    // copy.  Everything else (preserved non-contiguous strides, channels-last
    // destinations, cross-device) retains copy_'s layout and transfer
    // semantics.  Note is_contiguous() alone is not enough here: it does not
    // distinguish row-major from channels-last layouts, so compare strides.
    const auto rm_strides = SizesAndStrides::compute_contiguous_strides(sizes_v);
    const bool row_major = strides == rm_strides &&
        static_cast<std::vector<int64_t>>(self.strides()) == rm_strides;
    if (self.device().is_cpu() && row_major) {
        std::memcpy(t.data_ptr(), self.data_ptr(),
                    static_cast<size_t>(self.numel()) * self.itemsize());
    } else {
        t.copy_(self);
    }
    // copy_ records a mutation on the destination; the clone result is a
    // 0), so clear the counter the internal copy bumped.
    t.unsafeGetTensorImpl()->reset_version();
    return t;
}

Tensor contiguous_clone(const Tensor& self) {
    if (!self.defined()) return Tensor();
    if (self.is_sparse()) return clone_impl(self);
    return clone_impl(self, MemoryFormat::Contiguous);
}

Tensor contiguous_impl(const Tensor& self, int64_t memory_format_raw) {
    auto format = static_cast<MemoryFormat>(memory_format_raw);
    if (format == MemoryFormat::Preserve) format = MemoryFormat::Contiguous;
    if (self.is_sparse()) return clone_impl(self);
    if (format != MemoryFormat::Contiguous && !self.is_contiguous(format)) {
        if (self.dim() < 3 ||
            (format == MemoryFormat::ChannelsLast && self.dim() != 4) ||
            (format == MemoryFormat::ChannelsLast3d && self.dim() != 5)) {
            // Channels-last layouts are only representable at these ranks;
            format = MemoryFormat::Contiguous;
        }
    }
    if (self.is_contiguous(format)) return self;
    const auto sizes_v = static_cast<std::vector<int64_t>>(self.shape());
    std::vector<int64_t> strides = get_strides_for(sizes_v, format);
    Storage storage(static_cast<size_t>(self.numel()) * self.itemsize(),
                    getAllocator(self.device().type()), self.device());
    auto out_impl = std::make_shared<TensorImpl>(storage, sizes_v, strides, self.dtype(), 0);
    out_impl->set_memory_format(format);
    if (self.unsafeGetTensorImpl()->has_quantizer()) {
        out_impl->set_quantizer(self.unsafeGetTensorImpl()->quantizer());
    }
    Tensor out(std::move(out_impl));
    out.copy_(self);
    out.unsafeGetTensorImpl()->reset_version();
    return out;
}

} // namespace detail


Tensor Tensor::to(DType dtype, bool non_blocking, bool copy) const {
    if (!impl_) return Tensor();
    if (is_sparse()) {
        if (dtype == this->dtype()) return copy ? clone() : *this;
        if (is_sparse_csr()) {
            return make_sparse_csr_tensor(
                _crow_indices(), _col_indices(),
                _values().to(dtype, non_blocking, copy),
                static_cast<std::vector<int64_t>>(shape()));
        }
        return make_sparse_coo_tensor(
            _indices(), _values().to(dtype, non_blocking, copy),
            static_cast<std::vector<int64_t>>(shape()), is_coalesced());
    }
    if (this->dtype() == dtype) {
        return copy ? clone() : *this;
    }
    Tensor t = ops::empty(
        impl_->sizes(), dtype, device(), /*pin_memory=*/false);
    t.copy_(*this, non_blocking);
    return t;
}

Tensor Tensor::to(Device device, bool non_blocking, bool copy) const {
    if (!impl_) return Tensor();
    if (is_sparse()) {
        if (this->device() == device) return copy ? clone() : *this;
        if (is_sparse_csr()) {
            return make_sparse_csr_tensor(
                _crow_indices().to(device, non_blocking, copy),
                _col_indices().to(device, non_blocking, copy),
                _values().to(device, non_blocking, copy),
                static_cast<std::vector<int64_t>>(shape()));
        }
        return make_sparse_coo_tensor(
            _indices().to(device, non_blocking, copy),
            _values().to(device, non_blocking, copy),
            static_cast<std::vector<int64_t>>(shape()), is_coalesced());
    }
    if (this->device() == device) {
        return copy ? clone() : *this;
    }
    Tensor t = ops::empty(
        impl_->sizes(), dtype(), device, /*pin_memory=*/false);
    t.copy_(*this, non_blocking);
    if (impl_->has_quantizer()) {
        t.unsafeGetTensorImpl()->set_quantizer(impl_->quantizer());
    }
    return t;
}

Tensor Tensor::to(Device device, DType dtype, bool non_blocking, bool copy) const {
    if (!impl_) return Tensor();
    if (is_sparse()) {
        if (this->device() == device && this->dtype() == dtype) {
            return copy ? clone() : *this;
        }
        if (is_sparse_csr()) {
            return make_sparse_csr_tensor(
                _crow_indices().to(device, non_blocking, copy),
                _col_indices().to(device, non_blocking, copy),
                _values().to(device, dtype, non_blocking, copy),
                static_cast<std::vector<int64_t>>(shape()));
        }
        return make_sparse_coo_tensor(
            _indices().to(device, non_blocking, copy),
            _values().to(device, dtype, non_blocking, copy),
            static_cast<std::vector<int64_t>>(shape()), is_coalesced());
    }
    if (this->device() == device && this->dtype() == dtype) {
        return copy ? clone() : *this;
    }
    Tensor t = ops::empty(
        impl_->sizes(), dtype, device, /*pin_memory=*/false);
    t.copy_(*this, non_blocking);
    return t;
}



bool Tensor::is_contiguous(MemoryFormat format) const {
    if (!impl_) return false;
    if (!defined()) return false;
    switch (format) {
        case MemoryFormat::Contiguous:
            return is_contiguous();
        case MemoryFormat::Preserve:
            return true;
        case MemoryFormat::ChannelsLast:
            return dim() == 4 && impl_->is_contiguous_in(format);
        case MemoryFormat::ChannelsLast3d:
            return dim() == 5 && impl_->is_contiguous_in(format);
    }
    return false;
}

MemoryFormat Tensor::memory_format() const { return impl_ ? impl_->memory_format() : MemoryFormat::Contiguous; }
bool Tensor::is_channels_last() const { return dim() == 4 && impl_ && impl_->is_channels_last(); }
bool Tensor::is_channels_last_2d() const { return is_channels_last(); }
bool Tensor::is_channels_last_3d() const { return dim() == 5 && impl_ && impl_->is_channels_last_3d(); }

} // namespace tensorplay
