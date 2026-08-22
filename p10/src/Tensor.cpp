 #include <iomanip>
#include <type_traits>
#include "Tensor.h"
#include "TensorImpl.h"
#include "Storage.h"
#include "SparseKernels.h"
#include "Utils.h"
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
        // ATen canonicalizes both int32 and int64 COO indices to int64.
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

bool Tensor::requires_grad() const {
    return impl_ && impl_->autograd_meta() && impl_->autograd_meta()->requires_grad();
}

void Tensor::set_requires_grad(bool requires_grad) {
    if (!impl_) return;
    if (auto* meta = impl_->autograd_meta()) {
        meta->set_requires_grad(requires_grad);
    } else if (requires_grad) {
        // No autograd metadata attached yet. The tpx layer owns the concrete
        // AutogradMeta type; without it, requires_grad cannot be enabled.
        // This mirrors a fresh TensorImpl in PyTorch, where requires_grad is
        // only meaningful once the autograd layer attaches metadata.
    }
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
    if (!is_sparse()) {
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
    if (!is_sparse()) TP_THROW(RuntimeError, "_indices() is only defined for sparse COO tensors");
    return Tensor(impl_->sparse_indices_impl());
}

Tensor Tensor::_values() const {
    if (!is_sparse()) TP_THROW(RuntimeError, "_values() is only defined for sparse COO tensors");
    return Tensor(impl_->sparse_values_impl());
}

Tensor Tensor::coalesce() const {
    if (!is_sparse()) TP_THROW(RuntimeError, "coalesce() is only defined for sparse COO tensors");
    if (is_coalesced()) return *this;
    if (device().is_cpu()) return cpu::coalesce_sparse_cpu(*this);
#ifdef USE_CUDA
    if (device().is_cuda()) return cuda::coalesce_sparse_cuda(*this);
#endif
    TP_THROW(NotImplementedError, "coalesce() is not implemented for this device");
}

Tensor Tensor::sparse_mask(const Tensor& mask) const {
    if (is_sparse()) TP_THROW(RuntimeError, "sparse_mask(): self must be dense");
    if (!mask.is_sparse()) TP_THROW(RuntimeError, "sparse_mask(): mask must be sparse COO");
    if (shape() != mask.shape()) {
        TP_THROW(RuntimeError, "sparse_mask(): operands have incompatible sizes");
    }
    if (device() != mask.device()) {
        TP_THROW(DeviceMismatchError, "sparse_mask(): operands must be on the same device");
    }
    if (device().is_cpu()) return cpu::sparse_mask_cpu(*this, mask);
#ifdef USE_CUDA
    if (device().is_cuda()) return cuda::sparse_mask_cuda(*this, mask);
#endif
    TP_THROW(NotImplementedError, "sparse_mask() is not implemented for this device");
}

void* Tensor::data_ptr() const {
    if (is_sparse()) {
        TP_THROW(RuntimeError, "data_ptr() is not supported for sparse COO tensors");
    }
    return impl_ ? impl_->data() : nullptr;
}

Scalar Tensor::item() const {
    if (is_sparse()) {
        TP_THROW(RuntimeError, "item() is not supported for sparse COO tensors");
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
        default: TP_THROW(NotImplementedError, "item() not implemented for this dtype");
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
    // 这样可以保持与PyTorch一致的体验
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
        default:
            ss << "Tensor(shape=" << shape() << ", dtype=" << dtype() << ", device=" << device() << ")";
            return ss.str();
    }
    
    if (dtype() != DType::Float32) {
        ss << ", dtype=" << dtype();
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

// View methods implementation

Tensor Tensor::as_strided(const std::vector<int64_t>& size, const std::vector<int64_t>& stride, std::optional<int64_t> storage_offset) const {
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");
    size_t offset = storage_offset.value_or(impl_->storage_offset());
    Tensor out = Tensor(impl_->storage(), size, stride, impl_->dtype(), offset);
    // A view aliases the base memory: share the version counter so that
    // in-place writes through either alias are visible to mutation tracking
    // (mirrors PyTorch view semantics).
    out.unsafeGetTensorImpl()->share_version_counter(*impl_);
    return out;
}

Tensor Tensor::view(const std::vector<int64_t>& shape) const {
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");
    if (!is_contiguous()) TP_THROW(RuntimeError, "view(): tensor must be contiguous");
    
    int64_t new_numel = 1;
    int infer_dim = -1;
    
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == -1) {
            if (infer_dim != -1) TP_THROW(RuntimeError, "view(): only one dimension can be inferred");
            infer_dim = i;
        } else {
            if (shape[i] < 0) TP_THROW(RuntimeError, "view(): invalid negative dimension");
            new_numel *= shape[i];
        }
    }
    
    std::vector<int64_t> final_shape = shape;
    if (infer_dim != -1) {
        if (new_numel == 0) TP_THROW(RuntimeError, "view(): cannot infer shape when other dimensions are 0");
        if (numel() % new_numel != 0) TP_THROW(RuntimeError, "view(): shape inference failed");
        final_shape[infer_dim] = numel() / new_numel;
        new_numel *= final_shape[infer_dim];
    }
    
    if (new_numel != numel()) {
        TP_THROW(RuntimeError, "view(): invalid shape, numel mismatch");
    }
    
    std::vector<int64_t> new_strides = SizesAndStrides::compute_contiguous_strides(final_shape);
    return as_strided(final_shape, new_strides);
}

Tensor Tensor::select(int64_t dim, int64_t index) const {
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");
    int64_t ndim = this->dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) TP_THROW(IndexError, "Dimension out of range");
    
    int64_t size_dim = size(dim);
    if (index < 0) index += size_dim;
    if (index < 0 || index >= size_dim) TP_THROW(IndexError, "Index out of range");
    
    std::vector<int64_t> new_sizes = static_cast<std::vector<int64_t>>(shape());
    std::vector<int64_t> new_strides = strides();
    
    size_t new_offset = impl_->storage_offset() + index * new_strides[dim];
    
    new_sizes.erase(new_sizes.begin() + dim);
    new_strides.erase(new_strides.begin() + dim);
    
    return as_strided(new_sizes, new_strides, new_offset);
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end, int64_t step) const {
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");
    int64_t ndim = this->dim();
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) TP_THROW(IndexError, "Dimension out of range");
    
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

Tensor Tensor::expand(const std::vector<int64_t>& size) const {
    if (!impl_) TP_THROW(RuntimeError, "Tensor not defined");
    int64_t ndim = dim();
    int64_t new_ndim = size.size();
    
    if (new_ndim < ndim) {
        TP_THROW(RuntimeError, "expand(): the number of sizes provided must be greater or equal to the number of dimensions in the tensor.");
    }
    
    std::vector<int64_t> new_sizes(size);
    std::vector<int64_t> new_strides(new_ndim);
    
    for (int64_t i = new_ndim - 1; i >= 0; --i) {
        int64_t offset = new_ndim - 1 - i;
        int64_t dim_index = ndim - 1 - offset;
        
        if (dim_index >= 0) {
            int64_t size_dim = this->size(dim_index);
            int64_t stride_dim = this->stride(dim_index);
            
            if (size_dim == 1 && new_sizes[i] > 1) {
                new_strides[i] = 0;
            } else if (size_dim == new_sizes[i]) {
                new_strides[i] = stride_dim;
            } else {
                 TP_THROW(RuntimeError, "expand(): incompatible dimensions");
            }
        } else {
            if (new_sizes[i] == -1) TP_THROW(RuntimeError, "expand(): invalid size -1");
            new_strides[i] = 0; 
        }
    }
    
    return as_strided(new_sizes, new_strides);
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

Tensor Tensor::clone() const {
    if (!impl_) return Tensor();
    if (is_sparse()) {
        return make_sparse_coo_tensor(_indices().clone(), _values().clone(),
                                      static_cast<std::vector<int64_t>>(shape()),
                                      is_coalesced());
    }
    Tensor t(impl_->sizes(), dtype(), device());
    // Match the native contiguous clone path used by Torch: avoid routing
    // every same-dtype contiguous clone through the dispatcher.  Optimizer
    // momentum initialization creates one clone per parameter, so this
    // dispatch overhead is visible even though the operation is just a byte
    // copy.  Non-contiguous and cross-device cases retain copy_'s layout and
    // transfer semantics.
    if (device().is_cpu() && is_contiguous()) {
        std::memcpy(t.data_ptr(), data_ptr(),
                    static_cast<size_t>(numel()) * itemsize());
    } else {
        t.copy_(*this);
    }
    // copy_ records a mutation on the destination; the clone result is a
    // freshly materialized tensor and must start unmutated (PyTorch: version
    // 0), so clear the counter the internal copy bumped.
    t.unsafeGetTensorImpl()->reset_version();
    return t;
}

Tensor Tensor::to(DType dtype, bool non_blocking, bool copy) const {
    if (!impl_) return Tensor();
    if (is_sparse()) {
        if (dtype == this->dtype()) return copy ? clone() : *this;
        return make_sparse_coo_tensor(
            _indices(), _values().to(dtype, non_blocking, copy),
            static_cast<std::vector<int64_t>>(shape()), is_coalesced());
    }
    if (this->dtype() == dtype) {
        return copy ? clone() : *this;
    }
    Tensor t(impl_->sizes(), dtype, device());
    t.copy_(*this, non_blocking);
    return t;
}

Tensor Tensor::to(Device device, bool non_blocking, bool copy) const {
    if (!impl_) return Tensor();
    if (is_sparse()) {
        if (this->device() == device) return copy ? clone() : *this;
        return make_sparse_coo_tensor(
            _indices().to(device, non_blocking, copy),
            _values().to(device, non_blocking, copy),
            static_cast<std::vector<int64_t>>(shape()), is_coalesced());
    }
    if (this->device() == device) {
        return copy ? clone() : *this;
    }
    Tensor t(impl_->sizes(), dtype(), device);
    t.copy_(*this, non_blocking);
    return t;
}

Tensor Tensor::to(Device device, DType dtype, bool non_blocking, bool copy) const {
    if (!impl_) return Tensor();
    if (is_sparse()) {
        if (this->device() == device && this->dtype() == dtype) {
            return copy ? clone() : *this;
        }
        return make_sparse_coo_tensor(
            _indices().to(device, non_blocking, copy),
            _values().to(device, dtype, non_blocking, copy),
            static_cast<std::vector<int64_t>>(shape()), is_coalesced());
    }
    if (this->device() == device && this->dtype() == dtype) {
        return copy ? clone() : *this;
    }
    Tensor t(impl_->sizes(), dtype, device);
    t.copy_(*this, non_blocking);
    return t;
}



Tensor Tensor::contiguous() const {
    if (is_sparse()) return clone();
    if (is_contiguous()) return *this;
    return clone();
}

} // namespace tensorplay
