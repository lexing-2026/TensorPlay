 #include <iomanip>
#include <type_traits>
#include "Tensor.h"
#include "TensorImpl.h"
#include "Storage.h"
#include "Utils.h"
#include <iostream>
#include <cstring>
#include <sstream>

namespace tensorplay {

// Helper for DType output
std::ostream& operator<<(std::ostream& os, DType dt) {
    switch (dt) {
        case DType::Float32: os << "float32"; break;
        case DType::Float64: os << "float64"; break;
        case DType::Int32: os << "int32"; break;
        case DType::Int64: os << "int64"; break;
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

int64_t Tensor::dim() const { return impl_ ? impl_->dim() : 0; }
int64_t Tensor::numel() const { return impl_ ? impl_->numel() : 0; }
Size Tensor::shape() const { return impl_ ? Size(impl_->sizes()) : Size(); }
std::vector<int64_t> Tensor::strides() const { return impl_ ? impl_->strides() : std::vector<int64_t>(); }
int64_t Tensor::size(int64_t dim) const {
    if (!impl_) return 0;
    if (dim < 0) dim += impl_->dim();
    return impl_->size(dim);
}
int64_t Tensor::stride(int64_t dim) const {
    if (!impl_) return 0;
    if (dim < 0) dim += impl_->dim();
    return impl_->stride(dim);
}
DType Tensor::dtype() const { return impl_ ? impl_->dtype() : DType::Undefined; }
Device Tensor::device() const { return impl_ ? impl_->device() : Device(DeviceType::CPU); }
size_t Tensor::itemsize() const { return impl_ ? impl_->itemsize() : 0; }
bool Tensor::is_contiguous() const { return impl_ ? impl_->is_contiguous() : false; }

bool Tensor::is_sparse() const { return false; }

void* Tensor::data_ptr() const { return impl_ ? impl_->data() : nullptr; }

Scalar Tensor::item() const {
    if (numel() != 1) {
        TP_THROW(ValueError, "item() only supported for 1-element tensors");
    }

    if (device().type() != DeviceType::CPU) {
        return to(Device(DeviceType::CPU)).item();
    }
    
    switch (dtype()) {
        case DType::Float32: return Scalar(static_cast<double>(*data_ptr<float>()));
        case DType::Float64: return Scalar(*data_ptr<double>());
        case DType::Int32: return Scalar(static_cast<int64_t>(*data_ptr<int32_t>()));
        case DType::Int64: return Scalar(*data_ptr<int64_t>());
        case DType::Bool: return Scalar(static_cast<bool>(*data_ptr<bool>()));
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
    return Tensor(impl_->storage(), size, stride, impl_->dtype(), offset);
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
    Tensor t = Tensor::full({}, other, other.dtype(), device());
    return add(t);
}
Tensor Tensor::operator-(Scalar other) const {
    Tensor t = Tensor::full({}, other, other.dtype(), device());
    return sub(t);
}
Tensor Tensor::operator*(Scalar other) const {
    Tensor t = Tensor::full({}, other, other.dtype(), device());
    return mul(t);
}
Tensor Tensor::operator/(Scalar other) const {
    Tensor t = Tensor::full({}, other, other.dtype(), device());
    return div(t);
}

Tensor& Tensor::operator+=(const Tensor& other) { return add_(other); }
Tensor& Tensor::operator-=(const Tensor& other) { return sub_(other); }
Tensor& Tensor::operator*=(const Tensor& other) { return mul_(other); }
Tensor& Tensor::operator/=(const Tensor& other) { return div_(other); }

Tensor& Tensor::operator+=(Scalar other) {
    Tensor t = Tensor::full({}, other, other.dtype(), device());
    return add_(t);
}
Tensor& Tensor::operator-=(Scalar other) {
    Tensor t = Tensor::full({}, other, other.dtype(), device());
    return sub_(t);
}
Tensor& Tensor::operator*=(Scalar other) {
    Tensor t = Tensor::full({}, other, other.dtype(), device());
    return mul_(t);
}
Tensor& Tensor::operator/=(Scalar other) {
    Tensor t = Tensor::full({}, other, other.dtype(), device());
    return div_(t);
}

Tensor Tensor::clone() const {
    if (!impl_) return Tensor();
    Tensor t(impl_->sizes(), dtype(), device());
    t.copy_(*this);
    return t;
}

Tensor Tensor::to(DType dtype, bool non_blocking, bool copy) const {
    if (!impl_) return Tensor();
    if (this->dtype() == dtype) {
        return copy ? clone() : *this;
    }
    Tensor t(impl_->sizes(), dtype, device());
    t.copy_(*this);
    return t;
}

Tensor Tensor::to(Device device, bool non_blocking, bool copy) const {
    if (!impl_) return Tensor();
    if (this->device() == device) {
        return copy ? clone() : *this;
    }
    Tensor t(impl_->sizes(), dtype(), device);
    t.copy_(*this);
    return t;
}

Tensor Tensor::to(Device device, DType dtype, bool non_blocking, bool copy) const {
    if (!impl_) return Tensor();
    if (this->device() == device && this->dtype() == dtype) {
        return copy ? clone() : *this;
    }
    Tensor t(impl_->sizes(), dtype, device);
    t.copy_(*this);
    return t;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    return clone();
}

} // namespace tensorplay
