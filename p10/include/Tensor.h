#pragma once

#include <memory>
#include <vector>
#include <optional>
#include <string>
#include <cstring>
#include "Macros.h"
#include "DType.h"
#include "Device.h"
#include "Generator.h"
#include "Scalar.h"
#include "SymInt.h"
#include "SymBool.h"
#include "SymFloat.h"
#include "TensorImpl.h"
#include "Dispatcher.h"
#include "Exception.h"

namespace tensorplay {

class P10_API Size {
public:
    using iterator = std::vector<int64_t>::iterator;
    using const_iterator = std::vector<int64_t>::const_iterator;
    using value_type = int64_t;

    Size() = default;
    Size(std::vector<int64_t> sizes) : sizes_(std::move(sizes)) {}
    Size(std::initializer_list<int64_t> sizes) : sizes_(sizes) {}
    
    // Implicit conversion to std::vector<int64_t> for backward compatibility
    operator std::vector<int64_t>() const { return sizes_; }
    
    int64_t operator[](size_t index) const { return sizes_[index]; }
    int64_t& operator[](size_t index) { return sizes_[index]; }
    
    size_t size() const { return sizes_.size(); }
    bool empty() const { return sizes_.empty(); }
    const int64_t* data() const { return sizes_.data(); }
    
    const_iterator begin() const { return sizes_.begin(); }
    const_iterator end() const { return sizes_.end(); }
    iterator begin() { return sizes_.begin(); }
    iterator end() { return sizes_.end(); }
    
    bool operator==(const Size& other) const { return sizes_ == other.sizes_; }
    bool operator!=(const Size& other) const { return sizes_ != other.sizes_; }
    
    std::string toString() const;

private:
    std::vector<int64_t> sizes_;
};

P10_API std::ostream& operator<<(std::ostream& os, const Size& s);

// Helper for value casting
template <typename Target, typename Source>
inline Target cast_value(const Source& src) {
    if constexpr (std::is_same_v<Source, Scalar>) {
        return src.template to<Target>();
    } else if constexpr (is_complex_type_v<Target>) {
        using target_value_t = typename is_complex_type<Target>::value_type;
        if constexpr (is_complex_type_v<Source>) {
            return Target(static_cast<target_value_t>(src.real()),
                          static_cast<target_value_t>(src.imag()));
        } else {
            return Target(static_cast<target_value_t>(src), target_value_t(0));
        }
    } else if constexpr (is_complex_type_v<Source>) {
        // Route through double so reduced-precision element types (e.g.
        // Float16) without direct integral conversions still compile.
        return static_cast<Target>(static_cast<double>(src.real()));
    } else {
        return static_cast<Target>(src);
    }
}
 
class P10_API Tensor {
private:
    std::shared_ptr<TensorImpl> impl_;

public:
    // Constructors
    Tensor() = default;
    
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
    
    Tensor(const std::vector<int64_t>& sizes, DType dtype, const Device& device = Device());
    
    // Constructor from Storage (for advanced usage)
    Tensor(Storage storage, const std::vector<int64_t>& sizes, DType dtype);
    Tensor(Storage storage, const std::vector<int64_t>& sizes, const std::vector<int64_t>& strides, DType dtype, size_t storage_offset = 0);

    // Utils
    bool defined() const { return impl_ != nullptr; }
    std::shared_ptr<TensorImpl> impl() const { return impl_; }
    void swap_impl(Tensor& other) noexcept { impl_.swap(other.impl_); }
    
    // Constructor with Scalar fill value
    Tensor(const std::vector<int64_t>& sizes, Scalar fill_value, const Device& device = Device());

    template <typename T>
    static Tensor tensor(const std::vector<T>& data, std::optional<DType> dtype = std::nullopt, const Device& device = Device(DeviceType::CPU)) {
        DType inferred_dtype = dtype.value_or(TypeTraits<T>::dtype);
        if (inferred_dtype == DType::Undefined) {
             TP_THROW(RuntimeError, "Could not infer dtype from C++ type");
        }
        
        std::vector<int64_t> size = {static_cast<int64_t>(data.size())};
        Tensor t = Tensor(size, inferred_dtype, device);
        
        // Copy data
        if (device.is_cpu()) {
            if (inferred_dtype == TypeTraits<T>::dtype) {
                 std::memcpy(t.data_ptr(), data.data(), data.size() * sizeof(T));
            } else {
                 // Cast through the same complete dtype list used by CPU
                 // copy/fill dispatch.  This keeps the C++ tensor factory in
                 // sync with the Python factory for unsigned, reduced-float,
                 // and complex dtypes.
                 #define COPY_CASE(ctype, name) \
                     case DType::name: { \
                         auto* ptr = t.data_ptr<ctype>(); \
                         for (size_t i = 0; i < data.size(); ++i) { \
                             ptr[i] = cast_value<ctype>(data[i]); \
                         } \
                         break; \
                     }
                 switch (inferred_dtype) {
                     TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(COPY_CASE)
                     default:
                         TP_THROW(NotImplementedError, "Type conversion in tensor() not fully implemented for this dtype");
                 }
                 #undef COPY_CASE
            }
        } else {
             TP_THROW(NotImplementedError, "tensor() currently only supports CPU");
        }
        
        return t;
    }
    
    template <typename T>
    static Tensor tensor(std::initializer_list<T> data, std::optional<DType> dtype = std::nullopt, const Device& device = Device(DeviceType::CPU)) {
        return tensor(std::vector<T>(data), dtype, device);
    }

    // Accessors
    int64_t dim() const;
    int64_t numel() const;
    Size shape() const;
    std::vector<int64_t> strides() const;
    int64_t size(int64_t dim) const;
    int64_t stride(int64_t dim) const;
    
    DType dtype() const;
    Device device() const;
    size_t itemsize() const;
    bool is_contiguous() const;
    bool is_contiguous(MemoryFormat format) const;
    MemoryFormat memory_format() const;
    bool is_channels_last() const;
    bool is_channels_last_2d() const;
    bool is_channels_last_3d() const;
    DispatchKeySet key_set() const { return impl_ ? impl_->key_set() : DispatchKeySet(); }
    bool is_batched() const { return impl_ && impl_->is_batched(); }
    int64_t batch_dim() const { return is_batched() ? impl_->batch_dim() : -1; }
    int64_t batch_level() const { return is_batched() ? impl_->batch_level() : -1; }
    int64_t batch_size() const { return is_batched() ? impl_->batch_size() : 0; }
    Tensor transform_value() const;
    bool is_pinned() const;
    Tensor pin_memory() const;
    // Tensor contiguous() const; // Generated
    // Materializes *this with the canonical strides of `format` (NHWC /
    // NDHWC); returns *this unchanged when it already matches. Preserve
    // resolves to plain contiguous().
    // Tensor contiguous(MemoryFormat format) const; // Generated
    bool is_sparse() const;
    bool is_coalesced() const;
    int64_t sparse_dim() const;
    int64_t dense_dim() const;
    Tensor _indices() const;
    Tensor _values() const;
    // CSR layout accessors (2D); throw for COO/dense tensors.
    Tensor _crow_indices() const;
    Tensor _col_indices() const;
    bool is_sparse_csr() const;
    Tensor coalesce() const;
    Tensor sparse_mask(const Tensor& mask) const;

    // Internal constructor used by the native sparse COO factory and
    // sparse embedding backward.  The public Python surface is generated
    // from sparse_coo_tensor(...) below; keeping the representation builder
    // here avoids routing component tensors through a dense temporary.
    static Tensor make_sparse_coo_tensor(const Tensor& indices,
                                         const Tensor& values,
                                         const std::vector<int64_t>& size,
                                         bool is_coalesced = false);
    // Internal CSR constructor used by to_sparse_csr; it installs crow/col
    // components alongside the values tensor.
    static Tensor make_sparse_csr_tensor(const Tensor& crow,
                                         const Tensor& col,
                                         const Tensor& values,
                                         const std::vector<int64_t>& size);
    
    // Autograd methods (delegated to the AutogradMeta extension point on
    // TensorImpl; the concrete implementation lives in the tpx library).
    bool requires_grad() const;
    void set_requires_grad(bool requires_grad);
    Tensor grad() const;
    void set_grad(const Tensor& grad);
    bool retains_grad() const;
    void set_retains_grad(bool retains_grad);
    Tensor detach() const;

    // Data access
    template<typename T>
    T* data_ptr() const {
        if (is_sparse()) {
            TP_THROW(RuntimeError, "data_ptr() is not supported for sparse COO tensors");
        }
        return impl_ ? impl_->data<T>() : nullptr;
    }
    
    void* data_ptr() const;
    
    std::shared_ptr<TensorImpl> unsafeGetTensorImpl() const { return impl_; }

    // Operators / Methods
    std::string toString() const;
    
    // View methods
    Tensor view(const std::vector<int64_t>& shape) const;
    Tensor view_dtype(DType dtype) const;
    // Tensor reshape(const std::vector<int64_t>& shape) const; // Generated
    Tensor as_strided(const std::vector<int64_t>& size,
                     const std::vector<int64_t>& stride,
                     std::optional<int64_t> storage_offset = std::nullopt) const;
    Tensor select(int64_t dim, int64_t index) const;
    Tensor slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1) const;
    // Tensor expand(const std::vector<int64_t>& size) const; // Generated
    
    // Tensor transpose(int64_t dim0, int64_t dim1) const; // Generated
    // Tensor t() const; // Generated
    // Tensor permute(const std::vector<int64_t>& dims) const; // Generated
    // Tensor squeeze() const; // Generated
    // Tensor squeeze(int64_t dim) const; // Generated
    // Tensor unsqueeze(int64_t dim) const; // Generated

    // std::vector<Tensor> unbind(int64_t dim = 0) const; // Generated

    // std::vector<Tensor> split(int64_t split_size, int64_t dim = 0) const; // Generated
    // std::vector<Tensor> split(const std::vector<int64_t>& split_sizes, int64_t dim = 0) const; // Generated
    // std::vector<Tensor> chunk(int64_t chunks, int64_t dim = 0) const; // Generated

    // Modification
    // Tensor& copy_(const Tensor& src);

    
    // Scalar access
    Scalar item() const;

    template<typename T>
    T item() const {
        return item().to<T>();
    }
    
    // Clone
    // Tensor clone() const; // Generated

    // Factories (static)
    // static Tensor empty(const std::vector<int64_t>& size, DType dtype = DType::Float32, Device device = Device(DeviceType::CPU)); // Generated
    // static Tensor full(const std::vector<int64_t>& size, Scalar fill_value, DType dtype = DType::Undefined, Device device = Device(DeviceType::CPU)); // Generated
    // static Tensor zeros(const std::vector<int64_t>& size, DType dtype = DType::Float32, Device device = Device(DeviceType::CPU)); // Generated
    // static Tensor ones(const std::vector<int64_t>& size, DType dtype = DType::Float32, Device device = Device(DeviceType::CPU)); // Generated
    // static Tensor eye(int64_t n, int64_t m = -1, DType dtype = DType::Float32, Device device = Device(DeviceType::CPU)); // Generated
    // static Tensor arange(Scalar start, Scalar end, Scalar step = Scalar(1), DType dtype = DType::Undefined, Device device = Device(DeviceType::CPU)); // Generated
    // static Tensor arange(Scalar end, DType dtype = DType::Undefined, Device device = Device(DeviceType::CPU)); // Generated
    // static Tensor linspace(Scalar start, Scalar end, int64_t steps, DType dtype = DType::Float32, Device device = Device(DeviceType::CPU)); // Generated
    // static Tensor logspace(Scalar start, Scalar end, int64_t steps, double base = 10.0, DType dtype = DType::Float32, Device device = Device(DeviceType::CPU)); // Generated
    // static Tensor rand(const std::vector<int64_t>& size, DType dtype = DType::Float32, Device device = Device(DeviceType::CPU)); // Generated
    
    // static Tensor cat(const std::vector<Tensor>& tensors, int64_t dim = 0); // Generated
    // static Tensor stack(const std::vector<Tensor>& tensors, int64_t dim = 0); // Generated // Generated

    // *_like factories
    // static Tensor empty_like(const Tensor& input, std::optional<DType> dtype = std::nullopt, std::optional<Device> device = std::nullopt); // Generated
    // static Tensor full_like(const Tensor& input, Scalar fill_value, std::optional<DType> dtype = std::nullopt, std::optional<Device> device = std::nullopt); // Generated
    // static Tensor zeros_like(const Tensor& input, std::optional<DType> dtype = std::nullopt, std::optional<Device> device = std::nullopt); // Generated
    // static Tensor ones_like(const Tensor& input, std::optional<DType> dtype = std::nullopt, std::optional<Device> device = std::nullopt); // Generated

    // Explicit arithmetic methods
    // Tensor add(const Tensor& other, Scalar alpha = Scalar(1)) const; // Generated
    // Tensor sub(const Tensor& other, Scalar alpha = Scalar(1)) const; // Generated
    // Tensor mul(const Tensor& other) const; // Generated
    // Tensor div(const Tensor& other) const; // Generated
    // Tensor mm(const Tensor& other) const; // Generated
    // Tensor matmul(const Tensor& other) const;
    
    // Type conversion
    Tensor to(DType dtype, bool non_blocking = false, bool copy = false) const;
    Tensor to(Device device, DType dtype, bool non_blocking = false, bool copy = false) const;
    Tensor to(Device device, bool non_blocking = false, bool copy = false) const;

    // Arithmetic Operators
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor operator+(Scalar other) const;
    Tensor operator-(Scalar other) const;
    Tensor operator*(Scalar other) const;
    Tensor operator/(Scalar other) const;

    // In-place operators
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    
    Tensor& operator+=(Scalar other);
    Tensor& operator-=(Scalar other);
    Tensor& operator*=(Scalar other);
    Tensor& operator/=(Scalar other);

    // Generated methods
    #include "tensorplay/ops/TensorGenerated.h"
};

// Unary operators
P10_API Tensor operator-(const Tensor& t);

// Dispatcher-level implementations shared by the backend kernels; the
// clone/contiguous Tensor members are generated and route here through the
// dispatcher.
namespace detail {
P10_API Tensor clone_impl(const Tensor& self,
                          std::optional<MemoryFormat> memory_format = std::nullopt);
P10_API Tensor contiguous_impl(const Tensor& self, int64_t memory_format);
// Fresh row-major copy for internal kernels that do flat/contiguous pointer
// arithmetic: unlike clone() (which preserves non-overlapping-and-dense
// inputs), this always materializes a new contiguous buffer.
P10_API Tensor contiguous_clone(const Tensor& self);
} // namespace detail

// Global operators for Scalar first
P10_API inline Tensor operator+(Scalar s, const Tensor& t) { return t + s; }
P10_API inline Tensor operator-(Scalar s, const Tensor& t) { return (-t) + s; } // unary minus needed
P10_API inline Tensor operator*(Scalar s, const Tensor& t) { return t * s; }
// inline Tensor operator/(Scalar s, const Tensor& t) { ... } // Need specialized impl

P10_API std::ostream& operator<<(std::ostream& os, const Tensor& t);

inline DispatchKey dispatchKeyForTensor(const Tensor& tensor) {
    const DispatchKey key = tensor.key_set().highest_priority_key();
    return key == DispatchKey::EndOfKeys ? computeDispatchKey(tensor.device()) : key;
}

inline DispatchKey dispatchKeyForTensorArg(const Tensor& tensor) {
    return dispatchKeyForTensor(tensor);
}

inline DispatchKey dispatchKeyForTensorArg(const std::optional<Tensor>& tensor) {
    return tensor.has_value() ? dispatchKeyForTensor(*tensor) : DispatchKey::EndOfKeys;
}

inline DispatchKey dispatchKeyForTensorArg(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) return DispatchKey::EndOfKeys;
    DispatchKey result = DispatchKey::EndOfKeys;
    for (const Tensor& tensor : tensors) {
        const DispatchKey key = dispatchKeyForTensor(tensor);
        if (result == DispatchKey::EndOfKeys ||
            static_cast<uint8_t>(key) > static_cast<uint8_t>(result)) {
            result = key;
        }
    }
    return result;
}

inline DispatchKey dispatchKeyForTensorArg(
    const std::vector<std::optional<Tensor>>& tensors) {
    DispatchKey result = DispatchKey::EndOfKeys;
    for (const auto& tensor : tensors) {
        if (!tensor.has_value()) continue;
        const DispatchKey key = dispatchKeyForTensor(*tensor);
        if (result == DispatchKey::EndOfKeys ||
            static_cast<uint8_t>(key) > static_cast<uint8_t>(result)) {
            result = key;
        }
    }
    return result;
}

inline Device deviceForTensorArg(const std::vector<Tensor>& tensors) {
    for (const auto& tensor : tensors) return tensor.device();
    return Device(DeviceType::CPU);
}

inline Device deviceForTensorArg(
    const std::vector<std::optional<Tensor>>& tensors) {
    for (const auto& tensor : tensors) {
        if (tensor.has_value()) return tensor->device();
    }
    return Device(DeviceType::CPU);
}

// Non-tensor arguments carry no dispatch key of their own; they are simply
// ignored when deriving a key from an argument pack.
template <typename T>
inline DispatchKey dispatchKeyForTensorArg(const T&) {
    return DispatchKey::EndOfKeys;
}

template <typename... Args>
inline DispatchKey dispatchKeyForTensorArgs(const Args&... args) {
    DispatchKey result = DispatchKey::EndOfKeys;
    const auto consider = [&result](DispatchKey key) {
        if (key == DispatchKey::EndOfKeys) return;
        if (result == DispatchKey::EndOfKeys ||
            static_cast<uint8_t>(key) > static_cast<uint8_t>(result)) {
            result = key;
        }
    };
    (consider(dispatchKeyForTensorArg(args)), ...);
    return result;
}

P10_API void set_printoptions(int64_t edge_items = -1, int64_t threshold = -1, int64_t precision = -1, int64_t linewidth = -1);

} // namespace tensorplay
