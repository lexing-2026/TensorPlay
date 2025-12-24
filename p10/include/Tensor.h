#pragma once

#include <memory>
#include <vector>
#include <optional>
#include <string>
#include <cstring>
#include "Macros.h"
#include "DType.h"
#include "Device.h"
#include "Scalar.h"
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
    
    // Constructor with Scalar fill value
    Tensor(const std::vector<int64_t>& sizes, Scalar fill_value, const Device& device = Device());

    // C++ Factory Methods (mirroring torch::tensor)
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
                 // Simple cast copy loop
                 // Note: Ideally use a dispatcher/kernel
                 if (inferred_dtype == DType::Float32) {
                     float* ptr = t.data_ptr<float>();
                     for(size_t i=0; i<data.size(); ++i) ptr[i] = cast_value<float>(data[i]);
                 } else if (inferred_dtype == DType::Int64) {
                     int64_t* ptr = t.data_ptr<int64_t>();
                     for(size_t i=0; i<data.size(); ++i) ptr[i] = cast_value<int64_t>(data[i]);
                 } else if (inferred_dtype == DType::Int32) {
                     int32_t* ptr = t.data_ptr<int32_t>();
                     for(size_t i=0; i<data.size(); ++i) ptr[i] = cast_value<int32_t>(data[i]);
                 } else if (inferred_dtype == DType::Float64) {
                     double* ptr = t.data_ptr<double>();
                     for(size_t i=0; i<data.size(); ++i) ptr[i] = cast_value<double>(data[i]);
                 } else if (inferred_dtype == DType::Bool) {
                     bool* ptr = t.data_ptr<bool>();
                     for(size_t i=0; i<data.size(); ++i) ptr[i] = cast_value<bool>(data[i]);
                 } else {
                     TP_THROW(NotImplementedError, "Type conversion in tensor() not fully implemented for this dtype");
                 }
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
    Tensor contiguous() const;
    bool is_sparse() const;
    
    // Autograd methods removed from P10
    // bool requires_grad() const;
    // void set_requires_grad(bool r);
    // Tensor grad() const;
    // void set_grad(const Tensor& grad);
    // void retain_grad();
    // bool is_leaf() const;
    // Tensor detach() const;
    
    // void set_grad_fn(std::shared_ptr<Node> grad_fn);
    // std::shared_ptr<Node> grad_fn() const;
    
    // Data access
    template<typename T>
    T* data_ptr() const { return impl_ ? impl_->data<T>() : nullptr; }
    
    void* data_ptr() const;
    
    std::shared_ptr<TensorImpl> unsafeGetTensorImpl() const { return impl_; }

    // Operators / Methods
    std::string toString() const;
    
    // View methods
    // Tensor view(const std::vector<int64_t>& shape) const; // Generated
    // Tensor reshape(const std::vector<int64_t>& shape) const; // Generated
    Tensor as_strided(const std::vector<int64_t>& size, const std::vector<int64_t>& stride, std::optional<int64_t> storage_offset = std::nullopt) const;
    Tensor select(int64_t dim, int64_t index) const;
    Tensor slice(int64_t dim, int64_t start, int64_t end, int64_t step = 1) const;
    Tensor expand(const std::vector<int64_t>& size) const;
    
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
    // Tensor& fill_(Scalar value);
    
    // Scalar access
    Scalar item() const;

    template<typename T>
    T item() const {
        return item().to<T>();
    }
    
    // Clone
    Tensor clone() const;

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

// Global operators for Scalar first
P10_API inline Tensor operator+(Scalar s, const Tensor& t) { return t + s; }
P10_API inline Tensor operator-(Scalar s, const Tensor& t) { return (-t) + s; } // unary minus needed
P10_API inline Tensor operator*(Scalar s, const Tensor& t) { return t * s; }
// inline Tensor operator/(Scalar s, const Tensor& t) { ... } // Need specialized impl

P10_API std::ostream& operator<<(std::ostream& os, const Tensor& t);

P10_API void set_printoptions(int64_t edge_items = -1, int64_t threshold = -1, int64_t precision = -1, int64_t linewidth = -1);

} // namespace tensorplay
