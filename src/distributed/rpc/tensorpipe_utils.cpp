#include "tensorpipe_utils.h"

#include "Allocator.h"
#include "Tensor.h"
#include "TensorImpl.h"
#include "agent_utils.h"
#include "python_functions.h"

#include <pybind11/stl.h>

#ifdef USE_CUDA
#include "CUDARuntime.h"
#include <tensorpipe/tensorpipe_cuda.h>
#endif

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace tensorplay::distributed::rpc {
namespace {

tensorpipe::Device to_tensorpipe_device(const tensorplay::Device& device) {
    if (device.is_cpu()) {
        return tensorpipe::Device(tensorpipe::kCpuDeviceType, 0);
    }
    if (device.is_cuda()) {
        return tensorpipe::Device(
            tensorpipe::kCudaDeviceType,
            static_cast<int>(device.index() < 0 ? 0 : device.index()));
    }
    throw std::runtime_error("RPC tensor device is not supported");
}

tensorplay::Device from_tensorpipe_device(const tensorpipe::Device& device) {
    if (device.type == tensorpipe::kCpuDeviceType) {
        return tensorplay::Device(tensorplay::DeviceType::CPU);
    }
    if (device.type == tensorpipe::kCudaDeviceType) {
#ifdef USE_CUDA
        return tensorplay::Device(
            tensorplay::DeviceType::CUDA,
            static_cast<int64_t>(device.index));
#else
        throw std::runtime_error("RPC received a CUDA tensor in a CPU build");
#endif
    }
    throw std::runtime_error("RPC tensor device is not supported");
}

tensorplay::Device mapped_device(
    const tensorplay::Device& source,
    const DeviceMap& device_map) {
    const auto source_key = source.toString();
    auto entry = device_map.find(source_key);
    if (entry == device_map.end() && source.is_cuda()) {
        entry = device_map.find("cuda");
    }
    if (entry == device_map.end()) {
        if (source.is_cpu()) {
            return source;
        }
        throw std::runtime_error(
            "RPC device mapping is required for non-CPU tensors");
    }
    return tensorplay::Device(entry->second);
}

tensorpipe::Buffer make_tensorpipe_buffer(
    const tensorplay::Storage& storage,
    const tensorplay::Device& device) {
    if (device.is_cpu()) {
        return tensorpipe::CpuBuffer{storage.data()};
    }
#ifdef USE_CUDA
    if (device.is_cuda()) {
        tensorplay::cuda::CUDAGuard guard(static_cast<int>(device.index()));
        return tensorpipe::CudaBuffer{
            storage.data(),
            tensorplay::cuda::getCurrentCUDAStream(
                static_cast<int>(device.index())).stream()};
    }
#endif
    throw std::runtime_error("RPC tensor buffer device is not supported");
}

std::vector<int64_t> shape_values(const tensorplay::Tensor& tensor) {
    const auto shape = tensor.shape();
    return std::vector<int64_t>(shape.begin(), shape.end());
}

struct TensorEncoder final {
    TensorPipeWriteState& state;
    const DeviceMap& device_map;

    size_t storage_index(const tensorplay::Tensor& tensor) {
        const auto impl = tensor.unsafeGetTensorImpl();
        if (!impl || !impl->has_storage()) {
            throw std::runtime_error("RPC tensor has no transferable storage");
        }
        const tensorplay::Storage storage = impl->storage();
        for (size_t index = 0; index < state.storages.size(); ++index) {
            if (state.storages[index].is_same(storage)) {
                return index;
            }
        }
        const size_t index = state.storages.size();
        state.storages.push_back(storage);
        tensorpipe::Message::Tensor buffer;
        buffer.buffer = make_tensorpipe_buffer(storage, tensor.device());
        buffer.length = storage.nbytes();
        buffer.targetDevice = to_tensorpipe_device(mapped_device(
            tensor.device(), device_map));
        buffer.metadata = std::to_string(index);
        state.message.tensors.push_back(std::move(buffer));
        return index;
    }

    py::object encode(const tensorplay::Tensor& tensor) {
        if (!tensor.defined()) {
            return py::make_tuple(py::str("undefined"));
        }
        if (tensor.is_sparse_csr()) {
            return py::make_tuple(
                py::str("csr"),
                encode(tensor._crow_indices()),
                encode(tensor._col_indices()),
                encode(tensor._values()),
                py::cast(shape_values(tensor)),
                py::cast(tensor.requires_grad()));
        }
        if (tensor.is_sparse()) {
            return py::make_tuple(
                py::str("coo"),
                encode(tensor._indices()),
                encode(tensor._values()),
                py::cast(shape_values(tensor)),
                py::cast(tensor.is_coalesced()),
                py::cast(tensor.requires_grad()));
        }
        const auto impl = tensor.unsafeGetTensorImpl();
        const size_t index = storage_index(tensor);
        return py::make_tuple(
            py::str("dense"),
            py::cast(index),
            py::cast(shape_values(tensor)),
            py::cast(tensor.strides()),
            py::cast(static_cast<int>(tensor.dtype())),
            py::cast(static_cast<int>(tensor.device().type())),
            py::cast(tensor.device().index()),
            py::cast(static_cast<uint64_t>(impl->storage_offset())),
            py::cast(tensor.requires_grad()));
    }
};

struct TensorDecoder final {
    const TensorPipeReadState& state;

    const tensorplay::Storage& storage(size_t index) const {
        if (index >= state.storages.size()) {
            throw std::runtime_error("RPC tensor storage index is out of range");
        }
        return state.storages[index];
    }

    tensorplay::Tensor decode(const py::handle& value) const {
        const py::tuple record = value.cast<py::tuple>();
        if (record.empty()) {
            throw std::runtime_error("RPC tensor metadata is empty");
        }
        const std::string kind = record[0].cast<std::string>();
        if (kind == "undefined") {
            if (record.size() != 1) {
                throw std::runtime_error("RPC undefined tensor metadata is invalid");
            }
            return tensorplay::Tensor();
        }
        if (kind == "dense") {
            if (record.size() != 9) {
                throw std::runtime_error("RPC dense tensor metadata is invalid");
            }
            const size_t storage_id = record[1].cast<size_t>();
            const auto shape = record[2].cast<std::vector<int64_t>>();
            const auto strides = record[3].cast<std::vector<int64_t>>();
            const auto dtype = static_cast<tensorplay::DType>(
                record[4].cast<int>());
            const auto offset = record[7].cast<uint64_t>();
            if (offset > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("RPC tensor storage offset is out of range");
            }
            tensorplay::Tensor result(
                storage(storage_id),
                shape,
                strides,
                dtype,
                static_cast<size_t>(offset));
            result.set_requires_grad(record[8].cast<bool>());
            return result;
        }
        if (kind == "coo") {
            if (record.size() != 6) {
                throw std::runtime_error("RPC sparse COO metadata is invalid");
            }
            tensorplay::Tensor result = tensorplay::Tensor::make_sparse_coo_tensor(
                decode(record[1]),
                decode(record[2]),
                record[3].cast<std::vector<int64_t>>(),
                record[4].cast<bool>());
            result.set_requires_grad(record[5].cast<bool>());
            return result;
        }
        if (kind == "csr") {
            if (record.size() != 6) {
                throw std::runtime_error("RPC sparse CSR metadata is invalid");
            }
            tensorplay::Tensor result = tensorplay::Tensor::make_sparse_csr_tensor(
                decode(record[1]),
                decode(record[2]),
                decode(record[3]),
                record[4].cast<std::vector<int64_t>>());
            result.set_requires_grad(record[5].cast<bool>());
            return result;
        }
        throw std::runtime_error("RPC tensor metadata has an unknown layout");
    }
};

}  // namespace

Endpoint parse_endpoint(const std::string& value, uint16_t default_port) {
    if (value.empty()) {
        return {guess_address(), default_port};
    }
    std::string host;
    std::string port_text;
    if (value.front() == '[') {
        const auto close = value.find(']');
        if (close == std::string::npos) {
            throw std::invalid_argument("endpoint has an unterminated IPv6 host");
        }
        host = value.substr(1, close - 1);
        if (close + 1 < value.size()) {
            if (value[close + 1] != ':') {
                throw std::invalid_argument("endpoint has invalid IPv6 syntax");
            }
            port_text = value.substr(close + 2);
        }
    } else {
        const auto first = value.find(':');
        const auto last = value.rfind(':');
        if (first != std::string::npos && first == last) {
            host = value.substr(0, first);
            port_text = value.substr(first + 1);
        } else {
            host = value;
        }
    }
    if (host.empty()) {
        throw std::invalid_argument("endpoint host is empty");
    }
    if (port_text.empty()) {
        return {std::move(host), default_port};
    }
    size_t consumed = 0;
    const auto port = std::stoll(port_text, &consumed);
    if (consumed != port_text.size() || port < 0 || port > 65535) {
        throw std::invalid_argument("endpoint port is out of range");
    }
    return {std::move(host), static_cast<uint16_t>(port)};
}

std::string guess_address() {
    const std::string interface_name = environment_value("TP_SOCKET_IFNAME");
    if (!interface_name.empty()) {
        auto [error, address] =
            tensorpipe::transport::uv::lookupAddrForIface(interface_name);
        if (!error && !address.empty()) {
            return address;
        }
    }
    auto [error, address] = tensorpipe::transport::uv::lookupAddrForHostname();
    if (!error && !address.empty()) {
        return address;
    }
    return "127.0.0.1";
}

std::shared_ptr<TensorPipeWriteState> make_tensorpipe_message(
    const Message& message,
    const DeviceMap& device_map) {
    auto state = std::make_shared<TensorPipeWriteState>();
    state->type = std::make_unique<MessageType>(message.type());
    state->id = std::make_unique<int64_t>(message.id());
    state->payload.assign(message.payload().begin(), message.payload().end());
    state->storages.reserve(message.tensors().size());
    state->message.tensors.reserve(message.tensors().size());
    py::list tensor_records;
    TensorEncoder encoder{*state, device_map};
    for (const auto& value : message.tensors()) {
        const auto tensor = py::cast<tensorplay::Tensor>(value);
        tensor_records.append(encoder.encode(tensor));
    }
    SerializedPyObj metadata = serialize_python_object(tensor_records);
    if (!metadata.tensors_.empty()) {
        throw std::runtime_error("RPC tensor metadata contains nested tensors");
    }
    state->tensor_metadata = std::move(metadata.payload_);
    state->message.payloads.reserve(4);
    state->message.payloads.push_back(
        {state->type.get(), sizeof(MessageType), std::string()});
    state->message.payloads.push_back(
        {state->id.get(), sizeof(int64_t), std::string()});
    state->message.payloads.push_back(
        {state->payload.empty() ? nullptr : state->payload.data(),
         state->payload.size(),
         std::string()});
    state->message.payloads.push_back(
        {state->tensor_metadata.empty() ? nullptr : state->tensor_metadata.data(),
         state->tensor_metadata.size(),
         std::string()});
    return state;
}

TensorPipeReadAllocation allocate_tensorpipe_message(
    const tensorpipe::Descriptor& descriptor) {
    if (descriptor.payloads.size() != 4 ||
        descriptor.payloads[0].length != sizeof(MessageType) ||
        descriptor.payloads[1].length != sizeof(int64_t)) {
        throw std::runtime_error("RPC TensorPipe descriptor has invalid payloads");
    }
    auto state = std::make_shared<TensorPipeReadState>();
    state->type = std::make_unique<MessageType>();
    state->id = std::make_unique<int64_t>();
    state->payload.resize(descriptor.payloads[2].length);
    state->tensor_metadata.resize(descriptor.payloads[3].length);
    state->storages.reserve(descriptor.tensors.size());

    tensorpipe::Allocation allocation;
    allocation.payloads.reserve(4);
    allocation.payloads.push_back({state->type.get()});
    allocation.payloads.push_back({state->id.get()});
    allocation.payloads.push_back({
        state->payload.empty() ? nullptr : state->payload.data()});
    allocation.payloads.push_back({
        state->tensor_metadata.empty() ? nullptr : state->tensor_metadata.data()});
    allocation.tensors.reserve(descriptor.tensors.size());

    for (const auto& descriptor_tensor : descriptor.tensors) {
        if (!descriptor_tensor.targetDevice) {
            throw std::runtime_error(
                "RPC TensorPipe descriptor is missing a target device");
        }
        const tensorpipe::Device target = *descriptor_tensor.targetDevice;
        const tensorplay::Device device = from_tensorpipe_device(target);
        if (device.is_cpu()) {
            state->storages.emplace_back(
                descriptor_tensor.length,
                tensorplay::getAllocator(tensorplay::DeviceType::CPU),
                device);
            allocation.tensors.push_back({
                tensorpipe::CpuBuffer{state->storages.back().data()}});
            continue;
        }
#ifdef USE_CUDA
        if (device.is_cuda()) {
            tensorplay::cuda::CUDAGuard guard(static_cast<int>(device.index()));
            state->storages.emplace_back(
                descriptor_tensor.length,
                tensorplay::getAllocator(tensorplay::DeviceType::CUDA),
                device);
            allocation.tensors.push_back({tensorpipe::CudaBuffer{
                state->storages.back().data(),
                tensorplay::cuda::getCurrentCUDAStream(
                    static_cast<int>(device.index())).stream()}});
            continue;
        }
#endif
        throw std::runtime_error("RPC TensorPipe descriptor has an invalid device");
    }
    return {std::move(allocation), std::move(state)};
}

MessagePtr decode_tensorpipe_message(
    const tensorpipe::Descriptor& descriptor,
    const TensorPipeReadState& state) {
    if (!state.type || !state.id || descriptor.payloads.size() != 4) {
        throw std::runtime_error("RPC TensorPipe message state is invalid");
    }
    const std::string metadata(
        state.tensor_metadata.begin(), state.tensor_metadata.end());
    const py::object records = deserialize_python_object(
        SerializedPyObj(metadata, {}));
    const py::list tensor_records = records.cast<py::list>();
    TensorDecoder decoder{state};
    std::vector<py::object> tensors;
    tensors.reserve(tensor_records.size());
    for (const auto& record : tensor_records) {
        tensors.emplace_back(py::cast(decoder.decode(record)));
    }
    return std::make_shared<Message>(
        state.payload,
        std::move(tensors),
        *state.type,
        *state.id);
}

}  // namespace tensorplay::distributed::rpc
