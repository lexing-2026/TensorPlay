#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Device.h>
#include <Tensor.h>

namespace tensorplay {
namespace distributed {

inline constexpr std::chrono::milliseconds kUnsetTimeout{-1};

struct _SupplementBase {
  virtual ~_SupplementBase() = default;
};

struct PreMulSumSupplement : _SupplementBase {
  explicit PreMulSumSupplement(double factor) : double_factor(factor) {}
  explicit PreMulSumSupplement(Tensor tensor)
      : tensor_factor(std::move(tensor)) {
    if (!tensor_factor->defined() || tensor_factor->numel() != 1) {
      throw std::invalid_argument(
          "pre-multiply reduction tensor must contain one element");
    }
  }

  double double_factor{0.0};
  std::optional<Tensor> tensor_factor;
};

using NCCLPreMulSumSupplement = PreMulSumSupplement;

class ReduceOp {
 public:
  enum RedOpType : uint8_t {
    SUM = 0,
    AVG = 1,
    PRODUCT = 2,
    MIN = 3,
    MAX = 4,
    BAND = 5,
    BOR = 6,
    BXOR = 7,
    PREMUL_SUM = 8,
    UNUSED = 9,
  };

  ReduceOp() noexcept = default;
  ReduceOp(RedOpType op) : op_(op) {
    if (op_ == PREMUL_SUM) {
      throw std::invalid_argument(
          "PREMUL_SUM requires a scaling factor");
    }
  }
  ReduceOp(
      RedOpType op,
      std::shared_ptr<_SupplementBase> supplement)
      : op_(op), supplement_(std::move(supplement)) {
    if (op_ == PREMUL_SUM && !supplement_) {
      throw std::invalid_argument(
          "PREMUL_SUM requires a scaling factor");
    }
  }
  ReduceOp(int value) : ReduceOp(fromInt(value)) {}

  RedOpType op() const noexcept {
    return op_;
  }

  operator RedOpType() const noexcept {
    return op_;
  }

  explicit operator int() const noexcept {
    return static_cast<int>(op_);
  }

  bool operator==(RedOpType other) const noexcept {
    return op_ == other;
  }

  bool operator!=(RedOpType other) const noexcept {
    return !(*this == other);
  }

  bool operator==(std::uint8_t other) const {
    if (other > static_cast<std::uint8_t>(UNUSED)) {
      throw std::invalid_argument("invalid reduction operation");
    }
    return static_cast<std::uint8_t>(op_) == other;
  }

  bool operator!=(std::uint8_t other) const {
    return !(*this == other);
  }

  bool operator==(const ReduceOp& other) const noexcept {
    return op_ == other.op_;
  }

  bool operator!=(const ReduceOp& other) const noexcept {
    return !(*this == other);
  }

  RedOpType op_{SUM};
  std::shared_ptr<_SupplementBase> supplement_;

 private:
  static RedOpType fromInt(int value) {
    if (value < static_cast<int>(SUM) ||
        value > static_cast<int>(UNUSED)) {
      throw std::invalid_argument("invalid reduction operation");
    }
    return static_cast<RedOpType>(value);
  }

};

template <typename T>
ReduceOp makePreMulSum(const T& factor) {
  ReduceOp op;
  op.op_ = ReduceOp::PREMUL_SUM;
  op.supplement_ = std::make_shared<PreMulSumSupplement>(factor);
  return op;
}

inline bool isComplexViewAsRealAllowed(const ReduceOp& reduceOp) {
  switch (reduceOp.op()) {
    case ReduceOp::SUM:
    case ReduceOp::AVG:
    case ReduceOp::PREMUL_SUM:
    case ReduceOp::UNUSED:
      return true;
    default:
      return false;
  }
}

struct BroadcastOptions {
  int64_t rootRank{0};
  int64_t rootTensor{0};
  std::chrono::milliseconds timeout{kUnsetTimeout};
  bool asyncOp{true};
};

struct AllreduceOptions {
  ReduceOp reduceOp{ReduceOp::SUM};
  std::chrono::milliseconds timeout{kUnsetTimeout};
  bool asyncOp{true};
  std::optional<Tensor> sparseIndices;
};

struct AllreduceCoalescedOptions : AllreduceOptions {};

struct ReduceOptions {
  ReduceOp reduceOp{ReduceOp::SUM};
  int64_t rootRank{0};
  int64_t rootTensor{0};
  std::chrono::milliseconds timeout{kUnsetTimeout};
  bool asyncOp{true};
};

struct AllgatherOptions {
  std::chrono::milliseconds timeout{kUnsetTimeout};
  bool asyncOp{true};
};

struct GatherOptions {
  int64_t rootRank{0};
  std::chrono::milliseconds timeout{kUnsetTimeout};
  bool asyncOp{true};
};

struct ScatterOptions {
  int64_t rootRank{0};
  std::chrono::milliseconds timeout{kUnsetTimeout};
  bool asyncOp{true};
};

struct ReduceScatterOptions {
  ReduceOp reduceOp{ReduceOp::SUM};
  std::chrono::milliseconds timeout{kUnsetTimeout};
  bool asyncOp{true};
};

struct AllToAllOptions {
  std::chrono::milliseconds timeout{kUnsetTimeout};
  bool asyncOp{true};
};

struct BarrierOptions {
  std::vector<int64_t> device_ids;
  std::chrono::milliseconds timeout{kUnsetTimeout};
  std::optional<Device> device;
  bool asyncOp{true};
};

} // namespace distributed
} // namespace tensorplay
