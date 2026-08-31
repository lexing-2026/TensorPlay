#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tensorplay {
namespace distributed {

// Abstract key/value store used for rendezvous and cross-process
// synchronization. Values are arbitrary byte strings; keys are text.
//
// Timeout contract: operations that can block take an explicit timeout and
// return a completion flag (or a sentinel) instead of throwing on expiry,
// so callers can implement retry policies.
class Store : public std::enable_shared_from_this<Store> {
 public:
  static constexpr std::chrono::milliseconds kDefaultTimeout{300000};

  explicit Store(std::chrono::milliseconds timeout = kDefaultTimeout)
      : timeout_(timeout) {}
  virtual ~Store();

  // Publishes a new version of `key`.
  virtual void set(
      const std::string& key,
      const std::vector<uint8_t>& value) = 0;

  // Compares the current value against `expectedValue` (a missing key
  // compares equal to the empty string) and swaps in `desiredValue` when
  // they match. Returns the pre-swap value; the swap happened exactly when
  // the returned value equals `expectedValue`.
  virtual std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& expectedValue,
      const std::vector<uint8_t>& desiredValue) = 0;

  // Returns the newest value for `key`, or the empty string when absent.
  virtual std::vector<uint8_t> get(const std::string& key) = 0;

  // Atomically increments the integer stored at `key` (missing keys start
  // from zero) and returns the new value.
  virtual int64_t add(const std::string& key, int64_t value) = 0;

  // Removes `key`; returns true when a value existed.
  virtual bool deleteKey(const std::string& key) = 0;

  // True when every key currently has a value.
  virtual bool check(const std::vector<std::string>& keys) = 0;

  // Number of distinct keys currently stored.
  virtual int64_t getNumKeys() = 0;

  // Blocks until every key has a value or the timeout expires. Returns
  // true on completion.
  virtual bool wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout);

  bool wait(const std::vector<std::string>& keys) {
    return wait(keys, timeout_);
  }

  const std::chrono::milliseconds& getTimeout() const noexcept {
    return timeout_;
  }

  virtual void setTimeout(const std::chrono::milliseconds& timeout) {
    timeout_ = timeout;
  }

 protected:
  std::chrono::milliseconds timeout_;
};

} // namespace distributed
} // namespace tensorplay
