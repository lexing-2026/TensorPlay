#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "store/Store.h"

namespace tensorplay {
namespace distributed {

// In-process key/value store, one instance per memory space. Used for
// single-process tests and as the reference semantics for the networked
// stores.
class HashStore : public Store {
 public:
  explicit HashStore(
      std::chrono::milliseconds timeout = Store::kDefaultTimeout)
      : Store(timeout) {}

  void set(
      const std::string& key,
      const std::vector<uint8_t>& value) override;
  std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& expectedValue,
      const std::vector<uint8_t>& desiredValue) override;
  std::vector<uint8_t> get(const std::string& key) override;
  int64_t add(const std::string& key, int64_t value) override;
  bool deleteKey(const std::string& key) override;
  bool check(const std::vector<std::string>& keys) override;
  int64_t getNumKeys() override;

 private:
  std::mutex mutex_;
  std::map<std::string, std::vector<uint8_t>> data_;
};

} // namespace distributed
} // namespace tensorplay
