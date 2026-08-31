#pragma once

#include <memory>
#include <string>
#include <vector>

#include "store/Store.h"

namespace tensorplay {
namespace distributed {

// Wraps another store and prepends a fixed prefix to every key, giving
// independent namespaces over the same backend (one per device, group, or
// run id).
class PrefixStore : public Store {
 public:
  PrefixStore(
      std::string prefix,
      std::shared_ptr<Store> store,
      std::chrono::milliseconds timeout = Store::kDefaultTimeout);

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
  bool wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;

  // The two-argument overload above hides the single-argument default.
  using Store::wait;

 private:
  std::string prefix_;
  std::shared_ptr<Store> store_;
};

} // namespace distributed
} // namespace tensorplay
