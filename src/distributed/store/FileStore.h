#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "store/Store.h"

namespace tensorplay {
namespace distributed {

// Flock-protected append-log file store. Every mutation appends one record
// under an exclusive lock; readers scan the log and apply the newest record
// per key. Records are length-prefixed, so keys and values are arbitrary
// binary strings.
class FileStore : public Store {
 public:
  FileStore(
      std::string path,
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

  struct Snapshot {
    std::map<std::string, std::vector<uint8_t>> latest;
  };

 private:
  Snapshot snapshotLocked();

  std::string path_;
};

} // namespace distributed
} // namespace tensorplay
