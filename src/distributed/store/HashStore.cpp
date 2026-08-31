#include "store/HashStore.h"

#include <cstring>

namespace tensorplay {
namespace distributed {

void HashStore::set(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  std::lock_guard<std::mutex> lock(mutex_);
  data_[key] = value;
}

std::vector<uint8_t> HashStore::compareSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& desiredValue) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = data_.find(key);
  std::vector<uint8_t> current =
      it == data_.end() ? std::vector<uint8_t>{} : it->second;
  if (current == expectedValue) {
    data_[key] = desiredValue;
  }
  return current;
}

std::vector<uint8_t> HashStore::get(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = data_.find(key);
  return it == data_.end() ? std::vector<uint8_t>{} : it->second;
}

int64_t HashStore::add(const std::string& key, int64_t value) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = data_.find(key);
  int64_t current = 0;
  if (it != data_.end()) {
    try {
      current = std::stoll(
          std::string(it->second.begin(), it->second.end()));
    } catch (const std::exception&) {
      current = 0;
    }
  }
  const int64_t updated = current + value;
  const std::string text = std::to_string(updated);
  data_[key] = std::vector<uint8_t>(text.begin(), text.end());
  return updated;
}

bool HashStore::deleteKey(const std::string& key) {
  std::lock_guard<std::mutex> lock(mutex_);
  return data_.erase(key) > 0;
}

bool HashStore::check(const std::vector<std::string>& keys) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& key : keys) {
    if (data_.find(key) == data_.end()) {
      return false;
    }
  }
  return true;
}

int64_t HashStore::getNumKeys() {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<int64_t>(data_.size());
}

} // namespace distributed
} // namespace tensorplay
