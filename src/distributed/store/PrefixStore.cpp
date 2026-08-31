#include "store/PrefixStore.h"

namespace tensorplay {
namespace distributed {

PrefixStore::PrefixStore(
    std::string prefix,
    std::shared_ptr<Store> store,
    std::chrono::milliseconds timeout)
    : Store(timeout), prefix_(std::move(prefix)), store_(std::move(store)) {}

namespace {

std::string join(const std::string& prefix, const std::string& key) {
  return prefix.empty() ? key : prefix + "/" + key;
}

} // namespace

void PrefixStore::set(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  store_->set(join(prefix_, key), value);
}

std::vector<uint8_t> PrefixStore::compareSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& desiredValue) {
  return store_->compareSet(join(prefix_, key), expectedValue, desiredValue);
}

std::vector<uint8_t> PrefixStore::get(const std::string& key) {
  return store_->get(join(prefix_, key));
}

int64_t PrefixStore::add(const std::string& key, int64_t value) {
  return store_->add(join(prefix_, key), value);
}

bool PrefixStore::deleteKey(const std::string& key) {
  return store_->deleteKey(join(prefix_, key));
}

bool PrefixStore::check(const std::vector<std::string>& keys) {
  std::vector<std::string> prefixed;
  prefixed.reserve(keys.size());
  for (const auto& key : keys) {
    prefixed.push_back(join(prefix_, key));
  }
  return store_->check(prefixed);
}

int64_t PrefixStore::getNumKeys() {
  return store_->getNumKeys();
}

bool PrefixStore::wait(
    const std::vector<std::string>& keys,
    const std::chrono::milliseconds& timeout) {
  std::vector<std::string> prefixed;
  prefixed.reserve(keys.size());
  for (const auto& key : keys) {
    prefixed.push_back(join(prefix_, key));
  }
  return store_->wait(prefixed, timeout);
}

} // namespace distributed
} // namespace tensorplay
