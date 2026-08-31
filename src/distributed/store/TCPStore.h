#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "store/Store.h"

namespace tensorplay {
namespace distributed {

// Key/value store over TCP, split into a server (owning the authoritative
// table plus a listening socket) and clients talking a small binary
// protocol: one request per exchange, every field length-prefixed so keys
// and values stay byte-transparent.
//
// With isServer=true the instance starts the server thread; port 0 asks the
// kernel for a free port, exposed through port(). Otherwise the instance is
// a client and `port` must reference a running server.
class TCPStore : public Store {
 public:
  static constexpr uint16_t kDefaultPort = 29500;

  TCPStore(
      std::string host,
      uint16_t port = kDefaultPort,
      bool isServer = false,
      std::chrono::milliseconds timeout = Store::kDefaultTimeout);

  ~TCPStore() override;

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

  const std::string& host() const noexcept {
    return host_;
  }
  uint16_t port() const noexcept {
    return port_;
  }

  // Shuts the server thread down early; without this call the destructor
  // performs the same join.
  void stop() {
    server_.reset();
  }

 private:
  class Server;

  void startServer();

  std::string host_;
  uint16_t port_;
  bool isServer_;
  std::shared_ptr<Server> server_;
};

} // namespace distributed
} // namespace tensorplay
