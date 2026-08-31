#include "store/TCPStore.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <thread>

#include "Exception.h"

namespace tensorplay {
namespace distributed {

namespace {

// Wire protocol. Every exchange is one request followed by one reply:
//   request: [u8 opcode][u32 keyLen][key][payload]
//   reply:   [u8 status][payload]
// Status 0 marks success; 1 signals a failed lookup; 2 signals a decode
// error. Payloads are length-prefixed the same way.
enum class Op : uint8_t {
  SET = 0,
  GET = 1,
  ADD = 2,
  HAS = 3,
  CAS = 4,
  DEL = 5,
  CHECK = 6,
  NUMKEYS = 7,
};

enum class Status : uint8_t {
  OK = 0,
  NOT_FOUND = 1,
  ERROR = 2,
};

bool sendAll(int fd, const void* buf, size_t len) {
  const auto* bytes = static_cast<const uint8_t*>(buf);
  size_t sent = 0;
  while (sent < len) {
    ssize_t n = ::send(fd, bytes + sent, len - sent, MSG_NOSIGNAL);
    if (n <= 0) {
      return false;
    }
    sent += static_cast<size_t>(n);
  }
  return true;
}

bool sendFrame(int fd, const void* buf, size_t len) {
  const uint32_t size = static_cast<uint32_t>(len);
  return sendAll(fd, &size, sizeof(size)) && sendAll(fd, buf, len);
}

bool recvAll(int fd, void* buf, size_t len) {
  auto* bytes = static_cast<uint8_t*>(buf);
  size_t got = 0;
  while (got < len) {
    ssize_t n = ::recv(fd, bytes + got, len - got, 0);
    if (n <= 0) {
      return false;
    }
    got += static_cast<size_t>(n);
  }
  return true;
}

bool recvFrame(int fd, std::vector<uint8_t>* out) {
  uint32_t size = 0;
  if (!recvAll(fd, &size, sizeof(size))) {
    return false;
  }
  out->assign(size, 0);
  return size == 0 || recvAll(fd, out->data(), size);
}

bool recvKey(int fd, std::string* key) {
  uint32_t keyLen = 0;
  if (!recvAll(fd, &keyLen, sizeof(keyLen))) {
    return false;
  }
  std::string bytes(keyLen, '\0');
  if (keyLen > 0 && !recvAll(fd, bytes.data(), keyLen)) {
    return false;
  }
  *key = std::move(bytes);
  return true;
}

std::vector<uint8_t> i64Bytes(int64_t v) {
  return std::vector<uint8_t>(
      reinterpret_cast<const uint8_t*>(&v),
      reinterpret_cast<const uint8_t*>(&v) + sizeof(int64_t));
}

int64_t bytesI64(const std::vector<uint8_t>& bytes) {
  int64_t v = 0;
  if (bytes.size() == sizeof(int64_t)) {
    std::memcpy(&v, bytes.data(), sizeof(int64_t));
  }
  return v;
}

std::vector<uint8_t> packKeys(const std::vector<std::string>& keys) {
  std::vector<uint8_t> blob;
  for (const auto& key : keys) {
    uint32_t len = static_cast<uint32_t>(key.size());
    const auto* lenBytes = reinterpret_cast<const uint8_t*>(&len);
    blob.insert(blob.end(), lenBytes, lenBytes + sizeof(len));
    blob.insert(blob.end(), key.begin(), key.end());
  }
  return blob;
}

std::vector<std::string> unpackKeys(const std::vector<uint8_t>& blob) {
  std::vector<std::string> keys;
  size_t offset = 0;
  while (offset + sizeof(uint32_t) <= blob.size()) {
    uint32_t len = 0;
    std::memcpy(&len, blob.data() + offset, sizeof(len));
    offset += sizeof(len);
    if (offset + len > blob.size()) {
      break;
    }
    keys.emplace_back(blob.begin() + offset, blob.begin() + offset + len);
    offset += len;
  }
  return keys;
}

std::vector<uint8_t> strBytes(const std::string& s) {
  return std::vector<uint8_t>(s.begin(), s.end());
}

} // namespace

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

class TCPStore::Server {
 public:
  Server(std::string host, uint16_t requestedPort);

  ~Server() {
    stop();
  }

  uint16_t port() const {
    return port_;
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stop_) {
        return;
      }
      stop_ = true;
      // Unblocks per-connection handlers so their threads can finish.
      for (int fd : conns_) {
        ::shutdown(fd, SHUT_RDWR);
      }
    }
    if (listenFd_ >= 0) {
      ::shutdown(listenFd_, SHUT_RDWR);
      ::close(listenFd_);
      listenFd_ = -1;
    }
    if (acceptThread_.joinable()) {
      acceptThread_.join();
    }
  }

 private:
  struct Request {
    Op op;
    std::string key;
    std::vector<uint8_t> value;
    std::vector<uint8_t> expected;
    std::vector<std::string> keys;
  };

  bool readRequest(int fd, Request* request);
  void handleConnection(int fd);
  void acceptLoop();

  std::map<std::string, std::vector<uint8_t>> data_;
  std::mutex mutex_;
  std::set<int> conns_;
  int listenFd_{-1};
  uint16_t port_{0};
  std::thread acceptThread_;
  bool stop_{false};
};

TCPStore::Server::Server(std::string host, uint16_t requestedPort) {
  listenFd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  TP_CHECK(listenFd_ >= 0, "TCPStore server: socket() failed");
  int reuse = 1;
  ::setsockopt(
      listenFd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

  struct sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(requestedPort);
  if (host.empty() || host == "localhost") {
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  } else if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
    struct hostent* entry = ::gethostbyname(host.c_str());
    TP_CHECK(entry != nullptr, "TCPStore server: cannot resolve host");
    std::memcpy(&addr.sin_addr, entry->h_addr_list[0], sizeof(addr.sin_addr));
  }
  TP_CHECK(
      ::bind(listenFd_, reinterpret_cast<struct sockaddr*>(&addr),
             sizeof(addr)) == 0,
      "TCPStore server: bind failed");
  TP_CHECK(
      ::listen(listenFd_, 64) == 0, "TCPStore server: listen failed");

  struct sockaddr_in bound{};
  socklen_t boundLen = sizeof(bound);
  TP_CHECK(
      ::getsockname(listenFd_, reinterpret_cast<struct sockaddr*>(&bound),
                    &boundLen) == 0,
      "TCPStore server: getsockname failed");
  port_ = ntohs(bound.sin_port);

  acceptThread_ = std::thread(&Server::acceptLoop, this);
}

bool TCPStore::Server::readRequest(int fd, Request* request) {
  uint8_t opByte = 0;
  if (!recvAll(fd, &opByte, sizeof(opByte))) {
    return false;
  }
  request->op = static_cast<Op>(opByte);
  if (!recvKey(fd, &request->key)) {
    return false;
  }
  switch (request->op) {
    case Op::SET:
      return recvFrame(fd, &request->value);
    case Op::CAS:
      return recvFrame(fd, &request->expected) &&
          recvFrame(fd, &request->value);
    case Op::ADD:
      return recvFrame(fd, &request->value);
    case Op::CHECK: {
      std::vector<uint8_t> blob;
      if (!recvFrame(fd, &blob)) {
        return false;
      }
      request->keys = unpackKeys(blob);
      return true;
    }
    case Op::GET:
    case Op::HAS:
    case Op::DEL:
    case Op::NUMKEYS:
      return true;
  }
  return false;
}

void TCPStore::Server::handleConnection(int fd) {
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stop_) {
      ::close(fd);
      return;
    }
    conns_.insert(fd);
  }
  for (;;) {
    Request request;
    if (!readRequest(fd, &request)) {
      break;
    }
    const uint8_t ok = static_cast<uint8_t>(Status::OK);
    const uint8_t notFound = static_cast<uint8_t>(Status::NOT_FOUND);
    switch (request.op) {
      case Op::SET: {
        std::lock_guard<std::mutex> lock(mutex_);
        data_[request.key] = request.value;
        sendAll(fd, &ok, 1);
        break;
      }
      case Op::GET: {
        std::vector<uint8_t> value;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          auto it = data_.find(request.key);
          if (it == data_.end()) {
            sendAll(fd, &notFound, 1);
            break;
          }
          value = it->second;
        }
        sendAll(fd, &ok, 1);
        sendFrame(fd, value.data(), value.size());
        break;
      }
      case Op::ADD: {
        std::lock_guard<std::mutex> lock(mutex_);
        // Counters are stored as decimal ASCII text, matching the set/get
        // representation clients observe.
        int64_t current = 0;
        auto it = data_.find(request.key);
        if (it != data_.end()) {
          try {
            current = std::stoll(
                std::string(it->second.begin(), it->second.end()));
          } catch (const std::exception&) {
            current = 0;
          }
        }
        const int64_t updated = current + bytesI64(request.value);
        const std::string text = std::to_string(updated);
        data_[request.key] = std::vector<uint8_t>(text.begin(), text.end());
        sendAll(fd, &ok, 1);
        sendFrame(fd, text.data(), text.size());
        break;
      }
      case Op::HAS: {
        std::lock_guard<std::mutex> lock(mutex_);
        const uint8_t present =
            data_.count(request.key) > 0 ? ok : notFound;
        sendAll(fd, &present, 1);
        break;
      }
      case Op::CAS: {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = data_.find(request.key);
        std::vector<uint8_t> current =
            it == data_.end() ? std::vector<uint8_t>{} : it->second;
        if (current == request.expected) {
          data_[request.key] = request.value;
        }
        sendAll(fd, &ok, 1);
        sendFrame(fd, current.data(), current.size());
        break;
      }
      case Op::DEL: {
        std::lock_guard<std::mutex> lock(mutex_);
        const uint8_t removed =
            data_.erase(request.key) > 0 ? ok : notFound;
        sendAll(fd, &removed, 1);
        break;
      }
      case Op::CHECK: {
        std::lock_guard<std::mutex> lock(mutex_);
        bool allPresent = true;
        for (const auto& key : request.keys) {
          if (data_.find(key) == data_.end()) {
            allPresent = false;
            break;
          }
        }
        sendAll(fd, &ok, 1);
        const uint8_t flag = allPresent ? ok : notFound;
        sendAll(fd, &flag, 1);
        break;
      }
      case Op::NUMKEYS: {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto count = static_cast<int64_t>(data_.size());
        sendAll(fd, &ok, 1);
        sendFrame(fd, i64Bytes(count).data(), sizeof(int64_t));
        break;
      }
      default:
        ::close(fd);
        {
          std::lock_guard<std::mutex> lock(mutex_);
          conns_.erase(fd);
        }
        return;
    }
  }
  ::close(fd);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    conns_.erase(fd);
  }
}

void TCPStore::Server::acceptLoop() {
  while (true) {
    int fd = ::accept(listenFd_, nullptr, nullptr);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stop_) {
        if (fd >= 0) {
          ::close(fd);
        }
        return;
      }
    }
    if (fd < 0) {
      return;
    }
    // One thread per connection keeps the server logic sequential per
    // client, matching the rendezvous-scale request rate.
    std::thread(&Server::handleConnection, this, fd).detach();
  }
}

// ---------------------------------------------------------------------------
// Client-side store
// ---------------------------------------------------------------------------

TCPStore::TCPStore(
    std::string host,
    uint16_t port,
    bool isServer,
    std::chrono::milliseconds timeout)
    : Store(timeout), host_(std::move(host)), port_(port),
      isServer_(isServer) {
  if (isServer_) {
    startServer();
  }
}

TCPStore::~TCPStore() {
  server_.reset();
}

void TCPStore::startServer() {
  server_ = std::make_shared<Server>(host_, port_);
  port_ = server_->port();
}

class TCPStoreClientHelper {
  // Connection-per-request: rendezvous traffic is low-rate, and this keeps
  // error handling free of reconnect state machines.
 public:
  static int connect(
      const std::string& host,
      uint16_t port,
      const std::chrono::milliseconds& timeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
      int fd = attemptConnect(host, port);
      if (fd >= 0) {
        return fd;
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        TP_CHECK(
            false,
            "TCPStore: could not connect to ",
            host,
            ":",
            port);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  static int attemptConnect(const std::string& host, uint16_t port) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      return -1;
    }
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (host.empty() || host == "localhost") {
      addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    } else if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1) {
      struct hostent* entry = ::gethostbyname(host.c_str());
      if (entry == nullptr) {
        ::close(fd);
        return -1;
      }
      std::memcpy(&addr.sin_addr, entry->h_addr_list[0], sizeof(addr.sin_addr));
    }
    if (::connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) !=
        0) {
      ::close(fd);
      return -1;
    }
    return fd;
  }
};

namespace {

// Builds one request on the wire and consumes the reply. The payload
// follows the opcode: SET/CAS carry values, ADD an integer, CHECK the
// packed key list.
template <typename ConsumeReply>
bool exchange(
    const std::string& host,
    uint16_t port,
    const std::chrono::milliseconds& timeout,
    Op op,
    const std::string& key,
    const std::vector<uint8_t>& value,
    const std::vector<uint8_t>& expected,
    const std::vector<std::string>& extraKeys,
    ConsumeReply&& consumeReply) {
  int fd = TCPStoreClientHelper::connect(host, port, timeout);
  if (fd < 0) {
    return false;
  }
  const uint8_t opByte = static_cast<uint8_t>(op);
  bool ok = sendAll(fd, &opByte, 1) && sendFrame(fd, key.data(), key.size());
  if (ok && op == Op::SET) {
    ok = sendFrame(fd, value.data(), value.size());
  } else if (ok && op == Op::CAS) {
    ok = sendFrame(fd, expected.data(), expected.size()) &&
        sendFrame(fd, value.data(), value.size());
  } else if (ok && op == Op::ADD) {
    const auto bytes = i64Bytes(bytesI64(value));
    ok = sendFrame(fd, bytes.data(), bytes.size());
  } else if (ok && op == Op::CHECK) {
    const auto blob = packKeys(extraKeys);
    ok = sendFrame(fd, blob.data(), blob.size());
  }
  uint8_t status = 0;
  if (ok) {
    ok = recvAll(fd, &status, 1);
  }
  if (ok) {
    consumeReply(fd, static_cast<Status>(status));
  }
  ::close(fd);
  return ok;
}

} // namespace

void TCPStore::set(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  bool ok = exchange(
      host_, port_, timeout_, Op::SET, key, value, {}, {},
      [](int, Status) {});
  TP_CHECK(ok, "TCPStore set: request failed");
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
  std::vector<uint8_t> value;
  bool ok = exchange(
      host_, port_, timeout_, Op::GET, key, {}, {}, {},
      [&](int fd, Status status) {
        if (status == Status::NOT_FOUND) {
          return;
        }
        recvFrame(fd, &value);
      });
  TP_CHECK(ok, "TCPStore get: request failed");
  return value;
}

int64_t TCPStore::add(const std::string& key, int64_t value) {
  int64_t updated = 0;
  bool ok = exchange(
      host_, port_, timeout_, Op::ADD, key, i64Bytes(value), {}, {},
      [&](int fd, Status status) {
        if (status != Status::OK) {
          return;
        }
        std::vector<uint8_t> bytes;
        recvFrame(fd, &bytes);
        try {
          updated = std::stoll(std::string(bytes.begin(), bytes.end()));
        } catch (const std::exception&) {
          updated = 0;
        }
      });
  TP_CHECK(ok, "TCPStore add: request failed");
  return updated;
}

bool TCPStore::deleteKey(const std::string& key) {
  bool removed = false;
  bool ok = exchange(
      host_, port_, timeout_, Op::DEL, key, {}, {}, {},
      [&](int fd, Status status) { removed = status == Status::OK; });
  TP_CHECK(ok, "TCPStore delete: request failed");
  return removed;
}

std::vector<uint8_t> TCPStore::compareSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& desiredValue) {
  std::vector<uint8_t> current;
  bool ok = exchange(
      host_, port_, timeout_, Op::CAS, key, desiredValue, expectedValue, {},
      [&](int fd, Status status) {
        if (status != Status::OK) {
          return;
        }
        recvFrame(fd, &current);
      });
  TP_CHECK(ok, "TCPStore compareSet: request failed");
  return current;
}

bool TCPStore::check(const std::vector<std::string>& keys) {
  bool allPresent = false;
  bool ok = exchange(
      host_, port_, timeout_, Op::CHECK, "", {}, {}, keys,
      [&](int fd, Status status) {
        if (status != Status::OK) {
          return;
        }
        uint8_t flag = 0;
        recvAll(fd, &flag, 1);
        allPresent = flag == static_cast<uint8_t>(Status::OK);
      });
  TP_CHECK(ok, "TCPStore check: request failed");
  return allPresent;
}

int64_t TCPStore::getNumKeys() {
  int64_t count = 0;
  bool ok = exchange(
      host_, port_, timeout_, Op::NUMKEYS, "", {}, {}, {},
      [&](int fd, Status status) {
        if (status != Status::OK) {
          return;
        }
        std::vector<uint8_t> bytes;
        recvFrame(fd, &bytes);
        count = bytesI64(bytes);
      });
  TP_CHECK(ok, "TCPStore numKeys: request failed");
  return count;
}

} // namespace distributed
} // namespace tensorplay
