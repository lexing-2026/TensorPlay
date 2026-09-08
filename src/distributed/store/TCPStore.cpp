#include "store/TCPStore.h"

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <thread>
#include <vector>

#include "Exception.h"

namespace tensorplay {
namespace distributed {

#if defined(_WIN32)
// Winsock2 keeps the BSD-socket call shapes; only the type spellings and
// process-wide initialization differ. Descriptors stay `int` (SOCKET values
// fit and INVALID_SOCKET maps to -1), so every call site keeps its shape.
namespace {

using Ssize = int;

void winsock_init() {
  static const bool ready = [] {
    WSADATA data{};
    return WSAStartup(MAKEWORD(2, 2), &data) == 0;
  }();
  (void)ready;
}

int close_socket(int fd) { return closesocket(fd); }

constexpr int kMsgNoSignal = 0;  // no SIGPIPE semantics on Windows
constexpr int kShutdownBoth = SD_BOTH;

#else
namespace {

using Ssize = ssize_t;

void winsock_init() {}
int close_socket(int fd) { return ::close(fd); }

constexpr int kMsgNoSignal = MSG_NOSIGNAL;
constexpr int kShutdownBoth = SHUT_RDWR;

#endif

} // namespace

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

bool setSocketTimeout(
    int fd,
    const std::chrono::milliseconds& timeout) {
  if (timeout.count() < 0) {
    return true;
  }
  struct timeval value{};
  using Seconds = decltype(value.tv_sec);
  const int64_t seconds = timeout.count() / 1000;
  value.tv_sec = static_cast<Seconds>(std::min<int64_t>(
      seconds, static_cast<int64_t>(std::numeric_limits<Seconds>::max())));
  value.tv_usec = static_cast<decltype(value.tv_usec)>(
      (timeout.count() % 1000) * 1000);
  return ::setsockopt(
             fd,
             SOL_SOCKET,
             SO_RCVTIMEO,
             &value,
             sizeof(value)) == 0 &&
      ::setsockopt(
             fd,
             SOL_SOCKET,
             SO_SNDTIMEO,
             &value,
             sizeof(value)) == 0;
}

bool sendAll(int fd, const void* buf, size_t len) {
  const auto* bytes = static_cast<const uint8_t*>(buf);
  size_t sent = 0;
  while (sent < len) {
    Ssize n = ::send(fd, bytes + sent, len - sent, kMsgNoSignal);
    if (n <= 0) {
      return false;
    }
    sent += static_cast<size_t>(n);
  }
  return true;
}

bool sendFrame(int fd, const void* buf, size_t len) {
  TP_CHECK(
      len <= std::numeric_limits<uint32_t>::max(),
      "TCPStore: frame is too large");
  const uint32_t size = static_cast<uint32_t>(len);
  return sendAll(fd, &size, sizeof(size)) && sendAll(fd, buf, len);
}

bool recvAll(int fd, void* buf, size_t len) {
  auto* bytes = static_cast<uint8_t*>(buf);
  size_t got = 0;
  while (got < len) {
    Ssize n = ::recv(fd, bytes + got, len - got, 0);
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
  if (size > std::vector<uint8_t>().max_size()) {
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
  if (keyLen > std::string().max_size()) {
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

int64_t checkedAdd(int64_t left, int64_t right) {
  if ((right > 0 && left > std::numeric_limits<int64_t>::max() - right) ||
      (right < 0 && left < std::numeric_limits<int64_t>::min() - right)) {
    TP_THROW(RuntimeError, "TCPStore: counter overflow");
  }
  return left + right;
}

std::vector<uint8_t> packKeys(const std::vector<std::string>& keys) {
  std::vector<uint8_t> blob;
  for (const auto& key : keys) {
    TP_CHECK(
        key.size() <= std::numeric_limits<uint32_t>::max(),
        "TCPStore: key is too large");
    const uint32_t len = static_cast<uint32_t>(key.size());
    const auto* lenBytes = reinterpret_cast<const uint8_t*>(&len);
    blob.insert(blob.end(), lenBytes, lenBytes + sizeof(len));
    blob.insert(blob.end(), key.begin(), key.end());
  }
  return blob;
}

bool unpackKeys(
    const std::vector<uint8_t>& blob,
    std::vector<std::string>* keys) {
  keys->clear();
  size_t offset = 0;
  while (offset <= blob.size() &&
         blob.size() - offset >= sizeof(uint32_t)) {
    uint32_t len = 0;
    std::memcpy(&len, blob.data() + offset, sizeof(len));
    offset += sizeof(len);
    if (len > blob.size() - offset) {
      return false;
    }
    keys->emplace_back(blob.begin() + offset, blob.begin() + offset + len);
    offset += len;
  }
  return offset == blob.size();
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
        ::shutdown(fd, kShutdownBoth);
      }
    }
    if (listenFd_ >= 0) {
      ::shutdown(listenFd_, kShutdownBoth);
      close_socket(listenFd_);
      listenFd_ = -1;
    }
    if (acceptThread_.joinable()) {
      acceptThread_.join();
    }
    for (auto& thread : connectionThreads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
    connectionThreads_.clear();
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
  std::vector<std::thread> connectionThreads_;
  int listenFd_{-1};
  uint16_t port_{0};
  std::thread acceptThread_;
  bool stop_{false};
};

TCPStore::Server::Server(std::string host, uint16_t requestedPort) {
  winsock_init();
  listenFd_ = static_cast<int>(::socket(AF_INET, SOCK_STREAM, 0));
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
      return recvFrame(fd, &request->value) &&
          request->value.size() == sizeof(int64_t);
    case Op::CHECK: {
      std::vector<uint8_t> blob;
      if (!recvFrame(fd, &blob)) {
        return false;
      }
      return unpackKeys(blob, &request->keys);
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
      close_socket(fd);
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
        int64_t updated = 0;
        try {
          updated = checkedAdd(current, bytesI64(request.value));
        } catch (const std::exception&) {
          const uint8_t error = static_cast<uint8_t>(Status::ERROR);
          sendAll(fd, &error, 1);
          break;
        }
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
        close_socket(fd);
        {
          std::lock_guard<std::mutex> lock(mutex_);
          conns_.erase(fd);
        }
        return;
    }
  }
  close_socket(fd);
  {
    std::lock_guard<std::mutex> lock(mutex_);
    conns_.erase(fd);
  }
}

void TCPStore::Server::acceptLoop() {
  while (true) {
    int fd = static_cast<int>(::accept(listenFd_, nullptr, nullptr));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stop_) {
        if (fd >= 0) {
          close_socket(fd);
        }
        return;
      }
    }
    if (fd < 0) {
      return;
    }
    // One thread per connection keeps the server logic sequential per
    // client, matching the rendezvous-scale request rate. Keeping the
    // handles lets shutdown wait until no handler still references Server.
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stop_) {
        close_socket(fd);
        continue;
      }
      connectionThreads_.emplace_back(&Server::handleConnection, this, fd);
    }
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
    const bool hasDeadline = timeout.count() >= 0;
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    for (;;) {
      int fd = attemptConnect(host, port);
      if (fd >= 0) {
        return fd;
      }
      const auto now = std::chrono::steady_clock::now();
      if (hasDeadline && now >= deadline) {
        TP_CHECK(
            false,
            "TCPStore: could not connect to ",
            host,
            ":",
            port);
      }
      auto delay = std::chrono::milliseconds(50);
      if (hasDeadline) {
        const auto remaining = std::chrono::duration_cast<
            std::chrono::milliseconds>(deadline - now);
        if (remaining <= std::chrono::milliseconds(0)) {
          TP_CHECK(
              false,
              "TCPStore: could not connect to ",
              host,
              ":",
              port);
        }
        delay = std::min(delay, remaining);
      }
      std::this_thread::sleep_for(delay);
    }
  }

  static int attemptConnect(const std::string& host, uint16_t port) {
    winsock_init();
    int fd = static_cast<int>(::socket(AF_INET, SOCK_STREAM, 0));
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
        close_socket(fd);
        return -1;
      }
      std::memcpy(&addr.sin_addr, entry->h_addr_list[0], sizeof(addr.sin_addr));
    }
    if (::connect(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) !=
        0) {
      close_socket(fd);
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
  const auto start = std::chrono::steady_clock::now();
  int fd = TCPStoreClientHelper::connect(host, port, timeout);
  if (fd < 0) {
    return false;
  }
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);
  const auto remaining = timeout.count() < 0
      ? std::chrono::milliseconds(-1)
      : (elapsed >= timeout ? std::chrono::milliseconds(0) : timeout - elapsed);
  if (!setSocketTimeout(fd, remaining)) {
    close_socket(fd);
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
    ok = consumeReply(fd, static_cast<Status>(status));
  }
  close_socket(fd);
  return ok;
}

} // namespace

void TCPStore::set(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  bool ok = exchange(
      host_, port_, timeout_, Op::SET, key, value, {}, {},
      [](int, Status status) { return status == Status::OK; });
  TP_CHECK(ok, "TCPStore set: request failed");
}

std::vector<uint8_t> TCPStore::get(const std::string& key) {
  std::vector<uint8_t> value;
  bool ok = exchange(
      host_, port_, timeout_, Op::GET, key, {}, {}, {},
      [&](int fd, Status status) {
        if (status == Status::NOT_FOUND) {
          return true;
        }
        return status == Status::OK && recvFrame(fd, &value);
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
          return false;
        }
        std::vector<uint8_t> bytes;
        if (!recvFrame(fd, &bytes)) {
          return false;
        }
        try {
          const std::string text(bytes.begin(), bytes.end());
          size_t parsed = 0;
          updated = std::stoll(text, &parsed);
          return parsed == text.size();
        } catch (const std::exception&) {
          return false;
        }
      });
  TP_CHECK(ok, "TCPStore add: request failed");
  return updated;
}

bool TCPStore::deleteKey(const std::string& key) {
  bool removed = false;
  bool ok = exchange(
      host_, port_, timeout_, Op::DEL, key, {}, {}, {},
      [&](int, Status status) {
        removed = status == Status::OK;
        return status == Status::OK || status == Status::NOT_FOUND;
      });
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
          return false;
        }
        return recvFrame(fd, &current);
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
          return false;
        }
        uint8_t flag = 0;
        if (!recvAll(fd, &flag, 1)) {
          return false;
        }
        allPresent = flag == static_cast<uint8_t>(Status::OK);
        return flag == static_cast<uint8_t>(Status::OK) ||
            flag == static_cast<uint8_t>(Status::NOT_FOUND);
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
          return false;
        }
        std::vector<uint8_t> bytes;
        if (!recvFrame(fd, &bytes) || bytes.size() != sizeof(int64_t)) {
          return false;
        }
        count = bytesI64(bytes);
        return true;
      });
  TP_CHECK(ok, "TCPStore numKeys: request failed");
  return count;
}

} // namespace distributed
} // namespace tensorplay
