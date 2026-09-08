#include "store/FileStore.h"

#if defined(_WIN32)
// windows.h first: winnt.h refuses to compile without an architecture
// target, which the toolchain headers only set up through this include.
#include <windows.h>
#include <direct.h>
#include <fileapi.h>
#include <io.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>

#include "Exception.h"

namespace tensorplay {
namespace distributed {

#if defined(_WIN32)
// Thin POSIX-shape wrappers over the Win32 file APIs, following the c10d
// FileStore arrangement: same call sites, platform-specific only here.
namespace {

using Ssize = int;

// Win32 CRT spellings; kOpenBinary strips the text-mode translation.
constexpr int kOpenBinary = _O_BINARY;
constexpr int kOpenRdWr = _O_RDWR;
constexpr int kOpenCreat = _O_CREAT;
constexpr int kOpenRdOnly = _O_RDONLY;
constexpr int kOpenWrOnly = _O_WRONLY;
constexpr int kOpenAppend = _O_APPEND;

int file_open(const char* path, int flags, int mode = 0) {
  return _open(path, flags, mode);
}
int file_close(int fd) { return _close(fd); }
Ssize file_write(int fd, const void* buf, size_t len) {
  return _write(fd, buf, static_cast<unsigned>(len));
}
long file_seek_end(int fd) { return _lseek(fd, 0, SEEK_END); }
int file_mkdir(const char* path) { return _mkdir(path); }

// Whole-file advisory lock via LockFileEx; blocking unless the flag says
// otherwise, matching flock's semantics for LOCK_EX / LOCK_SH / LOCK_UN.
int file_flock(int fd, int operation) {
  void* handle = reinterpret_cast<void*>(_get_osfhandle(fd));
  OVERLAPPED overlapped{};
  if (operation & LOCK_UN) {
    return UnlockFileEx(handle, 0, 1, 0, &overlapped) ? 0 : -1;
  }
  DWORD flags = operation & LOCK_EX ? LOCKFILE_EXCLUSIVE_LOCK : 0;
  return LockFileEx(handle, flags, 0, 1, 0, &overlapped) ? 0 : -1;
}

constexpr int kLockEx = 10;  // distinct values; mapped through file_flock only
constexpr int kLockSh = 20;
constexpr int kLockUn = 30;

} // namespace
#else
namespace {

using Ssize = ssize_t;

constexpr int kOpenBinary = 0;

// POSIX spellings; the Windows branch maps them onto the _O_* flags.
constexpr int kOpenRdWr = O_RDWR;
constexpr int kOpenCreat = O_CREAT;
constexpr int kOpenRdOnly = O_RDONLY;
constexpr int kOpenWrOnly = O_WRONLY;
constexpr int kOpenAppend = O_APPEND;

int file_open(const char* path, int flags, int mode = 0) {
  return ::open(path, flags, mode);
}
int file_close(int fd) { return file_close(fd); }
Ssize file_write(int fd, const void* buf, size_t len) {
  return ::write(fd, buf, len);
}
long file_seek_end(int fd) { return ::lseek(fd, 0, SEEK_END); }
int file_mkdir(const char* path) { return ::mkdir(path, 0755); }
int file_flock(int fd, int operation) { return ::flock(fd, operation); }

constexpr int kLockEx = LOCK_EX;
constexpr int kLockSh = LOCK_SH;
constexpr int kLockUn = LOCK_UN;

} // namespace
#endif

namespace {

// Record framing: [u32 key length][key][u32 value length][value][\n].
// A tombstone value marks deletion; the marker cannot collide with user
// bytes because values are length-scoped.
const std::vector<uint8_t> kTombstone{
    0x00, 0x5F, 0x5F, 0x64, 0x65, 0x6C, 0x65, 0x74, 0x65, 0x64, 0x5F, 0x5F};

void writeAll(int fd, const void* buf, size_t len) {
  const auto* bytes = static_cast<const uint8_t*>(buf);
  size_t written = 0;
  while (written < len) {
    Ssize n = file_write(fd, bytes + written, len - written);
    TP_CHECK(n > 0, "FileStore: write failed");
    written += static_cast<size_t>(n);
  }
}

uint32_t wireLength(size_t length, const char* field) {
  TP_CHECK(
      length <= std::numeric_limits<uint32_t>::max(),
      "FileStore: ",
      field,
      " is too large");
  return static_cast<uint32_t>(length);
}

int64_t checkedAdd(int64_t left, int64_t right) {
  if ((right > 0 && left > std::numeric_limits<int64_t>::max() - right) ||
      (right < 0 && left < std::numeric_limits<int64_t>::min() - right)) {
    TP_THROW(RuntimeError, "FileStore: counter overflow");
  }
  return left + right;
}

// Reads the whole file into memory under the shared lock, then decodes the
// record log. Small rendezvous logs make a full scan the cheapest correct
// strategy; compaction can come later without changing the format.
FileStore::Snapshot decodeLog(const std::string& path) {
  FileStore::Snapshot snapshot;
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return snapshot;
  }
  std::string blob((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  size_t offset = 0;
  auto readU32 = [&](uint32_t* out) {
    if (offset > blob.size() || blob.size() - offset < sizeof(uint32_t)) {
      return false;
    }
    std::memcpy(out, blob.data() + offset, sizeof(uint32_t));
    offset += sizeof(uint32_t);
    return true;
  };
  while (offset < blob.size()) {
    uint32_t keyLen = 0;
    uint32_t valLen = 0;
    if (!readU32(&keyLen) || keyLen > blob.size() - offset) {
      break;
    }
    std::string key(blob.data() + offset, keyLen);
    offset += keyLen;
    if (!readU32(&valLen) || valLen > blob.size() - offset) {
      break;
    }
    std::vector<uint8_t> value(
        blob.begin() + offset, blob.begin() + offset + valLen);
    offset += valLen;
    if (offset < blob.size() && blob[offset] == '\n') {
      offset += 1;
    }
    if (value == kTombstone) {
      snapshot.latest.erase(key);
      continue;
    }
    snapshot.latest[key] = std::move(value);
  }
  return snapshot;
}

} // namespace

FileStore::FileStore(std::string path, std::chrono::milliseconds timeout)
    : Store(timeout), path_(std::move(path)) {
  const auto separator = path_.find_last_of('/');
  if (separator != std::string::npos && separator > 0) {
    const auto parent = path_.substr(0, separator);
    file_mkdir(parent.c_str());
  }
  // Create eagerly so all ranks agree on the path before rendezvous.
  int fd = file_open(path_.c_str(), kOpenRdWr | kOpenCreat | kOpenBinary, 0644);
  TP_CHECK(fd >= 0, "FileStore: cannot create file");
  file_close(fd);
}

FileStore::Snapshot FileStore::snapshotLocked() {
  return decodeLog(path_);
}

void FileStore::set(
    const std::string& key,
    const std::vector<uint8_t>& value) {
  int fd = file_open(path_.c_str(), kOpenWrOnly | kOpenAppend | kOpenBinary);
  TP_CHECK(fd >= 0, "FileStore: cannot open for append");
  file_flock(fd, kLockEx);
  const uint32_t keyLen = wireLength(key.size(), "key");
  const uint32_t valLen = wireLength(value.size(), "value");
  writeAll(fd, &keyLen, sizeof(keyLen));
  writeAll(fd, key.data(), key.size());
  writeAll(fd, &valLen, sizeof(valLen));
  writeAll(fd, value.data(), value.size());
  writeAll(fd, "\n", 1);
  file_flock(fd, kLockUn);
  file_close(fd);
}

std::vector<uint8_t> FileStore::compareSet(
    const std::string& key,
    const std::vector<uint8_t>& expectedValue,
    const std::vector<uint8_t>& desiredValue) {
  int fd = file_open(path_.c_str(), kOpenRdWr | kOpenBinary);
  TP_CHECK(fd >= 0, "FileStore: cannot open file");
  file_flock(fd, kLockEx);
  Snapshot snapshot = decodeLog(path_);
  auto it = snapshot.latest.find(key);
  std::vector<uint8_t> current =
      it == snapshot.latest.end() ? std::vector<uint8_t>{} : it->second;
  if (current == expectedValue) {
    const uint32_t keyLen = wireLength(key.size(), "key");
    const uint32_t valLen = wireLength(desiredValue.size(), "value");
    (void)file_seek_end(fd);
    writeAll(fd, &keyLen, sizeof(keyLen));
    writeAll(fd, key.data(), key.size());
    writeAll(fd, &valLen, sizeof(valLen));
    writeAll(fd, desiredValue.data(), desiredValue.size());
    writeAll(fd, "\n", 1);
  }
  file_flock(fd, kLockUn);
  file_close(fd);
  return current;
}

std::vector<uint8_t> FileStore::get(const std::string& key) {
  int fd = file_open(path_.c_str(), kOpenRdOnly | kOpenBinary);
  TP_CHECK(fd >= 0, "FileStore: cannot open file");
  file_flock(fd, kLockSh);
  Snapshot snapshot = decodeLog(path_);
  file_flock(fd, kLockUn);
  file_close(fd);
  auto it = snapshot.latest.find(key);
  return it == snapshot.latest.end() ? std::vector<uint8_t>{} : it->second;
}

int64_t FileStore::add(const std::string& key, int64_t value) {
  int fd = file_open(path_.c_str(), kOpenRdWr | kOpenBinary);
  TP_CHECK(fd >= 0, "FileStore: cannot open file");
  file_flock(fd, kLockEx);
  Snapshot snapshot = decodeLog(path_);
  int64_t current = 0;
  auto it = snapshot.latest.find(key);
  if (it != snapshot.latest.end()) {
    try {
      current = std::stoll(
          std::string(it->second.begin(), it->second.end()));
    } catch (const std::exception&) {
      current = 0;
    }
  }
  const int64_t updated = checkedAdd(current, value);
  const std::string text = std::to_string(updated);
  const std::vector<uint8_t> bytes(text.begin(), text.end());
  const uint32_t keyLen = wireLength(key.size(), "key");
  const uint32_t valLen = wireLength(bytes.size(), "value");
  (void)file_seek_end(fd);
  writeAll(fd, &keyLen, sizeof(keyLen));
  writeAll(fd, key.data(), key.size());
  writeAll(fd, &valLen, sizeof(valLen));
  writeAll(fd, bytes.data(), bytes.size());
  writeAll(fd, "\n", 1);
  file_flock(fd, kLockUn);
  file_close(fd);
  return updated;
}

bool FileStore::deleteKey(const std::string& key) {
  int fd = file_open(path_.c_str(), kOpenRdWr | kOpenBinary);
  TP_CHECK(fd >= 0, "FileStore: cannot open file");
  file_flock(fd, kLockEx);
  Snapshot snapshot = decodeLog(path_);
  const bool existed = snapshot.latest.count(key) > 0;
  if (existed) {
    const uint32_t keyLen = wireLength(key.size(), "key");
    const uint32_t valLen = wireLength(kTombstone.size(), "value");
    (void)file_seek_end(fd);
    writeAll(fd, &keyLen, sizeof(keyLen));
    writeAll(fd, key.data(), key.size());
    writeAll(fd, &valLen, sizeof(valLen));
    writeAll(fd, kTombstone.data(), kTombstone.size());
    writeAll(fd, "\n", 1);
  }
  file_flock(fd, kLockUn);
  file_close(fd);
  return existed;
}

bool FileStore::check(const std::vector<std::string>& keys) {
  int fd = file_open(path_.c_str(), kOpenRdOnly | kOpenBinary);
  TP_CHECK(fd >= 0, "FileStore: cannot open file");
  file_flock(fd, kLockSh);
  Snapshot snapshot = decodeLog(path_);
  file_flock(fd, kLockUn);
  file_close(fd);
  for (const auto& key : keys) {
    if (snapshot.latest.find(key) == snapshot.latest.end()) {
      return false;
    }
  }
  return true;
}

int64_t FileStore::getNumKeys() {
  int fd = file_open(path_.c_str(), kOpenRdOnly | kOpenBinary);
  TP_CHECK(fd >= 0, "FileStore: cannot open file");
  file_flock(fd, kLockSh);
  Snapshot snapshot = decodeLog(path_);
  file_flock(fd, kLockUn);
  file_close(fd);
  return static_cast<int64_t>(snapshot.latest.size());
}

} // namespace distributed
} // namespace tensorplay
