// GEMM tuning state: enable switches, search budget, results database and
// CSV persistence for the cuBLASLt plan path.

#include "CudaTunable.h"

#include "CUDARuntime.h"
#include "Exception.h"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tensorplay {
namespace cuda {
namespace tunable {

namespace {

// Measurement defaults: every candidate is timed for at most this many
// iterations or this many milliseconds, whichever allows fewer runs.
constexpr int kDefaultMaxTuningDurationMs = 30;
constexpr int kDefaultMaxTuningSamples = 100;

constexpr const char* kDefaultResultsFilename = "tunableop_results.csv";
constexpr const char* kDefaultUntunedFilename = "tunableop_untuned.csv";
constexpr const char* kKernelDefault = "Default";

// Recorded winner identifier for the library heuristic's top choice.
const std::string& kernelDefault() {
    static const std::string value(kKernelDefault);
    return value;
}

// Bump whenever the CSV layout or the kernel identifier grammar changes in
// a way old files cannot express; files stamped with another value are
// rejected instead of misread.
constexpr const char* kFileFormatVersion = "1";

// Embeds a device ordinal in a filename: a "%d" token is replaced in
// place, otherwise the ordinal lands just before the extension (or at the
// end when there is none).
std::string insertOrdinal(const std::string& filename, int device) {
    std::string out = filename;
    const std::string ord = std::to_string(device);
    const std::string token("%d");
    auto found = out.find(token);
    if (found != std::string::npos) {
        out.replace(found, token.length(), ord);
        return out;
    }
    found = out.rfind('.');
    if (found != std::string::npos) {
        out.insert(found, ord);
    } else {
        out.append(ord);
    }
    return out;
}

std::string formatTime(double ms) {
    std::ostringstream ss;
    ss << ms;
    return ss.str();
}

}  // namespace

struct TuningContext::Impl {
    // Hot-path flags are atomic: GEMM execution reads them without a lock
    // while the Python API writes them.
    std::atomic<bool> enabled{false};
    std::atomic<bool> tuning_enabled{true};
    std::atomic<bool> record_untuned{false};
    std::atomic<bool> verbose{false};
    std::atomic<int> max_duration_ms{kDefaultMaxTuningDurationMs};
    std::atomic<int> max_samples{kDefaultMaxTuningSamples};
    std::atomic<uint64_t> epoch{1};

    mutable std::mutex lock;
    std::string filename;  // resolved (ordinal-embedded) results path
    bool filename_set = false;

    // op signature -> params signature -> winner
    std::unordered_map<std::string,
                       std::unordered_map<std::string, TuningResult>> results;
    std::unordered_set<std::string> untuned_logged;

    bool validators_cached = false;
    std::unordered_map<std::string, std::string> validators;

    std::once_flag init_once;
    std::ofstream append_stream;
    std::string append_stream_filename;
    std::ofstream untuned_stream;
};

namespace {

void logLine(const TuningContext::Impl& impl, const std::string& message) {
    if (impl.verbose.load(std::memory_order_relaxed)) {
        std::fprintf(stderr, "[tensorplay.cuda.tunable] %s\n", message.c_str());
    }
}

// Validator values describe the exact software and hardware a winner was
// measured on; a file whose validators differ from the current build is
// rejected because its entries no longer describe runnable choices.
void ensureValidatorsLocked(TuningContext::Impl& impl) {
    if (impl.validators_cached) {
        return;
    }
    cudaDeviceProp prop{};
    const int device = currentDevice();
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        TP_WARN("could not query device properties; tuning file validators "
                "cannot be established");
        return;
    }
    std::ostringstream device_id;
    device_id << prop.major << '.' << prop.minor << ':' << prop.name;
    impl.validators["TP_TUNABLEOP_FORMAT"] = kFileFormatVersion;
    impl.validators["CUDA_DEVICE"] = device_id.str();
    impl.validators["CUBLASLT_VERSION"] = std::to_string(cublasLtGetVersion());
    impl.validators_cached = true;
}

// Results path for output: the explicitly set filename, or the default
// with the current device ordinal embedded.
std::string outputFilenameLocked(TuningContext::Impl& impl) {
    if (impl.filename_set) {
        return impl.filename;
    }
    return insertOrdinal(kDefaultResultsFilename, currentDevice());
}

// Appends one result line, opening the file (and stamping validators into
// a fresh one) as needed. Caller holds impl.lock.
void appendResultLocked(TuningContext::Impl& impl, const std::string& op,
                        const std::string& params, const TuningResult& result) {
    const std::string target = outputFilenameLocked(impl);
    if (target.empty()) {
        return;
    }
    if (!impl.append_stream.is_open() || impl.append_stream_filename != target) {
        if (impl.append_stream.is_open()) {
            impl.append_stream.close();
        }
        bool needs_validators = true;
        {
            std::ifstream probe(target);
            needs_validators = !probe.good() ||
                               probe.peek() == std::ifstream::traits_type::eof();
        }
        impl.append_stream.open(target, std::ios::out | std::ios::app);
        if (!impl.append_stream.good()) {
            TP_WARN("could not open '", target, "' for appending tuning results");
            impl.append_stream.clear();
            return;
        }
        impl.append_stream_filename = target;
        if (needs_validators) {
            ensureValidatorsLocked(impl);
            for (const auto& [key, value] : impl.validators) {
                impl.append_stream << "Validator," << key << ',' << value << '\n';
            }
            impl.append_stream.flush();
        }
    }
    impl.append_stream << op << ',' << params << ',' << result.kernel << ','
                       << formatTime(result.time_ms) << '\n';
    impl.append_stream.flush();
}

// A file is acceptable when its validator key set matches ours exactly and
// every value agrees; anything else means the file was written by another
// build or machine.
bool validatorsMatchLocked(TuningContext::Impl& impl,
                           const std::unordered_map<std::string, std::string>& file) {
    bool ok = true;
    for (const auto& [key, value] : impl.validators) {
        auto it = file.find(key);
        if (it == file.end()) {
            TP_WARN("tuning results file lacks the validator '", key, "'");
            ok = false;
        } else if (it->second != value) {
            TP_WARN("tuning results validator '", key, "' mismatch: file has '",
                    it->second, "', this build has '", value, "'");
            ok = false;
        }
    }
    for (const auto& [key, value] : file) {
        if (impl.validators.find(key) == impl.validators.end()) {
            TP_WARN("tuning results file carries unknown validator '", key,
                    "' (value '", value, "')");
            ok = false;
        }
    }
    return ok;
}

}  // namespace

TuningContext::TuningContext() : impl_(new Impl()) {}

TuningContext& TuningContext::get() {
    static TuningContext* context = new TuningContext();
    return *context;
}

void TuningContext::setEnabled(bool value) {
    impl_->enabled.store(value, std::memory_order_relaxed);
    impl_->epoch.fetch_add(1, std::memory_order_relaxed);
    logLine(*impl_, value ? "enabled" : "disabled");
}

bool TuningContext::isEnabled() const {
    return impl_->enabled.load(std::memory_order_relaxed);
}

void TuningContext::setTuningEnabled(bool value) {
    impl_->tuning_enabled.store(value, std::memory_order_relaxed);
    logLine(*impl_, value ? "tuning enabled" : "tuning disabled");
}

bool TuningContext::isTuningEnabled() const {
    return impl_->tuning_enabled.load(std::memory_order_relaxed);
}

void TuningContext::setRecordUntuned(bool value) {
    impl_->record_untuned.store(value, std::memory_order_relaxed);
    if (!value) {
        std::lock_guard<std::mutex> guard(impl_->lock);
        if (impl_->untuned_stream.is_open()) {
            impl_->untuned_stream.close();
        }
        impl_->untuned_logged.clear();
    }
    logLine(*impl_, value ? "untuned recording enabled" : "untuned recording disabled");
}

bool TuningContext::isRecordUntunedEnabled() const {
    return impl_->record_untuned.load(std::memory_order_relaxed);
}

void TuningContext::setVerbose(bool value) {
    impl_->verbose.store(value, std::memory_order_relaxed);
    logLine(*impl_, value ? "verbose output enabled" : "verbose output disabled");
}

bool TuningContext::isVerbose() const {
    return impl_->verbose.load(std::memory_order_relaxed);
}

void TuningContext::setMaxTuningDurationMs(int value) {
    impl_->max_duration_ms.store(value < 0 ? 0 : value, std::memory_order_relaxed);
}

int TuningContext::maxTuningDurationMs() const {
    return impl_->max_duration_ms.load(std::memory_order_relaxed);
}

void TuningContext::setMaxTuningSamples(int value) {
    impl_->max_samples.store(value < 0 ? 0 : value, std::memory_order_relaxed);
}

int TuningContext::maxTuningSamples() const {
    return impl_->max_samples.load(std::memory_order_relaxed);
}

bool TuningContext::lookup(const std::string& op, const std::string& params,
                           TuningResult* out) const {
    std::lock_guard<std::mutex> guard(impl_->lock);
    auto it = impl_->results.find(op);
    if (it == impl_->results.end()) {
        return false;
    }
    auto jt = it->second.find(params);
    if (jt == it->second.end()) {
        return false;
    }
    if (out != nullptr) {
        *out = jt->second;
    }
    return true;
}

void TuningContext::record(const std::string& op, const std::string& params,
                           const TuningResult& result) {
    bool is_new = false;
    {
        std::lock_guard<std::mutex> guard(impl_->lock);
        auto& kernel_map = impl_->results[op];
        is_new = kernel_map.find(params) == kernel_map.end();
        if (is_new) {
            kernel_map.emplace(params, result);
        }
    }
    if (!is_new) {
        return;
    }
    logLine(*impl_, "new winner " + op + "(" + params + ") -> " + result.kernel +
                        " " + formatTime(result.time_ms) + " ms");
    // Real-time persistence: entries land in the results file the moment
    // they are measured, unless the run is collecting untuned signatures
    // instead of tuning.
    if (impl_->tuning_enabled.load(std::memory_order_relaxed) &&
        !impl_->record_untuned.load(std::memory_order_relaxed)) {
        std::lock_guard<std::mutex> guard(impl_->lock);
        appendResultLocked(*impl_, op, params, result);
    }
}

void TuningContext::recordUntuned(const std::string& op, const std::string& params) {
    std::lock_guard<std::mutex> guard(impl_->lock);
    if (!impl_->untuned_stream.is_open()) {
        const std::string target = insertOrdinal(kDefaultUntunedFilename, currentDevice());
        impl_->untuned_stream.open(target, std::ios::out | std::ios::app);
        if (!impl_->untuned_stream.good()) {
            TP_WARN("could not open '", target, "' for recording untuned GEMMs");
            impl_->untuned_stream.clear();
            return;
        }
    }
    const std::string key = op + ',' + params;
    if (impl_->untuned_logged.insert(key).second) {
        impl_->untuned_stream << op << ',' << params << '\n';
        impl_->untuned_stream.flush();
        logLine(*impl_, "recorded untuned " + key);
    }
}

void TuningContext::setFilename(const std::string& filename,
                                bool insert_device_ordinal) {
    {
        std::lock_guard<std::mutex> guard(impl_->lock);
        // An empty filename turns file persistence off; no ordinal is
        // embedded into it.
        impl_->filename =
            (filename.empty() || !insert_device_ordinal)
                ? filename
                : insertOrdinal(filename, currentDevice());
        impl_->filename_set = true;
        // Reopen lazily so subsequent appends target the new path.
        if (impl_->append_stream.is_open()) {
            impl_->append_stream.close();
        }
        impl_->append_stream_filename.clear();
    }
    logLine(*impl_, "results filename set to '" + filename + "'");
}

std::string TuningContext::getFilename() const {
    std::lock_guard<std::mutex> guard(impl_->lock);
    return impl_->filename;
}

bool TuningContext::readFile(const std::string& filename) {
    std::string target = filename;
    {
        std::lock_guard<std::mutex> guard(impl_->lock);
        ensureValidatorsLocked(*impl_);
        if (target.empty()) {
            target = outputFilenameLocked(*impl_);
        }
    }
    if (target.empty()) {
        return false;
    }

    std::unordered_map<std::string, std::string> file_validators;
    std::vector<std::tuple<std::string, std::string, TuningResult>> entries;
    std::ifstream file(target);
    if (!file) {
        logLine(*impl_, "could not open '" + target + "' for reading results");
        return false;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> parts;
        std::stringstream stream(line);
        std::string part;
        while (std::getline(stream, part, ',')) {
            parts.push_back(part);
        }
        if (parts[0] == "Validator" && parts.size() >= 3) {
            file_validators[parts[1]] = parts[2];
        } else if (parts.size() >= 4) {
            entries.emplace_back(parts[0], parts[1],
                                 TuningResult{parts[2], std::atof(parts[3].c_str())});
        } else if (parts.size() == 3) {
            // The measured time is optional.
            entries.emplace_back(parts[0], parts[1], TuningResult{parts[2], 0.0});
        } else {
            logLine(*impl_, "could not parse line: " + line);
        }
    }

    {
        std::lock_guard<std::mutex> guard(impl_->lock);
        if (!validatorsMatchLocked(*impl_, file_validators)) {
            logLine(*impl_, "rejected tuning results file '" + target + "'");
            return false;
        }
        // Merge: entries already measured in this process win over the
        // file's, the file fills the rest.
        for (auto& [op, params, result] : entries) {
            impl_->results[op].emplace(params, result);
        }
    }
    impl_->epoch.fetch_add(1, std::memory_order_relaxed);
    logLine(*impl_, "read " + std::to_string(entries.size()) +
                        " tuning results from '" + target + "'");
    return true;
}

void TuningContext::writeFile() {
    std::string target;
    {
        std::lock_guard<std::mutex> guard(impl_->lock);
        ensureValidatorsLocked(*impl_);
        target = outputFilenameLocked(*impl_);
        if (target.empty()) {
            TP_WARN("no results filename configured; nothing to write");
            return;
        }
        std::ofstream out(target, std::ios::out | std::ios::trunc);
        if (!out) {
            TP_WARN("could not open '", target, "' for writing tuning results");
            return;
        }
        for (const auto& [key, value] : impl_->validators) {
            out << "Validator," << key << ',' << value << '\n';
        }
        for (const auto& [op, kernel_map] : impl_->results) {
            for (const auto& [params, result] : kernel_map) {
                out << op << ',' << params << ',' << result.kernel << ','
                    << formatTime(result.time_ms) << '\n';
            }
        }
    }
    logLine(*impl_, "wrote tuning results to '" + target + "'");
}

size_t TuningContext::resultsSize() const {
    std::lock_guard<std::mutex> guard(impl_->lock);
    size_t size = 0;
    for (const auto& [op, kernel_map] : impl_->results) {
        size += kernel_map.size();
    }
    return size;
}

std::vector<std::tuple<std::string, std::string, std::string, double>>
TuningContext::resultsSnapshot() const {
    std::lock_guard<std::mutex> guard(impl_->lock);
    std::vector<std::tuple<std::string, std::string, std::string, double>> out;
    out.reserve(impl_->results.size());
    for (const auto& [op, kernel_map] : impl_->results) {
        for (const auto& [params, result] : kernel_map) {
            out.emplace_back(op, params, result.kernel, result.time_ms);
        }
    }
    return out;
}

uint64_t TuningContext::epoch() const {
    return impl_->epoch.load(std::memory_order_relaxed);
}

void TuningContext::ensureInitialized() {
    std::call_once(impl_->init_once, [this]() {
        std::string target;
        bool read_it = false;
        {
            std::lock_guard<std::mutex> guard(impl_->lock);
            if (!impl_->filename_set) {
                impl_->filename = insertOrdinal(kDefaultResultsFilename, currentDevice());
            }
            // Collecting untuned signatures and consuming tuned ones are
            // separate workflows; a collection run starts from an empty
            // database on purpose.
            read_it = !impl_->record_untuned.load(std::memory_order_relaxed);
            target = impl_->filename;
        }
        if (read_it && !target.empty()) {
            readFile(target);
        }
    });
}

std::string gemmOpSignature(DType dtype, bool tf32) {
    if (tf32) {
        return "GemmTunableOp_Tf32";
    }
    return std::string("GemmTunableOp_") + toString(dtype);
}

std::string gemmParamsSignature(int64_t m, int64_t n, int64_t k, bool has_bias,
                                bool other_transposed, int device) {
    std::ostringstream ss;
    ss << (other_transposed ? "taT" : "taN") << "_m" << m << "_n" << n << "_k" << k
       << "_bias" << (has_bias ? 1 : 0) << "_dev" << device;
    return ss.str();
}

}  // namespace tunable
}  // namespace cuda
}  // namespace tensorplay
