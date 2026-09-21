#pragma once

#include "DType.h"
#include "Macros.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace tensorplay {
namespace cuda {
namespace tunable {

// A recorded tuning outcome: the winning kernel identifier together with
// its measured average duration in milliseconds. The identifier "Default"
// names the library heuristic's top choice; every other identifier encodes
// one concrete cuBLASLt algorithm configuration.
struct P10_API TuningResult {
    std::string kernel;
    double time_ms = 0.0;
};

// Process-wide GEMM tuning state: the enable switches, the search budget,
// the in-memory results database and its CSV persistence.
//
// GEMM execution consults the context only while the feature is enabled;
// disabled, dispatch is exactly the untuned behavior. The database maps an
// operator signature plus a parameter signature to the kernel that measured
// fastest for that pair, so a machine is tuned once and the recorded
// winners are replayed by later runs instead of being re-measured.
class P10_API TuningContext {
public:
    // Opaque state; defined in the translation unit that implements the
    // context. Public so the implementation's file-local helpers can name
    // it.
    struct Impl;

    static TuningContext& get();

    // Master switch. While on, the cuBLASLt plan path reuses recorded
    // winners (and measures untuned shapes when tuning is on) instead of
    // relying only on per-process selection.
    void setEnabled(bool value);
    bool isEnabled() const;

    // Whether an untuned signature triggers a measurement pass. Newly found
    // winners are appended to the results file as they are measured while
    // this is on.
    void setTuningEnabled(bool value);
    bool isTuningEnabled() const;

    // Whether GEMMs that ran without a tuned choice are logged to the
    // untuned file (one line per unique signature).
    void setRecordUntuned(bool value);
    bool isRecordUntunedEnabled() const;

    // Diagnostic logging of state changes, searches and file activity.
    void setVerbose(bool value);
    bool isVerbose() const;

    // Emits one diagnostic line when verbose logging is on; the GEMM path
    // uses it to report why a recorded winner was or was not applied.
    void logVerbose(const std::string& message) const;

    // Measurement budget per candidate kernel, applied while tuning.
    // Zero disables a limit; a search always runs at least one timed
    // sample; when both limits are set the smaller one wins.
    void setMaxTuningDurationMs(int value);
    int maxTuningDurationMs() const;
    void setMaxTuningSamples(int value);
    int maxTuningSamples() const;

    // Results database. lookup returns false when no winner is recorded.
    // record inserts only entries not already present and, while tuning
    // writes are enabled, appends new entries to the results file in real
    // time.
    bool lookup(const std::string& op, const std::string& params,
                TuningResult* out) const;
    void record(const std::string& op, const std::string& params,
                const TuningResult& result);

    // Logs one line per unique signature that ran without a tuned choice.
    void recordUntuned(const std::string& op, const std::string& params);

    // CSV persistence. The stored filename receives real-time appends and
    // is rewritten by writeFile; readFile merges a file into the database
    // after its validator lines match this build. With
    // insert_device_ordinal the current device ordinal is embedded in the
    // name so one-process-per-device runs never share a file. An empty
    // filename disables file persistence entirely.
    void setFilename(const std::string& filename, bool insert_device_ordinal);
    std::string getFilename() const;

    bool readFile(const std::string& filename);
    void writeFile();

    // Number of recorded (op, params) winners.
    size_t resultsSize() const;

    // Snapshot of every recorded winner as (op, params, kernel, time_ms).
    std::vector<std::tuple<std::string, std::string, std::string, double>>
    resultsSnapshot() const;

    // Generation counter for the database and enable state. Callers that
    // cache a resolution against the context compare the value they
    // resolved under with the current one to know when to re-resolve.
    uint64_t epoch() const;

    // One-time lazy initialization: picks the default results filename and
    // reads it. Runs on the first GEMM that consults the context.
    void ensureInitialized();

private:
    TuningContext();

    TuningContext(const TuningContext&) = delete;
    TuningContext& operator=(const TuningContext&) = delete;

    Impl* impl_;
};

// Operator signature: the tunable GEMM flavor identified by dtype and
// reduced-precision compute mode.
P10_API std::string gemmOpSignature(DType dtype, bool tf32);

// Parameter signature: the shape/dispatch key of one tunable GEMM. The
// device ordinal is part of the key so winners never leak across GPUs.
P10_API std::string gemmParamsSignature(int64_t m, int64_t n, int64_t k,
                                        bool has_bias, bool other_transposed,
                                        int device);

}  // namespace tunable
}  // namespace cuda
}  // namespace tensorplay
