// Monitoring primitives: events, handler registry, windowed statistics and
// scope timers.  The core keeps per-window accumulators under a mutex; a new
// window is opened lazily on the next add/get after the previous one elapses,
// at which point the finished window is exported as an event.

#include <pybind11/chrono.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace py = pybind11;

namespace tensorplay {
namespace monitor {

namespace {
constexpr int kNumAggregations = 7;
}

// Aggregation selectors for Stat windows; values are fixed so they can be
// exchanged with external tooling.
enum class Aggregation : int {
    NONE = 0,
    VALUE = 1,
    MEAN = 2,
    COUNT = 3,
    SUM = 4,
    MAX = 5,
    MIN = 6,
};

const char* aggregationName(Aggregation agg) {
    switch (agg) {
        case Aggregation::VALUE: return "VALUE";
        case Aggregation::MEAN: return "MEAN";
        case Aggregation::COUNT: return "COUNT";
        case Aggregation::SUM: return "SUM";
        case Aggregation::MAX: return "MAX";
        case Aggregation::MIN: return "MIN";
        default: return "NONE";
    }
}

// A single event: a stable name, an epoch timestamp and a payload of scalar
// values.  Handlers receive every logged event.
struct Event {
    std::string name;
    double timestamp{0.0};
    std::unordered_map<std::string,
                       std::variant<std::string, double, int64_t, bool>>
        data;
};

// One exported statistic: which aggregation produced it, for which stat, and
// the aggregated value.
struct StatResult {
    std::string name;
    Aggregation type{Aggregation::NONE};
    double value{0.0};
};

namespace {

struct Registry {
    std::mutex mu;
    int64_t next_handle{1};
    std::unordered_map<int64_t, py::function> handlers;

    static Registry& instance() {
        static Registry r;
        return r;
    }
};

double epochNow() {
    return std::chrono::duration<double>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

void logEventLocked(const Event& e) {
    // Handlers run under the GIL because they are python callables.
    py::gil_scoped_acquire acquire;
    py::dict data;
    for (const auto& [k, v] : e.data) {
        if (std::holds_alternative<double>(v)) {
            data[k.c_str()] = std::get<double>(v);
        } else if (std::holds_alternative<int64_t>(v)) {
            data[k.c_str()] = std::get<int64_t>(v);
        } else if (std::holds_alternative<bool>(v)) {
            data[k.c_str()] = std::get<bool>(v);
        } else {
            data[k.c_str()] = std::get<std::string>(v);
        }
    }
    py::object event = py::cast(e);
    for (auto& [_, fn] : Registry::instance().handlers) {
        fn(event);
    }
}

}  // namespace

// Fixed-interval summary statistics.  add() feeds the open window; when a
// window closes (on the next add or the destructor) the aggregates are
// exported through the event handlers.
struct Stat {
    std::string name;
    std::deque<std::pair<double, double>> samples;  // (arrival, value)
    std::vector<Aggregation> aggregations;
    std::chrono::milliseconds window_size;
    int64_t max_samples;
    std::mutex mu;

    Stat(std::string name_, std::vector<Aggregation> aggs,
         std::chrono::milliseconds window, int64_t max_samples_)
        : name(std::move(name_)), aggregations(std::move(aggs)),
          window_size(window), max_samples(max_samples_) {}

    void pruneLocked(double now) {
        const double cutoff = now - window_size.count() / 1000.0;
        while (!samples.empty() && samples.front().first < cutoff) {
            samples.pop_front();
        }
    }

    void exportWindowLocked() {
        if (samples.empty()) {
            return;
        }
        double sum = 0.0, min_v = 0.0, max_v = 0.0;
        int64_t count = 0;
        for (const auto& [_, v] : samples) {
            sum += v;
            if (count == 0) {
                min_v = v;
                max_v = v;
            } else {
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
            }
            ++count;
        }
        Event e;
        e.name = "tensorplay.monitor.Stat";
        e.timestamp = epochNow();
        for (Aggregation agg : aggregations) {
            double value = 0.0;
            switch (agg) {
                case Aggregation::VALUE:
                    value = samples.back().second;
                    break;
                case Aggregation::MEAN:
                    value = count > 0 ? sum / count : 0.0;
                    break;
                case Aggregation::COUNT:
                    value = static_cast<double>(count);
                    break;
                case Aggregation::SUM:
                    value = sum;
                    break;
                case Aggregation::MAX:
                    value = max_v;
                    break;
                case Aggregation::MIN:
                    value = min_v;
                    break;
                default:
                    continue;
            }
            e.data[name + "." + aggregationName(agg)] = value;
        }
        samples.clear();
        logEventLocked(e);
    }

    void add(double v) {
        std::lock_guard<std::mutex> guard(mu);
        const double now = epochNow();
        // an aged-out window is exported whole before the new sample opens
        // the next one
        if (!samples.empty() &&
            samples.back().first < now - window_size.count() / 1000.0) {
            exportWindowLocked();
        }
        pruneLocked(now);
        if (static_cast<int64_t>(samples.size()) >= max_samples) {
            return;
        }
        samples.emplace_back(now, v);
    }

    int64_t count() {
        std::lock_guard<std::mutex> guard(mu);
        pruneLocked(epochNow());
        return static_cast<int64_t>(samples.size());
    }

    std::vector<StatResult> get() {
        std::lock_guard<std::mutex> guard(mu);
        pruneLocked(epochNow());
        std::vector<StatResult> out;
        double sum = 0.0, min_v = 0.0, max_v = 0.0;
        int64_t count_v = 0;
        for (const auto& [_, v] : samples) {
            sum += v;
            if (count_v == 0) {
                min_v = v;
                max_v = v;
            } else {
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
            }
            ++count_v;
        }
        for (Aggregation agg : aggregations) {
            StatResult r;
            r.name = name;
            r.type = agg;
            switch (agg) {
                case Aggregation::VALUE:
                    r.value = samples.empty() ? 0.0 : samples.back().second;
                    break;
                case Aggregation::MEAN:
                    r.value = count_v > 0 ? sum / count_v : 0.0;
                    break;
                case Aggregation::COUNT:
                    r.value = static_cast<double>(count_v);
                    break;
                case Aggregation::SUM:
                    r.value = sum;
                    break;
                case Aggregation::MAX:
                    r.value = max_v;
                    break;
                case Aggregation::MIN:
                    r.value = min_v;
                    break;
                default:
                    continue;
            }
            out.push_back(r);
        }
        return out;
    }

    ~Stat() {
        std::lock_guard<std::mutex> guard(mu);
        exportWindowLocked();
    }
};

// A timed scope: entering starts a timer, leaving logs the elapsed seconds.
struct WaitCounter {
    std::string name;
    std::chrono::steady_clock::time_point start;

    explicit WaitCounter(std::string name_) : name(std::move(name_)) {}

    void enter() { start = std::chrono::steady_clock::now(); }

    void exit() {
        const double elapsed =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start)
                .count();
        Event e;
        e.name = "tensorplay.monitor.WaitCounter";
        e.timestamp = epochNow();
        e.data[name] = elapsed;
        logEventLocked(e);
    }
};

int64_t registerEventHandler(py::function fn) {
    auto& r = Registry::instance();
    std::lock_guard<std::mutex> guard(r.mu);
    const int64_t handle = r.next_handle++;
    r.handlers[handle] = fn;
    return handle;
}

void unregisterEventHandler(int64_t handle) {
    auto& r = Registry::instance();
    std::lock_guard<std::mutex> guard(r.mu);
    r.handlers.erase(handle);
}

void logEventPy(const Event& e) { logEventLocked(e); }

}  // namespace monitor
}  // namespace tensorplay

void init_monitor(py::module_& m) {
    using namespace tensorplay::monitor;
    py::module_ mon = m.def_submodule("_monitor");

    py::enum_<Aggregation>(mon, "Aggregation")
        .value("NONE", Aggregation::NONE)
        .value("VALUE", Aggregation::VALUE)
        .value("MEAN", Aggregation::MEAN)
        .value("COUNT", Aggregation::COUNT)
        .value("SUM", Aggregation::SUM)
        .value("MAX", Aggregation::MAX)
        .value("MIN", Aggregation::MIN)
        .export_values();

    py::class_<Event>(mon, "Event")
        .def(py::init<std::string, double,
                      std::unordered_map<
                          std::string,
                          std::variant<std::string, double, int64_t, bool>>>(),
             py::arg("name"), py::arg("timestamp"), py::arg("data"))
        .def_readwrite("name", &Event::name)
        .def_readwrite("timestamp", &Event::timestamp)
        .def_readwrite("data", &Event::data)
        .def("__repr__", [](const Event& e) {
            return "Event(name='" + e.name + "', data_size=" +
                   std::to_string(e.data.size()) + ")";
        });

    py::class_<StatResult>(mon, "StatResult")
        .def_readonly("name", &StatResult::name)
        .def_readonly("type", &StatResult::type)
        .def_readonly("value", &StatResult::value)
        .def("__repr__", [](const StatResult& r) {
            return "StatResult(name='" + r.name +
                   "', type=" + aggregationName(r.type) + ", value=" +
                   std::to_string(r.value) + ")";
        });

    py::class_<Stat>(mon, "Stat")
        .def(py::init<std::string, std::vector<Aggregation>,
                      std::chrono::milliseconds, int64_t>(),
             py::arg("name"), py::arg("aggregations"), py::arg("window_size"),
             py::arg("max_samples") = std::numeric_limits<int64_t>::max())
        .def("add", &Stat::add, py::call_guard<py::gil_scoped_release>())
        .def("count", &Stat::count, py::call_guard<py::gil_scoped_release>())
        .def("get", &Stat::get, py::call_guard<py::gil_scoped_release>())
        .def_readonly("name", &Stat::name);

    py::class_<WaitCounter>(mon, "_WaitCounter")
        .def(py::init<std::string>())
        .def("__enter__", [](WaitCounter& self) {
            self.enter();
            return &self;
        })
        .def("__exit__",
             [](WaitCounter& self, const py::object&, const py::object&,
                const py::object&) {
                 self.exit();
                 return false;
             });

    mon.def("register_event_handler", &registerEventHandler,
            py::arg("callback"));
    mon.def("unregister_event_handler", &unregisterEventHandler,
            py::arg("handle"));
    mon.def("log_event", &logEventPy, py::arg("event"));
}
