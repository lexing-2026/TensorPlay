
#pragma once
#include "Graph.h"
#include "Macros.h"
#include <functional>
#include <string>
#include <vector>
#include <memory>
#include <map>

namespace tensorplay {
namespace stax {

// --- Pass Infrastructure ---

class STAX_API Pass {
public:
    virtual ~Pass() = default;
    virtual std::string name() const = 0;
    virtual bool run(Graph& graph) = 0; // Returns true if graph was modified
};

class STAX_API PassRegistry {
public:
    using Creator = std::function<std::unique_ptr<Pass>()>;
    
    static PassRegistry& instance() {
        static PassRegistry inst;
        return inst;
    }
    
    void registerPass(const std::string& name, Creator creator) {
        creators_[name] = creator;
    }
    
    std::unique_ptr<Pass> createPass(const std::string& name) {
        if (creators_.count(name)) {
            return creators_[name]();
        }
        return nullptr;
    }
    
    std::vector<std::string> listPasses() const {
        std::vector<std::string> names;
        for (auto& kv : creators_) names.push_back(kv.first);
        return names;
    }

private:
    std::map<std::string, Creator> creators_;
};

#define REGISTER_STAX_PASS(name, ClassName) \
    struct ClassName##Register { \
        ClassName##Register() { \
            PassRegistry::instance().registerPass(name, []() { return std::make_unique<ClassName>(); }); \
        } \
    }; \
    static ClassName##Register global_##ClassName##_register;

// --- Optimizer ---

class STAX_API Optimizer {
public:
    void addPass(const std::string& pass_name);
    void run(Graph& graph);

private:
    std::vector<std::string> passes_;
};

// --- Legacy (for compatibility) ---
STAX_API void fuseGraph(Graph& graph);

} // namespace stax
} // namespace tensorplay
