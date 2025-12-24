#pragma once

#include "Macros.h"
#include <atomic>

#ifdef USE_ONEDNN
#include <oneapi/dnnl/dnnl.hpp>
#endif

namespace tensorplay {

class P10_API OneDNNContext {
public:
    static bool is_available();
    static bool is_enabled();
    static void set_enabled(bool enabled);
    
#ifdef USE_ONEDNN
    static dnnl::engine& get_engine();
    static dnnl::stream& get_stream();
#endif

private:
    static std::atomic<bool> enabled_;
};

} // namespace tensorplay
