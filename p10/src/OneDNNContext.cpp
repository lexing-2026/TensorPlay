#include "OneDNNContext.h"

namespace tensorplay {

#ifdef USE_ONEDNN
std::atomic<bool> OneDNNContext::enabled_(true);

bool OneDNNContext::is_available() {
    return true;
}

dnnl::engine& OneDNNContext::get_engine() {
    static dnnl::engine* eng = new dnnl::engine(dnnl::engine::kind::cpu, 0);
    return *eng;
}

dnnl::stream& OneDNNContext::get_stream() {
    static dnnl::stream* s = new dnnl::stream(get_engine());
    return *s;
}

#else
std::atomic<bool> OneDNNContext::enabled_(false);

bool OneDNNContext::is_available() {
    return false;
}
#endif

bool OneDNNContext::is_enabled() {
    return enabled_.load();
}

void OneDNNContext::set_enabled(bool enabled) {
    enabled_.store(enabled);
}

} // namespace tensorplay
