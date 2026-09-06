#pragma once

#include "DispatchKey.h"
#include "Macros.h"

namespace tensorplay::impl {

struct LocalDispatchKeySet {
    DispatchKeySet included;
    DispatchKeySet excluded;
};

P10_API LocalDispatchKeySet tls_local_dispatch_key_set();
P10_API void force_tls_local_dispatch_key_set(LocalDispatchKeySet state);
P10_API void tls_set_dispatch_key_included(DispatchKey key, bool included);

class IncludeDispatchKeyGuard {
public:
    explicit IncludeDispatchKeyGuard(DispatchKeySet keys)
        : saved_(tls_local_dispatch_key_set().included) {
        auto state = tls_local_dispatch_key_set();
        state.included |= keys;
        force_tls_local_dispatch_key_set(state);
    }
    explicit IncludeDispatchKeyGuard(DispatchKey key)
        : IncludeDispatchKeyGuard(DispatchKeySet::make(key)) {}
    ~IncludeDispatchKeyGuard() {
        auto state = tls_local_dispatch_key_set();
        state.included = saved_;
        force_tls_local_dispatch_key_set(state);
    }
    IncludeDispatchKeyGuard(const IncludeDispatchKeyGuard&) = delete;
    IncludeDispatchKeyGuard& operator=(const IncludeDispatchKeyGuard&) = delete;
private:
    DispatchKeySet saved_;
};

class ExcludeDispatchKeyGuard {
public:
    explicit ExcludeDispatchKeyGuard(DispatchKeySet keys)
        : saved_(tls_local_dispatch_key_set().excluded) {
        auto state = tls_local_dispatch_key_set();
        state.excluded |= keys;
        force_tls_local_dispatch_key_set(state);
    }
    explicit ExcludeDispatchKeyGuard(DispatchKey key)
        : ExcludeDispatchKeyGuard(DispatchKeySet::make(key)) {}
    ~ExcludeDispatchKeyGuard() {
        auto state = tls_local_dispatch_key_set();
        state.excluded = saved_;
        force_tls_local_dispatch_key_set(state);
    }
    ExcludeDispatchKeyGuard(const ExcludeDispatchKeyGuard&) = delete;
    ExcludeDispatchKeyGuard& operator=(const ExcludeDispatchKeyGuard&) = delete;
private:
    DispatchKeySet saved_;
};

class ForceDispatchKeyGuard {
public:
    explicit ForceDispatchKeyGuard(LocalDispatchKeySet state)
        : saved_(tls_local_dispatch_key_set()) {
        force_tls_local_dispatch_key_set(state);
    }
    ~ForceDispatchKeyGuard() { force_tls_local_dispatch_key_set(saved_); }
    ForceDispatchKeyGuard(const ForceDispatchKeyGuard&) = delete;
    ForceDispatchKeyGuard& operator=(const ForceDispatchKeyGuard&) = delete;
private:
    LocalDispatchKeySet saved_;
};

} // namespace tensorplay::impl
