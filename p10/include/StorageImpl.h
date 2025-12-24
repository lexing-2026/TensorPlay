#pragma once

#include <cstdint>
#include <cstring>
#include <algorithm>
#include "DataPtr.h"
#include "Allocator.h"
#include "Macros.h"
#include "Exception.h"

namespace tensorplay {

struct P10_API StorageImpl {
    DataPtr data_ptr;
    size_t nbytes;
    Allocator* allocator;
    bool resizable;

    StorageImpl(size_t size, Allocator* allocator, bool resizable = true)
        : nbytes(size), allocator(allocator), resizable(resizable) {
        if (allocator) {
            data_ptr = allocator->allocate(nbytes);
        }
    }
    
    StorageImpl(DataPtr&& ptr, size_t size, Allocator* alloc, bool resizable = false)
        : data_ptr(std::move(ptr)), nbytes(size), allocator(alloc), resizable(resizable) {}
        
    void* data() {
        return data_ptr.get();
    }
    
    const void* data() const {
        return data_ptr.get();
    }
    
    void set_nbytes(size_t new_nbytes) {
        if (nbytes == new_nbytes) return;
        if (!resizable || !allocator) {
            TP_THROW(RuntimeError, "Storage is not resizable");
        }
        
        DataPtr new_data = allocator->allocate(new_nbytes);
        
        size_t copy_size = std::min(nbytes, new_nbytes);
        if (data_ptr.get() && new_data.get()) {
            std::memcpy(new_data.get(), data_ptr.get(), copy_size);
        }
        
        data_ptr = std::move(new_data);
        nbytes = new_nbytes;
    }
};

} // namespace tensorplay
