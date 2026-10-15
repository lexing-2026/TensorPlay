#include "backend/mps/MetalShaderLibrary.h"

#include "Exception.h"
#include "backend/mps/MPSDevice.h"

#include <dispatch/dispatch.h>
#include <string>

// Generated from the compiled .metallib by tools/write_metallib_headers.py.
// Defines `tensorplay::mps::kQuantizedMetallib` / `kQuantizedMetallibSize`.
#include "mps/Quantized_metallib.h"

namespace tensorplay {
namespace mps {

MetalShaderLibrary::MetalShaderLibrary(const unsigned char* data,
                                       unsigned long size) {
  NSError* error = nil;
  dispatch_data_t dd = dispatch_data_create(
      data, size, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
  library_ = [MPSDevice::getInstance()->device() newLibraryWithData:dd
                                                                error:&error];
  dispatch_release(dd);
  if (!library_) {
    TP_THROW(RuntimeError,
             "Failed to load embedded metallib: ",
             error ? [[error localizedDescription] UTF8String]
                    : "unknown error");
  }
}

MetalShaderLibrary::~MetalShaderLibrary() {
  [library_ release];
  library_ = nil;
}

MTLComputePipelineState_t MetalShaderLibrary::getPipelineStateForFunc(
    const std::string& fname) {
  auto it = pipeline_cache_.find(fname);
  if (it != pipeline_cache_.end()) {
    return it->second;
  }
  NSError* error = nil;
  MTLFunction_t func = [library_ newFunctionWithName:
      [NSString stringWithUTF8String:fname.c_str()]];
  if (!func) {
    TP_THROW(RuntimeError, "Metal shader function not found: ", fname);
  }
  MTLComputePipelineState_t cpl =
      [[library_ device] newComputePipelineStateWithFunction:func error:&error];
  [func release];
  if (!cpl) {
    TP_THROW(RuntimeError,
             "Failed to create compute pipeline state for ",
             fname,
             ": ",
             error ? [[error localizedDescription] UTF8String]
                    : "unknown error");
  }
  pipeline_cache_.emplace(fname, cpl);
  return cpl;
}

MetalShaderLibrary& MetalShaderLibrary::getBundledLibrary() {
  static MetalShaderLibrary lib(kQuantizedMetallib, kQuantizedMetallibSize);
  return lib;
}

} // namespace mps
} // namespace tensorplay