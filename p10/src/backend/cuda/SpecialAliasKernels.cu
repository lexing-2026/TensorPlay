// The special_* operator family needs no CUDA registration.
//
// Its adapters (SpecialAliasKernels.cpp) are registered once as composites:
// they call the de-prefixed twin operator through its public entry point,
// whose CUDA kernel serves the call and whose derivative is recorded.  A
// backend registration here would take precedence over the composite and
// bypass that derivative.
