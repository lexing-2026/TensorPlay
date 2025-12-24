import tensorplay._C as _C

def is_available():
    return _C.has_mkl()
