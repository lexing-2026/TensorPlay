# Complex numbers

TensorPlay supports complex tensors — elements stored as a real and an imaginary part. Complex
dtypes are `tensorplay.complex64` and `tensorplay.complex128`.

```python
import tensorplay as tp

x = tp.tensor([1 + 2j, 3 - 4j])
print(x.dtype)          # tensorplay.complex64
print(x.real)           # the real part
print(x.imag)           # the imaginary part
```

## Creating complex tensors

`tensorplay.complex` builds a complex tensor from separate real and imaginary tensors, and
`tensorplay.polar` builds one from magnitude and angle:

```python
re = tp.tensor([1.0, 3.0])
im = tp.tensor([2.0, -4.0])
z = tp.complex(re, im)             # tensor([1.+2.j, 3.-4.j])

mag = tp.tensor([2.0, 1.0])
ang = tp.tensor([0.0, 1.5708])
p = tp.polar(mag, ang)             # magnitude, angle
```

## Reading and converting

`tensorplay.real` and `tensorplay.imag` extract the real and imaginary parts. `conj` returns
the complex conjugate and `resolve_conj` produces a tensor whose conjugation is applied;
`view_as_real` reinterprets a complex tensor as a float tensor with an extra trailing
dimension of size 2, while `view_as_complex` removes that pairing:

```python
z = tp.tensor([1 + 2j, 3 + 4j])
print(tp.conj(z))                   # [1.-2.j, 3.-4.j]
print(tp.view_as_real(z).shape)     # (2, 2) — the trailing 2 is (real, imag)
```

`tensorplay.is_complex` reports whether a tensor has a complex dtype.

```{eval-rst}
.. autosummary::
    :nosignatures:

    tensorplay.complex
    tensorplay.polar
    tensorplay.view_as_complex
    tensorplay.view_as_real
    tensorplay.is_complex
    tensorplay.conj
    tensorplay.resolve_conj
    tensorplay.real
    tensorplay.imag
```

## Where to go next

- [Tensors](tensorplay.md) — the full tensor API.
- [Type information](type_info.md) — the limits of a complex dtype.
