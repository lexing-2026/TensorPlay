# Type information

`tensorplay.finfo` and `tensorplay.iinfo` describe the limits of a floating-point or integer
data type. You can query a dtype, or construct one from a tensor.

```python
import tensorplay as tp

print(tp.finfo(tp.float32))          # machine limits for float32
print(tp.finfo(tp.float32).max)      # 3.4028235e+38
print(tp.iinfo(tp.int64).max)        # 9223372036854775807
```

`finfo` works for floating-point dtypes (`float16`, `float32`, `float64`, and the complex
dtypes) and exposes values such as `max`, `min`, `eps`, `tiny`, and `resolution`. `iinfo`
works for integer dtypes (`int8` through `int64`, and the unsigned types) and exposes `min`
and `max`.

Both accept a dtype or the data type of a tensor:

```python
x = tp.tensor([1.0, 2.0])
print(tp.finfo(x.dtype).eps)         # machine epsilon for float32
```

```{eval-rst}
.. autosummary::
    :nosignatures:

    tensorplay.finfo
    tensorplay.iinfo
```

## Where to go next

- [Tensor attributes](tensor_attributes.md) — the dtype and device of a tensor.
