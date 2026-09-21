```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# tensorplay.linalg

Common linear algebra operations.
See {ref}`Linear Algebra Stability` for some common numerical edge-cases.

## Matrix Properties

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.linalg.norm
    tensorplay.linalg.vector_norm
    tensorplay.linalg.matrix_norm
    tensorplay.linalg.diagonal
    tensorplay.linalg.det
    tensorplay.linalg.slogdet
    tensorplay.linalg.cond
    tensorplay.linalg.matrix_rank
```

## Decompositions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.linalg.cholesky
    tensorplay.linalg.qr
    tensorplay.linalg.polar
    tensorplay.linalg.lu
    tensorplay.linalg.lu_factor
    tensorplay.linalg.eig
    tensorplay.linalg.eigvals
    tensorplay.linalg.eigh
    tensorplay.linalg.eigvalsh
    tensorplay.linalg.svd
    tensorplay.linalg.svdvals
```

(linalg solvers)=

## Solvers

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.linalg.solve
    tensorplay.linalg.solve_triangular
    tensorplay.linalg.lu_solve
    tensorplay.linalg.lstsq
```

(linalg inverses)=

## Inverses

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.linalg.inv
    tensorplay.linalg.pinv
```

## Matrix Functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.linalg.matrix_exp
    tensorplay.linalg.matrix_sqrth
    tensorplay.linalg.matrix_power
```

## Matrix Products

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.linalg.cross
    tensorplay.linalg.matmul
    tensorplay.linalg.vecdot
    tensorplay.linalg.vdot
    tensorplay.linalg.multi_dot
    tensorplay.linalg.householder_product
```

## Tensor Operations

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.linalg.tensorinv
    tensorplay.linalg.tensorsolve
```

## Misc

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.linalg.vander
```

## Experimental Functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.linalg.cholesky_ex
    tensorplay.linalg.inv_ex
    tensorplay.linalg.solve_ex
    tensorplay.linalg.lu_factor_ex
    tensorplay.linalg.ldl_factor
    tensorplay.linalg.ldl_factor_ex
    tensorplay.linalg.ldl_solve
```

## Result Types

Solvers and decompositions that return several tensors package them in the
named tuples below; the `_ex` variants report failures through an ``info``
field instead of raising.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.linalg.LinAlgError
    tensorplay.linalg.EigResult
    tensorplay.linalg.EighResult
    tensorplay.linalg.LstsqResult
    tensorplay.linalg.QRResult
    tensorplay.linalg.SVDResult
    tensorplay.linalg.SlogdetResult
    tensorplay.linalg.CholeskyExResult
```
