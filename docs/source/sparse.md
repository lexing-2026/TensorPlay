```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# tensorplay.sparse

## Accelerating nn.Linear with semi-structured sparsity

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.sparse.semi_structured.to_sparse_semi_structured
    tensorplay.sparse.semi_structured.SparseSemiStructuredTensorCUSPARSELT
    tensorplay.sparse.semi_structured.SparseSemiStructuredTensorCUTLASS
```

### Tensor methods and sparse

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.dense_dim
    tensorplay.sparse_dim
    tensorplay.sparse_mask
    tensorplay.to_sparse
    tensorplay.to_sparse_csr
    tensorplay.to_sparse_csc
    tensorplay.to_sparse_bsr
    tensorplay.to_sparse_bsc
    tensorplay.to_dense
    tensorplay.values
    tensorplay.coalesce
    tensorplay.sparse_resize_
    tensorplay.sparse_resize_and_clear_
    tensorplay.is_coalesced
    tensorplay.indices
    tensorplay.crow_indices
    tensorplay.col_indices
    tensorplay.row_indices
    tensorplay.ccol_indices
```

### Torch functions specific to sparse Tensors

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.functional.sparse_coo_tensor
    tensorplay.functional.sparse_csr_tensor
    tensorplay.functional.sparse_csc_tensor
    tensorplay.functional.sparse_bsr_tensor
    tensorplay.functional.sparse_bsc_tensor
    tensorplay.functional.sparse_compressed_tensor
    tensorplay.sparse._ops.sum
    tensorplay.sparse._ops.addmm
    tensorplay.sparse._ops.mm
    tensorplay.functional.sspaddmm
    tensorplay.functional.hspmm
    tensorplay.functional.smm
    tensorplay.sparse._ops.softmax
    tensorplay.sparse._ops.log_softmax
    tensorplay.sparse._construction.spdiags
```

### Other functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.sparse._invariants.check_sparse_tensor_invariants
    tensorplay.sparse._gradcheck.as_sparse_gradcheck
```

