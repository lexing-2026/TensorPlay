"""Backend-specific graph partitioning passes."""

from .cudagraphs import CudaGraphsSupport, partition_cudagraphs

__all__ = ["CudaGraphsSupport", "partition_cudagraphs"]
