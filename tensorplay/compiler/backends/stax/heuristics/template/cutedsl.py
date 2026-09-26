"""The configurations worth trying for a product written in the device dialect.

That dialect is written for one shape of hardware, and the shapes a kernel may
be cut into are not all of them workable: a tile that fits is a tile whose
extents and whose cluster divide the device the way that hardware is divided.
So the configurations here are the ones the hardware admits rather than every
combination of numbers that could be written down, and a cluster shape appears
only alongside the tiles it was measured to go with.

Measuring across all of them is opt-in, and the reason is stated rather than
assumed: a configuration that does not fit is not one that measured badly, it is
one that will not launch, so a search that included them would spend its time
finding that out. The default is therefore a single known-good configuration,
and the wider sets are what a program opts into by saying it wants them measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import auto, Enum
from itertools import product

from ... import config as inductor_config


class TensorMapUpdateMode(Enum):
    """Where a tile's descriptor is refreshed: in shared memory, or in global.

    Named here rather than taken from the library that also defines it, so that
    the table below can be read without that library being present, and so that
    a value read out of a log has a name that means something on its own.
    """

    SMEM = auto()
    GMEM = auto()


@dataclass(frozen=True)
class CuTeGemmConfig:
    TILE_M: int = 128
    TILE_N: int = 192
    CLUSTER_M: int = 2
    CLUSTER_N: int = 1
    USE_2_CTA: bool = False
    TENSORMAP_UPDATE_MODE: TensorMapUpdateMode = TensorMapUpdateMode.SMEM


def get_exhaustive_groupgemm_configs() -> list[CuTeGemmConfig]:
    """Every configuration the hardware admits, for a program that will measure them all.

    The extents along the second axis do not depend on whether the tile is
    spread over one or two cooperating blocks, so they are listed once; what
    changes with that choice is which cluster shapes fit and which row extents
    go with them, so those are paired rather than varied independently.
    """

    tile_n_vals = [32, 64, 96, 128, 160, 192, 224, 256]

    # The cluster shapes that fit without the tile being spread over two blocks.
    clusters_no_2cta = [
        (1, 1),
        (1, 2),
        (1, 4),
        (1, 8),
        (1, 16),
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 1),
        (8, 2),
        (16, 1),
    ]
    # The ones that fit when it is, which is a subset: the shape has to divide
    # the two blocks as well as the tile.
    clusters_2cta = [
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 1),
        (8, 2),
        (16, 1),
    ]

    configs: list[CuTeGemmConfig] = []

    for use_2cta, cluster_set, tile_m_range in [
        (False, clusters_no_2cta, [64, 128]),
        (True, clusters_2cta, [128, 256]),
    ]:
        for tensormap_update_mode, tile_m, tile_n, (cluster_m, cluster_n) in product(
            [TensorMapUpdateMode.SMEM, TensorMapUpdateMode.GMEM],
            tile_m_range,
            tile_n_vals,
            cluster_set,
        ):
            configs.append(
                CuTeGemmConfig(
                    tile_m,
                    tile_n,
                    cluster_m,
                    cluster_n,
                    USE_2_CTA=use_2cta,
                    TENSORMAP_UPDATE_MODE=tensormap_update_mode,
                )
            )

    return configs


def get_default_groupgemm_configs() -> list[CuTeGemmConfig]:
    """The configurations measured to be worth having, for a program that wants a choice.

    A set rather than a ranking, because which of them wins depends on the shape
    being multiplied and nothing here knows the shape. They are written in the
    order they were found to be worth trying, so the first is the one to use when
    no measuring was asked for.
    """

    config_tuples = [
        (128, 256, 2, 1, False, TensorMapUpdateMode.SMEM),
        (256, 160, 2, 1, True, TensorMapUpdateMode.GMEM),
        (256, 256, 2, 1, True, TensorMapUpdateMode.GMEM),
        (64, 32, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 256, 1, 2, False, TensorMapUpdateMode.SMEM),
        (128, 256, 1, 2, False, TensorMapUpdateMode.SMEM),
        (256, 256, 2, 2, True, TensorMapUpdateMode.GMEM),
        (128, 256, 1, 2, False, TensorMapUpdateMode.GMEM),
        (64, 32, 1, 1, False, TensorMapUpdateMode.SMEM),
        (256, 256, 2, 1, True, TensorMapUpdateMode.SMEM),
        (128, 256, 1, 1, False, TensorMapUpdateMode.GMEM),
        (256, 256, 8, 1, True, TensorMapUpdateMode.GMEM),
        (64, 32, 1, 2, False, TensorMapUpdateMode.SMEM),
        (256, 192, 2, 1, True, TensorMapUpdateMode.GMEM),
        (256, 256, 2, 2, True, TensorMapUpdateMode.SMEM),
        (128, 96, 1, 2, False, TensorMapUpdateMode.SMEM),
        (64, 192, 1, 1, False, TensorMapUpdateMode.SMEM),
        (64, 64, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 192, 1, 1, False, TensorMapUpdateMode.GMEM),
        (128, 64, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 160, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 256, 1, 1, False, TensorMapUpdateMode.GMEM),
    ]

    return [CuTeGemmConfig(*args) for args in config_tuples]


def get_groupgemm_configs() -> list[CuTeGemmConfig]:
    """The configurations to offer, as wide as the program asked for.

    Measuring this dialect is opt-in and the width of the search is a separate
    choice again: a program that wants it measured may still want only the set
    worth having rather than everything the hardware admits. With neither asked
    for, the single known-good configuration is what is offered, because a
    configuration that will not launch is not a candidate that lost.
    """

    if (
        inductor_config.cutedsl_enable_autotuning
        and inductor_config.max_autotune_gemm_search_space == "EXHAUSTIVE"
    ):
        return get_exhaustive_groupgemm_configs()
    elif inductor_config.cutedsl_enable_autotuning:
        return get_default_groupgemm_configs()
    else:
        return [get_default_groupgemm_configs()[0]]
