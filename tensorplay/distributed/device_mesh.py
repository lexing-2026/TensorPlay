#
# plain sizes/strides bookkeeping; the public API (init_device_mesh,
# get_group, get_local_rank, get_coordinate, submesh __getitem__, context
# manager, from_group) is preserved.

import math
import threading
from typing import Any

import tensorplay as tp
import tensorplay.distributed as dist


__all__ = ["DeviceMesh", "init_device_mesh"]


class _MeshEnv(threading.local):
    def __init__(self) -> None:
        # root_mesh_to_mesh
        self.root_to_flat_mesh: dict[Any, dict[str, Any]] = {}
        self.mesh_stack: list[Any] = []

    @staticmethod
    def get() -> "_MeshEnv":
        if not hasattr(_MeshEnv, "_local"):
            _MeshEnv._local = _MeshEnv()
        return _MeshEnv._local


class _MeshResources:

    def __init__(self) -> None:
        # map from root_mesh to list of all meshes with root_mesh as parent
        self.root_to_2d_mesh: dict = {}

    def create_sub_mesh(
        self, root_mesh, sub_mesh, mesh_dim_names
    ) -> None:
        self.root_to_2d_mesh.setdefault(root_mesh, {})[mesh_dim_names] = sub_mesh


_mesh_resources = _MeshResources()


def _get_device_handle(device_type: str = "cuda"):
    return getattr(tp, device_type, None)


def _flatten(sizes):
    out = []
    strides = [1] * len(sizes)
    acc = 1
    for i in reversed(range(len(sizes))):
        strides[i] = acc
        acc *= sizes[i]
    for idx in range(math.prod(sizes)):
        coords = []
        rem = idx
        for s in sizes:
            rem, c = divmod(rem, s)
            coords.append(c)
        coords.reverse()
        flat = sum(c * st for c, st in zip(coords, strides))
        out.append(flat)
    return out


def _coord_at(sizes, strides, flat_idx):
    coords = []
    rem = flat_idx
    for d in sizes:
        coords.append(rem // strides[0] if False else 0)
        break
    # general row-major decode using cumulative strides of this dim group
    return coords


class DeviceMesh:
    """

    The mesh is an n-d array whose values are global ranks. Process groups
    are created per mesh dimension so collectives can run on each dimension
    independently.

    Example::

        >>> from tensorplay.distributed.device_mesh import init_device_mesh
        >>> mesh = init_device_mesh("cuda", mesh_shape=(2, 4),
        ...                         mesh_dim_names=("dp", "tp"))
    """

    def __init__(
        self,
        device_type: str,
        mesh=None,
        *,
        mesh_dim_names=None,
        _dim_group_names=None,
        _rank_map=None,
        _sizes=None,
        _strides=None,
        _root_mesh=None,
    ):
        if mesh is not None and (_rank_map is not None or _sizes is not None):
            raise TypeError(
                "Cannot provide internal fields when passing an explicit mesh"
            )
        if mesh is not None:
            if isinstance(mesh, tp.Tensor):
                mesh = mesh.cpu().tolist()
            if isinstance(mesh, int):
                mesh = [mesh]
            try:
                flat = [int(r) for r in mesh]
                nested = len(mesh) > 0 and isinstance(mesh[0], (list, tuple))
                if not nested:
                    mesh = [[r] for r in flat]
                    sizes = [len(mesh)]
                else:
                    sizes = [len(mesh), len(mesh[0])]
            except TypeError:
                sizes = [len(d) for d in mesh] if isinstance(mesh[0], (list, tuple)) \
                    else [len(mesh)]
            rank_map = [int(r) for row in mesh for r in (
                row if isinstance(row, (list, tuple)) else [row])]
        else:
            if _rank_map is None or _sizes is None:
                raise TypeError("The mesh argument is required")
            rank_map = list(_rank_map)
            sizes = list(_sizes)

        total = math.prod(sizes)
        if len(rank_map) != total:
            raise AssertionError(
                f"rank map length {len(rank_map)} != product of sizes {total}"
            )

        # row-major strides
        strides = [1] * len(sizes)
        acc = 1
        for i in reversed(range(len(sizes))):
            strides[i] = acc
            acc *= sizes[i]

        if mesh_dim_names is not None:
            if len(set(mesh_dim_names)) != len(mesh_dim_names):
                raise ValueError("Each mesh_dim_name must be unique.")
            if len(mesh_dim_names) != len(sizes):
                raise ValueError(
                    "mesh_shape and mesh_dim_names should have same length!"
                )
            self._mesh_dim_names = tuple(mesh_dim_names)
        else:
            self._mesh_dim_names = None

        self.device_type = device_type
        self._sizes = tuple(sizes)
        self._strides = tuple(strides)
        self._rank_map = rank_map  # position in mesh -> global rank
        self.mesh = (
            [[rank_map[i * sizes[1] + j] for j in range(sizes[1])]
             for i in range(sizes[0])] if len(sizes) == 2
            else [rank_map[i] for i in range(total)] if len(sizes) == 1
            else None
        )

        self._root_mesh = _root_mesh
        self._thread_id: int | None = None
        self._flatten_mapping: dict[str, "DeviceMesh"] = {}

        if _dim_group_names is not None:
            self._dim_group_names = list(_dim_group_names)
        else:
            self._dim_group_names = self._init_process_groups()

    # ------------------------------------------------------------------
    # process-group setup
    # ------------------------------------------------------------------
    def _my_rank(self) -> int:
        try:
            return dist.get_rank()
        except Exception:
            return -1

    def _coords_of(self, pos: int) -> tuple[int, ...]:
        coords = []
        rem = pos
        for d, st in zip(self._sizes, self._strides):
            coords.append(rem // st)
            rem %= st
        return tuple(coords)

    def _pos_of_coords(self, coords) -> int:
        return sum(c * s for c, s in zip(coords, self._strides))

    def _ranks_along_dim(self, dim: int, coord: int) -> list[int]:
        """Global ranks sharing `coord` on every axis except `dim`."""
        ranks = []
        my_pos_coords = None
        base_coord = None
        if self._root_mesh is None:
            my_pos = None
            for p in range(len(self._rank_map)):
                pass
        # enumerate all positions whose coordinate vector matches `coord`
        # on every axis except `dim`
        ranges = []
        for d, size in enumerate(self._sizes):
            ranges.append([coord] if d == dim else range(size))
        from itertools import product

        for combo in product(*ranges):
            pos = self._pos_of_coords(combo)
            ranks.append(self._rank_map[pos])
        return sorted(set(ranks))

    def _init_process_groups(self) -> list[str]:
        names = []
        for dim in range(len(self._sizes)):
            name = (self._mesh_dim_names[dim] if self._mesh_dim_names
                    else f"dim_{dim}")
            names.append(name)
        # Defer actual subgroup creation until get_group is called; the
        # per-dim groups are created lazily by the SPMD ranks together.
        self._lazy_groups = True
        return names

    def _get_or_create_group_for_dim(self, mesh_dim) :
        """Create (once) the subgroup along `mesh_dim` containing this rank."""
        if isinstance(mesh_dim, str):
            dim = self._mesh_dim_names.index(mesh_dim) \
                if self._mesh_dim_names else None
            if dim is None:
                raise KeyError(mesh_dim)
        else:
            dim = int(mesh_dim)
        my_global = dist.get_rank()
        # find my coordinate along each axis via rank_map lookup
        try:
            pos = self._rank_map.index(my_global)
        except ValueError as e:
            raise RuntimeError(
                f"Rank {my_global} is not part of this DeviceMesh"
            ) from e
        coords = self._coords_of(pos)
        ranks = self._ranks_along_dim(dim, coords[dim])
        if len(ranks) == 1:
            # single-rank line: reuse default group semantics via new_group
            pg = dist.new_group(ranks=ranks)
        else:
            pg = dist.new_group(ranks=ranks)
        return pg

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_group(self, mesh_dim=None):
        if mesh_dim is None:
            if self._mesh_dim_names and len(self._mesh_dim_names) == 1:
                mesh_dim = 0
            else:
                raise RuntimeError(
                    "Must specify mesh_dim for multi-dimensional mesh."
                )
        return self._get_or_create_group_for_dim(mesh_dim)

    def size(self, mesh_dim: int | None = None) -> int:
        if mesh_dim is None:
            return len(self._rank_map)
        return self._sizes[mesh_dim]

    def ndim(self) -> int:
        return len(self._sizes)

    @property
    def mesh_dim_names(self):
        return self._mesh_dim_names

    @property
    def ndimension(self) -> int:
        return len(self._sizes)

    def get_local_rank(self, mesh_dim=None) -> int:
        my_global = dist.get_rank()
        pos = self._rank_map.index(my_global)
        coords = self._coords_of(pos)
        if mesh_dim is None:
            if self._mesh_dim_names and len(self._mesh_dim_names) == 1:
                mesh_dim = 0
            else:
                raise RuntimeError("Must specify mesh_dim.")
        dim = (self._mesh_dim_names.index(mesh_dim)
               if isinstance(mesh_dim, str) else int(mesh_dim))
        return coords[dim]

    def get_coordinate(self) -> tuple[int, ...] | None:
        """Returns this rank's coordinate in the mesh, or None if absent."""
        try:
            my_global = dist.get_rank()
            pos = self._rank_map.index(my_global)
        except ValueError:
            return None
        return self._coords_of(pos)

    def __getitem__(self, mesh_dim_names) -> "DeviceMesh":
        if isinstance(mesh_dim_names, str):
            mesh_dim_names = (mesh_dim_names,)
        if self._mesh_dim_names is None:
            raise RuntimeError(
                "No `mesh_dim_names` found; cannot slice the mesh."
            )
        dims = tuple(self._mesh_dim_names.index(n) for n in mesh_dim_names)
        sub_sizes = tuple(self._sizes[d] for d in dims)
        sub_strides = tuple(self._strides[d] for d in dims)
        # Build sliced rank map preserving selected axes' ordering.
        rank_map = []
        other_dims = [d for d in range(len(self._sizes)) if d not in dims]
        other_ranges = [range(self._sizes[d]) for d in other_dims]
        from itertools import product

        for sel in product(*[
            range(self._sizes[d]) for d in range(len(self._sizes))
            if d in dims
        ]):
            dim_to_coord = dict(zip(dims, sel))
            for others in product(*other_ranges) if other_ranges else [()]:
                it = iter(others)
                coords = [
                    next(it) if d not in dims else dim_to_coord[d]
                    for d in range(len(self._sizes))
                ]
                rank_map.append(self._rank_map[self._pos_of_coords(coords)])
        sub = DeviceMesh(
            device_type=self.device_type,
            _rank_map=rank_map,
            _sizes=sub_sizes,
            mesh_dim_names=mesh_dim_names,
            _root_mesh=self._get_root_mesh(),
        )
        _mesh_resources.create_sub_mesh(
            self._get_root_mesh(), sub, mesh_dim_names
        )
        return sub

    def _get_root_mesh(self) -> "DeviceMesh":
        return self._root_mesh if self._root_mesh else self

    def __enter__(self) -> "DeviceMesh":
        _MeshEnv.get().mesh_stack.append(self)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        _MeshEnv.get().mesh_stack.pop()

    def __repr__(self) -> str:
        if self._mesh_dim_names:
            dims_repr = ", ".join(
                f"{k}={v}" for k, v in zip(self._mesh_dim_names, self._sizes)
            )
        else:
            dims_repr = str(tuple(self._sizes))
        return f"DeviceMesh({dims_repr}, '{self.device_type}')"

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, DeviceMesh):
            return False
        return (
            self._rank_map == other._rank_map
            and self._sizes == other._sizes
            and self.device_type == other.device_type
            and self._mesh_dim_names == other._mesh_dim_names
        )

    def __hash__(self):
        return hash((
            tuple(self._rank_map), tuple(self._sizes),
            self.device_type, self._mesh_dim_names,
        ))

    @classmethod
    def from_group(cls, group, device_type=None, mesh=None,
                   mesh_dim_names=None) -> "DeviceMesh":
        """Construct a 1-D DeviceMesh from an existing ProcessGroup."""
        if mesh is not None:
            raise NotImplementedError(
                "Multi-dim from_group pending; use a 1d group."
            )
        device_type = device_type or "cuda"
        if mesh_dim_names is not None and len(mesh_dim_names) != 1:
            raise ValueError("1d group maps to exactly one mesh_dim_name")
        ranks = dist.get_process_group_ranks(group)
        return cls(device_type=device_type, mesh=[ranks],
                   mesh_dim_names=mesh_dim_names)


def init_device_mesh(
    device_type: str,
    mesh_shape: tuple[int, ...],
    *,
    mesh_dim_names: tuple[str, ...] | None = None,
    backend_override=None,
) -> DeviceMesh:
    """

    This creates a DeviceMesh with an n-dimensional array layout, where `n`
    is the length of `mesh_shape`. If `mesh_dim_names` is provided, each
    dimension is labeled as `mesh_dim_names[i]`.

    .. note::
        Follows SPMD: ensure `mesh_shape` is identical across all ranks.

    Example::

        >>> mesh_1d = init_device_mesh("cuda", mesh_shape=(8,))
        >>> mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 8),
        ...                             mesh_dim_names=("dp", "tp"))
    """
    if mesh_dim_names is not None:
        if len(set(mesh_dim_names)) != len(mesh_dim_names):
            raise RuntimeError(
                "Each mesh_dim_name must be unique. "
                f"Found repeated mesh_dim_name in mesh_dim_names {mesh_dim_names}"
            )
        if len(mesh_shape) != len(mesh_dim_names):
            raise RuntimeError(
                "mesh_shape and mesh_dim_names should have same length! "
                f"Found len(mesh_dim_names): {len(mesh_dim_names)} and "
                f"len(mesh_shape):{len(mesh_shape)}."
            )

    if device_type and not device_type.isalpha():
        raise RuntimeError(
            f"Device type with index is not supported but got {device_type}. ",
            "If you maintained a 'tp.device' object, it's recommended to "
            "pass in 'device.type'.",
        )
    if not dist.is_initialized():
        raise RuntimeError(
            "init_device_mesh requires tensorplay.distributed to be "
            "initialized first (call dist.init_process_group)."
        )

    world_size = dist.get_world_size()
    if math.prod(mesh_shape) > world_size:
        raise RuntimeError(
            f"Created more ranks than world size! {math.prod(mesh_shape)} > {world_size}."
        )

    return DeviceMesh(
        device_type=device_type,
        mesh=list(range(world_size)).reshape(list(mesh_shape))
        if False else _reshape_ranks(list(range(world_size)), mesh_shape),
        mesh_dim_names=mesh_dim_names,
    )


def _reshape_ranks(ranks: list[int], shape: tuple[int, ...]):
    """Row-major reshape of a flat rank list into nested lists (n-d)."""
    if len(shape) == 1:
        return ranks
    if len(shape) == 2:
        r, c = shape
        return [ranks[i * c : (i + 1) * c] for i in range(r)]
    # generic n-d nesting
    outer = shape[0]
    chunk = len(ranks) // outer
    rest = shape[1:]
    return [_reshape_ranks(ranks[i * chunk : (i + 1) * chunk], rest)
            for i in range(outer)]
