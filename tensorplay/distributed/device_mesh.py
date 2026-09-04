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


class _MeshNDimensionality(int):
    def __new__(cls, value: int) -> "_MeshNDimensionality":
        return int.__new__(cls, value)

    def __call__(self) -> int:
        return int(self)


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
    if flat_idx < 0 or flat_idx >= math.prod(sizes):
        raise IndexError("flat mesh index is out of range")
    rem = int(flat_idx)
    coords = []
    for stride, size in zip(strides, sizes):
        coord, rem = divmod(rem, stride)
        if coord >= size:
            raise IndexError("flat mesh index is out of range")
        coords.append(coord)
    return tuple(coords)


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
        _backend_override=None,
        _axis_root_dims=None,
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

            def flatten(value):
                if not isinstance(value, (list, tuple)):
                    return (), [int(value)]
                if not value:
                    return (0,), []
                child_shapes = []
                flat_values = []
                for child in value:
                    child_shape, child_values = flatten(child)
                    child_shapes.append(child_shape)
                    flat_values.extend(child_values)
                if any(shape != child_shapes[0] for shape in child_shapes[1:]):
                    raise ValueError("all mesh dimensions must be rectangular")
                return (len(value),) + child_shapes[0], flat_values

            sizes, rank_map = flatten(mesh)
            if not sizes:
                sizes = (1,)
        else:
            if _rank_map is None or _sizes is None:
                raise TypeError("The mesh argument is required")
            rank_map = list(_rank_map)
            sizes = list(_sizes)

        if not sizes or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in sizes
        ):
            raise ValueError("mesh dimensions must be positive integers")
        sizes = tuple(sizes)

        total = math.prod(sizes)
        if len(rank_map) != total:
            raise AssertionError(
                f"rank map length {len(rank_map)} != product of sizes {total}"
            )
        if any(not isinstance(rank, int) or rank < 0 for rank in rank_map):
            raise ValueError("mesh ranks must be non-negative integers")
        if len(set(rank_map)) != len(rank_map):
            raise ValueError("mesh ranks must be unique")

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
        self._rank_map = list(rank_map)

        def nest(values, shape):
            if len(shape) == 1:
                return list(values)
            width = math.prod(shape[1:])
            return [nest(values[i * width:(i + 1) * width], shape[1:])
                    for i in range(shape[0])]

        self.mesh = nest(self._rank_map, self._sizes)

        self._root_mesh = _root_mesh
        self._thread_id: int | None = None
        self._flatten_mapping: dict[str, "DeviceMesh"] = {}
        self._dim_groups: dict[int, Any] = {}
        self._backend_override = _backend_override
        if _axis_root_dims is not None:
            if len(_axis_root_dims) != len(self._sizes):
                raise ValueError("axis metadata must match mesh dimensions")
            self._axis_root_dims = tuple(
                tuple(int(dim) for dim in dims) for dims in _axis_root_dims
            )
        else:
            root = self._get_root_mesh()
            root_names = getattr(root, "_mesh_dim_names", None)
            axis_root_dims = []
            for index, name in enumerate(self._mesh_dim_names or ()):
                if root_names is not None and name in root_names:
                    axis_root_dims.append((root_names.index(name),))
                else:
                    axis_root_dims.append((index,))
            if len(axis_root_dims) != len(self._sizes):
                axis_root_dims = [(index,) for index in range(len(self._sizes))]
            self._axis_root_dims = tuple(axis_root_dims)

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
        if len(coords) != len(self._sizes):
            raise ValueError("coordinate rank does not match mesh dimensions")
        if any(coord < 0 or coord >= size
               for coord, size in zip(coords, self._sizes)):
            raise IndexError("mesh coordinate is out of range")
        return sum(c * s for c, s in zip(coords, self._strides))

    def _ranks_along_dim(self, dim: int, coords: tuple[int, ...]) -> list[int]:
        """Return ranks on the mesh line containing ``coords``."""
        from itertools import product

        ranges = [
            range(size) if axis == dim else (coords[axis],)
            for axis, size in enumerate(self._sizes)
        ]
        return [
            self._rank_map[self._pos_of_coords(combo)]
            for combo in product(*ranges)
        ]

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
            if self._mesh_dim_names is None:
                raise KeyError(mesh_dim)
            try:
                dim = self._mesh_dim_names.index(mesh_dim)
            except ValueError as exc:
                raise KeyError(mesh_dim) from exc
        else:
            if isinstance(mesh_dim, bool):
                raise TypeError("mesh_dim must be an integer or string")
            dim = int(mesh_dim)
        if dim < 0 or dim >= self.ndim():
            raise IndexError("mesh dimension is out of range")
        if dim in self._dim_groups:
            return self._dim_groups[dim]
        my_global = dist.get_rank()
        try:
            pos = self._rank_map.index(my_global)
        except ValueError as e:
            raise RuntimeError(
                f"Rank {my_global} is not part of this DeviceMesh"
            ) from e
        coords = self._coords_of(pos)
        group = None
        from itertools import product

        other_ranges = [
            range(size) if axis != dim else (0,)
            for axis, size in enumerate(self._sizes)
        ]
        for line_coords in product(*other_ranges):
            ranks = self._ranks_along_dim(dim, tuple(line_coords))
            kwargs = {"ranks": ranks}
            if self._backend_override is not None:
                kwargs["backend"] = self._backend_override
            candidate = dist.new_group(**kwargs)
            if my_global in ranks:
                group = candidate
        if group is None:
            raise RuntimeError(f"Rank {my_global} is not part of mesh dimension {dim}")
        self._dim_groups[dim] = group
        return group

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def get_group(self, mesh_dim=None):
        if mesh_dim is None:
            if int(self.ndim) == 1:
                mesh_dim = 0
            else:
                raise RuntimeError(
                    "Must specify mesh_dim for multi-dimensional mesh."
                )
        return self._get_or_create_group_for_dim(mesh_dim)

    def size(self, mesh_dim: int | None = None) -> int:
        if mesh_dim is None:
            return len(self._rank_map)
        if isinstance(mesh_dim, str):
            if self._mesh_dim_names is None:
                raise KeyError(mesh_dim)
            mesh_dim = self._mesh_dim_names.index(mesh_dim)
        mesh_dim = int(mesh_dim)
        if mesh_dim < 0 or mesh_dim >= self.ndim():
            raise IndexError("mesh dimension is out of range")
        return self._sizes[mesh_dim]

    @property
    def ndim(self) -> int:
        return _MeshNDimensionality(len(self._sizes))

    @property
    def shape(self) -> tuple[int, ...]:
        return self._sizes

    def numel(self) -> int:
        return math.prod(self._sizes)

    def get_rank(self) -> int:
        return dist.get_rank()

    def get_all_groups(self) -> list[Any]:
        return [self.get_group(index) for index in range(int(self.ndim))]

    @property
    def mesh_dim_names(self):
        return self._mesh_dim_names

    @property
    def ndimension(self) -> int:
        return len(self._sizes)

    def get_local_rank(self, mesh_dim=None) -> int:
        try:
            my_global = dist.get_rank()
        except RuntimeError:
            my_global = 0
        try:
            pos = self._rank_map.index(my_global)
        except ValueError as exc:
            raise RuntimeError(
                f"Rank {my_global} is not part of this DeviceMesh"
            ) from exc
        coords = self._coords_of(pos)
        if mesh_dim is None:
            if int(self.ndim) != 1:
                raise RuntimeError("Must specify mesh_dim.")
            mesh_dim = 0
        if isinstance(mesh_dim, str):
            if self._mesh_dim_names is None:
                raise KeyError(mesh_dim)
            dim = self._mesh_dim_names.index(mesh_dim)
        else:
            dim = int(mesh_dim)
        if dim < 0 or dim >= self.ndim():
            raise IndexError("mesh dimension is out of range")
        return coords[dim]

    def get_coordinate(self) -> tuple[int, ...] | None:
        """Returns this rank's coordinate in the mesh, or None if absent."""
        try:
            my_global = dist.get_rank()
        except RuntimeError:
            my_global = 0
        try:
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
        if not mesh_dim_names:
            raise ValueError("at least one mesh dimension must be selected")
        if len(set(mesh_dim_names)) != len(mesh_dim_names):
            raise ValueError("mesh dimensions must be unique")
        if tuple(mesh_dim_names) == self._mesh_dim_names:
            return self
        try:
            dims = tuple(self._mesh_dim_names.index(n) for n in mesh_dim_names)
        except ValueError as exc:
            root = self._get_root_mesh()
            if len(mesh_dim_names) == 1 and mesh_dim_names[0] in root._flatten_mapping:
                return root._flatten_mapping[mesh_dim_names[0]]
            raise KeyError(mesh_dim_names) from exc
        sub_sizes = tuple(self._sizes[d] for d in dims)
        coordinate = self.get_coordinate()
        if coordinate is None:
            coordinate = tuple(0 for _ in self._sizes)
        # A slice over a subset of dimensions is the line through this rank;
        # selecting every dimension retains the complete mesh.
        selected_ranges = [range(self._sizes[d]) for d in dims]
        from itertools import product

        rank_map = []
        for sel in product(*selected_ranges):
            dim_to_coord = dict(zip(dims, sel))
            coords = [dim_to_coord.get(d, coordinate[d])
                      for d in range(len(self._sizes))]
            rank_map.append(self._rank_map[self._pos_of_coords(coords)])
        sub = DeviceMesh(
            device_type=self.device_type,
            _rank_map=rank_map,
            _sizes=sub_sizes,
            mesh_dim_names=mesh_dim_names,
            _root_mesh=self._get_root_mesh(),
            _backend_override=self._backend_override,
            _axis_root_dims=tuple(
                self._get_axis_root_dims()[dim] for dim in dims
            ),
        )
        _mesh_resources.create_sub_mesh(
            self._get_root_mesh(), sub, mesh_dim_names
        )
        return sub

    def _flatten(self, mesh_dim_name: str | None = None, backend_override=None) -> "DeviceMesh":
        if not self._mesh_dim_names:
            raise RuntimeError("Cannot flatten a mesh without dimension names")
        if mesh_dim_name is None:
            mesh_dim_name = "_".join(self._mesh_dim_names)
        if not isinstance(mesh_dim_name, str) or not mesh_dim_name:
            raise ValueError("flattened mesh dimension name must be non-empty")
        if self.ndim == 1 and mesh_dim_name == self._mesh_dim_names[0]:
            return self
        root = self._get_root_mesh()
        if root._mesh_dim_names and mesh_dim_name in root._mesh_dim_names:
            raise ValueError(
                f"{mesh_dim_name} already exists in the root mesh dimensions"
            )
        existing = root._flatten_mapping.get(mesh_dim_name)
        if existing is not None:
            if existing._rank_map != self._rank_map or existing._sizes != (len(self._rank_map),):
                raise ValueError(
                    f"flattened mesh dimension {mesh_dim_name!r} already has a different layout"
                )
            return existing
        flattened = DeviceMesh(
            self.device_type,
            list(self._rank_map),
            mesh_dim_names=(mesh_dim_name,),
            _root_mesh=root,
            _backend_override=(
                backend_override if backend_override is not None else self._backend_override
            ),
            _axis_root_dims=(
                tuple(
                    dim
                    for axis in self._get_axis_root_dims()
                    for dim in axis
                ),
            ),
        )
        root._flatten_mapping[mesh_dim_name] = flattened
        return flattened

    def _get_axis_root_dims(self) -> tuple[tuple[int, ...], ...]:
        value = getattr(self, "_axis_root_dims", None)
        if value is not None:
            return value
        root = self._get_root_mesh()
        root_names = getattr(root, "_mesh_dim_names", None)
        result = []
        for index, name in enumerate(self._mesh_dim_names or ()):
            if root_names is not None and name in root_names:
                result.append((root_names.index(name),))
            else:
                result.append((index,))
        if len(result) != len(self._sizes):
            result = [(index,) for index in range(len(self._sizes))]
        self._axis_root_dims = tuple(result)
        return self._axis_root_dims

    @staticmethod
    def _concatenate(device_mesh_list: list["DeviceMesh"]) -> "DeviceMesh":
        if not device_mesh_list:
            raise ValueError("at least one DeviceMesh is required")
        first = device_mesh_list[0]
        root = first._get_root_mesh()
        if any(not isinstance(mesh, DeviceMesh) for mesh in device_mesh_list):
            raise TypeError("all entries must be DeviceMesh instances")
        if any(mesh._get_root_mesh() is not root for mesh in device_mesh_list):
            raise RuntimeError(
                "Cannot concatenate DeviceMeshes derived from different device meshes"
            )
        if any(mesh.device_type != first.device_type for mesh in device_mesh_list):
            raise RuntimeError("Cannot concatenate DeviceMeshes with different device types")

        names: list[str] = []
        axes: list[tuple[int, ...]] = []
        for mesh in device_mesh_list:
            mesh_names = mesh.mesh_dim_names
            if mesh_names is None or len(mesh_names) != int(mesh.ndim):
                raise ValueError("all DeviceMeshes must have mesh dimension names")
            mesh_axes = mesh._get_axis_root_dims()
            if len(mesh_axes) != len(mesh_names):
                raise ValueError("mesh axis metadata is invalid")
            names.extend(str(name) for name in mesh_names)
            axes.extend(mesh_axes)

        root_sizes = root._sizes
        used_dims: set[int] = set()
        axis_sizes: list[int] = []
        for axis in axes:
            if not axis or any(dim < 0 or dim >= len(root_sizes) for dim in axis):
                raise RuntimeError("Cannot concatenate invalid mesh axes")
            if used_dims.intersection(axis):
                raise RuntimeError(
                    f"Cannot concatenate overlapping meshes: {device_mesh_list}"
                )
            used_dims.update(axis)
            axis_sizes.append(math.prod(root_sizes[dim] for dim in axis))

        coordinate = root.get_coordinate()
        if coordinate is None:
            coordinate = tuple(0 for _ in root_sizes)
        rank_map: list[int] = []
        from itertools import product

        for axis_coordinates in product(*(range(size) for size in axis_sizes)):
            full_coordinate = list(coordinate)
            for axis, axis_coordinate in zip(axes, axis_coordinates):
                remaining = int(axis_coordinate)
                for index, root_dim in enumerate(axis):
                    inner_size = math.prod(
                        root_sizes[other_dim] for other_dim in axis[index + 1 :]
                    )
                    value = remaining // inner_size
                    remaining %= inner_size
                    full_coordinate[root_dim] = value
            rank_map.append(root._rank_map[root._pos_of_coords(full_coordinate)])

        result = DeviceMesh(
            first.device_type,
            _rank_map=rank_map,
            _sizes=tuple(axis_sizes),
            mesh_dim_names=tuple(names),
            _root_mesh=root,
            _backend_override=first._backend_override,
            _axis_root_dims=tuple(axes),
        )
        output_dim = 0
        for mesh in device_mesh_list:
            for dim in range(int(mesh.ndim)):
                if dim in mesh._dim_groups:
                    result._dim_groups[output_dim] = mesh._dim_groups[dim]
                output_dim += 1
        return result

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

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_dim_groups"] = {}
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._dim_groups = {}

    @classmethod
    def from_group(cls, group, device_type=None, mesh=None,
                   mesh_dim_names=None) -> "DeviceMesh":
        """Construct a DeviceMesh from one or more existing process groups."""
        device_type = device_type or "cuda"
        groups = list(group) if isinstance(group, (list, tuple)) else [group]
        if not groups:
            raise ValueError("at least one process group is required")
        if mesh_dim_names is not None and len(mesh_dim_names) != len(groups):
            raise ValueError("mesh_dim_names must match the number of groups")

        group_ranks = [dist.get_process_group_ranks(item) for item in groups]
        if len(groups) == 1:
            ranks = group_ranks[0]
            if mesh is None:
                mesh = ranks
            else:
                if isinstance(mesh, tp.Tensor):
                    mesh = mesh.cpu().tolist()
                flat = []

                def collect(value):
                    if isinstance(value, (list, tuple)):
                        for item in value:
                            collect(item)
                    else:
                        flat.append(int(value))

                collect(mesh)
                if flat != ranks:
                    raise ValueError("mesh must list process-group ranks in order")
        elif mesh is None:
            raise ValueError("mesh is required when multiple groups are provided")

        result = cls(device_type=device_type, mesh=mesh,
                     mesh_dim_names=mesh_dim_names)
        result._dim_groups = {index: item for index, item in enumerate(groups)}
        return result


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

    if not isinstance(device_type, str) or not device_type or not device_type.isalpha():
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
    if not mesh_shape or any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in mesh_shape
    ):
        raise ValueError("mesh_shape must contain positive integers")
    mesh_size = math.prod(mesh_shape)
    if mesh_size != world_size:
        raise RuntimeError(
            f"mesh_shape product ({mesh_size}) must equal world size ({world_size})"
        )

    return DeviceMesh(
        device_type=device_type,
        mesh=_reshape_ranks(list(range(world_size)), mesh_shape),
        mesh_dim_names=mesh_dim_names,
        _backend_override=backend_override,
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
