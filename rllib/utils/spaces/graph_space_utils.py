"""Structure traversal that treats graph observations as leaves.

dm_tree descends into a `GraphInstance` (it is a NamedTuple) and stacks a batch
of graphs as if they had one shape. These functions instead follow the
conventions the `_reference_*` implementations at the bottom of this file
define:

  * A `GraphInstance`, or a tuple whose first item is one, is a leaf. A tuple
    of graphs is a batch of graphs, and its length may differ from one
    structure to the next.
  * `map_structure` and `map_structure_with_path` pass a bare
    `GraphInstance` to `func` as a 1-tuple `(g,)`, i.e. a batch of one.
  * A Mapping is walked, and rebuilt as its own type, in sorted key order.
  * Any other tuple or list is walked item by item and rebuilt as a plain
    tuple or list.
  * Anything else, None included, is a leaf.

The public functions walk each structure once. The reference versions made four
passes (an unbatching copy, a shape check, a flatten and an unflatten), each
built on `isinstance(.., Mapping)`, an ABC check, at every node. For a batch of
graph observations those passes were 90-95% of the time spent batching it and
converting it to tensors. The real work, `np.stack` and `torch.from_numpy`,
was the rest.

The fast path does not try to support everything. It handles each structure
whose shape it can show is the same as the first one's. In every other case the
reference code may pair leaves wrongly without saying so, or may raise. Those
cases go straight to the reference code, before `func` is called even once. So
the results, the errors, and the order `func` is called in are always the
reference's. There is one exception. Suppose a Mapping type cannot be rebuilt
from its (key, value) pairs. The reference raises before calling `func`; these
functions raise once `func` has run.
"""

import gymnasium as gym
from collections.abc import Mapping
import numpy as np
import torch

_GRAPH = gym.spaces.GraphInstance

# What a node of a structure is. The first three are leaves.
_LEAF, _GRAPH_ONE, _GRAPHS, _TUPLE, _LIST, _MAPPING = range(6)

# Types already found to be leaves: no Graph, tuple, list or Mapping
# subclass. Whether a tuple holds graphs depends on what is in it, so a
# tuple type is never cached.
_LEAF_TYPES = {np.ndarray, torch.Tensor, type(None), int, float, bool, str, bytes}


def _kind(x):
    t = type(x)
    if t in _LEAF_TYPES:
        return _LEAF
    if t is dict:
        return _MAPPING
    if t is tuple:
        return _GRAPHS if x and isinstance(x[0], _GRAPH) else _TUPLE
    if t is _GRAPH:
        return _GRAPH_ONE
    if t is list:
        return _LIST
    # Subclasses, in the order the reference tests them.
    if isinstance(x, _GRAPH):
        return _GRAPH_ONE
    if isinstance(x, tuple):
        return _GRAPHS if len(x) > 0 and isinstance(x[0], _GRAPH) else _TUPLE
    if isinstance(x, list):
        return _LIST
    if isinstance(x, Mapping):
        return _MAPPING
    _LEAF_TYPES.add(t)
    return _LEAF


class _Mismatch(Exception):
    """The structures are not shaped alike: the reference decides what happens."""


# A treedef is None for a leaf, (_TUPLE | _LIST, [child treedefs]), or
# (_MAPPING, type, sorted keys, [child treedefs]).

def _gather(structs, leaves, paths, path, dicts_only):
    """Check `structs` against structs[0], appending each leaf's arguments to
    `leaves` (and its path to `paths`, when given). Returns structs[0]'s
    treedef. Raises _Mismatch wherever the reference would not pair the
    structures leaf for leaf."""
    s0 = structs[0]
    k = _kind(s0)

    if k <= _GRAPHS:
        if len(structs) == 1:
            leaves.append(((s0,) if k == _GRAPH_ONE else s0,))
        else:
            args = []
            for s in structs:
                ks = _kind(s)
                # A container against a leaf: the reference flattens it into
                # a different number of leaves. A plain leaf against a graph:
                # the reference refuses it.
                if ks > _GRAPHS or (k != _LEAF and ks == _LEAF):
                    raise _Mismatch
                args.append((s,) if ks == _GRAPH_ONE else s)
            leaves.append(tuple(args))
        if paths is not None:
            paths.append(path)
        return None

    if k == _TUPLE or k == _LIST:
        n = len(s0)
        for s in structs[1:]:
            if _kind(s) != k or len(s) != n:
                raise _Mismatch
        children = []
        for i in range(n):
            children.append(_gather(
                [s[i] for s in structs], leaves, paths,
                path + (i,) if paths is not None else None, dicts_only))
        return (k, children)

    # Mapping.
    if dicts_only and not isinstance(s0, dict):
        # map_structure_with_path's reference walks only dicts, and makes any
        # other Mapping a leaf.
        raise _Mismatch
    try:
        keys = sorted(s0.keys())
    except TypeError:
        raise _Mismatch
    for s in structs[1:]:
        # keys(), not len(s): SampleBatch's len is its row count.
        if _kind(s) != _MAPPING or len(s.keys()) != len(keys) or \
                any(key not in s for key in keys):
            raise _Mismatch
    children = []
    for key in keys:
        children.append(_gather(
            [s[key] for s in structs], leaves, paths,
            path + (key,) if paths is not None else None, dicts_only))
    return (_MAPPING, type(s0), keys, children)


def _build(treedef, it):
    if treedef is None:
        return next(it)
    if treedef[0] == _TUPLE:
        return tuple([_build(child, it) for child in treedef[1]])
    if treedef[0] == _LIST:
        return [_build(child, it) for child in treedef[1]]
    _, mapping_type, keys, children = treedef
    # A generator, as the reference passes: a Mapping type's constructor sees
    # the same argument it did.
    return mapping_type(
        (key, _build(child, it)) for key, child in zip(keys, children))


def _check_map_args(func, structures, kwargs):
    if not callable(func):
        raise TypeError(f"func must be callable, got: {type(func)}")
    if not structures:
        raise ValueError("Must provide at least one structure")
    check_types = kwargs.pop("check_types", True)
    if kwargs:
        raise ValueError(
            f"Only valid keyword argument is `check_types`, not: `{kwargs.keys()}`")
    return check_types


def map_structure(func, *structures, **kwargs):
    """Maps `func` through given structures, preserving variable-length tuples of GraphInstance."""
    check_types = _check_map_args(func, structures, kwargs)
    leaves = []
    try:
        treedef = _gather(structures, leaves, None, None, False)
    except _Mismatch:
        return _reference_map_structure(func, *structures, check_types=check_types)
    return _build(treedef, iter([func(*args) for args in leaves]))


def map_structure_with_path(func, *structures, **kwargs):
    """Maps `func(path, *leaves)` through given structures.

    A path is the tuple of indices and keys leading to the leaf, as in
    `tree.map_structure_with_path`:

    >>> map_structure_with_path(lambda path, v: (path, v**2), [{"foo": 42}])
    [{'foo': ((0, 'foo'), 1764)}]

    Graphs are leaves, as in `map_structure`. Only `check_types` is a valid
    keyword argument. Structures not shaped alike are the reference's to
    accept or refuse.
    """
    check_types = _check_map_args(func, structures, kwargs)
    leaves, paths = [], []
    try:
        treedef = _gather(structures, leaves, paths, (), True)
    except _Mismatch:
        return _reference_map_structure_with_path(
            func, *structures, check_types=check_types)
    results = [func(path, *args) for path, args in zip(paths, leaves)]
    mapped = _build(treedef, iter(results))
    # The reference flattens what `func` returned before laying it back out
    # along the first structure. That changes nothing unless `func` returned a
    # container, which flattens into several leaves (or none). In that case,
    # repeat the reference's flatten and unflatten, along with its errors.
    for result in results:
        if _kind(result) > _GRAPHS:
            return _reference_unflatten_as(
                unbatch_graph_instance(structures[0]), _reference_flatten(mapped))
    return mapped


def _flatten_into(x, out):
    k = _kind(x)
    if k <= _GRAPHS:
        out.append(x)
    elif k == _TUPLE or k == _LIST:
        for item in x:
            _flatten_into(item, out)
    else:
        try:
            for key in sorted(x.keys()):
                _flatten_into(x[key], out)
        except TypeError as e:
            raise TypeError("Cannot flatten dict with non-sortable keys") from e


def flatten(structure):
    """Flattens a possibly nested structure into a list."""
    out = []
    _flatten_into(structure, out)
    return out


def _unflatten_from(x, it):
    k = _kind(x)
    if k <= _GRAPHS:
        return next(it)
    if k == _TUPLE:
        return tuple([_unflatten_from(item, it) for item in x])
    if k == _LIST:
        return [_unflatten_from(item, it) for item in x]
    return type(x)((key, _unflatten_from(x[key], it)) for key in sorted(x.keys()))


def unflatten_as(structure, flat_sequence):
    """Unflattens a flat sequence into a given structure, preserving variable-length GraphInstance tuples."""
    flat = flat_sequence if isinstance(flat_sequence, (list, tuple)) else list(flat_sequence)
    it = iter(flat)
    try:
        result = _unflatten_from(structure, it)
    except (StopIteration, TypeError):
        # Too few elements, or unsortable keys: the reference's error.
        return _reference_unflatten_as(structure, flat)
    for _ in it:
        return _reference_unflatten_as(structure, flat)  # too many: same
    return result


# The reference implementations. They define the behaviour above, and they
# handle every pair of structures not shaped alike. Everything after this
# line is the original module, with the four functions above renamed.

def _reference_map_structure(func, *structures, **kwargs):
    """Maps `func` through given structures, preserving variable-length tuples of GraphInstance."""

    if not callable(func):
        raise TypeError(f"func must be callable, got: {type(func)}")

    if not structures:
        raise ValueError("Must provide at least one structure")

    check_types = kwargs.pop("check_types", True)
    if kwargs:
        raise ValueError(
            f"Only valid keyword argument is `check_types`, not: `{kwargs.keys()}`")

    structures = [unbatch_graph_instance(structure)
                  for structure in structures]

    # Ensure all structures have the same shape
    for other in structures[1:]:
        assert_same_structure(structures[0], other, check_types=check_types)

    # Flatten each structure
    flat_structures = list(map(_reference_flatten, structures))

    # Apply function element-wise
    mapped_values = [func(*args) for args in zip(*flat_structures)]

    # Unflatten back to original structure
    return _reference_unflatten_as(structures[0], mapped_values)


def convert_list_to_graph_structure(structure):
    if isinstance(structure, gym.spaces.GraphInstance):
        return structure
    elif isinstance(structure, tuple):
        if len(structure) > 0 and isinstance(structure[0], gym.spaces.GraphInstance):
            return structure
        else:
            return tuple(convert_list_to_graph_structure(sub) for sub in structure)
    elif isinstance(structure, list):
        if len(structure) == 3 and all(isinstance(sub, np.ndarray) or sub is None for sub in structure):
            return gym.spaces.GraphInstance(
                structure[0], structure[1], structure[2]
            )

        return [
            convert_list_to_graph_structure(sub) for sub in structure]
    elif isinstance(structure, Mapping):
        return type(structure)(
            (key, convert_list_to_graph_structure(structure[key])) for key in sorted(structure.keys())
        )
    else:
        return structure


def _reference_unflatten_as(structure, flat_sequence):
    """Unflattens a flat sequence into a given structure, preserving variable-length GraphInstance tuples."""

    def _unflatten(struct, flat_iter):
        """Recursively reconstructs the structure from flat_iter."""

        # Special handling for GraphInstance
        if isinstance(struct, gym.spaces.GraphInstance):
            return next(flat_iter)  # GraphInstances are returned directly

        # Handle tuple: Special case for variable-length tuples containing GraphInstance
        elif isinstance(struct, tuple):
            if len(struct) > 0 and isinstance(struct[0], gym.spaces.GraphInstance):
                return next(flat_iter)

            # If not GraphInstances, recursively process the tuple
            return tuple(_unflatten(sub, flat_iter) for sub in struct)

        # Handle list: Recursively process each element
        elif isinstance(struct, list):
            return [_unflatten(sub, flat_iter) for sub in struct]

        # Handle dict (Mapping): Ensure keys are sorted and process each key-value pair recursively
        elif isinstance(struct, Mapping):
            try:
                return type(struct)(
                    (key, _unflatten(struct[key], flat_iter)) for key in sorted(struct.keys())
                )
            except TypeError as e:
                raise TypeError(
                    "Cannot unflatten dict with non-sortable keys") from e

        # If it's a scalar or any other type, just return the next element
        else:
            return next(flat_iter)

    # Create an iterator from the flat_sequence
    flat_iter = iter(flat_sequence)

    # Start unflattening
    result = _unflatten(structure, flat_iter)

    # Ensure the flat sequence has no extra elements (check if fully consumed)
    try:
        next(flat_iter)
        raise ValueError(
            "Flat sequence contains more elements than required by structure")
    except StopIteration:
        pass  # No extra elements, all good

    return result


def _reference_flatten(structure):
    """Flattens a possibly nested structure into a list."""

    # Use lists for flattening
    if isinstance(structure, gym.spaces.GraphInstance):
        return [structure]

    elif isinstance(structure, tuple):
        # Preserve tuples of GraphInstances (they should not be flattened)
        if len(structure) > 0 and isinstance(structure[0], gym.spaces.GraphInstance):
            return [structure]

        return [item for sub in structure for item in _reference_flatten(sub)]

    elif isinstance(structure, list):
        return [item for sub in structure for item in _reference_flatten(sub)]

    elif isinstance(structure, Mapping):
        try:
            # Flatten dictionary values in sorted order of keys
            return [item for key in sorted(structure.keys()) for item in _reference_flatten(structure[key])]
        except TypeError as e:
            raise TypeError(
                "Cannot flatten dict with non-sortable keys") from e
    else:
        # Scalars are returned as a single element in a list
        return [structure]


def unbatch_graph_instance(structure):
    if isinstance(structure, gym.spaces.GraphInstance):
        return tuple([structure])
    elif isinstance(structure, tuple):
        if len(structure) > 0 and isinstance(structure[0], gym.spaces.GraphInstance):
            return structure
        return tuple(unbatch_graph_instance(sub) for sub in structure)
    elif isinstance(structure, list):
        return [unbatch_graph_instance(sub) for sub in structure]
    elif isinstance(structure, Mapping):
        return type(structure)(
            (key, unbatch_graph_instance(structure[key])) for key in sorted(structure.keys())
        )
    else:
        return structure


def unbatch_graph(structure):
    if isinstance(structure, gym.spaces.GraphInstance):
        return structure
    elif isinstance(structure, tuple):
        if len(structure) > 0 and isinstance(structure[0], gym.spaces.GraphInstance):
            if len(structure) == 1:
                return structure[0]
            return structure
        return tuple(unbatch_graph(sub) for sub in structure)
    elif isinstance(structure, list):
        return [unbatch_graph(sub) for sub in structure]
    elif isinstance(structure, Mapping):
        return type(structure)(
            (key, unbatch_graph_instance(structure[key])) for key in sorted(structure.keys())
        )
    else:
        return structure


def _reference_map_structure_with_path(func, *structures, **kwargs):
    """Maps `func` through given structures.

    This is a variant of :func:`~tree.map_structure` which accumulates
    a *path* while mapping through the structures. A path is a tuple of
    indices and/or keys which uniquely identifies the positions of the
    arguments passed to `func`.

    >>> tree.map_structure_with_path(
    ...     lambda path, v: (path, v**2),
    ...     [{"foo": 42}])
    [{'foo': ((0, 'foo'), 1764)}]

    Args:
      func: A callable that accepts a path and as many arguments as there are
        structures.
      *structures: Arbitrarily nested structures of the same layout.
      **kwargs: The only valid keyword argument is `check_types`. If `True`
        (default) the types of components within the structures have to be match,
        e.g. ``tree.map_structure_with_path(func, [1], (1,))`` will raise a
        `TypeError`, otherwise this is not enforced. Note that namedtuples with
        identical name and fields are considered to be the same type.

    Returns:
      A new structure with the same layout as the given ones. If the
      `structures` have components of varying types, the resulting structure
      will use the same types as ``structures[0]``.

    Raises:
      TypeError: If `func` is not callable or if the `structures` do not
        have the same layout.
      TypeError: If `check_types` is `True` and any two `structures`
        differ in the types of their components.
      ValueError: If no structures were given or if a keyword argument other
        than `check_types` is provided.
    """

    if not callable(func):
        raise TypeError(f"func must be callable, got: {type(func)}")

    if not structures:
        raise ValueError("Must provide at least one structure")

    check_types = kwargs.pop("check_types", True)
    if kwargs:
        raise ValueError(
            f"Only valid keyword argument is `check_types`, not: `{kwargs.keys()}`")

    structures = [unbatch_graph_instance(structure)
                  for structure in structures]

    for other in structures[1:]:
        assert_same_structure(structures[0], other, check_types=check_types)

    def recursive_map(path, *elems):
        if isinstance(elems[0], gym.spaces.GraphInstance):
            return func(path, *elems)  # GraphInstances are returned directly

        elif isinstance(elems[0], tuple):
            if len(elems[0]) > 0 and isinstance(elems[0][0], gym.spaces.GraphInstance):
                # Variable-length GraphInstance tuples are returned directly
                return func(path, *elems)

            return tuple(recursive_map(path + (i,), *childs) for i, childs in enumerate(zip(*elems)))

        elif isinstance(elems[0], list):
            return [recursive_map(path + (i,), *childs) for i, childs in enumerate(zip(*elems))]

        elif isinstance(elems[0], dict):
            return {k: recursive_map(path + (k,), *[e[k] for e in elems]) for k in elems[0]}

        else:
            return func(path, *elems)

    mapped_structure = recursive_map((), *structures)
    return _reference_unflatten_as(structures[0], _reference_flatten(mapped_structure))


def print_structure(structure, indent=0):
    """Recursively print the structure of nested data with indentation."""
    prefix = "  " * indent

    if isinstance(structure, gym.spaces.GraphInstance):
        nodes_shape = structure.nodes.shape if structure.nodes is not None else None
        edges_shape = structure.edges.shape if structure.edges is not None else None
        edge_links_shape = structure.edge_links.shape if structure.edge_links is not None else None
        print(f"{prefix}GraphInstance(nodes: {nodes_shape}, edges: {edges_shape}, edge_links: {edge_links_shape})")

    elif isinstance(structure, tuple):
        if len(structure) > 0 and isinstance(structure[0], gym.spaces.GraphInstance):
            print(f"{prefix}Tuple of {len(structure)} GraphInstance(s):")
            for i, item in enumerate(structure):
                print(f"{prefix}  [{i}]:")
                print_structure(item, indent + 2)
        else:
            print(f"{prefix}Tuple (length {len(structure)}):")
            for i, item in enumerate(structure):
                print(f"{prefix}  [{i}]:")
                print_structure(item, indent + 2)

    elif isinstance(structure, list):
        print(f"{prefix}List (length {len(structure)}):")
        for i, item in enumerate(structure):
            print(f"{prefix}  [{i}]:")
            print_structure(item, indent + 2)

    elif isinstance(structure, Mapping):
        print(f"{prefix}Dict (keys: {list(structure.keys())}):")
        for key in sorted(structure.keys()):
            print(f"{prefix}  '{key}':")
            print_structure(structure[key], indent + 2)

    elif isinstance(structure, (np.ndarray, torch.Tensor)):
        type_name = "ndarray" if isinstance(
            structure, np.ndarray) else "Tensor"
        print(
            f"{prefix}{type_name}(shape: {structure.shape}, dtype: {structure.dtype})")

    else:
        print(f"{prefix}{type(structure).__name__}: {structure}")


def assert_same_structure(a, b, check_types=True):
    """Asserts that two structures are nested in the same way."""

    if isinstance(a, gym.spaces.GraphInstance):
        if not isinstance(b, gym.spaces.GraphInstance):
            raise AssertionError(
                f"Structures have different types: {type(a)} vs {type(b)}")

    elif isinstance(a, tuple):
        if not isinstance(b, tuple):
            raise AssertionError(
                f"Structures have different types: {type(a)} vs {type(b)}")

        # If tuples contain GraphInstances, allow different lengths
        if len(a) > 0 and isinstance(a[0], gym.spaces.GraphInstance):
            if len(b) == 0 or not isinstance(b[0], gym.spaces.GraphInstance):
                raise AssertionError(
                    "Both tuples must start with GraphInstances")
        else:
            if len(a) != len(b):
                raise AssertionError(
                    f"Structures have different lengths: {len(a)} vs {len(b)}")
            # for sub_a, sub_b in zip(a, b):
            #    assert_same_structure(sub_a, sub_b, check_types=check_types)

    elif isinstance(a, list):
        if not isinstance(b, list):
            # PATCH: Allow list vs scalar mismatch for graph structures - skip check
            # This handles cases where graph observations have variable structures
            if not check_types:
                return
            raise AssertionError(
                f"Structures have different types: {type(a)} vs {type(b)}")
        if len(a) != len(b):
            raise AssertionError(
                f"Structures have different lengths: {len(a)} vs {len(b)}")
        for sub_a, sub_b in zip(a, b):
            assert_same_structure(sub_a, sub_b, check_types=check_types)

    elif isinstance(a, Mapping):
        if not isinstance(b, Mapping):
            raise AssertionError(
                f"Structures have different types: {type(a)} vs {type(b)}")

        if set(a.keys()) != set(b.keys()):
            raise AssertionError(
                f"Dictionaries have different keys: {set(a.keys())} vs {set(b.keys())}")

        for key in a:
            assert_same_structure(a[key], b[key], check_types=check_types)
