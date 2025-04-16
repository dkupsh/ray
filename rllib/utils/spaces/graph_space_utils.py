import gymnasium as gym
from collections.abc import Mapping


def map_structure(func, *structures, **kwargs):
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
    flat_structures = list(map(flatten, structures))

    # Apply function element-wise
    mapped_values = [func(*args) for args in zip(*flat_structures)]

    # Unflatten back to original structure
    return unflatten_as(structures[0], mapped_values)


def unflatten_as(structure, flat_sequence):
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


def flatten(structure):
    """Flattens a possibly nested structure into a list."""

    # Use lists for flattening
    if isinstance(structure, gym.spaces.GraphInstance):
        return [structure]

    elif isinstance(structure, tuple):
        # Preserve tuples of GraphInstances (they should not be flattened)
        if len(structure) > 0 and isinstance(structure[0], gym.spaces.GraphInstance):
            return [structure]

        return [item for sub in structure for item in flatten(sub)]

    elif isinstance(structure, list):
        return [item for sub in structure for item in flatten(sub)]

    elif isinstance(structure, Mapping):
        try:
            # Flatten dictionary values in sorted order of keys
            return [item for key in sorted(structure.keys()) for item in flatten(structure[key])]
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


def map_structure_with_path(func, *structures, **kwargs):
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
    return unflatten_as(structures[0], flatten(mapped_structure))


def print_structure(structure, indent=0):
    if isinstance(structure, gym.spaces.GraphInstance):
        print(" " * indent + f"GraphInstance")
    elif isinstance(structure, tuple):
        if len(structure) > 0 and isinstance(structure[0], gym.spaces.GraphInstance):
            print(" " * indent + f"Tuple: GraphInstances")
        else:
            print(" " * indent + "Tuple: ")
            for sub in structure:
                print_structure(sub, indent + 2)
    elif isinstance(structure, list):
        print(" " * indent + "List:")
        for sub in structure:
            print_structure(sub, indent + 2)
    elif isinstance(structure, Mapping):
        print(" " * indent + "Dict:")
        for key in sorted(structure.keys()):
            print(" " * indent + f"Key: {key}")
            print_structure(structure[key], indent + 2)
    else:
        print(" " * indent + f"Scalar: {type(structure)}")


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
