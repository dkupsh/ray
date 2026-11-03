"""graph_space_utils: the one-pass functions against the reference ones.

The public functions must return what the `_reference_*` implementations
return, raise what they raise, and call `func` with the same arguments in the
same order. That includes structures not shaped alike, which the fast path hands
to the reference.
"""

import collections
import enum
import random
import sys
import types
from collections.abc import Mapping

import gymnasium as gym
import numpy as np
import pytest
import torch

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.spaces import graph_space_utils as G
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

GI = gym.spaces.GraphInstance
NT = collections.namedtuple("NT", "a b")


def _graph(rng, batched=False, edges=True):
    n, e = rng.randint(1, 5), rng.randint(0, 6)
    lead = (rng.randint(1, 3),) if batched else ()
    return GI(
        nodes=np.zeros(lead + (n, 3), np.float32),
        edges=np.zeros(lead + (e, 2), np.float32) if edges else None,
        edge_links=np.zeros(lead + (e, 2), np.int64) if edges else None,
    )


def _leaf(rng):
    r = rng.random()
    if r < 0.3:
        return np.arange(rng.randint(0, 4))
    if r < 0.4:
        return None
    if r < 0.5:
        return rng.randint(0, 9)
    if r < 0.6:
        return torch.zeros(2)
    if r < 0.7:
        return "text"
    if r < 0.85:
        return _graph(rng, batched=rng.random() < 0.3, edges=rng.random() < 0.6)
    return tuple(_graph(rng, edges=rng.random() < 0.6) for _ in range(rng.randint(1, 4)))


def _is_graph(s):
    return isinstance(s, GI) or (isinstance(s, tuple) and s and isinstance(s[0], GI))


def _structure(rng, depth=0):
    if depth > 3 or rng.random() < 0.3:
        return _leaf(rng)
    r, width = rng.random(), rng.randint(0, 3)
    kids = [_structure(rng, depth + 1) for _ in range(width)]
    keys = rng.sample(["b", "a", "z", "m", "q"], width)
    if r < 0.2:
        return tuple(kids)
    if r < 0.35:
        return kids
    if r < 0.4:
        return NT(_structure(rng, depth + 1), _structure(rng, depth + 1))
    if r < 0.85:
        return dict(zip(keys, kids))
    if r < 0.95:
        return collections.OrderedDict(zip(keys, kids))
    # SampleBatch: its len() is the row count, not the key count.
    return SampleBatch({k: np.arange(3) for k in keys}) if width else {}


def _like(rng, s):
    """Shaped as `s` with fresh leaves. Graph batches change length, and a bare
    graph may become a batch of graphs."""
    if _is_graph(s):
        if rng.random() < 0.3:
            return _graph(rng)
        return tuple(_graph(rng) for _ in range(rng.randint(1, 4)))
    if isinstance(s, NT):
        return NT(_like(rng, s.a), _like(rng, s.b))
    if isinstance(s, (tuple, list)):
        return type(s)(_like(rng, x) for x in s)
    if isinstance(s, collections.abc.Mapping):
        keys = list(s.keys())
        rng.shuffle(keys)  # insertion order is not part of the shape
        return type(s)({k: _like(rng, s[k]) for k in keys}) if keys else type(s)()
    v = _leaf(rng)
    while _is_graph(v):
        v = _leaf(rng)
    return v


def _mutate(rng, s):
    """`s` with one change that may break the shape."""
    if isinstance(s, collections.abc.Mapping) and s and rng.random() < 0.6:
        out, k, r = dict(s), rng.choice(list(s.keys())), rng.random()
        if r < 0.3:
            del out[k]
        elif r < 0.5:
            out["new"] = _leaf(rng)
        else:
            out[k] = _mutate(rng, s[k])
        try:
            return type(s)(out)
        except Exception:
            return out
    if (isinstance(s, (list, tuple)) and s and not _is_graph(s)
            and not isinstance(s, NT) and rng.random() < 0.6):
        items, i = list(s), rng.randrange(len(s))
        if rng.random() < 0.3:
            items.pop(i)
        else:
            items[i] = _mutate(rng, items[i])
        return type(s)(items)
    r = rng.random()
    if r < 0.25:
        return _leaf(rng)
    if r < 0.5:
        return [_leaf(rng)]
    if r < 0.75:
        return (_leaf(rng), _leaf(rng))
    return {"a": _leaf(rng)}


def _fp(x, by_content=False):
    """Container types, key order, and leaf identity. A view or a tensor is a
    new object each time it is taken from its parent, so those compare by
    content; so does every array when `by_content`, for structures whose
    rebuild makes new arrays on each run."""
    if isinstance(x, GI):
        return ("GI",) + tuple(_fp(i, by_content) for i in x)
    if isinstance(x, tuple):
        return ("T", type(x).__name__) + tuple(_fp(i, by_content) for i in x)
    if isinstance(x, list):
        return ("L", [_fp(i, by_content) for i in x])
    if isinstance(x, collections.abc.Mapping):
        return ("M", type(x).__name__,
                [((type(k).__name__, repr(k)), _fp(v, by_content)) for k, v in x.items()])
    if isinstance(x, torch.Tensor):
        return ("tensor", tuple(x.shape), x.tolist())
    if isinstance(x, np.ndarray):
        if by_content or x.base is not None:
            return ("array", x.dtype.str, x.shape, x.tolist())
        return ("id", id(x))
    return ("v", repr(x))


def _outcome(fn, by_content=False):
    calls = []
    try:
        return ("ok", _fp(fn(calls), by_content), calls)
    except Exception as e:
        return ("err", type(e).__name__, str(e), calls)


def _recorder(calls, with_path=False, containers=False, by_content=False):
    count = [0]

    def func(*args):
        if with_path:
            calls.append(("path", args[0]))
            args = args[1:]
        calls.append(tuple(_fp(a, by_content) for a in args))
        count[0] += 1
        result = ("r", count[0])
        # Every third result a container: the reference re-flattens those.
        if containers and count[0] % 3 == 0:
            return [result] if count[0] % 2 else (result, result)
        return result

    return func


def _assert_same(fast, ref, by_content=False):
    assert _outcome(fast, by_content) == _outcome(ref, by_content)


def _check_maps(structs, check_types, containers=False, by_content=False):
    def run(fn, with_path):
        return lambda calls: fn(
            _recorder(calls, with_path, containers and with_path, by_content),
            *structs, check_types=check_types)

    _assert_same(run(G.map_structure, False), run(G._reference_map_structure, False),
                 by_content)
    _assert_same(run(G.map_structure_with_path, True),
                 run(G._reference_map_structure_with_path, True), by_content)


@pytest.mark.parametrize("seed", range(8))
def test_random_structures_match_reference(seed):
    rng = random.Random(seed)
    for _ in range(250):
        s0 = _structure(rng)
        if rng.random() < 0.45:
            others = [_like(rng, s0) for _ in range(rng.randint(0, 3))]
        else:
            others = [_mutate(rng, s0) if rng.random() < 0.7 else _like(rng, s0)
                      for _ in range(rng.randint(1, 2))]
        _check_maps([s0] + others, check_types=rng.random() < 0.5,
                    containers=rng.random() < 0.15)

        _assert_same(lambda c: G.flatten(s0), lambda c: G._reference_flatten(s0))
        try:
            flat = G._reference_flatten(s0)
        except TypeError:
            flat = []
        for variant in (flat, flat[:-1], flat + ["extra"]):
            _assert_same(lambda c: G.unflatten_as(s0, list(variant)),
                         lambda c: G._reference_unflatten_as(s0, list(variant)))


_g1 = GI(np.zeros((2, 3)), np.zeros((1, 2)), np.zeros((1, 2), np.int64))
_g2 = GI(np.zeros((4, 3)), None, None)


@pytest.mark.parametrize("structs", [
    pytest.param([_g1, _g2], id="bare graphs"),
    pytest.param([(_g1, _g2), (_g2,)], id="graph batches of differing length"),
    pytest.param([_g1, (_g1, _g2, _g1)], id="bare graph against a batch"),
    pytest.param([{"x": _g1, "y": np.ones(2)}, {"y": np.ones(2), "x": (_g2,)}],
                 id="graphs in dicts, keys in other orders"),
    pytest.param([[_g1, (_g2,)], [(_g1, _g1), _g2]], id="lists of graphs"),
    pytest.param([GI(np.ones((2, 3, 4)), None, None)], id="batched graph, no edges"),
    pytest.param([{3: 1, "a": 2}], id="unsortable keys"),
    pytest.param([{"a": 1}, {"b": 1}], id="different keys"),
    pytest.param([SampleBatch({"a": np.ones(3)}),
                  SampleBatch({"a": np.ones(3), "b": np.ones(3)})],
                 id="SampleBatch with an extra key"),
    pytest.param([(1, 2), [1, 2]], id="tuple against list"),
    pytest.param([[1, 2], 5], id="list against leaf"),
    pytest.param([5, [1, 2]], id="leaf against list"),
    pytest.param([_g1, np.ones(2)], id="graph against leaf"),
    pytest.param([np.ones(2), _g1], id="leaf against graph"),
    pytest.param([(), ()], id="empty tuples"),
    pytest.param([{}, {}], id="empty dicts"),
])
@pytest.mark.parametrize("check_types", [True, False])
def test_edge_cases_match_reference(structs, check_types):
    _check_maps(structs, check_types)
    _assert_same(lambda c: G.flatten(structs[0]),
                 lambda c: G._reference_flatten(structs[0]))


def _sample_batch_with_list_column():
    batch = SampleBatch({"a": np.ones(2)})
    batch["a"] = [1, 2]  # the reference's rebuild turns this back into an array
    return batch


@pytest.mark.parametrize("make", [
    pytest.param(_sample_batch_with_list_column, id="SampleBatch list column"),
    pytest.param(lambda: SampleBatch({"a": np.ones(2), "is_training": True}),
                 id="SampleBatch is_training, popped on rebuild"),
    pytest.param(lambda: collections.defaultdict(int, a=1), id="defaultdict"),
    pytest.param(lambda: types.MappingProxyType({"a": 1}),
                 id="mappingproxy"),
    pytest.param(lambda: {"x": [collections.defaultdict(int, a=1)]},
                 id="defaultdict nested in a dict"),
])
def test_mappings_the_reference_rebuilds(make):
    """The reference rebuilds a Mapping before calling func: what func sees,
    and any constructor error, come from that rebuild."""
    _check_maps([make()], check_types=True, by_content=True)
    _check_maps([make(), make()], check_types=True, by_content=True)


def _logging_mappings(log):
    """dict subclasses whose constructors and reads are observable: each
    records what it does in `log`."""

    class Counting(dict):
        def __init__(self, pairs=()):
            log.append("Counting()")
            super().__init__(pairs)

    class Reversing(dict):
        def __init__(self, pairs=()):
            log.append("Reversing()")
            super().__init__(reversed(list(pairs)))

    class Rejecting(dict):
        def __init__(self, pairs=()):
            pairs = list(pairs)
            log.append("Rejecting()")
            if any(value is None for _, value in pairs):
                raise TypeError("None forbidden")
            super().__init__(pairs)

    class Changing(dict):
        reads = 0

        def __getitem__(self, key):
            type(self).reads += 1
            log.append(f"read {type(self).reads}")
            return type(self).reads

    return Counting, Reversing, Rejecting, Changing


@pytest.mark.parametrize("case", [
    "counting, mismatched", "counting, unflatten too long", "reversing order",
    "rejecting results", "changing reads",
])
def test_observable_mappings_match_reference(case):
    """Mappings whose constructors or reads have effects see the same calls,
    in the same order, as with the reference."""

    def run(fast):
        ms = G.map_structure if fast else G._reference_map_structure
        msp = G.map_structure_with_path if fast else G._reference_map_structure_with_path
        unflatten = G.unflatten_as if fast else G._reference_unflatten_as

        def go(calls):
            Counting, Reversing, Rejecting, Changing = _logging_mappings(calls)
            record = _recorder(calls)
            if case == "counting, mismatched":
                return (ms(record, {"x": Counting([("a", [1])])}, {"x": {"a": [1, 2]}}),)
            if case == "counting, unflatten too long":
                return unflatten(Counting([("a", 1)]), [2, 3])
            if case == "reversing order":
                return msp(lambda p, v: (calls.append(p), v)[1],
                           Reversing([("a", 1), ("b", 2)]))
            if case == "rejecting results":
                return ms(lambda v: None, Rejecting([("a", 1)]))
            changing = Changing(a=0)
            return msp(lambda p, v: changing, 0)

        return go

    _assert_same(run(True), run(False))


@pytest.mark.parametrize("case", [
    "tuple subclass indexing", "tuple subclass read twice", "flat list subclass",
    "callback mutates input", "logging key hash", "StrEnum keys",
])
def test_observable_containers_and_keys_match_reference(case):
    """Containers and keys with observable reads, iteration or hashing see the
    same operations as with the reference; ordinary ones keep the fast path."""

    def run(fast):
        ms = G.map_structure if fast else G._reference_map_structure
        msp = G.map_structure_with_path if fast else G._reference_map_structure_with_path
        unflatten = G.unflatten_as if fast else G._reference_unflatten_as

        def go(calls):
            if case == "tuple subclass indexing":
                class T(tuple):
                    def __getitem__(self, i):
                        return 99 if i == 1 else super().__getitem__(i)
                return ms(_recorder(calls), T((1, 2)))
            if case == "tuple subclass read twice":
                class T(tuple):
                    def __getitem__(self, i):
                        calls.append(("getitem", i))
                        if len(calls) > 1:
                            raise ValueError("read twice")
                        return super().__getitem__(i)
                return unflatten([T((0,)), {}], [7, 8])
            if case == "flat list subclass":
                class L(list):
                    def __iter__(self):
                        calls.append("iter")
                        return iter([7, 8] if calls.count("iter") == 1 else [9])
                return unflatten(0, L([7, 8]))
            if case == "callback mutates input":
                x = [1, 2]

                def callback(path, value):
                    calls.append((path, value))
                    x.clear()
                    return [value + 10]
                return msp(callback, x)
            if case == "logging key hash":
                class Key:
                    def __hash__(self):
                        calls.append("hash")
                        return 1

                    def __lt__(self, other):
                        return False

                    def __repr__(self):
                        return "Key"
                key = Key()
                d = {key: 0}
                calls.clear()
                return ms(lambda v: len(calls), d)
            members = enum.StrEnum("Names", ["b", "a"])
            return ms(_recorder(calls), {members.b: 1, members.a: (_g1, _g2)})

        return go

    _assert_same(run(True), run(False))


def test_graph_results_regroup_like_the_reference():
    """A graph returned for a tuple's first item makes the reference read that
    tuple as one graph batch when it flattens the results."""
    g = GI(np.ones((1, 1)), None, None)

    def run(fn):
        return lambda calls: fn(lambda p, x: (calls.append(p), g)[1], (1, 2))

    _assert_same(run(G.map_structure_with_path),
                 run(G._reference_map_structure_with_path))


def test_unflatten_as_reads_an_iterator_as_far_as_the_reference():
    def values(read):
        for i in range(5):
            read.append(i)
            yield i

    _assert_same(lambda read: G.unflatten_as(0, values(read)),
                 lambda read: G._reference_unflatten_as(0, values(read)))


def test_types_registered_as_mappings_later():
    class Late:
        def keys(self):
            return ["a"]

        def __getitem__(self, key):
            return 7

    x = Late()
    assert G.flatten(x) == [x]
    Mapping.register(Late)
    assert G.flatten(x) == G._reference_flatten(x) == [7]


def test_graph_conventions():
    seen = []
    G.map_structure(lambda g: seen.append(g), {"g": _g1, "batch": (_g1, _g2)})
    # Sorted key order; a bare graph arrives as a batch of one.
    assert seen[0] == (_g1, _g2) and seen[1] == (_g1,)
    assert G.flatten({"g": _g1, "batch": (_g1, _g2)}) == [(_g1, _g2), _g1]
    paths = []
    G.map_structure_with_path(lambda p, v: paths.append(p), {"o": [_g1, {"x": 1}]})
    assert paths == [("o", 0), ("o", 1, "x")]


def test_convert_to_torch_tensor_graphs():
    out = convert_to_torch_tensor({"g": (_g1, _g2), "x": np.ones(2, np.float64)})
    graphs = out["g"]
    assert isinstance(graphs, tuple) and all(isinstance(g, GI) for g in graphs)
    assert graphs[0].nodes.dtype == torch.float32  # float64 becomes float32
    assert graphs[0].edge_links.dtype == torch.int64
    assert graphs[1].edges is None and graphs[1].edge_links is None
    assert out["x"].dtype == torch.float32
    # A bare graph is converted as a batch of one, as before.
    single = convert_to_torch_tensor(_g1)
    assert isinstance(single, tuple) and isinstance(single[0], GI)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
