"""An extent with no upper bound, and what arithmetic does with it.

The two infinities are objects of this package's own.  The symbolic library
keeps one registry of such objects for the whole process, keyed by class
name, so a second package that defines a class called the same thing takes the
entry over.  None of the arithmetic here may then start answering with, or
looking for, that other object.
"""
import pytest
import sympy
from sympy.core.singleton import Singleton

from tensorplay.graph.experimental import sympy_functions as sf

_NAMES = ("IntInfinity", "NegativeIntInfinity")


def test_integer_bounds_remain_integer_under_powers():
    extent = sympy.Symbol("extent", integer=True, positive=True)
    bounds = sf.bound_sympy(extent**2)
    assert bounds.is_int
    assert bounds.lower == 1
    assert bounds.upper is sf.int_oo


@pytest.fixture
def taken_names():
    """The two registry entries, taken over for the length of one test.

    This is what importing a package with classes of these names does.  The
    entries are put back as they were found afterwards: whoever owned them
    before is still in the process and still reads them.
    """

    registry = type(sympy.S)
    pending = sympy.S._classes_to_install
    installed_before = {n: vars(registry)[n] for n in _NAMES if n in vars(registry)}
    pending_before = {n: pending[n] for n in _NAMES if n in pending}

    class IntInfinity(sympy.Basic, metaclass=Singleton):
        pass

    class NegativeIntInfinity(sympy.Basic, metaclass=Singleton):
        pass

    try:
        yield IntInfinity(), NegativeIntInfinity()
    finally:
        for name in _NAMES:
            if name in vars(registry):
                delattr(registry, name)
            pending.pop(name, None)
        for name, value in installed_before.items():
            setattr(registry, name, value)
        pending.update(pending_before)


def test_arithmetic_keeps_its_own_infinity_when_the_name_is_taken(taken_names):
    theirs, their_negative = taken_names
    assert sympy.S.IntInfinity is theirs
    assert sympy.S.NegativeIntInfinity is their_negative
    assert sf.int_oo is not theirs

    two = sympy.Integer(2)
    assert sf.FloorDiv(sf.int_oo, two) is sf.int_oo
    assert sf.FloorDiv(-sf.int_oo, two) is -sf.int_oo
    assert -(-sf.int_oo) is sf.int_oo
    assert sf.int_oo + 1 is sf.int_oo
    assert (sf.int_oo > 5) is sympy.true
    assert (-sf.int_oo < 5) is sympy.true
    assert str(sf.int_oo) == "int_oo"


def test_a_bound_through_a_division_stays_a_range(taken_names):
    x = sympy.Symbol("x", integer=True)
    bound = sf.bound_sympy(x // 2, {x: sf.ValueRanges(0, sf.int_oo)})
    assert bound.lower == 0
    assert bound.upper is sf.int_oo
    halved = sf.bound_sympy(x // 2, {x: sf.ValueRanges(0, 9)})
    assert (halved.lower, halved.upper) == (0, 4)


def test_the_entries_are_put_back(taken_names):
    # Nothing to do with the names while they are taken; the next test reads
    # what is left behind.
    assert sympy.S.IntInfinity is taken_names[0]


def test_the_entries_were_put_back():
    for name in _NAMES:
        entry = getattr(sympy.S, name, None)
        assert entry is None or "taken_names" not in type(entry).__qualname__


def test_this_package_leaves_the_same_named_entries_alone():
    registered = {type(sf.int_oo).__name__, type(-sf.int_oo).__name__}
    assert registered.isdisjoint({"IntInfinity", "NegativeIntInfinity"})
