#!/usr/bin/env python3
"""Holding a call that leaves an argument out to the answer it gives.

A schema may name an ordinary parameter "self" -- the value a search looks
up, the number an assertion checks, the list a walk reduces -- while the
surface it is reached through is a plain function.  That parameter is an
argument like any other, and the generated entry treats it as the receiver
of a method, so it never gets the check that a missing argument is
answered rather than dereferenced.

The check is in two places: the entry that takes the arguments refuses a
missing one by name, and the unpackers that read the argument refuse a
null one instead of asking the interpreter what a null object holds.

Every assertion here reads the answer the call gave, since that is the
whole of what the caller is entitled to.
"""
import pytest
import tensorplay as tp
from tensorplay._ops import NATIVE_NAMESPACE

ops = getattr(tp.ops, NATIVE_NAMESPACE)

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])

# Schemas whose ordinary parameter is named "self" while the surface they
# are reached through is a plain function.
SELF_NAMED = [
    ("searchsorted", "Scalar_out"),
    ("searchsorted", "Tensor_out"),
    ("remainder", "Scalar_Tensor"),
    ("_assert_scalar", "default"),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,overload", SELF_NAMED)
def test_a_left_out_argument_is_answered_not_dereferenced(device, name, overload):
    # The ordinary parameter is named "self" while the surface it is reached
    # through is a plain function, so it is an argument like any other.  Left
    # out, the call has to be answered -- the answer says which parameter was
    # left out -- rather than dereferenced.
    overload_object = getattr(getattr(ops, name), overload)
    with pytest.raises(TypeError) as raised:
        overload_object()
    message = str(raised.value)
    assert message, "the answer carried no text"
    assert "missing required argument" in message, message


@pytest.mark.parametrize("device", DEVICES)
def test_a_left_out_argument_of_an_out_variant_is_answered(device):
    # The same fault reached through the destination keyword alone: the
    # destination is named, the ordinary parameter is not.
    destination = tp.empty(3, dtype=tp.int64, device=device)
    with pytest.raises(TypeError) as raised:
        ops.searchsorted.Scalar_out(out=destination)
    message = str(raised.value)
    assert message, "the answer carried no text"
    assert "missing required argument" in message, message
