"""Higher-order operators.

A higher-order operator takes callables (graphs) among its inputs.  The
operators in this package expose a dispatch-key registry whose default
registration is the composite eager implementation; under graph capture the
call is recorded as one opaque node.
"""
