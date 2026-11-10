import builtins

import tensorplay as tp
import tensorplay.nn.functional as F


def test_distributed_types_resolve_once(monkeypatch):
    # linear and embedding check for distributed tensors on every call; the
    # lookup must not repeat the import (a failed import is not cached and
    # costs a full module search each time).
    monkeypatch.setattr(F, "_distributed_types", F._DISTRIBUTED_UNRESOLVED)
    imports = []
    real_import = builtins.__import__

    def counting_import(name, *args, **kwargs):
        if name.startswith("tensorplay.distributed"):
            imports.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", counting_import)
    x, w = tp.ones(2, 3), tp.ones(4, 3)
    idx, table = tp.tensor([0, 1]), tp.ones(5, 3)
    for _ in range(3):
        assert tp.equal(F.linear(x, w), tp.full((2, 4), 3.0))
        assert tp.equal(F.embedding(idx, table), tp.ones(2, 3))
    assert len(imports) <= 1, imports
    assert F._distributed_types is not F._DISTRIBUTED_UNRESOLVED
