"""Whether two pieces may be joined, where the answer is a matter of policy.

Joining pieces is always correct and never necessary: it produces the same
values either way.  What it changes is how much is held while the work runs and
how many launches there are, and how much of that is worth trading away is a
decision about the program rather than a fact about it.  So the questions asked
here are not whether a join would be right but whether it is worth it, and the
answer is allowed to differ between two programs that do the same arithmetic.

Nothing here is needed for the answer to be correct.  A pair refused here is one
that would have worked, so what is lost is a possible saving; that is why the
reasons are recorded, since a surprising refusal is worth being able to read.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import typing
from typing import Any

from . import config
from ....graph.experimental.sympy_functions import is_infinite
from .ir import OrderedSet
from .loops import V
from .memory import estimate_peak_memory
from .runtime.hints import ReductionHint
from .runtime.runtime_utils import next_power_of_2
from .utils import bound_sympy

log = logging.getLogger(__name__)


def can_fuse(scheduler, node1, node2, shared_data_score: int) -> bool:
    """Whether joining these is worth doing.

    Three things are asked.  Do the two share a value at all -- if not, the join
    saves nothing, and joining purely to have one launch fewer would make the
    program bigger for no reason.  Would the join be so large that the code
    stops fitting in what a program may hold?  And would it mean holding more
    memory than before, which is a cost paid for a saving and may not be worth
    it?
    """

    if shared_data_score == 0 and (
        not config.aggressive_fusion or node1.is_reduction() or node2.is_reduction()
    ):
        # Nothing is shared, so there is nothing to save.  Where a value is
        # read by both but at different places, the reason is worth having
        # since it looks like a pair that should have joined.
        common_buf_names: OrderedSet = (
            node1.read_writes.buffer_names() & node2.read_writes.buffer_names()
        )
        if len(common_buf_names) > 0:
            _why(node1, node2)("no shared data due to indexing mismatch")
        else:
            _why(node1, node2)("no shared data")
        return False

    if (
        not node1.is_foreach()
        and not node2.is_foreach()
        and len(node1.get_nodes()) + len(node2.get_nodes()) > config.max_fusion_size
    ):
        _why(node1, node2)("exceeds max fusion")
        return False

    if scheduler.can_fusion_increase_peak_memory(node1, node2):
        _why(node1, node2)("Fusion will increase peak memory")
        return False

    if (
        config.max_fusion_unique_io_buffers is not None
        and scheduler.fusion_prevent_too_many_reads_and_writes(
            node1,
            node2,
            config.max_fusion_unique_io_buffers,
        )
    ):
        _why(node1, node2)("fusion_prevent_too_many_reads_and_writes")
        return False

    return True


def can_fuse_vertical(
    scheduler, node1, node2, shared_data_score: int
) -> bool:
    """Whether one piece may be written as part of another that feeds it.

    Whether the two fit is a property of the machine and is asked of the
    backend; what is left to ask here is whether it is worth it, and for this
    shape there is nothing further to say.
    """

    return True


def can_fuse_horizontal(
    scheduler, node1, node2, shared_data_score: int
) -> bool:
    """Whether two pieces walking the same axes may become one.

    Unlike the other shape, this one is worth a bound: two pieces that read and
    write a great deal relative to what they share would be joined into a kernel
    that holds far more open than either did alone, and a kernel that does not
    fit in what a program may hold is worse than two that each fit.
    """

    from .scheduler import MixOrderReduction

    if MixOrderReduction.can_fuse(node1, node2):
        # The two share every element of the value, so neither what they share
        # nor how far apart they are says anything.
        return True
    if shared_data_score < config.score_fusion_memory_threshold:
        _why(node1, node2)("score_fusion_memory_threshold")
        return False
    if scheduler.are_long_distant_nodes(node1, node2):
        _why(node1, node2)(
            "Nodes are too far away. Fusing them may increase peak memory."
        )
        return False

    return True


def _why(node1, node2):
    from .scheduler import WhyNoFuse

    return WhyNoFuse(node1, node2)




class Sortable(typing.Protocol):
    """Anything that can be used as a list.sort() key (int/tuple/etc)"""

    def __lt__(self, other: typing.Self) -> bool: ...


@dataclasses.dataclass
class FusionScore:
    """How much a join is worth, and of what kind.

    Five numbers rather than one, so that two joins of the same worth can be
    ordered by something other than worth -- which matters because the ties are
    common and the order they are broken in is what decides what runs first.
    """

    template_score: int
    node_type_score: int
    memory_score: int
    buffer_overlap_score: int
    proximity_score: int

    def __lt__(self, other):
        """Which of two joins runs first.

        Two joins of the same kind are ordered by what they save, except when one
        saves so much more than the other that the difference is not worth the
        kind: a join that saves sixteen times more is worth having even if it is
        the wrong kind, and a join that saves twice more is not.

        A reduction nested inside another is ranked below an ordinary join of
        mixed kinds, because the two cannot be ordered without one of them
        running its reduction before the other is ready for it.

        The buffer score is ranked below the memory score, so a join that saves
        reads and writes outright is preferred to one that saves a buffer being
        read twice -- the first is a saving and the second is an accounting.
        """

        threshold = 16
        if self.template_score != other.template_score:
            return self.template_score < other.template_score

        if (
            max(self.memory_score, other.memory_score)
            > min(self.memory_score, other.memory_score) * threshold
        ):
            return self.memory_score < other.memory_score

        return (
            self.node_type_score,
            self.memory_score,
            self.buffer_overlap_score,
            self.proximity_score,
        ) < (
            other.node_type_score,
            other.memory_score,
            other.buffer_overlap_score,
            other.proximity_score,
        )


class TpChoices:
    """The policy that decides what is worth doing to a program.

    Nothing here is needed for an answer to be correct: a pair refused here is
    one that would have worked, so what is lost is a possible saving.  What is
    here is a set of opinions about cost, and a program is free to replace the
    whole set by putting its own in place of this one.

    The tables themselves live with the code that uses them rather than here, so
    that adding a kind of tiled operation does not mean editing the policy and
    the table together.
    """

    def get_config_heuristics(self, device_type: str | None = "cuda"):
        """The tilings worth trying on a kind of device."""

        from .templates.triton import CHOICES

        return CHOICES.get_config_heuristics(device_type)

    def get_omni_decode_configs(
        self, head_dim: int, dtype: Any, device_type: str | None = "cuda"
    ) -> list:
        """The tilings for attention that asks one question at a time."""

        omni_heuristics = self.get_config_heuristics(device_type)
        return omni_heuristics.get_omni_decode_configs(head_dim, dtype)

    def get_omni_attention_fwd_configs(
        self,
        head_dim: int,
        seq_len: Any,
        dtype: Any,
        device_type: str | None = "cuda",
    ) -> list:
        """The tilings for the forward pass of attention."""

        omni_heuristics = self.get_config_heuristics(device_type)
        return omni_heuristics.get_omni_attn_fwd_configs(head_dim, seq_len, dtype)

    def get_omni_attention_bwd_configs(
        self, head_dim: int, dtype: Any, device_type: str | None = "cuda"
    ) -> list:
        """The tilings for the pass that goes back the way attention came."""

        omni_heuristics = self.get_config_heuristics(device_type)
        return omni_heuristics.get_omni_attn_bwd_configs(head_dim, dtype)

    def append_omni_attention_choices(
        self,
        choices: list,
        configs: list,
        input_nodes: list,
        subgraphs: list,
        layout: Any,
        kernel_options: dict,
        sparse_q_block_size: int,
        sparse_kv_block_size: int,
    ) -> list:
        """Add whatever candidates this policy has that the tables do not name.

        Nothing by default.  A policy that knows of a way to write this kernel
        that the tables here do not describe adds it, which is the only reason
        this exists: the tables are one policy's opinion and a program may
        replace it.
        """

        return choices

    can_fuse = staticmethod(can_fuse)
    can_fuse_vertical = staticmethod(can_fuse_vertical)
    can_fuse_horizontal = staticmethod(can_fuse_horizontal)

    def score_fusion(
        self,
        scheduler: Any,
        node1: Any,
        node2: Any,
    ) -> Sortable:
        """How much joining these two is worth, and of what kind.

        Five things, each answering a question the others cannot.  What kind of
        join it is, which decides whether it can be had at all.  How much memory
        traffic it saves, which is the saving itself.  How much of a buffer the
        two share, which is a saving of a different kind -- a buffer read twice
        rather than a read removed.  And how far apart the two are in the order
        they were written, which is what says whether holding the first's
        results until the second runs costs anything.
        """

        memory_score, buffer_overlap_score, is_mix_order_reduction = (
            scheduler.score_fusion_memory(
                node1, node2, return_is_mix_order_reduction=True
            )
        )
        proximity_score = -max(
            abs(node1.min_order - node2.max_order),
            abs(node2.min_order - node1.max_order),
        )

        # A join whose second half writes the whole answer goes last: it cannot
        # be started until everything it writes over is finished, so putting it
        # first means holding every one of those results until it runs.
        if node2.is_template():
            template_score = 0
        else:
            template_score = 1 + (
                (node1.is_template() == config.epilogue_fusion_first)
                and memory_score > 0
            )

        # Two of the same kind are worth more than two of different kinds, but
        # only if they save something -- otherwise the kind is the only thing
        # being compared and the two are equally arbitrary.
        node_type_score = int(
            node1.is_reduction() == node2.is_reduction() and memory_score > 0
        )
        if (
            config.triton.nested_reduction
            and node1.is_reduction()
            and node2.is_reduction()
            and node1.get_operation_names() & node2.ancestors
        ):
            # A reduction inside another cannot be run before the outer one has
            # produced what it reduces, so the two cannot be ordered without one
            # of them being held.  Only reductions of different sizes are like
            # this: two of the same size can be run side by side.
            _, (_, rnumel1) = node1.group
            _, (_, rnumel2) = node2.group
            if not V.graph.sizevars.statically_known_equals(rnumel1, rnumel2):
                node_type_score = -1

        return FusionScore(
            template_score,
            node_type_score,
            memory_score,
            buffer_overlap_score,
            proximity_score,
        )

    @staticmethod
    def should_use_cooperative_reduction(
        device: Any, numel: Any, reduction_numel: Any
    ) -> bool:
        """Whether the warps of a launch should cooperate on one reduction.

        Rather than each reducing a part of it and a later step combining the
        parts, which costs a round trip through memory for a reduction small
        enough that one launch can finish.

        Whether it is small enough depends on how many elements there are in
        total, because that is what decides how much each warp is given: a
        reduction over a few elements gives each warp one, and a reduction over
        many gives each a great many, and the same reduction length is cheap in
        one case and not in the other.
        """

        if config.triton.force_cooperative_reductions:
            return True
        if not config.triton.cooperative_reductions or device.type == "cpu":
            return False

        xhint = V.graph.sizevars.optimization_hint(numel, fallback=2)
        if xhint <= 8:
            threshold = 32768 * xhint
        elif xhint <= 16:
            threshold = 2097152
        else:
            return False
        return V.graph.sizevars.statically_known_geq(reduction_numel, threshold)

    @staticmethod
    def should_use_persistent_reduction(
        features: Any,
        cooperative_reduction: bool,
    ) -> bool:
        """Whether a reduction small enough to finish in one launch should be.

        A reduction that one launch can finish writes its answer once.  One that
        is stepped over in pieces writes a partial answer per step and combines
        them, which costs the round trip.  So the question is whether the
        reduction is small enough, and the answer depends on what else the launch
        is doing -- which is why the kernel's own shape is asked about rather
        than the reduction's length alone.

        A reduction whose length is not yet a number is refused rather than
        assumed: a launch that would have been persistent may be much larger
        than it looks, and a launch that would not may be small.
        """

        if not config.triton.persistent_reductions:
            return False
        reduction_hint = features.get_reduction_hint()
        rblock = features.strict_reduction_rblock()
        if rblock is not None and not features.has_strict_multirow_reduction():
            if not (
                V.graph.sizevars.statically_known_geq(rblock, features.reduction_numel)
            ):
                return False
        threshold = {
            ReductionHint.INNER: 1024,
        }.get(reduction_hint, 64)

        if reduction_hint not in (
            ReductionHint.INNER,
            ReductionHint.OUTER_TINY,
        ):
            bounds = bound_sympy(features.reduction_numel)
            lower = bounds.lower
            upper = bounds.upper

            if not all(
                (
                    (isinstance(bound, int) or bound.is_constant())
                    and not is_infinite(bound)
                )
                for bound in (lower, upper)
            ):
                return False

            lower = next_power_of_2(int(lower))
            upper = next_power_of_2(int(upper))

            # Widening the rows walked, rather than widening the block that
            # reduces: a persistent reduction finishes in one launch and so
            # cannot mask off part of its block, and if the length could be
            # anywhere between two very different values then for the smaller
            # ones most of what it writes would be masked away on every step.
            if lower != upper:
                return False

        if cooperative_reduction:
            # The warps of a cooperating launch each get fewer elements, so the
            # same reduction length is a larger share of what one warp holds.
            threshold *= 32 // min(
                V.graph.sizevars.optimization_hint(features.numel), 32
            )

        if config.triton.multi_kernel:
            threshold *= 16

        return V.graph.sizevars.statically_known_leq(
            features.reduction_numel, threshold
        )

    def uuid(self) -> tuple:
        """What identifies this policy, for a compiled answer to be keyed by.

        A compiled answer is only reusable by a program that would have made the
        same choices, so the key has to say which choices were in place.  This
        one is the default policy, and the key says so.
        """

        return ("tp_choices",)


def create_tp_choices(factory: Any) -> TpChoices:
    """The policy in place, which is the default one unless a program said."""

    return _create_tp_choices(factory, registered_tp_choices())


def _create_tp_choices(
    factory: Any,
    registrations: tuple,
) -> TpChoices:
    if not registrations:
        return TpChoices() if factory is None else factory()

    factories = list(registrations)
    if factory is not None:
        factories.insert(0, ("config", factory))

    choices = []
    for key, choice_factory in factories:
        choice = choice_factory()
        if not isinstance(choice, TpChoices):
            raise TypeError(
                "Tp choices factories must return TpChoices instances, "
                f"but {key!r} returned {type(choice)}"
            )
        choices.append(choice)

    if len(choices) == 1:
        return choices[0]
    return _ComposedTpChoices(choices)


class _ComposedTpChoices(TpChoices):
    """Use the first contributor that overrides each policy hook.

    Composed rather than merged, because two policies answering the same
    question differently is not a question with two answers -- it is a program
    that has to say which of them it meant.  So the order is the answer, and a
    question two of them answer differently is said out loud rather than
    resolved silently in favour of the earlier one.

    Which contributor answers is decided by whether the contributor's own class
    says something about the name, not by whether its object happens to have an
    attribute: a policy that inherits a hook has not overridden it, and treating
    an inherited hook as an override would let the first policy in the list
    silence every policy after it.
    """

    def __init__(self, choices: list) -> None:
        super().__init__()
        self._choices = tuple(choices)
        self._dispatchers: dict = {}

    def __getattribute__(self, name: str) -> Any:
        try:
            attributes = object.__getattribute__(self, "__dict__")
        except AttributeError:
            return object.__getattribute__(self, name)
        if name in attributes:
            return attributes[name]

        dispatchers = attributes.get("_dispatchers")
        if dispatchers is None:
            return object.__getattribute__(self, name)
        if name in dispatchers:
            dispatcher = dispatchers[name]
            if dispatcher is None:
                return object.__getattribute__(self, name)
            return dispatcher

        default = inspect.getattr_static(TpChoices, name, None)
        if name == "uuid" or not (
            callable(default) or isinstance(default, (classmethod, staticmethod))
        ):
            dispatchers[name] = None
            return object.__getattribute__(self, name)

        dispatcher = None
        owner = ""
        choices = object.__getattribute__(self, "_choices")
        for choice in choices:
            if inspect.getattr_static(choice, name) is default:
                continue
            if dispatcher is not None:
                log.warning(
                    "TpChoices hook %r is overridden by both %s and %s; "
                    "list order selects %s",
                    name,
                    owner,
                    type(choice).__name__,
                    owner,
                )
                continue
            dispatcher = getattr(choice, name)
            owner = type(choice).__name__

        dispatchers[name] = dispatcher
        if dispatcher is None:
            return object.__getattribute__(self, name)
        return dispatcher

    def uuid(self) -> tuple:
        return (
            "composed_tp_choices",
            tuple(
                _validate_choice_uuid(f"config:{index}", choice)()
                for index, choice in enumerate(self._choices)
            ),
        )


_registered_tp_choices: dict = {}


def register_tp_choices(key: str, choice_factory: Any) -> None:
    """Add a policy to the set, under a name to refer to it by."""

    _registered_tp_choices[key] = choice_factory


def registered_tp_choices() -> tuple:
    return tuple(_registered_tp_choices.items())


def _validate_choice_uuid(key: str, choice: TpChoices) -> Any:
    uuid = getattr(choice, "uuid", None)
    if not callable(uuid):
        raise RuntimeError(
            f"TpChoices contributor {key!r} does not implement uuid(). "
            "Implement uuid() for cache key participation."
        )
    return uuid
