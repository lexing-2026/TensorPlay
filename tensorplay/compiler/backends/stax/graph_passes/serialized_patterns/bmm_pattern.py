# noqa: F401, E501
# This is an auto-generated file. Please do not modify it by hand.
# To re-generate, run the generator named in the file it writes.

import operator
import tensorplay as tp

operator_set = tp.ops.tp
prims = tp.ops.prims

from tensorplay.compiler.backends.stax.pattern_matcher import (
   Arg,
   CallFunction,
   CallFunctionVarArgs,
   CallMethod,
   CallMethodVarArgs,
   CallModule,
   CallModuleVarArgs,
   ExclusiveKeywordArg,
   GetAttr,
   Ignored,
   KeywordArg,
   ListOf,
   MultiOutputPattern,
   PatternExpr,
   RepeatedExpr,
   _TargetArgsExpr,
   _TargetExpr,
   _TargetExprVarArgs,
)
bmm_default = CallFunction(operator_set.bmm.default, KeywordArg('mat1'), KeywordArg('mat2'))
broadcast_to_default = CallFunction(operator_set.broadcast_to.default, KeywordArg('tangents_1'), Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default = CallFunction(operator_set.permute.default, KeywordArg('mat2'), Ignored())
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
permute_default_0 = CallFunction(operator_set.permute.default, KeywordArg('mat1'), Ignored())
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, permute_default_0, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, KeywordArg('tangents_1'), Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_1 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_1, Ignored())
bmm_pattern_training = MultiOutputPattern([bmm_default,
  reshape_default_1,
  reshape_default_4
])


bmm_pattern_inference = CallFunction(operator_set.bmm.default, KeywordArg('mat1'), KeywordArg('mat2'))
