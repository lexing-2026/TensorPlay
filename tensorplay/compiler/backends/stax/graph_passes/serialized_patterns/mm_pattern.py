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
mm_default = CallFunction(operator_set.mm.default, KeywordArg('mat1'), KeywordArg('mat2'))
permute_default = CallFunction(operator_set.permute.default, KeywordArg('mat2'), Ignored())
mm_default_0 = CallFunction(operator_set.mm.default, KeywordArg('tangents_1'), permute_default)
permute_default_0 = CallFunction(operator_set.permute.default, KeywordArg('mat1'), Ignored())
mm_default_1 = CallFunction(operator_set.mm.default, permute_default_0, KeywordArg('tangents_1'))
mm_pattern_training = MultiOutputPattern([mm_default,
  mm_default_0,
  mm_default_1
])


mm_pattern_inference = CallFunction(operator_set.mm.default, KeywordArg('mat1'), KeywordArg('mat2'))
