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
addmm_default = CallFunction(operator_set.addmm.default, KeywordArg('input'), KeywordArg('mat1'), KeywordArg('mat2'), beta=KeywordArg('beta'), alpha=KeywordArg('alpha'))
mul_Scalar = CallFunction(operator_set.mul.Scalar, KeywordArg('tangents_1'), KeywordArg('beta'))
sum_dim_IntList = CallFunction(operator_set.sum.dim_IntList, mul_Scalar, Ignored(), True, dtype=Ignored())
reshape_default = CallFunction(operator_set.reshape.default, sum_dim_IntList, Ignored())
permute_default = CallFunction(operator_set.permute.default, KeywordArg('mat2'), Ignored())
mm_default = CallFunction(operator_set.mm.default, KeywordArg('tangents_1'), permute_default)
mul_Scalar_0 = CallFunction(operator_set.mul.Scalar, mm_default, KeywordArg('alpha'))
permute_default_0 = CallFunction(operator_set.permute.default, KeywordArg('mat1'), Ignored())
mm_default_0 = CallFunction(operator_set.mm.default, permute_default_0, KeywordArg('tangents_1'))
mul_Scalar_1 = CallFunction(operator_set.mul.Scalar, mm_default_0, KeywordArg('alpha'))
addmm_pattern_training = MultiOutputPattern([addmm_default,
  reshape_default,
  mul_Scalar_0,
  mul_Scalar_1
])


addmm_pattern_inference = CallFunction(operator_set.addmm.default, KeywordArg('input'), KeywordArg('mat1'), KeywordArg('mat2'), beta=KeywordArg('beta'), alpha=KeywordArg('alpha'))
