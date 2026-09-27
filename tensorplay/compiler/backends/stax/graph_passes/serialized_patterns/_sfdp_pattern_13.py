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
permute_default = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored())
bmm_default = CallFunction(operator_set.bmm.default, KeywordArg('query'), permute_default)
softmax_default = CallFunction(operator_set.softmax.default, bmm_default, Ignored(), Ignored())
_sfdp_pattern_13_inference = CallFunction(operator_set.bmm.default, softmax_default, KeywordArg('value'), _users=0)


permute_default = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored())
bmm_default = CallFunction(operator_set.bmm.default, KeywordArg('query'), permute_default)
softmax_default = CallFunction(operator_set.softmax.default, bmm_default, Ignored(), Ignored())
_sfdp_pattern_13_half_inference = CallFunction(operator_set.bmm.default, softmax_default, KeywordArg('value'), _users=0)
