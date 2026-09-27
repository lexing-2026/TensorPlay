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
view_default = CallFunction(operator_set.view.default, KeywordArg('query'), Ignored())
reshape_default = CallFunction(operator_set.reshape.default, KeywordArg('key'), Ignored())
permute_default = CallFunction(operator_set.permute.default, reshape_default, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, view_default, permute_default)
view_default_0 = CallFunction(operator_set.view.default, bmm_default, Ignored())
add_Tensor = CallFunction(operator_set.add.Tensor, view_default_0, KeywordArg('attention_mask'), alpha=Ignored())
view_default_1 = CallFunction(operator_set.view.default, add_Tensor, Ignored())
softmax_default = CallFunction(operator_set.softmax.default, view_default_1, Ignored(), Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, KeywordArg('value'), Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, softmax_default, reshape_default_0)
_sfdp_pattern_24_inference = CallFunction(operator_set.view.default, bmm_default_0, Ignored())


view_default = CallFunction(operator_set.view.default, KeywordArg('query'), Ignored())
reshape_default = CallFunction(operator_set.reshape.default, KeywordArg('key'), Ignored())
permute_default = CallFunction(operator_set.permute.default, reshape_default, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, view_default, permute_default)
view_default_0 = CallFunction(operator_set.view.default, bmm_default, Ignored())
add_Tensor = CallFunction(operator_set.add.Tensor, view_default_0, KeywordArg('attention_mask'), alpha=Ignored())
view_default_1 = CallFunction(operator_set.view.default, add_Tensor, Ignored())
softmax_default = CallFunction(operator_set.softmax.default, view_default_1, Ignored(), Ignored())
to_dtype = CallFunction(operator_set.to.dtype, softmax_default, Ignored(), False, False, None)
reshape_default_0 = CallFunction(operator_set.reshape.default, KeywordArg('value'), Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, to_dtype, reshape_default_0)
_sfdp_pattern_24_half_inference = CallFunction(operator_set.view.default, bmm_default_0, Ignored())
