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
broadcast_to_default = CallFunction(operator_set.broadcast_to.default, KeywordArg('query'), Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored())
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
full_default = CallFunction(operator_set.full.default, Ignored(), KeywordArg('inv_scale'), dtype=Ignored(), device=Ignored(), pin_memory=False)
div_Tensor = CallFunction(operator_set.div.Tensor, reshape_default_1, full_default)
full_default_0 = CallFunction(operator_set.full.default, Ignored(), Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
where_self = CallFunction(operator_set.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_0)
add_Tensor = CallFunction(operator_set.add.Tensor, where_self, KeywordArg('attn_mask'), alpha=Ignored())
softmax_default = CallFunction(operator_set.softmax.default, add_Tensor, Ignored(), Ignored())
to_dtype = CallFunction(operator_set.to.dtype, softmax_default, Ignored(), False, False, None)
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, to_dtype, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, KeywordArg('value'), Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
_sfdp_pattern_19_inference = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored(), _users=0)


broadcast_to_default = CallFunction(operator_set.broadcast_to.default, KeywordArg('query'), Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored())
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
full_default = CallFunction(operator_set.full.default, Ignored(), KeywordArg('inv_scale'), dtype=Ignored(), device=Ignored(), pin_memory=False)
div_Tensor = CallFunction(operator_set.div.Tensor, reshape_default_1, full_default)
full_default_0 = CallFunction(operator_set.full.default, Ignored(), Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
where_self = CallFunction(operator_set.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_0)
add_Tensor = CallFunction(operator_set.add.Tensor, where_self, KeywordArg('attn_mask'), alpha=Ignored())
softmax_default = CallFunction(operator_set.softmax.default, add_Tensor, Ignored(), Ignored())
to_dtype = CallFunction(operator_set.to.dtype, softmax_default, Ignored(), False, False, None)
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, to_dtype, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, KeywordArg('value'), Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
_sfdp_pattern_19_half_inference = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored(), _users=0)
