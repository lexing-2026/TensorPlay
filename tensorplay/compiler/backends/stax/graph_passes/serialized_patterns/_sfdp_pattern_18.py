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
permute_default = CallFunction(operator_set.permute.default, KeywordArg('query'), Ignored())
broadcast_to_default = CallFunction(operator_set.broadcast_to.default, permute_default, Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default_0 = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored(), _users=2)
permute_default_1 = CallFunction(operator_set.permute.default, permute_default_0, Ignored())
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default_1, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
full_default = CallFunction(operator_set.full.default, Ignored(), KeywordArg('inv_scale'), dtype=Ignored(), device=Ignored(), pin_memory=False)
div_Tensor = CallFunction(operator_set.div.Tensor, reshape_default_1, full_default)
full_default_0 = CallFunction(operator_set.full.default, Ignored(), Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
where_self = CallFunction(operator_set.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_0)
softmax_default = CallFunction(operator_set.softmax.default, where_self, Ignored(), Ignored())
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, softmax_default, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored(), _users=2)
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, permute_default_2, Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
_sfdp_pattern_18_inference = MultiOutputPattern([reshape_default_4,
  permute_default_0,
  permute_default_2
])


permute_default = CallFunction(operator_set.permute.default, KeywordArg('query'), Ignored())
broadcast_to_default = CallFunction(operator_set.broadcast_to.default, permute_default, Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default_0 = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored(), _users=2)
permute_default_1 = CallFunction(operator_set.permute.default, permute_default_0, Ignored())
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default_1, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
full_default = CallFunction(operator_set.full.default, Ignored(), KeywordArg('inv_scale'), dtype=Ignored(), device=Ignored(), pin_memory=False)
div_Tensor = CallFunction(operator_set.div.Tensor, reshape_default_1, full_default)
full_default_0 = CallFunction(operator_set.full.default, Ignored(), Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
where_self = CallFunction(operator_set.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_0)
softmax_default = CallFunction(operator_set.softmax.default, where_self, Ignored(), Ignored())
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, softmax_default, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored(), _users=2)
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, permute_default_2, Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
_sfdp_pattern_18_bs1_inference = MultiOutputPattern([reshape_default_4,
  permute_default_0,
  permute_default_2
])


permute_default = CallFunction(operator_set.permute.default, KeywordArg('query'), Ignored())
broadcast_to_default = CallFunction(operator_set.broadcast_to.default, permute_default, Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default_0 = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored(), _users=2)
permute_default_1 = CallFunction(operator_set.permute.default, permute_default_0, Ignored())
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default_1, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
full_default = CallFunction(operator_set.full.default, Ignored(), KeywordArg('inv_scale'), dtype=Ignored(), device=Ignored(), pin_memory=False)
div_Tensor = CallFunction(operator_set.div.Tensor, reshape_default_1, full_default)
full_default_0 = CallFunction(operator_set.full.default, Ignored(), Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
where_self = CallFunction(operator_set.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_0)
softmax_default = CallFunction(operator_set.softmax.default, where_self, Ignored(), Ignored())
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, softmax_default, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored(), _users=2)
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, permute_default_2, Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
_sfdp_pattern_18_half_inference = MultiOutputPattern([reshape_default_4,
  permute_default_0,
  permute_default_2
])


permute_default = CallFunction(operator_set.permute.default, KeywordArg('query'), Ignored())
broadcast_to_default = CallFunction(operator_set.broadcast_to.default, permute_default, Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default_0 = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored(), _users=2)
permute_default_1 = CallFunction(operator_set.permute.default, permute_default_0, Ignored())
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default_1, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
full_default = CallFunction(operator_set.full.default, Ignored(), KeywordArg('inv_scale'), dtype=Ignored(), device=Ignored(), pin_memory=False)
div_Tensor = CallFunction(operator_set.div.Tensor, reshape_default_1, full_default)
full_default_0 = CallFunction(operator_set.full.default, Ignored(), Ignored(), dtype=Ignored(), device=Ignored(), pin_memory=False)
where_self = CallFunction(operator_set.where.self, KeywordArg('causal_mask'), div_Tensor, full_default_0)
softmax_default = CallFunction(operator_set.softmax.default, where_self, Ignored(), Ignored())
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, softmax_default, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored(), _users=2)
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, permute_default_2, Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
_sfdp_pattern_18_half_bs1_inference = MultiOutputPattern([reshape_default_4,
  permute_default_0,
  permute_default_2
])
