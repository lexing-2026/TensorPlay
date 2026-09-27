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
div_Scalar = CallFunction(operator_set.div.Scalar, reshape_default_1, KeywordArg('inv_scale'))
add_Tensor = CallFunction(operator_set.add.Tensor, div_Scalar, KeywordArg('attn_mask'), alpha=Ignored())
softmax_default = CallFunction(operator_set.softmax.default, add_Tensor, Ignored(), Ignored(), _users=2)
rand_like_default = CallFunction(operator_set.rand_like.default, softmax_default, dtype=Ignored(), device=None)
gt_Scalar = CallFunction(operator_set.gt.Scalar, rand_like_default, KeywordArg('dropout_p'))
mul_Tensor = CallFunction(operator_set.mul.Tensor, gt_Scalar, softmax_default)
mul_Scalar = CallFunction(operator_set.mul.Scalar, mul_Tensor, Ignored(), _users=2)
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, mul_Scalar, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, KeywordArg('value'), Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
permute_default_0 = CallFunction(operator_set.permute.default, mul_Scalar, Ignored())
broadcast_to_default_3 = CallFunction(operator_set.broadcast_to.default, permute_default_0, Ignored())
reshape_default_5 = CallFunction(operator_set.reshape.default, broadcast_to_default_3, Ignored())
broadcast_to_default_4 = CallFunction(operator_set.broadcast_to.default, KeywordArg('tangents_1'), Ignored())
reshape_default_6 = CallFunction(operator_set.reshape.default, broadcast_to_default_4, Ignored())
bmm_default_1 = CallFunction(operator_set.bmm.default, reshape_default_5, reshape_default_6)
reshape_default_7 = CallFunction(operator_set.reshape.default, bmm_default_1, Ignored())
_sfdp_pattern_6_training = MultiOutputPattern([reshape_default_4,
  None,
  None,
  reshape_default_7
])


broadcast_to_default = CallFunction(operator_set.broadcast_to.default, KeywordArg('query'), Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored())
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
div_Scalar = CallFunction(operator_set.div.Scalar, reshape_default_1, KeywordArg('inv_scale'))
add_Tensor = CallFunction(operator_set.add.Tensor, div_Scalar, KeywordArg('attn_mask'), alpha=Ignored())
softmax_default = CallFunction(operator_set.softmax.default, add_Tensor, Ignored(), Ignored())
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, softmax_default, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, KeywordArg('value'), Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
_sfdp_pattern_6_inference = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored(), _users=0)


broadcast_to_default = CallFunction(operator_set.broadcast_to.default, KeywordArg('query'), Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored())
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
div_Scalar = CallFunction(operator_set.div.Scalar, reshape_default_1, KeywordArg('inv_scale'))
add_Tensor = CallFunction(operator_set.add.Tensor, div_Scalar, KeywordArg('attn_mask'), alpha=Ignored())
softmax_default = CallFunction(operator_set.softmax.default, add_Tensor, Ignored(), Ignored(), _users=2)
rand_like_default = CallFunction(operator_set.rand_like.default, softmax_default, dtype=Ignored(), device=None)
gt_Scalar = CallFunction(operator_set.gt.Scalar, rand_like_default, KeywordArg('dropout_p'))
mul_Tensor = CallFunction(operator_set.mul.Tensor, gt_Scalar, softmax_default)
mul_Scalar = CallFunction(operator_set.mul.Scalar, mul_Tensor, Ignored(), _users=2)
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, mul_Scalar, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, KeywordArg('value'), Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
permute_default_0 = CallFunction(operator_set.permute.default, mul_Scalar, Ignored())
broadcast_to_default_3 = CallFunction(operator_set.broadcast_to.default, permute_default_0, Ignored())
reshape_default_5 = CallFunction(operator_set.reshape.default, broadcast_to_default_3, Ignored())
broadcast_to_default_4 = CallFunction(operator_set.broadcast_to.default, KeywordArg('tangents_1'), Ignored())
reshape_default_6 = CallFunction(operator_set.reshape.default, broadcast_to_default_4, Ignored())
bmm_default_1 = CallFunction(operator_set.bmm.default, reshape_default_5, reshape_default_6)
reshape_default_7 = CallFunction(operator_set.reshape.default, bmm_default_1, Ignored())
_sfdp_pattern_6_half_training = MultiOutputPattern([reshape_default_4,
  None,
  None,
  reshape_default_7
])


broadcast_to_default = CallFunction(operator_set.broadcast_to.default, KeywordArg('query'), Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored())
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
div_Scalar = CallFunction(operator_set.div.Scalar, reshape_default_1, KeywordArg('inv_scale'))
add_Tensor = CallFunction(operator_set.add.Tensor, div_Scalar, KeywordArg('attn_mask'), alpha=Ignored())
softmax_default = CallFunction(operator_set.softmax.default, add_Tensor, Ignored(), Ignored())
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, softmax_default, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, KeywordArg('value'), Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
_sfdp_pattern_6_half_inference = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored(), _users=0)
