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
permute_default = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored(), _users=2)
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
mul_Tensor = CallFunction(operator_set.mul.Tensor, reshape_default_1, KeywordArg('scale_factor'))
softmax_default = CallFunction(operator_set.softmax.default, mul_Tensor, Ignored(), Ignored(), _users=4)
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, softmax_default, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, KeywordArg('value'), Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
broadcast_to_default_3 = CallFunction(operator_set.broadcast_to.default, KeywordArg('tangents_1'), Ignored())
reshape_default_5 = CallFunction(operator_set.reshape.default, broadcast_to_default_3, Ignored())
permute_default_0 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored())
broadcast_to_default_4 = CallFunction(operator_set.broadcast_to.default, permute_default_0, Ignored())
reshape_default_6 = CallFunction(operator_set.reshape.default, broadcast_to_default_4, Ignored())
bmm_default_1 = CallFunction(operator_set.bmm.default, reshape_default_5, reshape_default_6)
reshape_default_7 = CallFunction(operator_set.reshape.default, bmm_default_1, Ignored())
mul_Tensor_0 = CallFunction(operator_set.mul.Tensor, reshape_default_7, softmax_default, _users=2)
sum_dim_IntList = CallFunction(operator_set.sum.dim_IntList, mul_Tensor_0, Ignored(), True, dtype=Ignored())
mul_Tensor_1 = CallFunction(operator_set.mul.Tensor, softmax_default, sum_dim_IntList)
sub_Tensor = CallFunction(operator_set.sub.Tensor, mul_Tensor_0, mul_Tensor_1, alpha=Ignored())
mul_Tensor_2 = CallFunction(operator_set.mul.Tensor, sub_Tensor, KeywordArg('scale_factor'), _users=2)
broadcast_to_default_5 = CallFunction(operator_set.broadcast_to.default, mul_Tensor_2, Ignored())
reshape_default_8 = CallFunction(operator_set.reshape.default, broadcast_to_default_5, Ignored())
permute_default_1 = CallFunction(operator_set.permute.default, permute_default, Ignored())
broadcast_to_default_6 = CallFunction(operator_set.broadcast_to.default, permute_default_1, Ignored())
reshape_default_9 = CallFunction(operator_set.reshape.default, broadcast_to_default_6, Ignored())
bmm_default_2 = CallFunction(operator_set.bmm.default, reshape_default_8, reshape_default_9)
reshape_default_10 = CallFunction(operator_set.reshape.default, bmm_default_2, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('query'), Ignored())
broadcast_to_default_7 = CallFunction(operator_set.broadcast_to.default, permute_default_2, Ignored())
reshape_default_11 = CallFunction(operator_set.reshape.default, broadcast_to_default_7, Ignored())
broadcast_to_default_8 = CallFunction(operator_set.broadcast_to.default, mul_Tensor_2, Ignored())
reshape_default_12 = CallFunction(operator_set.reshape.default, broadcast_to_default_8, Ignored())
bmm_default_3 = CallFunction(operator_set.bmm.default, reshape_default_11, reshape_default_12)
reshape_default_13 = CallFunction(operator_set.reshape.default, bmm_default_3, Ignored())
permute_default_3 = CallFunction(operator_set.permute.default, reshape_default_13, Ignored())
permute_default_4 = CallFunction(operator_set.permute.default, softmax_default, Ignored())
broadcast_to_default_9 = CallFunction(operator_set.broadcast_to.default, permute_default_4, Ignored())
reshape_default_14 = CallFunction(operator_set.reshape.default, broadcast_to_default_9, Ignored())
broadcast_to_default_10 = CallFunction(operator_set.broadcast_to.default, KeywordArg('tangents_1'), Ignored())
reshape_default_15 = CallFunction(operator_set.reshape.default, broadcast_to_default_10, Ignored())
bmm_default_4 = CallFunction(operator_set.bmm.default, reshape_default_14, reshape_default_15)
reshape_default_16 = CallFunction(operator_set.reshape.default, bmm_default_4, Ignored())
_sfdp_pattern_2_training = MultiOutputPattern([reshape_default_4,
  reshape_default_10,
  permute_default_3,
  reshape_default_16
])


broadcast_to_default = CallFunction(operator_set.broadcast_to.default, KeywordArg('query'), Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored())
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
mul_Tensor = CallFunction(operator_set.mul.Tensor, reshape_default_1, KeywordArg('scale_factor'))
softmax_default = CallFunction(operator_set.softmax.default, mul_Tensor, Ignored(), Ignored())
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, softmax_default, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, KeywordArg('value'), Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
_sfdp_pattern_2_inference = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored(), _users=0)


broadcast_to_default = CallFunction(operator_set.broadcast_to.default, KeywordArg('query'), Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored(), _users=2)
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
mul_Tensor = CallFunction(operator_set.mul.Tensor, reshape_default_1, KeywordArg('scale_factor'))
softmax_default = CallFunction(operator_set.softmax.default, mul_Tensor, Ignored(), Ignored(), _users=4)
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, softmax_default, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, KeywordArg('value'), Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
broadcast_to_default_3 = CallFunction(operator_set.broadcast_to.default, KeywordArg('tangents_1'), Ignored())
reshape_default_5 = CallFunction(operator_set.reshape.default, broadcast_to_default_3, Ignored())
permute_default_0 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored())
broadcast_to_default_4 = CallFunction(operator_set.broadcast_to.default, permute_default_0, Ignored())
reshape_default_6 = CallFunction(operator_set.reshape.default, broadcast_to_default_4, Ignored())
bmm_default_1 = CallFunction(operator_set.bmm.default, reshape_default_5, reshape_default_6)
reshape_default_7 = CallFunction(operator_set.reshape.default, bmm_default_1, Ignored())
mul_Tensor_0 = CallFunction(operator_set.mul.Tensor, reshape_default_7, softmax_default, _users=2)
sum_dim_IntList = CallFunction(operator_set.sum.dim_IntList, mul_Tensor_0, Ignored(), True, dtype=Ignored())
mul_Tensor_1 = CallFunction(operator_set.mul.Tensor, softmax_default, sum_dim_IntList)
sub_Tensor = CallFunction(operator_set.sub.Tensor, mul_Tensor_0, mul_Tensor_1, alpha=Ignored())
mul_Tensor_2 = CallFunction(operator_set.mul.Tensor, sub_Tensor, KeywordArg('scale_factor'), _users=2)
broadcast_to_default_5 = CallFunction(operator_set.broadcast_to.default, mul_Tensor_2, Ignored())
reshape_default_8 = CallFunction(operator_set.reshape.default, broadcast_to_default_5, Ignored())
permute_default_1 = CallFunction(operator_set.permute.default, permute_default, Ignored())
broadcast_to_default_6 = CallFunction(operator_set.broadcast_to.default, permute_default_1, Ignored())
reshape_default_9 = CallFunction(operator_set.reshape.default, broadcast_to_default_6, Ignored())
bmm_default_2 = CallFunction(operator_set.bmm.default, reshape_default_8, reshape_default_9)
reshape_default_10 = CallFunction(operator_set.reshape.default, bmm_default_2, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('query'), Ignored())
broadcast_to_default_7 = CallFunction(operator_set.broadcast_to.default, permute_default_2, Ignored())
reshape_default_11 = CallFunction(operator_set.reshape.default, broadcast_to_default_7, Ignored())
broadcast_to_default_8 = CallFunction(operator_set.broadcast_to.default, mul_Tensor_2, Ignored())
reshape_default_12 = CallFunction(operator_set.reshape.default, broadcast_to_default_8, Ignored())
bmm_default_3 = CallFunction(operator_set.bmm.default, reshape_default_11, reshape_default_12)
reshape_default_13 = CallFunction(operator_set.reshape.default, bmm_default_3, Ignored())
permute_default_3 = CallFunction(operator_set.permute.default, reshape_default_13, Ignored())
permute_default_4 = CallFunction(operator_set.permute.default, softmax_default, Ignored())
broadcast_to_default_9 = CallFunction(operator_set.broadcast_to.default, permute_default_4, Ignored())
reshape_default_14 = CallFunction(operator_set.reshape.default, broadcast_to_default_9, Ignored())
broadcast_to_default_10 = CallFunction(operator_set.broadcast_to.default, KeywordArg('tangents_1'), Ignored())
reshape_default_15 = CallFunction(operator_set.reshape.default, broadcast_to_default_10, Ignored())
bmm_default_4 = CallFunction(operator_set.bmm.default, reshape_default_14, reshape_default_15)
reshape_default_16 = CallFunction(operator_set.reshape.default, bmm_default_4, Ignored())
_sfdp_pattern_2_half_training = MultiOutputPattern([reshape_default_4,
  reshape_default_10,
  permute_default_3,
  reshape_default_16
])


broadcast_to_default = CallFunction(operator_set.broadcast_to.default, KeywordArg('query'), Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored())
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
mul_Tensor = CallFunction(operator_set.mul.Tensor, reshape_default_1, KeywordArg('scale_factor'))
softmax_default = CallFunction(operator_set.softmax.default, mul_Tensor, Ignored(), Ignored())
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, softmax_default, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, KeywordArg('value'), Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
_sfdp_pattern_2_half_inference = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored(), _users=0)
