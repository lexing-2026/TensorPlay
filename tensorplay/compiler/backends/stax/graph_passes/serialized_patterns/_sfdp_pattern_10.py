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
div_Scalar = CallFunction(operator_set.div.Scalar, permute_default, KeywordArg('inv_scale'), _users=2)
broadcast_to_default = CallFunction(operator_set.broadcast_to.default, div_Scalar, Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default_0 = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored())
permute_default_1 = CallFunction(operator_set.permute.default, permute_default_0, Ignored(), _users=2)
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default_1, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
to_dtype = CallFunction(operator_set.to.dtype, reshape_default_1, Ignored(), False, False, None)
softmax_default = CallFunction(operator_set.softmax.default, to_dtype, Ignored(), Ignored(), _users=3)
to_dtype_0 = CallFunction(operator_set.to.dtype, softmax_default, Ignored(), False, False, None, _users=2)
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, to_dtype_0, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored())
to_dtype_1 = CallFunction(operator_set.to.dtype, permute_default_2, Ignored(), False, False, None, _users=2)
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, to_dtype_1, Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
broadcast_to_default_3 = CallFunction(operator_set.broadcast_to.default, KeywordArg('tangents_1'), Ignored())
reshape_default_5 = CallFunction(operator_set.reshape.default, broadcast_to_default_3, Ignored())
permute_default_3 = CallFunction(operator_set.permute.default, to_dtype_1, Ignored())
broadcast_to_default_4 = CallFunction(operator_set.broadcast_to.default, permute_default_3, Ignored())
reshape_default_6 = CallFunction(operator_set.reshape.default, broadcast_to_default_4, Ignored())
bmm_default_1 = CallFunction(operator_set.bmm.default, reshape_default_5, reshape_default_6)
reshape_default_7 = CallFunction(operator_set.reshape.default, bmm_default_1, Ignored())
to_device = CallFunction(operator_set.to.device, reshape_default_7, Ignored(), Ignored(), False, False, None)
mul_Tensor = CallFunction(operator_set.mul.Tensor, to_device, softmax_default, _users=2)
sum_dim_IntList = CallFunction(operator_set.sum.dim_IntList, mul_Tensor, Ignored(), True, dtype=Ignored())
mul_Tensor_0 = CallFunction(operator_set.mul.Tensor, softmax_default, sum_dim_IntList)
sub_Tensor = CallFunction(operator_set.sub.Tensor, mul_Tensor, mul_Tensor_0, alpha=Ignored(), _users=2)
broadcast_to_default_5 = CallFunction(operator_set.broadcast_to.default, sub_Tensor, Ignored())
reshape_default_8 = CallFunction(operator_set.reshape.default, broadcast_to_default_5, Ignored())
permute_default_4 = CallFunction(operator_set.permute.default, permute_default_1, Ignored())
broadcast_to_default_6 = CallFunction(operator_set.broadcast_to.default, permute_default_4, Ignored())
reshape_default_9 = CallFunction(operator_set.reshape.default, broadcast_to_default_6, Ignored())
bmm_default_2 = CallFunction(operator_set.bmm.default, reshape_default_8, reshape_default_9)
reshape_default_10 = CallFunction(operator_set.reshape.default, bmm_default_2, Ignored())
div_Scalar_0 = CallFunction(operator_set.div.Scalar, reshape_default_10, KeywordArg('inv_scale'))
permute_backward_default = CallFunction(operator_set.permute_backward.default, div_Scalar_0, KeywordArg('query'), Ignored())
permute_default_5 = CallFunction(operator_set.permute.default, div_Scalar, Ignored())
broadcast_to_default_7 = CallFunction(operator_set.broadcast_to.default, permute_default_5, Ignored())
reshape_default_11 = CallFunction(operator_set.reshape.default, broadcast_to_default_7, Ignored())
broadcast_to_default_8 = CallFunction(operator_set.broadcast_to.default, sub_Tensor, Ignored())
reshape_default_12 = CallFunction(operator_set.reshape.default, broadcast_to_default_8, Ignored())
bmm_default_3 = CallFunction(operator_set.bmm.default, reshape_default_11, reshape_default_12)
reshape_default_13 = CallFunction(operator_set.reshape.default, bmm_default_3, Ignored())
permute_default_6 = CallFunction(operator_set.permute.default, reshape_default_13, Ignored())
permute_backward_default_0 = CallFunction(operator_set.permute_backward.default, permute_default_6, KeywordArg('key'), Ignored())
permute_default_7 = CallFunction(operator_set.permute.default, to_dtype_0, Ignored())
broadcast_to_default_9 = CallFunction(operator_set.broadcast_to.default, permute_default_7, Ignored())
reshape_default_14 = CallFunction(operator_set.reshape.default, broadcast_to_default_9, Ignored())
broadcast_to_default_10 = CallFunction(operator_set.broadcast_to.default, KeywordArg('tangents_1'), Ignored())
reshape_default_15 = CallFunction(operator_set.reshape.default, broadcast_to_default_10, Ignored())
bmm_default_4 = CallFunction(operator_set.bmm.default, reshape_default_14, reshape_default_15)
reshape_default_16 = CallFunction(operator_set.reshape.default, bmm_default_4, Ignored())
to_device_0 = CallFunction(operator_set.to.device, reshape_default_16, Ignored(), Ignored(), False, False, None)
permute_backward_default_1 = CallFunction(operator_set.permute_backward.default, to_device_0, KeywordArg('value'), Ignored())
_sfdp_pattern_10_training = MultiOutputPattern([reshape_default_4,
  permute_backward_default,
  permute_backward_default_0,
  permute_backward_default_1
])


permute_default = CallFunction(operator_set.permute.default, KeywordArg('query'), Ignored())
div_Scalar = CallFunction(operator_set.div.Scalar, permute_default, KeywordArg('inv_scale'))
broadcast_to_default = CallFunction(operator_set.broadcast_to.default, div_Scalar, Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default_0 = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored())
permute_default_1 = CallFunction(operator_set.permute.default, permute_default_0, Ignored())
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default_1, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
to_dtype = CallFunction(operator_set.to.dtype, reshape_default_1, Ignored(), False, False, None)
softmax_default = CallFunction(operator_set.softmax.default, to_dtype, Ignored(), Ignored())
to_dtype_0 = CallFunction(operator_set.to.dtype, softmax_default, Ignored(), False, False, None)
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, to_dtype_0, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored())
to_dtype_1 = CallFunction(operator_set.to.dtype, permute_default_2, Ignored(), False, False, None)
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, to_dtype_1, Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
_sfdp_pattern_10_inference = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored(), _users=0)


permute_default = CallFunction(operator_set.permute.default, KeywordArg('query'), Ignored())
div_Scalar = CallFunction(operator_set.div.Scalar, permute_default, KeywordArg('inv_scale'), _users=2)
broadcast_to_default = CallFunction(operator_set.broadcast_to.default, div_Scalar, Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default_0 = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored())
permute_default_1 = CallFunction(operator_set.permute.default, permute_default_0, Ignored(), _users=2)
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default_1, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
to_dtype = CallFunction(operator_set.to.dtype, reshape_default_1, Ignored(), False, False, None)
softmax_default = CallFunction(operator_set.softmax.default, to_dtype, Ignored(), Ignored(), _users=3)
to_dtype_0 = CallFunction(operator_set.to.dtype, softmax_default, Ignored(), False, False, None, _users=2)
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, to_dtype_0, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored())
to_dtype_1 = CallFunction(operator_set.to.dtype, permute_default_2, Ignored(), False, False, None, _users=2)
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, to_dtype_1, Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
broadcast_to_default_3 = CallFunction(operator_set.broadcast_to.default, KeywordArg('tangents_1'), Ignored())
reshape_default_5 = CallFunction(operator_set.reshape.default, broadcast_to_default_3, Ignored())
permute_default_3 = CallFunction(operator_set.permute.default, to_dtype_1, Ignored())
broadcast_to_default_4 = CallFunction(operator_set.broadcast_to.default, permute_default_3, Ignored())
reshape_default_6 = CallFunction(operator_set.reshape.default, broadcast_to_default_4, Ignored())
bmm_default_1 = CallFunction(operator_set.bmm.default, reshape_default_5, reshape_default_6)
reshape_default_7 = CallFunction(operator_set.reshape.default, bmm_default_1, Ignored())
to_device = CallFunction(operator_set.to.device, reshape_default_7, Ignored(), Ignored(), False, False, None)
mul_Tensor = CallFunction(operator_set.mul.Tensor, to_device, softmax_default, _users=2)
sum_dim_IntList = CallFunction(operator_set.sum.dim_IntList, mul_Tensor, Ignored(), True, dtype=Ignored())
mul_Tensor_0 = CallFunction(operator_set.mul.Tensor, softmax_default, sum_dim_IntList)
sub_Tensor = CallFunction(operator_set.sub.Tensor, mul_Tensor, mul_Tensor_0, alpha=Ignored())
to_device_0 = CallFunction(operator_set.to.device, sub_Tensor, Ignored(), Ignored(), False, False, None, _users=2)
broadcast_to_default_5 = CallFunction(operator_set.broadcast_to.default, to_device_0, Ignored())
reshape_default_8 = CallFunction(operator_set.reshape.default, broadcast_to_default_5, Ignored())
permute_default_4 = CallFunction(operator_set.permute.default, permute_default_1, Ignored())
broadcast_to_default_6 = CallFunction(operator_set.broadcast_to.default, permute_default_4, Ignored())
reshape_default_9 = CallFunction(operator_set.reshape.default, broadcast_to_default_6, Ignored())
bmm_default_2 = CallFunction(operator_set.bmm.default, reshape_default_8, reshape_default_9)
reshape_default_10 = CallFunction(operator_set.reshape.default, bmm_default_2, Ignored())
div_Scalar_0 = CallFunction(operator_set.div.Scalar, reshape_default_10, KeywordArg('inv_scale'))
permute_backward_default = CallFunction(operator_set.permute_backward.default, div_Scalar_0, KeywordArg('query'), Ignored())
permute_default_5 = CallFunction(operator_set.permute.default, div_Scalar, Ignored())
broadcast_to_default_7 = CallFunction(operator_set.broadcast_to.default, permute_default_5, Ignored())
reshape_default_11 = CallFunction(operator_set.reshape.default, broadcast_to_default_7, Ignored())
broadcast_to_default_8 = CallFunction(operator_set.broadcast_to.default, to_device_0, Ignored())
reshape_default_12 = CallFunction(operator_set.reshape.default, broadcast_to_default_8, Ignored())
bmm_default_3 = CallFunction(operator_set.bmm.default, reshape_default_11, reshape_default_12)
reshape_default_13 = CallFunction(operator_set.reshape.default, bmm_default_3, Ignored())
permute_default_6 = CallFunction(operator_set.permute.default, reshape_default_13, Ignored())
permute_backward_default_0 = CallFunction(operator_set.permute_backward.default, permute_default_6, KeywordArg('key'), Ignored())
permute_default_7 = CallFunction(operator_set.permute.default, to_dtype_0, Ignored())
broadcast_to_default_9 = CallFunction(operator_set.broadcast_to.default, permute_default_7, Ignored())
reshape_default_14 = CallFunction(operator_set.reshape.default, broadcast_to_default_9, Ignored())
broadcast_to_default_10 = CallFunction(operator_set.broadcast_to.default, KeywordArg('tangents_1'), Ignored())
reshape_default_15 = CallFunction(operator_set.reshape.default, broadcast_to_default_10, Ignored())
bmm_default_4 = CallFunction(operator_set.bmm.default, reshape_default_14, reshape_default_15)
reshape_default_16 = CallFunction(operator_set.reshape.default, bmm_default_4, Ignored())
permute_backward_default_1 = CallFunction(operator_set.permute_backward.default, reshape_default_16, KeywordArg('value'), Ignored())
_sfdp_pattern_10_half_training = MultiOutputPattern([reshape_default_4,
  permute_backward_default,
  permute_backward_default_0,
  permute_backward_default_1
])


permute_default = CallFunction(operator_set.permute.default, KeywordArg('query'), Ignored())
div_Scalar = CallFunction(operator_set.div.Scalar, permute_default, KeywordArg('inv_scale'))
broadcast_to_default = CallFunction(operator_set.broadcast_to.default, div_Scalar, Ignored())
reshape_default = CallFunction(operator_set.reshape.default, broadcast_to_default, Ignored())
permute_default_0 = CallFunction(operator_set.permute.default, KeywordArg('key'), Ignored())
permute_default_1 = CallFunction(operator_set.permute.default, permute_default_0, Ignored())
broadcast_to_default_0 = CallFunction(operator_set.broadcast_to.default, permute_default_1, Ignored())
reshape_default_0 = CallFunction(operator_set.reshape.default, broadcast_to_default_0, Ignored())
bmm_default = CallFunction(operator_set.bmm.default, reshape_default, reshape_default_0)
reshape_default_1 = CallFunction(operator_set.reshape.default, bmm_default, Ignored())
to_dtype = CallFunction(operator_set.to.dtype, reshape_default_1, Ignored(), False, False, None)
softmax_default = CallFunction(operator_set.softmax.default, to_dtype, Ignored(), Ignored())
to_dtype_0 = CallFunction(operator_set.to.dtype, softmax_default, Ignored(), False, False, None)
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, to_dtype_0, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored())
to_dtype_1 = CallFunction(operator_set.to.dtype, permute_default_2, Ignored(), False, False, None)
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, to_dtype_1, Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
_sfdp_pattern_10_half_inference = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored(), _users=0)
