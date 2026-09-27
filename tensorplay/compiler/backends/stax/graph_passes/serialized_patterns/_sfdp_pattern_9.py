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
rand_like_default = CallFunction(operator_set.rand_like.default, softmax_default, dtype=Ignored(), device=None)
gt_Scalar = CallFunction(operator_set.gt.Scalar, rand_like_default, KeywordArg('dropout_p'))
mul_Tensor = CallFunction(operator_set.mul.Tensor, gt_Scalar, softmax_default)
mul_Scalar = CallFunction(operator_set.mul.Scalar, mul_Tensor, Ignored())
to_dtype_0 = CallFunction(operator_set.to.dtype, mul_Scalar, Ignored(), False, False, None)
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, to_dtype_0, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored())
to_dtype_1 = CallFunction(operator_set.to.dtype, permute_default_2, Ignored(), False, False, None)
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, to_dtype_1, Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
permute_default_3 = CallFunction(operator_set.permute.default, to_dtype_0, Ignored())
broadcast_to_default_3 = CallFunction(operator_set.broadcast_to.default, permute_default_3, Ignored())
reshape_default_5 = CallFunction(operator_set.reshape.default, broadcast_to_default_3, Ignored())
broadcast_to_default_4 = CallFunction(operator_set.broadcast_to.default, KeywordArg('tangents_1'), Ignored())
reshape_default_6 = CallFunction(operator_set.reshape.default, broadcast_to_default_4, Ignored())
bmm_default_1 = CallFunction(operator_set.bmm.default, reshape_default_5, reshape_default_6)
reshape_default_7 = CallFunction(operator_set.reshape.default, bmm_default_1, Ignored())
to_device = CallFunction(operator_set.to.device, reshape_default_7, Ignored(), Ignored(), False, False, None)
permute_backward_default = CallFunction(operator_set.permute_backward.default, to_device, KeywordArg('value'), Ignored())
_sfdp_pattern_9_training = MultiOutputPattern([reshape_default_4,
  None,
  None,
  permute_backward_default
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
_sfdp_pattern_9_inference = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())


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
rand_like_default = CallFunction(operator_set.rand_like.default, softmax_default, dtype=Ignored(), device=None)
gt_Scalar = CallFunction(operator_set.gt.Scalar, rand_like_default, KeywordArg('dropout_p'))
mul_Tensor = CallFunction(operator_set.mul.Tensor, gt_Scalar, softmax_default)
mul_Scalar = CallFunction(operator_set.mul.Scalar, mul_Tensor, Ignored())
to_dtype_0 = CallFunction(operator_set.to.dtype, mul_Scalar, Ignored(), False, False, None)
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, to_dtype_0, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored())
to_dtype_1 = CallFunction(operator_set.to.dtype, permute_default_2, Ignored(), False, False, None)
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, to_dtype_1, Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
permute_default_3 = CallFunction(operator_set.permute.default, to_dtype_0, Ignored())
broadcast_to_default_3 = CallFunction(operator_set.broadcast_to.default, permute_default_3, Ignored())
reshape_default_5 = CallFunction(operator_set.reshape.default, broadcast_to_default_3, Ignored())
broadcast_to_default_4 = CallFunction(operator_set.broadcast_to.default, KeywordArg('tangents_1'), Ignored())
reshape_default_6 = CallFunction(operator_set.reshape.default, broadcast_to_default_4, Ignored())
bmm_default_1 = CallFunction(operator_set.bmm.default, reshape_default_5, reshape_default_6)
reshape_default_7 = CallFunction(operator_set.reshape.default, bmm_default_1, Ignored())
permute_backward_default = CallFunction(operator_set.permute_backward.default, reshape_default_7, KeywordArg('value'), Ignored())
_sfdp_pattern_9_half_training = MultiOutputPattern([reshape_default_4,
  None,
  None,
  permute_backward_default
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
_sfdp_pattern_9_half_inference = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
