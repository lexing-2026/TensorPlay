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
to_dtype = CallFunction(operator_set.to.dtype, reshape_default_1, Ignored(), False, False, None)
to_dtype_0 = CallFunction(operator_set.to.dtype, to_dtype, Ignored(), False, False, None)
softmax_default = CallFunction(operator_set.softmax.default, to_dtype_0, Ignored(), Ignored())
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, softmax_default, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored(), _users=2)
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, permute_default_2, Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
_sfdp_pattern_23_inference = MultiOutputPattern([reshape_default_4,
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
to_dtype = CallFunction(operator_set.to.dtype, reshape_default_1, Ignored(), False, False, None)
to_dtype_0 = CallFunction(operator_set.to.dtype, to_dtype, Ignored(), False, False, None)
softmax_default = CallFunction(operator_set.softmax.default, to_dtype_0, Ignored(), Ignored())
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, softmax_default, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored(), _users=2)
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, permute_default_2, Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
_sfdp_pattern_23_bs1_inference = MultiOutputPattern([reshape_default_4,
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
to_dtype = CallFunction(operator_set.to.dtype, reshape_default_1, Ignored(), False, False, None)
to_dtype_0 = CallFunction(operator_set.to.dtype, to_dtype, Ignored(), False, False, None)
to_dtype_1 = CallFunction(operator_set.to.dtype, to_dtype_0, Ignored(), False, False, None)
softmax_default = CallFunction(operator_set.softmax.default, to_dtype_1, Ignored(), Ignored())
to_dtype_2 = CallFunction(operator_set.to.dtype, softmax_default, Ignored(), False, False, None)
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, to_dtype_2, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored(), _users=2)
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, permute_default_2, Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
_sfdp_pattern_23_half_inference = MultiOutputPattern([reshape_default_4,
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
to_dtype = CallFunction(operator_set.to.dtype, reshape_default_1, Ignored(), False, False, None)
to_dtype_0 = CallFunction(operator_set.to.dtype, to_dtype, Ignored(), False, False, None)
to_dtype_1 = CallFunction(operator_set.to.dtype, to_dtype_0, Ignored(), False, False, None)
softmax_default = CallFunction(operator_set.softmax.default, to_dtype_1, Ignored(), Ignored())
to_dtype_2 = CallFunction(operator_set.to.dtype, softmax_default, Ignored(), False, False, None)
broadcast_to_default_1 = CallFunction(operator_set.broadcast_to.default, to_dtype_2, Ignored())
reshape_default_2 = CallFunction(operator_set.reshape.default, broadcast_to_default_1, Ignored())
permute_default_2 = CallFunction(operator_set.permute.default, KeywordArg('value'), Ignored(), _users=2)
broadcast_to_default_2 = CallFunction(operator_set.broadcast_to.default, permute_default_2, Ignored())
reshape_default_3 = CallFunction(operator_set.reshape.default, broadcast_to_default_2, Ignored())
bmm_default_0 = CallFunction(operator_set.bmm.default, reshape_default_2, reshape_default_3)
reshape_default_4 = CallFunction(operator_set.reshape.default, bmm_default_0, Ignored())
_sfdp_pattern_23_half_bs1_inference = MultiOutputPattern([reshape_default_4,
  permute_default_0,
  permute_default_2
])
