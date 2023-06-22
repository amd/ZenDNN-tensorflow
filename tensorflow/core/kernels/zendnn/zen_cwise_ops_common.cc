/*******************************************************************************
 * Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 *******************************************************************************/

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0(the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/math_ops.cc.
#include "tensorflow/core/kernels/zendnn/zen_cwise_ops_common.h"

namespace tensorflow {

ZenBinaryOpShared::ZenBinaryOpShared(OpKernelConstruction *ctx, DataType out,
                                     DataType in)
    : OpKernel(ctx) {
  OP_REQUIRES_OK(ctx, ctx->MatchSignature({in, in}, {out}));
}

void ZenBinaryOpShared::SetUnimplementedError(OpKernelContext *ctx) {
  ctx->SetStatus(errors::Unimplemented(
      "Broadcast between ", ctx->input(0).shape().DebugString(), " and ",
      ctx->input(1).shape().DebugString(), " is not supported yet."));
}

void ZenBinaryOpShared::SetComputeError(OpKernelContext *ctx) {
  // For speed, errors during compute are caught only via boolean flag, with no
  // associated information.  This is sufficient for now, since the only binary
  // ops that have compute errors are integer division and mod, and the only
  // error they produce is zero division.
  const string &op = ctx->op_kernel().type_string();
  if ((op == "Div" || op == "Mod" || op == "FloorMod" || op == "FloorDiv") &&
      DataTypeIsInteger(ctx->op_kernel().input_type(0))) {
    ctx->CtxFailure(errors::InvalidArgument("Integer division by zero"));
  } else if ((op == "Pow") &&
             DataTypeIsInteger(ctx->op_kernel().input_type(0)) &&
             DataTypeIsSigned(ctx->op_kernel().input_type(1))) {
    ctx->CtxFailure(errors::InvalidArgument(
        "Integers to negative integer powers are not allowed"));
  } else {
    ctx->CtxFailure(
        errors::Internal("Unexpected error in binary operator "
                         "(only integer div and mod should have errors)"));
  }
}

ZenBinaryOpShared::ZenBinaryOpState::ZenBinaryOpState(OpKernelContext *ctx,
                                                      int in_links,
                                                      int out_links, bool reset)
    : in0(ctx->input(0)),
      in1(ctx->input(1)),
      bcast(BCast::FromShape(in0.shape()), BCast::FromShape(in1.shape())),
      in_links(in_links),
      out_links(out_links),
      reset(reset),
      in0_reuse(false),
      in1_reuse(false) {
  if (!bcast.IsValid()) {
    bool incompatible_shape_error;
    bool has_attr =
        TryGetNodeAttr(ctx->op_kernel().def(), "incompatible_shape_error",
                       &(incompatible_shape_error));
    if (has_attr && !incompatible_shape_error) {
      const string &op = ctx->op_kernel().type_string();
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
      result = (op == "NotEqual");
      return;
    }

    ctx->SetStatus(errors::InvalidArgument(
        "Incompatible shapes: ", in0.shape().DebugString(), " vs. ",
        in1.shape().DebugString()));
    return;
  }

  const TensorShape output_shape = BCast::ToShape(bcast.output_shape());
  out_num_elements = output_shape.num_elements();
  in0_num_elements = in0.NumElements();
  in1_num_elements = in1.NumElements();
  zenTensorType out_type = zenTensorType::FLOAT;

  zendnnEnv zenEnvObj = readEnv();
  int zenEnableMemPool =
      zenEnvObj.zenEnableMemPool && ctx->expected_output_dtype(0) == DT_FLOAT;
  ZenMemoryPool<float> *zenPoolBuffer = NULL;

  // Output buffer:
  // (1) Reuse any of the input buffers for output
  // (2) If not (1), then get buffer from ZenMemPool
  // (3) If not (1) and (2), then allocate the buffer for output
  if (ctx->forward_input_to_output_with_shape(0, 0, output_shape, &out)) {
    in0_reuse = true;
  } else if (ctx->forward_input_to_output_with_shape(1, 0, output_shape,
                                                     &out)) {
    in1_reuse = true;
  } else {
    // ZenMemPool Optimization reuse o/p tensors from the pool. By default
    // its enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory
    // pool optimization
    // Cases where tensors in pool are not free or requested size is more
    // than available tensor size in Pool, control will fall back to
    // default way of allocation i.e. with allocate_output(..)
    if (zenEnableMemPool) {
      unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
      zenPoolBuffer = ZenMemoryPool<float>::getZenMemPool(threadID);
      if (zenPoolBuffer) {
        int status = zenPoolBuffer->acquireZenPoolTensor(
            ctx, &out, output_shape, out_links, reset, out_type);
        if (status) {
          zenEnableMemPool = false;
        }
      } else {
        zenEnableMemPool = false;
      }
    }
    if (!zenEnableMemPool) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &out));
    }
  }

  ndims = static_cast<int>(bcast.x_reshape().size());
}

REGISTER(ZenBinaryOp, CPU, "ZenAdd", functor::add, float);
REGISTER(ZenBinaryOp, CPU, "ZenAddV2", functor::add, float);
REGISTER(ZenBinaryOp, CPU, "ZenSub", functor::sub, float);
REGISTER(ZenBinaryOp, CPU, "ZenMul", functor::mul, float);
REGISTER(ZenBinaryOp, CPU, "ZenMaximum", functor::maximum, float);
REGISTER(ZenBinaryOp, CPU, "ZenSquaredDifference", functor::squared_difference,
         float);

}  // end namespace tensorflow
