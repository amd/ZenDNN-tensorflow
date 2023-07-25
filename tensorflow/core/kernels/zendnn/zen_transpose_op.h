/*******************************************************************************
 * Modifications Copyright (c) 2023 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 *******************************************************************************/

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_ZEN_TRANSPOSE_OP_H_
#define TENSORFLOW_KERNELS_ZEN_TRANSPOSE_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

template <typename T>
class ZenTransposeOp : public OpKernel {
 public:
  bool reorder_before, reorder_after, reset;
  int in_links, out_links;

  explicit ZenTransposeOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reorder_before", &reorder_before));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reorder_after", &reorder_after));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reset", &reset));
  }

  void Compute(OpKernelContext *ctx) override;

 protected:
  virtual Status DoTranspose(OpKernelContext *ctx, const Tensor &in,
                             gtl::ArraySlice<int32> perm, Tensor *out) = 0;
  virtual bool IsConjugate() const { return false; }
};
template <typename T>
class ZenTransposeCpuOp : public ZenTransposeOp<T> {
 public:
  explicit ZenTransposeCpuOp(OpKernelConstruction *ctx)
      : ZenTransposeOp<T>(ctx) {}

 protected:
  Status DoTranspose(OpKernelContext *ctx, const Tensor &in,
                     gtl::ArraySlice<int32> perm, Tensor *out) override;
};

// Conjugating transpose ops.
template <typename T>
class ConjugateZenTransposeCpuOp : public ZenTransposeOp<T> {
 public:
  explicit ConjugateZenTransposeCpuOp<T>(OpKernelConstruction *ctx)
      : ZenTransposeOp<T>(ctx) {}

 protected:
  Status DoTranspose(OpKernelContext *ctx, const Tensor &in,
                     gtl::ArraySlice<int32> perm, Tensor *out) override;
  bool IsConjugate() const override { return true; }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_ZEN_TRANSPOSE_OP_H_
