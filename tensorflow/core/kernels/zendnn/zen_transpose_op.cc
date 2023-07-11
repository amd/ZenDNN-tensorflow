/*******************************************************************************
 * Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights
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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/zendnn/zen_transpose_op.h"

#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "zendnn.hpp"
#include "zendnn_helper.hpp"
#include "zendnn_logging.hpp"

namespace tensorflow {

// inv = ZenInvertPermutationOp(T<int32/int64> p) takes a permutation of
// integers 0, 1, ..., n - 1 and returns the inverted
// permutation of p. I.e., inv[p[i]] == i, for i in [0 .. n).
//
// REQUIRES: input is a vector of int32 or int64.
// REQUIRES: input is a permutation of 0, 1, ..., n-1.

template <typename T>
class ZenInvertPermutationOp : public OpKernel {
 public:
  bool reorder_before, reorder_after, reset;
  int in_links, out_links;

  explicit ZenInvertPermutationOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reorder_before", &reorder_before));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reorder_after", &reorder_after));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reset", &reset));
  }

  void Compute(OpKernelContext *context) override {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: ZenInvertPermutationOp (TF kernel): In Compute!");

    const Tensor &input = context->input(0);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input.shape()),
        errors::InvalidArgument("invert_permutation expects a 1D vector."));
    auto Tin = input.vec<T>();
    OP_REQUIRES(context,
                FastBoundsCheck(Tin.size(), std::numeric_limits<int32>::max()),
                errors::InvalidArgument("permutation of nonnegative int32s "
                                        "must have <= int32 max elements"));
    const T N = static_cast<T>(Tin.size());  // Safe: bounds-checked above.
    // Update the output type
    zenTensorType out_type = zenTensorType::FLOAT;

    zendnnEnv zenEnvObj = readEnv();
    Tensor *output = nullptr;
    int zenEnableMemPool = zenEnvObj.zenEnableMemPool &&
                           context->expected_output_dtype(0) == DT_FLOAT;
    ZenMemoryPool<T> *zenPoolBuffer = NULL;

    // ZenMemPool Optimization reuse o/p tensors from the pool. By default
    //  its enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory
    //  pool optimization
    //  Cases where tensors in pool are not free or requested size is more
    //  than available tensor size in Pool, control will fall back to
    //  default way of allocation i.e. with allocate_output(..)
    if (zenEnableMemPool) {
      unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
      zenPoolBuffer = ZenMemoryPool<T>::getZenMemPool(threadID);
      if (zenPoolBuffer) {
        int status = zenPoolBuffer->acquireZenPoolTensor(
            context, &output, input.shape(), out_links, reset, out_type);
        if (status) {
          zenEnableMemPool = false;
        }
      } else {
        zenEnableMemPool = false;
      }
    }
    if (!zenEnableMemPool) {
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, input.shape(), &output));
    }

    auto Tout = output->vec<T>();
    std::fill_n(Tout.data(), N, -1);
    for (int i = 0; i < N; ++i) {
      const T d = internal::SubtleMustCopy(Tin(i));
      OP_REQUIRES(context, FastBoundsCheck(d, N),
                  errors::InvalidArgument(d, " is not between 0 and ", N));
      OP_REQUIRES(context, Tout(d) == -1,
                  errors::InvalidArgument(d, " is duplicated in the input."));
      Tout(d) = i;
    }

    // If ZenMemPool Optimization is enabled(default), update the state of
    //  Memory pool based on input_array address
    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      T *input_array = const_cast<T *>(input.template flat<T>().data());
      zenPoolBuffer->zenMemPoolFree(context, (void *)input_array);
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("_ZenInvertPermutation").Device(DEVICE_CPU).TypeConstraint<int32>("T"),
    ZenInvertPermutationOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("_ZenInvertPermutation").Device(DEVICE_CPU).TypeConstraint<int64>("T"),
    ZenInvertPermutationOp<int64>);

namespace {
template <typename Tperm>
Status PermutationHelper(const Tensor &perm, const int dims,
                         std::vector<int32> *permutation) {
  auto Vperm = perm.vec<Tperm>();
  if (dims != Vperm.size()) {
    return errors::InvalidArgument("transpose expects a vector of size ", dims,
                                   ". But input(1) is a vector of size ",
                                   Vperm.size());
  }
  // using volatile instead of SubtleMustCopy here so that the
  // asynchrony boundary is permutation.
  const volatile Tperm *perm_begin =
      reinterpret_cast<const volatile Tperm *>(Vperm.data());
  *permutation = std::vector<int32>(perm_begin, perm_begin + dims);

  return OkStatus();
}
}  // namespace

// output = ZenTransposeOp(T<any> input, T<int32> perm) takes a tensor
// of type T and rank N, and a permutation of 0, 1, ..., N-1. It
// shuffles the dimensions of the input tensor according to permutation.
//
// Specifically, the returned tensor output meets the following condition:
// 1) output.dims() == input.dims();
// 2) output.dim_size(i) == input.dim_size(perm[i]);
// 3) output.tensor<T, N>(i_0, i_1, ..., i_N-1) ==
//      input.tensor<T, N>(j_0, j_1, ..., j_N-1),
//    where i_s == j_{perm[s]}
//
// REQUIRES: perm is a vector of int32.
// REQUIRES: input.dims() == perm.size().
// REQUIRES: perm is a permutation.

void ZenTransposeOp::Compute(OpKernelContext *ctx) {
  zendnnInfo(ZENDNN_FWKLOG,
             "ZEN-OP-DEF: ZenTransposeOp (TF kernel): In Compute!");

  const Tensor &input = ctx->input(0);
  const Tensor &perm = ctx->input(1);
  // Preliminary validation of sizes.
  OP_REQUIRES(ctx, TensorShapeUtils::IsVector(perm.shape()),
              errors::InvalidArgument("perm must be a vector, not ",
                                      perm.shape().DebugString()));

  // Although Tperm may be an int64 type, an int32 is sufficient to hold
  // dimension range values, so the narrowing here should be safe.
  std::vector<int32> permutation;
  const int dims = input.dims();
  if (perm.dtype() == DT_INT32) {
    OP_REQUIRES_OK(ctx, PermutationHelper<int32>(perm, dims, &permutation));
  } else {
    OP_REQUIRES_OK(ctx, PermutationHelper<int64>(perm, dims, &permutation));
  }
  TensorShape shape;

  // Check whether permutation is a permutation of integers of [0 .. dims).
  gtl::InlinedVector<bool, 8> bits(dims);
  bool is_identity = true;
  for (int i = 0; i < dims; ++i) {
    const int32 d = permutation[i];
    OP_REQUIRES(
        ctx, 0 <= d && d < dims,
        errors::InvalidArgument(d, " is out of range [0 .. ", dims, ")"));
    bits[d] = true;
    const auto dim_size = input.dim_size(d);
    shape.AddDim(dim_size);
    if (d != i) {
      is_identity = false;
    }
  }
  for (int i = 0; i < dims; ++i) {
    OP_REQUIRES(ctx, bits[i],
                errors::InvalidArgument(i, " is missing from {",
                                        absl::StrJoin(permutation, ","), "}."));
  }

  // 0-D, 1-D, and identity transposes do nothing.
  if (!IsConjugate() && (dims <= 1 || is_identity)) {
    ctx->set_output(0, input);
    return;
  } else if (!IsConjugate() && internal::NonSingletonDimensionsAlign(
                                   input.shape(), permutation)) {
    Tensor output;
    OP_REQUIRES(ctx, output.CopyFrom(input, shape),
                errors::Unknown("Error reshaping Tensor."));
    ctx->set_output(0, output);
    return;
  }
  zenTensorType out_type = zenTensorType::FLOAT;

  zendnnEnv zenEnvObj = readEnv();
  Tensor *output = nullptr;
  int zenEnableMemPool =
      zenEnvObj.zenEnableMemPool && ctx->expected_output_dtype(0) == DT_FLOAT;
  ZenMemoryPool<float> *zenPoolBuffer = NULL;

  // ZenMemPool Optimization reuse o/p tensors from the pool. By default
  //  its enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory
  //  pool optimization
  //  Cases where tensors in pool are not free or requested size is more
  //  than available tensor size in Pool, control will fall back to
  //  default way of allocation i.e. with allocate_output(..)
  if (zenEnableMemPool) {
    unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
    zenPoolBuffer = ZenMemoryPool<float>::getZenMemPool(threadID);
    if (zenPoolBuffer) {
      int status = zenPoolBuffer->acquireZenPoolTensor(
          ctx, &output, shape, out_links, reset, out_type);
      if (status) {
        zenEnableMemPool = false;
      }
    } else {
      zenEnableMemPool = false;
    }
  }
  if (!zenEnableMemPool) {
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
  }

  if (shape.num_elements() > 0) {
    OP_REQUIRES_OK(ctx, DoTranspose(ctx, input, permutation, output));
  }

  // If ZenMemPool Optimization is enabled(default), update the state of
  //  Memory pool based on input_array address
  if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
    float *input_array =
        const_cast<float *>(input.template flat<float>().data());
    zenPoolBuffer->zenMemPoolFree(ctx, (void *)input_array);
  }
}

Status ZenTransposeCpuOp::DoTranspose(OpKernelContext *ctx, const Tensor &in,
                                      gtl::ArraySlice<int32> perm,
                                      Tensor *out) {
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<CPUDevice>(), in, perm,
                                   out);
}

Status ConjugateZenTransposeCpuOp::DoTranspose(OpKernelContext *ctx,
                                               const Tensor &in,
                                               gtl::ArraySlice<int32> perm,
                                               Tensor *out) {
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoConjugateTranspose(ctx->eigen_device<CPUDevice>(), in,
                                            perm, out);
}

#define REGISTER(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("_ZenTranspose")          \
                              .Device(DEVICE_CPU)       \
                              .TypeConstraint<T>("T")   \
                              .HostMemory("perm"),      \
                          ZenTransposeCpuOp);           \
  REGISTER_KERNEL_BUILDER(Name("_ZenConjugateTranspose") \
                              .Device(DEVICE_CPU)       \
                              .TypeConstraint<T>("T")   \
                              .HostMemory("perm"),      \
                          ConjugateZenTransposeCpuOp);

TF_CALL_ALL_TYPES(REGISTER)
#undef REGISTER

}  // namespace tensorflow
