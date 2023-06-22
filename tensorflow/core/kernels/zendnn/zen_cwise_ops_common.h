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

#ifndef TENSORFLOW_CORE_KERNELS_ZEN_CWISE_OPS_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_ZEN_CWISE_OPS_COMMON_H_

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS
#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "zendnn.hpp"
#include "zendnn_helper.hpp"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

class ZenBinaryOpShared : public OpKernel {
 public:
  explicit ZenBinaryOpShared(OpKernelConstruction *ctx, DataType out,
                             DataType in);

 protected:
  struct ZenBinaryOpState {
    // Sets up bcast with the shape of in0 and in1, ensures that the bcast
    // is valid, and if so, set out, either by allocating a new buffer using
    // ctx->output(...) or by creating an alias for an owned input buffer for
    // in-place computation.
    // Caller must check ctx->status() upon return for non-ok status.
    // If ctx->status().ok() is true, then out is guaranteed to be allocated.
    explicit ZenBinaryOpState(OpKernelContext *ctx, int in_links, int out_links,
                              bool reset);

    const Tensor &in0;
    const Tensor &in1;

    BCast bcast;
    Tensor *out = nullptr;
    int64 out_num_elements;

    int64 in0_num_elements;
    int64 in1_num_elements;

    int ndims;
    bool result;

    int in_links;
    int out_links;
    bool reset;
    bool in0_reuse;
    bool in1_reuse;
  };

  void SetUnimplementedError(OpKernelContext *ctx);
  void SetComputeError(OpKernelContext *ctx);
};

// Coefficient-wise binary operations:
//   Device: E.g., CPUDevice
//   Functor: defined in cwise_ops.h. E.g., functor::add.
template <typename Device, typename Functor>
class ZenBinaryOp : public ZenBinaryOpShared {
 private:
  bool reset;
  int in_links, out_links;

 public:
  typedef typename Functor::in_type Tin;    // Input scalar data type.
  typedef typename Functor::out_type Tout;  // Output scalar data type.

  explicit ZenBinaryOp(OpKernelConstruction *ctx)
      : ZenBinaryOpShared(ctx, DataTypeToEnum<Tout>::v(),
                          DataTypeToEnum<Tin>::v()) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reset", &reset));
  }

  void Compute(OpKernelContext *ctx) override {
    // 'state': Shared helper not dependent on T to reduce code size
    ZenBinaryOpState state(ctx, in_links, out_links, reset);
    auto &bcast = state.bcast;
    const Device &eigen_device = ctx->eigen_device<Device>();
    Tensor *out = state.out;
    if (!bcast.IsValid()) {
      if (ctx->status().ok()) {
        if (state.result) {
          functor::SetOneFunctor<Device, bool>()(eigen_device,
                                                 out->flat<bool>());
        } else {
          functor::SetZeroFunctor<Device, bool>()(eigen_device,
                                                  out->flat<bool>());
        }
      }
      return;
    }

    auto &in0 = state.in0;
    auto &in1 = state.in1;
    if (state.out_num_elements == 0) {
      return;
    }

    const int ndims = state.ndims;
    bool error = false;
    bool *const error_ptr = Functor::has_errors ? &error : nullptr;
    if (ndims <= 1) {
      auto out_flat = out->flat<Tout>();
      if (state.in1_num_elements == 1) {
        // tensor op scalar
        functor::BinaryFunctor<Device, Functor, 1>().Right(
            eigen_device, out_flat, in0.template flat<Tin>(),
            in1.template scalar<Tin>(), error_ptr);
      } else if (state.in0_num_elements == 1) {
        // scalar op tensor
        functor::BinaryFunctor<Device, Functor, 1>().Left(
            eigen_device, out_flat, in0.template scalar<Tin>(),
            in1.template flat<Tin>(), error_ptr);
      } else {
        functor::BinaryFunctor<Device, Functor, 1>()(
            eigen_device, out_flat, in0.template flat<Tin>(),
            in1.template flat<Tin>(), error_ptr);
      }
    } else if (ndims == 2) {
      functor::BinaryFunctor<Device, Functor, 2>().BCast(
          eigen_device, out->shaped<Tout, 2>(bcast.result_shape()),
          in0.template shaped<Tin, 2>(bcast.x_reshape()),
          BCast::ToIndexArray<2>(bcast.x_bcast()),
          in1.template shaped<Tin, 2>(bcast.y_reshape()),
          BCast::ToIndexArray<2>(bcast.y_bcast()), error_ptr);
    } else if (ndims == 3) {
      functor::BinaryFunctor<Device, Functor, 3>().BCast(
          eigen_device, out->shaped<Tout, 3>(bcast.result_shape()),
          in0.template shaped<Tin, 3>(bcast.x_reshape()),
          BCast::ToIndexArray<3>(bcast.x_bcast()),
          in1.template shaped<Tin, 3>(bcast.y_reshape()),
          BCast::ToIndexArray<3>(bcast.y_bcast()), error_ptr);
    } else if (ndims == 4) {
      functor::BinaryFunctor<Device, Functor, 4>().BCast(
          eigen_device, out->shaped<Tout, 4>(bcast.result_shape()),
          in0.template shaped<Tin, 4>(bcast.x_reshape()),
          BCast::ToIndexArray<4>(bcast.x_bcast()),
          in1.template shaped<Tin, 4>(bcast.y_reshape()),
          BCast::ToIndexArray<4>(bcast.y_bcast()), error_ptr);
    } else if (ndims == 5) {
      functor::BinaryFunctor<Device, Functor, 5>().BCast(
          eigen_device, out->shaped<Tout, 5>(bcast.result_shape()),
          in0.template shaped<Tin, 5>(bcast.x_reshape()),
          BCast::ToIndexArray<5>(bcast.x_bcast()),
          in1.template shaped<Tin, 5>(bcast.y_reshape()),
          BCast::ToIndexArray<5>(bcast.y_bcast()), error_ptr);
    } else {
      SetUnimplementedError(ctx);
    }
    if (Functor::has_errors && error) {
      SetComputeError(ctx);
    }

    // If ZenMemPool is enabled then,
    // - Update the buffer use status if input buffer is reused for output
    // - Free the buffers of inputs if they aren't reused for output
    zendnnEnv zenEnvObj = readEnv();
    ZenMemoryPool<float> *zenPoolBuffer = NULL;
    if (zenEnvObj.zenEnableMemPool) {
      unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
      zenPoolBuffer = ZenMemoryPool<float>::getZenMemPool(threadID);
      if (zenPoolBuffer) {
        auto in0_ptr = const_cast<float *>(in0.template flat<float>().data());
        auto in1_ptr = const_cast<float *>(in1.template flat<float>().data());
        if (state.in0_reuse) {
          zenPoolBuffer->zenMemPoolUpdateTensorPtrStatus(ctx, (void *)in0_ptr,
                                                         out_links, reset);
        } else {
          zenPoolBuffer->zenMemPoolFree(ctx, (void *)in0_ptr);
        }
        if (state.in1_reuse) {
          zenPoolBuffer->zenMemPoolUpdateTensorPtrStatus(ctx, (void *)in1_ptr,
                                                         out_links, reset);
        } else {
          zenPoolBuffer->zenMemPoolFree(ctx, (void *)in1_ptr);
        }
      }
    }
  }
};

}  // end namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_ZEN_CWISE_OPS_COMMON_H_
