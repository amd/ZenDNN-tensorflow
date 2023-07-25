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

#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/zendnn/zen_matmul_ops_util.h"
#include "zendnn.hpp"
#include "zendnn_helper.hpp"

using namespace zendnn;
using namespace std;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T, bool isBiasAddGelu = false>
class ZenMatMulOp : public OpKernel {
 public:
  explicit ZenMatMulOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reset", &reset));
  }

  void Compute(OpKernelContext *ctx) override {
    const Tensor &a = ctx->input(0);
    const Tensor &b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(a.shape()),
        errors::InvalidArgument("In[0] is not a matrix. Instead it has shape ",
                                a.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(b.shape()),
        errors::InvalidArgument("In[1] is not a matrix. Instead it has shape ",
                                b.shape().DebugString()));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    OP_REQUIRES(
        ctx, a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", a.shape().DebugString(),
            ", In[1]: ", b.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    // Update the output type
    zenTensorType out_type = zenTensorType::FLOAT;

    zendnnEnv zenEnvObj = readEnv();
    Tensor *out = nullptr;
    int zenEnableMemPool =
        zenEnvObj.zenEnableMemPool && ctx->expected_output_dtype(0) == DT_FLOAT;
    ZenMemoryPool *zenPoolBuffer = NULL;

    // ZenMemPool Optimization reuse o/p tensors from the pool. By default
    //  its enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory
    //  pool optimization
    //  Cases where tensors in pool are not free or requested size is more
    //  than available tensor size in Pool, control will fall back to
    //  default way of allocation i.e. with allocate_output(..)
    if (zenEnableMemPool) {
      unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
      zenPoolBuffer = ZenMemoryPool::getZenMemPool(threadID);
      if (zenPoolBuffer) {
        int status = zenPoolBuffer->acquireZenPoolTensor(
            ctx, &out, out_shape, out_links, reset, out_type);
        if (status) {
          zenEnableMemPool = false;
        }
      } else {
        zenEnableMemPool = false;
      }
    }
    if (!zenEnableMemPool) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    }

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a.NumElements() == 0 && b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      // functor::SetZeroFunctor<Device, T> f;
      // f(ctx->eigen_device<Device>(), out->flat<T>());
      return;
    }

    const int m = a.dim_size(1 - dim_pair[0].first);
    const int k = a.dim_size(dim_pair[0].first);
    const int n = b.dim_size(1 - dim_pair[0].second);
    bool transpose_a = dim_pair[0].first == 0;
    bool transpose_b = dim_pair[0].second == 1;

    auto a_ptr = const_cast<float *>(a.template flat<T>().data());
    auto b_ptr = const_cast<float *>(b.template flat<T>().data());
    auto c_ptr = (out->template flat<T>().data());

    // dimensions of matmul source, weights, bias and destination tensors
    memory::dims src_dims = {m, k};
    memory::dims weight_dims = {n, k};
    memory::dims bias_dims = {n};
    memory::dims dst_dims = {m, n};
    memory::format_tag src_format = memory::format_tag::nc;
    memory::format_tag weight_format =
        transpose_b ? memory::format_tag::oi : memory::format_tag::io;

    ZenMatMulParams matmul_params(src_dims, weight_dims, bias_dims, dst_dims,
                                  src_format, weight_format);

    ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
        ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 0);

    if (isBiasAddGelu) {
      const Tensor &bias = ctx->input(2);
      auto bias_ptr = const_cast<float *>(bias.template flat<T>().data());
      ZenMatMulParams matmul_params(src_dims, weight_dims, bias_dims, dst_dims,
                                    src_format, weight_format);
      matmul_params.post_op_params.push_back({"gelu", {1.0, 0.0, 0.0}});
      ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
          ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 1);
      matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr);
    } else {
      matmul_prim->Execute(a_ptr, b_ptr, NULL, c_ptr);
    }

    // If ZenMemPool Optimization is enabled(default), update the state of
    //  Memory pool based on input_array address
    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      zenPoolBuffer->zenMemPoolFree(ctx, a_ptr);
      zenPoolBuffer->zenMemPoolFree(ctx, b_ptr);
    }
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  int in_links, out_links;
  bool reset;
};

REGISTER_KERNEL_BUILDER(
    Name("ZenMatMul").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    ZenMatMulOp<float, false>);

REGISTER_KERNEL_BUILDER(
    Name("_ZenMatMulBiasAddGelu").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    ZenMatMulOp<float, true>);

REGISTER_KERNEL_BUILDER(
    Name("MatMulBiasAddGelu").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    ZenMatMulOp<float, true>);

}  // namespace tensorflow
