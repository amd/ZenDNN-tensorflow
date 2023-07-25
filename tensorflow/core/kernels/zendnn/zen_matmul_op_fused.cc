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

#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/fused_eigen_output_kernels.h"
#include "tensorflow/core/kernels/zendnn/zen_matmul_ops_util.h"
#include "tensorflow/core/util/tensor_format.h"
#include "zendnn.hpp"

using namespace zendnn;
using namespace std;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
struct LaunchZenFusedMatMulOp {
  void operator()(
      OpKernelContext *context, const Tensor &a, const Tensor &b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> &dim_pair,
      FusedComputationType fusion, const FusedComputationArgs &fusion_args,
      Tensor *output);
};

template <typename T>
struct LaunchZenFusedMatMulOp<CPUDevice, T> {
  void operator()(
      OpKernelContext *context, const Tensor &a, const Tensor &b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> &dim_pair,
      FusedComputationType fusion, const FusedComputationArgs &fusion_args,
      Tensor *output) {
    BiasAddArgs<T> bias_add_args;
    if (BiasAddArgs<T>::IsSupported(fusion)) {
      OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add_args));
    }

    const int m = a.dim_size(1 - dim_pair[0].first);
    const int k = a.dim_size(dim_pair[0].first);
    const int n = b.dim_size(1 - dim_pair[0].second);
    bool transpose_a = dim_pair[0].first == 0;
    bool transpose_b = dim_pair[0].second == 1;
    bool isBiasAdd = true;

    auto a_ptr = const_cast<T *>(a.template flat<T>().data());
    auto b_ptr = const_cast<T *>(b.template flat<T>().data());
    auto c_ptr = (output->template flat<T>().data());
    auto bias_ptr = const_cast<T *>(bias_add_args.bias_add_data);

    // dimensions of matmul source, weights, bias and destination tensors
    memory::dims src_dims = {m, k};
    memory::dims weight_dims = {k, n};
    memory::dims bias_dims = {1, n};
    memory::dims dst_dims = {m, n};
    memory::format_tag src_format = memory::format_tag::nc;
    memory::format_tag weight_format =
        transpose_b ? memory::format_tag::io : memory::format_tag::oi;

    ZenMatMulParams matmul_params(src_dims, weight_dims, bias_dims, dst_dims,
                                  src_format, weight_format, isBiasAdd);

    switch (fusion) {
      case FusedComputationType::kBiasAdd: {
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 0);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr, isBiasAdd);
        break;
      }
      case FusedComputationType::kBiasAddWithAdd: {
        matmul_params.post_op_params.push_back({"sum", {1.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 1);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr, isBiasAdd);
        break;
      }
      case FusedComputationType::kBiasAddWithRelu: {
        matmul_params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 0);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr, isBiasAdd);
        break;
      }
      case FusedComputationType::kBiasAddWithAddAndRelu: {
        matmul_params.post_op_params.push_back({"sum", {1.0}});
        matmul_params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 1);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr, isBiasAdd);
        break;
      }
      case FusedComputationType::kBiasAddWithGeluApproximate: {
        matmul_params.post_op_params.push_back(
            {"GeluApproximate", {1.0, 1.0, 0.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 1);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr, isBiasAdd);
        break;
      }
      case FusedComputationType::kBiasAddWithGeluExact: {
        matmul_params.post_op_params.push_back({"GeluExact", {1.0, 1.0, 0.0}});
        ZenMatMulPrimitive<T, T, T, T> *matmul_prim =
            ZenMatMulPrimitiveFactory<T, T, T, T>::Get(matmul_params, 1);
        matmul_prim->Execute(a_ptr, b_ptr, bias_ptr, c_ptr, isBiasAdd);
        break;
      }
      case FusedComputationType::kBiasAddWithRelu6:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
        break;
      case FusedComputationType::kBiasAddWithElu:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
        break;
      case FusedComputationType::kUndefined:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type is undefined"));
        break;
      default:
        OP_REQUIRES_OK(context,
                       errors::Internal("Fusion type is not supported"));
    }
  }
};

template <typename Device, typename T>
class ZenFusedMatMulOp : public OpKernel {
 public:
  explicit ZenFusedMatMulOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));
    OP_REQUIRES_OK(context, context->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(context, context->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(context, context->GetAttr("reset", &reset));

    std::vector<FusedComputationPattern> patterns;

    using FCT = FusedComputationType;
    if (std::is_same<Device, CPUDevice>::value) {
      patterns = {
          {FCT::kBiasAdd, {"BiasAdd"}},
          {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
          {FCT::kBiasAddWithRelu6, {"BiasAdd", "Relu6"}},
          {FCT::kBiasAddWithAdd, {"BiasAdd", "Add"}},
          {FCT::kBiasAddWithAddAndRelu, {"BiasAdd", "Add", "Relu"}},
          {FCT::kBiasAddWithElu, {"BiasAdd", "Elu"}},
          {FCT::kBiasAddWithGeluApproximate, {"BiasAdd", "GeluApproximate"}},
          {FCT::kBiasAddWithGeluExact, {"BiasAdd", "GeluExact"}}};
    }

    OP_REQUIRES_OK(context, InitializeFusedComputation(
                                context, "_ZenMatMul", patterns,
                                &fused_computation_, &fused_computation_args_));
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
    bool is_float = std::is_same<T, float>::value;
    zenTensorType out_type =
        (is_float) ? zenTensorType::FLOAT : zenTensorType::BFLOAT16;

    zendnnEnv zenEnvObj = readEnv();
    Tensor *out = nullptr;
    int zenEnableMemPool = zenEnvObj.zenEnableMemPool &&
                           (ctx->expected_output_dtype(0) == DT_FLOAT ||
                            ctx->expected_output_dtype(0) == DT_BFLOAT16);
    ZenMemoryPool<T> *zenPoolBuffer = NULL;

    if ((fused_computation_ == FusedComputationType::kBiasAddWithAdd) ||
        (fused_computation_ == FusedComputationType::kBiasAddWithAddAndRelu)) {
      const Tensor &add_tensor = ctx->input(3);
      ctx->set_output(0, add_tensor);
      out = ctx->mutable_output(0);
      if (zenEnableMemPool) {
        unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
        zenPoolBuffer = ZenMemoryPool<T>::getZenMemPool(threadID);
        if (zenPoolBuffer) {
          const T *output_array = const_cast<T *>(out->flat<T>().data());
          zenPoolBuffer->zenMemPoolUpdateTensorPtrStatus(ctx, (T *)output_array,
                                                         out_links, reset);
        }
      }
    } else {
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
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), out->flat<T>());
      return;
    }

    auto launch = LaunchZenFusedMatMulOp<Device, T>();
    launch(ctx, a, b, dim_pair, fused_computation_, fused_computation_args_,
           out);

    // If ZenMemPool Optimization is enabled(default), update the state of
    //  Memory pool based on input_array address
    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      auto a_ptr = const_cast<T *>(a.template flat<T>().data());
      auto b_ptr = const_cast<T *>(b.template flat<T>().data());
      zenPoolBuffer->zenMemPoolFree(ctx, a_ptr);
      zenPoolBuffer->zenMemPoolFree(ctx, b_ptr);
    }
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  bool reset;
  int in_links, out_links;

  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;

  TF_DISALLOW_COPY_AND_ASSIGN(ZenFusedMatMulOp);
};

// Registration of the CPU implementations.
#define REGISTER_ZEN_FUSED_CPU_MATMUL(T)                                 \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_ZenFusedMatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ZenFusedMatMulOp<CPUDevice, T>);

TF_CALL_float(REGISTER_ZEN_FUSED_CPU_MATMUL);
TF_CALL_bfloat16(REGISTER_ZEN_FUSED_CPU_MATMUL);

#undef REGISTER_FUSED_CPU_MATMUL

}  // namespace tensorflow
