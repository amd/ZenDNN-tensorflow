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
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/kernels/zendnn/zen_quantized_ops.h"
#include "tensorflow/core/util/tensor_format.h"
#include "zendnn.hpp"

namespace tensorflow {
using namespace zendnn;
using dt = zendnn::memory::data_type;
using tag = zendnn::memory::format_tag;
typedef Eigen::ThreadPoolDevice CPUDevice;
template <typename Tinput, typename Tfilter, typename Toutput>
class ZenVitisAIMatMulop : public OpKernel {
 public:
  explicit ZenVitisAIMatMulop(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));
    OP_REQUIRES_OK(context, context->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(context, context->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(context, context->GetAttr("reset", &reset));
    OP_REQUIRES_OK(context, context->GetAttr("in_scale", &input_scale));
    OP_REQUIRES_OK(context, context->GetAttr("weight_scale", &filter_scale));
    OP_REQUIRES_OK(context, context->GetAttr("out_scale", &output_scale));
  }
  void Compute(OpKernelContext *ctx) override {
    const Tensor &a = ctx->input(0);
    const Tensor &b = ctx->input(1);
    ZenExecutor *ex = ex->getInstance();
    zendnn::engine eng = ex->getEngine();
    zendnn::stream s = ex->getStream();
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
    auto out_type = zenTensorType::FLOAT;

    zendnnEnv zenEnvObj = readEnv();
    Tensor *out = nullptr;
    int zenEnableMemPool = zenEnvObj.zenEnableMemPool &&
                           (ctx->expected_output_dtype(0) == DT_FLOAT);
    ZenMemoryPool<Toutput> *zenPoolBuffer = NULL;
    // ZenMemPool Optimization reuse o/p tensors from the pool. By default
    //  its enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory
    //  pool optimization
    //  Cases where tensors in pool are not free or requested size is more
    //  than available tensor size in Pool, control will fall back to
    //  default way of allocation i.e. with allocate_output(..)
    if (zenEnableMemPool) {
      unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
      zenPoolBuffer = ZenMemoryPool<Toutput>::getZenMemPool(threadID);
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
    zendnnInfo(ZENDNN_FWKLOG, "ZenVitisAIMatMulop in compute");
    const int m = a.dim_size(1 - dim_pair[0].first);
    const int k = a.dim_size(dim_pair[0].first);
    const int n = b.dim_size(1 - dim_pair[0].second);
    bool transpose_a = dim_pair[0].first == 0;
    bool transpose_b = dim_pair[0].second == 1;
    Tinput *a_ptr = static_cast<Tinput *>(
        const_cast<Tinput *>(a.template flat<Tinput>().data()));
    float *b_ptr = static_cast<float *>(
        const_cast<float *>(b.template flat<float>().data()));
    auto output_map = out->tensor<Toutput, 2>();
    Toutput *c_ptr = const_cast<Toutput *>(output_map.data());

    // dimensions of matmul source, weights, bias and destination tensors
    memory::dims src_dims = {m, k};
    memory::dims weight_dims = {k, n};
    // memory::dims bias_dims = {n};
    memory::dims dst_dims = {m, n};
    memory::format_tag src_format = memory::format_tag::nc;
    memory::format_tag weight_format =
        transpose_b ? memory::format_tag::ba : memory::format_tag::ab;
    zendnn::memory user_src_memory;
    zendnn::memory matmul_dst_memory;
    user_src_memory =
        memory({{src_dims}, DataTypetoZen<Tinput>(), src_format}, eng, a_ptr);
    matmul_dst_memory =
        memory({{dst_dims}, DataTypetoZen<Toutput>(), tag::nc}, eng, c_ptr);

    memory::desc src_memory_desc =
        memory::desc({src_dims}, DataTypetoZen<Tinput>(), src_format);
    memory::desc filter_memory_desc = memory::desc(
        {weight_dims}, DataTypetoZen<Tfilter>(), memory::format_tag::any);
    memory::desc dst_memory_desc =
        memory::desc({dst_dims}, DataTypetoZen<Toutput>(), tag::nc);
    int dequantize_scale = -input_scale - filter_scale;
    std::vector<float> matmul_output_scales(1);
    matmul_output_scales[0] = std::pow(2, dequantize_scale);
    primitive_attr matmul_attr;
    matmul_attr.set_output_scales(0, matmul_output_scales);
    matmul::desc matmul_d =
        matmul::desc(src_memory_desc, filter_memory_desc, dst_memory_desc);
    matmul::primitive_desc matmul_pd =
        matmul::primitive_desc(matmul_d, matmul_attr, eng);
    zendnn::memory matmul_filter_memory;
    zendnn::memory matmul_filter_reordered_memory;
    if (!cached_filter.tensorvalue()) {
      matmul_filter_memory =
          memory({{weight_dims}, dt::f32, weight_format}, eng, b_ptr);
      matmul_filter_reordered_memory = memory(matmul_pd.weights_desc(), eng);
      zp.AddReorder(matmul_filter_memory, matmul_filter_reordered_memory,
                    {filter_scale});
    } else {
      Tfilter *filter_data = cached_filter.GetTensorHandle();
      matmul_filter_reordered_memory =
          memory(matmul_pd.weights_desc(), eng, filter_data);
    }
    std::unordered_map<int, memory> matmul_prim_args;
    matmul_prim_args.insert({ZENDNN_ARG_SRC, user_src_memory});  // input
    matmul_prim_args.insert(
        {ZENDNN_ARG_WEIGHTS, matmul_filter_reordered_memory});     // filter
    matmul_prim_args.insert({ZENDNN_ARG_DST, matmul_dst_memory});  // output

    auto matmul_prim = matmul(matmul_pd);
    zp.AddPrimitive(matmul_prim, matmul_prim_args);
    // Execute all the added primitives
    zp.Execute(s);

    // Reset any primitives added yet
    zp.reset();
    cached_filter.SetTensorHandle(ctx, matmul_filter_reordered_memory);
    // If ZenMemPool Optimization is enabled(default), update the state of
    //  Memory pool based on input_array address
    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      zenPoolBuffer->zenMemPoolFree(ctx, (void *)a_ptr);
      zenPoolBuffer->zenMemPoolFree(ctx, (void *)b_ptr);
    }
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  bool reset;
  int in_links, out_links;
  int input_scale;
  int filter_scale;
  int output_scale;
  ZenPersistentTensor<Tfilter> cached_filter;
  ZenPrimitives zp;
  TF_DISALLOW_COPY_AND_ASSIGN(ZenVitisAIMatMulop);
};
REGISTER_KERNEL_BUILDER(Name("VitisAIMatMul").Device(DEVICE_CPU), NoOp);
REGISTER_KERNEL_BUILDER(Name("_ZenVitisAIMatMul")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("Tinput")
                            .TypeConstraint<float>("Tfilter")
                            .TypeConstraint<float>("Toutput"),
                        ZenVitisAIMatMulop<qint8, qint8, float>);

REGISTER_KERNEL_BUILDER(Name("_ZenVitisAIMatMul")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<float>("Tfilter")
                            .TypeConstraint<float>("Toutput"),
                        ZenVitisAIMatMulop<quint8, quint8, float>);
}  // namespace tensorflow
