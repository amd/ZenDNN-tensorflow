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

#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/kernels/zendnn/zen_quantized_ops.h"

namespace tensorflow {

using namespace zendnn;
using dt = zendnn::memory::data_type;
using tag = zendnn::memory::format_tag;
template <typename Tinput, typename Toutput, bool is_dequantize>
class ZenVitisAIQuantizeOp : public OpKernel {
 public:
  explicit ZenVitisAIQuantizeOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("scale", &scale));
    OP_REQUIRES_OK(context, context->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(context, context->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(context, context->GetAttr("reset", &reset));
  }
  void Compute(OpKernelContext *ctx) override {
    const Tensor &input = ctx->input(0);
    Tinput *input_array = static_cast<Tinput *>(
        const_cast<Tinput *>(input.flat<Tinput>().data()));

    ZenExecutor *ex = ex->getInstance();
    zendnn::engine eng = ex->getEngine();
    zendnn::stream s = ex->getStream();
    TensorShape out_shape = input.shape();
    const int ndims = input.dims();
    memory::dims src_dims;
    if (ndims == 2) {
      src_dims = {input.dim_size(0), input.dim_size(1)};
    } else {
      src_dims = {input.dim_size(0), input.dim_size(3), input.dim_size(1),
                  input.dim_size(2)};
    }

    zendnnEnv zenEnvObj = readEnv();
    int zenPoolEnable = zenEnvObj.zenEnableMemPool &&
                        (ctx->expected_output_dtype(0) == DT_QINT8 ||
                         ctx->expected_output_dtype(0) == DT_FLOAT);
    // Update the output type
    auto out_type =
        std::is_same<Toutput, qint8>::value
            ? zenTensorType::QINT8 : zenTensorType::FLOAT;
    ZenMemoryPool<Toutput> *zenPoolBuffer = nullptr;
    Tensor *output = nullptr;

    if (zenPoolEnable) {
      unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
      zenPoolBuffer = ZenMemoryPool<Toutput>::getZenMemPool(threadID);
      if (zenPoolBuffer) {
        int status = zenPoolBuffer->acquireZenPoolTensor(
            ctx, &output, out_shape, out_links, reset, out_type);
        if (status) {
          zenPoolEnable = false;
        }
      } else {
        zenPoolEnable = false;
      }
    }
    if (!zenPoolEnable) {
      // Outtype is not required for default allocation because context
      // maintains allocation data Type for outputs
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
    }
    Toutput *output_array;
    if (ndims == 2) {
      auto output_map = output->tensor<Toutput, 2>();
      output_array = const_cast<Toutput *>(output_map.data());
    } else {
      auto output_map = output->tensor<Toutput, 4>();
      output_array = const_cast<Toutput *>(output_map.data());
    }
    memory::format_tag src_format;
    if (ndims == 2) {
      src_format = memory::format_tag::nc;
    } else {
      src_format = memory::format_tag::acdb;
    }
    if (is_dequantize) {
      // for dequant scale should be -ve
      scale = -scale;
      zendnnInfo(ZENDNN_FWKLOG, "ZenVitisAIDequantize in compute");
    } else {
      zendnnInfo(ZENDNN_FWKLOG, "ZenVitisAIQuantize in compute");
    }
    zendnn::memory user_src_memory;
    zendnn::memory dst_memory;
    user_src_memory = memory({{src_dims}, DataTypetoZen<Tinput>(), src_format},
                             eng, input_array);
    dst_memory = memory({{src_dims}, DataTypetoZen<Toutput>(), src_format}, eng,
                        output_array);
    zp.AddReorder(user_src_memory, dst_memory, {scale});
    zp.Execute(s);
    zp.reset();
    // If ZenMemPool Optimization is enabled(default), update the state of
    //  Memory pool based on input_array address
    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      zenPoolBuffer->zenMemPoolFree(ctx, (void *)input_array);
    }
  }

 private:
  int scale;
  ZenPrimitives zp;
  TF_DISALLOW_COPY_AND_ASSIGN(ZenVitisAIQuantizeOp);
  bool reset;
  int in_links, out_links;
};

REGISTER_KERNEL_BUILDER(Name("VitisAIQuantize").Device(DEVICE_CPU), NoOp);
REGISTER_KERNEL_BUILDER(Name("VitisAIDequantize").Device(DEVICE_CPU), NoOp);
REGISTER_KERNEL_BUILDER(Name("_ZenVitisAIQuantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Tinput")
                            .TypeConstraint<qint8>("Toutput"),
                        ZenVitisAIQuantizeOp<float, qint8, false>);

REGISTER_KERNEL_BUILDER(Name("_ZenVitisAIQuantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("Tinput")
                            .TypeConstraint<quint8>("Toutput"),
                        ZenVitisAIQuantizeOp<float, quint8, false>);

REGISTER_KERNEL_BUILDER(Name("_ZenVitisAIDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("Tinput")
                            .TypeConstraint<float>("Toutput"),
                        ZenVitisAIQuantizeOp<qint8, float, true>);

REGISTER_KERNEL_BUILDER(Name("_ZenVitisAIDequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<float>("Toutput"),
                        ZenVitisAIQuantizeOp<quint8, float, true>);
}  // namespace tensorflow
