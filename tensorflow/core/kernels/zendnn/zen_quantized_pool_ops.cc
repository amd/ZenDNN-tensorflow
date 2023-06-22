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

#include <numeric>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"
#include "tensorflow/core/kernels/zendnn/zen_quantized_ops.h"

namespace tensorflow {

using tag = memory::format_tag;
using dt = memory::data_type;

/**
 * @brief ZenVitisAIPoolOp implements all pooling kernels:
 * ZenVitisAIAvgPool and ZenVitisAIMaxPool
 *
 * @tparam Tinput Input dtype
 * @tparam Toutput Output dtype
 * @tparam pooling_algo AvgPool or MaxPool
 */
template <typename Tinput, typename Toutput, zendnn::algorithm pooling_algo>
class ZenVitisAIPoolOp : public OpKernel {
 public:
  explicit ZenVitisAIPoolOp(OpKernelConstruction *context) : OpKernel(context) {
    string data_format;
    ReadParameterFromContext<string>(context, "data_format", &data_format);
    ReadParameterFromContext<std::vector<int32>>(context, "ksize", &ksize_);
    ReadParameterFromContext<std::vector<int32>>(context, "strides", &stride_);
    ReadParameterFromContext<Padding>(context, "padding", &padding_);

    // ZenDNN related params
    InitZendnnParameters(context, &zendnn_params_);

    // Only NHWC is allowed for VitisAI Ops
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(
        context, data_format_ == FORMAT_NHWC,
        errors::Unimplemented("ZenVitisAIPoolOp only supports NHWC Format"));

    // Only 4D (NHWC) pooling is allowed for VitisAI Ops for now
    OP_REQUIRES(
        context, ksize_.size() == 4,
        errors::InvalidArgument("Kernel size field must specify 4 dimensions"));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument(
                    "Sliding window stride field must specify 4 dimensions"));
  }

  void Compute(OpKernelContext *context) override {
    ZenExecutor *ex = ex->getInstance();
    zendnn::engine engine = ex->getEngine();
    zendnn::stream stream = ex->getStream();

    const Tensor &input = ReadInputFromContext(context, 0);
    Tinput *input_array = static_cast<Tinput *>(
        const_cast<Tinput *>(input.flat<Tinput>().data()));

    // Calcuate the dimensions
    PoolParameters params{
        context,
        ksize_,
        stride_,
        padding_,
        /*explicit_paddings=*/{},
        FORMAT_NHWC,
        input.shape(),
    };
    TensorShape out_shape;
    OP_REQUIRES_OK(context,
		  params.forward_output_shape(&out_shape));
    Tensor *output = nullptr;

    // Update the output type
    zenTensorType out_type = zenTensorType::FLOAT;
    if (std::is_same<Toutput, quint8>::value) {
      out_type = zenTensorType::QUINT8;
    } else if (std::is_same<Toutput, qint8>::value) {
      out_type = zenTensorType::QINT8;
    }
    // Allocate output
    zendnnEnv zenEnvObj = readEnv();
    int zenPoolEnable = zenEnvObj.zenEnableMemPool &&
                        (context->expected_output_dtype(0) == DT_QINT8 ||
                         context->expected_output_dtype(0) == DT_QUINT8 ||
                         context->expected_output_dtype(0) == DT_FLOAT);
    ZenMemoryPool<Toutput> *zenPoolBuffer=nullptr;

    if (zenPoolEnable) {
      unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
      zenPoolBuffer = ZenMemoryPool<Toutput>::getZenMemPool(threadID);
      if (zenPoolBuffer) {
        int status = zenPoolBuffer->acquireZenPoolTensor(
            context, &output, out_shape, zendnn_params_.out_links,
            zendnn_params_.reset, out_type);
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
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    }

    auto output_map = output->tensor<Toutput, 4>();
    Toutput *output_array = const_cast<Toutput *>(output_map.data());

    // Define the sizes for the primitive
    memory::dims pool_src_tz = {params.tensor_in_batch, params.depth,
                                params.tensor_in_rows, params.tensor_in_cols};
    memory::dims pool_dst_tz = {params.tensor_in_batch, params.depth,
                                params.out_height, params.out_width};
    memory::dims pool_kernel = {params.window_rows, params.window_cols};
    memory::dims pool_strides = {params.row_stride, params.col_stride};
    memory::dims pool_padding_l = {params.pad_top, params.pad_left};
    memory::dims pool_padding_r = {params.pad_bottom, params.pad_right};

    // Create pooling primitive
    memory::desc pool_src_md =
        memory::desc({pool_src_tz}, DataTypetoZen<Tinput>(), tag::acdb);
    memory::desc pool_dst_md =
        memory::desc({pool_dst_tz}, DataTypetoZen<Tinput>(), tag::acdb);
    pooling_forward::desc pool_desc = pooling_forward::desc(
        prop_kind::forward_inference, pooling_algo, pool_src_md, pool_dst_md,
        pool_strides, pool_kernel, pool_padding_l, pool_padding_r);
    pooling_forward::primitive_desc pool_pd =
        pooling_forward::primitive_desc(pool_desc, engine);

    // Create src and dst memory
    zendnn::memory pool_src_memory, pool_dst_temp_output, pool_dst_memory;
    pool_src_memory =
        memory({{pool_src_tz}, DataTypetoZen<Tinput>(), tag::acdb}, engine,
               input_array);
    if (!std::is_same<Toutput, float>::value) {
      pool_dst_temp_output =
          memory({{pool_dst_tz}, DataTypetoZen<Toutput>(), tag::acdb}, engine,
                 output_array);
    } else {
      pool_dst_temp_output =
          memory({{pool_dst_tz}, DataTypetoZen<Tinput>(), tag::acdb}, engine);
      pool_dst_memory =
          memory({{pool_dst_tz}, DataTypetoZen<Toutput>(), tag::acdb}, engine,
                 output_array);
    }

    // Create and add pool primitives
    auto pool_prim = pooling_forward(pool_pd);
    zp.AddPrimitive(pool_prim, {{ZENDNN_ARG_SRC, pool_src_memory},
                                {ZENDNN_ARG_DST, pool_dst_temp_output}});

    if (std::is_same<Toutput, float>::value) {
      zp.AddReorder(pool_dst_temp_output, pool_dst_memory);
    }

    // Execute all the added primitives
    zp.Execute(stream);

    // Reset any primitives added yet
    zp.reset();

    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      zenPoolBuffer->zenMemPoolFree(context, (void *)input_array);
    }
  }

 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;

  ZenPrimitives zp;
  ZendnnParameters zendnn_params_;
};

// clang-format off

REGISTER_KERNEL_BUILDER(Name("VitisAIAvgPool").Device(DEVICE_CPU), NoOp);
REGISTER_KERNEL_BUILDER(Name("VitisAIMaxPool").Device(DEVICE_CPU), NoOp);

// All the required ZenVitisAIAvgPool kernel combinations
#define REGISTER_VITISAI_AVGERAGE_POOL(Tinput, Toutput)            \
  REGISTER_KERNEL_BUILDER(Name("ZenVitisAIAvgPool")                \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<Tinput>("Tinput")    \
                              .TypeConstraint<Toutput>("Toutput"), \
                              ZenVitisAIPoolOp<Tinput, Toutput, zendnn::algorithm::pooling_avg>)

REGISTER_VITISAI_AVGERAGE_POOL(qint8, qint8);
REGISTER_VITISAI_AVGERAGE_POOL(quint8, quint8);
REGISTER_VITISAI_AVGERAGE_POOL(quint8, float);

#undef REGISTER_VITISAI_AVGERAGE_POOL

// All the required ZenVitisAIMaxPool kernel combinations
#define REGISTER_VITISAI_MAX_POOL(Tinput, Toutput)                 \
  REGISTER_KERNEL_BUILDER(Name("ZenVitisAIMaxPool")                \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<Tinput>("Tinput")    \
                              .TypeConstraint<Toutput>("Toutput"), \
                              ZenVitisAIPoolOp<Tinput, Toutput, zendnn::algorithm::pooling_max>)

REGISTER_VITISAI_MAX_POOL(qint8, qint8);
REGISTER_VITISAI_MAX_POOL(quint8, quint8);

#undef REGISTER_VITISAI_MAX_POOL

// clang-format on

}  // namespace tensorflow
