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

#include "tensorflow/core/kernels/zendnn/zen_quantized_conv_ops.h"

#include "tensorflow/core/kernels/no_op.h"

namespace tensorflow {

using dt = zendnn::memory::data_type;
using tag = zendnn::memory::format_tag;

#define TF_REQUIRES(EXP, STATUS)                \
  do {                                          \
    if (!TF_PREDICT_TRUE(EXP)) return (STATUS); \
  } while (false)

Status InitVitisAIConv2DParameters(OpKernelConstruction *context,
                                   VitisAIConv2DParameters *params) {
  // Initialize and copy standard conv2d params
  Conv2DParameters conv2d_params_;
  TF_RETURN_IF_ERROR(InitConv2DParameters(context, &conv2d_params_));
  params->dilations = conv2d_params_.dilations;
  params->strides = conv2d_params_.strides;
  params->padding = conv2d_params_.padding;
  params->data_format = conv2d_params_.data_format;
  params->explicit_paddings = conv2d_params_.explicit_paddings;
  TF_REQUIRES(
      params->data_format == FORMAT_NHWC,
      errors::Unimplemented("ZenVitisAIConv2DOp only supports NHWC Format"));

  // Initialize vitisai conv2d specific params. fp32 conv does not have scales
  ReadParameterFromContextIfAvailable<int>(context, "in_scale",
                                           &params->input_scale);
  ReadParameterFromContextIfAvailable<int>(context, "weight_scale",
                                           &params->filter_scale);
  ReadParameterFromContextIfAvailable<int>(context, "out_scale",
                                           &params->output_scale);
  ReadParameterFromContextIfAvailable<int>(context, "sum_scale",
                                           &params->sum_scale);
  ReadParameterFromContextIfAvailable<int>(context, "add_out_scale",
                                           &params->sum_out_scale);
  ReadParameterFromContextIfAvailable<int>(context, "intermediate_float_scale",
                                           &params->intermediate_float_scale);
  params->bias_scale = params->input_scale + params->filter_scale;

  ReadParameterFromContext<bool>(context, "is_relu", &params->is_relu);
  ReadParameterFromContextIfAvailable<float>(context, "relu_alpha",
                                             &params->relu_alpha);

  // Initialize vitisai conv2d with depthwise conv2d fused
  auto &dw_params = params->dw_params;
  ReadParameterFromContextIfAvailable<std::vector<int32>>(
      context, "dw_strides", &params->dw_params.strides);
  ReadParameterFromContextIfAvailable<Padding>(context, "dw_padding",
                                               &dw_params.padding);
  ReadParameterFromContextIfAvailable<int>(context, "dw_in_scale",
                                           &dw_params.input_scale);
  ReadParameterFromContextIfAvailable<int>(context, "dw_weight_scale",
                                           &dw_params.filter_scale);
  ReadParameterFromContextIfAvailable<int>(context, "dw_out_scale",
                                           &dw_params.output_scale);
  ReadParameterFromContextIfAvailable<bool>(context, "dw_is_relu",
                                            &dw_params.is_relu);
  ReadParameterFromContextIfAvailable<float>(context, "dw_relu_alpha",
                                             &dw_params.relu_alpha);
  dw_params.bias_scale = dw_params.input_scale + dw_params.filter_scale;

  return OkStatus();
}

Status ComputeVitisAIConv2DDimensions(VitisAIConv2DDimensions *dimensions,
                                      const VitisAIConv2DParameters &params,
                                      const Tensor &input, const Tensor &filter,
                                      bool is_depthwise) {
  // Check that 2D convolution input and filter have exactly 4 dimensions.
  // and that the dims are within int range
  TF_REQUIRES(input.dims() == 4,
              errors::InvalidArgument("input must be 4-dimensional",
                                      input.shape().DebugString()));
  TF_REQUIRES(filter.dims() == 4,
              errors::InvalidArgument("filter must be 4-dimensional: ",
                                      filter.shape().DebugString()));
  for (int i = 0; i < 3; i++) {
    TF_REQUIRES(
        FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
        errors::InvalidArgument("filter size too large"));
  }

  const auto ReadValidateInputDim = [](const Tensor &input, const char dim,
                                       const string error_msg) -> int {
    int64 val_raw = GetTensorDim(input, FORMAT_NHWC, dim);
    return static_cast<int>(val_raw);
  };

  // Get the input dims
  int batch = ReadValidateInputDim(input, 'N', "Input batch too large");
  int input_rows = ReadValidateInputDim(input, 'H', "Input rows too large");
  int input_cols = ReadValidateInputDim(input, 'W', "Input cols too large");
  int in_depth = ReadValidateInputDim(input, 'C', "Input depth too large");

  const auto ReadValidateFilterDim = [](const Tensor &filter,
                                        const int dim) -> int {
    int64 val_raw = filter.dim_size(dim);
    return static_cast<int>(val_raw);
  };

  // Get the filter dims
  int filter_rows = ReadValidateFilterDim(filter, 0);
  int filter_cols = ReadValidateFilterDim(filter, 1);
  int patch_depth = ReadValidateFilterDim(filter, 2);
  int out_depth = ReadValidateFilterDim(filter, 3);
  if (is_depthwise) out_depth = ReadValidateFilterDim(filter, 2);

  // Get the stride and dilation dims
  TF_REQUIRES(params.strides.size() == 4,
              errors::InvalidArgument("Sliding window strides field must "
                                      "specify 4 dimensions"));
  // Stride and dilation is only supported for H and W dims for now
  int stride_rows = GetTensorDim(params.strides, FORMAT_NHWC, 'H');
  int stride_cols = GetTensorDim(params.strides, FORMAT_NHWC, 'W');
  int dilation_rows = GetTensorDim(params.dilations, FORMAT_NHWC, 'H');
  int dilation_cols = GetTensorDim(params.dilations, FORMAT_NHWC, 'W');

  // If padding was explicit, read them from the attributes
  int64_t pad_rows_before, pad_rows_after, pad_cols_before, pad_cols_after;
  if (params.padding == Padding::EXPLICIT) {
    GetExplicitPaddingForDim(params.explicit_paddings, FORMAT_NHWC, 'H',
                             &pad_rows_before, &pad_rows_after);
    GetExplicitPaddingForDim(params.explicit_paddings, FORMAT_NHWC, 'W',
                             &pad_cols_before, &pad_cols_after);
  }

  // Compute output dims with the specified params
  int64_t out_rows = 0, out_cols = 0;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
      input_rows, filter_rows, dilation_rows, stride_rows, params.padding,
      &out_rows, &pad_rows_before, &pad_rows_after));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
      input_cols, filter_cols, dilation_cols, stride_cols, params.padding,
      &out_cols, &pad_cols_before, &pad_cols_after));

  dimensions->batch = batch;
  dimensions->input_rows = input_rows;
  dimensions->input_cols = input_cols;
  dimensions->in_depth = in_depth;
  dimensions->filter_rows = filter_rows;
  dimensions->filter_cols = filter_cols;
  dimensions->patch_depth = patch_depth;
  dimensions->out_depth = out_depth;
  dimensions->stride_rows = stride_rows;
  dimensions->stride_cols = stride_cols;
  // TF dilations starts from 1, whereas ZenDNN dilations starts from 0
  dimensions->dilation_rows = dilation_rows - 1;
  dimensions->dilation_cols = dilation_cols - 1;
  dimensions->out_rows = out_rows;
  dimensions->out_cols = out_cols;
  dimensions->pad_rows_before = pad_rows_before;
  dimensions->pad_rows_after = pad_rows_after;
  dimensions->pad_cols_before = pad_cols_before;
  dimensions->pad_cols_after = pad_cols_after;

  return OkStatus();
}

Status ComputeVitisAIConv2DDimensions(VitisAIConv2DDimensions *dimensions,
                                      const VitisAIConv2DParameters &params,
                                      const Tensor &input, const Tensor &filter,
                                      const Tensor &dw_filter) {
  ComputeVitisAIConv2DDimensions(dimensions, params, input, filter, false);

  auto dw_params = params.dw_params;
  auto &dw_dimensions = dimensions->dw_dimensions;

  const auto ReadValidateFilterDim = [](const Tensor &filter,
                                        const int dim) -> int {
    int64 val_raw = filter.dim_size(dim);
    return static_cast<int>(val_raw);
  };

  // Get the filter dims
  int dw_filter_rows = ReadValidateFilterDim(dw_filter, 0);
  int dw_filter_cols = ReadValidateFilterDim(dw_filter, 1);

  TF_REQUIRES(dw_params.strides.size() == 4,
              errors::InvalidArgument("Sliding window strides field must "
                                      "specify 4 dimensions"));
  int dw_stride_rows = GetTensorDim(dw_params.strides, FORMAT_NHWC, 'H');
  int dw_stride_cols = GetTensorDim(dw_params.strides, FORMAT_NHWC, 'W');

  int64_t out_rows = 0, out_cols = 0;
  int64_t pad_rows_before, pad_cols_before, dummy_padding;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
      dimensions->out_rows, dw_filter_rows, /*dilation*/ 1, dw_stride_rows,
      dw_params.padding, &out_rows, &pad_rows_before, &dummy_padding));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeVerboseV2(
      dimensions->out_cols, dw_filter_cols, /*dilation*/ 1, dw_stride_cols,
      dw_params.padding, &out_cols, &pad_cols_before, &dummy_padding));

  dw_dimensions.filter_rows = dw_filter_rows;
  dw_dimensions.filter_cols = dw_filter_cols;
  dw_dimensions.stride_rows = dw_stride_rows;
  dw_dimensions.stride_cols = dw_stride_cols;
  dw_dimensions.out_rows = out_rows;
  dw_dimensions.out_cols = out_cols;
  dw_dimensions.pad_rows_before = pad_rows_before;
  dw_dimensions.pad_cols_before = pad_cols_before;

  return OkStatus();
}

#undef TF_REQUIRES

template <typename Toutput, typename Tsum>
Status SetVitisAIConv2DAttributes(zendnn::primitive_attr *conv_attr,
                                  const VitisAIConv2DParameters &params,
                                  const VitisAIConv2DDimensions dimensions,
                                  bool is_fused_depthwise, bool is_conv_fp32) {
  zendnn::post_ops post_ops;
  int modified_output_scale = params.output_scale;

  // Handling Asymetric Scaling: Sum nodes can have different scaling for
  // different inputs |         |
  //  \(x)    /(y)
  //   \     /
  //     Add
  //      |
  //     (z)
  // x, y and z are equal: no scale required for append_sum
  // x, y or z is different
  // 1. requantization is not done, conv accumulates in s32
  // 2. sum input is converted from s8/u8 range to s32 range
  // 3. Both values are added in s32
  // 4. Output downscaled using the scaling from another post-op
  // Order of execution: PostOp(Sum(Conv(input), add_input))
  // If relu is present, PostOp is ReLU else, Linear
  bool symmetric_sum = (params.sum_scale == params.output_scale &&
                        params.sum_scale == params.sum_out_scale);
  if ((!std::is_same<Tsum, void>::value && !symmetric_sum) ||
      std::is_same<Toutput, float>::value) {
    // requantize_scale = -params.input_scale - params.filter_scale;
    modified_output_scale = 0;
  }

  float downscale_scale = 1.0f;
  if (!std::is_same<Tsum, void>::value) {
    float sum_scale = 1.0f;
    if (!symmetric_sum) {
      sum_scale = std::pow(2, -params.sum_scale);
      downscale_scale = std::pow(2, params.sum_out_scale);
    }
    post_ops.append_sum(sum_scale);
  }

  // If relu_alpha is present, this means that it would be either ReLU6
  // or LeakyRelu  (both will havealpha in float range).
  // ReLU6:eltwise_bounded_relu. LeakyRelu: eltwise_relu
  // alpha * (2 ^ output_scale) will be alpha for append_eltwise and
  // will be upper bound for ReLU6 and negative slope for LeakyRelu
  if (params.is_relu) {
    auto relu_algo = zendnn::algorithm::eltwise_relu;
    if (params.relu_alpha == 6) {  // ReLU6
      relu_algo = zendnn::algorithm::eltwise_bounded_relu;
    }
    float relu_alpha = params.relu_alpha * std::pow(2, modified_output_scale);
    post_ops.append_eltwise(downscale_scale, relu_algo, relu_alpha, 0.0f);
  } else if (downscale_scale != 1.0f) {
    post_ops.append_eltwise(downscale_scale, zendnn::algorithm::eltwise_linear,
                            1.0f, 0.0f);
  }

  // Depthwise conv is added as a post op to the existing convolution
  // with the required parameters
  // Currently depthwise conv fusion is only enabled/implemented for
  // convolutions which has input and output as int8
  if (is_fused_depthwise && !is_conv_fp32) {
    auto dw_params = params.dw_params;
    auto dw_dimensions = dimensions.dw_dimensions;

    std::vector<float> dw_scale(1);
    dw_scale[0] = std::pow(2, -dw_params.input_scale - dw_params.filter_scale +
                                  dw_params.output_scale);
    auto dw_output_dtype = dw_params.is_relu ? dt::u8 : dt::s8;
    post_ops.append_dw(
        /* depthwise weights data type = */ dt::s8,
        /* depthwise bias data type (undef implies no bias) = */ dt::s32,
        /* depthwise destination data type = */ dw_output_dtype,
        /* kernel size of fused depthwise convolution = */
        dw_dimensions.filter_rows,
        /* stride size of fused depthwise convolution = */
        dw_dimensions.stride_rows,
        /* padding size of fused depthwise convolution = */
        dw_dimensions.pad_rows_before,
        /* mask for output scales of depthwise output = */ 0,
        /* output scales for depthwise output = */ dw_scale);
    if (dw_params.is_relu) {
      auto dw_relu_algo = zendnn::algorithm::eltwise_relu;
      if (dw_params.relu_alpha) {
        dw_relu_algo = zendnn::algorithm::eltwise_bounded_relu;
      }
      float dw_relu_alpha =
          dw_params.relu_alpha * std::pow(2, dw_params.output_scale);
      post_ops.append_eltwise(1.0f, dw_relu_algo, dw_relu_alpha, 0.0f);
    }
  }

  int requantize_scale =
      -params.input_scale - params.filter_scale + modified_output_scale;
  std::vector<float> conv_output_scales(1);
  conv_output_scales[0] = std::pow(2, requantize_scale);

  // fp32 convolution does not need output scales
  if (!is_conv_fp32) {
    conv_attr->set_output_scales(0, conv_output_scales);
  }
  conv_attr->set_scratchpad_mode(zendnn::scratchpad_mode::user);
  conv_attr->set_post_ops(post_ops);

  return OkStatus();
}

/**
 * @brief ZenVitisAIConv2DOp implements all VitisAIConv2D related Ops
 *
 * @tparam Tinput Input dtype
 * @tparam Tfilter Filter dtype
 * @tparam Tbias Bias dtype
 * @tparam Toutput Output dtype
 * @tparam Tsum Sum dtype (void if not available)
 * @tparam bias_enabled If bias is available or not
 * @tparam is_depthwise Is depthwise convolution
 * @tparam is_fused_depthwise Is depthwise convolution fused as post-op
 */
template <typename Tinput, typename Tfilter, typename Tbias, typename Toutput,
          typename Tsum = void, bool bias_enabled = true,
          bool is_depthwise = false, bool is_fused_depthwise = false>
class ZenVitisAIConv2DOp : public OpKernel {
 public:
  explicit ZenVitisAIConv2DOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, InitVitisAIConv2DParameters(context, &params_));
    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));
  }

  void Compute(OpKernelContext *context) override {
    ZenExecutor *ex = ex->getInstance();
    zendnn::engine engine = ex->getEngine();
    zendnn::stream stream = ex->getStream();

    // Input tensor
    const Tensor &input = ReadInputFromContext(context, 0);
    Tinput *input_array = static_cast<Tinput *>(
        const_cast<Tinput *>(input.flat<Tinput>().data()));

    // Filter tensor (will always be float)
    const Tensor &filter = ReadInputFromContext(context, 1);
    float *filter_array =
        static_cast<float *>(const_cast<float *>(filter.flat<float>().data()));

    // Bias tensor (will always be float)
    float *bias_array;
    if (bias_enabled) {
      const Tensor &bias = ReadInputFromContext(context, 2);
      bias_array =
          static_cast<float *>(const_cast<float *>(bias.flat<float>().data()));
    }

    // Compute all the required dimensions and set output shape
    float *dw_filter_array, *dw_bias_array;
    TensorShape out_shape;
    if (!is_fused_depthwise) {
      OP_REQUIRES_OK(context,
                     ComputeVitisAIConv2DDimensions(&dimensions, params_, input,
                                                    filter, is_depthwise));
      out_shape = ShapeFromFormat(params_.data_format, dimensions.batch,
                                  dimensions.out_rows, dimensions.out_cols,
                                  dimensions.out_depth);
    } else {
      // Filter tensor (will always be float)
      const Tensor &dw_filter = ReadInputFromContext(context, 3);
      dw_filter_array = static_cast<float *>(
          const_cast<float *>(dw_filter.flat<float>().data()));

      // Bias tensor (will always be float)
      const Tensor &dw_bias = ReadInputFromContext(context, 4);
      dw_bias_array = static_cast<float *>(
          const_cast<float *>(dw_bias.flat<float>().data()));

      OP_REQUIRES_OK(context,
                     ComputeVitisAIConv2DDimensions(&dimensions, params_, input,
                                                    filter, dw_filter));
      out_shape = ShapeFromFormat(params_.data_format, dimensions.batch,
                                  dimensions.dw_dimensions.out_rows,
                                  dimensions.dw_dimensions.out_rows,
                                  dimensions.out_depth);
    }

    zendnnInfo(ZENDNN_FWKLOG,
               "ZenVitisAIConv2DOp:", " in_depth = ", dimensions.in_depth,
               ", input_cols = ", dimensions.input_cols,
               ", input_rows = ", dimensions.input_rows,
               ", patch_depth = ", dimensions.patch_depth,
               ", filter_cols = ", dimensions.filter_cols,
               ", filter_rows = ", dimensions.filter_rows,
               ", stride_rows = ", dimensions.stride_rows,
               ", stride_cols = ", dimensions.stride_cols,
               ", dilation_rows = ", dimensions.dilation_rows,
               ", dilation_cols = ", dimensions.dilation_cols,
               ", out_depth = ", dimensions.out_depth, ", out_rows = ",
               (!is_fused_depthwise) ? dimensions.out_rows
                                     : dimensions.dw_dimensions.out_rows,
               ", out_cols = ",
               (!is_fused_depthwise) ? dimensions.out_cols
                                     : dimensions.dw_dimensions.out_cols,
               ", pad_rows_before = ", dimensions.pad_rows_before,
               ", pad_cols_before = ", dimensions.pad_cols_before,
               ", pad_rows_after = ", dimensions.pad_rows_after,
               ", pad_cols_after = ", dimensions.pad_cols_after);

    // Update the output type
    zenTensorType out_type = zenTensorType::FLOAT;
    if (std::is_same<Toutput, quint8>::value) {
      out_type = zenTensorType::QUINT8;
    } else if (std::is_same<Toutput, qint8>::value) {
      out_type = zenTensorType::QINT8;
    }

    zendnnEnv zenEnvObj = readEnv();
    int zenPoolEnable = zenEnvObj.zenEnableMemPool &&
                        (context->expected_output_dtype(0) == DT_QINT8 ||
                         context->expected_output_dtype(0) == DT_QUINT8 ||
                         context->expected_output_dtype(0) == DT_FLOAT);
    ZenMemoryPool<Toutput> *zenPoolBuffer=nullptr;

    Tensor *output = nullptr;
    if (!std::is_same<Tsum, void>::value) {
      Tensor &add_tensor = const_cast<Tensor &>(context->input(3));
      OP_REQUIRES_OK(
          context,
          add_tensor.BitcastFrom(
              add_tensor,
              std::is_same<Toutput, quint8>::value ? DT_QUINT8 : DT_QINT8,
              add_tensor.shape()));
      context->set_output(0, add_tensor);
      output = context->mutable_output(0);
      if (zenPoolEnable) {
        unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
        zenPoolBuffer = ZenMemoryPool<Toutput>::getZenMemPool(threadID);
        if (zenPoolBuffer) {
          const void *output_array;
          if (out_type == zenTensorType::QINT8) {
            output_array = const_cast<qint8 *>(output->flat<qint8>().data());
            zenPoolBuffer->zenMemPoolUpdateTensorPtrStatus(
                context, (qint8 *)output_array, zendnn_params_.out_links,
                zendnn_params_.reset);
          } else if (out_type == zenTensorType::QUINT8) {
            output_array = const_cast<quint8 *>(output->flat<quint8>().data());
            zenPoolBuffer->zenMemPoolUpdateTensorPtrStatus(
                context, (quint8 *)output_array, zendnn_params_.out_links,
                zendnn_params_.reset);
          }
        }
      }
    } else {
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
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, out_shape, &output));
      }
    }

    Toutput *output_array;
    auto output_map = output->tensor<Toutput, 4>();
    output_array = const_cast<Toutput *>(output_map.data());

    // Define dims for creating primitives
    memory::dims src_dims = {dimensions.batch, dimensions.in_depth,
                             dimensions.input_rows, dimensions.input_cols};
    memory::dims filter_dims = {dimensions.out_depth, dimensions.patch_depth,
                                dimensions.filter_rows, dimensions.filter_cols};
    if (is_depthwise) {
      filter_dims = {dimensions.patch_depth, 1, 1, dimensions.filter_rows,
                     dimensions.filter_cols};
    }
    memory::dims bias_dims = {dimensions.out_depth};
    memory::dims dst_dims = {dimensions.batch, dimensions.out_depth,
                             dimensions.out_rows, dimensions.out_cols};
    memory::dims strides_dims = {dimensions.stride_rows,
                                 dimensions.stride_cols};
    memory::dims dilation_dims = {dimensions.dilation_rows,
                                  dimensions.dilation_cols};
    memory::dims padding_left = {dimensions.pad_rows_before,
                                 dimensions.pad_cols_before};
    memory::dims padding_right = {dimensions.pad_rows_after,
                                  dimensions.pad_cols_after};

    bool is_conv_fp32 = std::is_same<Tinput, float>::value &&
                        std::is_same<Tinput, Toutput>::value;

    // Define memory desc for creating primitives
    memory::desc src_memory_desc = memory::desc(
        {src_dims}, DataTypetoZenForInput<Tinput, Toutput>(), tag::nhwc);
    memory::desc filter_memory_desc =
        memory::desc({filter_dims}, DataTypetoZen<Tfilter>(), tag::any);
    memory::desc dst_memory_desc = memory::desc(
        {dst_dims}, DataTypetoZenForOutput<Toutput, Tsum>(), tag::nhwc);

    // create primitive without bias
    convolution_forward::desc conv_desc = zendnn::convolution_forward::desc(
        zendnn::prop_kind::forward_inference,
        zendnn::algorithm::convolution_direct, src_memory_desc,
        filter_memory_desc, dst_memory_desc, strides_dims, dilation_dims,
        padding_left, padding_right);

    if (bias_enabled) {  // create primitive with bias
      memory::desc bias_memory_desc =
          memory::desc({bias_dims}, DataTypetoZen<Tbias>(), tag::x);
      conv_desc = zendnn::convolution_forward::desc(
          zendnn::prop_kind::forward_inference,
          zendnn::algorithm::convolution_direct, src_memory_desc,
          filter_memory_desc, bias_memory_desc, dst_memory_desc, strides_dims,
          dilation_dims, padding_left, padding_right);
    }

    // Setting conv attributes
    primitive_attr conv_attr;
    OP_REQUIRES_OK(context, SetVitisAIConv2DAttributes<Toutput, Tsum>(
                                &conv_attr, params_, dimensions,
                                is_fused_depthwise, is_conv_fp32));
    convolution_forward::primitive_desc conv_prim_desc =
        convolution_forward::primitive_desc(conv_desc, conv_attr, engine);

    // Scratchpad memory
    zendnn::memory::desc scratchpad_md = conv_prim_desc.scratchpad_desc();
    if (!UserScratchPad.isallocated()) {
      UserScratchPad.Allocate(context, scratchpad_md.get_size());
    }
    zendnn::memory scratchpad(scratchpad_md, engine,
                              UserScratchPad.GetTensorHandle());

    // Define memory to be used for primitive execution
    zendnn::memory conv_src_reordered_memory;
    zendnn::memory conv_filter_reordered_memory;
    zendnn::memory conv_bias_reordered_memory;
    zendnn::memory conv_dst_memory;
    zendnn::memory conv_dw_filter_reordered_memory;
    zendnn::memory conv_dw_bias_reordered_memory;

    // If input is f32 and the convolution is not f32, reorder with scale to s8
    if (std::is_same<Tinput, float>::value && !is_conv_fp32) {
      zendnn::memory conv_src_memory =
          memory({{src_dims}, dt::f32, tag::nhwc}, engine, input_array);
      conv_src_reordered_memory = memory(conv_prim_desc.src_desc(), engine);
      zp.AddReorder(conv_src_memory, conv_src_reordered_memory,
                    {params_.input_scale});
    } else {
      conv_src_reordered_memory =
          memory(conv_prim_desc.src_desc(), engine, input_array);
    }

    // Filter is cached after the first iteration. If filter is not cached yet,
    // reorder the filter from f32 to s8. If not get the cached tensor handle.
    if (!cached_filter.tensorvalue()) {
      zendnn::memory conv_filter_memory =
          memory({{filter_dims},
                  dt::f32,
                  (is_depthwise != 1) ? tag::hwio : tag ::hwigo},
                 engine, filter_array);
      conv_filter_reordered_memory =
          memory(conv_prim_desc.weights_desc(), engine);
      // If convolution is f32 conv, then no need to use any scale value
      if (!is_conv_fp32) {
        zp.AddReorder(conv_filter_memory, conv_filter_reordered_memory,
                      {params_.filter_scale});
      } else {
        zp.AddReorder(conv_filter_memory, conv_filter_reordered_memory);
      }
    } else {
      Tfilter *filter_data = cached_filter.GetTensorHandle();
      conv_filter_reordered_memory =
          memory(conv_prim_desc.weights_desc(), engine, filter_data);
    }

    // Define destination using dest_desc
    conv_dst_memory = memory(conv_prim_desc.dst_desc(), engine, output_array);

    // Create a map to store the primitive args
    std::unordered_map<int, memory> conv_prim_args;
    conv_prim_args.insert(
        {ZENDNN_ARG_SRC, conv_src_reordered_memory});  // input
    conv_prim_args.insert(
        {ZENDNN_ARG_WEIGHTS, conv_filter_reordered_memory});     // filter
    conv_prim_args.insert({ZENDNN_ARG_DST, conv_dst_memory});    // output
    conv_prim_args.insert({ZENDNN_ARG_SCRATCHPAD, scratchpad});  // scratchpad

    if (bias_enabled) {
      // Bias is cached after the first iteration. If bias is not cached yet,
      // reorder the bias from f32 to s32, If not get cached tensor handle.
      if (!cached_bias.tensorvalue()) {
        zendnn::memory conv_bias_memory =
            memory({{bias_dims}, dt::f32, tag::a}, engine, bias_array);
        conv_bias_reordered_memory = memory(conv_prim_desc.bias_desc(), engine);
        // If convolution is f32 conv, then no need to use any scale value
        if (!is_conv_fp32) {
          zp.AddReorder(conv_bias_memory, conv_bias_reordered_memory,
                        {params_.bias_scale});
        } else {
          zp.AddReorder(conv_bias_memory, conv_bias_reordered_memory);
        }
      } else {
        Tbias *bias_data = cached_bias.GetTensorHandle();
        conv_bias_reordered_memory =
            memory(conv_prim_desc.bias_desc(), engine, bias_data);
      }
      conv_prim_args.insert(
          {ZENDNN_ARG_BIAS, conv_bias_reordered_memory});  // bias
    }

    if (is_fused_depthwise) {
      // Query out the descriptors for depthwise filter and depthwise bias
      auto dw_filter_md = conv_prim_desc.query_md(
          query::exec_arg_md, ZENDNN_ARG_ATTR_POST_OP_DW | ZENDNN_ARG_WEIGHTS);
      auto dw_bias_md = conv_prim_desc.query_md(
          query::exec_arg_md, ZENDNN_ARG_ATTR_POST_OP_DW | ZENDNN_ARG_BIAS);

      // Depthwise filter is cached after the first iteration. If filter is not
      // cached yet, reorder the filter from f32 to s8. If not get the cached
      // tensor handle.
      if (!dw_cached_filter.tensorvalue()) {
        memory::dims dw_filter_dims = {dimensions.out_depth, 1, 1,
                                       dimensions.dw_dimensions.filter_rows,
                                       dimensions.dw_dimensions.filter_cols};
        auto conv_dw_filter_memory = zendnn::memory(
            {dw_filter_dims, dt::f32, tag::decab}, engine, dw_filter_array);

        conv_dw_filter_reordered_memory = memory(dw_filter_md, engine);
        zp.AddReorder(conv_dw_filter_memory, conv_dw_filter_reordered_memory,
                      {params_.dw_params.filter_scale});
      } else {
        Tfilter *dw_filter_data = dw_cached_filter.GetTensorHandle();
        conv_dw_filter_reordered_memory =
            memory(dw_filter_md, engine, dw_filter_data);
      }

      // Depthwise bias is cached after the first iteration. If bias is not
      // cached yet, reorder the bias from f32 to s32, If not get cached tensor
      // handle.
      if (!dw_cached_bias.tensorvalue()) {
        auto dw_bias_memory = zendnn::memory(
            {{dimensions.out_depth}, dt::f32, tag::x}, engine, dw_bias_array);
        conv_dw_bias_reordered_memory = memory(dw_bias_md, engine);
        zp.AddReorder(dw_bias_memory, conv_dw_bias_reordered_memory,
                      {params_.dw_params.bias_scale});
      } else {
        Tbias *dw_bias_data = dw_cached_bias.GetTensorHandle();
        conv_dw_bias_reordered_memory =
            memory(dw_bias_md, engine, dw_bias_data);
      }

      conv_prim_args.insert({ZENDNN_ARG_ATTR_POST_OP_DW | ZENDNN_ARG_WEIGHTS,
                             conv_dw_filter_reordered_memory});  // dw_filter
      conv_prim_args.insert({ZENDNN_ARG_ATTR_POST_OP_DW | ZENDNN_ARG_BIAS,
                             conv_dw_bias_reordered_memory});  // dw_bias
    }

    // Add conv primitive
    auto conv_prim = convolution_forward(conv_prim_desc);
    zp.AddPrimitive(conv_prim, conv_prim_args);

    // Execute all the added primitives
    zp.Execute(stream);

    // Reset any primitives added yet
    zp.reset();

    // Copy the reordered filter and bias to the persistent tensor
    cached_filter.SetTensorHandle(context, conv_filter_reordered_memory);
    if (bias_enabled) {
      cached_bias.SetTensorHandle(context, conv_bias_reordered_memory);
    }

    if (is_fused_depthwise) {
      // Copy the reordered depthwise filter and bias to the persistent tensor
      dw_cached_filter.SetTensorHandle(context,
                                       conv_dw_filter_reordered_memory);
      dw_cached_bias.SetTensorHandle(context, conv_dw_bias_reordered_memory);
    }

    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      zenPoolBuffer->zenMemPoolFree(context, (void *)input_array);
    }
  }

 private:
  // Parameters and dimensions
  VitisAIConv2DParameters params_;
  ZendnnParameters zendnn_params_;
  VitisAIConv2DDimensions dimensions;

  // Primitive library
  ZenPrimitives zp;
  ZenPersistentTensor<unsigned char> UserScratchPad;

  // Persistent tensors for normal conv
  ZenPersistentTensor<Tfilter> cached_filter;
  ZenPersistentTensor<Tbias> cached_bias;

  // Persistent tensors for dw fused conv
  ZenPersistentTensor<Tfilter> dw_cached_filter;
  ZenPersistentTensor<Tbias> dw_cached_bias;
};

// clang-format off

// Registering dummy kernels as NoOps (needs to be overwritten)
REGISTER_KERNEL_BUILDER(Name("VitisAIConv2D").Device(DEVICE_CPU), NoOp)
REGISTER_KERNEL_BUILDER(Name("VitisAIConv2DWithSum").Device(DEVICE_CPU), NoOp);
REGISTER_KERNEL_BUILDER(Name("VitisAIDepthwiseConv2D").Device(DEVICE_CPU), NoOp);
REGISTER_KERNEL_BUILDER(Name("VitisAIConv2DWithoutBias").Device(DEVICE_CPU),NoOp);
REGISTER_KERNEL_BUILDER(Name("_FusedVitisAIConv2DWithDepthwise").Device(DEVICE_CPU),NoOp);

// Float in/out ZenVitisAIConv2D kernel combinations
#define REGISTER_VITISAI_CONV2D_FLOAT(Tinput, Toutput)             \
  REGISTER_KERNEL_BUILDER(Name("_ZenVitisAIConv2D")                 \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<Tinput>("Tinput")    \
                              .TypeConstraint<float>("Tfilter")    \
                              .TypeConstraint<float>("Tbias")      \
                              .TypeConstraint<Toutput>("Toutput"), \
                              ZenVitisAIConv2DOp<Tinput, float, float, Toutput, void, true, false, false>)

REGISTER_VITISAI_CONV2D_FLOAT(float, float);

#undef REGISTER_VITISAI_CONV2D_FLOAT

// All the required ZenVitisAIConv2D kernel combinations
#define REGISTER_VITISAI_CONV2D(Tinput, Toutput)                   \
  REGISTER_KERNEL_BUILDER(Name("_ZenVitisAIConv2D")                 \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<Tinput>("Tinput")    \
                              .TypeConstraint<float>("Tfilter")    \
                              .TypeConstraint<float>("Tbias")      \
                              .TypeConstraint<Toutput>("Toutput"), \
                              ZenVitisAIConv2DOp<Tinput, qint8, qint32, Toutput, void, true, false, false>)

REGISTER_VITISAI_CONV2D(float, qint8);
REGISTER_VITISAI_CONV2D(float, quint8);
REGISTER_VITISAI_CONV2D(qint8, qint8);
REGISTER_VITISAI_CONV2D(qint8, quint8);
REGISTER_VITISAI_CONV2D(quint8, qint8);
REGISTER_VITISAI_CONV2D(quint8, quint8);
REGISTER_VITISAI_CONV2D(qint8, float);
REGISTER_VITISAI_CONV2D(quint8, float);

#undef REGISTER_VITISAI_CONV2D

// All the required ZenVitisAIDepthwiseConv2D kernel combinations
#define REGISTER_VITISAI_CONV2D_DEPTHWISE(Tinput, Toutput)         \
  REGISTER_KERNEL_BUILDER(Name("_ZenVitisAIDepthwiseConv2D")        \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<Tinput>("Tinput")    \
                              .TypeConstraint<float>("Tfilter")    \
                              .TypeConstraint<float>("Tbias")      \
                              .TypeConstraint<Toutput>("Toutput"), \
                              ZenVitisAIConv2DOp<Tinput, qint8, qint32, Toutput, void, true, true, false>)

REGISTER_VITISAI_CONV2D_DEPTHWISE(quint8, qint8);
REGISTER_VITISAI_CONV2D_DEPTHWISE(quint8, quint8);

#undef REGISTER_VITISAI_CONV2D_DEPTHWISE

// All the required ZenVitisAIConv2DWithSum kernel combinations
#define REGISTER_VITISAI_CONV2D_SUM(Tinput, Toutput, Tsum)         \
  REGISTER_KERNEL_BUILDER(Name("_ZenVitisAIConv2DWithSum")          \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<Tinput>("Tinput")    \
                              .TypeConstraint<float>("Tfilter")    \
                              .TypeConstraint<float>("Tbias")      \
                              .TypeConstraint<Toutput>("Toutput")  \
                              .TypeConstraint<Tsum>("Tsum"),       \
                              ZenVitisAIConv2DOp<Tinput, qint8, qint32, Toutput, Tsum, true, false, false>)

REGISTER_VITISAI_CONV2D_SUM(quint8, qint8, qint8);
REGISTER_VITISAI_CONV2D_SUM(quint8, qint8, quint8);
REGISTER_VITISAI_CONV2D_SUM(quint8, quint8, qint8);
REGISTER_VITISAI_CONV2D_SUM(quint8, quint8, quint8);

#undef REGISTER_VITISAI_CONV2D_SUM

// All the required ZenVitisAIConv2DWithoutBias kernel combinations
#define REGISTER_VITISAI_CONV2D_WITHOUT_BIAS(Tinput, Toutput)      \
  REGISTER_KERNEL_BUILDER(Name("_ZenVitisAIConv2DWithoutBias")      \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<Tinput>("Tinput")    \
                              .TypeConstraint<float>("Tfilter")    \
                              .TypeConstraint<Toutput>("Toutput"), \
                              ZenVitisAIConv2DOp<Tinput, qint8, qint32, Toutput, void, false, false, false>)

REGISTER_VITISAI_CONV2D_WITHOUT_BIAS(quint8, quint8);

#undef REGISTER_VITISAI_CONV2D_WITHOUT_BIAS

// All the required ZenFusedVitisAIConv2DWithDepthwise kernel combinations
#define REGISTER_VITISAI_CONV2D_WITH_DEPTHWISE(Tinput, Toutput)      \
  REGISTER_KERNEL_BUILDER(Name("_ZenFusedVitisAIConv2DWithDepthwise")      \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<Tinput>("Tinput")    \
                              .TypeConstraint<float>("Tfilter")    \
                              .TypeConstraint<float>("Tbias")      \
                              .TypeConstraint<Toutput>("Toutput"), \
                              ZenVitisAIConv2DOp<Tinput, qint8, qint32, Toutput, void, true, false, true>)

REGISTER_VITISAI_CONV2D_WITH_DEPTHWISE(qint8, quint8);
REGISTER_VITISAI_CONV2D_WITH_DEPTHWISE(quint8, quint8);

#undef REGISTER_VITISAI_CONV2D_WITH_DEPTHWISE

// clang-format on

}  // namespace tensorflow
