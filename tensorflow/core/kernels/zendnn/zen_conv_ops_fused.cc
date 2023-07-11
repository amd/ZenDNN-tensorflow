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

#include <iostream>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/fused_eigen_output_kernels.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "tensorflow/core/kernels/zendnn/zen_conv_ops_util.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "zendnn_helper.hpp"

#define NEW_API 1
#if NEW_API
#include "zendnn.hpp"
using namespace zendnn;
using namespace std;

// TODO: It may be better to read this locally (July release)
zendnnEnv zenEnvObj = readEnv();

void zenGemmConvolution2D(void *input_array, int batch_size, int channels,
                          int height, int width, void *filter_array,
                          int output_channels, int kernel_h, int kernel_w,
                          float pad_t, float pad_l, float pad_b, float pad_r,
                          int stride_h, int stride_w, void *bias_array,
                          void *output_array, int out_height, int out_width,
                          bool reluFused, bool batchNormFused, bool addFused,
                          void *bn_scale, void *bn_mean, void *bn_offset);
template <typename T>
void zenBlockedConv2DBiasEltSum(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void *input_array, int batch_size, int channels, int height, int width,
    void *filter_array, int output_channels, int kernel_h, int kernel_w,
    int pad_t, int pad_l, int pad_b, int pad_r, int stride_h, int stride_w,
    void *bias_array, void *output_array, int out_height, int out_width,
    bool reorder_before, bool reorder_after, void *cached_filter_data_,
    void *context);

template <typename T>
void zenConvolution2DBiasOrRelu(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void *input_array, int batch_size, int channels, int height, int width,
    void *filter_array, int output_channels, int kernel_h, int kernel_w,
    float pad_t, float pad_l, float pad_b, float pad_r, int stride_h,
    int stride_w, void *bias_array, void *output_array, int out_height,
    int out_width, bool reorder_before, bool reorder_after,
    void *cached_filter_data_, void *context);

void zenConvolution2DBatchNormOrRelu(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void *input_array, int batch_size, int channels, int height, int width,
    void *filter_array, int output_channels, int kernel_h, int kernel_w,
    float pad_t, float pad_l, float pad_b, float pad_r, int stride_h,
    int stride_w, void *bias_array, void *batch_norm_scale,
    void *batch_norm_mean, void *batch_norm_offset, void *elementwise_input,
    void *output_array, int out_height, int out_width, bool reluFused,
    bool batchNormFused, bool reorder_before, bool reorder_after,
    void *cached_filter_data_, void *context);
void zenQuantized(zendnn::engine eng, zendnn::stream s, void *input_array,
                  int batch_size, int height, int width, int channels,
                  float scale_factor, bool out_type, void *output);

void zenQuantizedConvolution2DBiasOrRelu(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void *context, void *input_array, int bs, int channels, int height,
    int width, void *filter_array, int output_channels, int kernel_h,
    int kernel_w, int oh, int ow, int pad_t, int pad_l, int pad_b, int pad_r,
    int stride_h, int stride_w, void *bias_array, vector<float> &scale,
    void *output, void *output_min, void *output_max, bool in_type, bool outype,
    bool Tbias, vector<float> &bias_scale, bool is_relu, bool is_sum,
    bool is_signed, float factor, int depth, float scale_output,
    float scale_summand, void *cached_filter_data_, bool reset);

#endif

// Uncomment to enable complete upgraded API of QuantizeV2 op
// Commented (for now) to avoid any overhead
/*
namespace {
enum {
    QUANTIZE_MODE_MIN_COMBINED,
    QUANTIZE_MODE_MIN_FIRST,
    QUANTIZE_MODE_SCALED,
};
enum {
    // Round half away from zero: if the fraction of y is exactly 0.5, then
    // round(y) = y + 0.5 if y > 0
    // round(y) = y - 0.5 if y < 0
    // E.g., -5.5 gets rounded to -6, -5.4 goes to -5,
    // 5.4 goes to 5, and 5.5 goes to 6.
    ROUND_HALF_AWAY_FROM_ZERO,
    // Round half to even: if the fraction of y is exactly 0.5, then round(y) is
    // the nearest even integer to y.
    // E.g., 23.5 gets rounded to 24, 24.5 gets rounded to 24, while -23.5
becomes
    // -24, and -24.5 gets rounded to 24.
    ROUND_HALF_TO_EVEN,
};
}  // namespace
*/

namespace tensorflow {

// TODO: This function to be improved for July release
template <typename Toutput>
class ZenQuantizeV2Op : public OpKernel {
 public:
  explicit ZenQuantizeV2Op(OpKernelConstruction *context) : OpKernel(context) {
    // Uncomment to enable complete upgraded API of QuantizeV2 op
    // Commented (for now) to avoid any overhead
    /*
    half_range_ =
            !std::is_signed<Toutput>::value
            ? 0.0f
            : (static_cast<double>(std::numeric_limits<Toutput>::max()) -
               static_cast<double>(std::numeric_limits<Toutput>::min()) + 1) /
            2.0f;
    string mode_string;
    OP_REQUIRES_OK(context, context->GetAttr("mode", &mode_string));
    OP_REQUIRES(context,
                (mode_string == "MIN_COMBINED" || mode_string == "MIN_FIRST"
    || mode_string == "SCALED"), errors::InvalidArgument("Mode string must be
    'MIN_COMBINED'," " 'MIN_FIRST', or 'SCALED', is '" + mode_string + "'"));
    if (mode_string == "MIN_COMBINED") {
      mode_ = QUANTIZE_MODE_MIN_COMBINED;
    }
    else if (mode_string == "MIN_FIRST") {
      mode_ = QUANTIZE_MODE_MIN_FIRST;
    }
    else if (mode_string == "SCALED") {
      mode_ = QUANTIZE_MODE_SCALED;
    }

    string round_mode_string;
    OP_REQUIRES_OK(context, context->GetAttr("round_mode",
                   &round_mode_string));
    OP_REQUIRES(context, (round_mode_string ==
                "HALF_AWAY_FROM_ZERO" || round_mode_string == "HALF_TO_EVEN"),
                errors::InvalidArgument("Round mode string must be "
                                            "'HALF_AWAY_FROM_ZERO' or "
                                            "'HALF_TO_EVEN', is '" +
                                            round_mode_string + "'"));
    if (round_mode_string == "HALF_AWAY_FROM_ZERO") {
      round_mode_ = ROUND_HALF_AWAY_FROM_ZERO;
    }
    else if (round_mode_string == "HALF_TO_EVEN") {
      OP_REQUIRES(context, mode_string == "SCALED",
                  errors::InvalidArgument("Round mode 'HALF_TO_EVEN' "
                  "only supported for mode 'SCALED', " "but mode is '"
                  + mode_string + "'."));
      round_mode_ = ROUND_HALF_TO_EVEN;
    }
    */
    OP_REQUIRES_OK(context,
                   context->GetAttr("reorder_before", &reorder_before));
    OP_REQUIRES_OK(context, context->GetAttr("reorder_after", &reorder_after));
    OP_REQUIRES_OK(context, context->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(context, context->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(context, context->GetAttr("reset", &reset));
    OP_REQUIRES_OK(context, context->GetAttr("narrow_range", &narrow_range_));
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
    OP_REQUIRES_OK(context, context->GetAttr("ensure_minimum_range",
                                             &ensure_minimum_range_));
  }
  void Compute(OpKernelContext *context) override {
    ZenExecutor *ex = ex->getInstance();
    engine eng = ex->getEngine();
    stream s = ex->getStream();

    const float input_min_range = context->input(1).flat<float>()(0);
    const float input_max_range = context->input(2).flat<float>()(0);
    float min_range = std::min(0.0f, input_min_range);
    const float epsilon = std::max(1.0f, std::max(fabsf(input_min_range),
                                                  fabsf(input_max_range))) *
                          ensure_minimum_range_;
    float max_range = std::max(input_max_range, min_range + epsilon);

    const Tensor &input = context->input(0);
    int batch_size = input.dim_size(0);
    int height = input.dim_size(1);
    int width = input.dim_size(2);
    int channels = input.dim_size(3);

    TensorShape zen_out_shape_max, zen_out_shape_min;
    TensorShape out_shape = input.shape();
    Tensor *output = nullptr, *output_min = nullptr, *output_max = nullptr;

    float *input_array =
        static_cast<float *>(const_cast<float *>(input.flat<float>().data()));
    void *output_array;
    // Update the output type
    zenTensorType out_type = zenTensorType::QINT8;
    if (std::is_same<Toutput, quint8>::value) {
      out_type = zenTensorType::QUINT8;
    }

    zendnnEnv zenEnvObj = readEnv();
    int zenEnableMemPool = zenEnvObj.zenEnableMemPool &&
                           (context->expected_output_dtype(0) == DT_QINT8 ||
                            context->expected_output_dtype(0) == DT_QUINT8);
    ZenMemoryPool<Toutput> *zenPoolBuffer=NULL;

    if (zenEnableMemPool) {
      unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
      zenPoolBuffer = ZenMemoryPool<Toutput>::getZenMemPool(threadID);
      if (zenPoolBuffer) {
        // Quantized models have 3 outputs 1 input is used
        // for computatuion other 2 outputs are used during dequantize
        int status = zenPoolBuffer->acquireZenPoolTensor(
            context, &output, out_shape, out_links - 2, reset, out_type);
        if (status) {
          zenEnableMemPool = false;
        }
      } else {
        zenEnableMemPool = false;
      }
    }
    if (!zenEnableMemPool) {
      // Outtype is not required for default allocation because context
      // maintains allocation data Type for outputs
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    }

    auto output_map = output->tensor<Toutput, 4>();
    // Define dimension of output_array based on dimension of out_shape
    // Fix for ResNet50v1.5 INT8 model
    output_array = const_cast<Toutput *>(output_map.data());
    if (out_shape.dims() == 2) {
      output_array = const_cast<Toutput *>(output->tensor<Toutput, 2>().data());
    } else {
      output_array = const_cast<Toutput *>(output->tensor<Toutput, 4>().data());
    }

    OP_REQUIRES_OK(context,
                   context->allocate_output(1, zen_out_shape_min, &output_min));
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, zen_out_shape_max, &output_max));

    const int num_bits = sizeof(Toutput) * 8;
    const float max_abs = std::max(std::abs(min_range), std::abs(max_range));
    const bool is_signed = std::is_signed<Toutput>::value;
    float target_range;
    if (is_signed) {
      max_range = max_abs;
      min_range = -max_abs;
      target_range = static_cast<float>((uint64_t{1} << (num_bits - 1)) - 1);
    } else {
      max_range = max_abs;
      min_range = 0.0;
      target_range = static_cast<float>((uint64_t{1} << num_bits) - 1);
    }

    output_min->flat<float>()(0) = min_range;
    output_max->flat<float>()(0) = max_range;
    const float scale_factor = target_range / max_abs;

    zenQuantized(eng, s, input_array, batch_size, height, width, channels,
                 scale_factor, (bool)out_type, output_array);

    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      zenPoolBuffer->zenMemPoolFree(context, (void *)input_array);
    }
  }

 private:
  // Defined additional attributes to update QuantizeV2 op API
  Tensor cached_data_ TF_GUARDED_BY(mu_);
  bool reorder_before, reorder_after, reset, narrow_range_;
  int in_links, out_links, axis_;
  float ensure_minimum_range_;
  float half_range_;
  int mode_;
  int round_mode_;
};

template <typename Tinput, typename Tfilter, typename Tbias, typename Toutput,
          typename Ttemp_output, bool bias_enabled, bool is_depthwise,
          bool is_relu, bool is_sum, bool is_signed>
class ZenQuantizedConv2DOp : public OpKernel {
 public:
  explicit ZenQuantizedConv2DOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));
    // Support for new padding definition for Quantized Conv ops in TF
    // Fix for pad accuracy issue with ResNet50v1.5 model
    // For Quantized Conv ops, there is no EXPLICIT pad type (as in FP32 Conv
    // ops) A new attribute padding_list is used with VALID pad type If
    // padding_list has custom values, then it should be used. If no custom
    // values have been defined, then pad value of 0 is used (VALID type)
    if (context->HasAttr("padding_list")) {
      // padding_list_control = true;
      OP_REQUIRES_OK(context, context->GetAttr("padding_list", &padding_list_));
    }
    OP_REQUIRES_OK(context,
                   context->GetAttr("reorder_before", &reorder_before));
    OP_REQUIRES_OK(context, context->GetAttr("reorder_after", &reorder_after));
    OP_REQUIRES_OK(context, context->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(context, context->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(context, context->GetAttr("reset", &reset));
  }
  void Compute(OpKernelContext *context) override {
    ZenExecutor *ex = ex->getInstance();
    engine eng = ex->getEngine();
    stream s = ex->getStream();

    // Input Filter and Bias
    const Tensor &input = context->input(0);
    Tinput *input_array = static_cast<Tinput *>(
        const_cast<Tinput *>(input.flat<Tinput>().data()));

    const Tensor &filter = context->input(1);
    Tfilter *filter_array = static_cast<Tfilter *>(
        const_cast<Tfilter *>(filter.flat<Tfilter>().data()));

    BiasAddArgs<Tbias> bias_add_args;
    OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add_args));
    Tbias *bias_array = const_cast<Tbias *>(bias_add_args.bias_add_data);

    // Update the output type
    zenTensorType out_type = zenTensorType::QINT8;
    if (std::is_same<Toutput, quint8>::value) {
      out_type = zenTensorType::QUINT8;
    }
    bool bias_type = std::is_same<Tbias, qint32>::value;
    // Input type is defined (signed or unsigned)
    // Fix for ResNet50v1.5 INT8 model where signed INT8 input is used
    bool in_type = std::is_same<Tinput, quint8>::value;

    TensorShape zen_out_shape_max, zen_out_shape_min;

    Conv2DDimensions dimensions;

    // Compute Convolution / Quantization Specific parameters

    Tensor *output = nullptr, *output_min = nullptr, *output_max = nullptr;
    TensorShape out_shape;
    int batch_size, channels, height, width, output_channels, kernel_height,
        kernel_width;
    int bias_index_offset = bias_enabled ? 1 : 0;
    float scale_output, scale_summand;

    const int stride_rows =
        GetTensorDim(params_.strides, params_.data_format, 'H');
    const int stride_cols =
        GetTensorDim(params_.strides, params_.data_format, 'W');
    const int dilation_rows =
        GetTensorDim(params_.dilations, params_.data_format, 'H');
    const int dilation_cols =
        GetTensorDim(params_.dilations, params_.data_format, 'W');

    int64 out_rows = 0, out_cols = 0;
    int64 pad_rows_before = 0, pad_rows_after = 0, pad_cols_before = 0,
          pad_cols_after = 0;

    batch_size = input.dim_size(0);
    channels = input.dim_size(3);

    if (!is_depthwise) {
      kernel_width = filter.dim_size(1);
      kernel_height = filter.dim_size(0);
      output_channels = filter.dim_size(3);
    } else {
      kernel_width = filter.dim_size(1);
      kernel_height = filter.dim_size(0);
      output_channels = filter.dim_size(2);
    }

    height = input.dim_size(1);
    width = input.dim_size(2);

    GetWindowedOutputSizeVerboseV2(width, kernel_width, dilation_cols,
                                   stride_cols, params_.padding, &out_cols,
                                   &pad_cols_before, &pad_cols_after);
    GetWindowedOutputSizeVerboseV2(height, kernel_height, dilation_rows,
                                   stride_rows, params_.padding, &out_rows,
                                   &pad_rows_before, &pad_rows_after);

    // Support for new padding type in Quantized Conv ops
    for (auto const &padding_val : padding_list_) {
      if (padding_val > 0) {
        pad_rows_before = pad_rows_after = pad_cols_before = pad_cols_after =
            padding_val;
        out_rows = out_cols =
            (height + pad_cols_before + pad_cols_after - kernel_height) /
                (stride_rows) +
            1;
        break;
      }
    }

    out_shape = ShapeFromFormat(params_.data_format, batch_size, out_rows,
                                out_cols, output_channels);

    OP_REQUIRES_OK(context,
                   context->allocate_output(1, zen_out_shape_min, &output_min));
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, zen_out_shape_max, &output_max));

    const float min_input =
        context->input(2 + bias_index_offset).flat<float>()(0);
    const float max_input =
        context->input(3 + bias_index_offset).flat<float>()(0);
    const Tensor &min_filter_vector = context->input(4 + bias_index_offset);
    const Tensor &max_filter_vector = context->input(5 + bias_index_offset);
    const float min_freezed_output =
        context->input(6 + bias_index_offset).flat<float>()(0);
    const float max_freezed_output =
        context->input(7 + bias_index_offset).flat<float>()(0);

    output_min->flat<float>()(0) =
        context->input(6 + bias_index_offset).flat<float>()(0);
    output_max->flat<float>()(0) =
        context->input(7 + bias_index_offset).flat<float>()(0);

    output_min = context->mutable_output(1);
    output_max = context->mutable_output(2);

    float factor = is_signed ? 127.0f : 255.0f;
    float ftype = (bool)out_type ? 255.0f : 127.0f;
    size_t depth = 1;
    depth = min_filter_vector.NumElements();
    const float *min_filter = min_filter_vector.flat<float>().data();
    const float *max_filter = max_filter_vector.flat<float>().data();
    std::vector<float> scales(depth);
    std::vector<float> bias_scales(depth);
    float input_range = std::max(std::abs(min_input), std::abs(max_input));
    float output_range =
        std::max(std::abs(min_freezed_output), std::abs(max_freezed_output));

    for (size_t i = 0; i < depth; ++i) {
      float filter_range =
          std::max(std::abs(min_filter[i]), std::abs(max_filter[i]));
      // Changes to fix accuracy issue with ResNet50v1.5 first Conv layer
      const float int_const_scale_limit =
          (in_type) ? 255.0 * 127.0 : 127.0 * 127.0;
      scales[i] = (ftype * input_range * filter_range) /
                  (int_const_scale_limit * output_range);
      bias_scales[i] = int_const_scale_limit /
                       (input_range * std::max(std::abs(min_filter[i]),
                                               std::abs(max_filter[i])));
    }

    Toutput *output_array, *sum_array;
    int status = 0;
    Tensor *sum = nullptr;
    zendnnEnv zenEnvObj = readEnv();
    int zenEnableMemPool = zenEnvObj.zenEnableMemPool &&
                           (context->expected_output_dtype(0) == DT_QINT8 ||
                            context->expected_output_dtype(0) == DT_QUINT8);
    ZenMemoryPool<Toutput> *zenPoolBuffer=NULL;

    if (is_sum) {
      const float min_freezed_summand =
          context->input(9 + bias_index_offset).flat<float>()(0);
      const float max_freezed_summand =
          context->input(10 + bias_index_offset).flat<float>()(0);
      scale_output =
          std::max(std::abs(min_freezed_output), std::abs(max_freezed_output));
      scale_summand = std::max(std::abs(min_freezed_summand),
                               std::abs(max_freezed_summand));

      Tensor &add_tensor = const_cast<Tensor &>(context->input(9));

      OP_REQUIRES_OK(context, add_tensor.BitcastFrom(add_tensor, DT_QUINT8,
                                                     add_tensor.shape()));
      context->set_output(0, add_tensor);
      output = context->mutable_output(0);
      if (zenEnableMemPool) {
        unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
        zenPoolBuffer = ZenMemoryPool<Toutput>::getZenMemPool(threadID);
        if (zenPoolBuffer) {
          const void *output_array;
          if (out_type == zenTensorType::QINT8) {
            output_array = const_cast<qint8 *>(output->flat<qint8>().data());
            // Quantized models have 3 outputs 1 input is used
            // for computatuion other 2 outputs are used during dequantize
            zenPoolBuffer->zenMemPoolUpdateTensorPtrStatus(
                context, (qint8 *)output_array, out_links - 2, reset);
          } else if (out_type == zenTensorType::QUINT8) {
            output_array = const_cast<quint8 *>(output->flat<quint8>().data());
            zenPoolBuffer->zenMemPoolUpdateTensorPtrStatus(
                context, (quint8 *)output_array, out_links - 2, reset);
          }
        }
      }
    } else {
      if (zenEnableMemPool) {
        unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
        zenPoolBuffer = ZenMemoryPool<Toutput>::getZenMemPool(threadID);
        if (zenPoolBuffer) {
          // Quantized models have 3 outputs 1 input is used
          // for computatuion other 2 outputs are used during dequantize
          int status = zenPoolBuffer->acquireZenPoolTensor(
              context, &output, out_shape, out_links - 2, reset, out_type);
          if (status) {
            zenEnableMemPool = false;
          }
        } else {
          zenEnableMemPool = false;
        }
      }
      if (!zenEnableMemPool) {
        // Outtype is not required for default allocation because context
        // maintains allocation data Type for outputs
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, out_shape, &output));
      }
    }

    auto output_map = output->tensor<Toutput, 4>();
    output_array = const_cast<Toutput *>(output_map.data());

    // There are edge cases where destination memory type from registered Op is
    // unsigned but results of the Operations are signed Example patterns is
    // when convolution does not have relu and postops is signed In these cases
    // post memory allocation we cast them to signed based on the new out_type
    // TODO: Hardcoding to be removed with alternative patch
    if (depth == 1) {
      if (!is_relu || is_signed) {
        out_type = zenTensorType::QINT8;
      }
    } else {
      out_type = zenTensorType::QUINT8;
    }
    // Accuracy fix for ResNet50v1.5 INT8 model
    // TODO: Add an alternative fix
    // For specific Convolution layers, output type is unsigned instead of
    // signed. 7 Convolution layers involved in this fix.
    if ((channels == 64 && output_channels == 256 && height == 56) ||
        (channels == 256 && output_channels == 512 && height == 56) ||
        (channels == 128 && output_channels == 512 && height == 28) ||
        (channels == 512 && output_channels == 1024 && height == 28) ||
        (channels == 256 && output_channels == 1024 && height == 14) ||
        (channels == 1024 && output_channels == 2048 && height == 14) ||
        (channels == 512 && output_channels == 2048 && height == 7)) {
      out_type = zenTensorType::QINT8;
    }

    primitive_attr conv_attr;

    zenQuantizedConvolution2DBiasOrRelu(
        eng, s, conv_attr, context, input_array, batch_size, channels, height,
        width, filter_array, output_channels, kernel_height, kernel_width,
        out_rows, out_cols, pad_rows_before, pad_cols_before, pad_rows_after,
        pad_cols_after, stride_rows, stride_cols, bias_array, scales,
        output_array, output_min, output_max, in_type, (bool)out_type,
        bias_type, bias_scales, is_relu, is_sum, is_signed, factor,
        is_depthwise, scale_output, scale_summand, &cached_filter_data_, reset);
    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      zenPoolBuffer->zenMemPoolFree(context, (void *)input_array);
    }
  }

 private:
  Conv2DParameters params_;
  bool reorder_before, reorder_after, reset;
  int in_links, out_links;
  // Additional attributes to support new Padding definition and tensors
  std::vector<int64> padding_list_;
  Tensor cached_filter_data_ TF_GUARDED_BY(mu_);
  Tensor cached_data_ TF_GUARDED_BY(mu_);
};

template <typename T>
struct LaunchZenFusedConv2DOp {
  void operator()(OpKernelContext *context, const Tensor &input,
                  const Tensor &filter, const FusedComputationType fusion,
                  const FusedComputationArgs &fusion_args,
                  const Conv2DParameters &params,
                  const Conv2DDimensions &dimensions, Tensor *output,
                  bool reorder_before, bool reorder_after,
                  Tensor *cached_filter_data_, bool is_depthwise) {
    OP_REQUIRES(context, dimensions.in_depth == filter.dim_size(2),
                errors::Unimplemented("Fused conv implementation does not "
                                      "support grouped convolutions for now."));
    OP_REQUIRES(context, params.data_format == FORMAT_NHWC,
                errors::Unimplemented("Fused conv implementation only supports "
                                      "NHWC tensor format for now."));

    BiasAddArgs<T> bias_add_args;
    if (BiasAddArgs<T>::IsSupported(fusion)) {
      OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add_args));
    }

    FusedBatchNormArgs<T> fused_batch_norm_args;
    if (FusedBatchNormArgs<T>::IsSupported(fusion)) {
      OP_REQUIRES_OK(context,
                     InitFusedBatchNormArgs(context, fusion_args.epsilon,
                                            &fused_batch_norm_args));
    }

    auto input_map = input.tensor<T, 4>();
    const T *input_array = input_map.data();
    auto filter_map = filter.tensor<T, 4>();
    const T *filter_array = filter_map.data();
    auto output_map = output->tensor<T, 4>();
    T *output_array = const_cast<T *>(output_map.data());
    T *in_arr = const_cast<T *>(input_array);
    T *filt_arr = const_cast<T *>(filter_array);

    bool is_input_float = std::is_same<T, float>::value;

#if NEW_API
    zendnnEnv zenEnvObj = readEnv();
    bool blocked = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT1;
    bool blockedNHWC = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT2;
    ZenExecutor *ex = ex->getInstance();
    engine eng = ex->getEngine();
    stream s = ex->getStream();
#endif
    // Note: we are forcing it through BlockedNHWC when input is bf16 towards
    // beta support for this release.
    if (!is_input_float) {
      blocked = 0;
      blockedNHWC = 1;
    }
    switch (fusion) {
      case FusedComputationType::kUndefined:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type is undefined"));
        break;
      case FusedComputationType::kBiasAdd: {
#if NEW_API
        T *bias_arr = const_cast<T *>(bias_add_args.bias_add_data);
        primitive_attr conv_attr;
        if (is_depthwise) {
          zenConvolution2DDepthwise<T>(
              eng, s, conv_attr, in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, reorder_before,
              reorder_after, cached_filter_data_, context);
        } else if (blocked || blockedNHWC) {
          // Direct convolution
          zenConvolution2DBiasOrRelu<T>(
              eng, s, conv_attr, in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, reorder_before,
              reorder_after, cached_filter_data_, context);
        } else {
          // GEMM based convolution
          zenGemmConvolution2D(
              in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, false, false, false,
              nullptr, nullptr, nullptr);
        }
#else
        zenConvolution2DwithBias(
            input_array, dimensions.batch, dimensions.in_depth,
            dimensions.input_rows, dimensions.input_cols, filter_array,
            dimensions.out_depth, dimensions.filter_rows,
            dimensions.filter_cols, dimensions.pad_rows_before,
            dimensions.pad_cols_before, dimensions.pad_rows_after,
            dimensions.pad_cols_after, dimensions.stride_rows,
            dimensions.stride_cols, bias_add_args.bias_add_data, output_array,
            dimensions.out_rows, dimensions.out_cols);
#endif
        break;
      }
      case FusedComputationType::kBiasAddWithRelu: {
#if NEW_API
        T *bias_arr = const_cast<T *>(bias_add_args.bias_add_data);
        primitive_attr conv_attr;
        //[Configure post-ops]
        const float ops_scale = 1.f;
        const float ops_alpha = 0.f;  // relu negative slope
        const float ops_beta = 0.f;
        post_ops ops;
        ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha,
                           ops_beta);
        conv_attr.set_post_ops(ops);
        if (is_depthwise) {
          zenConvolution2DDepthwise<T>(
              eng, s, conv_attr, in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, reorder_before,
              reorder_after, cached_filter_data_, context);
        }

        else if (blocked || blockedNHWC) {
          // Direct convolution
          zenConvolution2DBiasOrRelu<T>(
              eng, s, conv_attr, in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, reorder_before,
              reorder_after, cached_filter_data_, context);
        } else {
          // GEMM based convolution
          zenGemmConvolution2D(
              in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, true, false, false,
              nullptr, nullptr, nullptr);
        }
#else
        zenConvolution2DwithBiasRelu(
            input_array, dimensions.batch, dimensions.in_depth,
            dimensions.input_rows, dimensions.input_cols, filter_array,
            dimensions.out_depth, dimensions.filter_rows,
            dimensions.filter_cols, dimensions.pad_rows_before,
            dimensions.pad_cols_before, dimensions.pad_rows_after,
            dimensions.pad_cols_after, dimensions.stride_rows,
            dimensions.stride_cols, bias_add_args.bias_add_data, output_array,
            dimensions.out_rows, dimensions.out_cols);
#endif
        break;
      }
      case FusedComputationType::kBiasAddWithRelu6: {
#if NEW_API
        T *bias_arr = const_cast<T *>(bias_add_args.bias_add_data);
        primitive_attr conv_attr;
        //[Configure post-ops]
        const float ops_scale = 1.f;
        const float ops_alpha = 6.0;  // relu negative slope
        const float ops_beta = 0.f;
        post_ops ops;
        ops.append_eltwise(ops_scale, algorithm::eltwise_bounded_relu,
                           ops_alpha, ops_beta);
        conv_attr.set_post_ops(ops);
        if (is_depthwise) {
          zenConvolution2DDepthwise<T>(
              eng, s, conv_attr, in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, reorder_before,
              reorder_after, cached_filter_data_, context);
        } else if (blocked || blockedNHWC) {
          // Direct convolution
          zenConvolution2DBiasOrRelu<T>(
              eng, s, conv_attr, in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, reorder_before,
              reorder_after, cached_filter_data_, context);
        } else {
          // GEMM based convolution
          zenGemmConvolution2D(
              in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, true, false, false,
              nullptr, nullptr, nullptr);
          zenClipOp(zenEnvObj, (float *)output_array, 6.0F,
                    dimensions.batch * dimensions.out_depth *
                        dimensions.out_rows * dimensions.out_cols);
        }
#else
        zenConvolution2DwithBiasRelu(
            input_array, dimensions.batch, dimensions.in_depth,
            dimensions.input_rows, dimensions.input_cols, filter_array,
            dimensions.out_depth, dimensions.filter_rows,
            dimensions.filter_cols, dimensions.pad_rows_before,
            dimensions.pad_cols_before, dimensions.pad_rows_after,
            dimensions.pad_cols_after, dimensions.stride_rows,
            dimensions.stride_cols, bias_add_args.bias_add_data, output_array,
            dimensions.out_rows, dimensions.out_cols);
#endif
        break;
      }
      case FusedComputationType::kBiasAddWithElu:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
        break;
      case FusedComputationType::kBiasAddWithAdd: {
#if NEW_API
        T *bias_arr = const_cast<T *>(bias_add_args.bias_add_data);
        if (is_depthwise) {
          // TODO: Handle this in graph layout pass and fall back to TF-Vanilla
          // for this
          OP_REQUIRES_OK(
              context,
              errors::Internal("DepthWise Fusion with ADD is not supported"));
        } else if (blocked || blockedNHWC) {
          // Direct convolution
          primitive_attr conv_attr;
          //[Configure post-ops]
          float ops_scale = 1.0;
          post_ops ops;
          ops.append_sum(ops_scale);
          conv_attr.set_post_ops(ops);
          //[Configure post-ops]
          zenBlockedConv2DBiasEltSum<T>(
              eng, s, conv_attr, in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, reorder_before,
              reorder_after, cached_filter_data_, context);
        } else {
          // GEMM based convolution
          zenGemmConvolution2D(
              in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, false, false, true,
              nullptr, nullptr, nullptr);
        }
#else
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
#endif
        break;
      }
      case FusedComputationType::kBiasAddWithAddAndRelu: {
#if NEW_API
        T *bias_arr = const_cast<T *>(bias_add_args.bias_add_data);
        if (is_depthwise) {
          // TODO: Handle this in graph layout pass and fall back to TF-Vanilla
          // for this
          OP_REQUIRES_OK(
              context,
              errors::Internal(
                  "DepthWise Fusion with ADD and Relu is not supported"));
        } else if (blocked || blockedNHWC) {
          // Direct convolution
          primitive_attr conv_attr;
          //[Configure post-ops]
          const float ops_scale = 1.f;
          const float ops_alpha = 0.f;  // relu negative slope
          const float ops_beta = 0.f;
          post_ops ops;
          ops.append_sum(ops_scale);
          ops.append_eltwise(ops_scale, algorithm::eltwise_relu, ops_alpha,
                             ops_beta);
          conv_attr.set_post_ops(ops);
          //[Configure post-ops]
          zenBlockedConv2DBiasEltSum<T>(
              eng, s, conv_attr, in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, reorder_before,
              reorder_after, cached_filter_data_, context);
        } else {
          // GEMM based convolution
          zenGemmConvolution2D(
              in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, true, false, true,
              nullptr, nullptr, nullptr);
        }
#else
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
#endif
        break;
      }
      case FusedComputationType::kFusedBatchNorm: {
#if NEW_API
        T *bias_arr = NULL;
        T *batch_norm_mean_data =
            const_cast<T *>(fused_batch_norm_args.estimated_mean_data);
        T *batch_norm_offset_data =
            const_cast<T *>(fused_batch_norm_args.offset_data);
        if (is_depthwise) {
          // TODO: Handle this in graph layout pass and fall back to TF-Vanilla
          // for this
          OP_REQUIRES_OK(
              context, errors::Internal(
                           "DepthWise Fusion with BatchNorm is not supported"));
        } else if (blocked || blockedNHWC) {
          primitive_attr conv_attr;
          zenConvolution2DBatchNormOrRelu(
              eng, s, conv_attr, in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr,
              fused_batch_norm_args.scaling_factor.data(), batch_norm_mean_data,
              batch_norm_offset_data,
              NULL,  // elementwise_input is not required
              output_array, dimensions.out_rows, dimensions.out_cols, false,
              true, reorder_before, reorder_after, cached_filter_data_,
              context);
        } else {
          // GEMM based convolution
          zenGemmConvolution2D(
              in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, false, true, false,
              fused_batch_norm_args.scaling_factor.data(), batch_norm_mean_data,
              batch_norm_offset_data);
        }
#else
        zenConvolution2DwithBatchNorm(
            input_array, dimensions.batch, dimensions.in_depth,
            dimensions.input_rows, dimensions.input_cols, filter_array,
            dimensions.out_depth, dimensions.filter_rows,
            dimensions.filter_cols, dimensions.pad_rows_before,
            dimensions.pad_cols_before, dimensions.pad_rows_after,
            dimensions.pad_cols_after, dimensions.stride_rows,
            dimensions.stride_cols, fused_batch_norm_args.scaling_factor.data(),
            fused_batch_norm_args.estimated_mean_data,
            fused_batch_norm_args.offset_data, output_array,
            dimensions.out_rows, dimensions.out_cols);
#endif
        break;
      }
      case FusedComputationType::kFusedBatchNormWithRelu: {
#if NEW_API
        T *bias_arr = NULL;
        T *batch_norm_mean_data =
            const_cast<T *>(fused_batch_norm_args.estimated_mean_data);
        T *batch_norm_offset_data =
            const_cast<T *>(fused_batch_norm_args.offset_data);
        if (is_depthwise) {
          // TODO: Handle this in graph layout pass and fall back to TF-Vanilla
          // for this
          OP_REQUIRES_OK(
              context,
              errors::Internal(
                  "DepthWise Fusion with BatchNorm and Relu is not supported"));
        } else if (blocked || blockedNHWC) {
          primitive_attr conv_attr;
          zenConvolution2DBatchNormOrRelu(
              eng, s, conv_attr, in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr,
              fused_batch_norm_args.scaling_factor.data(), batch_norm_mean_data,
              batch_norm_offset_data,
              NULL,  // elementwise_input is not required
              output_array, dimensions.out_rows, dimensions.out_cols, true,
              true, reorder_before, reorder_after, cached_filter_data_,
              context);
        } else {
          // GEMM based convolution
          zenGemmConvolution2D(
              in_arr, dimensions.batch, dimensions.in_depth,
              dimensions.input_rows, dimensions.input_cols, filt_arr,
              dimensions.out_depth, dimensions.filter_rows,
              dimensions.filter_cols, dimensions.pad_rows_before,
              dimensions.pad_cols_before, dimensions.pad_rows_after,
              dimensions.pad_cols_after, dimensions.stride_rows,
              dimensions.stride_cols, bias_arr, output_array,
              dimensions.out_rows, dimensions.out_cols, true, true, false,
              fused_batch_norm_args.scaling_factor.data(), batch_norm_mean_data,
              batch_norm_offset_data);
        }
#else
        zenConvolution2DwithBatchNormRelu(
            input_array, dimensions.batch, dimensions.in_depth,
            dimensions.input_rows, dimensions.input_cols, filter_array,
            dimensions.out_depth, dimensions.filter_rows,
            dimensions.filter_cols, dimensions.pad_rows_before,
            dimensions.pad_cols_before, dimensions.pad_rows_after,
            dimensions.pad_cols_after, dimensions.stride_rows,
            dimensions.stride_cols, fused_batch_norm_args.scaling_factor.data(),
            fused_batch_norm_args.estimated_mean_data,
            fused_batch_norm_args.offset_data, output_array,
            dimensions.out_rows, dimensions.out_cols);
#endif
        break;
      }
      case FusedComputationType::kFusedBatchNormWithRelu6:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
        break;
      case FusedComputationType::kFusedBatchNormWithElu:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
        break;
    }
  }
};

template <typename T, bool is_depthwise = false>
class ZenFusedConv2DOp : public OpKernel {
 public:
  explicit ZenFusedConv2DOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("reorder_before", &reorder_before));
    OP_REQUIRES_OK(context, context->GetAttr("reorder_after", &reorder_after));

    OP_REQUIRES_OK(context, context->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(context, context->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(context, context->GetAttr("reset", &reset));
    using FCT = FusedComputationType;

    std::vector<FusedComputationPattern> patterns;
    patterns = {
        {FCT::kBiasAdd, {"BiasAdd"}},
        {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
        {FCT::kBiasAddWithRelu6, {"BiasAdd", "Relu6"}},
        {FCT::kBiasAddWithElu, {"BiasAdd", "Elu"}},
        {FCT::kBiasAddWithAdd, {"BiasAdd", "Add"}},
        {FCT::kBiasAddWithAddAndRelu, {"BiasAdd", "Add", "Relu"}},
        {FCT::kFusedBatchNorm, {"FusedBatchNorm"}},
        {FCT::kFusedBatchNormWithRelu, {"FusedBatchNorm", "Relu"}},
        {FCT::kFusedBatchNormWithRelu6, {"FusedBatchNorm", "Relu6"}},
        {FCT::kFusedBatchNormWithElu, {"FusedBatchNorm", "Elu"}},
    };

    OP_REQUIRES_OK(context, InitializeFusedComputation(
                                context, "_ZenConv2D", patterns,
                                &fused_computation_, &fused_computation_args_));
  }

  void Compute(OpKernelContext *context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor &input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor &filter = context->input(1);
    bool is_float = std::is_same<T, float>::value;

    Conv2DDimensions dimensions;
    OP_REQUIRES_OK(context,
                   ComputeConv2DDimension(params_, input, filter, &dimensions));

    if (is_depthwise) {
      dimensions.out_depth *= dimensions.patch_depth;
    }
    // Update the output type
    zenTensorType out_type =
        (is_float) ? zenTensorType::FLOAT : zenTensorType::BFLOAT16;

    TensorShape out_shape = ShapeFromFormat(
        params_.data_format, dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);

    zendnnEnv zenEnvObj = readEnv();
    Tensor *output = nullptr;
    int zenEnableMemPool = zenEnvObj.zenEnableMemPool &&
                           (context->expected_output_dtype(0) == DT_FLOAT ||
                            context->expected_output_dtype(0) == DT_BFLOAT16);
    ZenMemoryPool<T> *zenPoolBuffer = NULL;

    if ((fused_computation_ == FusedComputationType::kBiasAddWithAdd) ||
        (fused_computation_ == FusedComputationType::kBiasAddWithAddAndRelu)) {
      const Tensor &add_tensor = context->input(3);
      context->set_output(0, add_tensor);
      output = context->mutable_output(0);
      if (zenEnableMemPool) {
        unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
        zenPoolBuffer = ZenMemoryPool<T>::getZenMemPool(threadID);
        if (zenPoolBuffer) {
          const T *output_array = const_cast<T *>(output->flat<T>().data());
          zenPoolBuffer->zenMemPoolUpdateTensorPtrStatus(
              context, (T *)output_array, out_links, reset);
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
              context, &output, out_shape, out_links, reset, out_type);
          if (status) {
            zenEnableMemPool = false;
          }
        } else {
          zenEnableMemPool = false;
        }
      }
      if (!zenEnableMemPool) {
        // Output tensor is of the following dimensions:
        // [ in_batch, out_rows, out_cols, out_depth ]
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, out_shape, &output));
      }
    }

    VLOG(2) << "ZenFusedConv2D: in_depth = " << dimensions.in_depth
            << ", patch_depth = " << dimensions.patch_depth
            << ", input_cols = " << dimensions.input_cols
            << ", filter_cols = " << dimensions.filter_cols
            << ", input_rows = " << dimensions.input_rows
            << ", filter_rows = " << dimensions.filter_rows
            << ", stride_rows = " << dimensions.stride_rows
            << ", stride_cols = " << dimensions.stride_cols
            << ", dilation_rows = " << dimensions.dilation_rows
            << ", dilation_cols = " << dimensions.dilation_cols
            << ", out_depth = " << dimensions.out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    LaunchZenFusedConv2DOp<T>()(context, input, filter, fused_computation_,
                                fused_computation_args_, params_, dimensions,
                                output, reorder_before, reorder_after,
                                &cached_filter_data_, is_depthwise);

    // If ZenMemPool Optimization is enabled(default), update the state of
    //  Memory pool based on input_array address
    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      auto input_map = input.tensor<T, 4>();
      const T *input_array = input_map.data();
      zenPoolBuffer->zenMemPoolFree(context, (void *)input_array);
    }
  }

 private:
  Conv2DParameters params_;
  bool reorder_before, reorder_after, reset;
  int in_links, out_links;
  Tensor cached_filter_data_ TF_GUARDED_BY(mu_);
  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;

  TF_DISALLOW_COPY_AND_ASSIGN(ZenFusedConv2DOp);
};

// Registration of the CPU implementations.
#define REGISTER_ZEN_FUSED_CPU_CONV2D(T)                                 \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("_ZenFusedConv2D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ZenFusedConv2DOp<T, false>);

#define REGISTER_ZEN_FUSED_CPU_DEPTHWISECONV2D(T)                \
  REGISTER_KERNEL_BUILDER(Name("_ZenFusedDepthwiseConv2dNative") \
                              .Device(DEVICE_CPU)                \
                              .TypeConstraint<T>("T"),           \
                          ZenFusedConv2DOp<T, true>);

REGISTER_ZEN_FUSED_CPU_CONV2D(float)
REGISTER_ZEN_FUSED_CPU_CONV2D(::tensorflow::bfloat16)
REGISTER_ZEN_FUSED_CPU_DEPTHWISECONV2D(float)
REGISTER_ZEN_FUSED_CPU_DEPTHWISECONV2D(::tensorflow::bfloat16)

REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedConv2DWithBiasAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint8>("out_type")
        .TypeConstraint<qint32>("Tbias"),
    ZenQuantizedConv2DOp<quint8, qint8, qint32, qint8, qint8, true, false,
                         false, false, false>);

REGISTER_KERNEL_BUILDER(Name("_ZenQuantizedConv2DWithBiasAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<quint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<qint8>("out_type")
                            .TypeConstraint<float>("Tbias"),
                        ZenQuantizedConv2DOp<quint8, qint8, float, qint8, qint8,
                                             true, false, false, false, false>);

REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<quint8>("out_type")
        .TypeConstraint<qint32>("Tbias"),
    ZenQuantizedConv2DOp<quint8, qint8, qint32, quint8, qint8, true, false,
                         true, false, false>);

REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<quint8>("out_type")
        .TypeConstraint<float>("Tbias"),
    ZenQuantizedConv2DOp<quint8, qint8, float, quint8, qint8, true, false, true,
                         false, false>);

REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedConv2DWithBiasSignedSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<quint8>("out_type")
        .TypeConstraint<qint32>("Tbias"),
    ZenQuantizedConv2DOp<quint8, qint8, qint32, quint8, qint8, true, false,
                         true, true, true>);

REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedConv2DWithBiasSignedSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<quint8>("out_type")
        .TypeConstraint<float>("Tbias"),
    ZenQuantizedConv2DOp<quint8, qint8, float, quint8, qint8, true, false, true,
                         true, true>);

REGISTER_KERNEL_BUILDER(Name("_ZenQuantizedConv2DWithBiasAndReluAndRequantize")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<qint8>("Tinput")
                            .TypeConstraint<qint8>("Tfilter")
                            .TypeConstraint<quint8>("out_type")
                            .TypeConstraint<float>("Tbias"),
                        ZenQuantizedConv2DOp<qint8, qint8, float, quint8, qint8,
                                             true, false, true, false, false>);

REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedConv2DWithBiasSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<quint8>("out_type")
        .TypeConstraint<qint32>("Tbias"),
    ZenQuantizedConv2DOp<quint8, qint8, qint32, quint8, quint8, true, false,
                         true, true, false>);

REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedConv2DWithBiasSumAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<quint8>("out_type")
        .TypeConstraint<float>("Tbias"),
    ZenQuantizedConv2DOp<quint8, qint8, float, quint8, quint8, true, false,
                         true, true, false>);

// Tbias -> float
REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<float>("Tbias")
        .TypeConstraint<quint8>("out_type"),
    ZenQuantizedConv2DOp<quint8, qint8, float, quint8, quint8, true, true, true,
                         false, false>);

// Tbias -> qint32
REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize")
        .Device(DEVICE_CPU)
        .TypeConstraint<quint8>("Tinput")
        .TypeConstraint<qint8>("Tfilter")
        .TypeConstraint<qint32>("Tbias")
        .TypeConstraint<quint8>("out_type"),
    ZenQuantizedConv2DOp<quint8, qint8, qint32, quint8, quint8, true, true,
                         true, false, false>);

REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizeV2").Device(DEVICE_CPU).TypeConstraint<quint8>("T"),
    ZenQuantizeV2Op<quint8>);

REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizeV2").Device(DEVICE_CPU).TypeConstraint<qint8>("T"),
    ZenQuantizeV2Op<qint8>);

}  // namespace tensorflow
