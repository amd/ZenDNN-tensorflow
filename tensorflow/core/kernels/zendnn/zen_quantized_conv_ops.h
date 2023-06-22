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

#ifndef TENSORFLOW_CORE_KERNELS_ZEN_QUANTIZED_CONV_OPS_H_
#define TENSORFLOW_CORE_KERNELS_ZEN_QUANTIZED_CONV_OPS_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/zendnn/zen_conv_ops_util.h"
#include "tensorflow/core/kernels/zendnn/zen_quantized_ops.h"

using namespace zendnn;

namespace tensorflow {

/**
 * @brief Paramerters that is used for all standard VitisAIConv2D node
 *
 */
struct VitisAIConv2DParameters {
  // standard conv2d params
  std::vector<int32> dilations;
  std::vector<int32> strides;
  Padding padding;
  TensorFormat data_format;
  std::vector<int64_t> explicit_paddings;

  // vitisai scale params
  int input_scale = 0;
  int filter_scale = 0;
  int bias_scale = 0;
  int output_scale = 0;
  int sum_scale = 0;
  int sum_out_scale = 0;
  // not implemented in conv as of now
  int intermediate_float_scale = 0;

  // relu fused params
  bool is_relu = false;
  float relu_alpha = 0.0f;

  // Paramerters that are used VitisAIConv2DWithDepthwise node
  struct VitisAIConv2DWithDepthwiseParameters {
    // standard conv2d params
    std::vector<int32> strides;
    Padding padding;

    // vitisai scale params
    int input_scale = 0;
    int filter_scale = 0;
    int bias_scale = 0;
    int output_scale = 0;

    // relu fused params
    bool is_relu = false;
    float relu_alpha = 0.0f;
  };
  VitisAIConv2DWithDepthwiseParameters dw_params;
};

/**
 * @brief Dimensions that is used for all standard VitisAIConv2D node
 *
 */
struct VitisAIConv2DDimensions : Conv2DDimensions {
  struct DWConv2DDimension {
    int filter_rows;
    int filter_cols;

    int stride_rows;
    int stride_cols;

    // The final output sizes might be different from original convolution size
    int64_t out_rows;
    int64_t out_cols;

    // Depthwise post-op only uses padding_l
    int64_t pad_cols_before;
    int64_t pad_rows_before;
  };
  DWConv2DDimension dw_dimensions;
};

/**
 * @brief Initializes and validates VitisAI Convolution parameters configured by
 * OpKernel attributes.
 *
 * @param context Context from which prameters are read
 * @param params Write the parameters to it
 * @return Status
 */
Status InitVitisAIConv2DParameters(const OpKernelConstruction* context,
                                   VitisAIConv2DParameters* params);

/**
 * @brief Compute and validate all sizes required for VitisAI Convolution
 *
 * @param dimensions Store all the dimensions
 * @param params Parameters for convolution
 * @param input Input tensor
 * @param filter Filter tensor
 * @param is_depthwise If convolution is depthwise
 * @return Status
 */
Status ComputeVitisAIConv2DDimensions(VitisAIConv2DDimensions* dimensions,
                                      const VitisAIConv2DParameters& params,
                                      const Tensor& input, const Tensor& filter,
                                      bool is_depthwise);

/**
 * @brief Compute and validate all sizes required for VitisAI Convolution fused
 * with VitisAI Depthwise Convolution
 *
 * @param dimensions Store dimensions for convolution
 * @param params Parameters for convolution
 * @param input Input tensor
 * @param filter Filter tensor
 * @param dw_filter Depthwise convolution filter tensor
 * @return Status
 */
Status ComputeVitisAIConv2DDimensions(VitisAIConv2DDimensions* dimensions,
                                      const VitisAIConv2DParameters& params,
                                      const Tensor& input, const Tensor& filter,
                                      const Tensor& dw_filter);

/**
 * @brief Sets all the required conv attributes. Includes:
 * 1. Sum post op, with appropirate scaling
 * 2. ReLU/ReLU6/LeakyRelu post op with appropritate alpha
 * 3. Depthwise Post-Op
 * 4. Output scales for INT8 kernels
 *
 * @tparam Toutput Output dtype
 * @tparam Tsum Sum dtype (void if sum not present)
 * @param conv_attr Attributes to store the post-ops and scales
 * @param params Parameters for convolution
 * @param dimensions Dimensions for convolution
 * @param is_fused_depthwise Is VitisAI Depthwise Convolution fused
 * @param is_conv_fp32 Is Convolution FP32
 * @return Status
 */
template <typename Toutput, typename Tsum>
Status SetVitisAIConv2DAttributes(zendnn::primitive_attr* conv_attr,
                                  const VitisAIConv2DParameters& params,
                                  const VitisAIConv2DDimensions dimensions,
                                  bool is_fused_depthwise, bool is_conv_fp32);

/**
 * @brief Returns the data type in zendnn memory data type format for inputs. If
 * Toutput and Tinput are float, then it's a fp32 conv, otherwise change data
 * type of the input from float to s8
 *
 * @tparam Tinput Input dtype
 * @tparam Toutput Output Dtype
 * @return zendnn::memory::data_type
 */
template <typename Tinput, typename Toutput>
inline zendnn::memory::data_type DataTypetoZenForInput() {
  if (std::is_same<Tinput, float>::value &&
      !std::is_same<Tinput, Toutput>::value) {
    return DataTypetoZen<qint8>();
  } else {
    return DataTypetoZen<Tinput>();
  }
}

/**
 * @brief Returns the data type in zendnn memory data type format for output
 * For specific layers, where Add is a post-op, Tsum is
 * output data type
 *
 * @tparam Toutput Output dtype
 * @tparam Tsum Sum dtype
 * @return zendnn::memory::data_type
 */
template <typename Toutput, typename Tsum>
inline zendnn::memory::data_type DataTypetoZenForOutput() {
  if (!std::is_same<Tsum, void>::value) {
    return DataTypetoZen<Tsum>();
  } else {
    return DataTypetoZen<Toutput>();
  }
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_ZEN_QUANTIZED_CONV_OPS_H_
