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

#include "tensorflow/core/ops/zen_shape_util.h"

#include <iostream>

#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace shape_inference {

// The V2 version computes windowed output size with arbitrary dilation_rate,
// while the original version only handles the cases where dilation_rates equal
// to 1.
Status ZenGetWindowedOutputSizeFromDimsV2(
    shape_inference::InferenceContext *c,
    shape_inference::DimensionHandle input_size,
    shape_inference::DimensionOrConstant filter_size, int64 dilation_rate,
    int64 stride, Padding padding_type, int64 padding_before,
    int64 padding_after, shape_inference::DimensionHandle *output_size) {
  if (stride <= 0) {
    return errors::InvalidArgument("Stride must be > 0, but got ", stride);
  }

  if (dilation_rate < 1) {
    return errors::InvalidArgument("Dilation rate must be >= 1, but got ",
                                   dilation_rate);
  }

  // See also the parallel implementation in GetWindowedOutputSizeVerbose.
  switch (padding_type) {
    case Padding::VALID:
      padding_before = padding_after = 0;
      TF_FALLTHROUGH_INTENDED;
    case Padding::EXPLICIT:
      TF_RETURN_IF_ERROR(
          c->Add(input_size, padding_before + padding_after, &input_size));
      if (dilation_rate > 1) {
        DimensionHandle window_size;
        TF_RETURN_IF_ERROR(
            c->Subtract(c->MakeDim(filter_size), 1, &window_size));
        TF_RETURN_IF_ERROR(
            c->Multiply(window_size, dilation_rate, &window_size));
        TF_RETURN_IF_ERROR(c->Add(window_size, 1, &window_size));
        TF_RETURN_IF_ERROR(c->Subtract(input_size, window_size, output_size));
      } else {
        TF_RETURN_IF_ERROR(c->Subtract(input_size, filter_size, output_size));
      }
      TF_RETURN_IF_ERROR(c->Add(*output_size, stride, output_size));
      TF_RETURN_IF_ERROR(c->Divide(*output_size, stride,
                                   /*evenly_divisible=*/false, output_size));
      break;
    case Padding::SAME:
      TF_RETURN_IF_ERROR(c->Add(input_size, stride - 1, output_size));
      TF_RETURN_IF_ERROR(c->Divide(*output_size, stride,
                                   /*evenly_divisible=*/false, output_size));
      break;
  }
  return OkStatus();
}

Status ZenShapeFromDimensions(DimensionHandle batch_dim,
                              gtl::ArraySlice<DimensionHandle> spatial_dims,
                              DimensionHandle filter_dim, TensorFormat format,
                              InferenceContext *context, ShapeHandle *shape) {
  const int32 rank = GetTensorDimsFromSpatialDims(spatial_dims.size(), format);
  std::vector<DimensionHandle> out_dims(rank);

  // Batch.
  out_dims[tensorflow::GetTensorBatchDimIndex(rank, format)] = batch_dim;
  // Spatial.
  for (int spatial_dim_index = 0; spatial_dim_index < spatial_dims.size();
       ++spatial_dim_index) {
    out_dims[tensorflow::GetTensorSpatialDimIndex(
        rank, format, spatial_dim_index)] = spatial_dims[spatial_dim_index];
  }
  // Channel.
  if (format == tensorflow::FORMAT_NCHW_VECT_C) {
    // When format is NCHW_VECT_C, factor the feature map count
    // into the outer feature count and the inner feature count (=4).
    TF_RETURN_IF_ERROR(context->Divide(
        filter_dim, 4, /*evenly_divisible=*/true,
        &out_dims[tensorflow::GetTensorFeatureDimIndex(rank, format)]));
    out_dims[GetTensorInnerFeatureDimIndex(rank, format)] = context->MakeDim(4);
  } else {
    out_dims[tensorflow::GetTensorFeatureDimIndex(rank, format)] = filter_dim;
  }

  *shape = context->MakeShape(out_dims);
  return tensorflow::OkStatus();
}

Status ZenDimensionsFromShape(
    ShapeHandle shape, TensorFormat format, DimensionHandle *batch_dim,
    gtl::MutableArraySlice<DimensionHandle> spatial_dims,
    DimensionHandle *filter_dim, InferenceContext *context) {
  const int32 rank = GetTensorDimsFromSpatialDims(spatial_dims.size(), format);
  // Batch.
  *batch_dim = context->Dim(shape, GetTensorBatchDimIndex(rank, format));
  // Spatial.
  for (int spatial_dim_index = 0; spatial_dim_index < spatial_dims.size();
       ++spatial_dim_index) {
    spatial_dims[spatial_dim_index] = context->Dim(
        shape, GetTensorSpatialDimIndex(rank, format, spatial_dim_index));
  }
  // Channel.
  *filter_dim = context->Dim(shape, GetTensorFeatureDimIndex(rank, format));
  if (format == FORMAT_NCHW_VECT_C) {
    TF_RETURN_IF_ERROR(context->Multiply(
        *filter_dim,
        context->Dim(shape, GetTensorInnerFeatureDimIndex(rank, format)),
        filter_dim));
  }
  return OkStatus();
}

Status ZenCheckFormatConstraintsOnShape(const TensorFormat tensor_format,
                                        const ShapeHandle shape_handle,
                                        const string &tensor_name,
                                        shape_inference::InferenceContext *c) {
  if (tensor_format == FORMAT_NCHW_VECT_C) {
    // Check that the vect dim has size 4.
    const int num_dims = c->Rank(shape_handle);
    DimensionHandle vect_dim = c->Dim(
        shape_handle, GetTensorInnerFeatureDimIndex(num_dims, tensor_format));
    DimensionHandle unused_vect_dim;
    TF_RETURN_IF_ERROR(c->WithValue(vect_dim, 4, &unused_vect_dim));
  }

  return OkStatus();
}

Status InceptionConv2DShapeImpl(shape_inference::InferenceContext *c,
                                bool supports_explicit_padding, int index,
                                ShapeHandle *result) {
  string data_format_str, filter_format_str;
  if (!c->GetAttr("data_format", &data_format_str).ok()) {
    data_format_str = "NHWC";
  }
  if (!c->GetAttr("filter_format", &filter_format_str).ok()) {
    filter_format_str = "HWIO";
  }

  TensorFormat data_format;
  if (!FormatFromString(data_format_str, &data_format)) {
    return errors::InvalidArgument("Invalid data format string: ",
                                   data_format_str);
  }
  FilterTensorFormat filter_format;
  if (!FilterFormatFromString(filter_format_str, &filter_format)) {
    return errors::InvalidArgument("Invalid filter format string: ",
                                   filter_format_str);
  }

  int conv_input_index = index;
  int filter_input_index = index + 1;

  constexpr int num_spatial_dims = 2;
  const int rank = GetTensorDimsFromSpatialDims(num_spatial_dims, data_format);
  ShapeHandle conv_input_shape;
  TF_RETURN_IF_ERROR(
      c->WithRank(c->input(conv_input_index), rank, &conv_input_shape));
  TF_RETURN_IF_ERROR(ZenCheckFormatConstraintsOnShape(
      data_format, conv_input_shape, "conv_input", c));

  // The filter rank should match the input (4 for NCHW, 5 for NCHW_VECT_C).
  ShapeHandle filter_shape;
  TF_RETURN_IF_ERROR(
      c->WithRank(c->input(filter_input_index), rank, &filter_shape));
  TF_RETURN_IF_ERROR(
      ZenCheckFormatConstraintsOnShape(data_format, filter_shape, "filter", c));

  std::vector<int32> dilations;
  TF_RETURN_IF_ERROR(c->GetAttr("dilations", &dilations));

  if (dilations.size() != 4) {
    return errors::InvalidArgument(
        "Conv2D requires the dilation attribute to contain 4 values, but got: ",
        dilations.size());
  }

  std::vector<int32> strides;
  TF_RETURN_IF_ERROR(c->GetAttr("strides", &strides));

  // strides.size() should be 4 (NCHW) even if the input is 5 (NCHW_VECT_C).
  if (strides.size() != 4) {
    return errors::InvalidArgument("Conv2D on data format ", data_format_str,
                                   " requires the stride attribute to contain"
                                   " 4 values, but got: ",
                                   strides.size());
  }

  const int32 stride_rows = GetTensorDim(strides, data_format, 'H');
  const int32 stride_cols = GetTensorDim(strides, data_format, 'W');
  const int32 dilation_rows = GetTensorDim(dilations, data_format, 'H');
  const int32 dilation_cols = GetTensorDim(dilations, data_format, 'W');

  DimensionHandle batch_size_dim;
  DimensionHandle input_depth_dim;
  gtl::InlinedVector<DimensionHandle, 2> input_spatial_dims(2);
  TF_RETURN_IF_ERROR(ZenDimensionsFromShape(
      conv_input_shape, data_format, &batch_size_dim,
      absl::MakeSpan(input_spatial_dims), &input_depth_dim, c));

  DimensionHandle output_depth_dim = c->Dim(
      filter_shape, GetFilterDimIndex<num_spatial_dims>(filter_format, 'O'));
  DimensionHandle filter_rows_dim = c->Dim(
      filter_shape, GetFilterDimIndex<num_spatial_dims>(filter_format, 'H'));
  DimensionHandle filter_cols_dim = c->Dim(
      filter_shape, GetFilterDimIndex<num_spatial_dims>(filter_format, 'W'));
  DimensionHandle filter_input_depth_dim;
  if (filter_format == FORMAT_OIHW_VECT_I) {
    TF_RETURN_IF_ERROR(c->Multiply(
        c->Dim(filter_shape,
               GetFilterDimIndex<num_spatial_dims>(filter_format, 'I')),
        c->Dim(filter_shape,
               GetFilterTensorInnerInputChannelsDimIndex(rank, filter_format)),
        &filter_input_depth_dim));
  } else {
    filter_input_depth_dim = c->Dim(
        filter_shape, GetFilterDimIndex<num_spatial_dims>(filter_format, 'I'));
  }

  // Check that the input tensor and the filter tensor agree on the channel
  // count.
  if (c->ValueKnown(input_depth_dim) && c->ValueKnown(filter_input_depth_dim)) {
    int64 input_depth_value = c->Value(input_depth_dim),
          filter_input_depth_value = c->Value(filter_input_depth_dim);
    if (input_depth_value % filter_input_depth_value != 0)
      return errors::InvalidArgument(
          "Depth of input (", input_depth_value,
          ") is not a multiple of input depth of filter (",
          filter_input_depth_value, ")");
    if (input_depth_value != filter_input_depth_value) {
      int64 num_groups = input_depth_value / filter_input_depth_value;
      if (c->ValueKnown(output_depth_dim)) {
        int64 output_depth_value = c->Value(output_depth_dim);
        if (output_depth_value % num_groups != 0)
          return errors::InvalidArgument(
              "Depth of output (", output_depth_value,
              ") is not a multiple of the number of groups (", num_groups, ")");
      }
    }
  }

  Padding padding;
  TF_RETURN_IF_ERROR(c->GetAttr("padding", &padding));

  std::vector<int64> explicit_paddings;
  if (supports_explicit_padding) {
    Status s = c->GetAttr("explicit_paddings", &explicit_paddings);
    // Use the default value, which is an empty list, if the attribute is not
    // found. Otherwise return the error to the caller.
    if (!s.ok() && !errors::IsNotFound(s)) {
      return s;
    }
    TF_RETURN_IF_ERROR(CheckValidPadding(padding, explicit_paddings,
                                         /*num_dims=*/4, data_format));
  } else {
    CHECK(padding != Padding::EXPLICIT);  // Crash ok.
  }

  DimensionHandle output_rows, output_cols;
  int64 pad_rows_before = -1, pad_rows_after = -1;
  int64 pad_cols_before = -1, pad_cols_after = -1;
  if (padding == Padding::EXPLICIT) {
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'H',
                             &pad_rows_before, &pad_rows_after);
    GetExplicitPaddingForDim(explicit_paddings, data_format, 'W',
                             &pad_cols_before, &pad_cols_after);
  }
  TF_RETURN_IF_ERROR(ZenGetWindowedOutputSizeFromDimsV2(
      c, input_spatial_dims[0], filter_rows_dim, dilation_rows, stride_rows,
      padding, pad_rows_before, pad_rows_after, &output_rows));
  TF_RETURN_IF_ERROR(ZenGetWindowedOutputSizeFromDimsV2(
      c, input_spatial_dims[1], filter_cols_dim, dilation_cols, stride_cols,
      padding, pad_cols_before, pad_cols_after, &output_cols));

  ShapeHandle output_shape;
  TF_RETURN_IF_ERROR(
      ZenShapeFromDimensions(batch_size_dim, {output_rows, output_cols},
                             output_depth_dim, data_format, c, &output_shape));
  *result = output_shape;
  return OkStatus();
}

Status InceptionFourConvsShapeInference(shape_inference::InferenceContext *c) {
  int num_args;
  c->GetAttr("num_args", &num_args);
  int total_inputs_per_conv = num_args + 2;  // 1 data, 1 filter

  int axis = 3;
  string data_format_str;
  if (!c->GetAttr("data_format", &data_format_str).ok()) {
    data_format_str = "NHWC";
  }
  if (data_format_str == "NCHW") {
    axis = 1;  // TODO: no kernel support for this currently
  }

  int num_inputs = 4;
  ShapeHandle shape_;
  InceptionConv2DShapeImpl(c, true, 0, &shape_);  // get output shape of first
  std::vector<DimensionHandle> dims = {c->Dim(shape_, 0), c->Dim(shape_, 1),
                                       c->Dim(shape_, 2), c->Dim(shape_, 3)};
  for (int i = 1; i < num_inputs; i++) {
    int index = i * total_inputs_per_conv;
    InceptionConv2DShapeImpl(c, true, index, &shape_);
    c->Add(dims[axis], c->Dim(shape_, axis), &dims[axis]);
  }

  ShapeHandle output_shape = c->MakeShape(dims);
  c->set_output(0, output_shape);
  return OkStatus();
}

}  // namespace shape_inference
}  // namespace tensorflow
