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

#include <iostream>

#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace shape_inference {

Status ZenGetWindowedOutputSizeFromDimsV2(
    shape_inference::InferenceContext *c,
    shape_inference::DimensionHandle input_size,
    shape_inference::DimensionOrConstant filter_size, int64 dilation_rate,
    int64 stride, Padding padding_type, int64 padding_before,
    int64 padding_after, shape_inference::DimensionHandle *output_size);

Status ZenShapeFromDimensions(DimensionHandle batch_dim,
                              gtl::ArraySlice<DimensionHandle> spatial_dims,
                              DimensionHandle filter_dim, TensorFormat format,
                              InferenceContext *context, ShapeHandle *shape);

Status ZenDimensionsFromShape(
    ShapeHandle shape, TensorFormat format, DimensionHandle *batch_dim,
    gtl::MutableArraySlice<DimensionHandle> spatial_dims,
    DimensionHandle *filter_dim, InferenceContext *context);

Status ZenCheckFormatConstraintsOnShape(const TensorFormat tensor_format,
                                        const ShapeHandle shape_handle,
                                        const string &tensor_name,
                                        shape_inference::InferenceContext *c);

Status InceptionConv2DShapeImpl(shape_inference::InferenceContext *c,
                                bool supports_explicit_padding, int index,
                                ShapeHandle *result);

Status InceptionFourConvsShapeInference(shape_inference::InferenceContext *c);

}  // namespace shape_inference
}  // namespace tensorflow
