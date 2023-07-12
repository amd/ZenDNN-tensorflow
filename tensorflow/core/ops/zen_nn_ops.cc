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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/ops/zen_shape_util.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

template <typename T>
std::vector<int64> AsInt64(const Tensor* tensor, int64 num_elements) {
  std::vector<int64> ret(num_elements);
  auto data = tensor->vec<T>();
  for (int64 i = 0; i < num_elements; ++i) {
    ret[i] = data(i);
  }
  return ret;
}

Status TransposeShapeFn(InferenceContext* c) {
  ShapeHandle input = c->input(0);
  ShapeHandle perm_shape = c->input(1);
  const Tensor* perm = c->input_tensor(1);
  DimensionHandle perm_elems = c->NumElements(perm_shape);
  // If we don't have rank information on the input or value information on
  // perm we can't return any shape information, otherwise we have enough
  // information to at least find the rank of the output.
  if (!c->RankKnown(input) && !c->ValueKnown(perm_elems) && perm == nullptr) {
    c->set_output(0, c->UnknownShape());
    return OkStatus();
  }

  // Find our value of the rank.
  int64 rank;
  if (c->RankKnown(input)) {
    rank = c->Rank(input);
  } else if (c->ValueKnown(perm_elems)) {
    rank = c->Value(perm_elems);
  } else {
    rank = perm->NumElements();
  }
  if (!c->RankKnown(input) && rank < 2) {
    // A permutation array containing a single element is ambiguous. It could
    // indicate either a scalar or a 1-dimensional array, both of which the
    // transpose op returns unchanged.
    c->set_output(0, input);
    return OkStatus();
  }
  std::vector<DimensionHandle> dims;
  dims.resize(rank);
  TF_RETURN_IF_ERROR(c->WithRank(input, rank, &input));
  // Ensure that perm is a vector and has rank elements.
  TF_RETURN_IF_ERROR(c->WithRank(perm_shape, 1, &perm_shape));
  TF_RETURN_IF_ERROR(c->WithValue(perm_elems, rank, &perm_elems));

  // If we know the rank of the input and the value of perm, we can return
  // all shape informantion, otherwise we can only return rank information,
  // but no information for the dimensions.
  if (perm != nullptr) {
    std::vector<int64> data;
    if (perm->dtype() == DT_INT32) {
      data = AsInt64<int32>(perm, rank);
    } else {
      data = AsInt64<int64>(perm, rank);
    }
    for (int32 i = 0; i < rank; ++i) {
      int64 in_idx = data[i];
      if (in_idx >= rank) {
        return errors::InvalidArgument("perm dim ", in_idx,
                                       " is out of range of input rank ", rank);
      }
      dims[i] = c->Dim(input, in_idx);
    }
  } else {
    for (int i = 0; i < rank; ++i) {
      dims[i] = c->UnknownDim();
    }
  }

  c->set_output(0, c->MakeShape(dims));
  return OkStatus();
}

Status SetOutputShapeForReshape(InferenceContext* c) {
  ShapeHandle in = c->input(0);
  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &out));

  if (!c->RankKnown(out)) {
    // We have no information about the shape of the output.
    c->set_output(0, out);
    return OkStatus();
  }
  if (c->RankKnown(in)) {
    // We don't know the number of output elements, but we can try to infer
    // the missing dimension.
    bool too_many_unknown = false;
    int32_t out_unknown_idx = -1;

    DimensionHandle known_out_elems = c->NumElements(out);
    if (!c->ValueKnown(known_out_elems)) {
      known_out_elems = c->MakeDim(1);
      for (int32_t i = 0; i < c->Rank(out); ++i) {
        DimensionHandle dim = c->Dim(out, i);
        if (!c->ValueKnown(dim)) {
          if (out_unknown_idx >= 0) {
            too_many_unknown = true;
            break;
          }
          out_unknown_idx = i;
        } else {
          TF_RETURN_IF_ERROR(
              c->Multiply(known_out_elems, dim, &known_out_elems));
        }
      }
    }
    int32_t in_unknown_idx = -1;
    DimensionHandle known_in_elems = c->NumElements(in);
    if (!c->ValueKnown(known_in_elems)) {
      known_in_elems = c->MakeDim(1);
      for (int32_t i = 0; i < c->Rank(in); ++i) {
        DimensionHandle dim = c->Dim(in, i);
        if (!c->ValueKnown(dim)) {
          if (in_unknown_idx >= 0) {
            too_many_unknown = true;
            break;
          }
          in_unknown_idx = i;
        } else {
          TF_RETURN_IF_ERROR(c->Multiply(known_in_elems, dim, &known_in_elems));
        }
      }
    }

    if (!too_many_unknown) {
      if (in_unknown_idx < 0 && out_unknown_idx < 0) {
        // Just check that the dimensions match.
        if (c->Value(known_in_elems) != c->Value(known_out_elems)) {
          return errors::InvalidArgument(
              "Cannot reshape a tensor with ", c->DebugString(known_in_elems),
              " elements to shape ", c->DebugString(out), " (",
              c->DebugString(known_out_elems), " elements)");
        }
      } else if (in_unknown_idx < 0 && out_unknown_idx >= 0 &&
                 c->Value(known_out_elems) > 0) {
        // Input fully known, infer the one missing output dim
        DimensionHandle inferred_dim;
        TF_RETURN_IF_ERROR(c->Divide(known_in_elems, c->Value(known_out_elems),
                                     true /* evenly_divisible */,
                                     &inferred_dim));
        TF_RETURN_IF_ERROR(
            c->ReplaceDim(out, out_unknown_idx, inferred_dim, &out));

      } else if (in_unknown_idx >= 0 && out_unknown_idx < 0 &&
                 c->Value(known_in_elems) != 0) {
        // Output fully known, infer the one missing input dim
        DimensionHandle inferred_dim;
        TF_RETURN_IF_ERROR(c->Divide(known_out_elems, c->Value(known_in_elems),
                                     true /* evenly_divisible */,
                                     &inferred_dim));
        DimensionHandle unknown_in_dim = c->Dim(in, in_unknown_idx);
        TF_RETURN_IF_ERROR(
            c->Merge(unknown_in_dim, inferred_dim, &unknown_in_dim));
      } else if (in_unknown_idx >= 0 && out_unknown_idx >= 0) {
        // Exactly one unknown dimension in both input and output. These 2 are
        // equal iff the known elements are equal.
        if (c->Value(known_in_elems) == c->Value(known_out_elems)) {
          DimensionHandle unknown_in_dim = c->Dim(in, in_unknown_idx);
          TF_RETURN_IF_ERROR(
              c->ReplaceDim(out, out_unknown_idx, unknown_in_dim, &out));
        }
      }
    }
  }
  c->set_output(0, out);
  return OkStatus();
}

REGISTER_OP("_ZenConv2D")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding);

REGISTER_OP("_ZenDepthwiseConv2dNative")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding);

REGISTER_OP("_ZenFusedConv2D")
    .Input("input: T")
    .Input("filter: T")
    .Input("args: num_args * T")
    .Output("output: T")
    .Attr("T: {float, double, bfloat16}")
    .Attr("num_args: int >= 0")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("fused_ops: list(string) = []")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    // Attributes for the FusedBatchNorm ------------------------------------ //
    .Attr("epsilon: float = 0.0001")
    // Attributes for the LeakyRelu ----------------------------------------- //
    .Attr("leakyrelu_alpha: float = 0.2")
    // ---------------------------------------------------------------------- //
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. ZenDNN graph rewrite pass is
expected to create this operator.
)doc");

REGISTER_OP("_ZenFusedDepthwiseConv2dNative")
    .Input("input: T")
    .Input("filter: T")
    .Input("args: num_args * T")
    .Output("output: T")
    .Attr("T: {half, bfloat16, float, double}")
    .Attr("num_args: int >= 0")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("fused_ops: list(string) = []")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    // Attributes for the FusedBatchNorm ------------------------------------ //
    .Attr("epsilon: float = 0.0001")
    // Attributes for the LeakyRelu ----------------------------------------- //
    .Attr("leakyrelu_alpha: float = 0.2")
    // ---------------------------------------------------------------------- //
    .SetShapeFn(shape_inference::DepthwiseConv2DNativeShape);

REGISTER_OP("_ZenMaxPool")
    .Attr(
        "T: {half, bfloat16, float, double, int32, int64, uint8, int16, int8, "
        "uint16, qint8} = DT_FLOAT")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr("data_format: {'NHWC', 'NCHW', 'NCHW_VECT_C'} = 'NHWC'")
    .Input("input: T")
    .Output("output: T")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::MaxPoolShape);

REGISTER_OP("_ZenAvgPool")
    .Attr(
        "T: {half, bfloat16, float, double, int32, int64, uint8, int16, int8, "
        "uint16, qint8} = DT_FLOAT")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr("data_format: {'NHWC', 'NCHW', 'NCHW_VECT_C'} = 'NHWC'")
    .Input("input: T")
    .Output("output: T")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::AvgPoolShape);

REGISTER_OP("_ZenEinsum")
    .Input("inputs: N * T")
    .Output("output: T")
    .Attr("equation: string")
    .Attr("N: int >= 1")
    .Attr("T: {bfloat16, float}")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int = 1")
    .Attr("reset: bool = 0")
    .SetShapeFn(shape_inference::EinsumShape);

REGISTER_OP("_ZenMatMul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr(
        "T: {bfloat16, half, float, double, int32, int64, complex64, "
        "complex128}")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::MatMulShape);

REGISTER_OP("_ZenBatchMatMul")
    .Input("x: T")
    .Input("y: T")
    .Output("output: T")
    .Attr(
        "T: {bfloat16, half, float, double, int32, int64, complex64, "
        "complex128}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::BatchMatMulShape);

REGISTER_OP("_ZenBatchMatMulV2")
    .Input("x: T")
    .Input("y: T")
    .Output("output: T")
    .Attr(
        "T: {bfloat16, half, float, double, int32, int64, complex64, "
        "complex128}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::BatchMatMulV2Shape);

REGISTER_OP("_ZenFusedBatchMatMulV2")
    .Input("x: T")
    .Input("y: T")
    .Input("args: num_args * T")
    .Output("output: T")
    .Attr(
        "T: {bfloat16, half, float, double, int32, int64, complex64, "
        "complex128}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .Attr("num_args: int >= 0")
    .Attr("fused_ops: list(string) = []")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::BatchMatMulV2Shape)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. Grappler is
expected to create these operators.
)doc");

REGISTER_OP("_ZenSoftmax")
    .Input("logits: T")
    .Output("softmax: T")
    .Attr("data_format: {'N', 'NC', 'TNC', 'NHWC'} = 'NHWC'")
    .Attr("T: {float, double} = DT_FLOAT")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    });

REGISTER_OP("_ZenConcatV2")
    .Input("values: N * T")
    .Input("axis: Tidx")
    .Output("output: T")
    .Attr("N: int >= 2")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .Attr("T: type")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ConcatV2Shape);

REGISTER_OP("_ZenConcat")
    .Input("concat_dim: int32")
    .Input("values: N * T")
    .Output("output: T")
    .Attr("N: int >= 2")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::ConcatShape(c, c->num_inputs() - 1);
    });

REGISTER_OP("_ZenQuantizedConcatV2")
    .Input("values: N * T")
    .Input("axis: Tidx")
    .Input("input_mins:  N * float32")
    .Input("input_maxes: N * float32")
    .Output("output: T")
    .Output("output_min: float")
    .Output("output_max: float")
    .Attr("N: int >= 2")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .Attr("T: type")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      const int n = (c->num_inputs() / 2 - 1) / 3;
      TF_RETURN_IF_ERROR(shape_inference::QuantizedConcatV2Shape(c, n));
      ShapeHandle unused;
      for (int i = n + 1; i < c->num_inputs() / 2; ++i) {
        TF_RETURN_IF_ERROR(c->WithRank(c->input(i), 0, &unused));
      }
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("_ZenFusedMatMul")
    .Input("a: T")
    .Input("b: T")
    .Input("args: num_args * T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {float}")
    .Attr("num_args: int >= 0")
    .Attr("fused_ops: list(string) = []")
    // Attributes for the FusedBatchNorm ----------- //
    .Attr("epsilon: float = 0.0001")
    // Attributes for the LeakyRelu ---------------- //
    .Attr("leakyrelu_alpha: float = 0.2")
    // --------------------------------------------- //
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::MatMulShape)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. Grappler is
expected to create these operators.
)doc");

REGISTER_OP("MatMulBiasAddGelu")
    .Input("a: T")
    .Input("b: T")
    .Input("bias: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {bfloat16, float}")
    .SetShapeFn(shape_inference::MatMulShape)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. Grappler is
expected to create these operators.
)doc");

REGISTER_OP("_ZenMatMulBiasAddGelu")
    .Input("a: T")
    .Input("b: T")
    .Input("bias: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {bfloat16, float}")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::MatMulShape)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. Grappler is
expected to create these operators.
)doc");

REGISTER_OP("_ZenFusedConv2DSum")
    .Input("input: T")
    .Input("filter: T")
    .Input("args: num_args * T")
    .Input("elementwiseinput: T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("num_args: int >= 0")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("fused_ops: list(string) = []")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    // Attributes for the FusedBatchNorm ------------------------------------ //
    .Attr("epsilon: float = 0.0001")
    // ---------------------------------------------------------------------- //
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. ZenDNN graph rewrite pass is
expected to create this operator.
)doc");

REGISTER_OP("_ZenInception")
    .Input("input1: T")
    .Input("filter1: T")
    .Input("args1: num_args *T")
    .Input("input2: T")
    .Input("filter2: T")
    .Input("args2: num_args *T")
    .Input("input3: T")
    .Input("filter3: T")
    .Input("args3: num_args *T")
    .Input("input4: T")
    .Input("filter4: T")
    .Input("args4: num_args *T")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Attr("num_args: int >= 0")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("fused_ops: list(string) = []")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    // Attributes for the FusedBatchNorm ------------------------------------ //
    .Attr("epsilon: float = 0.0001")
    // Attributes for the LeakyRelu ----------------------------------------- //
    .Attr("leakyrelu_alpha: float = 0.2")
    // ---------------------------------------------------------------------- //
    .SetShapeFn(shape_inference::InceptionFourConvsShapeInference)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. ZenDNN graph rewrite pass is
expected to create this operator.
)doc");

REGISTER_OP("_ZenAdd")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {float}")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("_ZenAddV2")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {float}")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)
    .SetIsAggregate()
    .SetIsCommutative();

REGISTER_OP("_ZenSub")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {float}")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("_ZenMul")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {float}")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetIsCommutative()
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("_ZenMaximum")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .Attr("T: {float}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("_ZenSquaredDifference")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {float}")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetIsCommutative()
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("_ZenReshape")
    .Input("tensor: T")
    .Input("shape: Tshape")
    .Output("output: T")
    .Attr("T: type")
    .Attr("Tshape: {int32, int64} = DT_INT32")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(SetOutputShapeForReshape);

REGISTER_OP("_ZenTranspose")
    .Input("x: T")
    .Input("perm: Tperm")
    .Output("y: T")
    .Attr("T: type")
    .Attr("Tperm: {int32, int64} = DT_INT32")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(TransposeShapeFn);

REGISTER_OP("_ZenConjugateTranspose")
    .Input("x: T")
    .Input("perm: Tperm")
    .Output("y: T")
    .Attr("T: type")
    .Attr("Tperm: {int32, int64} = DT_INT32")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(TransposeShapeFn);

REGISTER_OP("_ZenInvertPermutation")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {int32, int64} = DT_INT32")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle x;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &x));
      c->set_output(0, x);
      return OkStatus();
    });

REGISTER_OP("_ZenFusedBatchNorm")
    .Input("x: T")
    .Input("scale: T")
    .Input("offset: T")
    .Input("mean: T")
    .Input("variance: T")
    .Output("y: T")
    .Output("batch_mean: T")
    .Output("batch_variance: T")
    .Output("reserve_space_1: T")
    .Output("reserve_space_2: T")
    .Attr("T: {float}")
    .Attr("epsilon: float = 0.0001")
    .Attr("exponential_avg_factor: float = 1.0")
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("is_training: bool = false")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::FusedBatchNormShape);

REGISTER_OP("_ZenFusedBatchNormV2")
    .Input("x: T")
    .Input("scale: U")
    .Input("offset: U")
    .Input("mean: U")
    .Input("variance: U")
    .Output("y: T")
    .Output("batch_mean: U")
    .Output("batch_variance: U")
    .Output("reserve_space_1: U")
    .Output("reserve_space_2: U")
    .Attr("T: {half, bfloat16, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.0001")
    .Attr("exponential_avg_factor: float = 1.0")
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("is_training: bool = false")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::FusedBatchNormShape);

REGISTER_OP("_ZenFusedBatchNormV3")
    .Input("x: T")
    .Input("scale: U")
    .Input("offset: U")
    .Input("mean: U")
    .Input("variance: U")
    .Output("y: T")
    .Output("batch_mean: U")
    .Output("batch_variance: U")
    .Output("reserve_space_1: U")
    .Output("reserve_space_2: U")
    .Output("reserve_space_3: U")
    .Attr("T: {half, bfloat16, float}")
    .Attr("U: {float}")
    .Attr("epsilon: float = 0.0001")
    .Attr("exponential_avg_factor: float = 1.0")
    .Attr(GetConvnetDataFormat2D3DAttrString())
    .Attr("is_training: bool = false")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::FusedBatchNormV3Shape);

REGISTER_OP("_ZenQuantizedConv2DAndRequantize")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("T: quantizedtype")  // Additional attribute "T" for enabling zenToTf
                               // conversion
    .Attr("out_type: quantizedtype = DT_QINT8")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(4), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("_ZenQuantizedMaxPool")
    .Input("input:           T")
    .Input("min_input:       float")
    .Input("max_input:       float")
    .Output("output:         T")
    .Output("min_output:     float")
    .Output("max_output:     float")
    .Attr("T: quantizedtype")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MaxPoolShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("_ZenQuantizedAvgPool")
    .Input("input:           T")
    .Input("min_input:       float")
    .Input("max_input:       float")
    .Output("output:         T")
    .Output("min_output:     float")
    .Output("max_output:     float")
    .Attr("T: quantizedtype")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::AvgPoolShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("_ZenQuantizedConv2DWithBiasAndRequantize")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("Tbias: {float, qint32}")
    .Attr("T: quantizedtype")  // Additional attribute "T" for
                               // enabling zenToTf conversion
    .Attr("out_type: quantizedtype = DT_QINT8")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(6), 1, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("_ZenQuantizedConv2DAndReluAndRequantize")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("T: quantizedtype")  // Additional attribute "T" for enabling zenToTf
                               // conversion
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(4), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(6), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("_ZenQuantizedConv2DWithBiasAndReluAndRequantize")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("Tbias: {float, qint32}")
    .Attr("T: quantizedtype")  // Additional attribute "T" for
                               // enabling zenToTf conversion
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(6), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("_ZenQuantizedConv2DWithBiasSumAndReluAndRequantize")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Input("summand: Tsummand")
    .Input("min_summand: float")
    .Input("max_summand: float")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("Tbias: {float, qint32}")
    .Attr("Tsummand: quantizedtype")
    .Attr("T: quantizedtype")  // Additional attribute "T" for
                               // enabling zenToTf conversion
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(6), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("_ZenQuantizedConv2DWithBiasSignedSumAndReluAndRequantize")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Input("summand: Tsummand")
    .Input("min_summand: float")
    .Input("max_summand: float")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("Tbias: {float, qint32}")
    .Attr("Tsummand: quantizedtype")
    .Attr("T: quantizedtype")  // Additional attribute "T" for
                               // enabling zenToTf conversion
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("padding_list: list(int) = []")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(6), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("_ZenQuantizeV2")
    .Input("input: float")
    .Input("min_range: float")
    .Input("max_range: float")
    .Output("output: T")
    .Output("output_min: float")
    .Output("output_max: float")
    .Attr("T: quantizedtype")
    .Attr("mode: {'MIN_COMBINED', 'MIN_FIRST', 'SCALED'} = 'MIN_COMBINED'")
    .Attr(
        "round_mode: {'HALF_AWAY_FROM_ZERO', 'HALF_TO_EVEN'} = "
        "'HALF_AWAY_FROM_ZERO'")
    .Attr("narrow_range: bool = false")
    .Attr("axis: int = -1")
    .Attr("ensure_minimum_range: float = 0.01")

    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn([](InferenceContext* c) {
      int axis = -1;
      Status s = c->GetAttr("axis", &axis);
      if (!s.ok() && s.code() != error::NOT_FOUND) {
        return s;
      }
      const int minmax_rank = (axis == -1) ? 0 : 1;

      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle minmax;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), minmax_rank, &minmax));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), minmax_rank, &minmax));
      if (axis != -1) {
        ShapeHandle input;
        TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), axis + 1, &input));
        DimensionHandle depth;
        TF_RETURN_IF_ERROR(
            c->Merge(c->Dim(minmax, 0), c->Dim(input, axis), &depth));
      }
      c->set_output(1, minmax);
      c->set_output(2, minmax);
      return OkStatus();
    });

REGISTER_OP("_ZenQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Input("min_input: float")
    .Input("max_input: float")
    .Input("min_filter: float")
    .Input("max_filter: float")
    .Input("min_freezed_output: float")
    .Input("max_freezed_output: float")
    .Output("output: out_type")
    .Output("min_output: float")
    .Output("max_output: float")
    .Attr("Tinput: quantizedtype")
    .Attr("Tfilter: quantizedtype")
    .Attr("Tbias: {float, qint32}")
    .Attr("T: quantizedtype")
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .Attr("data_format: string = 'NHWC'")
    .Attr("strides: list(int)")
    .Attr("is_filter_const: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::Conv2DShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(5), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(6), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(7), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(8), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return OkStatus();
    });

REGISTER_OP("VitisAIConv2DWithoutBias")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Output("output: Toutput")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int)=[1, 1, 1, 1]")
    .Attr("in_scale: int=0")
    .Attr("weight_scale: int=0")
    .Attr("out_scale: int=0")
    .Attr("Tinput: type")
    .Attr("Tfilter: type")
    .Attr("Toutput: type")
    .Attr("is_relu: bool=false")
    .Attr("relu_alpha: float=0.0")
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding);

REGISTER_OP("VitisAIConv2D")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Output("output: Toutput")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int)=[1, 1, 1, 1]")
    .Attr("in_scale: int=0")
    .Attr("weight_scale: int=0")
    .Attr("out_scale: int=0")
    .Attr("Tinput: type")
    .Attr("Tfilter: type")
    .Attr("Tbias: type")
    .Attr("Toutput: type")
    .Attr("intermediate_float_scale: int=-1")
    .Attr("is_relu: bool=false")
    .Attr("relu_alpha: float=0.0")
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding);

REGISTER_OP("VitisAIConv2DWithSum")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Input("sum_input: Tsum")
    .Output("output: Toutput")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int)=[1, 1, 1, 1]")
    .Attr("in_scale: int=0")
    .Attr("weight_scale: int=0")
    .Attr("sum_scale: int=0")
    .Attr("out_scale: int=0")
    .Attr("add_out_scale: int=0")
    .Attr("Tinput: type")
    .Attr("Tfilter: type")
    .Attr("Tbias: type")
    .Attr("Toutput: type")
    .Attr("Tsum: type")
    .Attr("is_relu: bool=true")
    .Attr("relu_alpha: float=0.0")
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding);

REGISTER_OP("VitisAIDepthwiseConv2D")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Output("output: Toutput")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("in_scale: int=0")
    .Attr("weight_scale: int=0")
    .Attr("out_scale: int=0")
    .Attr("Tinput: type")
    .Attr("Tfilter: type")
    .Attr("Tbias: type")
    .Attr("Toutput: type")
    .Attr("is_relu: bool=true")
    .Attr("relu_alpha: float=0.0")
    .SetShapeFn(shape_inference::DepthwiseConv2DNativeShape);

REGISTER_OP("_FusedVitisAIConv2DWithDepthwise")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Input("dw_filter: Tfilter")
    .Input("dw_bias: Tbias")
    .Output("output: Toutput")
    .Attr("strides: list(int)")
    .Attr("dw_strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetVitisDwPaddingAttrString())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int)=[1, 1, 1, 1]")
    .Attr("in_scale: int=0")
    .Attr("dw_in_scale: int=0")
    .Attr("weight_scale: int=0")
    .Attr("dw_weight_scale: int=0")
    .Attr("out_scale: int=0")
    .Attr("dw_out_scale: int=0")
    .Attr("fused_ops: list(string) = []")
    .Attr("Tinput: type")
    .Attr("Tfilter: type")
    .Attr("Tbias: type")
    .Attr("Toutput: type")
    .Attr("is_relu: bool=false")
    .Attr("dw_is_relu: bool=false")
    .Attr("relu_alpha: float=0.0");

REGISTER_OP("VitisAIAvgPool")
    .Input("input: Tinput")
    .Output("output: Toutput")
    .Attr("Tinput: type")
    .Attr("Toutput: type")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn(shape_inference::AvgPoolShape);

REGISTER_OP("VitisAIMaxPool")
    .Input("input: Tinput")
    .Output("output: Toutput")
    .Attr("Tinput: type")
    .Attr("Toutput: type")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn(shape_inference::MaxPoolShape);

REGISTER_OP("VitisAIConcatV2")
    .Input("values: N * T")
    .Input("axis: Tidx")
    .Output("output: T")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ConcatV2Shape);

REGISTER_OP("VitisAIResize")
    .Input("images: T")
    .Input("multiplier: int32")
    .Output("resized_images: T")
    .Attr("T: {quint8, qint8}")
    .Attr("align_corners: bool = false")
    .Attr("half_pixel_centers: bool = false")
    .Attr("resize_algorithm: {'ResizeNearestNeighbor'}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 1, c->UnknownDim(), &input));
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 2, c->UnknownDim(), &input));
      c->set_output(0, input);
      return OkStatus();
    });

REGISTER_OP("VitisAIAddV2")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {qint8, quint8}")
    .Attr("in_scale_0: int=0")
    .Attr("in_scale_1: int=0")
    .Attr("out_scale: int=0")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)
    .SetIsAggregate();

REGISTER_OP("_ZenVitisAIConv2DWithoutBias")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Output("output: Toutput")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("in_scale: int=0")
    .Attr("weight_scale: int=0")
    .Attr("out_scale: int=0")
    .Attr("Tinput: type")
    .Attr("Tfilter: type")
    .Attr("Toutput: type")
    .Attr("is_relu: bool=false")
    .Attr("relu_alpha: float=0.0")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding);

REGISTER_OP("_ZenVitisAIConv2D")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Output("output: Toutput")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("in_scale: int=0")
    .Attr("weight_scale: int=0")
    .Attr("out_scale: int=0")
    .Attr("Tinput: type")
    .Attr("Tfilter: type")
    .Attr("Tbias: type")
    .Attr("Toutput: type")
    .Attr("intermediate_float_scale: int=-1")
    .Attr("is_relu: bool=false")
    .Attr("relu_alpha: float=0.0")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding);

REGISTER_OP("_ZenVitisAIConv2DWithSum")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Input("sum_input: Tsum")
    .Output("output: Toutput")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("in_scale: int=0")
    .Attr("weight_scale: int=0")
    .Attr("sum_scale: int=0")
    .Attr("out_scale: int=0")
    .Attr("add_out_scale: int=0")
    .Attr("Tinput: type")
    .Attr("Tfilter: type")
    .Attr("Tbias: type")
    .Attr("Toutput: type")
    .Attr("Tsum: type")
    .Attr("is_relu: bool=false")
    .Attr("relu_alpha: float=0.0")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::Conv2DShapeWithExplicitPadding);

REGISTER_OP("_ZenVitisAIDepthwiseConv2D")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Output("output: Toutput")
    .Attr("Tinput: type")
    .Attr("Tfilter: type")
    .Attr("Tbias: type")
    .Attr("Toutput: type")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("in_scale: int=0")
    .Attr("weight_scale: int=0")
    .Attr("out_scale: int=0")
    .Attr("is_relu: bool=true")
    .Attr("relu_alpha: float=0.0")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::DepthwiseConv2DNativeShape);

REGISTER_OP("_ZenFusedVitisAIConv2DWithDepthwise")
    .Input("input: Tinput")
    .Input("filter: Tfilter")
    .Input("bias: Tbias")
    .Input("dw_filter: Tfilter")
    .Input("dw_bias: Tbias")
    .Output("output: Toutput")
    .Attr("strides: list(int)")
    .Attr("dw_strides: list(int)")
    .Attr(GetPaddingAttrStringWithExplicit())
    .Attr(GetVitisDwPaddingAttrString())
    .Attr(GetExplicitPaddingsAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int)=[1, 1, 1, 1]")
    .Attr("in_scale: int=0")
    .Attr("dw_in_scale: int=0")
    .Attr("weight_scale: int=0")
    .Attr("dw_weight_scale: int=0")
    .Attr("out_scale: int=0")
    .Attr("dw_out_scale: int=0")
    .Attr("fused_ops: list(string) = []")
    .Attr("Tinput: type")
    .Attr("Tfilter: type")
    .Attr("Tbias: type")
    .Attr("Toutput: type")
    .Attr("is_relu: bool=false")
    .Attr("dw_is_relu: bool=false")
    .Attr("relu_alpha: float=0.0")
    .Attr("dw_relu_alpha: float=0.0")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::DepthwiseConv2DNativeShape);

REGISTER_OP("_ZenVitisAIMaxPool")
    .Input("input: Tinput")
    .Output("output: Toutput")
    .Attr("Tinput: type")
    .Attr("Toutput: type")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::MaxPoolShape);

REGISTER_OP("_ZenVitisAIAvgPool")
    .Input("input: Tinput")
    .Output("output: Toutput")
    .Attr("Tinput: type")
    .Attr("Toutput: type")
    .Attr("ksize: list(int) >= 4")
    .Attr("strides: list(int) >= 4")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::AvgPoolShape);

REGISTER_OP("_ZenVitisAIConcatV2")
    .Input("values: N * T")
    .Input("axis: Tidx")
    .Output("output: T")
    .Attr("N: int >= 2")
    .Attr("T: type")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::ConcatV2Shape);

  REGISTER_OP("_ZenVitisAIResize")
    .Input("images: T")
    .Input("multiplier: int32")
    .Output("resized_images: T")
    .Attr("T: {quint8, qint8}")
    .Attr("align_corners: bool = false")
    .Attr("half_pixel_centers: bool = false")
    .Attr("resize_algorithm: {'ResizeNearestNeighbor'}")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle input;
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &input));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &unused));
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 1, c->UnknownDim(), &input));
      TF_RETURN_IF_ERROR(c->ReplaceDim(input, 2, c->UnknownDim(), &input));
      c->set_output(0, input);
      return OkStatus();
    });

REGISTER_OP("_ZenVitisAIAddV2")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {qint8, quint8}")
    .Attr("in_scale_0: int=0")
    .Attr("in_scale_1: int=0")
    .Attr("out_scale: int=0")
    .Attr("reorder_before: bool")
    .Attr("reorder_after: bool")
    .Attr("in_links: int")
    .Attr("out_links: int")
    .Attr("reset: bool")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)
    .SetIsAggregate();

}  // namespace tensorflow
