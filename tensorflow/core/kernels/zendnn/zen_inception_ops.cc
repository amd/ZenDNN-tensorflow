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

#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/fused_eigen_output_kernels.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/util/tensor_format.h"
#include "zendnn.hpp"
#include "zendnn_helper.hpp"

// Taken from core/framework/common_shape_fns.cpp
#define TF_REQUIRES(EXP, STATUS)                \
  do {                                          \
    if (!TF_PREDICT_TRUE(EXP)) return (STATUS); \
  } while (false)

using namespace std;
using namespace zendnn;

namespace tensorflow {

template <typename T>
class ZenFusedInceptionOp : public OpKernel {
 private:
  Conv2DParameters
      params_;  // TODO: To be made vector if we handle dissimilar convolutions
  bool reorder_before,
      reorder_after;  // unused currently since only GEMM path is taken
  int num_args, total_inputs_per_conv;
  float epsilon;  // for batchnorm only

  bool reset;
  int in_links, out_links;

  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;

  TF_DISALLOW_COPY_AND_ASSIGN(ZenFusedInceptionOp);
  // Tensors to cache the weight filters in case of BLOCKED formats.
  // Caching the filters avoids unnecessary reorders.
  Tensor cached_filter_data_ptensor_[4] GUARDED_BY(mu_);

  Status GetBiasArray(OpKernelContext *context, int bias_index,
                      float **bias_array) {
    // Bias of the following dimensions: [ output_depth ]
    const Tensor &bias = context->input(bias_index);

    if (bias.dims() != 1)
      return errors::InvalidArgument("bias must be 1-dimensional",
                                     bias.shape().DebugString());

    const auto data_ptr = [](const Tensor &tensor) -> const T * {
      return reinterpret_cast<const T *>(tensor.tensor_data().data());
    };

    auto ptr = data_ptr(bias);
    *bias_array = const_cast<float *>(ptr);

    return Status::OK();
  }

  Status InitFusedBatchNormArgs(OpKernelContext *context, int start_index,
                                float **bn_mean, float **bn_offset,
                                float **scaling_factor) {
    const Tensor &scale = context->input(start_index);
    const Tensor &offset = context->input(start_index + 1);
    const Tensor &estimated_mean = context->input(start_index + 2);
    const Tensor &estimated_variance = context->input(start_index + 3);

    if (scale.dims() != 1)
      return errors::InvalidArgument("scale must be 1-dimensional",
                                     scale.shape().DebugString());
    if (offset.dims() != 1)
      return errors::InvalidArgument("offset must be 1-dimensional",
                                     offset.shape().DebugString());
    if (estimated_mean.dims() != 1)
      return errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                     estimated_mean.shape().DebugString());
    if (estimated_variance.dims() != 1)
      return errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                     estimated_variance.shape().DebugString());

    const auto data_ptr = [](const Tensor &tensor) -> const T * {
      return reinterpret_cast<const T *>(tensor.tensor_data().data());
    };

    auto mean_ptr = data_ptr(estimated_mean);
    *bn_mean = const_cast<float *>(mean_ptr);
    auto offset_ptr = data_ptr(offset);
    *bn_offset = const_cast<float *>(offset_ptr);

    // Precompute scaling factor once for all output blocks (kernels).
    Eigen::Tensor<T, 1, Eigen::RowMajor> scaling_factor_ =
        (estimated_variance.flat<T>() + static_cast<T>(epsilon)).rsqrt() *
        scale.flat<T>();

    *scaling_factor = scaling_factor_.data();

    return Status::OK();
  }
  // This function executes the convolution operators which are inputs to the
  // concat operator. The Conv primitives are created based on the format
  // provided.
  void run_convolution(OpKernelContext *context, zendnn::engine eng,
                       zendnn::stream s, int index, float *output_array,
                       Tensor &cached_filter_data_ptensor_,
                       zendnn::memory::desc &dst_mem_desc,
                       zendnn::memory::format_tag format, bool reorder_before) {
    int input_index = index;
    int filter_index = index + 1;
    int args_index = index + 2;

    const Tensor &input = context->input(input_index);
    auto input_map = input.tensor<float, 4>();
    const float *const_input_array = input_map.data();
    float *input_array = const_cast<float *>(const_input_array);

    const Tensor &filter = context->input(filter_index);
    auto filter_map = filter.tensor<float, 4>();
    const float *const_filter_array = filter_map.data();
    float *filter_array = const_cast<float *>(const_filter_array);

    Conv2DDimensions dimensions;
    ComputeConv2DDimension(params_, input, filter, &dimensions);

    TensorShape out_shape = ShapeFromFormat(
        params_.data_format, dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    float *bias_array = NULL;
    float *scaling_factor = NULL;
    float *bn_mean = NULL;
    float *bn_offset = NULL;
    float *bias_from_bn = NULL;
    bool is_relu_enabled = false;
    bool is_bn_enabled = false;
    using tag = zendnn::memory::format_tag;
    using dt = zendnn::memory::data_type;
    int batch_size = dimensions.batch;
    int channels = dimensions.in_depth;
    int height = dimensions.input_rows;
    int width = dimensions.input_cols;
    int output_channels = dimensions.out_depth;
    int kernel_h = dimensions.filter_rows;
    int kernel_w = dimensions.filter_cols;
    int out_height = dimensions.out_rows;
    int out_width = dimensions.out_cols;
    int stride_h = dimensions.stride_rows;
    int stride_w = dimensions.stride_cols;
    int pad_t = dimensions.pad_rows_before;
    int pad_l = dimensions.pad_cols_before;
    int pad_b = dimensions.pad_rows_after;
    int pad_r = dimensions.pad_cols_after;

    switch (fused_computation_) {
      case FusedComputationType::kUndefined:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type is undefined"));
        break;

      case FusedComputationType::kBiasAdd:
        GetBiasArray(context, args_index, &bias_array);
        break;

      case FusedComputationType::kBiasAddWithRelu:
        GetBiasArray(context, args_index, &bias_array);
        is_relu_enabled = true;
        break;

      case FusedComputationType::kBiasAddWithRelu6:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
        break;
      case FusedComputationType::kBiasAddWithElu:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
        break;

      case FusedComputationType::kFusedBatchNorm:
        // The mean, offsets and scaling factor parameters of the Batchnorm
        // operator are read
        InitFusedBatchNormArgs(context, args_index, &bn_mean, &bn_offset,
                               &scaling_factor);
        // The batchnorm paramertes are converted as bias to Conv operator.
        bias_from_bn = (float *)malloc(sizeof(float) * output_channels);
        for (int r = 0; r < output_channels; r++) {
          bias_from_bn[r] = bn_offset[r] - (scaling_factor[r] * bn_mean[r]);
        }
        bias_array = bias_from_bn;
        break;

      case FusedComputationType::kFusedBatchNormWithRelu:
        // The mean, offsets and scaling factor parameters of the Batchnorm
        // operator are read
        InitFusedBatchNormArgs(context, args_index, &bn_mean, &bn_offset,
                               &scaling_factor);
        // The batchnorm paramertes are converted as bias to Conv operator.
        bias_from_bn = (float *)malloc(sizeof(float) * output_channels);
        for (int r = 0; r < output_channels; r++) {
          bias_from_bn[r] = bn_offset[r] - (scaling_factor[r] * bn_mean[r]);
        }
        bias_array = bias_from_bn;
        is_relu_enabled = true;

        break;
      case FusedComputationType::kFusedBatchNormWithRelu6:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
        break;
      case FusedComputationType::kFusedBatchNormWithElu:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type not supported"));
        break;
    }

    memory::dims conv1_src_tz = {batch_size, channels, height, width};
    memory::dims conv1_weights_tz = {output_channels, channels, kernel_h,
                                     kernel_w};
    memory::dims conv1_bias_tz = {output_channels};
    memory::dims conv1_dst_tz = {batch_size, output_channels, out_height,
                                 out_width};
    memory::dims batch_norm_tz = {output_channels};
    memory::dims conv1_strides = {stride_h, stride_w};

    memory::dims conv1_padding1 = {pad_t, pad_l};
    memory::dims conv1_padding2 = {pad_b, pad_r};

    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    zendnn::memory conv1_user_bias_memory =
        zendnn::memory({{conv1_bias_tz}, dt::f32, tag::x}, eng, bias_array);

    zendnn::memory user_src_memory;
    // If the reorder_before flag has been set, then inputs must be reordered.
    // However in case of format is nhwc, then reordering is not required as the
    // inputs are already in the required format.
    if (format != tag::nhwc && reorder_before) {
      user_src_memory = zendnn::memory({{conv1_src_tz}, dt::f32, tag::nhwc},
                                       eng, input_array);
    } else {
      if (format == zendnn::memory::format_tag::aBcd8b) {
        user_src_memory = zendnn::memory({{conv1_src_tz}, dt::f32, tag::aBcd8b},
                                         eng, input_array);
      } else {
        user_src_memory = zendnn::memory({{conv1_src_tz}, dt::f32, tag::nhwc},
                                         eng, input_array);
      }
    }
    zendnn::memory conv1_dst_memory;
    conv1_dst_memory = zendnn::memory(dst_mem_desc, eng, output_array);
    zendnn::memory::desc conv1_src_md =
        memory::desc({conv1_src_tz}, dt::f32, tag::any);
    zendnn::memory::desc conv1_bias_md =
        memory::desc({conv1_bias_tz}, dt::f32, tag::any);
    zendnn::memory::desc conv1_weights_md =
        memory::desc({conv1_weights_tz}, dt::f32, tag::any);
    zendnn::memory::desc conv1_dst_md = dst_mem_desc;
    convolution_forward::desc *conv1_desc = NULL;

    if (format == zendnn::memory::format_tag::aBcd8b) {
      // Create a convolution descriptor for direct convolution
      conv1_desc = new convolution_forward::desc(
          prop_kind::forward_inference, algorithm::convolution_direct,
          conv1_src_md, conv1_weights_md, conv1_bias_md, conv1_dst_md,
          conv1_strides, conv1_padding1, conv1_padding2);
      if (!bias_array)
        conv1_desc = new convolution_forward::desc(
            prop_kind::forward_inference, algorithm::convolution_direct,
            conv1_src_md, conv1_weights_md, conv1_dst_md, conv1_strides,
            conv1_padding1, conv1_padding2);
    } else if (format == zendnn::memory::format_tag::nhwc) {
      // Create a convolution descriptor for gemm convolution
      conv1_desc = new convolution_forward::desc(
          prop_kind::forward_inference, algorithm::convolution_gemm,
          conv1_src_md, conv1_weights_md, conv1_bias_md, conv1_dst_md,
          conv1_strides, conv1_padding1, conv1_padding2);
      if (!bias_array)
        conv1_desc = new convolution_forward::desc(
            prop_kind::forward_inference, algorithm::convolution_gemm,
            conv1_src_md, conv1_weights_md, conv1_dst_md, conv1_strides,
            conv1_padding1, conv1_padding2);
    }

    // Prepare the postops for the conv primitive
    zendnn::primitive_attr conv_attr;
    if (is_relu_enabled == true) {
      const float ops_scale = 1.f;
      const float ops_alpha = 0.f;  // relu negative slope
      const float ops_beta = 0.f;
      zendnn::post_ops ops;
      ops.append_eltwise(ops_scale, zendnn::algorithm::eltwise_relu, ops_alpha,
                         ops_beta);
      conv_attr.set_post_ops(ops);
    }
    // Create a Conv Primitive Decriptor
    convolution_forward::primitive_desc conv1_prim_desc =
        convolution_forward::primitive_desc(*conv1_desc, conv_attr, eng);
    zendnn::memory conv1_src_memory = user_src_memory;
    // If the conv primitive source descriptor is NOT same the input source
    // descriptor, then reoroder of input takes place
    if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
      conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
      if (format != tag::nhwc && reorder_before) {
        net.push_back(reorder(user_src_memory, conv1_src_memory));
        net_args.push_back({{ZENDNN_ARG_SRC, user_src_memory},
                            {ZENDNN_ARG_DST, conv1_src_memory}});
      }
    }

    const Tensor &cached_filter_data_tensor = cached_filter_data_ptensor_;

    int res = cached_filter_data_tensor.NumElements();
    zendnn::memory conv1_weights_memory;

    // If there are no cached filters use the filter array  to create the
    // weights memory
    if (res <= 0) {
      zendnn::memory user_weights_memory = zendnn::memory(
          {{conv1_weights_tz}, dt::f32, tag::hwcn}, eng, filter_array);
      conv1_weights_memory = user_weights_memory;
      // If the weights' formats are not matching,then reorder happens.
      // This does NOT happen in case of GEMM based convolution,
      // as the weights are expected to be in hwcn format.
      if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
        net.push_back(reorder(user_weights_memory, conv1_weights_memory));
        net_args.push_back({{ZENDNN_ARG_SRC, user_weights_memory},
                            {ZENDNN_ARG_DST, conv1_weights_memory}});
      }
    } else {  // Case where there are cached filters
      void *filter_data = NULL;
      filter_data = static_cast<float *>(
          const_cast<float *>(cached_filter_data_tensor.flat<float>().data()));
      conv1_weights_memory =
          memory(conv1_prim_desc.weights_desc(), eng, filter_data);
    }
    // Create a Conv prmitive and  add it to the list of primitives
    net.push_back(convolution_forward(conv1_prim_desc));
    if (!bias_array)
      net_args.push_back({{ZENDNN_ARG_SRC, conv1_src_memory},
                          {ZENDNN_ARG_WEIGHTS, conv1_weights_memory},
                          {ZENDNN_ARG_DST, conv1_dst_memory}});
    else
      net_args.push_back({{ZENDNN_ARG_SRC, conv1_src_memory},
                          {ZENDNN_ARG_WEIGHTS, conv1_weights_memory},
                          {ZENDNN_ARG_BIAS, conv1_user_bias_memory},
                          {ZENDNN_ARG_DST, conv1_dst_memory}});
    assert(net.size() == net_args.size() && "something is missing");
    // Execute all the primitives added to the list of primitivies
    for (size_t i = 0; i < net.size(); ++i) {
      net.at(i).execute(s, net_args.at(i));
    }
    s.wait();

    // If the filters are NOT cached yet, cache them
    // No need to cache the filters if the format is nhwc
    if (res <= 0 && format != tag::nhwc) {
      TensorShape filter_tf_shape;
      Tensor *filter_tensor_ptr = nullptr;
      filter_tf_shape.AddDim(conv1_weights_memory.get_desc().get_size());
      Tensor *tpr = NULL;
      ((OpKernelContext *)context)
          ->allocate_temp(DT_FLOAT, filter_tf_shape,
                          &cached_filter_data_ptensor_);
      tpr = &cached_filter_data_ptensor_;
      size_t cached_filter_data_size =
          conv1_weights_memory.get_desc().get_size();
      float *weights_data =
          static_cast<float *>(conv1_weights_memory.get_data_handle());
      memcpy(static_cast<float *>(tpr->flat<float>().data()), weights_data,
             cached_filter_data_size);
    }
    if (conv1_desc != NULL) {
      delete conv1_desc;
    }
    if (bias_from_bn != NULL) {
      delete[] bias_from_bn;
    }
  }

  // NOTE: Keeping this code here as it might need modification later
  // for convolutions with dissimilar params
  Status InceptionInitConv2DParameters(const OpKernelConstruction *context,
                                       Conv2DParameters *params) {
    TF_RETURN_IF_ERROR(context->GetAttr("dilations", &params->dilations));
    TF_RETURN_IF_ERROR(context->GetAttr("strides", &params->strides));
    TF_RETURN_IF_ERROR(context->GetAttr("padding", &params->padding));
    if (context->HasAttr("explicit_paddings")) {
      TF_RETURN_IF_ERROR(
          context->GetAttr("explicit_paddings", &params->explicit_paddings));
    }
    string data_format_string;
    TF_RETURN_IF_ERROR(context->GetAttr("data_format", &data_format_string));
    TF_REQUIRES(FormatFromString(data_format_string, &params->data_format),
                errors::InvalidArgument("Invalid data format"));

    const auto &strides = params->strides;
    const auto &dilations = params->dilations;
    const auto &data_format = params->data_format;

    TF_REQUIRES(dilations.size() == 4,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 4 dimensions"));
    TF_REQUIRES(strides.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    const int64 stride_n = GetTensorDim(strides, data_format, 'N');
    const int64 stride_c = GetTensorDim(strides, data_format, 'C');
    const int64 stride_h = GetTensorDim(strides, data_format, 'H');
    const int64 stride_w = GetTensorDim(strides, data_format, 'W');
    TF_REQUIRES(
        stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    TF_REQUIRES(stride_h > 0 && stride_w > 0,
                errors::InvalidArgument(
                    "Row and column strides should be larger than 0."));

    const int64 dilation_n = GetTensorDim(dilations, data_format, 'N');
    const int64 dilation_c = GetTensorDim(dilations, data_format, 'C');
    const int64 dilation_h = GetTensorDim(dilations, data_format, 'H');
    const int64 dilation_w = GetTensorDim(dilations, data_format, 'W');
    TF_REQUIRES(dilation_n == 1 && dilation_c == 1,
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    TF_REQUIRES(
        dilation_h > 0 && dilation_w > 0,
        errors::InvalidArgument("Dilated rates should be larger than 0."));

    TF_RETURN_IF_ERROR(CheckValidPadding(params->padding,
                                         params->explicit_paddings,
                                         /*num_dims=*/4, data_format));

    return Status::OK();
  }

 public:
  explicit ZenFusedInceptionOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, InceptionInitConv2DParameters(context, &params_));
    OP_REQUIRES(
        context, params_.data_format == FORMAT_NHWC,
        errors::Unimplemented("ZenDNN InceptionOp implementation supports "
                              "NHWC tensor format only for now."));
    OP_REQUIRES_OK(context,
                   context->GetAttr("reorder_before", &reorder_before));
    OP_REQUIRES_OK(context, context->GetAttr("reorder_after", &reorder_after));
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    OP_REQUIRES_OK(context, context->GetAttr("num_args", &num_args));
    OP_REQUIRES_OK(context, context->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(context, context->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(context, context->GetAttr("reset", &reset));

    total_inputs_per_conv = num_args + 2;  // 1 data, 1 filter
    // currently ignoring reorder_before and reorder_after since InceptionOp
    // is expected to work only for NHWC
    using FCT = FusedComputationType;

    std::vector<FusedComputationPattern> patterns;
    patterns = {
        {FCT::kBiasAdd, {"BiasAdd"}},
        {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
        {FCT::kBiasAddWithRelu6, {"BiasAdd", "Relu6"}},
        {FCT::kBiasAddWithElu, {"BiasAdd", "Elu"}},
        {FCT::kFusedBatchNorm, {"FusedBatchNorm"}},
        {FCT::kFusedBatchNormWithRelu, {"FusedBatchNorm", "Relu"}},
        {FCT::kFusedBatchNormWithRelu6, {"FusedBatchNorm", "Relu6"}},
        {FCT::kFusedBatchNormWithElu, {"FusedBatchNorm", "Elu"}},
    };

    OP_REQUIRES_OK(context, InitializeFusedComputation(
                                context, "ZenConv2D", patterns,
                                &fused_computation_, &fused_computation_args_));
  }

  TensorShape getOutputShape(OpKernelContext *context) {
    int num_inputs = 4;  // TODO: Hardcoded. Needs to be dynamic
    int64 output_batch_size, output_height, output_width, output_channels;
    output_batch_size = output_height = output_width = output_channels = 0;

    const auto check_or_assign = [](int64 &old_value, const int64 &new_value) {
      if (old_value == 0) {
        old_value = new_value;
      } else {
        assert(old_value == new_value && "Inputs are of unequal dimensions");
      }
    };

    for (int i = 0; i < num_inputs; i++) {
      int input_index = i * total_inputs_per_conv;
      int filter_index = input_index + 1;

      const Tensor &input = context->input(input_index);
      const Tensor &filter = context->input(filter_index);

      Conv2DDimensions dims;
      ComputeConv2DDimension(params_, input, filter, &dims);

      check_or_assign(output_batch_size, dims.batch);
      check_or_assign(output_height, dims.out_rows);
      check_or_assign(output_width, dims.out_cols);

      output_channels += dims.out_depth;
    }

    TensorShape out_shape =
        ShapeFromFormat(params_.data_format, output_batch_size, output_height,
                        output_width, output_channels);
    return out_shape;
  }

  void Compute(OpKernelContext *context) override {
    TensorShape out_shape = getOutputShape(context);
    zendnn::engine cpu_engine_(zendnn::engine::kind::cpu, 0);
    zendnn::stream stream(cpu_engine_);
    int num_inputs = 4;  // TODO: Hardcoded. Needs to be dynamic
    int i = 0;
    // Update the output type
    zenTensorType out_type = zenTensorType::FLOAT;

    const auto check_or_assign = [](int64 &old_value, const int64 &new_value) {
      if (old_value == 0) {
        old_value = new_value;
      } else {
        assert(old_value == new_value && "Inputs are of unequal dimensions");
      }
    };
    zendnnEnv zenEnvObj = readEnv();
    Tensor *output = nullptr;
    int zenEnableMemPool = zenEnvObj.zenEnableMemPool &&
                           context->expected_output_dtype(0) == DT_FLOAT;
    ZenMemoryPool *zenPoolBuffer = NULL;

    // ZenMemPool Optimization reuse o/p tensors from the pool. By default
    //  its enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory
    //  pool optimization
    //  Cases where tensors in pool are not free or requested size is more
    //  than available tensor size in Pool, control will fall back to
    //  default way of allocation i.e. with allocate_output(..)
    if (zenEnableMemPool) {
      unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
      zenPoolBuffer = ZenMemoryPool::getZenMemPool(threadID);
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
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    }

    auto output_map = output->tensor<float, 4>();
    float *output_array = const_cast<float *>(output_map.data());

    int64 output_batch_size, output_height, output_width, output_channels;
    output_batch_size = output_height = output_width = output_channels = 0;
    std::vector<zendnn::primitive> prims;
    std::vector<zendnn::memory::desc> concat_parent_mem_desc;
    bool is_block_format = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT1;
    bool blockedNHWC = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT2;

    zendnn::memory::format_tag op_format = zendnn::memory::format_tag::any;
    if (is_block_format == 1) {
      op_format = zendnn::memory::format_tag::aBcd8b;
    } else {
      op_format = zendnn::memory::format_tag::nhwc;
    }
    for (i = 0; i < num_inputs; i++) {
      // TODO: slightly roundabout way. Can be made better.
      int input_index = i * total_inputs_per_conv;
      int filter_index = input_index + 1;

      const Tensor &input = context->input(input_index);
      const Tensor &filter = context->input(filter_index);

      Conv2DDimensions dims;
      ComputeConv2DDimension(params_, input, filter, &dims);
      check_or_assign(output_batch_size, dims.batch);
      check_or_assign(output_height, dims.out_rows);
      check_or_assign(output_width, dims.out_cols);

      zendnn::memory::dims src_dims = {dims.batch, dims.out_depth,
                                       dims.out_rows, dims.out_cols};
      zendnn::memory::desc src_desc(src_dims, zendnn::memory::data_type::f32,
                                    op_format);
      concat_parent_mem_desc.push_back(src_desc);
      output_channels += dims.out_depth;
    }

    // Dims for the concat op
    zendnn::memory::dims concat_op_dims = {output_batch_size, output_channels,
                                           output_height, output_width};
    // Mem Desc for the concat op
    zendnn::memory::desc concat_op_mem_desc(
        concat_op_dims, zendnn::memory::data_type::f32, op_format);
    zendnn::memory *concat_op_mem = NULL;
    float *output_array1;
    // In case reorder_after is set to true, the output must be reordered.
    // However in case of nhwc format, the output need not to be reordered
    // as the output is already in the required format.
    // So the required input and output memory objects are prepared for
    // reorder operation
    if (op_format != zendnn::memory::format_tag::nhwc && reorder_after) {
      concat_op_mem = new zendnn::memory(concat_op_mem_desc, cpu_engine_);
      output_array1 = (float *)concat_op_mem->get_data_handle();
    } else {
      output_array1 = output_array;
    }

    int offset = 0;
    int tmp_offset1 = 0;
    int axis = 1;
    zendnn::memory::dims ldims;
    zendnn::memory::dims loffsets;

    for (int n = 0; n < num_inputs; n++) {
      int index = n * total_inputs_per_conv;
      for (unsigned long int i = 0; i < concat_parent_mem_desc[n].dims().size();
           ++i) {
        ldims.push_back(concat_parent_mem_desc[n].dims()[i]);
        if (i == (unsigned int)axis) {
          loffsets.push_back(tmp_offset1);
        } else {
          loffsets.push_back(0);
        }
      }
      tmp_offset1 += ldims[axis];
      // Create output buffer  memory descriptor for the inputs of the concat
      // operator Incase Concat has 4 conv operators as inputs, then memory
      // descriptors for outputs of convolution operators are created using the
      // submemory_desc API. This api creates the memory descriptors with
      // appropriate offsets and strides set.
      zendnn::memory::desc ldesc =
          concat_op_mem_desc.submemory_desc(ldims, loffsets);
      ldims.clear();
      loffsets.clear();
      concat_parent_mem_desc[n] = ldesc;
      run_convolution(context, cpu_engine_, stream, index, output_array1,
                      cached_filter_data_ptensor_[n], concat_parent_mem_desc[n],
                      op_format, reorder_before);

      // If ZenMemPool Optimization is enabled(default), update the state of
      //  Memory pool based on input_array address
      const Tensor &input = context->input(index);
      if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
        auto input_map = input.tensor<float, 4>();
        const float *const_input_array = input_map.data();
        float *input_array = const_cast<float *>(const_input_array);

        zenPoolBuffer->zenMemPoolFree(context, (void *)input_array);
      }
    }
    // In case reorder_after is set to true, the output must be reordered.
    // However in case of nhwc format, the output need not to be reordered
    if (op_format != zendnn::memory::format_tag::nhwc && reorder_after) {
      zendnn::memory tmp_memory(concat_op_mem_desc, cpu_engine_, output_array);
      reorder(*concat_op_mem, tmp_memory)
          .execute(cpu_engine_, *concat_op_mem, tmp_memory);
    }
    if (concat_op_mem != NULL) {
      delete concat_op_mem;
    }
    return;
  }
};

// Registration of the CPU implementations.
#define REGISTER_ZEN_FUSED_CPU_INCEPTION(T)                            \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("_ZenInception").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ZenFusedInceptionOp<T>);

REGISTER_ZEN_FUSED_CPU_INCEPTION(float)

}  // namespace tensorflow
