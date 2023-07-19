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

#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/fused_eigen_output_kernels.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/util/tensor_format.h"

using namespace std;

#define NEW_API 1
#if NEW_API
#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "zendnn.hpp"
#include "zendnn_helper.hpp"
using namespace zendnn;
using namespace std;
void zenConvolution2DBatchNormOrRelu(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void *input_array, int batch_size, int channels, int height, int width,
    void *filter_array, int output_channels, int kernel_h, int kernel_w,
    float pad_t, float pad_l, float pad_b, float pad_r, int stride_h,
    int stride_w, void *bias_array, void *batch_norm_scale,
    void *batch_norm_mean, void *batch_norm_offset, void *elementwise_input,
    void *output_array, int out_height, int out_width, bool reluFused,
    bool batchNormFused, bool reorder_before, bool reorder_after,
    void *cached_filter_data_, void *context, const float alpha = 0.0f);
#else
#include "zendnn_helper.hpp"
#endif

namespace tensorflow {

template <typename T>
struct LaunchZenFusedConv2DSumOp {
  void operator()(OpKernelContext *context, const Tensor &input,
                  const Tensor &filter, const Tensor &dinput,
                  const FusedComputationType fusion,
                  const FusedComputationArgs &fusion_args,
                  const Conv2DParameters &params,
                  const Conv2DDimensions &dimensions, Tensor *output,
                  bool reorder_before, bool reorder_after,
                  Tensor *cached_filter_data_) {
    OP_REQUIRES(context, dimensions.in_depth == filter.dim_size(2),
                errors::Unimplemented("Fused conv implementation does not "
                                      "support grouped convolutions for now."));
    OP_REQUIRES(context, params.data_format == FORMAT_NHWC,
                errors::Unimplemented("Fused conv implementation only supports "
                                      "NHWC tensor format for now."));

    FusedBatchNormArgs<T> fused_batch_norm_args;
    if (FusedBatchNormArgs<T>::IsSupported(fusion)) {
      OP_REQUIRES_OK(context,
                     InitFusedBatchNormArgs(context, fusion_args.epsilon,
                                            &fused_batch_norm_args));
    }

    auto input_map = input.tensor<float, 4>();
    const float *input_array = input_map.data();
    auto filter_map = filter.tensor<float, 4>();
    const float *filter_array = filter_map.data();
    auto dinput_map = dinput.tensor<float, 4>();
    const float *dinput_array = dinput_map.data();

    auto output_map = output->tensor<float, 4>();
    float *output_array = const_cast<float *>(output_map.data());

    float *in_arr = const_cast<float *>(input_array);
    float *din_arr = const_cast<float *>(dinput_array);
    float *filt_arr = const_cast<float *>(filter_array);
    float *bia_arr = NULL;
    float *batch_norm_mean_data =
        const_cast<float *>(fused_batch_norm_args.estimated_mean_data);
    float *batch_norm_offset_data =
        const_cast<float *>(fused_batch_norm_args.offset_data);

#if NEW_API
    primitive_attr conv_attr;

    zendnnEnv zenEnvObj = readEnv();
    bool blocked = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT1;
    bool blockedNHWC = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT2;

    if (blocked || blockedNHWC) {
      ZenExecutor *ex = ex->getInstance();
      engine eng = ex->getEngine();
      stream s = ex->getStream();
      zenConvolution2DBatchNormOrRelu(
          eng, s, conv_attr, in_arr, dimensions.batch, dimensions.in_depth,
          dimensions.input_rows, dimensions.input_cols, filt_arr,
          dimensions.out_depth, dimensions.filter_rows, dimensions.filter_cols,
          dimensions.pad_rows_before, dimensions.pad_cols_before,
          dimensions.pad_rows_after, dimensions.pad_cols_after,
          dimensions.stride_rows, dimensions.stride_cols, bia_arr,
          fused_batch_norm_args.scaling_factor.data(), batch_norm_mean_data,
          batch_norm_offset_data, din_arr, output_array, dimensions.out_rows,
          dimensions.out_cols, true, true, reorder_before, reorder_after,
          cached_filter_data_, context);
    }
    // TODO: This else part will go once NEW API is supported for non blocked
    // format
    else {
      zenConvolution2DwithBatchNormsum(
          input_array, dimensions.batch, dimensions.in_depth,
          dimensions.input_rows, dimensions.input_cols, filter_array,
          dimensions.out_depth, dimensions.filter_rows, dimensions.filter_cols,
          dimensions.pad_rows_before, dimensions.pad_cols_before,
          dimensions.pad_rows_after, dimensions.pad_cols_after,
          dimensions.stride_rows, dimensions.stride_cols,
          fused_batch_norm_args.scaling_factor.data(),
          fused_batch_norm_args.estimated_mean_data,
          fused_batch_norm_args.offset_data, din_arr, output_array,
          dimensions.out_rows, dimensions.out_cols);
    }
#else

    // To Do , I will support this path with NEW API both for non blocked
    // formats in following checkin Add checks for fused computation - This path
    // is tested for Accuracy with Resnet50
    zenConvolution2DwithBatchNormsum(
        input_array, dimensions.batch, dimensions.in_depth,
        dimensions.input_rows, dimensions.input_cols, filter_array,
        dimensions.out_depth, dimensions.filter_rows, dimensions.filter_cols,
        dimensions.pad_rows_before, dimensions.pad_cols_before,
        dimensions.pad_rows_after, dimensions.pad_cols_after,
        dimensions.stride_rows, dimensions.stride_cols,
        fused_batch_norm_args.scaling_factor.data(),
        fused_batch_norm_args.estimated_mean_data,
        fused_batch_norm_args.offset_data, din_arr, output_array,
        dimensions.out_rows, dimensions.out_cols);
#endif
  }
};

template <typename T>
class ZenFusedConv2DSumOp : public OpKernel {
 public:
  explicit ZenFusedConv2DSumOp(OpKernelConstruction *context)
      : OpKernel(context) {
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
        {FCT::kFusedBatchNormWithRelu, {"FusedBatchNorm", "Relu"}},
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
    const Tensor &dinput = context->input(6);

    Conv2DDimensions dimensions;
    OP_REQUIRES_OK(context,
                   ComputeConv2DDimension(params_, input, filter, &dimensions));

    TensorShape out_shape = ShapeFromFormat(
        params_.data_format, dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);
    // Update the output type
    zenTensorType out_type = zenTensorType::FLOAT;

    zendnnEnv zenEnvObj = readEnv();
    Tensor *output = nullptr;
    int zenEnableMemPool = zenEnvObj.zenEnableMemPool &&
                           context->expected_output_dtype(0) == DT_FLOAT;
    ZenMemoryPool<T> *zenPoolBuffer = NULL;

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
      // To DO : Reusing the input could help to avoid memory allocation but
      // this results in crash
      //         Identify the root cause and avoid memory allocation
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    }

    VLOG(2) << "ZenFusedConv2DSum: in_depth = " << dimensions.in_depth
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
    LaunchZenFusedConv2DSumOp<T>()(context, input, filter, dinput,
                                   fused_computation_, fused_computation_args_,
                                   params_, dimensions, output, reorder_before,
                                   reorder_after, &cached_filter_data_);

    // If ZenMemPool Optimization is enabled(default), update the state of
    //  Memory pool based on input_array address
    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      auto input_map = input.tensor<float, 4>();
      const float *input_array = input_map.data();

      auto dinput_map = dinput.tensor<float, 4>();
      const float *dinput_array = dinput_map.data();

      zenPoolBuffer->zenMemPoolFree(context, (void *)input_array);
      zenPoolBuffer->zenMemPoolFree(context, (void *)dinput_array);
    }
  }

 private:
  Conv2DParameters params_;
  bool reorder_before, reorder_after, reset;
  int in_links, out_links;
  Tensor cached_filter_data_ TF_GUARDED_BY(mu_);
  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;

  TF_DISALLOW_COPY_AND_ASSIGN(ZenFusedConv2DSumOp);
};

// Registration of the CPU implementations.
#define REGISTER_ZEN_FUSED_CPU_CONV2D_SUM(T)                                \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_ZenFusedConv2DSum").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ZenFusedConv2DSumOp<T>);

REGISTER_ZEN_FUSED_CPU_CONV2D_SUM(float)

}  // namespace tensorflow
