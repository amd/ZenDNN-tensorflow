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

#include <cmath>
#include <iostream>
#include <vector>

#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/kernels/zendnn/zen_conv_ops_util.h"
#include "tensorflow/core/util/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "zendnn_helper.hpp"

#define NEW_API 1
#if NEW_API
#include "zendnn.hpp"
using namespace zendnn;
using namespace std;
void zenGemmConvolution2D(void *input_array, int batch_size, int channels,
                          int height, int width, void *filter_array,
                          int output_channels, int kernel_h, int kernel_w,
                          float pad_t, float pad_l, float pad_b, float pad_r,
                          int stride_h, int stride_w, void *bias_array,
                          void *output_array, int out_height, int out_width,
                          bool reluFused, bool batchNormFused, bool addFused,
                          void *bn_scale, void *bn_mean, void *bn_offset);
template <typename T>
void zenConvolution2DBiasOrRelu(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void *input_array, int batch_size, int channels, int height, int width,
    void *filter_array, int output_channels, int kernel_h, int kernel_w,
    float pad_t, float pad_l, float pad_b, float pad_r, int stride_h,
    int stride_w, void *bias_array, void *output_array, int out_height,
    int out_width, bool reorder_before, bool reorder_after,
    void *cached_filter_data_, void *context);
#endif

namespace tensorflow {

template <typename T, bool is_depthwise = false>
class ZenConvOp : public OpKernel {
 private:
  Tensor cached_filter_data_ TF_GUARDED_BY(mu_);
  Conv2DParameters params_;
  bool reorder_before, reorder_after, reset;
  int in_links, out_links;

 public:
  explicit ZenConvOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));
    OP_REQUIRES(context, params_.data_format == FORMAT_NHWC,
                errors::Unimplemented("ZenDNN Conv implementation supports "
                                      "NHWC tensor format only for now."));
    OP_REQUIRES_OK(context,
                   context->GetAttr("reorder_before", &reorder_before));
    OP_REQUIRES_OK(context, context->GetAttr("reorder_after", &reorder_after));
    OP_REQUIRES_OK(context, context->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(context, context->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(context, context->GetAttr("reset", &reset));
    /*TODO: Temporarily throw error if dilations are passed. Backend must
    support dilated convolutions later though
    for(int it : params_.dilations)
      if (it != 0)
        context->SetStatus(errors::Unimplemented("ZenConv2D does not
                           support dilations yet"));*/
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input = context->input(0);
    auto input_map =
        input.tensor<T, 4>();  // experimented and proven that it is row-major
    const T *input_array = input_map.data();

    const Tensor &filter = context->input(1);
    auto filter_map =
        filter.tensor<T, 4>();  // experimented and proven that it is row-major
    const T *filter_array = filter_map.data();

    Conv2DDimensions dimensions;
    OP_REQUIRES_OK(context,
                   ComputeConv2DDimension(params_, input, filter, &dimensions));

    if (is_depthwise) {
      dimensions.out_depth *= dimensions.patch_depth;
    }
    // Update the output type
    zenTensorType out_type = (std::is_same<T, float>::value)
                                 ? (zenTensorType::FLOAT)
                                 : (zenTensorType::BFLOAT16);

    TensorShape out_shape = ShapeFromFormat(
        params_.data_format, dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);
    zendnnEnv zenEnvObj = readEnv();
    Tensor *output = nullptr;
    int zenEnableMemPool = zenEnvObj.zenEnableMemPool &&
                           (context->expected_output_dtype(0) == DT_FLOAT ||
                            context->expected_output_dtype(0) == DT_BFLOAT16);
    ZenMemoryPool<T> *zenPoolBuffer = NULL;

    // ZenMemPool Optimization reuse o/p tensors from the pool. By default
    //  its enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory
    //  pool optimization
    //  Cases where tensors in pool are not free or requested size is more
    //  than available tensor size in Pool, control will fall back to
    //  default way of allocation i.e. with allocate_output(..)
    //  ZenMempool Optimization is not supported by Depthwise Convolution
    //  due to performance drop.
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
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    }

    auto output_map = output->tensor<T, 4>();
    T *output_array = const_cast<T *>(output_map.data());

#if NEW_API
    T *in_arr = const_cast<T *>(input_array);
    T *filt_arr = const_cast<T *>(filter_array);
    T *bias_arr = nullptr;

    bool blocked = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT1;
    bool blockedNHWC = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT2;
    // Direct convolution
    primitive_attr conv_attr;
    ZenExecutor *ex = ex->getInstance();
    engine eng = ex->getEngine();
    stream s = ex->getStream();
    if (is_depthwise) {
      zenConvolution2DDepthwise<T>(
          eng, s, conv_attr, in_arr, dimensions.batch, dimensions.in_depth,
          dimensions.input_rows, dimensions.input_cols, filt_arr,
          dimensions.out_depth, dimensions.filter_rows, dimensions.filter_cols,
          dimensions.pad_rows_before, dimensions.pad_cols_before,
          dimensions.pad_rows_after, dimensions.pad_cols_after,
          dimensions.stride_rows, dimensions.stride_cols, bias_arr,
          output_array, dimensions.out_rows, dimensions.out_cols,
          reorder_before, reorder_after, &cached_filter_data_, context);
    } else if (blocked || blockedNHWC) {
      zenConvolution2DBiasOrRelu<T>(
          eng, s, conv_attr, in_arr, dimensions.batch, dimensions.in_depth,
          dimensions.input_rows, dimensions.input_cols, filt_arr,
          dimensions.out_depth, dimensions.filter_rows, dimensions.filter_cols,
          dimensions.pad_rows_before, dimensions.pad_cols_before,
          dimensions.pad_rows_after, dimensions.pad_cols_after,
          dimensions.stride_rows, dimensions.stride_cols, bias_arr,
          output_array, dimensions.out_rows, dimensions.out_cols,
          reorder_before, reorder_after, &cached_filter_data_, context);
    } else {
      // GEMM based convolution
      zenGemmConvolution2D(
          in_arr, dimensions.batch, dimensions.in_depth, dimensions.input_rows,
          dimensions.input_cols, filt_arr, dimensions.out_depth,
          dimensions.filter_rows, dimensions.filter_cols,
          dimensions.pad_rows_before, dimensions.pad_cols_before,
          dimensions.pad_rows_after, dimensions.pad_cols_after,
          dimensions.stride_rows, dimensions.stride_cols, bias_arr,
          output_array, dimensions.out_rows, dimensions.out_cols, false, false,
          false, nullptr, nullptr, nullptr);
    }
#else
    // TF-Zen approach#2 integration
    zenConvolution2D(input_array, dimensions.batch, dimensions.in_depth,
                     dimensions.input_rows, dimensions.input_cols, filter_array,
                     dimensions.out_depth, dimensions.filter_rows,
                     dimensions.filter_cols, dimensions.pad_rows_before,
                     dimensions.pad_cols_before, dimensions.pad_rows_after,
                     dimensions.pad_cols_after, dimensions.stride_rows,
                     dimensions.stride_cols, output_array, dimensions.out_rows,
                     dimensions.out_cols);
#endif

    // If ZenMemPool Optimization is enabled(default), update the state of
    //  Memory pool based on input_array address
    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      zenPoolBuffer->zenMemPoolFree(context, (void *)input_array);
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("_ZenConv2D").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    ZenConvOp<float>);

REGISTER_KERNEL_BUILDER(Name("_ZenConv2D")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<::tensorflow::bfloat16>("T"),
                        ZenConvOp<::tensorflow::bfloat16>);

REGISTER_KERNEL_BUILDER(Name("_ZenDepthwiseConv2dNative")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        ZenConvOp<float, true>);

}  // namespace tensorflow
