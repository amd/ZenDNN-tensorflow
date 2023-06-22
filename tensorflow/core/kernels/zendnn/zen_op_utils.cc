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

#include <assert.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/zendnn/zen_conv_ops_util.h"
#include "zendnn.hpp"
#include "zendnn_helper.hpp"
#include "zendnn_logging.hpp"

using namespace tensorflow;
using namespace zendnn;
using namespace std;

/* Intermediate output results are not consistent.
 * Ouput results vary across runs for the same input.
 * Disabling the flag for now.
 * Please enable it after fixing the issues.
 */
#define CONV_PRIMITIVE_CACHE 0

// This routine converts FP32 into qint8/quint8 based on scale factors
void zenQuantized(zendnn::engine eng, zendnn::stream s, void *input_array,
                  int batch_size, int height, int width, int channels,
                  float scale_factor, bool out_type, void *output_array) {
  using tag = memory::format_tag;
  using dt = memory::data_type;

  zendnn::primitive_attr attr;
  std::vector<float> scales_;
  scales_.resize(1);
  scales_[0] = scale_factor;
  attr.set_output_scales(0, scales_);

  memory::dims src_tz = {batch_size, channels, height, width};
  memory::dims dst_tz = {batch_size, channels, height, width};
  memory::desc src_md = memory::desc({src_tz}, dt::f32, tag::acdb);
  memory::desc dst_md =
      memory::desc({dst_tz}, (channels == 32) ? dt::u8 : dt::s8, tag::acdb);

  zendnn::memory src_mem = memory(src_md, eng, input_array);
  zendnn::memory dst_mem = memory(dst_md, eng, output_array);

  auto reorder_pd = reorder::primitive_desc(eng, src_md, eng, dst_md, attr);
  auto val_reorder = reorder(reorder_pd);
  val_reorder.execute(s, src_mem, dst_mem);
}

// TODO: Code cleanup to be done for July Release 2021
void zenQuantizedConvolution2DBiasOrRelu(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void *context, void *input_array, int batch_size, int channels, int height,
    int width, void *filter_array, int output_channels, int kernel_h,
    int kernel_w, int out_height, int out_width, int pad_t, int pad_l,
    int pad_b, int pad_r, int stride_h, int stride_w, void *bias_array,
    vector<float> &scale, void *output_array, void *output_min,
    void *output_max, bool Tinput, bool Toutput, bool Tbias,
    vector<float> &bias_scale, bool is_relu, bool is_sum, bool is_signed,
    float factor, int depth, float scale_output, float scale_summand,
    void *cached_filter_data_, bool reset) {
  using tag = memory::format_tag;
  using dt = memory::data_type;

  // Support for Filter Caching
  const Tensor &cached_filter_data_tensor = *(Tensor *)cached_filter_data_;

  // TODO: Add an alternative fix
  // This fixes the accuracy issue in the first Convolution layer.
  // By setting res = -1, we are not using persistent caching
  // for this layer and the memory will be reallocated.
  // The check output_channels%8 fix accuracy issues with MobileNetV1
  // INT8 model for lower batch sizes.
  int res = cached_filter_data_tensor.NumElements();
  if (kernel_h == 7 || output_channels % 8 != 0) {
    res = -1;
  }
  void *filter_data = NULL;

  zendnn::post_ops post_ops;

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  memory::dims conv1_src_tz = {batch_size, channels, height, width};
  memory::dims conv1_weights_tz;
  if (depth != 1) {
    conv1_weights_tz = {output_channels, channels, kernel_h, kernel_w};
  } else {
    conv1_weights_tz = {channels, 1, 1, kernel_w, kernel_h};
    // TODO: pad values hardcoding to be removed for 2021 July release
    if (stride_h == 2 && kernel_h == 3) {
      pad_l = 0, pad_t = 0, pad_b = 1, pad_r = 1;
    } else {
      pad_l = 1, pad_t = 1;
      pad_b = 1, pad_r = 1;
    }
  }
  memory::dims bias_dims = {output_channels};
  memory::dims conv1_dst_tz = {batch_size, output_channels, out_height,
                               out_width};
  memory::dims conv1_strides = {stride_h, stride_w};
  memory::dims conv1_padding1 = {pad_t, pad_l};
  memory::dims conv1_padding2 = {pad_b, pad_r};

#if !CONV_PRIMITIVE_CACHE
  memory::desc conv1_src_md =
      memory::desc({conv1_src_tz}, (!Tinput) ? dt::s8 : dt::u8, tag::acdb);
  memory::desc conv1_bias_md =
      memory::desc({bias_dims}, (Tbias == 1) ? dt::s32 : dt::f32, tag::x);
  memory::desc conv1_weights_md =
      memory::desc({conv1_weights_tz}, dt::s8, tag::any);
  memory::desc conv1_dst_md =
      memory::desc({conv1_dst_tz}, (!Toutput) ? dt::s8 : dt::u8, tag::acdb);
  convolution_forward::desc conv1_desc = zendnn::convolution_forward::desc(
      zendnn::prop_kind::forward_inference,
      zendnn::algorithm::convolution_direct, conv1_src_md, conv1_weights_md,
      conv1_bias_md, conv1_dst_md, conv1_strides, conv1_padding1,
      conv1_padding2);
  // Convolution masks specify if a scalar value is used ( Per Channel mask ) or
  // Vector values are used for scaling ( Per Tensor masks )
  // mask value of 0 denotes scalar mask while 1 denotes vector masks
  if (scale.size() == 1) {
    conv_attr.set_output_scales(0, scale);
  } else {
    conv_attr.set_output_scales(2, scale);
  }
  if (is_sum) {
    post_ops.append_sum(255.0f * scale_summand / (scale_output * factor));
  }
  if (is_relu) {
    post_ops.append_eltwise(1.0f, zendnn::algorithm::eltwise_relu, 0.0f, 0.0f);
  }
  conv_attr.set_post_ops(post_ops);
  convolution_forward::primitive_desc prim_desc =
      convolution_forward::primitive_desc(conv1_desc, conv_attr, eng);
#else
  memory::data_type src_dt = (!Tinput) ? dt::s8 : dt::u8;
  memory::data_type filter_dt = dt::s8;
  memory::data_type bias_dt = (Tbias == 1) ? dt::s32 : dt::f32;
  memory::data_type dst_dt = (!Toutput) ? dt::s8 : dt::u8;

  memory::format_tag weight_tag = (depth != 1) ? tag::cdba : tag ::decab;

  ZenConvParams conv_params(conv1_src_tz, conv1_weights_tz, bias_dims,
                            conv1_dst_tz, conv1_strides, conv1_padding1,
                            conv1_padding2, src_dt, filter_dt, bias_dt, dst_dt,
                            weight_tag, cached_filter_data_,
                            zendnn::algorithm::convolution_direct, scale);

  if (scale.size() == 1) {
    conv_params.post_op_params.push_back({"scale", {0}});
  } else {
    conv_params.post_op_params.push_back({"scale", {2}});
  }
  if (is_sum) {
    conv_params.post_op_params.push_back(
        {"sum", {255.0f * scale_summand / (scale_output * factor)}});
  }
  if (is_relu) {
    conv_params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
  }
  bool disable_reuse_opt = ZenPrimitiveFactory::IsReuseOptDisabled();
  ZenConvPrimitive *conv_prim =
      ZenConvPrimitiveFactory::Get(conv_params, disable_reuse_opt);
#endif

#if !CONV_PRIMITIVE_CACHE
  auto user_src_memory = memory(prim_desc.src_desc(), eng, input_array);
  zendnn::memory usr_dst_mem =
      memory({{conv1_dst_tz}, (!Toutput) ? dt::s8 : dt::u8, tag::acdb}, eng,
             output_array);

  auto user_weights_memory = zendnn::memory(
      {{conv1_weights_tz}, dt::s8, (depth != 1) ? tag::cdba : tag ::decab}, eng,
      filter_array);

  auto conv1_bias_memory = memory(prim_desc.bias_desc(), eng, bias_array);
  auto conv2_bias_memory = memory(prim_desc.bias_desc(), eng);

  auto conv1_src_memory = user_src_memory;
  auto conv1_dst_memory = usr_dst_mem;
#endif
  // Mask value of 0 indicates vector of bias values is scaled by same scale
  // Mask value of 1 indicates vector of bias values are scaled by different
  // scale
  primitive_attr bias_attr;
  if (bias_scale.size() == 1) {
    bias_attr.set_output_scales(0, bias_scale);
  } else {
    bias_attr.set_output_scales(1, bias_scale);
  }
#if CONV_PRIMITIVE_CACHE
  auto conv1_bias_memory =
      memory(conv_prim->GetPrimitiveDesc().get()->bias_desc(), eng, bias_array);
  auto conv2_bias_memory =
      memory(conv_prim->GetPrimitiveDesc().get()->bias_desc(), eng);
  auto user_weights_memory = zendnn::memory(
      {{conv1_weights_tz}, dt::s8, (depth != 1) ? tag::cdba : tag ::decab}, eng,
      filter_array);
#endif
  if (!Tbias) {
    auto bias_reorder_pd =
        reorder::primitive_desc(eng, conv1_bias_memory.get_desc(), eng,
                                conv2_bias_memory.get_desc(), bias_attr);
    auto bias_reorder = reorder(bias_reorder_pd);
    bias_reorder.execute(s, conv1_bias_memory, conv2_bias_memory);
  }

  zendnn::memory conv1_weights_memory;
  if (res <= 0) {
#if !CONV_PRIMITIVE_CACHE
    conv1_weights_memory = memory(prim_desc.weights_desc(), eng);
#else
    conv1_weights_memory =
        memory(conv_prim->GetPrimitiveDesc().get()->weights_desc(), eng);
#endif
    net.push_back(reorder(user_weights_memory, conv1_weights_memory));
    net_args.push_back({{ZENDNN_ARG_SRC, user_weights_memory},
                        {ZENDNN_ARG_DST, conv1_weights_memory}});

    assert(net.size() == net_args.size() && "something is missing");
    for (size_t i = 0; i < net.size(); ++i) {
      net.at(i).execute(s, net_args.at(i));
    }
  } else {
    filter_data = static_cast<qint8 *>(
        const_cast<qint8 *>(cached_filter_data_tensor.flat<qint8>().data()));
#if !CONV_PRIMITIVE_CACHE
    conv1_weights_memory = memory(prim_desc.weights_desc(), eng, filter_data);
#else
    conv1_weights_memory = memory(
        conv_prim->GetPrimitiveDesc().get()->weights_desc(), eng, filter_data);
#endif
  }

#if !CONV_PRIMITIVE_CACHE
  if (!Tbias) {
    net.push_back(convolution_forward(prim_desc));
    net_args.push_back({{ZENDNN_ARG_SRC, user_src_memory},
                        {ZENDNN_ARG_WEIGHTS, conv1_weights_memory},
                        {ZENDNN_ARG_BIAS, conv2_bias_memory},
                        {ZENDNN_ARG_DST, usr_dst_mem}});
  } else {
    net.push_back(convolution_forward(prim_desc));
    net_args.push_back({{ZENDNN_ARG_SRC, user_src_memory},
                        {ZENDNN_ARG_WEIGHTS, conv1_weights_memory},
                        {ZENDNN_ARG_BIAS, conv1_bias_memory},
                        {ZENDNN_ARG_DST, usr_dst_mem}});
  }

  assert(net.size() == net_args.size() && "something is missing");
  for (size_t i = 0; i < net.size(); ++i) {
    net.at(i).execute(s, net_args.at(i));
  }
#else

  const float *filter_array_new =
      (const float *)conv1_weights_memory.get_data_handle();
  const float *bias_array_1 =
      (const float *)conv1_bias_memory.get_data_handle();
  const float *bias_array_2 =
      (const float *)conv2_bias_memory.get_data_handle();

  if (!Tbias) {
    // Get the convolution primitive
    conv_prim->Execute((const float *)input_array,
                       (const float *)filter_array_new,
                       (const float *)bias_array_2, (float *)output_array);
  } else {
    // Get the convolution primitive
    conv_prim->Execute((const float *)input_array,
                       (const float *)filter_array_new,
                       (const float *)bias_array_1, (float *)output_array);
  }
#endif
  TensorShape filter_tf_shape;
  Tensor *filter_tensor_ptr = nullptr;

  filter_tf_shape.AddDim(user_weights_memory.get_desc().get_size());

  if (res == 0) {
    ((OpKernelContext *)context)
        ->allocate_temp(DT_QINT8, filter_tf_shape,
                        (Tensor *)cached_filter_data_);
    size_t cached_filter_data_size = user_weights_memory.get_desc().get_size();
    qint8 *weights_data =
        static_cast<qint8 *>(conv1_weights_memory.get_data_handle());
    memcpy(static_cast<qint8 *>(
               ((Tensor *)cached_filter_data_)->flat<qint8>().data()),
           weights_data, cached_filter_data_size);
  }
}

void zenGemmConvolution2D(void *input_array, int batch_size, int channels,
                          int height, int width, void *filter_array,
                          int output_channels, int kernel_h, int kernel_w,
                          float pad_t, float pad_l, float pad_b, float pad_r,
                          int stride_h, int stride_w, void *bias_array,
                          void *output_array, int out_height, int out_width,
                          bool reluFused, bool batchNormFused, bool addFused,
                          void *bn_scale, void *bn_mean, void *bn_offset) {
  // GEMM based convolution
  memory::dims src_dims = {batch_size, channels, height, width};
  memory::dims weights_dims = {output_channels, channels, kernel_h, kernel_w};
  memory::dims bias_dims = {output_channels};
  memory::dims dst_dims = {batch_size, output_channels, out_height, out_width};
  memory::dims strides = {stride_h, stride_w};
  memory::dims padding_left = {pad_t, pad_l};
  memory::dims padding_right = {pad_b, pad_r};

  vector<float> vect;
  ZenConvParams conv_params(
      src_dims, weights_dims, bias_dims, dst_dims, strides, padding_left,
      padding_right, memory::data_type::f32, memory::data_type::f32,
      memory::data_type::f32, memory::data_type::f32, memory::format_tag::hwcn,
      NULL, zendnn::algorithm::convolution_gemm, vect);
  if (addFused) {
    conv_params.post_op_params.push_back({"sum", {1.0}});
  }
  if (reluFused) {
    conv_params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
  }
  if (batchNormFused) {
    conv_params.post_op_params.push_back({"batchnorm", {}});
  }
  // Get the convolution primitive
  bool disable_reuse_opt = ZenPrimitiveFactory::IsReuseOptDisabled();
  ZenConvPrimitive *conv_prim =
      ZenConvPrimitiveFactory::Get(conv_params, disable_reuse_opt);
  if (!batchNormFused) {
    conv_prim->Execute((const float *)input_array, (const float *)filter_array,
                       (const float *)bias_array, (float *)output_array);
  } else {
    conv_prim->Execute((const float *)input_array, (const float *)filter_array,
                       (const float *)bias_array, (float *)output_array,
                       (const float *)bn_scale, (const float *)bn_mean,
                       (const float *)bn_offset);
  }
}

/**
 * Function for Blocked Conv2D that fuses elementwise sum. This function
 * assumes one of the inputs of elementwise sum is available in output buffer.
 * TODO: (1) Get buffer for reorders from ZenMemoryPool
 *       (2) Reorder has been used to copy contents from one buffer to other.
 *           Need to check on memcpy instead for optimal performance.
 */
template <typename T>
void zenBlockedConv2DBiasEltSum(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr prim_attr,
    void *input_array, int batch_size, int channels, int height, int width,
    void *filter_array, int output_channels, int kernel_h, int kernel_w,
    int pad_t, int pad_l, int pad_b, int pad_r, int stride_h, int stride_w,
    void *bias_array, void *output_array, int out_height, int out_width,
    bool reorder_before, bool reorder_after, void *cached_filter_data_,
    void *context) {
  zendnnInfo(ZENDNN_FWKLOG,
             "zenBlockedConv2DBiasEltSum (TF kernel): New API for DIRECT \
               CONV2D with elementwise sum fused");

  using tag = memory::format_tag;
  using dt = memory::data_type;
  bool is_input_float = std::is_same<T, float>::value;

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  memory::dims src_tz = {batch_size, channels, height, width};
  memory::dims wts_tz = {output_channels, channels, kernel_h, kernel_w};
  memory::dims bias_tz = {output_channels};
  memory::dims dst_tz = {batch_size, output_channels, out_height, out_width};
  memory::dims strides = {stride_h, stride_w};
  memory::dims padding1 = {pad_t, pad_l};
  memory::dims padding2 = {pad_b, pad_r};
  // ZenDNN state
  zendnnEnv zenEnvObj = readEnv();
  bool blocked = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT1;
  bool blockedNHWC = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT2;
  // Note: we are forcing it through BlockedNHWC when input is bf16 towards beta
  // support for this release.
  if (!is_input_float) {
    bool result =
        tensorflow::port::TestCPUFeature(tensorflow::port::CPUFeature::AVX512F);
    if (!result) {
      OP_REQUIRES_OK(
          (OpKernelContext *)context,
          errors::Internal(
              "BF16 AVX512 instruction set is not supported in the machine."));
    }
    blocked = 0;
    blockedNHWC = 1;
  }
  // define memory descriptors
  memory::desc src_md = memory::desc({src_tz}, dt::f32, tag::any);
  memory::desc bias_md = memory::desc({bias_tz}, dt::f32, tag::any);
  memory::desc wts_md = memory::desc({wts_tz}, dt::f32, tag::any);
  memory::desc dst_md = memory::desc({dst_tz}, dt::f32, tag::aBcd8b);
  if (blockedNHWC) {
    src_md = memory::desc({src_tz}, (is_input_float) ? dt::f32 : dt::bf16,
                          tag::nhwc);
    bias_md =
        memory::desc({bias_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::x);
    wts_md =
        memory::desc({wts_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::any);
    dst_md = memory::desc({dst_tz}, (is_input_float) ? dt::f32 : dt::bf16,
                          tag::nhwc);
  }

  const Tensor &cached_filter_data_tensor = *(Tensor *)cached_filter_data_;

  int res = cached_filter_data_tensor.NumElements();
  void *filter_data = NULL;

  convolution_forward::desc op_desc = convolution_forward::desc(
      prop_kind::forward_inference, algorithm::convolution_direct, src_md,
      wts_md, bias_md, dst_md, strides, padding1, padding2);
  if (!bias_array) {
    op_desc = convolution_forward::desc(
        prop_kind::forward_inference, algorithm::convolution_direct, src_md,
        wts_md, dst_md, strides, padding1, padding2);
  }

  // define primitive descriptor
  convolution_forward::primitive_desc prim_desc =
      convolution_forward::primitive_desc(op_desc, prim_attr, eng);

  // define the source and destination memory
  zendnn::memory src_mem, dst_mem;
  if (blocked) {
    if (reorder_before) {
      // reorder before is enabled which means that the source inputs are in
      // NHWC format, hence both the source inputs needs to be reordered to
      // blocked data format.

      // one input via input_array is reordered for blocked format
      zendnn::memory usr_src_mem =
          memory({{src_tz}, dt::f32, tag::nhwc}, eng, input_array);
      src_mem = memory(prim_desc.src_desc(), eng);
      net.push_back(reorder(usr_src_mem, src_mem));
      net_args.push_back(
          {{ZENDNN_ARG_SRC, usr_src_mem}, {ZENDNN_ARG_DST, src_mem}});
      // second input via output_array is reordered for blocked format
      zendnn::memory usr_dst_mem =
          memory({{dst_tz}, dt::f32, tag::nhwc}, eng, output_array);
      dst_mem = memory(prim_desc.dst_desc(), eng);
      net.push_back(reorder(usr_dst_mem, dst_mem));
      net_args.push_back(
          {{ZENDNN_ARG_SRC, usr_dst_mem}, {ZENDNN_ARG_DST, dst_mem}});
    }

    else {
      // reorder before is disabled which means that the source inputs are in
      // blocked format, hence no reorder is required.

      // src mem definition with blocked data format
      src_mem = memory({{src_tz}, dt::f32, tag::aBcd8b}, eng, input_array);
      // dst mem definition
      if (reorder_after) {
        // For reorder after enabled, we do a reorder to transfer the data
        // to a temporary buffer but no data format is changed. This is
        // because we need output_array buffer for reordering after the
        // primitive execution.
        zendnn::memory usr_dst_mem =
            memory({{dst_tz}, dt::f32, tag::aBcd8b}, eng, output_array);
        dst_mem = memory(prim_desc.dst_desc(), eng);
        net.push_back(reorder(usr_dst_mem, dst_mem));
        net_args.push_back(
            {{ZENDNN_ARG_SRC, usr_dst_mem}, {ZENDNN_ARG_DST, dst_mem}});
      } else {
        dst_mem = memory({{dst_tz}, dt::f32, tag::aBcd8b}, eng, output_array);
      }
    }
  } else {
    src_mem =
        memory({{src_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::nhwc},
               eng, input_array);
    dst_mem =
        memory({{dst_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::nhwc},
               eng, output_array);
  }
  // define filters memory
  // To Do - will cleanup or improve over subsequent checkins
  zendnn::memory wts_mem;
  if (res <= 0) {
    wts_mem =
        memory({{wts_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::hwcn},
               eng, filter_array);
    if (prim_desc.weights_desc() != wts_mem.get_desc()) {
      // filters are in hwcn format in TF and hence for blocked format we
      // need to reorder the filters to blocked format
      zendnn::memory usr_wts_mem = wts_mem;
      wts_mem = memory(prim_desc.weights_desc(), eng);
      net.push_back(reorder(usr_wts_mem, wts_mem));
      net_args.push_back(
          {{ZENDNN_ARG_SRC, usr_wts_mem}, {ZENDNN_ARG_DST, wts_mem}});
    }
  } else {
    filter_data = static_cast<T *>(
        const_cast<T *>(cached_filter_data_tensor.flat<T>().data()));
    wts_mem = memory(prim_desc.weights_desc(), eng, filter_data);
  }

  // create primitive
  net.push_back(convolution_forward(prim_desc));

  if (bias_array) {
    zendnn::memory bias_mem =
        memory({{bias_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::x}, eng,
               bias_array);
    net_args.push_back({{ZENDNN_ARG_SRC, src_mem},
                        {ZENDNN_ARG_WEIGHTS, wts_mem},
                        {ZENDNN_ARG_BIAS, bias_mem},
                        {ZENDNN_ARG_DST, dst_mem}});
  } else {
    net_args.push_back({{ZENDNN_ARG_SRC, src_mem},
                        {ZENDNN_ARG_WEIGHTS, wts_mem},
                        {ZENDNN_ARG_DST, dst_mem}});
  }

  // primitive execution
  assert(net.size() == net_args.size() && "something is missing");
  for (size_t i = 0; i < net.size(); ++i) {
    net.at(i).execute(s, net_args.at(i));
  }

  if (res <= 0) {
    TensorShape filter_tf_shape;
    Tensor *filter_tensor_ptr = nullptr;
    filter_tf_shape.AddDim(wts_mem.get_desc().get_size());
    ((OpKernelContext *)context)
        ->allocate_temp(is_input_float ? DT_FLOAT : DT_BFLOAT16,
                        filter_tf_shape, ((Tensor *)cached_filter_data_));
    size_t cached_filter_data_size = wts_mem.get_desc().get_size();
    T *weights_data = static_cast<T *>(wts_mem.get_data_handle());
    memcpy(static_cast<T *>(((Tensor *)cached_filter_data_)->flat<T>().data()),
           weights_data, cached_filter_data_size);
  }

  if (reorder_after && blocked) {
    // output_array should have the data in nhwc data format as reorder
    // after is enabled.

    // Reorder after is enabled, so output memory has to be converted to
    // nhwc format from blocked format.
    zendnn::memory usr_dst_mem =
        memory({{dst_tz}, dt::f32, tag::nhwc}, eng, output_array);
    reorder(dst_mem, usr_dst_mem).execute(s, dst_mem, usr_dst_mem);
  } else {
    // output_array should have the data in blocked data format as reorder
    // after is disabled.

    if (reorder_before && blocked) {
      // Reorder after is disabled, but since reorder before was enabled
      // the primitive execution was having temp buffer for primitive
      // output. So to bring the primitive output to output_array from tmp
      // buffer we do a reorder here. (i.e. this reorder is just done for
      // moving output data from one buffer to the other)
      zendnn::memory usr_dst_mem =
          memory({{dst_tz}, dt::f32, tag::aBcd8b}, eng, output_array);
      reorder(dst_mem, usr_dst_mem).execute(s, dst_mem, usr_dst_mem);
    } else {
      // Do nothing as no reorder is required before and after primitive
      // execution (i.e. both inputs and output of this primitive
      // execution are in blocked format)
    }
  }
}
template void zenBlockedConv2DBiasEltSum<float>(
    zendnn::engine, zendnn::stream, zendnn::primitive_attr, void *, int, int,
    int, int, void *, int, int, int, int, int, int, int, int, int, void *,
    void *, int, int, bool, bool, void *, void *);
template void zenBlockedConv2DBiasEltSum<bfloat16>(
    zendnn::engine, zendnn::stream, zendnn::primitive_attr, void *, int, int,
    int, int, void *, int, int, int, int, int, int, int, int, int, void *,
    void *, int, int, bool, bool, void *, void *);

template <typename T>
void zenConvolution2DDepthwise(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void *input_array, int batch_size, int channels, int height, int width,
    void *filter_array, int output_channels, int kernel_h, int kernel_w,
    float pad_t, float pad_l, float pad_b, float pad_r, int stride_h,
    int stride_w, void *bias_array, void *output_array, int out_height,
    int out_width, bool reorder_before, bool reorder_after,
    void *cached_filter_data_, void *context) {
  using tag = memory::format_tag;
  using dt = memory::data_type;
  bool is_input_float = std::is_same<T, float>::value;

  memory::dims conv1_src_tz = {batch_size, channels, height, width};
  // Assumption: output_channel and input_channel should be the multiple of
  // groups. groups = input_channels, oc_per_group = output_channels/groups,
  // ic_per_group = input_channels/groups = 1.
  memory::dims conv1_weights_tz = {channels, output_channels / channels, 1,
                                   kernel_h, kernel_w};
  memory::dims conv1_bias_tz = {output_channels};
  memory::dims conv1_dst_tz = {batch_size, output_channels, out_height,
                               out_width};
  memory::dims conv1_strides = {stride_h, stride_w};
  memory::dims conv1_padding1 = {pad_t, pad_l};
  memory::dims conv1_padding2 = {pad_b, pad_r};

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  const Tensor &cached_filter_data_tensor = *(Tensor *)cached_filter_data_;

  int res = cached_filter_data_tensor.NumElements();
  void *filter_data = NULL;

  zendnnEnv zenEnvObj = readEnv();
  bool blocked = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT1;
  bool blockedNHWC = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT2;

  // filter Tag:: d = height,e = width, c = ic, a = group, b = oc.
  auto filter_format = tag::decab;

  zendnn::memory user_src_memory;
  zendnn::memory conv1_dst_memory, conv1_dst_memory_new;
  zendnn::memory conv1_bias_memory;
  zendnn::memory user_weights_memory =
      zendnn::memory({{conv1_weights_tz},
                      (is_input_float) ? dt::f32 : dt::bf16,
                      filter_format},
                     eng, filter_array);
  // Memory descriptors
  memory::desc conv1_src_md = memory::desc(
      {conv1_src_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::nhwc);
  memory::desc conv1_bias_md = memory::desc(
      {conv1_bias_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::x);
  memory::desc conv1_weights_md = memory::desc(
      {conv1_weights_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::any);
  memory::desc conv1_dst_md = memory::desc(
      {conv1_dst_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::nhwc);

  if (blocked) {
    if (reorder_before) {
      user_src_memory =
          memory({{conv1_src_tz}, dt::f32, tag::nhwc}, eng, input_array);
    } else {
      user_src_memory =
          memory({{conv1_src_tz}, dt::f32, tag::aBcd8b}, eng, input_array);
    }

    if (reorder_after) {
      conv1_dst_memory = memory({{conv1_dst_tz}, dt::f32, tag::aBcd8b}, eng);
      conv1_dst_memory_new =
          memory({{conv1_dst_tz}, dt::f32, tag::nhwc}, eng, output_array);
    } else {
      conv1_dst_memory =
          memory({{conv1_dst_tz}, dt::f32, tag::aBcd8b}, eng, output_array);
    }

    conv1_bias_memory =
        memory({{conv1_bias_tz}, dt::f32, tag::x}, eng, bias_array);

    conv1_src_md = memory::desc({conv1_src_tz}, dt::f32, tag::any);
    conv1_bias_md = memory::desc({conv1_bias_tz}, dt::f32, tag::any);
    conv1_weights_md = memory::desc({conv1_weights_tz}, dt::f32, tag::any);
    conv1_dst_md = memory::desc({conv1_dst_tz}, dt::f32, tag::aBcd8b);
  }

  convolution_forward::desc conv1_desc = convolution_forward::desc(
      prop_kind::forward_inference, algorithm::convolution_direct, conv1_src_md,
      conv1_weights_md, conv1_bias_md, conv1_dst_md, conv1_strides,
      conv1_padding1, conv1_padding2);

  if (!bias_array)
    conv1_desc = convolution_forward::desc(
        prop_kind::forward_inference, algorithm::convolution_direct,
        conv1_src_md, conv1_weights_md, conv1_dst_md, conv1_strides,
        conv1_padding1, conv1_padding2);

  convolution_forward::primitive_desc conv1_prim_desc =
      convolution_forward::primitive_desc(conv1_desc, conv_attr, eng);

  if (!blocked) {
    user_src_memory = memory(conv1_prim_desc.src_desc(), eng, input_array);
    conv1_dst_memory = memory(
        {{conv1_dst_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::acdb}, eng,
        output_array);
    conv1_bias_memory = memory(conv1_prim_desc.bias_desc(), eng, bias_array);
  }

  zendnn::memory conv1_src_memory = user_src_memory;
  if (blocked) {
    if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
      conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
      if (reorder_before) {
        net.push_back(reorder(user_src_memory, conv1_src_memory));
        net_args.push_back({{ZENDNN_ARG_SRC, user_src_memory},
                            {ZENDNN_ARG_DST, conv1_src_memory}});
      }
    }
  }
  zendnn::memory conv1_weights_memory = user_weights_memory;
  if (res <= 0) {
    if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
      conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
      net.push_back(reorder(user_weights_memory, conv1_weights_memory));
      net_args.push_back({{ZENDNN_ARG_SRC, user_weights_memory},
                          {ZENDNN_ARG_DST, conv1_weights_memory}});
    }
  } else {
    filter_data = static_cast<T *>(
        const_cast<T *>(cached_filter_data_tensor.flat<T>().data()));
    conv1_weights_memory =
        memory(conv1_prim_desc.weights_desc(), eng, filter_data);
  }
  net.push_back(convolution_forward(conv1_prim_desc));
  if (!bias_array)
    net_args.push_back({{ZENDNN_ARG_SRC, conv1_src_memory},
                        {ZENDNN_ARG_WEIGHTS, conv1_weights_memory},
                        {ZENDNN_ARG_DST, conv1_dst_memory}});
  else
    net_args.push_back({{ZENDNN_ARG_SRC, conv1_src_memory},
                        {ZENDNN_ARG_WEIGHTS, conv1_weights_memory},
                        {ZENDNN_ARG_BIAS, conv1_bias_memory},
                        {ZENDNN_ARG_DST, conv1_dst_memory}});

  assert(net.size() == net_args.size() && "something is missing");
  for (size_t i = 0; i < net.size(); ++i) {
    net.at(i).execute(s, net_args.at(i));
  }

  if (res <= 0) {
    TensorShape filter_tf_shape;
    Tensor *filter_tensor_ptr = nullptr;
    filter_tf_shape.AddDim(conv1_weights_memory.get_desc().get_size());
    ((OpKernelContext *)context)
        ->allocate_temp((is_input_float) ? DT_FLOAT : DT_BFLOAT16,
                        filter_tf_shape, ((Tensor *)cached_filter_data_));
    size_t cached_filter_data_size = conv1_weights_memory.get_desc().get_size();
    T *weights_data = static_cast<T *>(conv1_weights_memory.get_data_handle());
    memcpy(static_cast<T *>(((Tensor *)cached_filter_data_)->flat<T>().data()),
           weights_data, cached_filter_data_size);
  }

  if (reorder_after && blocked) {
    reorder(conv1_dst_memory, conv1_dst_memory_new)
        .execute(s, conv1_dst_memory, conv1_dst_memory_new);
  }
}
template void zenConvolution2DDepthwise<float>(
    zendnn::engine, zendnn::stream, zendnn::primitive_attr, void *, int, int,
    int, int, void *, int, int, int, float, float, float, float, int, int,
    void *, void *, int, int, bool, bool, void *, void *);

template void zenConvolution2DDepthwise<bfloat16>(
    zendnn::engine, zendnn::stream, zendnn::primitive_attr, void *, int, int,
    int, int, void *, int, int, int, float, float, float, float, int, int,
    void *, void *, int, int, bool, bool, void *, void *);

template <typename T>
void zenConvolution2DBiasOrRelu(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void *input_array, int batch_size, int channels, int height, int width,
    void *filter_array, int output_channels, int kernel_h, int kernel_w,
    float pad_t, float pad_l, float pad_b, float pad_r, int stride_h,
    int stride_w, void *bias_array, void *output_array, int out_height,
    int out_width, bool reorder_before, bool reorder_after,
    void *cached_filter_data_, void *context) {
  using tag = memory::format_tag;
  using dt = memory::data_type;
  bool is_input_float = std::is_same<T, float>::value;

  zendnnEnv zenEnvObj = readEnv();
  bool blocked = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT1;
  bool blockedNHWC = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT2;
  // Note: we are forcing it through BlockedNHWC when input is bf16 towards beta
  // support for this release.
  if (!is_input_float) {
    bool result =
        tensorflow::port::TestCPUFeature(tensorflow::port::CPUFeature::AVX512F);
    if (!result) {
      OP_REQUIRES_OK(
          (OpKernelContext *)context,
          errors::Internal(
              "BF16 AVX512 instruction set is not supported in the machine."));
    }
    blocked = 0;
    blockedNHWC = 1;
  }
  if (output_channels % 8 != 0 && blocked && !blockedNHWC) {
    OP_REQUIRES_OK(
        (OpKernelContext *)context,
        errors::Internal("ZENDNN_BLOCKED_FORMAT not supported for this model, "
                         "Please use another data format."));
  }

  memory::dims conv1_src_tz = {batch_size, channels, height, width};
  memory::dims conv1_weights_tz = {output_channels, channels, kernel_h,
                                   kernel_w};
  memory::dims conv1_bias_tz = {output_channels};
  memory::dims conv1_dst_tz = {batch_size, output_channels, out_height,
                               out_width};
  memory::dims conv1_strides = {stride_h, stride_w};
  memory::dims conv1_padding1 = {pad_t, pad_l};
  memory::dims conv1_padding2 = {pad_b, pad_r};

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  const Tensor &cached_filter_data_tensor = *(Tensor *)cached_filter_data_;

  int res = cached_filter_data_tensor.NumElements();
  void *filter_data = NULL;

  zendnn::memory user_weights_memory = memory(
      {{conv1_weights_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::hwcn},
      eng, filter_array);
  zendnn::memory conv1_user_bias_memory =
      memory({{conv1_bias_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::x},
             eng, bias_array);

  if (blocked || blockedNHWC) {
    zendnnInfo(ZENDNN_FWKLOG,
               "zenConvolution2DBiasOrRelu (TF kernel): New API for DIRECT "
               "CONV zenConvolution2DBiasOrRelu");

    zendnn::memory user_src_memory;
    zendnn::memory conv1_dst_memory, conv1_dst_memory_new;
    if (blocked) {
      if (reorder_before) {
        user_src_memory =
            memory({{conv1_src_tz}, dt::f32, tag::nhwc}, eng, input_array);
      } else {
        user_src_memory =
            memory({{conv1_src_tz}, dt::f32, tag::aBcd8b}, eng, input_array);
      }
      if (reorder_after) {
        conv1_dst_memory = memory({{conv1_dst_tz}, dt::f32, tag::aBcd8b}, eng);
        conv1_dst_memory_new =
            memory({{conv1_dst_tz}, dt::f32, tag::nhwc}, eng, output_array);
      } else {
        conv1_dst_memory =
            memory({{conv1_dst_tz}, dt::f32, tag::aBcd8b}, eng, output_array);
      }
    } else {
      user_src_memory = memory(
          {{conv1_src_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::nhwc},
          eng, input_array);
      conv1_dst_memory = memory(
          {{conv1_dst_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::nhwc},
          eng, output_array);
    }

    memory::desc conv1_src_md = memory::desc(
        {conv1_src_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::any);
    memory::desc conv1_bias_md = memory::desc(
        {conv1_bias_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::any);
    memory::desc conv1_weights_md = memory::desc(
        {conv1_weights_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::any);
    memory::desc conv1_dst_md = memory::desc(
        {conv1_dst_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::aBcd8b);

    if (blockedNHWC) {
      conv1_src_md = memory::desc(
          {conv1_src_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::nhwc);
      conv1_bias_md = memory::desc(
          {conv1_bias_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::x);
      conv1_weights_md = memory::desc(
          {conv1_weights_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::any);
      conv1_dst_md = memory::desc(
          {conv1_dst_tz}, (is_input_float) ? dt::f32 : dt::bf16, tag::nhwc);
    }
    // TODO: Current there is no default consructor to create conv desc
    convolution_forward::desc conv1_desc = convolution_forward::desc(
        prop_kind::forward_inference, algorithm::convolution_direct,
        conv1_src_md, conv1_weights_md, conv1_bias_md, conv1_dst_md,
        conv1_strides, conv1_padding1, conv1_padding2);
    if (!bias_array)
      conv1_desc = convolution_forward::desc(
          prop_kind::forward_inference, algorithm::convolution_direct,
          conv1_src_md, conv1_weights_md, conv1_dst_md, conv1_strides,
          conv1_padding1, conv1_padding2);

    convolution_forward::primitive_desc conv1_prim_desc =
        convolution_forward::primitive_desc(conv1_desc, conv_attr, eng);

    zendnn::memory conv1_src_memory = user_src_memory;
    if (blocked) {
      if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
        conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
        if (reorder_before) {
          net.push_back(reorder(user_src_memory, conv1_src_memory));
          net_args.push_back({{ZENDNN_ARG_SRC, user_src_memory},
                              {ZENDNN_ARG_DST, conv1_src_memory}});
        }
      }
    }

    zendnn::memory conv1_weights_memory = user_weights_memory;
    if (res <= 0) {
      if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
        net.push_back(reorder(user_weights_memory, conv1_weights_memory));
        net_args.push_back({{ZENDNN_ARG_SRC, user_weights_memory},
                            {ZENDNN_ARG_DST, conv1_weights_memory}});
      }
    } else {
      filter_data = static_cast<T *>(
          const_cast<T *>(cached_filter_data_tensor.flat<T>().data()));
      conv1_weights_memory =
          memory(conv1_prim_desc.weights_desc(), eng, filter_data);
    }

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
    for (size_t i = 0; i < net.size(); ++i) {
      net.at(i).execute(s, net_args.at(i));
    }
    // To Do - will cleanup or improve over subsequent checkins
    if (res <= 0) {
      TensorShape filter_tf_shape;
      Tensor *filter_tensor_ptr = nullptr;
      filter_tf_shape.AddDim(conv1_weights_memory.get_desc().get_size());
      ((OpKernelContext *)context)
          ->allocate_temp((is_input_float) ? DT_FLOAT : DT_BFLOAT16,
                          filter_tf_shape, ((Tensor *)cached_filter_data_));
      size_t cached_filter_data_size =
          conv1_weights_memory.get_desc().get_size();
      T *weights_data =
          static_cast<T *>(conv1_weights_memory.get_data_handle());
      memcpy(
          static_cast<T *>(((Tensor *)cached_filter_data_)->flat<T>().data()),
          weights_data, cached_filter_data_size);
    }

    if (reorder_after && blocked) {
      reorder(conv1_dst_memory, conv1_dst_memory_new)
          .execute(s, conv1_dst_memory, conv1_dst_memory_new);
    }
  } else {
    zendnnInfo(ZENDNN_FWKLOG,
               "zenConvolution2DBiasOrRelu (TF kernel): New API for GEMM CONV "
               "zenConvolution2DBiasOrRelu");
    zendnn::memory user_src_memory =
        memory({{conv1_src_tz}, dt::f32, tag::nhwc}, eng, input_array);
    zendnn::memory conv1_dst_memory =
        memory({{conv1_dst_tz}, dt::f32, tag::nhwc}, eng, output_array);
    memory::desc conv1_src_md =
        memory::desc({conv1_src_tz}, dt::f32, tag::nhwc);
    memory::desc conv1_bias_md =
        memory::desc({conv1_bias_tz}, dt::f32, tag::any);
    memory::desc conv1_weights_md =
        memory::desc({conv1_weights_tz}, dt::f32, tag::hwcn);
    memory::desc conv1_dst_md =
        memory::desc({conv1_dst_tz}, dt::f32, tag::nhwc);
    convolution_forward::desc conv1_desc = convolution_forward::desc(
        prop_kind::forward_inference, algorithm::convolution_gemm, conv1_src_md,
        conv1_weights_md, conv1_bias_md, conv1_dst_md, conv1_strides,
        conv1_padding1, conv1_padding2);
    convolution_forward::primitive_desc conv1_prim_desc =
        convolution_forward::primitive_desc(conv1_desc, conv_attr, eng);

    net.push_back(convolution_forward(conv1_prim_desc));
    net_args.push_back({{ZENDNN_ARG_SRC, user_src_memory},
                        {ZENDNN_ARG_WEIGHTS, user_weights_memory},
                        {ZENDNN_ARG_BIAS, conv1_user_bias_memory},
                        {ZENDNN_ARG_DST, conv1_dst_memory}});
    assert(net.size() == net_args.size() && "something is missing");
    for (size_t i = 0; i < net.size(); ++i) {
      net.at(i).execute(s, net_args.at(i));
    }
  }
}

template void zenConvolution2DBiasOrRelu<float>(
    zendnn::engine, zendnn::stream, zendnn::primitive_attr, void *, int, int,
    int, int, void *, int, int, int, float, float, float, float, int, int,
    void *, void *, int, int, bool, bool, void *, void *);
template void zenConvolution2DBiasOrRelu<bfloat16>(
    zendnn::engine, zendnn::stream, zendnn::primitive_attr, void *, int, int,
    int, int, void *, int, int, int, float, float, float, float, int, int,
    void *, void *, int, int, bool, bool, void *, void *);

void zenConvolution2DBatchNormOrRelu(
    zendnn::engine eng, zendnn::stream s, zendnn::primitive_attr conv_attr,
    void *input_array, int batch_size, int channels, int height, int width,
    void *filter_array, int output_channels, int kernel_h, int kernel_w,
    float pad_t, float pad_l, float pad_b, float pad_r, int stride_h,
    int stride_w, void *bias_array, void *batch_norm_scale,
    void *batch_norm_mean, void *batch_norm_offset, void *elementwise_input,
    void *output_array, int out_height, int out_width, bool reluFused,
    bool batchNormFused, bool reorder_before, bool reorder_after,
    void *cached_filter_data_, void *context) {
  using tag = memory::format_tag;
  using dt = memory::data_type;

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

  zendnnEnv zenEnvObj = readEnv();
  bool blocked = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT1;
  bool blockedNHWC = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT2;

  const Tensor &cached_filter_data_tensor = *(Tensor *)cached_filter_data_;

  int res = cached_filter_data_tensor.NumElements();
  void *filter_data = NULL;

  zendnn::memory user_weights_memory =
      memory({{conv1_weights_tz}, dt::f32, tag::hwcn}, eng, filter_array);
  zendnn::memory conv1_user_bias_memory =
      memory({{conv1_bias_tz}, dt::f32, tag::x}, eng, bias_array);

  if (blocked || blockedNHWC) {
    if (elementwise_input)
      zendnnInfo(ZENDNN_FWKLOG,
                 "zenConvolution2DBatchNormOrRelu (TF kernel): New API for "
                 "DIRECT CONV zenConvolution2DBatchNormSum");
    else
      zendnnInfo(ZENDNN_FWKLOG,
                 "zenConvolution2DBatchNormOrRelu (TF kernel): New API for "
                 "DIRECT CONV zenConvolution2DBatchNormOrRelu");

    zendnn::memory user_src_memory;
    zendnn::memory elementwise_memory, elementwise_memory_new;
    zendnn::memory conv1_dst_memory, conv1_dst_memory_new;
    if (blocked) {
      if (reorder_before) {
        user_src_memory =
            memory({{conv1_src_tz}, dt::f32, tag::nhwc}, eng, input_array);
        if (elementwise_input) {
          elementwise_memory = memory({{conv1_dst_tz}, dt::f32, tag::nhwc}, eng,
                                      elementwise_input);
          elementwise_memory_new =
              memory({{conv1_dst_tz}, dt::f32, tag::aBcd8b}, eng);
        }
      } else {
        user_src_memory =
            memory({{conv1_src_tz}, dt::f32, tag::aBcd8b}, eng, input_array);
        if (elementwise_input) {
          elementwise_memory = memory({{conv1_dst_tz}, dt::f32, tag::aBcd8b},
                                      eng, elementwise_input);
          elementwise_memory_new = elementwise_memory;
        }
      }

      if (reorder_after) {
        conv1_dst_memory = memory({{conv1_dst_tz}, dt::f32, tag::aBcd8b}, eng);
        conv1_dst_memory_new =
            memory({{conv1_dst_tz}, dt::f32, tag::nhwc}, eng, output_array);
      } else {
        conv1_dst_memory =
            memory({{conv1_dst_tz}, dt::f32, tag::aBcd8b}, eng, output_array);
      }
    }

    else {
      user_src_memory =
          memory({{conv1_src_tz}, dt::f32, tag::nhwc}, eng, input_array);
      elementwise_memory =
          memory({{conv1_dst_tz}, dt::f32, tag::nhwc}, eng, elementwise_input);
      conv1_dst_memory =
          memory({{conv1_dst_tz}, dt::f32, tag::nhwc}, eng, output_array);
    }

    memory::desc conv1_src_md = memory::desc({conv1_src_tz}, dt::f32, tag::any);
    memory::desc conv1_bias_md =
        memory::desc({conv1_bias_tz}, dt::f32, tag::any);
    memory::desc conv1_weights_md =
        memory::desc({conv1_weights_tz}, dt::f32, tag::any);
    memory::desc conv1_dst_md =
        memory::desc({conv1_dst_tz}, dt::f32, tag::aBcd8b);
    if (blockedNHWC) {
      conv1_src_md = memory::desc({conv1_src_tz}, dt::f32, tag::nhwc);
      conv1_bias_md = memory::desc({conv1_bias_tz}, dt::f32, tag::x);
      conv1_weights_md = memory::desc({conv1_weights_tz}, dt::f32, tag::any);
      conv1_dst_md = memory::desc({conv1_dst_tz}, dt::f32, tag::nhwc);
    }

    // TODO: Current there is no default consructor to create conv desc
    convolution_forward::desc conv1_desc = convolution_forward::desc(
        prop_kind::forward_inference, algorithm::convolution_direct,
        conv1_src_md, conv1_weights_md, conv1_bias_md, conv1_dst_md,
        conv1_strides, conv1_padding1, conv1_padding2);
    if (!bias_array)
      conv1_desc = convolution_forward::desc(
          prop_kind::forward_inference, algorithm::convolution_direct,
          conv1_src_md, conv1_weights_md, conv1_dst_md, conv1_strides,
          conv1_padding1, conv1_padding2);
    convolution_forward::primitive_desc conv1_prim_desc =
        convolution_forward::primitive_desc(conv1_desc, conv_attr, eng);

    zendnn::memory conv1_src_memory = user_src_memory;
    if (blocked) {
      if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
        conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
        if (reorder_before) {
          net.push_back(reorder(user_src_memory, conv1_src_memory));
          net_args.push_back({{ZENDNN_ARG_SRC, user_src_memory},
                              {ZENDNN_ARG_DST, conv1_src_memory}});
        }
      }
    }
    zendnn::memory conv1_weights_memory = user_weights_memory;
    if (res <= 0) {
      if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
        conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
        net.push_back(reorder(user_weights_memory, conv1_weights_memory));
        net_args.push_back({{ZENDNN_ARG_SRC, user_weights_memory},
                            {ZENDNN_ARG_DST, conv1_weights_memory}});
      }
    } else {
      filter_data = static_cast<float *>(
          const_cast<float *>(cached_filter_data_tensor.flat<float>().data()));
      conv1_weights_memory =
          memory(conv1_prim_desc.weights_desc(), eng, filter_data);
    }

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
    for (size_t i = 0; i < net.size(); ++i) {
      net.at(i).execute(s, net_args.at(i));
    }

    if (res <= 0) {
      TensorShape filter_tf_shape;

      filter_tf_shape.AddDim(conv1_weights_memory.get_desc().get_size());

      ((OpKernelContext *)context)
          ->allocate_temp(DT_FLOAT, filter_tf_shape,
                          ((Tensor *)cached_filter_data_));
      size_t cached_filter_data_size =
          conv1_weights_memory.get_desc().get_size();
      float *weights_data =
          static_cast<float *>(conv1_weights_memory.get_data_handle());
      memcpy(static_cast<float *>(
                 ((Tensor *)cached_filter_data_)->flat<float>().data()),
             weights_data, cached_filter_data_size);
    }
    int no_of_threads = zenEnvObj.omp_num_threads;

    float *elementwise_input_new = NULL;
    if (elementwise_input) {
      if (reorder_before && blocked) {
        reorder(elementwise_memory, elementwise_memory_new)
            .execute(s, elementwise_memory, elementwise_memory_new);
      }
      if (blocked) {
        elementwise_input_new =
            (float *)elementwise_memory_new.get_data_handle();
      } else {
        elementwise_input_new = (float *)elementwise_memory.get_data_handle();
      }
    }
    if (reorder_after && blocked) {
      zenPostOps(zenEnvObj, (float *)conv1_dst_memory.get_data_handle(),
                 (const float *)elementwise_input_new, out_height, out_width,
                 output_channels, 0, 0, NULL, reluFused, false,
                 (const float *)batch_norm_scale, no_of_threads,
                 (const float *)batch_norm_offset,
                 (const float *)batch_norm_mean, batch_size);

      reorder(conv1_dst_memory, conv1_dst_memory_new)
          .execute(s, conv1_dst_memory, conv1_dst_memory_new);

    } else {
      if (!blocked) {
        float *bias = (float *)malloc(sizeof(float) * output_channels);
#pragma omp parallel for num_threads(no_of_threads)
        for (int r = 0; r < output_channels; r++) {
          bias[r] =
              ((float *)batch_norm_offset)[r] -
              (((float *)batch_norm_scale)[r] * ((float *)batch_norm_mean)[r]);
        }
        unsigned long biasOffset = 0;
        for (int i = 0; i < batch_size; ++i) {
          biasOffset = (out_height * out_width) * output_channels * i;
          zenPostOps(zenEnvObj, (float *)output_array,
                     (const float *)elementwise_input_new, out_height,
                     out_width, output_channels, output_channels, biasOffset,
                     bias, reluFused, false, (const float *)batch_norm_scale,
                     no_of_threads);
        }
      } else {
        zenPostOps(zenEnvObj, (float *)output_array,
                   (const float *)elementwise_input_new, out_height, out_width,
                   output_channels, 0, 0, NULL, reluFused, false,
                   (const float *)batch_norm_scale, no_of_threads,
                   (const float *)batch_norm_offset,
                   (const float *)batch_norm_mean, batch_size);
      }
    }
  } else {
    if (elementwise_input)
      zendnnInfo(ZENDNN_FWKLOG,
                 "zenConvolution2DBatchNormOrRelu (TF kernel): New API for "
                 "GEMM CONV zenConvolution2DBatchNormSum");
    else
      zendnnInfo(ZENDNN_FWKLOG,
                 "zenConvolution2DBatchNormOrRelu (TF kernel): New API for "
                 "GEMM CONV zenConvolution2DBatchNormOrRelu");
    zendnn::memory user_src_memory =
        memory({{conv1_src_tz}, dt::f32, tag::nhwc}, eng, input_array);
    zendnn::memory conv1_dst_memory =
        memory({{conv1_dst_tz}, dt::f32, tag::nhwc}, eng, output_array);
    zendnn::memory batch_norm_scale_memory =
        memory({{batch_norm_tz}, dt::f32, tag::x}, eng, batch_norm_scale);
    zendnn::memory batch_norm_mean_memory =
        memory({{batch_norm_tz}, dt::f32, tag::x}, eng, batch_norm_mean);
    zendnn::memory batch_norm_offset_memory =
        memory({{batch_norm_tz}, dt::f32, tag::x}, eng, batch_norm_offset);

    memory::desc conv1_src_md =
        memory::desc({conv1_src_tz}, dt::f32, tag::nhwc);
    memory::desc conv1_bias_md =
        memory::desc({conv1_bias_tz}, dt::f32, tag::any);
    memory::desc conv1_weights_md =
        memory::desc({conv1_weights_tz}, dt::f32, tag::hwcn);
    memory::desc conv1_dst_md =
        memory::desc({conv1_dst_tz}, dt::f32, tag::nhwc);

    memory::desc batch_norm_scale_md =
        memory::desc({batch_norm_tz}, dt::f32, tag::any);
    memory::desc batch_norm_mean_md =
        memory::desc({batch_norm_tz}, dt::f32, tag::any);
    memory::desc batch_norm_offset_md =
        memory::desc({batch_norm_tz}, dt::f32, tag::any);

    convolution_forward::desc conv1_desc = convolution_forward::desc(
        prop_kind::forward_inference, algorithm::convolution_gemm, conv1_src_md,
        conv1_weights_md, conv1_bias_md, conv1_dst_md, conv1_strides,
        conv1_padding1, conv1_padding2, reluFused, batchNormFused,
        batch_norm_scale_md, batch_norm_mean_md, batch_norm_offset_md);
    convolution_forward::primitive_desc conv1_prim_desc =
        convolution_forward::primitive_desc(conv1_desc, conv_attr, eng);

    net.push_back(convolution_forward(conv1_prim_desc));
    net_args.push_back({{ZENDNN_ARG_SRC, user_src_memory},
                        {ZENDNN_ARG_WEIGHTS, user_weights_memory},
                        {ZENDNN_ARG_BIAS, conv1_user_bias_memory},
                        {ZENDNN_ARG_DST, conv1_dst_memory},
                        {ZENDNN_ARG_BN_SCALE, batch_norm_scale_memory},
                        {ZENDNN_ARG_BN_MEAN, batch_norm_mean_memory},
                        {ZENDNN_ARG_BN_OFFSET, batch_norm_offset_memory}

    });

    assert(net.size() == net_args.size() && "something is missing");
    for (size_t i = 0; i < net.size(); ++i) {
      net.at(i).execute(s, net_args.at(i));
    }
  }
}

void zenMatMulBiasRelu(zendnn::engine eng, zendnn::stream s,
                       zendnn::primitive_attr matmul_attr,
                       const bool transpose_input, const bool transpose_filter,
                       const int images, const int channels, const int filters,
                       void *input, void *filter, void *bias, void *output) {
  using tag = memory::format_tag;
  using dt = memory::data_type;

  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;

  // dimensions of matmul source, weights, bias and destination tensors
  memory::dims src_dims = {images, channels};
  memory::dims weights_dims = {filters, channels};
  memory::dims bias_dims = {filters};
  memory::dims dst_dims = {images, filters};

  // memory descriptors for matmul source, weights, bias and destination
  memory::desc src_md = memory::desc({src_dims}, dt::f32, tag::nc);
  tag weights_format = transpose_filter ? tag::oi : tag::io;
  memory::desc weights_md =
      memory::desc({weights_dims}, dt::f32, weights_format);
  memory::desc bias_md = memory::desc({bias_dims}, dt::f32, tag::x);
  memory::desc dst_md = memory::desc({dst_dims}, dt::f32, tag::nc);

  // memory objects for matmul source, weights, bias and destination
  zendnn::memory src_memory = memory(src_md, eng, input);
  zendnn::memory weights_memory = memory(weights_md, eng, filter);
  zendnn::memory bias_memory = memory(bias_md, eng, bias);
  zendnn::memory dst_memory = memory(dst_md, eng, output);

  // Operation descriptor for matmul
  inner_product_forward::desc matmul_d = inner_product_forward::desc(
      prop_kind::forward_inference, src_md, weights_md, bias_md, dst_md);

  // Primitive descriptor for matmul
  inner_product_forward::primitive_desc matmul_pd =
      inner_product_forward::primitive_desc(matmul_d, matmul_attr, eng);

  // Primitive for matmul
  net.push_back(inner_product_forward(matmul_pd));
  net_args.push_back({{ZENDNN_ARG_SRC, src_memory},
                      {ZENDNN_ARG_WEIGHTS, weights_memory},
                      {ZENDNN_ARG_BIAS, bias_memory},
                      {ZENDNN_ARG_DST, dst_memory}});

  // execute the matmul primitive
  assert(net.size() == net_args.size() && "something is missing");
  for (size_t i = 0; i < net.size(); ++i) {
    net.at(i).execute(s, net_args.at(i));
  }
}
