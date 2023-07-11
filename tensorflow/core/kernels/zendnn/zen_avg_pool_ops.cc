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

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/pooling_ops_common.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/zen_util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "zendnn.hpp"
#include "zendnn_helper.hpp"
#include "zendnn_logging.hpp"

#define NEW_API 1
using namespace zendnn;
using namespace std;

namespace tensorflow {

class NoOp : public OpKernel {
 public:
  explicit NoOp(OpKernelConstruction *context) : OpKernel(context) {}
  void Compute(OpKernelContext *context) override {}
  bool IsExpensive() override { return false; }
};

template <typename Toutput>
class ZenAvgQuantizedPoolOp : public OpKernel {
 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool reorder_before, reorder_after, reset;
  int in_links, out_links;

 public:
  explicit ZenAvgQuantizedPoolOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(
        context, ksize_.size() == 4,
        errors::InvalidArgument("Kernel size field must specify 4 dimensions"));

    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument(
                    "Sliding window stride field must specify 4 dimensions"));

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("reorder_before", &reorder_before));
    OP_REQUIRES_OK(context, context->GetAttr("reorder_after", &reorder_after));

    OP_REQUIRES_OK(context, context->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(context, context->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(context, context->GetAttr("reset", &reset));
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input = context->input(0);
    auto input_map = input.tensor<quint8, 4>();  // experimented and proven that
                                                 // it is row-major
    const quint8 *input_array = input_map.data();

    PoolParameters params{
        context,     ksize_,       stride_, padding_, /*explicit_padding*/ {},
        FORMAT_NHWC, input.shape()};
    TensorShape out_shape;
    OP_REQUIRES_OK(context,
		  params.forward_output_shape(&out_shape));
    Tensor *output = nullptr, *output_min = nullptr, *output_max = nullptr;
    Toutput *output_array;
    // Update the output type
    zenTensorType out_type = zenTensorType::QINT8;
    if (std::is_same<Toutput, quint8>::value) {
      out_type = zenTensorType::QUINT8;
    }

    // Allocate output
    zendnnEnv zenEnvObj = readEnv();
    int zenEnableMemPool = zenEnvObj.zenEnableMemPool &&
                           (context->expected_output_dtype(0) == DT_QINT8 ||
                            context->expected_output_dtype(0) == DT_QUINT8);
    ZenMemoryPool<Toutput> *zenPoolBuffer = NULL;

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
    output_array = const_cast<Toutput *>(output_map.data());

    const Tensor &min_input_t = context->input(1);
    const Tensor &max_input_t = context->input(2);
    const float min_input = min_input_t.flat<float>()(0);
    const float max_input = max_input_t.flat<float>()(0);

    TensorShape zen_out_shape_max, zen_out_shape_min;

    OP_REQUIRES_OK(context,
                   context->allocate_output(1, zen_out_shape_min, &output_min));
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, zen_out_shape_max, &output_max));

    output_min->flat<float>()(0) = min_input;
    output_max->flat<float>()(0) = max_input;

    // Compute Pooling Parameters
    const int batch_size = params.tensor_in_batch;
    const int image_height = params.tensor_in_rows;
    const int image_width = params.tensor_in_cols;
    const int image_channels = params.depth;

    int stride_h, stride_w, filter_height, filter_width;
    int padding_h_top, padding_h_bottom, padding_w_left, padding_w_right;
    int output_height, output_width;

    stride_h = stride_[1];
    stride_w = stride_[2];
    filter_height = ksize_[1];
    filter_width = ksize_[2];

    // Compute Padding Parameters
    if (!(padding_ == SAME)) {
      padding_h_top = padding_h_bottom = padding_w_left = padding_w_right = 0;
      output_height =
          std::floor(float(image_height - filter_height) / float(stride_h)) + 1;
      output_width =
          std::floor(float(image_width - filter_width) / float(stride_w)) + 1;
    } else {
      int total_pad_h, total_pad_w;
      int mod_h, mod_w;
      mod_h = image_height % stride_h;
      mod_w = image_width % stride_w;

      total_pad_h =
          std::max(filter_height - (mod_h == 0 ? stride_h : mod_h), 0);
      padding_h_top =
          (total_pad_h / 2);  // integer division equivalent to floor
      padding_h_bottom = total_pad_h - padding_h_top;

      total_pad_w = std::max(filter_width - (mod_w == 0 ? stride_w : mod_w), 0);
      padding_w_left =
          (total_pad_w / 2);  // integer division equivalent to floor
      padding_w_right = total_pad_w - padding_w_left;
      output_height = std::ceil(float(image_height) / float(stride_h));
      output_width = std::ceil(float(image_width) / float(stride_w));
    }

    // Primitive creation and Execution
    using tag = memory::format_tag;
    using dt = memory::data_type;
    ZenExecutor *ex = ex->getInstance();
    engine eng = ex->getEngine();
    stream s = ex->getStream();
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    memory::dim out_height, out_width;
    out_height = (params.tensor_in_rows - params.window_rows + padding_h_top +
                  padding_h_bottom) /
                     params.row_stride +
                 1;
    out_width = (params.tensor_in_cols - params.window_cols + padding_w_left +
                 padding_w_right) /
                    params.col_stride +
                1;

    memory::dims pool_src_tz = {params.tensor_in_batch, params.depth,
                                params.tensor_in_rows, params.tensor_in_cols};
    memory::dims pool_dst_tz = {params.tensor_in_batch, params.depth,
                                out_height, out_width};
    memory::dims pool_kernel = {params.window_rows, params.window_cols};
    memory::dims pool_strides = {params.row_stride, params.col_stride};
    memory::dims pool_padding_l = {padding_h_top, padding_w_left};
    memory::dims pool_padding_r = {padding_h_bottom, padding_w_right};

    zendnn::memory pool_src_memory, pool_dst_memory;
    pool_src_memory =
        memory({{pool_src_tz}, dt::u8, tag::acdb}, eng, (quint8 *)input_array);
    pool_dst_memory =
        memory({{pool_dst_tz}, dt::u8, tag::acdb}, eng, (quint8 *)output_array);

    memory::desc pool_src_md = memory::desc({pool_src_tz}, dt::u8, tag::acdb);
    memory::desc pool_dst_md = memory::desc({pool_dst_tz}, dt::u8, tag::acdb);
    //[Create pooling primitive]
    pooling_forward::desc pool_desc = pooling_forward::desc(
        prop_kind::forward_inference, algorithm::pooling_avg, pool_src_md,
        pool_dst_md, pool_strides, pool_kernel, pool_padding_l, pool_padding_r);
    pooling_forward::primitive_desc pool_pd =
        pooling_forward::primitive_desc(pool_desc, eng);

    net.push_back(pooling_forward(pool_pd));
    net_args.push_back(
        {{ZENDNN_ARG_SRC, pool_src_memory}, {ZENDNN_ARG_DST, pool_dst_memory}});
    for (size_t i = 0; i < net.size(); ++i) {
      net.at(i).execute(s, net_args.at(i));
    }
    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      zenPoolBuffer->zenMemPoolFree(context, (void *)input_array);
    }
  }
};

template <typename T>
class ZenAvgPoolOp : public OpKernel {
 private:
  std::vector<int32> ksize_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool reorder_before, reorder_after, reset;
  int in_links, out_links;

 public:
  explicit ZenAvgPoolOp(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
    OP_REQUIRES(
        context, ksize_.size() == 4,
        errors::InvalidArgument("Kernel size field must specify 4 dimensions"));

    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 4,
                errors::InvalidArgument(
                    "Sliding window stride field must specify 4 dimensions"));

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("reorder_before", &reorder_before));
    OP_REQUIRES_OK(context, context->GetAttr("reorder_after", &reorder_after));

    OP_REQUIRES_OK(context, context->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(context, context->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(context, context->GetAttr("reset", &reset));

    string data_format_str;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(context, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
  }

  void Compute(OpKernelContext *context) override {
    int data_format = (data_format_ == FORMAT_NCHW) ? 1 : 0;

    const Tensor &input = context->input(0);
    auto input_map =
        input.tensor<T, 4>();  // experimented and proven that it is row-major
    const T *input_array = input_map.data();

    PoolParameters params{
        context,      ksize_,       stride_, padding_, /*explict padding*/ {},
        data_format_, input.shape()};
    TensorShape out_shape;
    OP_REQUIRES_OK(context,
		  params.forward_output_shape(&out_shape));
    // Update the output type
    bool is_input_float = std::is_same<T, float>::value;
    zenTensorType out_type =
        (is_input_float) ? zenTensorType::FLOAT : zenTensorType::BFLOAT16;
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

    const int batch_size = params.tensor_in_batch;
    const int image_height = params.tensor_in_rows;
    const int image_width = params.tensor_in_cols;
    const int image_channels = params.depth;

    int stride_h, stride_w, filter_height, filter_width;
    int padding_h_top, padding_h_bottom, padding_w_left, padding_w_right;
    int output_height, output_width;

    stride_h = stride_[1];
    stride_w = stride_[2];

    filter_height = ksize_[1];
    filter_width = ksize_[2];

    // TODO: Define new function compute_padding for this as its used at
    // multiple places
    if (!(padding_ == SAME)) {
      padding_h_top = padding_h_bottom = padding_w_left = padding_w_right = 0;
      output_height =
          std::floor(float(image_height - filter_height) / float(stride_h)) + 1;
      output_width =
          std::floor(float(image_width - filter_width) / float(stride_w)) + 1;
    } else {
      int total_pad_h, total_pad_w;
      int mod_h, mod_w;
      mod_h = image_height % stride_h;
      mod_w = image_width % stride_w;

      total_pad_h =
          std::max(filter_height - (mod_h == 0 ? stride_h : mod_h), 0);
      padding_h_top =
          (total_pad_h / 2);  // integer division equivalent to floor
      padding_h_bottom = total_pad_h - padding_h_top;

      total_pad_w = std::max(filter_width - (mod_w == 0 ? stride_w : mod_w), 0);
      padding_w_left =
          (total_pad_w / 2);  // integer division equivalent to floor
      padding_w_right = total_pad_w - padding_w_left;
      output_height = std::ceil(float(image_height) / float(stride_h));
      output_width = std::ceil(float(image_width) / float(stride_w));
    }

#if NEW_API
    bool blocked = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT1;
    bool blockedNHWC = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT2;
    // Note: we are forcing it through BlockedNHWC when input is bf16 towards
    // beta support for this release.
    if (!is_input_float) {
      bool result = tensorflow::port::TestCPUFeature(
          tensorflow::port::CPUFeature::AVX512F);
      if (!result) {
        OP_REQUIRES_OK((OpKernelContext *)context,
                       errors::Internal("BF16 AVX512 instruction set is not "
                                        "supported in the machine."));
      }
      blocked = 0;
      blockedNHWC = 1;
    }
    if ((blocked || !is_input_float)) {
      zendnnInfo(ZENDNN_FWKLOG,
                 "zenAvgPool (TF kernel): New API for BLOCKED AVGPOOL "
                 "zenAvgPool blocked and blockednhwc",
                 blocked, blockedNHWC);
      using tag = memory::format_tag;
      using dt = memory::data_type;
      ZenExecutor *ex = ex->getInstance();
      engine eng = ex->getEngine();
      stream s = ex->getStream();
      std::vector<primitive> net;
      std::vector<std::unordered_map<int, memory>> net_args;
      auto dtype = std::is_same<T, float>::value ? dt::f32 : dt::bf16;

      memory::dim out_height, out_width;
      out_height = (params.tensor_in_rows - params.window_rows + padding_h_top +
                    padding_h_bottom) /
                       params.row_stride +
                   1;
      out_width = (params.tensor_in_cols - params.window_cols + padding_w_left +
                   padding_w_right) /
                      params.col_stride +
                  1;

      memory::dims pool_src_tz = {params.tensor_in_batch, params.depth,
                                  params.tensor_in_rows, params.tensor_in_cols};
      memory::dims pool_dst_tz = {params.tensor_in_batch, params.depth,
                                  out_height, out_width};
      memory::dims pool_kernel = {params.window_rows, params.window_cols};
      memory::dims pool_strides = {params.row_stride, params.col_stride};
      memory::dims pool_padding_l = {padding_h_top, padding_w_left};
      memory::dims pool_padding_r = {padding_h_bottom, padding_w_right};

      zendnn::memory pool_src_memory;
      zendnn::memory pool_dst_memory, pool_dst_memory_new;

      if (reorder_before || !is_input_float)
        pool_src_memory =
            memory({{pool_src_tz}, dtype, tag::nhwc}, eng, (T *)input_array);
      else
        pool_src_memory =
            memory({{pool_src_tz}, dtype, tag::aBcd8b}, eng, (T *)input_array);

      if (!is_input_float) {
        pool_dst_memory =
            memory({{pool_dst_tz}, dtype, tag::nhwc}, eng, (T *)output_array);
      } else if (reorder_after) {
        pool_dst_memory = memory({{pool_dst_tz}, dtype, tag::aBcd8b}, eng);
        pool_dst_memory_new =
            memory({{pool_dst_tz}, dtype, tag::nhwc}, eng, (T *)output_array);
      } else {
        pool_dst_memory =
            memory({{pool_dst_tz}, dtype, tag::aBcd8b}, eng, (T *)output_array);
        pool_dst_memory_new =
            memory({{pool_dst_tz}, dtype, tag::aBcd8b}, eng, (T *)output_array);
      }

      memory::desc pool_src_md, pool_dst_md;
      if (is_input_float) {
        pool_src_md = memory::desc({pool_src_tz}, dtype, tag::aBcd8b);
        pool_dst_md = memory::desc({pool_dst_tz}, dtype, tag::aBcd8b);
      } else {
        pool_src_md = memory::desc({pool_src_tz}, dtype, tag::nhwc);
        pool_dst_md = memory::desc({pool_dst_tz}, dtype, tag::nhwc);
      }

      //[Create pooling primitive]
      pooling_forward::desc pool_desc = pooling_forward::desc(
          prop_kind::forward_inference, algorithm::pooling_avg, pool_src_md,
          pool_dst_md, pool_strides, pool_kernel, pool_padding_l,
          pool_padding_r);
      pooling_forward::primitive_desc pool_pd =
          pooling_forward::primitive_desc(pool_desc, eng);

      zendnn::memory pool1_src_memory = pool_src_memory;
      if (pool_pd.src_desc() != pool_src_memory.get_desc()) {
        pool1_src_memory = memory(pool_pd.src_desc(), eng);
        if (reorder_before) {
          net.push_back(reorder(pool_src_memory, pool1_src_memory));
          //.execute(s, pool_src_memory, pool1_src_memory);
          net_args.push_back({{ZENDNN_ARG_SRC, pool_src_memory},
                              { ZENDNN_ARG_DST,
                                pool1_src_memory }});
        }
      }

      net.push_back(pooling_forward(pool_pd));
      net_args.push_back({{ZENDNN_ARG_SRC, pool1_src_memory},
                          { ZENDNN_ARG_DST,
                            pool_dst_memory }});

      //[Execute model]
      assert(net.size() == net_args.size() && "something is missing");
      for (size_t i = 0; i < net.size(); ++i) {
        net.at(i).execute(s, net_args.at(i));
      }
      if (reorder_after) {
        reorder(pool_dst_memory, pool_dst_memory_new)
            .execute(s, pool_dst_memory, pool_dst_memory_new);
      }
    } else {
      // TODO:: Create ZenDNN API for ZenDNN Library pooling
      zendnnInfo(ZENDNN_FWKLOG,
                 "zenAvgPool (TF kernel): New API for NHWC AVGPOOL zenAvgPool");
      avg_pooling((float *)input_array, params.tensor_in_batch, params.depth,
                  params.tensor_in_rows, params.tensor_in_cols,
                  params.window_rows, params.window_cols, params.row_stride,
                  params.col_stride, padding_h_top, padding_h_bottom,
                  padding_w_left, padding_w_right, (float *)output_array,
                  data_format);
    }
#else
    avg_pooling(input_array, params.tensor_in_batch, params.depth,
                params.tensor_in_rows, params.tensor_in_cols,
                params.window_rows, params.window_cols, params.row_stride,
                params.col_stride, padding_h_top, padding_h_bottom,
                padding_w_left, padding_w_right, output_array, data_format);

#endif

    // If ZenMemPool Optimization is enabled(default), update the state of
    //  Memory pool based on input_array address
    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      zenPoolBuffer->zenMemPoolFree(context, (void *)input_array);
    }
  }

 private:
  Tensor cached_data_ TF_GUARDED_BY(mu_);
};

REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedAvgPool").Device(DEVICE_CPU).TypeConstraint<quint8>("T"),
    ZenAvgQuantizedPoolOp<quint8>);

REGISTER_KERNEL_BUILDER(
    Name("_ZenQuantizedAvgPool").Device(DEVICE_CPU).TypeConstraint<qint8>("T"),
    ZenAvgQuantizedPoolOp<qint8>);

REGISTER_KERNEL_BUILDER(
    Name("_ZenAvgPool").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    ZenAvgPoolOp<float>);

REGISTER_KERNEL_BUILDER(
    Name("_ZenAvgPool").Device(DEVICE_CPU).TypeConstraint<bfloat16>("T"),
    ZenAvgPoolOp<bfloat16>);

}  // namespace tensorflow
