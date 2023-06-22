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

#include <vector>

#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/tensor_format.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "zendnn.hpp"
#include "zendnn_helper.hpp"
#include "zendnn_logging.hpp"

using namespace zendnn;
using namespace std;

namespace tensorflow {

template <typename T>
class ZenSoftmaxOp : public OpKernel {
 private:
  bool reorder_before, reorder_after, reset;
  int in_links, out_links;
  TensorFormat data_format_;

 public:
  explicit ZenSoftmaxOp(OpKernelConstruction *ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reorder_before", &reorder_before));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reorder_after", &reorder_after));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reset", &reset));
    string data_format_str;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("data_format", &data_format_str));
    OP_REQUIRES(ctx, FormatFromString(data_format_str, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES(ctx, data_format_ == FORMAT_NHWC,
                errors::Unimplemented("ZenDNN Softmax implementation supports "
                                      "NHWC tensor format only for now."));
  }

  void Compute(OpKernelContext *ctx) override {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: ZenSoftmax (TF kernel): In Compute!");

    ZenExecutor *ex = ex->getInstance();
    engine eng = ex->getEngine();
    stream s = ex->getStream();

    vector<primitive> net;
    vector<unordered_map<int, memory>> net_args;

    // src_tensor now points to the 0-th input of
    // global data struct "ctx"
    const Tensor &input = ctx->input(0);
    const int input_dims = input.shape().dims();
    T *input_array = const_cast<T *>(input.template flat<T>().data());
    memory::dims src_dims(input_dims);
    for (int d = 0; d < input_dims; ++d) {
      src_dims[d] = input.shape().dim_size(d);
    }
    // Update the output type
    zenTensorType out_type = zenTensorType::FLOAT;

    // Allocating memory for output tensor
    // Output tensor shape is same as input
    TensorShape out_shape = input.shape();

    zendnnEnv zenEnvObj = readEnv();
    Tensor *output = nullptr;
    int zenEnableMemPool =
        zenEnvObj.zenEnableMemPool && ctx->expected_output_dtype(0) == DT_FLOAT;
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
            ctx, &output, out_shape, out_links, reset, out_type);
        if (status) {
          zenEnableMemPool = false;
        }
      } else {
        zenEnableMemPool = false;
      }
    }
    if (!zenEnableMemPool) {
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &output));
    }

    T *output_array = output->template flat<T>().data();
    memory::dims output_dims = src_dims;
    int axis;

    memory::format_tag layout_type;
    // We use axis to define on which dimension to do softmax. Softmax axis
    // is attached to logical dimensions which always go in a specific
    // order. For softmax it would be N, C, H, W.
    // For a 4D Tensor with softmax axis = 1:
    // {{<physical layout>N,C,H,W}, data_type::f32, format_tag::nhwc};
    // For a 4D Tensor with softmax axis = 3:
    // {{<physical layout>N,H,W,C}, data_type::f32, format_tag::nchw};
    switch (input_dims) {
      case 1:
        layout_type = memory::format_tag::x;  // a
        axis = 0;
        break;
      case 2:
        layout_type = memory::format_tag::nc;  // ab
        axis = 1;
        break;
      case 3:
        layout_type = memory::format_tag::tnc;  // abc
        axis = 2;
        break;
      case 4:
        layout_type = memory::format_tag::nchw;  // abcd
        axis = 3;
        break;
      // case 5:
      //    layout_type = memory::format_tag::ndhwc;
      // break;
      default:
        OP_REQUIRES_OK(ctx, errors::Aborted("Input dims must be <= 4 and >=1"));
        return;
    }

    // Create softmax memory for src, dst,
    using tag = memory::format_tag;
    using dt = memory::data_type;
    zendnn::memory src_memory =
        memory({{src_dims}, dt::f32, layout_type}, eng, input_array);
    zendnn::memory dst_memory =
        memory({{output_dims}, dt::f32, layout_type}, eng, output_array);

    // Create memory descriptor for src
    memory::desc src_md = memory::desc({src_dims}, dt::f32, layout_type);

    // Create forward and primitive descriptor for softmax op
    softmax_forward::desc softmax_fwd_desc =
        softmax_forward::desc(prop_kind::forward_inference, src_md, axis);
    softmax_forward::primitive_desc softmax_fwd_pd =
        softmax_forward::primitive_desc(softmax_fwd_desc, eng);

    // Finally creating the "softmax op" using the primitive descriptor, src
    // and dst
    auto softmax_fwd = softmax_forward(softmax_fwd_pd);
    net.push_back(softmax_fwd);
    net_args.push_back(
        {{ZENDNN_ARG_SRC, src_memory}, {ZENDNN_ARG_DST, dst_memory}});
    assert(net.size() == net_args.size() && "something is missing");
    for (size_t i = 0; i < net.size(); ++i) {
      net.at(i).execute(s, net_args.at(i));
    }

    zendnnInfo(ZENDNN_FWKLOG,
               "ZEN-OP-DEF: ZenSoftmax (TF kernel): Out Compute!");

    // If ZenMemPool Optimization is enabled(default), update the state of
    //  Memory pool based on input_array address
    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      zenPoolBuffer->zenMemPoolFree(ctx, (void *)input_array);
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("ZenSoftmax").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    ZenSoftmaxOp<float>);

}  // namespace tensorflow
