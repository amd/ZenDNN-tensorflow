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

#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/kernels/zendnn/zen_quantized_ops.h"

namespace tensorflow {

using namespace zendnn;
using dt = zendnn::memory::data_type;
using tag = zendnn::memory::format_tag;

template <typename T, zendnn::algorithm binary_algo>
class ZenVitisAIBinaryOP : public OpKernel {
 public:
  explicit ZenVitisAIBinaryOP(OpKernelConstruction *context)
      : OpKernel(context) {
    ReadParameterFromContext<int>(context, "in_scale_0", &in_scale_0_);
    ReadParameterFromContext<int>(context, "in_scale_1", &in_scale_1_);
    ReadParameterFromContext<int>(context, "out_scale", &out_scale_);

    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));
  }

  void Compute(OpKernelContext *context) override {
    ZenExecutor *ex = ex->getInstance();
    zendnn::engine engine = ex->getEngine();
    zendnn::stream stream = ex->getStream();

    // Grab input 0
    const Tensor &in0 = context->input(0);
    T *in0_array = static_cast<T *>(const_cast<T *>(in0.flat<T>().data()));

    // Grab input 1:
    const Tensor &in1 = context->input(1);
    T *in1_array = static_cast<T *>(const_cast<T *>(in1.flat<T>().data()));

    // Find the largest dims in the two inputs
    int in0_dims = in0.dims();
    int in1_dims = in1.dims();
    int ndims = in0_dims >= in1_dims ? in0_dims : in1_dims;
    OP_REQUIRES(
        context, ndims >= 1,
        errors::InvalidArgument(
            "Input dims must be greater than 1 dimensions, got: ", ndims));

    // Supports two conditions
    // 1. Both inputs should have same number of dims and both should be of the
    // same shape
    // 2. One input has a dimension of 1 and is broadcastable
    bool is_supported = true;
    if (in0_dims == in1_dims) {
      if (in0.shape() != in1.shape()) {
        is_supported = false;
      }
    } else if ((in0.NumElements() != 1) && (in1.NumElements() != 1)) {
      is_supported = false;
    }

    // If not supported, throw error and exit
    if (!is_supported) {
      OP_REQUIRES(context, false,
                  errors::InvalidArgument("Input dims are not compatible: ",
                                          in0.shape().DebugString(), "and ",
                                          in1.shape().DebugString()));
    }

    // output_shape will be the largest shape of the two inputs
    TensorShape out_shape;
    if (in0_dims == ndims) {
      out_shape = in0.shape();
    } else {
      out_shape = in1.shape();
    }

    zendnnEnv zenEnvObj = readEnv();
    int zenPoolEnable = zenEnvObj.zenEnableMemPool &&
                        (context->expected_output_dtype(0) == DT_QINT8 ||
                         context->expected_output_dtype(0) == DT_QUINT8);

    auto out_type = std::is_same<T, quint8>::value ? zenTensorType::QUINT8
                                                   : zenTensorType::QINT8;
    ZenMemoryPool<T> *zenPoolBuffer = nullptr;
    Tensor *output = nullptr;

    bool in0_reuse = false;
    bool in1_reuse = false;

    // Output buffer:
    // (1) Reuse any of the input buffers for output
    // (2) If not (1), then get buffer from ZenMemPool
    // (3) If not (1) and (2), then allocate the buffer for output
    if (context->forward_input_to_output_with_shape(0, 0, out_shape, &output)) {
      in0_reuse = true;
    } else if (context->forward_input_to_output_with_shape(1, 0, out_shape,
                                                           &output)) {
      in1_reuse = true;
    } else {
      // ZenMemPool Optimization reuse o/p tensors from the pool. By default
      // its enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory
      // pool optimization
      // Cases where tensors in pool are not free or requested size is more
      // than available tensor size in Pool, control will fall back to
      // default way of allocation i.e. with allocate_output(..)
      if (zenPoolEnable) {
        unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
        zenPoolBuffer = ZenMemoryPool<T>::getZenMemPool(threadID);
        if (zenPoolBuffer) {
          int status = zenPoolBuffer->acquireZenPoolTensor(
              context, &output, out_shape, zendnn_params_.out_links,
              zendnn_params_.reset, out_type);
          if (status) {
            zenPoolEnable = false;
          }
        } else {
          zenPoolEnable = false;
        }
      }

      if (!zenPoolEnable) {
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, out_shape, &output));
      }
    }

    T *output_array;
    if (ndims == 2) {
      output_array = const_cast<T *>(output->tensor<T, 2>().data());
    } else {
      output_array = const_cast<T *>(output->tensor<T, 4>().data());
    }

    // Set the input shapes for both src
    // If an src is a scalar, the shape needs to be of the same
    // dimension as the other input, filled with 1
    memory::dims src0_dims, src1_dims, dst_dims;
    if (in0.NumElements() == 1) {
      src0_dims.push_back(1);
    } else {
      src0_dims = {in0.dim_size(0), in0.dim_size(3), in0.dim_size(1),
                   in0.dim_size(2)};
    }

    if (in1.NumElements() == 1) {
      src1_dims.push_back(1);
    } else {
      src1_dims = {in1.dim_size(0), in1.dim_size(3), in1.dim_size(1),
                   in1.dim_size(2)};
    }

    dst_dims = {out_shape.dim_size(0), out_shape.dim_size(3),
                out_shape.dim_size(1), out_shape.dim_size(2)};

    zendnnInfo(ZENDNN_FWKLOG,
               "ZenVitisAIBinaryOP:", " src0 = ", in0.shape().DebugString(),
               ", src1 = ", in1.shape().DebugString(),
               ", dst = ", out_shape.DebugString());

    memory::desc src0_memory_desc = memory::desc(
        {src0_dims}, DataTypetoZen<T>(), GetDefaultTagForDim(ndims));
    memory::desc src1_memory_desc = memory::desc(
        {src1_dims}, DataTypetoZen<T>(), GetDefaultTagForDim(ndims));
    memory::desc dst_memory_desc = memory::desc({dst_dims}, DataTypetoZen<T>(),
                                                GetDefaultTagForDim(ndims));

    // Set input scales for both the inputs
    primitive_attr binary_attr;

    std::vector<float> src0_scale = {std::pow(2, -in_scale_0_ + out_scale_)};
    std::vector<float> src1_scale = {std::pow(2, -in_scale_1_ + out_scale_)};

    binary_attr.set_scales(ZENDNN_ARG_SRC_0, 0, src0_scale);
    binary_attr.set_scales(ZENDNN_ARG_SRC_1, 0, src1_scale);

    binary_attr.set_scratchpad_mode(zendnn::scratchpad_mode::user);

    // Create primitive descriptor.
    binary::desc binary_desc = binary::desc(binary_algo, src0_memory_desc,
                                            src1_memory_desc, dst_memory_desc);
    binary::primitive_desc binary_prim_desc =
        binary::primitive_desc(binary_desc, binary_attr, engine);

    memory binary_src0_memory =
        zendnn::memory(src0_memory_desc, engine, in0_array);
    memory binary_src1_memory =
        zendnn::memory(src1_memory_desc, engine, in1_array);
    memory binary_dst_memory =
        zendnn::memory(dst_memory_desc, engine, output_array);

    zendnn::memory::desc scratchpad_md = binary_prim_desc.scratchpad_desc();
    if (!UserScratchPad.isallocated()) {
      UserScratchPad.Allocate(context, scratchpad_md.get_size());
    }
    zendnn::memory scratchpad(scratchpad_md, engine,
                              UserScratchPad.GetTensorHandle());

    // Create and add binary primitives
    auto binary_prim = binary(binary_prim_desc);
    zp.AddPrimitive(binary_prim, {{ZENDNN_ARG_SRC_0, binary_src0_memory},
                                  {ZENDNN_ARG_SRC_1, binary_src1_memory},
                                  {ZENDNN_ARG_DST, binary_dst_memory},
                                  {ZENDNN_ARG_SCRATCHPAD, scratchpad}});

    // Execute all the added primitives
    zp.Execute(stream);

    // Reset any primitives added yet
    zp.reset();

    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      // If in0 is not reused, free
      if (in0_reuse) {
        zenPoolBuffer->zenMemPoolUpdateTensorPtrStatus(
            context, (void *)in0_array, zendnn_params_.out_links,
            zendnn_params_.reset);
      } else {
        zenPoolBuffer->zenMemPoolFree(context, (void *)in0_array);
      }
      // If in1 is not reused, free
      if (in1_reuse) {
        zenPoolBuffer->zenMemPoolUpdateTensorPtrStatus(
            context, (void *)in1_array, zendnn_params_.out_links,
            zendnn_params_.reset);
      } else {
        zenPoolBuffer->zenMemPoolFree(context, (void *)in1_array);
      }
    }
  }

 private:
  // Parameters
  int in_scale_0_, in_scale_1_, out_scale_;
  ZendnnParameters zendnn_params_;

  // Primitive library
  ZenPrimitives zp;
  ZenPersistentTensor<unsigned char> UserScratchPad;
};

// clang-format off

// Registering dummy kernels as NoOps (needs to be overwritten)
REGISTER_KERNEL_BUILDER(Name("VitisAIAddV2").Device(DEVICE_CPU), NoOp);

// All the required ZenVitisAIAddV2 kernel combinations
#define REGISTER_VITISAI_BINARY(T)                        \
  REGISTER_KERNEL_BUILDER(Name("_ZenVitisAIAddV2")         \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<T>("T"),    \
                              ZenVitisAIBinaryOP<T, zendnn::algorithm::binary_add>)

REGISTER_VITISAI_BINARY(qint8);
REGISTER_VITISAI_BINARY(quint8);

#undef REGISTER_VITISAI_BINARY

// clang-format on

}  // namespace tensorflow