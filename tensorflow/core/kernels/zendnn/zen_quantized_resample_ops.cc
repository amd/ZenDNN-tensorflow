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

template <typename T>
class ZenVitisAIResampleOp : public OpKernel {
 public:
  explicit ZenVitisAIResampleOp(OpKernelConstruction *context)
      : OpKernel(context) {
    ReadParameterFromContext<bool>(context, "align_corners", &align_corners_);
    ReadParameterFromContext<bool>(context, "half_pixel_centers",
                                   &half_pixel_centers_);
    ReadParameterFromContext<string>(context, "resize_algorithm", &algorithm_);
    OP_REQUIRES(context, algorithm_ == "ResizeNearestNeighbor",
                errors::Unimplemented("ZenVitisAIResampleOp currently only "
                                      "supports ResizeNearestNeighbor algo"));

    OP_REQUIRES_OK(context, InitZendnnParameters(context, &zendnn_params_));
  }

  void Compute(OpKernelContext *context) override {
    ZenExecutor *ex = ex->getInstance();
    zendnn::engine engine = ex->getEngine();
    zendnn::stream stream = ex->getStream();

    // Grab and validate the input:
    const Tensor &input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    T *input_array = static_cast<T *>(const_cast<T *>(input.flat<T>().data()));

    // Grab and validate the output shape factor:
    const Tensor &shape_factor = context->input(1);
    OP_REQUIRES(context, shape_factor.dims() == 1,
                errors::InvalidArgument("shape_factor must be 1-dimensional",
                                        shape_factor.shape().DebugString()));
    OP_REQUIRES(context, shape_factor.NumElements() == 2,
                errors::InvalidArgument("shape_factor must have two elements",
                                        shape_factor.shape().DebugString()));

    auto sizes = shape_factor.vec<int32>();
    OP_REQUIRES(
        context, sizes(0) > 0 && sizes(1) > 0,
        errors::InvalidArgument("shape_factor's elements must be positive"));

    const int64_t batch_size = input.dim_size(0);
    const int64_t in_height = input.dim_size(1);
    const int64_t in_width = input.dim_size(2);
    const int64_t channels = input.dim_size(3);

    // Assuming that the operation is mul, multiply the factor with the current
    // input shapes
    const int64_t out_height = in_height * sizes(0);
    const int64_t out_width = in_width * sizes(1);

    // Set shapes
    TensorShape out_shape;
    OP_REQUIRES_OK(context, TensorShape::BuildTensorShape(
                                {batch_size, out_height, out_width, channels},
                                &out_shape));

    zendnnEnv zenEnvObj = readEnv();
    int zenPoolEnable = zenEnvObj.zenEnableMemPool &&
                        (context->expected_output_dtype(0) == DT_QINT8 ||
                         context->expected_output_dtype(0) == DT_QUINT8);

    auto out_type = std::is_same<T, quint8>::value ? zenTensorType::QUINT8
                                                   : zenTensorType::QINT8;
    ZenMemoryPool<T> *zenPoolBuffer = nullptr;
    Tensor *output = nullptr;

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
      // Outtype is not required for default allocation because context
      // maintains allocation data Type for outputs
      OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    }

    auto output_map = output->tensor<T, 4>();
    T *output_array = const_cast<T *>(output_map.data());

    memory::dims src_dims = {batch_size, channels, in_height, in_width};
    memory::dims dst_dims = {batch_size, channels, out_height, out_width};

    zendnnInfo(ZENDNN_FWKLOG, "ZenVitisAIResampleOp:", " depth = ", channels,
               ", input_cols = ", in_width, ", input_rows = ", in_height,
               ", out_cols = ", out_width, ", out_rows = ", out_height);

    memory::desc src_memory_desc =
        memory::desc({src_dims}, DataTypetoZen<T>(), tag::nhwc);
    memory::desc dst_memory_desc =
        memory::desc({dst_dims}, DataTypetoZen<T>(), tag::nhwc);

    // Check for which algo to use (currently only support resampling_nearest)
    algorithm resample_algo = algorithm::resampling_nearest;

    primitive_attr resample_attr;
    resampling_forward::desc resample_desc =
        resampling_forward::desc(prop_kind::forward_inference, resample_algo,
                                 src_memory_desc, dst_memory_desc);
    resampling_forward::primitive_desc resample_prim_desc =
        resampling_forward::primitive_desc(resample_desc, resample_attr,
                                           engine);

    memory resample_src_memory =
        zendnn::memory(src_memory_desc, engine, input_array);
    memory resample_dst_memory =
        zendnn::memory(dst_memory_desc, engine, output_array);

    // Create and add resample primitives
    auto resample_prim = resampling_forward(resample_prim_desc);
    zp.AddPrimitive(resample_prim, {{ZENDNN_ARG_SRC, resample_src_memory},
                                    {ZENDNN_ARG_DST, resample_dst_memory}});

    // // Execute all the added primitives
    zp.Execute(stream);

    // Reset any primitives added yet
    zp.reset();

    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      zenPoolBuffer->zenMemPoolFree(context, (void *)input_array);
    }
  }

 private:
  // Parameters
  bool align_corners_;
  bool half_pixel_centers_;
  string algorithm_;
  ZendnnParameters zendnn_params_;

  // Primitive library
  ZenPrimitives zp;
};

// clang-format off

// Registering dummy kernels as NoOps (needs to be overwritten)
REGISTER_KERNEL_BUILDER(Name("VitisAIResize").Device(DEVICE_CPU), NoOp);

// All the required ZenVitisAIResize kernel combinations
#define REGISTER_VITISAI_RESIZE(T)                        \
  REGISTER_KERNEL_BUILDER(Name("_ZenVitisAIResize")        \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<T>("T") ,   \
                              ZenVitisAIResampleOp<T>)

REGISTER_VITISAI_RESIZE(qint8);
REGISTER_VITISAI_RESIZE(quint8);

#undef REGISTER_VITISAI_RESIZE

// clang-format on

}  // namespace tensorflow