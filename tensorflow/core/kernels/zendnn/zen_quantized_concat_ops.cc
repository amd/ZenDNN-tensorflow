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

#include <numeric>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/kernels/zendnn/zen_quantized_ops.h"

namespace tensorflow {

using tag = memory::format_tag;
using dt = memory::data_type;

enum AxisArgumentName { NAME_IS_AXIS, NAME_IS_CONCAT_DIM };

/**
 * @brief ZenVitisAIConcatBaseOp implements ZenVitisAIConcatV2
 *
 * @tparam T Concat dtype
 * @tparam AxisArgName axis/dim to be concated
 */
template <typename T, AxisArgumentName AxisArgName>
class ZenVitisAIConcatBaseOp : public OpKernel {
 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;

  explicit ZenVitisAIConcatBaseOp(OpKernelConstruction *context)
      : OpKernel(context),
        axis_attribute_name_(AxisArgName == NAME_IS_AXIS
                                 ? "axis"
                                 : AxisArgName == NAME_IS_CONCAT_DIM
                                       ? "concat_dim"
                                       : "<invalid>") {
    int unused;
    OP_REQUIRES_OK(
        context, InputRange(axis_attribute_name_, &axis_input_index_, &unused));
    OP_REQUIRES_OK(context, InputRange("values", &values_input_start_index_,
                                       &values_input_end_index_));

    InitZendnnParameters(context, &zendnn_params_);
  }

  void Compute(OpKernelContext *context) override {
    ZenExecutor *ex = ex->getInstance();
    zendnn::engine engine = ex->getEngine();
    zendnn::stream stream = ex->getStream();

    const Tensor &concat_dim_tensor = context->input(axis_input_index_);

    // Checks from kernels/concat_op.cc to verify concat op
    int64_t concat_dim;
    // In case of ConcatV2, "axis" could be int32 or int64
    if (AxisArgName == NAME_IS_AXIS) {
      OP_REQUIRES(
          context,
          (concat_dim_tensor.dtype() == DT_INT32 ||
           concat_dim_tensor.dtype() == DT_INT64),
          errors::InvalidArgument(axis_attribute_name_,
                                  " tensor should be int32 or int64, but got ",
                                  DataTypeString(concat_dim_tensor.dtype())));
    } else {
      OP_REQUIRES(context, (concat_dim_tensor.dtype() == DT_INT32),
                  errors::InvalidArgument(
                      axis_attribute_name_, " tensor should be int32, but got ",
                      DataTypeString(concat_dim_tensor.dtype())));
    }
    if (concat_dim_tensor.dtype() == DT_INT32) {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor.scalar<int32>()());
    } else {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor.scalar<int64_t>()());
    }

    OpInputList values;
    OP_REQUIRES_OK(context, context->input_list("values", &values));
    const int N = values_input_end_index_ - values_input_start_index_;
    const Tensor &first_input = values[0];
    const int input_dims = first_input.dims();
    const TensorShape &input_shape = first_input.shape();

    int32_t axis = concat_dim < 0 ? concat_dim + input_dims : concat_dim;
    OP_REQUIRES(context, (0 <= axis && axis < input_dims) || concat_dim == 0,
                errors::InvalidArgument(
                    "ConcatOp : Expected concatenating dimensions in the range "
                    "[",
                    -input_dims, ", ", input_dims, "), but got ", concat_dim));

    // Note that we reduce the concat of n-dimensional tensors into a two
    // dimensional concat. Assuming the dimensions of any input/output
    // tensor are {x0, x1,...,xn-1, y0, y1,...,ym-1}, where the concat is along
    // the dimension indicated with size y0, we flatten it to {x, y}, where y =
    // Prod_i(yi) and x = ((n > 0) ? Prod_i(xi) : 1).
    int64_t inputs_flat_dim0 = 1;
    for (int d = 0; d < axis; ++d) {
      inputs_flat_dim0 *= input_shape.dim_size(d);
    }
    int64_t output_concat_dim = 0;
    for (int i = 0; i < N; ++i) {
      const auto &input = values[values_input_start_index_ + i];
      OP_REQUIRES(
          context, input.dims() == input_dims,
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              input_shape.DebugString(), " vs. shape[", i,
              "] = ", input.shape().DebugString()));
      for (int j = 0; j < input_dims; ++j) {
        if (j == axis) {
          continue;
        }
        OP_REQUIRES(
            context, input.dim_size(j) == input_shape.dim_size(j),
            errors::InvalidArgument("ConcatOp : Dimension ", j,
                                    " in both shapes must be equal: "
                                    "shape[0] = ",
                                    input_shape.DebugString(), " vs. shape[", i,
                                    "] = ", input.shape().DebugString()));
      }
      output_concat_dim += input.dims() > 0 ? input.dim_size(axis) : 1;
    }

    // Set the output shape and allocate output
    TensorShape output_shape(input_shape);
    output_shape.set_dim(axis, output_concat_dim);
    Tensor *output = nullptr;

    // Update the output type
    zenTensorType out_type = zenTensorType::FLOAT;
    if (std::is_same<T, quint8>::value) {
      out_type = zenTensorType::QUINT8;
    } else if (std::is_same<T, qint8>::value) {
      out_type = zenTensorType::QINT8;
    }

    zendnnEnv zenEnvObj = readEnv();
    int zenEnableMemPool = zenEnvObj.zenEnableMemPool &&
                           (context->expected_output_dtype(0) == DT_QINT8 ||
                            context->expected_output_dtype(0) == DT_QUINT8 ||
                            context->expected_output_dtype(0) == DT_FLOAT);
    ZenMemoryPool<T> *zenPoolBuffer = NULL;

    if (zenEnableMemPool) {
      unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
      zenPoolBuffer = ZenMemoryPool<T>::getZenMemPool(threadID);
      if (zenPoolBuffer) {
        int status = 0;
        status = zenPoolBuffer->acquireZenPoolTensor(
            context, &output, output_shape, zendnn_params_.out_links,
            zendnn_params_.reset, out_type);
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
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, output_shape, &output));
    }

    // The output is assumed to be a 2D tensor (even though the output is multi
    // dimensional)
    int64_t output_dim1 = output->NumElements() / inputs_flat_dim0;
    auto output_flat = output->shaped<T, 2>({inputs_flat_dim0, output_dim1});
    T *output_array = output_flat.data();

    std::vector<T *> inputs_array;
    for (auto eigen_tensor : values) {
      const T *data_ptr =
          const_cast<T *>(eigen_tensor.template flat<T>().data());
      T *mutable_data_ptr = const_cast<T *>(data_ptr);
      inputs_array.push_back(mutable_data_ptr);
    }

    // Each dimension for the memory desc will be
    // prod(before axis dims), prod(after axis dims) (line: 105)
    std::vector<memory::desc> src_mds;
    std::vector<memory> src_mems;
    for (int i = 0; i < N; i++) {
      const auto &input = values[i];
      int64_t inputs_flat_dim1 = input.NumElements() / inputs_flat_dim0;
      memory::dims src_dims = {inputs_flat_dim0, inputs_flat_dim1};
      auto md = memory::desc(src_dims, DataTypetoZen<T>(), tag::nc);
      src_mds.push_back(md);

      auto mem = memory(md, engine, inputs_array[i]);
      src_mems.push_back(mem);
    }

    // Since zendnn dims are always in NCHW, axis will be 1
    int zen_axis = 1;
    auto concat_pd = concat::primitive_desc(zen_axis, src_mds, engine);

    // Create destination (dst) memory object using the memory descriptor
    // created by the primitive.
    memory dst_mem = memory(concat_pd.dst_desc(), engine, output_array);

    auto concat_prim = concat(concat_pd);
    // Primitive arguments.
    std::unordered_map<int, memory> concat_args;
    for (int n = 0; n < N; ++n)
      concat_args.insert({ZENDNN_ARG_MULTIPLE_SRC + n, src_mems[n]});
    concat_args.insert({ZENDNN_ARG_DST, dst_mem});

    zp.AddPrimitive(concat_prim, concat_args);

    // Execute all the added primitives
    zp.Execute(stream);

    // Reset any primitives added yet
    zp.reset();

    // If ZenMemPool Optimization is enabled(default), update the state of
    //  Memory pool based on input_array address
    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      for (int i = 0; i < N; i++) {
        const T *data_ptr = const_cast<T *>(inputs_array[i]);
        zenPoolBuffer->zenMemPoolFree(context, (void *)data_ptr);
      }
    }
  }

 private:
  const char *const axis_attribute_name_;
  int axis_input_index_;
  int values_input_start_index_;
  int values_input_end_index_;

  ZenPrimitives zp;
  ZendnnParameters zendnn_params_;
};

// Not used fow now
// template <typename T>
// using ZenVitisAIConcatOp = ZenVitisAIConcatBaseOp<T, NAME_IS_CONCAT_DIM>;
template <typename T>
using ZenVitisAIConcatV2Op = ZenVitisAIConcatBaseOp<T, NAME_IS_AXIS>;

// clang-format off
REGISTER_KERNEL_BUILDER(Name("VitisAIConcatV2").Device(DEVICE_CPU), NoOp);

// All the required ZenVitisAIConcatV2 kernel combinations
#define REGISTER_VITISAI_QUANTIZEV2(T) \
  REGISTER_KERNEL_BUILDER(             \
      Name("ZenVitisAIConcatV2")      \
          .Device(DEVICE_CPU)          \
          .TypeConstraint<T>("T"),     \
      ZenVitisAIConcatV2Op<T>)

REGISTER_VITISAI_QUANTIZEV2(quint8);

#undef REGISTER_VITISAI_QUANTIZEV2

// clang-format on

}  // namespace tensorflow
