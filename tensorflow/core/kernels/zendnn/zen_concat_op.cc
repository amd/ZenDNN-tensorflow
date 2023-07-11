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

#include <cassert>
#include <chrono>
#include <limits>
#include <vector>

#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "zendnn.hpp"
#include "zendnn_helper.hpp"
#include "zendnn_logging.hpp"

using namespace zendnn;

using tag = memory::format_tag;
using dt = memory::data_type;

namespace tensorflow {
/// @input None
/// @return dt corresponding to type T
template <typename T>
dt ZenDnnType();

/// Instantiation for float type. Add similar instantiations for other
/// type if needed.
template <>
dt ZenDnnType<float>() {
  return dt::f32;
}

template <>
dt ZenDnnType<quint8>() {
  return dt::u8;
}

template <>
dt ZenDnnType<qint8>() {
  return dt::s8;
}

template <>
dt ZenDnnType<qint32>() {
  return dt::s32;
}

template <>
dt ZenDnnType<uint8>() {
  return dt::u8;
}

template <>
dt ZenDnnType<bfloat16>() {
  // Currently, falling back to f32 to get compilation working.
  return dt::f32;
}

enum AxisArgumentName { NAME_IS_AXIS, NAME_IS_CONCAT_DIM };

class NoOp : public OpKernel {
 public:
  explicit NoOp(OpKernelConstruction *context) : OpKernel(context) {}
  void Compute(OpKernelContext *context) override {}
  bool IsExpensive() override { return false; }
};

// --------------------------------------------------------------------------
template <typename T, AxisArgumentName AxisArgName>
class ZenConcatBaseOp : public OpKernel {
  bool reorder_before, reorder_after, reset;
  int in_links, out_links;

  memory::dims getNCHWShapeFromNHWC(const Tensor &t) {
    memory::dims result = {t.shape().dim_size(0), t.shape().dim_size(3),
                           t.shape().dim_size(1), t.shape().dim_size(2)};
    return result;
  }

 public:
  typedef std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>
      ConstMatrixVector;
  explicit ZenConcatBaseOp(OpKernelConstruction *c)
      : OpKernel(c),
        axis_attribute_name_(AxisArgName == NAME_IS_AXIS
                                 ? "axis"
                                 : AxisArgName == NAME_IS_CONCAT_DIM
                                       ? "concat_dim"
                                       : "<invalid>") {
    int unused;
    OP_REQUIRES_OK(
        c, InputRange(axis_attribute_name_, &axis_input_index_, &unused));
    OP_REQUIRES_OK(c, c->GetAttr("reorder_before", &reorder_before));
    OP_REQUIRES_OK(c, c->GetAttr("reorder_after", &reorder_after));
    OP_REQUIRES_OK(c, c->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(c, c->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(c, c->GetAttr("reset", &reset));
  }

  void Compute(OpKernelContext *c) override {
    zendnnInfo(ZENDNN_FWKLOG, "ZEN-OP-DEF: ZenConcat (TF kernel): In Compute!");
    const Tensor &concat_dim_tensor = c->input(axis_input_index_);
    OP_REQUIRES(c,
                (TensorShapeUtils::IsScalar(concat_dim_tensor.shape()) ||
                 (TensorShapeUtils::IsVector(concat_dim_tensor.shape()) &&
                  concat_dim_tensor.shape().dim_size(0) == 1)),
                errors::InvalidArgument(
                    axis_attribute_name_,
                    " tensor should be a scalar integer, but got shape ",
                    concat_dim_tensor.shape().DebugString()));

    int64 concat_dim;
    // In case of ConcatV2, "axis" could be int32 or int64
    if (AxisArgName == NAME_IS_AXIS) {
      OP_REQUIRES(
          c,
          (concat_dim_tensor.dtype() == DT_INT32 ||
           concat_dim_tensor.dtype() == DT_INT64),
          errors::InvalidArgument(axis_attribute_name_,
                                  " tensor should be int32 or int64, but got ",
                                  DataTypeString(concat_dim_tensor.dtype())));
    } else {
      OP_REQUIRES(c, (concat_dim_tensor.dtype() == DT_INT32),
                  errors::InvalidArgument(
                      axis_attribute_name_, " tensor should be int32, but got ",
                      DataTypeString(concat_dim_tensor.dtype())));
    }
    if (concat_dim_tensor.dtype() == DT_INT32) {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor.scalar<int32>()());
    } else {
      concat_dim =
          internal::SubtleMustCopy(concat_dim_tensor.scalar<int64>()());
    }

    OpInputList values;
    OP_REQUIRES_OK(c, c->input_list("values", &values));
    const int N = values.size();
    const int input_dims = values[0].dims();
    const TensorShape &input_shape = values[0].shape();

    int32 axis = concat_dim < 0 ? concat_dim + input_dims : concat_dim;
    OP_REQUIRES(c, (0 <= axis && axis < input_dims) || concat_dim == 0,
                errors::InvalidArgument(
                    "ConcatOp : Expected concatenating dimensions in the range "
                    "[",
                    -input_dims, ", ", input_dims, "), but got ", concat_dim));
    // Note that we reduce the concat of n-dimensional tensors into a two
    // dimensional concat. Assuming the dimensions of any input/output
    // tensor are {x0, x1,...,xn-1, y0, y1,...,ym-1}, where the concat is along
    // the dimension indicated with size y0, we flatten it to {x, y}, where y =
    // Prod_i(yi) and x = ((n > 0) ? Prod_i(xi) : 1).
    ConstMatrixVector inputs_flat;
    inputs_flat.reserve(N);
    int64 inputs_flat_dim0 = 1;
    for (int d = 0; d < axis; ++d) {
      inputs_flat_dim0 *= input_shape.dim_size(d);
    }

    int count = 0;
    int num_of_empty_inputs = 0;
    bool invoke_eigen = false;
    bool are_all_zen_inputs = true, are_all_tf_inputs = true;
    int64 output_concat_dim = 0;
    const bool input_is_scalar = TensorShapeUtils::IsScalar(input_shape);
    for (int i = 0; i < N; ++i) {
      const auto &in = values[i];
      const bool in_is_scalar = TensorShapeUtils::IsScalar(in.shape());
      size_t s_dims = in.dims();

      OP_REQUIRES(
          c, in.dims() == input_dims || (input_is_scalar && in_is_scalar),
          errors::InvalidArgument(
              "ConcatOp : Ranks of all input tensors should match: shape[0] = ",
              input_shape.DebugString(), " vs. shape[", i,
              "] = ", in.shape().DebugString()));
      for (int j = 0; j < input_dims; ++j) {
        if (j == axis) {
          continue;
        }
        OP_REQUIRES(
            c, in.dim_size(j) == input_shape.dim_size(j),
            errors::InvalidArgument(
                "ConcatOp : Dimensions of inputs should match: shape[0] = ",
                input_shape.DebugString(), " vs. shape[", i,
                "] = ", in.shape().DebugString()));
      }
      if (in.NumElements() > 0) {
        int64 inputs_flat_dim1 = in.NumElements() / inputs_flat_dim0;
        inputs_flat.emplace_back(new typename TTypes<T, 2>::ConstMatrix(
            in.shaped<T, 2>({inputs_flat_dim0, inputs_flat_dim1})));
      }
      output_concat_dim += in.dims() > 0 ? in.dim_size(axis) : 1;

      if (in_is_scalar) {
        are_all_tf_inputs = false;
      } else {
        are_all_zen_inputs = false;
      }

      if (s_dims != 4 && s_dims != 2) {
        invoke_eigen = true;
      }

      if (in.NumElements() == 0) {
        num_of_empty_inputs++;
      }
      ++count;
    }

    if (num_of_empty_inputs == count) {
      invoke_eigen = true;
    }

    // All inputs are not in one format (TF or ZEN). This is mixed input case.
    // We can potentially optimize this case by converting all TF inputs
    // to Zen format. But currently, we fall to Eigen for this case.
    // It may be possible to convert inputs that in TF format to Zen
    // format and avoid calling eigen version.
    if (!are_all_tf_inputs && !are_all_zen_inputs) {
      invoke_eigen = true;
    }

    OpInputList input_mins, input_maxes;
    bool quantized_input =
        std::is_same<T, qint8>::value || std::is_same<T, quint8>::value;
    if (quantized_input) {
      // Check if the ranges of the all input tensors are the same.
      // If not, forward it to Eigen implementation.

      OP_REQUIRES_OK(c, c->input_list("input_mins", &input_mins));
      OP_REQUIRES(c, (input_mins.size() == N),
                  errors::InvalidArgument(
                      "QuantizedConcatOp : Expected mins input list length ",
                      input_mins.size(), " to equal values length ", N));

      OP_REQUIRES_OK(c, c->input_list("input_maxes", &input_maxes));
      OP_REQUIRES(c, (input_maxes.size() == N),
                  errors::InvalidArgument(
                      "QuantizedConcatOp : Expected maxes input list length ",
                      input_maxes.size(), " to equal values length ", N));
      float input_min = input_mins[0].flat<float>()(0);
      float input_max = input_maxes[0].flat<float>()(0);
      const float eps = 1.0e-6;
      for (int i = 1; i < N; ++i) {
        float min = input_mins[i].flat<float>()(0);
        float max = input_maxes[i].flat<float>()(0);

        if (fabs(input_min - min) > eps || fabs(input_max - max) > eps) {
          zendnnError(ZENDNN_ALGOLOG,
                      "Zen QuantizedConcat requires inputs of same range");
          /* Invoke Eigen call when implementation is present
            * invoke_eigen = true;
            */
          return;
        }
      }
    }

    TensorShape output_shape(input_shape);
    if (output_shape.dims() == 0) {
      output_shape.AddDim(output_concat_dim);
    } else {
      output_shape.set_dim(axis, output_concat_dim);
    }
    // Update the output type
    zenTensorType out_type = zenTensorType::FLOAT;
    if (std::is_same<T, quint8>::value) {
      out_type = zenTensorType::QUINT8;
    } else if (std::is_same<T, qint8>::value) {
      out_type = zenTensorType::QINT8;
    }

    zendnnEnv zenEnvObj = readEnv();
    int use_blocked_format = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT1;
    int use_blocked_nhwc = zenEnvObj.zenConvAlgo == zenConvAlgoType::DIRECT2;

    Tensor *output = nullptr, *output_min = nullptr, *output_max = nullptr;
    int zenEnableMemPool = zenEnvObj.zenEnableMemPool &&
                           (c->expected_output_dtype(0) == DT_QINT8 ||
                            c->expected_output_dtype(0) == DT_QUINT8 ||
                            c->expected_output_dtype(0) == DT_FLOAT);
    ZenMemoryPool<T> *zenPoolBuffer = NULL;

    if (quantized_input) {
      TensorShape zen_out_shape_max, zen_out_shape_min;
      OP_REQUIRES_OK(c,
                      c->allocate_output(1, zen_out_shape_min, &output_min));
      OP_REQUIRES_OK(c,
                      c->allocate_output(2, zen_out_shape_max, &output_max));

      output_min->flat<float>()(0) = input_mins[0].flat<float>()(0);
      output_max->flat<float>()(0) = input_maxes[0].flat<float>()(0);
      if (zenEnableMemPool) {
        unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
        zenPoolBuffer = ZenMemoryPool<T>::getZenMemPool(threadID);
        if (zenPoolBuffer) {
          int status = 0;
          // Quantized models have 3 outputs 1 input is used
          // for computatuion other 2 outputs are used during dequantize
          status = zenPoolBuffer->acquireZenPoolTensor(
              c, &output, output_shape, out_links - 2, reset, out_type);
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
        OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));
      }
    } else {
      // ZenMemPool Optimization reuse o/p tensors from the pool. By default
      //  its enabled, export ZENDNN_ENABLE_MEMPOOL=0 will disable memory
      //  pool optimization
      //  Cases where tensors in pool are not free or requested size is more
      //  than available tensor size in Pool, control will fall back to
      //  default way of allocation i.e. with allocate_output(..)
      if (zenEnableMemPool && !invoke_eigen) {
        unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
        zenPoolBuffer = ZenMemoryPool<T>::getZenMemPool(threadID);
        if (zenPoolBuffer) {
          int status = zenPoolBuffer->acquireZenPoolTensor(
              c, &output, output_shape, out_links, reset, out_type);
          if (status) {
            zenEnableMemPool = false;
          }
        } else {
          zenEnableMemPool = false;
        }
      }
      if (!zenEnableMemPool || invoke_eigen) {
        // Output tensor is of the following dimensions:
        // [ in_batch, out_rows, out_cols, out_depth ]
        OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output));
      }
    }

    if (output->NumElements() <= 0) {
      return;
    }

    int64 output_dim1 = output->NumElements() / inputs_flat_dim0;
    auto output_flat = output->shaped<T, 2>({inputs_flat_dim0, output_dim1});

    // TODO: Enable below TF path for all sizes and evaluate performance
    if (invoke_eigen) {
      int64 output_dim1 = output->NumElements() / inputs_flat_dim0;
      auto output_flat = output->shaped<T, 2>({inputs_flat_dim0, output_dim1});
      ConcatCPU<T>(c->device(), inputs_flat, &output_flat);
      return;
    }

    /*
    // heuristic: axis will be passed in accordance to NHWC format TODO:verify
    // thus, if reorder_before==false, incoming data format is NC8HWC and we
    need
    // to redirect the axis to correct dimension (1). No explicit reorder is
    // required since concat performance is largely unaffected by axis
    dimension.
    // if reorder_before == true, then incoming data format is NHWC, which
    aligns
    // with how axis would be set, and no change is necessary.
    if (!reorder_before) {
        axis = axis == 3 ? 1 : axis;
    }*/
    ZenExecutor *ex = ex->getInstance();
    engine eng = ex->getEngine();
    stream engine_stream = ex->getStream();

    // fetch the pointers for input and output from Eigen data structures
    std::vector<T *> inputs_array;
    for (auto eigen_tensor : values) {
      const T *data_ptr =
          const_cast<T *>(eigen_tensor.template flat<T>().data());
      T *mutable_data_ptr = const_cast<T *>(data_ptr);
      inputs_array.push_back(mutable_data_ptr);
    }

    T *output_array = output_flat.data();

    int num_src = inputs_array.size();
    std::vector<memory::desc> src_mds;
    std::vector<memory> src_mems;
    int zen_axis;

    // TODO: Check blocked format path with densenet for reorder issues
    if (use_blocked_format) {
      // assumes inputs are in nChw8c
      for (int i = 0; i < num_src; i++) {
        const auto &in = values[i];
        memory::dims src_dims = getNCHWShapeFromNHWC(in);

        if (reorder_before) {
          zendnnInfo(ZENDNN_FWKLOG,
                     "ZEN-OP-DEF: ZenConcat (TF kernel): Using blocked format");
          auto md_tmp = memory::desc(src_dims, ZenDnnType<T>(), tag::nhwc);
          auto mem_tmp = memory(md_tmp, eng, inputs_array[i]);

          auto md = memory::desc(src_dims, ZenDnnType<T>(), tag::aBcd8b);
          src_mds.push_back(md);
          auto mem = memory(md, eng);

          auto reorder_prim = reorder(mem_tmp, mem);
          reorder_prim.execute(engine_stream, mem_tmp, mem);
          src_mems.push_back(mem);
        } else if (!reorder_before) {
          zendnnInfo(ZENDNN_FWKLOG,
                     "ZEN-OP-DEF: ZenConcat (TF kernel): Using blocked format");
          auto md = memory::desc(src_dims, ZenDnnType<T>(), tag::aBcd8b);
          src_mds.push_back(md);

          auto mem = memory(md, eng, inputs_array[i]);
          src_mems.push_back(mem);
        }
      }
      // convert NHWC axis to nChw8c axis only for blocked_format
      zen_axis = (axis == 3) ? 1 : ((axis == 0) ? 0 : axis + 1);
    } else {
      // assumes inputs are in NHWC
      zendnnInfo(ZENDNN_FWKLOG,
                 "ZEN-OP-DEF: ZenConcat (TF kernel): Using nc format");
      for (int i = 0; i < num_src; i++) {
        const auto &in = values[i];
        int64 inputs_flat_dim1 = in.NumElements() / inputs_flat_dim0;
        memory::dims src_dims = {inputs_flat_dim0, inputs_flat_dim1};
        auto md = memory::desc(src_dims, ZenDnnType<T>(), tag::nc);
        src_mds.push_back(md);

        auto mem = memory(md, eng, inputs_array[i]);
        src_mems.push_back(mem);
      }
      zen_axis = 1;
    }

    // Create primitive descriptor.
    auto concat_pd = concat::primitive_desc(zen_axis, src_mds, eng);

    // Create destination (dst) memory object using the memory descriptor
    // created by the primitive.
    memory dst_mem, dst_mem_new;
    if (reorder_after && use_blocked_format) {
      memory::dims current_dims = {
          output_shape.dim_size(0), output_shape.dim_size(3),
          output_shape.dim_size(1), output_shape.dim_size(2)};
      dst_mem = memory(concat_pd.dst_desc(), eng);
      auto dst_md_tmp = memory::desc(current_dims, ZenDnnType<T>(), tag::nhwc);
      dst_mem_new = memory(dst_md_tmp, eng, output_array);
    } else {
      dst_mem = memory(concat_pd.dst_desc(), eng, output_array);
    }

    // Create the primitive.
    auto concat_prim = concat(concat_pd);

    // Primitive arguments.
    std::unordered_map<int, memory> concat_args;
    for (int n = 0; n < num_src; ++n)
      concat_args.insert({ZENDNN_ARG_MULTIPLE_SRC + n, src_mems[n]});
    concat_args.insert({ZENDNN_ARG_DST, dst_mem});

    auto start = std::chrono::high_resolution_clock::now();
    // Primitive execution: concatenation.
    concat_prim.execute(engine_stream, concat_args);

    // Wait for the computation to finalize.
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    auto duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    zendnnInfo(ZENDNN_FWKLOG,
               "ZenConcat (TF kernel): Time taken = ", duration_ms.count(),
               "ms");

    if (reorder_after && use_blocked_format) {
      int num_dims = output_shape.dims();
      assert(num_dims == 4 &&
             "ZenConcat (TF kernel): Reorder not defined for shapes other than "
             "4D");

      auto reorder_prim = reorder(dst_mem, dst_mem_new);
      reorder_prim.execute(engine_stream, dst_mem, dst_mem_new);
    }

    // If ZenMemPool Optimization is enabled(default), update the state of
    //  Memory pool based on input_array address
    if (zenEnvObj.zenEnableMemPool && zenPoolBuffer) {
      for (int i = 0; i < num_src; i++) {
        const Tensor &input = values[i];
        if ((input.dtype() != DT_FLOAT) && (input.dtype() != DT_QINT8) &&
            (input.dtype() != DT_QUINT8)) {
          continue;
        }
        const T *data_ptr = const_cast<T *>(input.template flat<T>().data());
        zenPoolBuffer->zenMemPoolFree(c, (void *)data_ptr);
      }
    }
  }

 private:
  const char *const axis_attribute_name_;
  int axis_input_index_;
  int values_input_start_index_;
  int values_input_end_index_;
};

template <typename T>
using ZenConcatOp = ZenConcatBaseOp<T, NAME_IS_CONCAT_DIM>;
template <typename T>
using ZenConcatV2Op = ZenConcatBaseOp<T, NAME_IS_AXIS>;

#define REGISTER_CONCAT(type)                            \
  REGISTER_KERNEL_BUILDER(Name("_ZenConcat")              \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("concat_dim"), \
                          ZenConcatOp<type>)             \
  REGISTER_KERNEL_BUILDER(Name("_ZenConcatV2")            \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("axis"),       \
                          ZenConcatV2Op<type>)

REGISTER_CONCAT(quint8);
REGISTER_CONCAT(qint8);
REGISTER_CONCAT(uint8);
REGISTER_CONCAT(qint32);
REGISTER_CONCAT(float);
REGISTER_CONCAT(bfloat16);

#undef REGISTER_CONCAT
#define REGISTER_QUANTIZED_CONCAT_V2(type)               \
  REGISTER_KERNEL_BUILDER(Name("_ZenQuantizedConcatV2")  \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("axis"),       \
                          ZenConcatV2Op<type>)

REGISTER_QUANTIZED_CONCAT_V2(quint8);
REGISTER_QUANTIZED_CONCAT_V2(qint8);

#undef REGISTER_QUANTIZED_CONCAT

}  // namespace tensorflow
