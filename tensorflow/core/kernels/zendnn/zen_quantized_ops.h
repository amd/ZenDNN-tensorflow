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

#ifndef TENSORFLOW_CORE_KERNELS_ZEN_QUANTIZED_OPS_H_
#define TENSORFLOW_CORE_KERNELS_ZEN_QUANTIZED_OPS_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fused_eigen_output_kernels.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/util/tensor_format.h"
#include "zendnn.hpp"

using namespace zendnn;

namespace tensorflow {

/**
 * @brief Parameters that is used for standard ZenDNN node.
 *
 */
struct ZendnnParameters {
  bool reorder_before;
  bool reorder_after;
  int in_links;
  int out_links;
  bool reset;
};

/**
 * @brief Read attributes from context.
 *
 * @tparam T dtype of the parameter
 * @param context Context from which prameters are read
 * @param attribute_name Name of the attribute to be read
 * @param value Store the parameter value
 */
template <typename T>
inline void ReadParameterFromContext(OpKernelConstruction* context,
                                     StringPiece attribute_name, T* value) {
  OP_REQUIRES_OK(context, context->GetAttr(attribute_name, value));
}

/**
 * @brief Read attributes from context if availalable.
 *
 * @tparam T dtype of the parameter
 * @param context Context from which prameters are read
 * @param attribute_name Name of the attribute to be read
 * @param value Store the parameter value
 */
template <typename T>
inline void ReadParameterFromContextIfAvailable(OpKernelConstruction* context,
                                                StringPiece attribute_name,
                                                T* value) {
  if (context->HasAttr(attribute_name))
    ReadParameterFromContext<T>(context, attribute_name, value);
}

/**
 * @brief Read input from Context as tensors.
 *
 * @param context Context from which prameters are read
 * @param idx Index of input
 * @return const Tensor&
 */
inline const Tensor& ReadInputFromContext(const OpKernelContext* context,
                                          int idx) {
  return context->input(idx);
}

/**
 * @brief Initializes and validates ZenDNN parameters configured
 * by OpKernel attributes.
 *
 * @param context Context from which prameters are read
 * @param params Parameters for ZenDNN Op
 * @return Status
 */
inline Status InitZendnnParameters(OpKernelConstruction* context,
                                   ZendnnParameters* params) {
  ReadParameterFromContext<bool>(context, "reorder_before",
                                 &params->reorder_before);
  ReadParameterFromContext<bool>(context, "reorder_after",
                                 &params->reorder_after);
  ReadParameterFromContext<int>(context, "in_links", &params->in_links);
  ReadParameterFromContext<int>(context, "out_links", &params->out_links);
  ReadParameterFromContext<bool>(context, "reset", &params->reset);

  return OkStatus();
}

/**
 * @brief Returns the C++ data type in zendnn memory data type format.
 *
 * @tparam T C++ dtype
 * @return zendnn::memory::data_type
 */
template <typename T>
inline zendnn::memory::data_type DataTypetoZen() {
  if (std::is_same<T, quint8>::value)
    return zendnn::memory::data_type::u8;
  else if (std::is_same<T, qint8>::value)
    return zendnn::memory::data_type::s8;
  else if (std::is_same<T, qint32>::value)
    return zendnn::memory::data_type::s32;
  else if (std::is_same<T, float>::value)
    return zendnn::memory::data_type::f32;
  else
    return zendnn::memory::data_type::undef;
}

/**
 * @brief Returns the default memory tag to be used
 *
 * @param ndims number of dimensions of the tensor
 * @return zendnn::memory::format_tag
 */
inline zendnn::memory::format_tag GetDefaultTagForDim(int ndims) {
  if (ndims == 1)
    return zendnn::memory::format_tag::x;
  else if (ndims == 2)
    return zendnn::memory::format_tag::nc;
  else if (ndims == 4)
    return zendnn::memory::format_tag::nhwc;
}

/**
 * @brief Allocate and manage TF Persistent Tensor.
 *
 * @tparam T dtype of the Persistent Tensor
 */
template <typename T>
struct ZenPersistentTensor {
  // Wrapper method to check if tensor has memory allocated
  inline bool isallocated() { return allocated_; }

  // Wrapper method to check if tensor has value
  inline bool tensorvalue() { return (allocated_ && set_); }

  // Allocate required memory as temp memory
  inline void Allocate(OpKernelContext* context, size_t size) {
    TensorShape cached_tensor_shape;
    cached_tensor_shape.AddDim(size);
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                        cached_tensor_shape, &cached_tensor_));
    cached_tensor_size_ = size;
    allocated_ = true;
  }

  // Return the handle of the allocated tensor
  inline T* GetTensorHandle() {
    if (allocated_) {
      return static_cast<T*>(const_cast<T*>(cached_tensor_.flat<T>().data()));
    }
    return nullptr;
  }

  // Set value to the tensor
  inline void SetTensorHandle(T* value, size_t size) {
    // Make an error if size of the allocate memory and size of
    // the value is not the same
    if (!allocated_ || size != cached_tensor_size_) {
      // Ideally throw an error
      // errors::Internal("Persistent Tensor size and value "
      // "to be assinged have different sizes");
    }
    memcpy(static_cast<T*>(cached_tensor_.flat<T>().data()), value, size);
    set_ = true;
  }

  // Allocate required memory and set value to the tensor
  inline void SetTensorHandle(OpKernelContext* context, T* value, size_t size) {
    Allocate(context, size);
    SetTensorHandle(value, size);
  }

  inline void SetTensorHandle(OpKernelContext* context, zendnn::memory mem,
                              bool overwrite = false) {
    if (!set_ || overwrite) {
      T* value = static_cast<T*>(mem.get_data_handle());
      size_t size = mem.get_desc().get_size();
      SetTensorHandle(context, value, size);
    }
  }

 private:
  bool allocated_ = false;
  bool set_ = false;
  size_t cached_tensor_size_;
  Tensor cached_tensor_ GUARDED_BY(mu_);
};

/**
 * @brief Appends and executes all the primitives as a subgraph.
 *
 */
struct ZenPrimitives {
  // Add primitives to the subgraph
  inline void AddPrimitive(zendnn::primitive prim,
                           std::unordered_map<int, memory> prim_args) {
    net_.push_back(prim);
    net_args_.push_back(prim_args);
    num_primitives_++;
  }

  // Wrappers for reorders to be added to the subgraph
  inline void AddReorder(zendnn::memory src_memory,
                         zendnn::memory& dst_memory) {
    auto reoder_prim = reorder(src_memory, dst_memory);
    AddPrimitive(reoder_prim,
                 {{ZENDNN_ARG_SRC, src_memory}, {ZENDNN_ARG_DST, dst_memory}});
  }

  // Wrappers for reorders to be added to the subgraph
  inline void AddReorder(zendnn::memory src_memory, zendnn::memory& dst_memory,
                         const std::vector<int> scale, int scale_depth = 1) {
    primitive_attr reorder_attr;
    std::vector<float> reorder_scales(scale_depth);
    for (int idx = 0; idx < scale_depth; idx++)
      reorder_scales[idx] = std::pow(2, scale[idx]);
    reorder_attr.set_output_scales((scale_depth > 1) ? 1 : 0, reorder_scales);

    auto reoder_prim = reorder(src_memory, dst_memory, reorder_attr);
    AddPrimitive(reoder_prim,
                 {{ZENDNN_ARG_SRC, src_memory}, {ZENDNN_ARG_DST, dst_memory}});
  }

  // Execute the subgraph, after checking for validation
  inline void Execute(zendnn::stream stream) {
    assert(num_primitives_ == net_.size() && "something is missing");
    assert(num_primitives_ == net_args_.size() && "something is missing");

    for (size_t i = 0; i < num_primitives_; ++i) {
      net_.at(i).execute(stream, net_args_.at(i));
    }
  }

  // Clear everything within the subgraph
  // @TODO: Add a way to use context and cache the primitives
  // directly so that we don't need to create any later on.
  inline void reset() {
    num_primitives_ = 0;
    net_.clear();
    net_args_.clear();
  }

 private:
  size_t num_primitives_ = 0;
  std::vector<zendnn::primitive> net_;
  std::vector<std::unordered_map<int, memory>> net_args_;
};

}  // namespace tensorflow

#endif // TENSORFLOW_CORE_KERNELS_ZEN_QUANTIZED_OPS_H_
