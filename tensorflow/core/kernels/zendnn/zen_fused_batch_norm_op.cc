/*******************************************************************************
 * Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 *******************************************************************************/

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/fused_batch_norm_op.h"
#include "tensorflow/core/kernels/no_op.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/zen_util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "zendnn.hpp"
#include "zendnn_helper.hpp"

#define GET_FLAG(bn_flag) static_cast<int>(zendnn::normalization_flags::bn_flag)
#define IS_SET(cflag) (context_.flags & GET_FLAG(cflag))

using zendnn::batch_normalization_forward;
using zendnn::prop_kind;
using zendnn::stream;
using namespace zendnn;

using BatchNormFwdPd = batch_normalization_forward::primitive_desc;

namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;

using FusedBNActivationMode = functor::FusedBatchNormActivationMode;

struct ZenBatchNormFwdParams {
  memory::dims src_dims;
  int depth;
  float eps;
  bool training;
  FusedBNActivationMode activation_mode;
  memory::desc src_md;

  ZenBatchNormFwdParams(const memory::dims &src_dims, int depth, float eps,
                        bool training, memory::desc src_md,
                        FusedBNActivationMode activation_mode)
      : src_dims(src_dims),
        depth(depth),
        eps(eps),
        training(training),
        activation_mode(activation_mode),
        src_md(src_md) {}
};

template <typename T, typename U>
class ZenFusedBatchNormFwdPrimitive : public ZenPrimitive {
 public:
  explicit ZenFusedBatchNormFwdPrimitive(const ZenBatchNormFwdParams &fwdParams)
      : ZenPrimitive() {
    ZenExecutor *ex = ex->getInstance();
    std::shared_ptr<stream> s = ex->getStreamPtr();
    context_.bn_stream = s;
    if (context_.bn_fwd == nullptr) {
      Setup(fwdParams);
    }
  }

  ~ZenFusedBatchNormFwdPrimitive() {}

  // BatchNormalization forward execute
  //   src_data:     input data buffer of src
  //   weights_data: input data buffer of weights
  //   dst_data:     output data buffer of dst
  //   mean_data:     output data buffer of means
  //   variance_data: output data buffer of variances
  void Execute(const T *src_data, const U *weights_data, T *dst_data,
               U *mean_data, U *variance_data, U *workspace_data) {
    context_.src_mem->set_data_handle(
        static_cast<void *>(const_cast<T *>(src_data)));
    context_.dst_mem->set_data_handle(static_cast<void *>(dst_data));

    if (IS_SET(use_scale_shift))
      context_.weights_mem->set_data_handle(
          static_cast<void *>(const_cast<U *>(weights_data)));

    if ((context_.pkind == prop_kind::forward_training) ||
        (IS_SET(use_global_stats))) {
      context_.mean_mem->set_data_handle(static_cast<void *>(mean_data));
      context_.variance_mem->set_data_handle(
          static_cast<void *>(variance_data));
    }
    if (workspace_data != nullptr) {
      context_.ws_mem->set_data_handle(workspace_data);
    }

    // Execute batch-normalization forward primitives.
    execute_primitives(context_.fwd_primitives, context_.bn_stream,
                       context_.net_args);

    context_.src_mem->set_data_handle(DummyData);
    context_.dst_mem->set_data_handle(DummyData);

    if (IS_SET(use_scale_shift)) {
      context_.weights_mem->set_data_handle(DummyData);
    }

    if ((context_.pkind == prop_kind::forward_training) ||
        (IS_SET(use_global_stats))) {
      context_.mean_mem->set_data_handle(DummyData);
      context_.variance_mem->set_data_handle(DummyData);
    }

    if (workspace_data != nullptr) {
      context_.ws_mem->set_data_handle(DummyData);
    }
  }

  memory::desc GetDstPd() const { return context_.dst_mem->get_desc(); }

  std::shared_ptr<BatchNormFwdPd> GetBatchNormFwdPd() const {
    return context_.fwd_pd;
  }

 private:
  // Primitive reuse context for BatchNorm forward op.
  struct BatchNormFwdContext {
    // Flags indicating if it is training or inference mode.
    int64 flags;

    // Algorithm kind.
    prop_kind pkind;

    // Inputs/outputs memory.
    std::shared_ptr<memory> src_mem;
    std::shared_ptr<memory> weights_mem;
    std::shared_ptr<memory> dst_mem;
    std::shared_ptr<memory> mean_mem;
    std::shared_ptr<memory> variance_mem;
    std::shared_ptr<memory> ws_mem;

    // Forward BatchNorm primitive descriptor.
    std::shared_ptr<BatchNormFwdPd> fwd_pd;

    // BatchNorm forward primitive.
    std::shared_ptr<primitive> bn_fwd;
    std::vector<primitive> fwd_primitives;

    std::vector<std::unordered_map<int, memory>> net_args;

    std::shared_ptr<stream> bn_stream;

    BatchNormFwdContext()
        : flags(0),
          pkind(prop_kind::forward_training),
          src_mem(nullptr),
          weights_mem(nullptr),
          dst_mem(nullptr),
          mean_mem(nullptr),
          variance_mem(nullptr),
          ws_mem(nullptr),
          bn_fwd(nullptr),
          bn_stream(nullptr) {}
  };

  void Setup(const ZenBatchNormFwdParams &fwdParams) {
    context_.flags =
        fwdParams.training
            ? GET_FLAG(use_scale_shift)
            : (GET_FLAG(use_scale_shift) | GET_FLAG(use_global_stats));
    context_.pkind = fwdParams.training ? prop_kind::forward_training
                                        : prop_kind::forward_scoring;

    if (fwdParams.activation_mode == FusedBNActivationMode::kRelu) {
      context_.flags |= GET_FLAG(fuse_norm_relu);
    }
    // Memory descriptor
    auto src_md = fwdParams.src_md;
    // Create forward BatchNorm descriptor and primitive descriptor.
    auto fwd_desc = batch_normalization_forward::desc(
        context_.pkind, src_md, fwdParams.eps,
        static_cast<zendnn::normalization_flags>(context_.flags));

    context_.fwd_pd.reset(new BatchNormFwdPd(fwd_desc, cpu_engine_));

    // Create memory primitive based on dummy data
    context_.src_mem.reset(
        new memory(context_.fwd_pd->src_desc(), cpu_engine_, DummyData));
    context_.dst_mem.reset(
        new memory(context_.fwd_pd->dst_desc(), cpu_engine_, DummyData));

    memory::dims s_dims = {2, fwdParams.depth};
    memory::dims m_dims = {1, fwdParams.depth};
    if (IS_SET(use_scale_shift)) {
      context_.weights_mem.reset(
          new memory({{s_dims}, memory::data_type::f32, memory::format_tag::nc},
                     cpu_engine_, DummyData));
    }

    if (fwdParams.training || (IS_SET(use_global_stats))) {
      context_.mean_mem.reset(
          new memory({{m_dims}, memory::data_type::f32, memory::format_tag::nc},
                     cpu_engine_, DummyData));

      context_.variance_mem.reset(
          new memory({{m_dims}, memory::data_type::f32, memory::format_tag::nc},
                     cpu_engine_, DummyData));
    }

    if (IS_SET(fuse_norm_relu)) {
      context_.ws_mem.reset(new memory(context_.fwd_pd->workspace_desc(),
                                       cpu_engine_, DummyData));
    }

    // BatchNorm forward primitive.
    if (!fwdParams.training && !(IS_SET(use_global_stats))) {
      if ((IS_SET(use_scale_shift)) && zendnn_use_scaleshift) {
        context_.net_args.push_back(
            {{ZENDNN_ARG_SRC, *context_.src_mem},
             {ZENDNN_ARG_WEIGHTS, *context_.weights_mem},
             {ZENDNN_ARG_DST, *context_.dst_mem}});
      } else {
        context_.net_args.push_back({{ZENDNN_ARG_SRC, *context_.src_mem},
                                     {ZENDNN_ARG_DST, *context_.dst_mem}});
      }
      context_.bn_fwd.reset(new batch_normalization_forward(*context_.fwd_pd));
    } else if (IS_SET(use_global_stats)) {
      if ((IS_SET(use_scale_shift)) && GET_FLAG(use_scale_shift)) {
        if (IS_SET(fuse_norm_relu)) {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem},
               {ZENDNN_ARG_WEIGHTS, *context_.weights_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem},
               {ZENDNN_ARG_WORKSPACE, *context_.ws_mem}});
        } else {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem},
               {ZENDNN_ARG_WEIGHTS, *context_.weights_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem}});
        }
      } else {
        if (IS_SET(fuse_norm_relu)) {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem},
               {ZENDNN_ARG_WORKSPACE, *context_.ws_mem}});
        } else {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem}});
        }
      }
      context_.bn_fwd.reset(new batch_normalization_forward(*context_.fwd_pd));
    } else {
      if ((IS_SET(use_scale_shift)) && GET_FLAG(use_scale_shift)) {
        if (IS_SET(fuse_norm_relu)) {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_WEIGHTS, *context_.weights_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem},
               {ZENDNN_ARG_WORKSPACE, *context_.ws_mem}});
        } else {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_WEIGHTS, *context_.weights_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem}});
        }
      } else {
        if (IS_SET(fuse_norm_relu)) {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem},
               {ZENDNN_ARG_WORKSPACE, *context_.ws_mem}});
        } else {
          context_.net_args.push_back(
              {{ZENDNN_ARG_SRC, *context_.src_mem},
               {ZENDNN_ARG_DST, *context_.dst_mem},
               {ZENDNN_ARG_MEAN, *context_.mean_mem},
               {ZENDNN_ARG_VARIANCE, *context_.variance_mem}});
        }
      }
      context_.bn_fwd.reset(new batch_normalization_forward(*context_.fwd_pd));
    }

    context_.fwd_primitives.push_back(*context_.bn_fwd);
  }

  struct BatchNormFwdContext context_;
};

template <typename T, typename U>
class ZenFusedBatchNormFwdPrimitiveFactory : public ZenPrimitiveFactory {
 public:
  static ZenFusedBatchNormFwdPrimitive<T, U> *Get(
      const ZenBatchNormFwdParams &fwdParams) {
    auto bn_fwd = static_cast<ZenFusedBatchNormFwdPrimitive<T, U> *>(
        ZenFusedBatchNormFwdPrimitiveFactory<T, U>::GetInstance()
            .GetBatchNormFwd(fwdParams));

    if (bn_fwd == nullptr) {
      bn_fwd = new ZenFusedBatchNormFwdPrimitive<T, U>(fwdParams);
      ZenFusedBatchNormFwdPrimitiveFactory<T, U>::GetInstance().SetBatchNormFwd(
          fwdParams, bn_fwd);
    }
    return bn_fwd;
  }

  static ZenFusedBatchNormFwdPrimitiveFactory &GetInstance() {
    static ZenFusedBatchNormFwdPrimitiveFactory instance_;
    return instance_;
  }

 private:
  ZenFusedBatchNormFwdPrimitiveFactory() {}
  ~ZenFusedBatchNormFwdPrimitiveFactory() {}

  static string CreateKey(const ZenBatchNormFwdParams &fwdParams) {
    string prefix = "bn_fwd";
    FactoryKeyCreator key_creator;
    key_creator.AddAsKey(prefix);
    key_creator.AddAsKey(fwdParams.src_dims);
    key_creator.AddAsKey<int>(fwdParams.depth);
    key_creator.AddAsKey<float>(fwdParams.eps);
    key_creator.AddAsKey<bool>(fwdParams.training);
    key_creator.AddAsKey<FusedBNActivationMode>(fwdParams.activation_mode);
    key_creator.AddAsKey(typeid(T).name());
    key_creator.AddAsKey(typeid(U).name());
    return key_creator.GetKey();
  }

  ZenPrimitive *GetBatchNormFwd(const ZenBatchNormFwdParams &fwdParams) {
    string key = CreateKey(fwdParams);
    return this->GetOp(key);
  }

  void SetBatchNormFwd(const ZenBatchNormFwdParams &fwdParams,
                       ZenPrimitive *op) {
    string key = CreateKey(fwdParams);
    this->SetOp(key, op);
  }
};

//  Adding a third parameter to the template to support FusedBatchNormV3
//  This is different from default where the classes are
//  derived. Moves enabling to compile-time rather than runtime.
template <typename Device, typename T, typename U, bool reserved_space,
          bool is_batch_norm_ex = false>
class ZenFusedBatchNormOp : public OpKernel {
 public:
  explicit ZenFusedBatchNormOp(OpKernelConstruction *context)
      : OpKernel(context) {
    float epsilon;
    OP_REQUIRES_OK(context, context->GetAttr("epsilon", &epsilon));
    epsilon_ = epsilon;
    float exponential_avg_factor;
    OP_REQUIRES_OK(context, context->GetAttr("exponential_avg_factor",
                                             &exponential_avg_factor));
    exponential_avg_factor_ = static_cast<U>(exponential_avg_factor);
    string tensor_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &tensor_format));
    OP_REQUIRES(context, FormatFromString(tensor_format, &tensor_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
    OP_REQUIRES_OK(context, context->GetAttr("in_links", &in_links));
    OP_REQUIRES_OK(context, context->GetAttr("out_links", &out_links));
    OP_REQUIRES_OK(context, context->GetAttr("reset", &reset));

    depth_ = 0;
    mean_values_ = nullptr;
    variance_values_ = nullptr;

    if (!is_batch_norm_ex) {
      activation_mode_ = FusedBNActivationMode::kIdentity;
    } else {
      int num_side_inputs;
      OP_REQUIRES_OK(context,
                     context->GetAttr("num_side_inputs", &num_side_inputs));
      OP_REQUIRES(context, num_side_inputs == 0,
                  errors::InvalidArgument(
                      "ZenFusedBatchNorm do not support side input now."));

      OP_REQUIRES_OK(context, ParseActivationMode(context, &activation_mode_));
      OP_REQUIRES(context, activation_mode_ == FusedBNActivationMode::kRelu,
                  errors::InvalidArgument(
                      "ZenFusedBatchNorm only support Relu activation"));
    }
  }

  void Compute(OpKernelContext *context) override {
    const size_t kSrcIndex = 0;       // index of src input tensor
    const size_t kScaleIndex = 1;     // index of scale tensor
    const size_t kShiftIndex = 2;     // index of shift tensor
    const size_t kMeanIndex = 3;      // index of est_mean tensor
    const size_t kVarianceIndex = 4;  // index of est_variance tensor

    const Tensor &src_tensor = context->input(kSrcIndex);
    const Tensor &scale_tensor = context->input(kScaleIndex);
    const Tensor &shift_tensor = context->input(kShiftIndex);
    const Tensor &est_mean_tensor = context->input(kMeanIndex);
    const Tensor &est_variance_tensor = context->input(kVarianceIndex);

    TensorShape tf_shape_src;
    tf_shape_src = src_tensor.shape();
    OP_REQUIRES(context, src_tensor.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        src_tensor.shape().DebugString()));
    OP_REQUIRES(context, scale_tensor.dims() == 1,
                errors::InvalidArgument("scale must be 1-dimensional",
                                        scale_tensor.shape().DebugString()));
    OP_REQUIRES(context, shift_tensor.dims() == 1,
                errors::InvalidArgument("offset must be 1-dimensional",
                                        shift_tensor.shape().DebugString()));
    OP_REQUIRES(context, est_mean_tensor.dims() == 1,
                errors::InvalidArgument("estimated_mean must be 1-dimensional",
                                        est_mean_tensor.shape().DebugString()));
    OP_REQUIRES(
        context, est_variance_tensor.dims() == 1,
        errors::InvalidArgument("estimated_variance must be 1-dimensional",
                                est_variance_tensor.shape().DebugString()));

    // Handle the special case: input with 0 element and 0 batch size.
    Tensor *dst_tensor = nullptr;
    TensorShape workspace_tf_shape;
    if (tf_shape_src.num_elements() == 0) {
      size_t workspace_bytes = 0;
      workspace_tf_shape.AddDim(workspace_bytes);
      HandleEmptyInput(context, tf_shape_src, workspace_tf_shape,
                       scale_tensor.shape(), &dst_tensor);
      return;
    }

    depth_ = static_cast<int>(GetTensorDim(src_tensor, tensor_format_, 'C'));

    // Index of output tensor.
    const size_t kDstIndex = 0;

    // Allocate 5 output TF tensors.
    Tensor *batch_mean_tensor = nullptr;
    Tensor *batch_variance_tensor = nullptr;
    Tensor *saved_mean_tensor = nullptr;
    Tensor *saved_variance_tensor = nullptr;
    Tensor *reserved_space_tensor = nullptr;

    memory::format_tag dnn_fmt;
    if (tensor_format_ == FORMAT_NHWC) {
      dnn_fmt = memory::format_tag::nhwc;
    } else if (tensor_format_ == FORMAT_NCHW) {
      dnn_fmt = memory::format_tag::nchw;
    } else {
      TF_CHECK_OK(
          Status(error::Code::INVALID_ARGUMENT, "Unsupported data format"));
    }

    int inp_batch =
        tf_shape_src.dim_size(GetTensorDimIndex(tensor_format_, 'N'));
    int inp_depth =
        tf_shape_src.dim_size(GetTensorDimIndex(tensor_format_, 'C'));
    int inp_rows =
        tf_shape_src.dim_size(GetTensorDimIndex(tensor_format_, 'H'));
    int inp_cols =
        tf_shape_src.dim_size(GetTensorDimIndex(tensor_format_, 'W'));
    // Set src memory descriptor.
    memory::dims src_dims =
        memory::dims({inp_batch, inp_depth, inp_rows, inp_cols});

    auto src_md = memory::desc(src_dims, memory::data_type::f32, dnn_fmt);

    ZenBatchNormFwdParams fwdParams(src_dims, depth_, epsilon_, is_training_,
                                    src_md, activation_mode_);

    // Get forward batch-normalization op from the primitive caching pool.
    ZenFusedBatchNormFwdPrimitive<T, U> *bn_fwd =
        ZenFusedBatchNormFwdPrimitiveFactory<T, U>::Get(fwdParams);

    // Allocate workspace tensor
    U *ws_data = nullptr;
    if (fwdParams.activation_mode == FusedBNActivationMode::kRelu) {
      memory::desc workspace_md = bn_fwd->GetBatchNormFwdPd()->workspace_desc();
      size_t workspace_bytes = workspace_md.get_size();
      workspace_tf_shape.AddDim(workspace_bytes);

      AllocateTFOutputs(context, scale_tensor.shape(), workspace_tf_shape,
                        &batch_mean_tensor, &batch_variance_tensor,
                        &saved_mean_tensor, &saved_variance_tensor,
                        &reserved_space_tensor);
      if (reserved_space) {
        ws_data = static_cast<U *>(reserved_space_tensor->flat<U>().data());
      }
    } else {
      // There is actually no workspace tensor out, so we make a dummy one.
      size_t workspace_bytes = 0;
      workspace_tf_shape.AddDim(workspace_bytes);
      AllocateTFOutputs(context, scale_tensor.shape(), workspace_tf_shape,
                        &batch_mean_tensor, &batch_variance_tensor,
                        &saved_mean_tensor, &saved_variance_tensor,
                        &reserved_space_tensor);
    }

    if (is_training_) {
      SetMeanVariance(*batch_mean_tensor, *batch_variance_tensor);
    } else {
      SetMeanVariance(est_mean_tensor, est_variance_tensor);
    }

    // Pack scale & shift as "weights":
    // <scale>...<scale><shift>...<shift>
    Tensor weights_tensor;
    TensorShape weights_shape({2, depth_});
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<U>::value, weights_shape,
                                        &weights_tensor));
    U *weights_data = weights_tensor.flat<U>().data();
    const U *scale_tf = scale_tensor.flat<U>().data();
    const U *shift_tf = shift_tensor.flat<U>().data();

    std::memcpy(weights_data, scale_tf, depth_ * sizeof(U));
    std::memcpy(weights_data + depth_, shift_tf, depth_ * sizeof(U));
    char *saved_mean_data_tf =
        reinterpret_cast<char *>(saved_mean_tensor->flat<U>().data());
    std::memcpy(saved_mean_data_tf, reinterpret_cast<char *>(mean_values_),
                depth_ * sizeof(U));

    char *saved_variance_data_tf =
        reinterpret_cast<char *>(saved_variance_tensor->flat<U>().data());
    std::memcpy(saved_variance_data_tf,
                reinterpret_cast<char *>(variance_values_), depth_ * sizeof(U));

    const T *src_data =
        static_cast<T *>(const_cast<T *>(src_tensor.flat<T>().data()));

    // Update the output type
    zenTensorType out_type = zenTensorType::FLOAT;
    // Allocate output (dst) tensor
    TensorShape tf_shape_dst = tf_shape_src;
    zendnnEnv zenEnvObj = readEnv();
    int zenEnableMemPool = zenEnvObj.zenEnableMemPool &&
                           context->expected_output_dtype(0) == DT_FLOAT;
    ZenMemoryPool<T> *zenPoolBuffer=NULL;

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
            context, &dst_tensor, tf_shape_dst, out_links, reset, out_type);
        if (status) {
          zenEnableMemPool = false;
        }
      } else {
        zenEnableMemPool = false;
      }
    }
    if (!zenEnableMemPool) {
      OP_REQUIRES_OK(context, context->allocate_output(kDstIndex, tf_shape_dst,
                                                       &dst_tensor));
    }

    U *weights_op_data = weights_data;
    U *mean_op_data = saved_mean_tensor->flat<U>().data();
    U *variance_op_data = saved_variance_tensor->flat<U>().data();
    T *dst_data = dst_tensor->flat<T>().data();

    // Execute
    bn_fwd->Execute(src_data, weights_op_data, dst_data, mean_op_data,
                    variance_op_data, ws_data);
    float adjust_factor = 1.0;
    if (is_training_) {
      size_t orig_size = src_dims[0] * src_dims[2] * src_dims[3];
      size_t adjust_size = (orig_size > 1) ? (orig_size - 1) : 1;
      adjust_factor = (static_cast<float>(orig_size)) / adjust_size;
    }

    auto mean_data = reinterpret_cast<U *>(saved_mean_data_tf);
    auto variance_data = reinterpret_cast<U *>(saved_variance_data_tf);
    auto batch_mean_data = batch_mean_tensor->flat<U>().data();
    auto batch_variance_data = batch_variance_tensor->flat<U>().data();
    auto est_mean_data = est_mean_tensor.flat<U>().data();
    auto est_variance_data = est_variance_tensor.flat<U>().data();
    if (is_training_) {
      if (exponential_avg_factor_ == U(1.0)) {
        for (int k = 0; k < depth_; k++) {
          batch_mean_data[k] = mean_data[k];
          batch_variance_data[k] =
              static_cast<U>(adjust_factor) * variance_data[k];
        }
      } else {
        U one_minus_factor = U(1.0) - exponential_avg_factor_;
        for (int k = 0; k < depth_; k++) {
          batch_mean_data[k] = one_minus_factor * est_mean_data[k] +
                               exponential_avg_factor_ * mean_data[k];
          batch_variance_data[k] = one_minus_factor * est_variance_data[k] +
                                   exponential_avg_factor_ *
                                       static_cast<U>(adjust_factor) *
                                       variance_data[k];
        }
      }
    } else {
      std::memcpy(batch_mean_data, mean_data, depth_ * sizeof(U));
      std::memcpy(batch_variance_data, variance_data, depth_ * sizeof(U));
    }
    if (zenEnvObj.zenEnableMemPool) {
      unsigned int threadID = getZenTFthreadId(std::this_thread::get_id());
      zenPoolBuffer = ZenMemoryPool<T>::getZenMemPool(threadID);
      if (zenPoolBuffer) {
        auto src_tensor_map = src_tensor.tensor<float, 4>();
        const float *src_tensor_array = src_tensor_map.data();
        zenPoolBuffer->zenMemPoolFree(context, (void *)src_tensor_array);
      }
    }
  }

 private:
  float epsilon_;
  U exponential_avg_factor_;
  TensorFormat tensor_format_;
  bool is_training_;
  U *mean_values_;
  U *variance_values_;
  size_t depth_;  // Batch normalization is performed for per channel.
  FusedBNActivationMode activation_mode_;
  bool reset;
  int in_links, out_links;

  void SetMeanVariance(const Tensor &mean, const Tensor &variance) {
    mean_values_ =
        reinterpret_cast<U *>(const_cast<U *>(mean.flat<U>().data()));
    variance_values_ =
        reinterpret_cast<U *>(const_cast<U *>(variance.flat<U>().data()));
  }

  void HandleEmptyInput(OpKernelContext *context, TensorShape tf_shape_src,
                        TensorShape workspace_tf_shape,
                        TensorShape tf_shape_scale, Tensor **dst_tensor) {
    DCHECK(dst_tensor);
    const size_t kDstIndex = 0;
    OP_REQUIRES_OK(
        context, context->allocate_output(kDstIndex, tf_shape_src, dst_tensor));
    DCHECK(*dst_tensor);
    memset(const_cast<char *>((*dst_tensor)->tensor_data().data()), 0,
           (*dst_tensor)->tensor_data().size());

    Tensor *batch_mean_tensor = nullptr;
    Tensor *batch_variance_tensor = nullptr;
    Tensor *saved_mean_tensor = nullptr;
    Tensor *saved_variance_tensor = nullptr;
    Tensor *reserved_space_tensor = nullptr;
    AllocateTFOutputs(context, tf_shape_scale, workspace_tf_shape,
                      &batch_mean_tensor, &batch_variance_tensor,
                      &saved_mean_tensor, &saved_variance_tensor,
                      &reserved_space_tensor);
  }

  void AllocateTFOutputs(OpKernelContext *context, TensorShape tf_shape_scale,
                         TensorShape workspace_tf_shape,
                         Tensor **batch_mean_tensor,
                         Tensor **batch_variance_tensor,
                         Tensor **saved_mean_tensor,
                         Tensor **saved_variance_tensor,
                         Tensor **reserved_space_tensor) {
    DCHECK(batch_mean_tensor);
    DCHECK(batch_variance_tensor);
    DCHECK(saved_mean_tensor);
    DCHECK(saved_variance_tensor);

    const size_t kBatchMeanIndex = 1;
    const size_t kBatchVarianceIndex = 2;
    const size_t kSavedMeanIndex = 3;
    const size_t kSavedVarianceIndex = 4;
    const size_t kReservedSpaceIndex = 5;

    int num_elements = tf_shape_scale.num_elements();

    // Allocate batch mean output tensor.
    OP_REQUIRES_OK(context,
                   context->allocate_output(kBatchMeanIndex, tf_shape_scale,
                                            batch_mean_tensor));
    DCHECK(*batch_mean_tensor);
    // Set NAN mean value in case of empty input tensor
    auto batch_mean_data = (*batch_mean_tensor)->flat<U>().data();
    std::fill_n(batch_mean_data, num_elements, static_cast<U>(NAN));

    // Allocate batch variance output tensor.
    OP_REQUIRES_OK(context,
                   context->allocate_output(kBatchVarianceIndex, tf_shape_scale,
                                            batch_variance_tensor));
    DCHECK(*batch_variance_tensor);
    // Set NAN variance value in case of empty input tensor
    auto batch_variance_data = (*batch_variance_tensor)->flat<U>().data();
    std::fill_n(batch_variance_data, num_elements, static_cast<U>(NAN));

    // Mean and variance (without Bessel's correction) saved for backward
    // computation to serve as pre-computed mean and variance.
    OP_REQUIRES_OK(context,
                   context->allocate_output(kSavedMeanIndex, tf_shape_scale,
                                            saved_mean_tensor));
    DCHECK(*saved_mean_tensor);
    // Set 0 mean value in case of empty input tensor
    auto saved_mean_data = (*saved_mean_tensor)->flat<U>().data();
    std::fill_n(saved_mean_data, num_elements, static_cast<U>(0));

    OP_REQUIRES_OK(context,
                   context->allocate_output(kSavedVarianceIndex, tf_shape_scale,
                                            saved_variance_tensor));
    DCHECK(*saved_variance_tensor);
    // Set 0 variance value in case of empty input tensor
    auto saved_variance_data = (*saved_variance_tensor)->flat<U>().data();
    std::fill_n(saved_variance_data, num_elements, static_cast<U>(0));

    // Changes to support reserved_space_3 parameter in FusedBatchNormV3.
    if (reserved_space) {
      DCHECK(reserved_space_tensor != nullptr);
      OP_REQUIRES_OK(context, context->allocate_output(kReservedSpaceIndex,
                                                       workspace_tf_shape,
                                                       reserved_space_tensor));
      DCHECK((*reserved_space_tensor) != nullptr);
    }
  }
};

#define REGISTER_ZEN_FUSED_BATCHNORM_CPU(T)                                \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("_ZenFusedBatchNorm").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ZenFusedBatchNormOp<CPUDevice, T, T, false, false>);

TF_CALL_float(REGISTER_ZEN_FUSED_BATCHNORM_CPU);
#undef REGISTER_ZEN_FUSED_BATCHNORM_CPU

#define REGISTER_ZEN_FUSED_BATCHNORM_V2_CPU(T, U)      \
  REGISTER_KERNEL_BUILDER(Name("_ZenFusedBatchNormV2")  \
                              .Device(DEVICE_CPU)      \
                              .TypeConstraint<T>("T")  \
                              .TypeConstraint<U>("U"), \
                          ZenFusedBatchNormOp<CPUDevice, T, U, false, false>);

REGISTER_ZEN_FUSED_BATCHNORM_V2_CPU(float, float);
#undef REGISTER_ZEN_FUSED_BATCHNORM_V2_CPU

// TODO: FusedBatchNormV3 has an additional output that is used to
//       hold intermediate results. This parameter functionality is
//       not implemented on CPU.
#define REGISTER_ZEN_FUSED_BATCHNORM_V3_CPU(T, U)      \
  REGISTER_KERNEL_BUILDER(Name("_ZenFusedBatchNormV3")  \
                              .Device(DEVICE_CPU)      \
                              .TypeConstraint<T>("T")  \
                              .TypeConstraint<U>("U"), \
                          ZenFusedBatchNormOp<CPUDevice, T, U, true, false>);

REGISTER_ZEN_FUSED_BATCHNORM_V3_CPU(float, float);
#undef REGISTER_ZEN_FUSED_BATCHNORM_V3_CPU
}  // namespace tensorflow

#undef GET_FLAG
#undef IS_SET
