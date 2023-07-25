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

#ifdef AMD_ZENDNN

#include "tensorflow/core/common_runtime/zen_layout_pass.h"

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/dump_graph.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/util.h"
#include "zendnn_helper.hpp"
#include "zendnn_logging.hpp"

using namespace zendnn;

namespace tensorflow {

// initialize the executor object for use by the kernels
// declared in zen_layout_pass.h
ZenExecutor *ZenExecutor::instance = 0;
ZenMemoryPool *ZenMemoryPool::zenMemPoolArr[ZEN_MEM_POOL_LIMIT] = {NULL};
int ZenMemoryPool::zenMemPoolCount = 0;

const string zen_node_prefix = "Zen";

// For protecting ThreadID Map
std::mutex mtx;

// Map for storing TF thread id
std::map<std::thread::id, unsigned int> TFthreadIDmap;

class ZenLayoutRewritePass : public GraphOptimizationPass {
 public:
  ZenLayoutRewritePass() {
    // Zen Op rewrite information records
    // Caution: Altering the order of records may break some assumptions
    // of this layout pass.
    // Currently, It is not supporting any conv fusion.
    // TODO:: Support Relu+Other fusion
    zrwdb_.push_back({"Conv2D", "ZenConv2D", CheckValidityForDTypeSupported,
                      UpdateZenOpAttrsConv2D});
    zrwdb_.push_back({"_FusedConv2D", "_ZenFusedConv2D",
                      CheckValidityFusedConv2D, UpdateZenOpAttrsFusedConv2D});
    zrwdb_.push_back({"DepthwiseConv2dNative", "ZenDepthwiseConv2dNative",
                      CheckValidityForDTypeSupported, UpdateZenOpAttrsConv2D});
    zrwdb_.push_back({"_FusedDepthwiseConv2dNative",
                      "_ZenFusedDepthwiseConv2dNative",
                      CheckValidityFusedConv2D, UpdateZenOpAttrsFusedConv2D});
    zrwdb_.push_back({"MatMul", "ZenMatMul", CheckValidityForDTypeSupported,
                      UpdateZenOpAttrs});
    zrwdb_.push_back({"_FusedMatMul", "_ZenFusedMatMul",
                      CheckValidityForDTypeSupported, UpdateZenOpAttrs});

    // MatMulBiasAddGelu Op is not defined for TF-Vanilla path.
    // This is converted into _ZenMatMulBiasAddGelu which is
    // supported in TF-ZenDNN stack.
    zrwdb_.push_back({"MatMulBiasAddGelu", "_ZenMatMulBiasAddGelu",
                      CheckValidityForDTypeSupported, UpdateZenOpAttrs});

    zrwdb_.push_back({"BatchMatMul", "ZenBatchMatMul",
                      CheckValidityForDTypeSupported, UpdateZenOpAttrs});
    zrwdb_.push_back({"BatchMatMulV2", "ZenBatchMatMulV2",
                      CheckValidityForDTypeSupported, UpdateZenOpAttrs});
    zrwdb_.push_back(
        {"_ZenFusedConv2D", "_ZenFusedConv2DSum",
         CheckValidityConvBatchNormAddOptimization,
         UpdateZenOpAttrsFusedConv2D});  // Attributes of ZenOpAttrs is same as
                                         // FusedConv2D
    zrwdb_.push_back(
        {"ConcatV2", "ZenConcatV2", CheckValidityZenConcat, UpdateZenOpAttrs});
    zrwdb_.push_back(
        {"Concat", "ZenConcat", CheckValidityZenConcat, UpdateZenOpAttrs});

    // VitisAI specific ops
    zrwdb_.push_back({"VitisAIConv2DWithoutBias", "ZenVitisAIConv2DWithoutBias",
                      CheckValidityVitisAIWithoutBiasSupported,
                      UpdateZenOpAttrsVitisAIConv2D});
    zrwdb_.push_back({"VitisAIConv2D", "ZenVitisAIConv2D",
                      CheckValidityVitisAISupported,
                      UpdateZenOpAttrsVitisAIConv2D});
    zrwdb_.push_back({"VitisAIConv2DWithSum", "ZenVitisAIConv2DWithSum",
                      CheckValidityVitisAIWithSumSupported,
                      UpdateZenOpAttrsVitisAIConv2D});
    zrwdb_.push_back({"VitisAIDepthwiseConv2D", "ZenVitisAIDepthwiseConv2D",
                      CheckValidityVitisAISupported,
                      UpdateZenOpAttrsVitisAIConv2D});
    zrwdb_.push_back({"VitisAIConcatV2", "_ZenVitisAIConcatV2",
                      CheckValidityQuantizedZenConcat, UpdateZenOpAttrs});
    zrwdb_.push_back({"VitisAIMaxPool", "ZenVitisAIMaxPool", RewriteValid,
                      UpdateZenOpAttrs});
    zrwdb_.push_back({"VitisAIAvgPool", "ZenVitisAIAvgPool", RewriteValid,
                      UpdateZenOpAttrs});

    // Quantization Specific Functions
    if (zendnn_getenv_int("ZENDNN_INT8_SUPPORT", 0) == 1) {
      // When ZENDNN_INT8_SUPPORT is set to 1, ZENDNN_ENABLE_MEMPOOL is
      // overwritten to 0. Disabled Memory pool for unified memory model
      // TODO: Add an alternative fix
      zrwdb_.push_back({"QuantizedMaxPool", "ZenQuantizedMaxPool", RewriteValid,
                        UpdateZenOpAttrs});
      zrwdb_.push_back({"QuantizedAvgPool", "ZenQuantizedAvgPool", RewriteValid,
                        UpdateZenOpAttrs});
      zrwdb_.push_back({"QuantizedConv2DWithBiasAndReluAndRequantize",
                        "_ZenQuantizedConv2DWithBiasAndReluAndRequantize",
                        RewriteValid, UpdateQCBR});
      zrwdb_.push_back({"QuantizedConv2DWithBiasAndRequantize",
                        "_ZenQuantizedConv2DWithBiasAndRequantize",
                        RewriteValid, UpdateQCBR});
      zrwdb_.push_back(
          {"QuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
           "_ZenQuantizedConv2DWithBiasSignedSumAndReluAndRequantize",
           RewriteValid, UpdateQCBR});
      zrwdb_.push_back({"QuantizedConv2DWithBiasSumAndReluAndRequantize",
                        "_ZenQuantizedConv2DWithBiasSumAndReluAndRequantize",
                        RewriteValid, UpdateQCBR});
      zrwdb_.push_back(
          {"QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize",
           "_ZenQuantizedDepthwiseConv2DWithBiasAndReluAndRequantize",
           RewriteValid, UpdateQCBR});
      zrwdb_.push_back(
          {"QuantizeV2", "_ZenQuantizeV2", RewriteValid, UpdateZenOpAttrs});
      zrwdb_.push_back({"QuantizedConcatV2", "_ZenQuantizedConcatV2",
                        CheckValidityQuantizedZenConcat, UpdateZenOpAttrs});
    }

    zrwdb_.push_back({"MaxPool", "ZenMaxPool", CheckValidityForDTypeSupported,
                      UpdateZenOpAttrs});
    zrwdb_.push_back({"AvgPool", "ZenAvgPool", CheckValidityForDTypeSupported,
                      UpdateZenOpAttrs});
    zrwdb_.push_back({"_ZenFusedConv2D", "_ZenInception",
                      CheckValidityInception, UpdateZenOpAttrsFusedConv2D});

    // Update Zen Op rewrite information for NHWC format path only. The
    // information not written within the below 'if' true code block is
    // common for both NHWC format path and blocked format path.
    if (zendnn_getenv_int("ZENDNN_CONV_ALGO") != zenConvAlgoType::DIRECT1) {
      // Following Ops are supported for NLP models performance
      // improvement. Currently these ZenOps are supported for NHWC format
      // only.
      // TODO: Supported these Ops for BLOCKED format
      zrwdb_.push_back(
          {"Add", "ZenAdd", CheckValidityForDTypeSupported, UpdateZenOpAttrs});
      zrwdb_.push_back({"AddV2", "ZenAddV2", CheckValidityForDTypeSupported,
                        UpdateZenOpAttrs});
      zrwdb_.push_back(
          {"Sub", "ZenSub", CheckValidityForDTypeSupported, UpdateZenOpAttrs});
      zrwdb_.push_back(
          {"Mul", "ZenMul", CheckValidityForDTypeSupported, UpdateZenOpAttrs});
      zrwdb_.push_back({"Maximum", "ZenMaximum", CheckValidityForDTypeSupported,
                        UpdateZenOpAttrs});
      zrwdb_.push_back({"SquaredDifference", "ZenSquaredDifference",
                        CheckValidityForDTypeSupported, UpdateZenOpAttrs});
      zrwdb_.push_back({"Softmax", "ZenSoftmax", CheckValidityForDTypeSupported,
                        UpdateZenOpAttrs});
      zrwdb_.push_back({"ConjugateTranspose", "ZenConjugateTranspose",
                        RewriteValid, UpdateZenOpAttrs});
      zrwdb_.push_back(
          {"Transpose", "ZenTranspose", RewriteValid, UpdateZenOpAttrs});
      zrwdb_.push_back({"InvertPermutation", "ZenInvertPermutation",
                        RewriteValid, UpdateZenOpAttrs});
      zrwdb_.push_back({"FusedBatchNorm", "ZenFusedBatchNorm", RewriteValid,
                        UpdateZenOpAttrs});
      zrwdb_.push_back({"FusedBatchNormV2", "ZenFusedBatchNormV2", RewriteValid,
                        UpdateZenOpAttrs});
      zrwdb_.push_back({"FusedBatchNormV3", "ZenFusedBatchNormV3", RewriteValid,
                        UpdateZenOpAttrs});
      zrwdb_.push_back(
          {"Reshape", "ZenReshape", RewriteValid, UpdateZenOpAttrs});
    }
    // TF training ops list. This ops list is taken from following file,
    // - tensorflow/core/kernels/training_ops.cc
    // Used to check whether graph has inference ops only as TF+ZenDNN
    // currently does not support training, it supports inference only.
    tf_training_ops.push_back("ApplyGradientDescent");
    tf_training_ops.push_back("ApplyAdadelta");
    tf_training_ops.push_back("ResourceSparseApplyAdadelta");
    tf_training_ops.push_back("ApplyProximalGradientDescent");
    tf_training_ops.push_back("SparseApplyProximalGradientDescent");
    tf_training_ops.push_back("ApplyAdagrad");
    tf_training_ops.push_back("ApplyAdagradV2");
    tf_training_ops.push_back("ApplyProximalAdagrad");
    tf_training_ops.push_back("SparseApplyAdagrad");
    tf_training_ops.push_back("SparseApplyAdagradV2");
    tf_training_ops.push_back("SparseApplyProximalAdagrad");
    tf_training_ops.push_back("ApplyAdagradDA");
    tf_training_ops.push_back("SparseApplyAdagradDA");
    tf_training_ops.push_back("ApplyFtrl");
    tf_training_ops.push_back("ApplyFtrlV2");
    tf_training_ops.push_back("SparseApplyFtrl");
    tf_training_ops.push_back("SparseApplyFtrlV2");
    tf_training_ops.push_back("ApplyMomentum");
    tf_training_ops.push_back("ApplyKerasMomentum");
    tf_training_ops.push_back("ApplyAdam");
    tf_training_ops.push_back("ApplyAdaMax");
    tf_training_ops.push_back("ApplyRMSProp");
    tf_training_ops.push_back("ApplyCenteredRMSProp");
    tf_training_ops.push_back("ApplyAddSign");
    tf_training_ops.push_back("ApplyPowerSign");
  }

  Status Run(const GraphOptimizationPassOptions &options);

  bool ZenOpRewritePass(std::unique_ptr<Graph> *g);
  bool ZenOpUpdate(std::unique_ptr<Graph> *g);
  typedef struct {
    string tfop_name;
    string zenop_name;
    std::function<bool(const Node *)> check_validity;
    std::function<void(const Node *, NodeBuilder *)> update_zenopattr;
  } ZenOpRewriteRecord;

 private:
  std::vector<ZenOpRewriteRecord> zrwdb_;
  std::vector<string> tf_training_ops;

  inline bool ArgIsList(const OpDef::ArgDef &arg) const {
    return !arg.type_list_attr().empty() || !arg.number_attr().empty();
  }

  inline int GetTensorListLength(const OpDef::ArgDef &arg, const Node *n) {
    CHECK_EQ(ArgIsList(arg), true);
    int N = 0;
    const string attr_name = !arg.type_list_attr().empty()
                                 ? arg.type_list_attr()
                                 : arg.number_attr();
    if (!arg.type_list_attr().empty()) {
      std::vector<DataType> value;
      TF_CHECK_OK(GetNodeAttr(n->def(), attr_name, &value));
      N = value.size();
    } else {
      TF_CHECK_OK(GetNodeAttr(n->def(), attr_name, &N));
    }
    return N;
  }

  inline bool substrmatch(const std::string primary,
                          const std::string sub) const {
    return primary.find(sub) != std::string::npos;
  }

  bool CanOpRunOnCPUDevice(const Node *n) {
    bool result = true;
    const char *const substrCPU = "CPU";

    if (!n->assigned_device_name().empty() &&
        !absl::StrContains(n->assigned_device_name(), substrCPU)) {
      result = false;
    } else if (!n->def().device().empty() &&
               !absl::StrContains(n->def().device(), substrCPU)) {
      result = false;
    }

    if (result == false) {
      zendnnInfo(
          ZENDNN_FWKLOG,
          "ZenLayoutRewritePass::CanOpRunOnCPUDevice: Node skipped for rewrite",
          n->type_string(), ", Op cannot run on CPU.");
    }

    return result;
  }

  const ZenOpRewriteRecord *CheckNodeForZenOpRewrite(const Node *n) const;

  void GetNodesProducingTFTensorList(
      const gtl::InlinedVector<std::pair<Node *, int>, 4> &inputs,
      int *input_idx, int list_length,
      std::vector<NodeBuilder::NodeOut> *output_nodes);

  static bool CheckValidityFusedConv2D(const Node *n) {
    // Return false if the node is not with data type supported by Zen
    // inference. Currently Zen supports inference in float only.
    if (!CheckValidityForDTypeSupported(n)) {
      return false;
    }
    std::vector<string> fused_ops;
    TF_CHECK_OK(GetNodeAttr(n->def(), "fused_ops", &fused_ops));

    return (fused_ops == std::vector<string>{"BiasAdd"} ||
            fused_ops == std::vector<string>{"FusedBatchNorm"} ||
            fused_ops == std::vector<string>{"Relu"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Relu"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Relu6"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Add"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Add", "Relu"} ||
            fused_ops == std::vector<string>{"FusedBatchNorm", "Relu"});
  }

  // Currently TF+ZenDNN supports FP32 inference only, hence this method
  // checks the given node is of float data type.
  // Returns, true if node is of float dataype, false otherwise
  static bool CheckValidityForDTypeSupported(const Node *n) {
    DataType nT;
    TF_CHECK_OK(GetNodeAttr(n->def(), "T", &nT));
    if ((nT == DT_FLOAT) || (nT == DT_BFLOAT16)) {
      return true;
    }
  }

  static bool CheckValidityVitisAIWithoutBiasSupported(const Node *n) {
    if (n->in_edges().size() == 2) {
      return true;
    }
    return false;
  }

  static bool CheckValidityVitisAISupported(const Node *n) {
    if (n->in_edges().size() == 3) {
      return true;
    }
    return false;
  }

  static bool CheckValidityVitisAIWithSumSupported(const Node *n) {
    if (n->in_edges().size() == 4) {
      return true;
    }
    return false;
  }

  // This Routine checks Validity of Inception Ops
  // To Do : Make the function generic for Inception modules with different
  // Convolution sizes ( other than 4 ) To Do : Handle convolution with skip
  // connections that connect to concat ( No known existing network with such
  // pattern )
  static bool CheckValidityInception(const Node *n) {
    // Handle Special pattern
    string concat_pattern = "ZenConcatV2";
    string pattern = "_ZenFusedConv2D";

    // Check all attributes have same value
    bool flag = 0;  // Flag variable is used for return
    int count = 0;  // Count variable keeps tracks of convolutions preceded by
    // concat in Inception Block
    Node *concat_node = NULL;

    // The variables below keep track of the attributes of convolution in
    // Inception Block
    int num_args = 0, prev_num_args = 0;
    string data_format, prev_data_format;
    string padding, prev_padding;
    std::vector<int32> strides, prev_strides;
    std::vector<int32> dilations, prev_dilations;
    std::vector<string> fused_ops, prev_fused_ops;

    // Check the outedge is concat
    for (const Edge *m : n->out_edges()) {
      if (m->dst()->type_string() == concat_pattern) {
        concat_node = m->dst();
        break;
      }
    }

    if (concat_node == NULL) {
      return flag;
    }

    // All incoming edges to Concat are Fused ZenConvD
    for (const Edge *m : concat_node->in_edges()) {
      {
        if (!m->IsControlEdge()) {
          if (m->src()->type_string() == "Const") {
            continue;
          }
          // If there is pattern mismatch Return failure
          if (m->src()->type_string() != pattern) {
            return 0;
          }
          // Since ReadVariableOP is considered a Non-Zen Op the Reorder
          // infrastructure fails and passes reordered weights to InceptionOp
          // causing a failure when blocked format is used, this condition does
          // not affect Frozen Graphs as weights will be passed as Const
          for (const Edge *cm : m->src()->in_edges()) {
            if (!cm->IsControlEdge()) {
              if (cm->src()->type_string() == "ReadVariableOp") {
                return 0;
              }
            }
          }

          fused_ops.clear();
          strides.clear();
          dilations.clear();

          TF_CHECK_OK(GetNodeAttr((m->src())->def(), "num_args", &num_args));
          TF_CHECK_OK(GetNodeAttr((m->src())->def(), "strides", &strides));
          TF_CHECK_OK(GetNodeAttr((m->src())->def(), "padding", &padding));
          TF_CHECK_OK(
              GetNodeAttr((m->src())->def(), "data_format", &data_format));
          TF_CHECK_OK(GetNodeAttr((m->src())->def(), "fused_ops", &fused_ops));
          // Verify if the Attributes of preceding convolution operation in
          // Inception match the current one Skip the equality check for First
          // iteration

          if (count) {
            if (prev_num_args != num_args) {
              return 0;
            }
            if (prev_data_format != data_format) {
              return 0;
            }
            if (prev_padding != padding) {
              return 0;
            }
            if (prev_strides != strides) {
              return 0;
            }
            if (prev_dilations != dilations) {
              return 0;
            }
            if (prev_fused_ops.size() != fused_ops.size()) {
              return 0;
            }

            for (int i = 0; i < fused_ops.size(); i++) {
              if (prev_fused_ops[i] != fused_ops[i]) {
                return 0;
              }
            }
          }
          count++;

          prev_num_args = num_args;
          prev_data_format = data_format;
          prev_padding = padding;
          prev_strides = strides;
          prev_dilations = dilations;
          prev_fused_ops = fused_ops;
        }
      }
    }

    // Ensure count of ZenConv2D in Inception Block is 4
    // Also Ensure it is a fused convolution
    flag = (fused_ops == std::vector<string>{"BiasAdd"} ||
            fused_ops == std::vector<string>{"FusedBatchNorm"} ||
            fused_ops == std::vector<string>{"Relu"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Relu"} ||
            fused_ops == std::vector<string>{"FusedBatchNorm", "Relu"}) &&
           (count == 4);
    return flag;
  }

  // Fall back to Vanilla path if Data Type is not Float32 or BFloat16
  static bool CheckValidityZenConcat(const Node *n) {
    DataType dtype;
    TF_CHECK_OK(GetNodeAttr(n->def(), "T", &dtype));
    int dype_Float32 = 1, dytpe_BFloat16 = 4;
    bool is_concat_supported = 1;
    bool is_blocked =
        zendnn_getenv_int("ZENDNN_CONV_ALGO") == zenConvAlgoType::DIRECT1;
    // To Do
    // Concat support needs better design and rigorous testing for Dec release
    // Better design for Reorder

    // Fall back to vanilla for now for the following cases
    // case 1 : Support limited Data Type

    bool is_dtype_supported =
        (dtype == dype_Float32 || dtype == dytpe_BFloat16);

    // Case 2 : Fall back to vanilla if concat is preceded by ZenMatMul
    // or Non Zen Operations

    // Temporary patch to avoid regression in untested paths
    if (is_blocked) {
      // Check if in coming edges of Node support Blocked Format
      // Convoluton or Maxpool or Average pool supports Blocked Format
      for (const Edge *e : n->in_edges()) {
        if (!e->IsControlEdge()) {
          if ((e->src()->type_string() != "_ZenFusedConv2D") &&
              (e->src()->type_string() != "ZenConv2D") &&
              (e->src()->type_string() != "_FusedConv2D") &&
              (e->src()->type_string() != "Conv2D") &&
              (e->src()->type_string() != "ZenMaxPool") &&
              (e->src()->type_string() != "MaxPool") &&
              (e->src()->type_string() != "ZenAvgPool") &&
              (e->src()->type_string() != "AvgPool") &&
              (e->src()->type_string() != "Const")) {
            is_concat_supported = 0;
            break;
          }
        }
      }
    }
    // Return True only when DataType is Float16 or BFloat16 and concat is not
    // followed by Matrix Multiplication for Direct Path

    return is_dtype_supported && is_concat_supported;
  }

  static bool CheckValidityQuantizedZenConcat(const Node *n) {
    DataType dtype;
    TF_CHECK_OK(GetNodeAttr(n->def(), "T", &dtype));

    if (dtype != DT_QUINT8) return false;

    /* Please check for similar conditions in BLOCKED path
     * as we do in FP32 version of zenConcatV2, if we are seeing any regressions
     * in untested paths for quantized version.
     */
    return true;
  }

  static bool CheckValidityConvBatchNormAddOptimization(const Node *n) {
    std::vector<string> fused_ops;
    int num_inputs = 0, num_args = 0, num_data_inputs = 0;
    TF_CHECK_OK(
        GetNodeAttr(n->def(), "fused_ops",
                    &fused_ops));  // check Validitity based on Fused-ops
    TF_CHECK_OK(GetNodeAttr(n->def(), "num_args", &num_args));
    num_inputs = n->op_def().input_arg_size();
    num_data_inputs =
        num_inputs + num_args;  // check Validity based on Node input

    // validity depends on total num_inputs
    // conv input , filter input , data input , Fused inputs ( scale , mean ,
    // variance , epsilon )
    for (const Edge *e : n->in_edges()) {
      if (!e->IsControlEdge()) {
        num_data_inputs--;
      }
    }
    return ((fused_ops == std::vector<string>{"FusedBatchNorm"} ||
             fused_ops == std::vector<string>{"FusedBatchNorm", "Relu"}) &&
            !num_data_inputs);
  }

  static bool RewriteValid(const Node *n) { return true; }

  // Method to find whether the graph has inference ops only. It returns
  // error status if the graph has training ops.
  Status AreAllInferenceOps(std::unique_ptr<Graph> *g);

  Status ZenOpNodeRewrite(std::unique_ptr<Graph> *g, Node *n,
                          const ZenOpRewriteRecord *ri,
                          std::pair<bool, bool> reorder_flags);

  Status ZenOpInceptionNodeRewrite(std::unique_ptr<Graph> *g, Node *n,
                                   const ZenOpRewriteRecord *ri,
                                   std::pair<bool, bool> reorder_flags);

  static void UpdateZenOpAttrs(const Node *orig_node, NodeBuilder *nb);

  static void UpdateZenOpAttrsConv2D(const Node *orig_node, NodeBuilder *nb);

  static void UpdateZenOpAttrsVitisAIConv2D(const Node *orig_node,
                                            NodeBuilder *nb);

  static void UpdateZenOpAttrsFusedConv2D(const Node *orig_node,
                                          NodeBuilder *nb);
  static void UpdateQCBR(const Node *orig_node, NodeBuilder *nb);

  Status CopyInputs(const Node *orig_node,
                    const gtl::InlinedVector<std::pair<Node *, int>, 4> &inputs,
                    NodeBuilder *nb);

  std::unordered_map<Node *, std::pair<bool, bool>> GetReorderFlags(
      std::vector<Node *> &nodes);

  bool AddReorderAttrs(std::unique_ptr<Graph> *g);
};

REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 1,
                      ZenLayoutRewritePass);
// REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 1,
// ZenLayoutRewritePass);

bool delete_successor_node(std::unique_ptr<Graph> *, Node *, Node *, int);

const ZenLayoutRewritePass::ZenOpRewriteRecord *
ZenLayoutRewritePass::CheckNodeForZenOpRewrite(const Node *n) const {
  CHECK_NOTNULL(n);

  for (auto zrwr = zrwdb_.cbegin(); zrwr != zrwdb_.cend(); ++zrwr) {
    if (n->type_string().compare(zrwr->tfop_name) == 0 &&
        zrwr->check_validity(n)) {
      return &*zrwr;
    }
  }

  return nullptr;
}

// Check if Node is the first Zen Node of a graph
bool check_is_first_zennode(std::unique_ptr<Graph> *g, Node *m,
                            std::string zen_prefix) {
  std::vector<Node *> order;
  GetReversePostOrder(**g, &order);

  for (Node *n : order) {
    if ((n->type_string()).find(zen_prefix) != std::string::npos) {
      // Found a ZenOp
      return (n == m);
    }
  }
  return 0;
}

// Check if Node is the last Zen Node of a graph
bool check_is_last_zennode(std::unique_ptr<Graph> *g, Node *m,
                           std::string zen_prefix) {
  std::vector<Node *> order;
  GetReversePostOrder(**g, &order);

  Node *tmp;
  for (Node *n : order) {
    if ((n->type_string()).find(zen_prefix) != std::string::npos) {
      tmp = n;
    }
  }
  if (tmp == m) {
    return 1;
  } else {
    return 0;
  }
}

// Count incoming edges of a node
int check_incoming_edges(const Node *n) {
  int count = 0;
  for (const Edge *e : n->in_edges()) {
    if (e->src()->type_string() == "Const") {
      continue;
    }
    if (!e->IsControlEdge()) {
      count++;
    }
  }
  return count;
}

// Count Outgoing edges of a node
int check_outgoing_edges(const Node *n) {
  int count = 0;
  for (const Edge *e : n->out_edges()) {
    if (!e->IsControlEdge()) {
      count++;
    }
  }
  return count;
}

// Count Outgoing edges of a node for a specific pattern
int pattern_match_outedges(const Node *n, string pattern) {
  int count = 0;
  for (const Edge *e : n->out_edges()) {
    if (!e->IsControlEdge()) {
      if (e->dst()->type_string() == pattern) {
        count = 1;
        break;
      }
    }
  }
  return count;
}

// conv_pattern is expected to be variants of Conv2D patterns
// pad_pattern is expected to be Pad op
// Upon Pattern Match, the Pad op is removed. Conv2D op has been updated
// previously. This pattern (Pad -> Conv2D/FusedConv2D) is observed in ResNet50
// and other ResNet variants
bool ZenFusePadConv(std::unique_ptr<Graph> *g, Node *orig_node,
                    string conv_pattern, string pad_pattern) {
  bool flag = 0;                   // return value of ZenFusePadConv Function
  Node *explicit_pad_node = NULL;  // Explicit Pad op will be removed
  int source_slot;                 // Source output of incoming edge for Pad op
  string padding = "";  // To check that padding type is set to EXPLICIT
  string explicit_pad = "EXPLICIT";  // Padding type that should be in Conv2D

  // If current node is not Pad, return false
  if (orig_node->type_string() != pad_pattern) {
    return flag;
  }
  // Check incoming edges to Pad op (orig_node)
  for (const Edge *n : orig_node->in_edges()) {
    if (!n->IsControlEdge()) {
      // Skip if previous node is Const
      if (n->src()->type_string() == "Const") {
        continue;
      }
      // Store the Pad node
      explicit_pad_node = n->dst();
      // Store source output of incoming edge for Pad op
      source_slot = n->src_output();
      // Check outgoing edges from Pad op (orig_node)
      for (const Edge *e : explicit_pad_node->out_edges()) {
        if (!e->IsControlEdge()) {
          // check for 2nd  pattern (Conv2D)
          if (e->dst()->type_string() == conv_pattern) {
            // If padding type is not EXPLICIT_VALID, fusion of Pad op
            // cannot be performed
            TF_CHECK_OK(GetNodeAttr((e->dst())->def(), "padding", &padding));
            if (padding != explicit_pad) {
              return 0;
            }
            // Remove Pad node as its Fused with Conv2D (FusedPadConv2D)
            if (delete_successor_node(g, explicit_pad_node, n->src(),
                                      source_slot)) {
              // return true if deletion is successful
              return 1;
            }  // end of if condition to validate deletion of Pad node
          }    // end of if condition to check conv_pattern
        }      // end of if condition to check control edges
      }        // end of for loop for out edges
    }          // end of if condition to check control edge
  }            // end of for loop for in edges
  // return false/ flag as Pad removal is not performed
  return flag;
}  // End of ZenFusePadConv function

// conv_pattern is expected to be variants of Conv2D patterns
// add_pattern is expected to be variants of Add pattern
// Upon Pattern Match update outgoing and incoming Edges of Fused node
// Delete Fused node

// To DO : Extend to FuseConvBias when there is some usecase
bool ZenFuseConvBatchnormAdd(std::unique_ptr<Graph> *g, const Node *orig_node,
                             string conv_pattern, string add_pattern,
                             string activation_pattern) {
  bool flag = 0;                  // return value of FuseConvAdd Function
  std::vector<string> fused_ops;  // Ensure the fused_ops is for Batchnorm
  Node *fuse_node,
      *curr_node;  // fuse_node is node that will be fused with convolution
  std::unordered_set<Node *> unique_node;

  if (check_outgoing_edges(orig_node) != 1) {
    return flag;
  }

  for (const Edge *n : orig_node->out_edges()) {
    if (!n->IsControlEdge()) {
      // check for conv_pattern
      if (n->dst()->type_string() == conv_pattern) {
        curr_node = n->dst();
        TF_CHECK_OK(GetNodeAttr(curr_node->def(), "fused_ops", &fused_ops));
        if (fused_ops[0] != "FusedBatchNorm") {
          break;
        }
        for (const Edge *e : curr_node->out_edges()) {
          if (!e->IsControlEdge()) {
            // check for 2nd  pattern
            // add_pattern from argument is asssumed to be of Vanilla ADDV2
            // optype. Vanilla AddV2 may be rewritten to Zen optype 'ZenAddV2'
            // during Zen Op rewrite pass. In such case the pattern match has to
            // be checked against Zen optype. Zen AddV2 optype differs from
            // Vanilla AddV2 optype by having the prefix 'Zen'.
            if ((e->dst()->type_string() == add_pattern) ||
                (e->dst()->type_string() == "Zen" + add_pattern)) {
              fuse_node = e->dst();
              // This check ensures 2 incoming edges - Current implementation
              // only supports AddV2 Check for pattern can be removed in
              // subsequent versions
              if (check_incoming_edges(fuse_node) == 2 &&
                  pattern_match_outedges(fuse_node, activation_pattern)) {
                // Handle Outgoing Edges
                for (const Edge *f : fuse_node->out_edges()) {
                  if (f->IsControlEdge()) {
                    auto result = unique_node.insert(e->src());
                    if (result.second) {
                      (*g)->AddControlEdge(e->src(), f->dst(), true);
                    }
                  } else {
                    auto result = (*g)->AddEdge(e->src(), e->src_output(),
                                                f->dst(), f->dst_input());
                    DCHECK(result != nullptr);
                  }
                }
                unique_node.clear();

                // Handle Incoming edges
                for (const Edge *d : fuse_node->in_edges()) {
                  if (d->IsControlEdge()) {
                    auto result = unique_node.insert(d->src());
                    if (result.second) {
                      (*g)->AddControlEdge(d->src(), n->dst(), true);
                    }
                  } else {
                    if (d->src() != curr_node) {
                      flag = 1;
                      int input_slot_index = 0;
                      // Note - We are using additional slot to store the data
                      // input from Add Operation
                      int output_slot_index = 6;
                      // first 5 outputs are being used by Conv , Filter , Fused
                      // batchnorm params Use 6th index
                      auto result = (*g)->AddEdge(d->src(), input_slot_index,
                                                  n->dst(), output_slot_index);
                      DCHECK(result != nullptr);
                      break;
                    }
                  }
                }

                // Remove Add node as its Fused with Convolution
                unique_node.clear();
                if (flag == 1) {
                  (*g)->RemoveNode(fuse_node);
                }
                break;
              }
            }
            break;
          }
        }
      }
      break;
    }
  }
  return flag;
}

// Check for a specific pattern in all incoming edges of a node
// Example case - check if all incoming edges of a node are _FuseConv2D
int pattern_match_inedges(const Node *curr_node, string pattern) {
  int count = 0;
  for (const Edge *n : curr_node->in_edges()) {
    {
      if (!n->IsControlEdge()) {
        // To do : Store these strings in a vector , FusedConv2D can be added
        // later on This routine check is specific to ZenOperations  for
        // attribute updation currently
        if (n->src()->type_string() == "Const" ||
            n->src()->type_string() == ("_ZenFusedConv2D")) {
          continue;
        }
        if (n->src()->type_string() != pattern) {
          return 0;
        }
        count++;
      }
    }
  }
  return count;
}

// update links to successor and delete
bool delete_successor_node(std::unique_ptr<Graph> *g, Node *node,
                           Node *source_node, int source_outputslot) {
  std::unordered_set<Node *> unique_node;

  // Handle outdoing edges
  for (const Edge *f : node->out_edges()) {
    if (f->IsControlEdge()) {
      auto result = unique_node.insert(source_node);
      if (result.second) {
        (*g)->AddControlEdge(source_node, f->dst(), true);
      }
    } else {
      auto result = (*g)->AddEdge(source_node, source_outputslot, f->dst(),
                                  f->dst_input());
      DCHECK(result != nullptr);
    }
  }
  unique_node.clear();

  // Handle Incoming edges
  for (const Edge *d : node->in_edges()) {
    if (d->IsControlEdge()) {
      auto result = unique_node.insert(d->src());
      if (result.second) {
        for (const Edge *f : node->out_edges()) {
          if (!f->IsControlEdge()) {
            (*g)->AddControlEdge(d->src(), f->dst(), true);
          }
        }
      }
    }
  }
  unique_node.clear();
  (*g)->RemoveNode(node);
  return 1;
}

// The routine Checks if MaxPool is followed by Relu and updates edge for a
// pattern match To Do : MaxPoolRelu Fusion
const Edge *CheckMaxPoolRelu(const Edge *e, string pattern) {
  Node *tmp_node;
  if (e->dst()->type_string() == "MaxPool" && pattern == "Relu") {
    tmp_node = e->dst();
    for (const Edge *d : tmp_node->out_edges()) {
      if (!d->IsControlEdge()) {
        if (d->dst()->type_string() == pattern) {
          e = d;
          break;
        }
      }
    }
  }
  return e;
}

// The routine Checks if AvgPool is followed by Relu and updates edge for a
// pattern match To Do : AvgPoolRelu Fusion
const Edge *CheckAvgPoolRelu(const Edge *e, string pattern) {
  Node *tmp_node;
  if (e->dst()->type_string() == "AvgPool" && pattern == "Relu") {
    tmp_node = e->dst();
    for (const Edge *d : tmp_node->out_edges()) {
      if (!d->IsControlEdge()) {
        if (d->dst()->type_string() == pattern) {
          e = d;
          break;
        }
      }
    }
  }
  return e;
}

// Reorder Activate -
// Example use case - [ConvolutionBias - Mpool - Relu ] -> [ ConvolutionBiasRelu
// - Mpool ] Example use case - [ConvolutionBias - Concat - Relu ] -> [
// ConvolutionBiasRelu - Concat ] Example use case - [ConvolutionBias - Concat -
// Mpool - Relu ] -> [ ConvolutionBiasRelu - Concat - Mpool ]
// To Do : MaxPoolRelu Fusion
int ReorderActivation(std::unique_ptr<Graph> *g, const Node *orig_node,
                      string pattern1, string pattern2, string pattern3) {
  bool flag = 0;
  int count;
  Node *o_node, *curr_node;
  std::unordered_set<Node *> unique_node;

  if (check_outgoing_edges(orig_node) != 1) {
    return flag;
  }

  for (const Edge *n : orig_node->out_edges()) {
    if (!n->IsControlEdge()) {
      count = pattern_match_inedges(n->dst(), pattern3);
      if ((n->dst()->type_string() == pattern1) && count) {
        curr_node = n->dst();
        for (const Edge *e : curr_node->out_edges()) {
          // check for Maxpool Relu pattern
          e = CheckMaxPoolRelu(e, pattern2);
          if (e->dst()->type_string() == pattern2) {
            o_node = e->dst();
            // For successful pattern match with count > 1 Attribute updation
            // happens Relu gets deleted  when count equals 1
            if (check_incoming_edges(o_node) == 1 && count == 1) {
              delete_successor_node(g, o_node, e->src(), e->src_output());
            }
          }
          flag = 1;
          break;
        }
      }
    }
  }
  return flag;
}

// Remove Successor Node
bool ZenOpRemoveSuccessor(std::unique_ptr<Graph> *g, const Node *orig_node,
                          string pattern) {
  bool pattern_found = 0;
  Node *o_node;
  std::unordered_set<Node *> unique_node;
  if (check_outgoing_edges(orig_node) != 1) {
    return pattern_found;
  }

  for (const Edge *e : orig_node->out_edges()) {
    if (!e->IsControlEdge()) {
      if (e->dst()->type_string() == pattern) {  // check for Pattern Match
        o_node = e->dst();

        if (check_incoming_edges(o_node) ==
            1) {  // ensure the incoming edges are 1
          delete_successor_node(g, o_node, e->src(), e->src_output());
          pattern_found = 1;
          break;
        }
      }
    }
  }
  return pattern_found;
}

bool FuseCBR(std::unique_ptr<Graph> *g, const Node *orig_node, string pattern) {
  return ZenOpRemoveSuccessor(g, orig_node, pattern) ||
         ReorderActivation(g, orig_node, "MaxPool", pattern, "_FusedConv2D") ||
         ReorderActivation(g, orig_node, "ConcatV2", pattern, "_FusedConv2D");
}

void ZenLayoutRewritePass::UpdateQCBR(const Node *orig_node, NodeBuilder *nb) {
  DataType Tinput, Tfilter, out_type;
  string padding;
  string data_format("NHWC");
  std::vector<int32> strides, dilations, padding_list;
  bool has_padding_list = HasNodeAttr(orig_node->def(), "padding_list");

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "Tinput", &Tinput));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "Tfilter", &Tfilter));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "out_type", &out_type));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "dilations", &dilations));
  if (has_padding_list) {
    TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding_list", &padding_list));
  }

  Node *filter_node = nullptr;
  TF_CHECK_OK(orig_node->input_node(1, &filter_node));

  // Add attributes to new node.
  nb->Attr("Tinput", Tinput);
  nb->Attr("Tfilter", Tfilter);
  nb->Attr("out_type", out_type);
  nb->Attr("padding", padding);
  nb->Attr("is_filter_const", filter_node->IsConstant());
  nb->Attr("strides", strides);
  nb->Attr("dilations", dilations);
  nb->Attr("T", out_type);
  nb->Attr("data_format", data_format);
  if (has_padding_list) {
    nb->Attr("padding_list", padding_list);
  }

  // Requantization attr Tbias.
  DataType Tbias;
  Status bias_status = GetNodeAttr(orig_node->def(), "Tbias", &Tbias);
  if (bias_status.ToString() == "OK") {
    nb->Attr("Tbias", Tbias);
  }
}

void ZenLayoutRewritePass::UpdateZenOpAttrs(const Node *orig_node,
                                            NodeBuilder *nb) {
  string name;
  AttrSlice attr_list(orig_node->def());

  for (auto iter = attr_list.begin(); iter != attr_list.end(); ++iter) {
    name = iter->first;
    // since the reorder attributes , links and reset  are handled separately
    // we skip their inclusion here to avoid duplicate attrs
    // To Do : UpdateZenOpAttrs to be replaced by UpdateMatmul and UpdateConcat
    if (name == "reorder_before" || name == "reorder_after" ||
        name == "in_links" || name == "out_links" || name == "reset") {
      continue;
    }
    auto attr = iter->second;

    nb->Attr(name, attr);
  }
}

// Used internally in UpdateZenOpAttrsConv2D and UpdateZenOpAttrsFusedConv2D
// to update padding attribute according to PadConv2D fusion
// -----------------------------------------------------------------
// Input parameters
// padding: padding attribute of orig_node (expected to be variants of Conv2D)
// orig_node: Node with which Pad op needs to be fused
// explicit_paddings: a vector of padding values for each dimension
// Returns: true if fusion can take place (otherwise false)
// -----------------------------------------------------------------
bool updateAttributePadConv2D(string padding, const Node *orig_node,
                              std::vector<int32> &explicit_paddings) {
  // Part of PadConv2D fusion
  // If padding is VALID and the current FusedConv2D op is preceded by Pad op,
  // then we are updating the padding attribute to EXPLICIT and setting
  // explicit_paddings attribute
  // If padding is EXPLICIT and the pattern Pad op -> FusedConv2D op exists,
  // then we are updating the explicit_paddings attribute only
  // TODO: To handle padding = SAME, we need to introduce a new padding
  // type (this case has not been observed in real models)
  // In ResNet models, explicit padding will be stride-1 but in this
  // optimization, we are fusing all padding values, hence the check is not
  // needed.
  string valid_pad = "VALID";
  string explicit_pad = "EXPLICIT";
  string pad_pattern = "Pad";

  // if padding is not VALID or EXPLICIT, fusion cannot be performed
  if (padding != valid_pad && padding != explicit_pad) {
    return false;
  }

  // Check incoming edges to origin node (FusedConv2D)
  for (const Edge *m : orig_node->in_edges()) {
    // Skip if previous node is Const
    if (m->src()->type_string() == "Const") {
      continue;
    }
    // If previous node is pad_pattern, pattern (Pad op -> FusedConv2D op) has
    // been found
    if (m->src()->type_string() == pad_pattern) {
      // Get original explicit padding values if padding = EXPLICIT
      std::vector<int32> explicit_paddings_orig = {};
      if (padding == explicit_pad) {
        TF_CHECK_OK(GetNodeAttr(orig_node->def(), "explicit_paddings",
                                &explicit_paddings_orig));
      }
      // input will hold the const op before Pad op
      Node *input = nullptr;
      // Index 0 has the input data and Index 1 has the padding values (which is
      // needed)
      TF_CHECK_OK((m->src())->input_node(1, &input));
      // Check if input is constant
      if (input->IsConstant()) {
        Tensor explicit_padding_tensor;
        // value attribute has the Tensor with explicit padding values
        TF_CHECK_OK(
            GetNodeAttr((input)->def(), "value", &explicit_padding_tensor));
        // Number of elements in explicit_padding_tensor (should be 8)
        int num_elements = explicit_padding_tensor.NumElements();
        // Padding values are of datatype int32
        typedef int32 T;
        // padding_1d_tensor is an Eigen Tensor
        auto padding_1d_tensor = explicit_padding_tensor.flat<T>();
        // For dimension i (starting from 0), the padding values
        // will be at 2*i and 2*i + 1
        for (int index_pad = 0; index_pad < num_elements; index_pad++) {
          if (padding == valid_pad)
          // Insert the padding value at index i
          {
            explicit_paddings.insert(explicit_paddings.begin() + index_pad,
                                     padding_1d_tensor(index_pad));
          } else if (padding == explicit_pad) {
            explicit_paddings.insert(explicit_paddings.begin() + index_pad,
                                     padding_1d_tensor(index_pad) +
                                         explicit_paddings_orig.at(index_pad));
          }
        }  // end of for loop for padding values
        // Set padding_update to 1 (as PadConv2D can be performed)
        return true;
      }  // end of if condition to check constant op
    }    // end of if condition for Pad op
  }      // end of for loop for input edges for FusedConv2D op
  return false;
}  // end of updateAttributePadConv2D()

// Copies the attributes from Conv2D op to ZenConv2D op
// padding and explicit_paddings attributes are updated accordingly to PadConv2D
// fusion
void ZenLayoutRewritePass::UpdateZenOpAttrsConv2D(const Node *orig_node,
                                                  NodeBuilder *nb) {
  DataType T;
  string data_format;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> dilations;
  std::vector<int32> explicit_paddings = {};

  // Get attributes from TfOp node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "dilations", &dilations));

  // padding_update determines if padding attributes needs to be modified
  bool padding_update = false;
  // PadConv2D fusion can be done for VALID and EXPLICIT padding
  if (padding != "SAME")
    // Check if PadConv2D fusion can be done and get the padding values
    padding_update =
        updateAttributePadConv2D(padding, orig_node, explicit_paddings);
  // Update ZenOp with attributes from TfOp
  nb->Attr("T", T);
  nb->Attr("strides", strides);
  // Update padding attribute for PadConv2D fusion
  if (padding_update == true) {
    nb->Attr("padding", "EXPLICIT");                   // Updates padding type
    nb->Attr("explicit_paddings", explicit_paddings);  // sets padding values
  }
  // Padding attribute for condition when fusion is not performed
  else {
    nb->Attr("padding", padding);
    // If padding is EXPLICIT, then explicit_paddings attribute needs to be set
    if (padding == "EXPLICIT") {
      std::vector<int32> explicit_paddings_tmp = {};
      TF_CHECK_OK(GetNodeAttr(orig_node->def(), "explicit_paddings",
                              &explicit_paddings_tmp));
      nb->Attr("explicit_paddings", explicit_paddings_tmp);
    }
  }
  nb->Attr("data_format", data_format);
  nb->Attr("dilations", dilations);
}

// Copies the attributes from FusedConv2D op to ZenFusedConv2D op
// padding and explicit_paddings attributes are updated accordingly to
// PadFusedConv2D fusion
void ZenLayoutRewritePass::UpdateZenOpAttrsFusedConv2D(const Node *orig_node,
                                                       NodeBuilder *nb) {
  DataType T;
  int num_args;
  float epsilon;
  string data_format;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> dilations;
  std::vector<int32> explicit_paddings = {};

  // Get attributes from TfOp node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "num_args", &num_args));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "dilations", &dilations));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "epsilon", &epsilon));

  // padding_update determines if padding attributes needs to be modified
  bool padding_update = false;
  // PadFusedConv2D fusion can be done for VALID and EXPLICIT padding
  if (padding != "SAME")
    // Check if PadFusedConv2D fusion can be done and get the padding values
    padding_update =
        updateAttributePadConv2D(padding, orig_node, explicit_paddings);
  // Update ZenOp with attributes from TfOp
  nb->Attr("T", T);
  nb->Attr("num_args", num_args);
  nb->Attr("strides", strides);
  // Update padding attribute for PadConv2D fusion
  if (padding_update == true) {
    nb->Attr("padding", "EXPLICIT");                   // Updates padding type
    nb->Attr("explicit_paddings", explicit_paddings);  // sets padding values
  }
  // Padding attribute for condition when fusion is not performed
  else {
    nb->Attr("padding", padding);
    // If padding is EXPLICIT, then explicit_paddings attribute needs to be set
    if (padding == "EXPLICIT") {
      std::vector<int32> explicit_paddings_tmp = {};
      TF_CHECK_OK(GetNodeAttr(orig_node->def(), "explicit_paddings",
                              &explicit_paddings_tmp));
      nb->Attr("explicit_paddings", explicit_paddings_tmp);
    }
  }
  nb->Attr("data_format", data_format);
  nb->Attr("dilations", dilations);
  nb->Attr("epsilon", epsilon);
}

void ZenLayoutRewritePass::UpdateZenOpAttrsVitisAIConv2D(const Node *orig_node,
                                                         NodeBuilder *nb) {
  DataType Tinput, Tfilter, Tbias, Toutput, Tsum;
  string data_format;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> dilations;
  std::vector<int32> explicit_paddings = {};

  int in_scale, weight_scale, bias_scale, out_scale, sum_scale, add_out_scale,
      intermediate_float_scale;
  bool is_relu;
  bool has_Tsum = HasNodeAttr(orig_node->def(), "Tsum");
  bool has_Tbias = HasNodeAttr(orig_node->def(), "Tbias");
  float relu_alpha = 0.0f;

  // Get attributes from TfOp node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "Tinput", &Tinput));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "Tfilter", &Tfilter));
  if (has_Tbias) {
    TF_CHECK_OK(GetNodeAttr(orig_node->def(), "Tbias", &Tbias));
  }
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "Toutput", &Toutput));
  if (has_Tsum) {
    TF_CHECK_OK(GetNodeAttr(orig_node->def(), "Tsum", &Tsum));
  }
  if (HasNodeAttr(orig_node->def(), "intermediate_float_scale")) {
    TF_CHECK_OK(GetNodeAttr(orig_node->def(), "intermediate_float_scale",
                            &intermediate_float_scale));
  }
  if (HasNodeAttr(orig_node->def(), "relu_alpha")) {
    TF_CHECK_OK(GetNodeAttr(orig_node->def(), "relu_alpha", &relu_alpha));
  }

  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "dilations", &dilations));

  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "in_scale", &in_scale));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "weight_scale", &weight_scale));
  if (HasNodeAttr(orig_node->def(), "bias_scale")) {
    TF_CHECK_OK(GetNodeAttr(orig_node->def(), "bias_scale", &bias_scale));
  }
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "out_scale", &out_scale));
  if (HasNodeAttr(orig_node->def(), "sum_scale")) {
    TF_CHECK_OK(GetNodeAttr(orig_node->def(), "sum_scale", &sum_scale));
  }
  if (HasNodeAttr(orig_node->def(), "add_out_scale")) {
    TF_CHECK_OK(GetNodeAttr(orig_node->def(), "add_out_scale", &add_out_scale));
  }

  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "is_relu", &is_relu));

  // padding_update determines if padding attributes needs to be modified
  bool padding_update = false;
  // PadConv2D fusion can be done for VALID and EXPLICIT padding
  if (padding != "SAME")
    // Check if PadConv2D fusion can be done and get the padding values
    padding_update =
        updateAttributePadConv2D(padding, orig_node, explicit_paddings);
  // Update ZenOp with attributes from TfOp
  nb->Attr("Tfilter", Tfilter);
  nb->Attr("strides", strides);
  // Update padding attribute for PadConv2D fusion
  if (padding_update == true) {
    nb->Attr("padding", "EXPLICIT");                   // Updates padding type
    nb->Attr("explicit_paddings", explicit_paddings);  // sets padding values
  }
  // Padding attribute for condition when fusion is not performed
  else {
    nb->Attr("padding", padding);
    // If padding is EXPLICIT, then explicit_paddings attribute needs to be set
    if (padding == "EXPLICIT") {
      std::vector<int32> explicit_paddings_tmp = {};
      TF_CHECK_OK(GetNodeAttr(orig_node->def(), "explicit_paddings",
                              &explicit_paddings_tmp));
      nb->Attr("explicit_paddings", explicit_paddings_tmp);
    }
  }
  nb->Attr("data_format", data_format);
  nb->Attr("dilations", dilations);

  nb->Attr("Tinput", Tinput);
  nb->Attr("Toutput", Toutput);

  nb->Attr("in_scale", in_scale);
  nb->Attr("weight_scale", weight_scale);
  nb->Attr("bias_scale", bias_scale);
  nb->Attr("out_scale", out_scale);
  if (has_Tbias) {
    nb->Attr("Tbias", Tbias);
    nb->Attr("bias_scale", bias_scale);
  }
  if (has_Tsum) {
    nb->Attr("add_out_scale", add_out_scale);
    nb->Attr("sum_scale", sum_scale);
    nb->Attr("Tsum", Tsum);
  }
  if (HasNodeAttr(orig_node->def(), "intermediate_float_scale")) {
    nb->Attr("intermediate_float_scale", intermediate_float_scale);
  }
  nb->Attr("is_relu", is_relu);
  nb->Attr("relu_alpha", relu_alpha);
}

static void FillInputs(const Node *n,
                       gtl::InlinedVector<Node *, 4> *control_edges,
                       gtl::InlinedVector<std::pair<Node *, int>, 4> *in) {
  control_edges->clear();
  for (const Edge *e : n->in_edges()) {
    if (e->IsControlEdge()) {
      control_edges->push_back(e->src());
    } else {
      (*in)[e->dst_input()] = std::make_pair(e->src(), e->src_output());
    }
  }
  std::sort(control_edges->begin(), control_edges->end());
}

void ZenLayoutRewritePass::GetNodesProducingTFTensorList(
    const gtl::InlinedVector<std::pair<Node *, int>, 4> &inputs, int *input_idx,
    int list_length, std::vector<NodeBuilder::NodeOut> *output_nodes) {
  CHECK_LT(*input_idx, inputs.size());
  CHECK_GT(list_length, 0);
  CHECK_NOTNULL(output_nodes);
  output_nodes->reserve(list_length);

  while (list_length != 0) {
    CHECK_GT(list_length, 0);
    CHECK_LT(*input_idx, inputs.size());
    Node *n = inputs[*input_idx].first;
    int slot = inputs[*input_idx].second;
    // If input node 'n' is just producing a single tensor at
    // output slot 'slot' then we just add that single node.
    output_nodes->push_back(NodeBuilder::NodeOut(n, slot));
    (*input_idx)++;
    list_length--;
  }
}

Status ZenLayoutRewritePass::CopyInputs(
    const Node *old_node,
    const gtl::InlinedVector<std::pair<Node *, int>, 4> &old_node_inputs,
    NodeBuilder *nb) {
  // Number of input slots to old node
  // Input slots are represented by .Input() calls in REGISTER_OP.
  int old_node_input_slots = old_node->op_def().input_arg_size();
  // Actual number of inputs can be greater than or equal to number
  // of Input slots because inputs of type list could be unfolded.
  auto old_node_input_size = old_node_inputs.size();
  DCHECK_GE(old_node_input_size, old_node_input_slots);

  // Copy all inputs of old node to new node.
  int iidx = 0;
  for (int on_slot_idx = 0; on_slot_idx < old_node_input_slots; on_slot_idx++) {
    // An input slot could be a single tensor or a list. We need
    // to handle this case accordingly.
    DCHECK_LT(iidx, old_node_input_size);
    const OpDef::ArgDef &arg = old_node->op_def().input_arg(on_slot_idx);
    if (ArgIsList(arg)) {
      std::vector<NodeBuilder::NodeOut> new_node_inputs;
      int N = GetTensorListLength(arg, old_node);
      GetNodesProducingTFTensorList(old_node_inputs, &iidx, N,
                                    &new_node_inputs);
      nb->Input(new_node_inputs);
    } else {
      nb->Input(old_node_inputs[iidx].first, old_node_inputs[iidx].second);
      iidx++;
    }
  }
  return Status::OK();
}

// This rewrite is specific to Inception Module
// To Do : Make this generic in subsequent checkins
Status ZenLayoutRewritePass::ZenOpInceptionNodeRewrite(
    std::unique_ptr<Graph> *g, Node *orig_node, const ZenOpRewriteRecord *zrwr,
    std::pair<bool, bool> reorder_flags) {
  Node *tmp_node, *concat_node;
  Node *new_node = nullptr;
  std::vector<string> fused_ops = {};
  int num_data_inputs = orig_node->in_edges().size();
  int count = 0;
  string concat_pattern = "ZenConcatV2";
  Status ret_status = Status::OK();

  // Count data inputs
  for (const Edge *e : orig_node->in_edges()) {
    if (e->IsControlEdge()) {
      num_data_inputs--;
    }
  }

  for (const Edge *m : orig_node->out_edges()) {
    if (m->dst()->type_string() == concat_pattern) {
      concat_node = m->dst();
      break;
    }
  }

  gtl::InlinedVector<Node *, 4> control_edges;
  gtl::InlinedVector<std::pair<Node *, int>, 4> inputs(num_data_inputs);
  NodeBuilder nb(orig_node->name().c_str(), zrwr->zenop_name.c_str());
  nb.Device(orig_node->def().device());

  // Copy information from all incoming edges of concat Node ( From all
  // ZenConv2D to ZenInceptionOp
  for (const Edge *m : concat_node->in_edges()) {
    if (!m->IsControlEdge()) {
      if (m->src()->type_string() == "Const") {
        continue;
      }

      // Fill Inputs
      FillInputs(m->src(), &control_edges, &inputs);

      // Copy Multi-Inputs
      // To Do - Make this work with List Datastructure
      ret_status = CopyInputs(m->src(), inputs, &nb);
      if (ret_status != Status::OK()) {
        return ret_status;
      }

      // Its sufficient to update attributes for the one of the incoming concat
      // nodes Handled as part of CheckValidity
      if (count == 0) {
        zrwr->update_zenopattr(const_cast<const Node *>(m->src()), &nb);
        TF_CHECK_OK(GetNodeAttr((m->src())->def(), "fused_ops", &fused_ops));
        nb.Attr("fused_ops", fused_ops);
      }
      count++;
    }
  }
  // Update Reorder Attributes
  nb.Attr("reorder_before", reorder_flags.first);
  nb.Attr("reorder_after", reorder_flags.second);
  nb.Attr("in_links", check_incoming_edges(orig_node));
  nb.Attr("out_links", check_outgoing_edges(orig_node));
  // TODO: Merge the two function calls to one function call
  nb.Attr("reset", check_is_last_zennode(g, orig_node, zen_node_prefix));

  ret_status = nb.Finalize(&**g, &new_node);

  if (ret_status != Status::OK()) {
    return ret_status;
  }

  std::unordered_set<Node *> unique_node;

  // Loop Over all inedges of concat ( Find predecesors )
  for (const Edge *m : concat_node->in_edges()) {
    count++;
    if (!m->IsControlEdge()) {
      if (m->src()->type_string() == "Const") {
        continue;
      }
    }
    // Handle incoming edges
    tmp_node = m->src();
    for (const Edge *e : tmp_node->in_edges()) {
      if (e->IsControlEdge()) {
        auto result = unique_node.insert(e->src());
        if (result.second) {
          (*g)->AddControlEdge(e->src(), new_node, true);
        } else {
          auto result = (*g)->AddEdge(e->src(), e->src_output(), new_node, 0);
        }
      }
    }

    // Handle outgoing edges
    for (const Edge *e : tmp_node->out_edges()) {
      if (e->IsControlEdge()) {
        auto result = unique_node.insert(e->dst());
        if (result.second) {
          (*g)->AddControlEdge(new_node, e->dst(), true);
        }
      }
      auto result =
          (*g)->AddEdge(new_node, e->src_output(), e->dst(), e->dst_input());
      DCHECK(result != nullptr);
    }

    unique_node.clear();
    // Remove Original Node
    new_node->set_assigned_device_name(tmp_node->assigned_device_name());
    (*g)->RemoveNode(tmp_node);

    fused_ops.clear();
    TF_CHECK_OK(GetNodeAttr(new_node->def(), "fused_ops", &fused_ops));
  }

  // Update Links to Concat Node
  // Remove Concat
  for (const Edge *e : new_node->out_edges()) {
    orig_node = e->dst();
    delete_successor_node(g, orig_node, e->src(), e->src_output());
    break;
  }

  return ret_status;
}

Status ZenLayoutRewritePass::ZenOpNodeRewrite(
    std::unique_ptr<Graph> *g, Node *orig_node, const ZenOpRewriteRecord *zrwr,
    std::pair<bool, bool> reorder_flags) {
  int zenconv_maxinputs = 7;  // Maximum inputs supported by ZenConvolution
  DCHECK(zrwr != nullptr);
  DCHECK(orig_node != nullptr);

  Status ret_status = Status::OK();
  Node *new_node = nullptr;
  std::vector<string> fused_ops = {};
  int num_data_inputs = orig_node->in_edges().size();
  for (const Edge *e : orig_node->in_edges()) {
    if (e->IsControlEdge()) {
      num_data_inputs--;
    }
  }

  gtl::InlinedVector<Node *, 4> control_edges;
  gtl::InlinedVector<std::pair<Node *, int>, 4> inputs(num_data_inputs);
  FillInputs(orig_node, &control_edges, &inputs);

  NodeBuilder nb(orig_node->name().c_str(), zrwr->zenop_name.c_str());

  nb.Device(orig_node->def().device());
  ret_status = CopyInputs(orig_node, inputs, &nb);
  if (ret_status != Status::OK()) {
    return ret_status;
  }
  zrwr->update_zenopattr(const_cast<const Node *>(orig_node), &nb);

  // To Do : Update as part of Update ZenSumAttributes
  if ((num_data_inputs == zenconv_maxinputs) &&
      (orig_node->type_string() == "_ZenFusedConv2D")) {
    nb.Input(inputs[num_data_inputs - 1].first,
             inputs[num_data_inputs - 1].second);
  }

  nb.Attr("reorder_before", reorder_flags.first);
  nb.Attr("reorder_after", reorder_flags.second);
  nb.Attr("in_links", check_incoming_edges(orig_node));
  nb.Attr("out_links", check_outgoing_edges(orig_node));
  nb.Attr("reset", check_is_last_zennode(g, orig_node, zen_node_prefix));

  // Add/Update Fused Op Attribute
  if (orig_node->type_string() == "_ZenFusedConv2D" ||
      orig_node->type_string() == "_FusedConv2D" ||
      orig_node->type_string() == "_FusedDepthwiseConv2dNative" ||
      orig_node->type_string() == "_ZenFusedDepthwiseConv2dNative" ||
      orig_node->type_string() == "_ZenFusedConv2DSum" ||
      orig_node->type_string() == "_ZenInception") {
    TF_CHECK_OK(GetNodeAttr(orig_node->def(), "fused_ops", &fused_ops));
    if (FuseCBR(g, orig_node, "Relu")) {
      if (fused_ops.size() == 1) {
        fused_ops.push_back("Relu");
      }
    }
    if (FuseCBR(g, orig_node, "Relu6")) {
      if (fused_ops.size() == 1) {
        fused_ops.push_back("Relu6");
      }
    }
    nb.Attr("fused_ops", fused_ops);
  }
  ret_status = nb.Finalize(&**g, &new_node);
  if (ret_status != Status::OK()) {
    return ret_status;
  }

  std::unordered_set<Node *> unique_node;
  for (const Edge *e : orig_node->in_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->src());
      if (result.second) {
        (*g)->AddControlEdge(e->src(), new_node, true);
      }
    }
  }
  unique_node.clear();

  for (const Edge *e : orig_node->out_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->dst());
      if (result.second) {
        (*g)->AddControlEdge(new_node, e->dst(), true);
      }
    } else {
      auto result =
          (*g)->AddEdge(new_node, e->src_output(), e->dst(), e->dst_input());
      DCHECK(result != nullptr);
    }
  }
  new_node->set_assigned_device_name(orig_node->assigned_device_name());
  (*g)->RemoveNode(orig_node);

  return ret_status;
}

std::unordered_map<Node *, std::pair<bool, bool>>
ZenLayoutRewritePass::GetReorderFlags(std::vector<Node *> &nodes) {
  // nodes is a vector of original nodes marked for rewrite with Zen ops

  // map from node to [reorder_before, reorder_after]
  std::unordered_map<Node *, std::pair<bool, bool>> reorder_flags;
  bool first_reorder_completed = false;  // assuming only one input #TODO
  // When setting reorder_before, we check if the input ops are read ops
  // typically to avoid considering read ops from filter weights as they
  // are reordered anyway in the Zen op. However, for the first op,
  // there will be two read ops, one from weights, and one from input
  // data. To handle this special case, this bool variable is used.

  for (Node *n : nodes) {
    bool reorder_before, reorder_after;
    reorder_before = reorder_after = false;

    for (const Edge *e : n->out_edges()) {
      Node *dst = e->dst();
      if (!dst->IsOp() || e->IsControlEdge()) {
        continue;
      }

      auto it = std::find(nodes.begin(), nodes.end(), dst);
      if (it == nodes.end()) {
        zendnnInfo(ZENDNN_FWKLOG, "ZenLayoutRewritePass::GetReorderFlags: At ",
                   n->name(), " ", n->type_string(), ", non-Zen output - ",
                   dst->name(), " ", dst->type_string());
        // didn't find the next node
        // this means that the next node is not a Zen node
        // thus, we must reorder
        reorder_after = true;
        // can exit the loop since remaining edges won't
        // change this flag
        break;
      }
    }

    for (const Edge *e : n->in_edges()) {
      Node *src = e->src();
      if (!src->IsOp() || e->IsControlEdge()) {
        continue;
      }

      if (substrmatch(src->type_string(), "Const")) {
        // ignore Const ops
        continue;
      }

      // TODO: Needs more testing in new scripts/envs
      if (substrmatch(src->type_string(), "_Arg")) {
        // found a placeholder op
        zendnnInfo(ZENDNN_FWKLOG, "ZenLayoutRewritePass::GetReorderFlags: At ",
                   n->name(), " ", n->type_string(), ", a placeholder op ",
                   src->name(), " ", src->type_string());
        // in this case, we don't need to worry about
        // a read op from data
        first_reorder_completed = true;
        reorder_before = true;
        break;
      }

      // ignore read ops coming from weights
      // TODO: Is this sufficient? Needs testing outside of tf_cnn
      if (substrmatch(src->name(), "read")) {
        // found read op
        // check if it is the first
        if (!first_reorder_completed) {
          // it's the first!
          zendnnInfo(ZENDNN_FWKLOG,
                     "ZenLayoutRewritePass::GetReorderFlags: At ", n->name(),
                     " ", n->type_string(), ", encountered first read op ",
                     src->name(), " ", src->type_string());
          first_reorder_completed = true;
          reorder_before = true;
          break;
        }
        // read op was not first
        // ignore it
        continue;
      }

      auto it = std::find(nodes.begin(), nodes.end(), src);
      if (it == nodes.end()) {
        zendnnInfo(ZENDNN_FWKLOG, "ZenLayoutRewritePass::GetReorderFlags: At ",
                   n->name(), " ", n->type_string(), ", non-Zen input - ",
                   src->name(), " ", src->type_string());
        // didn't find the previous node
        // this means that the previous node is not a Zen node
        // thus, we must reorder
        reorder_before = true;
        // can exit the loop since remaining edges won't
        // change this flag
        break;
      }
    }

    std::pair<bool, bool> n_flags(reorder_before, reorder_after);
    reorder_flags[n] = n_flags;
  }

  // Handle the case of branches separately
  for (Node *n : nodes) {
    // Let A and B be Zen nodes, and X be a non-Zen node
    // rb - reorder_before, ra - reorder_after
    // Handle first case of branching:
    //       A (rb=True, ra)
    //     /   \
        //    X     B(rb, ra=False)
    if (reorder_flags[n].second == false) {
      for (const Edge *e : n->out_edges()) {
        Node *dst = e->dst();
        auto it = std::find(nodes.begin(), nodes.end(), dst);
        if (it != nodes.end()) {
          // found Zen node
          if (reorder_flags[dst].first == true) {
            reorder_flags[n].second = true;
            break;
          }
        }
      }
    }
    // reorder flags set to true cannot be altered
  }

  // Case 2
  for (Node *n : nodes) {
    // Let A and B be Zen nodes, and X be a non-Zen node
    // rb - reorder_before, ra - reorder_after
    // Handle second case of branching:
    //    B(rb=False, ra)   X
    //                  \  /
    //                   A(rb,ra=True)
    if (reorder_flags[n].first == false) {
      for (const Edge *e : n->in_edges()) {
        Node *src = e->src();
        auto it = std::find(nodes.begin(), nodes.end(), src);
        if (it != nodes.end()) {
          // found Zen node
          if (reorder_flags[src].second == true) {
            reorder_flags[n].first = true;
            break;
          }
        }
      }
    }
    // reorder flags set to true cannot be altered
  }

  // Case 3
  for (Node *n : nodes) {
    // Let A be a Zen nodes, and B and X be a Zen/Non Zen node
    // rb - reorder_before, ra - reorder_after
    // Handle third case of branching:
    //    B(rb, ra=True)    X (set ra=True) if one of the siblings has ra=True
    //                  \  /
    //                   A
    if (reorder_flags[n].second == false) {
      for (const Edge *e : n->out_edges()) {
        Node *dst = e->dst();
        for (const Edge *f : dst->in_edges()) {
          Node *src = f->src();
          auto it = std::find(nodes.begin(), nodes.end(), src);
          if (it != nodes.end()) {
            // Found a sibling with reorder after set to True
            if (src != n && reorder_flags[src].second == true) {
              reorder_flags[n].second = true;
              break;
            }
          }
        }
      }
    }
    // reorder flags set to true cannot be altered
  }

  return reorder_flags;
}

bool ZenLayoutRewritePass::AddReorderAttrs(std::unique_ptr<Graph> *g) {
  bool result = false;
  CHECK_NOTNULL(g);

  std::vector<Node *> order;
  GetReversePostOrder(**g, &order);
  std::vector<Node *> zen_nodes;

  for (Node *n : order) {
    std::string op_name = n->type_string();

    // NOTE: Every Zen op must have the prefix "Zen"
    auto found = op_name.find("Zen");
    if (found != std::string::npos) {
      // found a Zen op
      zen_nodes.push_back(n);
    }
  }

  std::unordered_map<Node *, std::pair<bool, bool>> reorder_flags =
      GetReorderFlags(zen_nodes);

  for (Node *n : zen_nodes) {
    std::string node_name = n->name();
    std::string op_name = n->type_string();
    std::pair<bool, bool> n_reorder = reorder_flags[n];

    ZenOpRewriteRecord zrwr;
    for (auto it = zrwdb_.begin(); it < zrwdb_.end(); it++) {
      if (op_name == it->zenop_name) {
        zrwr = *it;  // make a copy of it
        break;
      }
    }

    // rewrite op with a copy containing the new reorder flags
    if (ZenOpNodeRewrite(g, n, &zrwr, n_reorder) == Status::OK()) {
      zendnnInfo(ZENDNN_FWKLOG, "ZenLayoutRewritePass::AddReorderAttrs: Node ",
                 node_name, " ", op_name, " updated reorders to ",
                 n_reorder.first, " ", n_reorder.second);
      result = true;
    }
  }

  return result;
}

bool ZenLayoutRewritePass::ZenOpUpdate(std::unique_ptr<Graph> *g) {
  bool result = false;
  std::vector<Node *> order;
  GetReversePostOrder(**g, &order);
  for (Node *n : order) {
    if (!n->IsOp() || !CanOpRunOnCPUDevice(n)) {
      continue;
    }

    const ZenOpRewriteRecord *zrwr = nullptr;
    if ((zrwr = CheckNodeForZenOpRewrite(n)) != nullptr) {
      string node_name = n->name();
      string op_name = n->type_string();
      std::pair<bool, bool> n_reorder(true, true);
      // To Do : Handle the comparision is better way
      if (!(zrwr->zenop_name).compare("_ZenInception")) {
        if (ZenOpInceptionNodeRewrite(g, n, zrwr, n_reorder) == Status::OK()) {
          zendnnInfo(ZENDNN_FWKLOG, "ZenLayoutRewritePass::ZenOpUpdate: Node ",
                     op_name, " rewritten with ZenOp ", zrwr->zenop_name);
          result = true;
        }
      } else if (ZenOpNodeRewrite(g, n, zrwr, n_reorder) == Status::OK()) {
        zendnnInfo(ZENDNN_FWKLOG, "ZenLayoutRewritePass::ZenOpUpdate: Node ",
                   op_name, " rewritten with ZenOp ", zrwr->zenop_name);
        result = true;
      } else {
        // overwriting the node failed
        zendnnInfo(ZENDNN_FWKLOG,
                   "ZenLayoutRewritePass::ZenOpUpdate: Failed to rewrite node ",
                   node_name, " with ZenOp ", op_name);
        // TODO: Throw error in case ZenDNN permanently corrupts graph after
        // deleting
        //      old node and failing to write new one in its place
      }
    }
  }
  return result;
}

// Method to find whether the graph has inference ops only. It returns error
// status if the graph has training ops.
Status ZenLayoutRewritePass::AreAllInferenceOps(std::unique_ptr<Graph> *g) {
  Status ret_status = Status::OK();
  std::vector<Node *> order;
  GetReversePostOrder(**g, &order);
  for (Node *n : order) {
    if (!n->IsOp()) {
      continue;
    }
    for (auto op = tf_training_ops.cbegin(); op != tf_training_ops.cend();
         ++op) {
      if (n->type_string().find(*op) != string::npos) {
        return Status(error::Code::UNIMPLEMENTED,
                      "Training operation found! Currently TF+ZenDNN "
                      "does not support training. Set environment "
                      "variable ZENDNN_INFERENCE_ONLY to '0' for "
                      "training.");
      }
    }
  }
  return ret_status;
}

bool ZenLayoutRewritePass::ZenOpRewritePass(std::unique_ptr<Graph> *g) {
  bool result = false;
  CHECK_NOTNULL(g);

  // Before we proceed further for Zen Op rewrites first the graph shall be
  // checked for inference ops only as TF+ZenDNN currently does not support
  // training, it supports inference only.
  // TODO: Remove this check for env variable ZENDNN_INFERENCE_ONLY.
  // Currently this check is added only for feb2021 release for any
  // unforeseen worst case of this support.
  if (zendnn_getenv_int("ZENDNN_INFERENCE_ONLY", 1) == 1) {
    TF_CHECK_OK(AreAllInferenceOps(g));
  }

  std::vector<Node *> order;

  DumpGraph("\nBefore ZenRewritePass:\n", &**g);

  // Graph dump to a file before Zenrewrite if FWK is set to 4
  if (zendnnGetLogLevel("FWK") == 5) {
    Graph *graph_before = g->get();
    DumpGraphToFile("ZENDNN_before_rewrite", *graph_before, nullptr, "./");
  }

  // Two passes of Graph optimization

  // First pass implements Basic Fusion Eg. CBR
  result = ZenOpUpdate(g);
  if (!result) {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZenLayoutRewritePass::ZenOpRewritePass: No opportunity for Zen "
               "op conversion found");
  }
  // Second Pass - Enable Fused Optimizations
  // Enable Advanced Graph Optimizations
  GetReversePostOrder(**g, &order);
  for (Node *n : order) {
    if (!n->IsOp() || !CanOpRunOnCPUDevice(n)) {
      continue;
    }
    // Fused Optimizations

    // TODO: Both calls to ZenFusePadConv() (for FusedConv2D and Conv2D) may be
    // merged Check and perform Pad fusion with FusedConv2D (Removes Pad op and
    // expects n to be Pad op)
    if (ZenFusePadConv(g, n, "_ZenFusedConv2D", "Pad")) {
      zendnnInfo(
          ZENDNN_FWKLOG,
          "ZenLayoutRewritePass::ZenOpRewritePass: FusedConvPad Successful");
    }
    // Check and perform Pad fusion with Conv2D (Removes Pad op and expects n to
    // be Pad op)
    else if (ZenFusePadConv(g, n, "ZenConv2D", "Pad")) {
      zendnnInfo(ZENDNN_FWKLOG,
                 "ZenLayoutRewritePass::ZenOpRewritePass: ConvPad Successful");
    } else if (ZenFusePadConv(g, n, "ZenVitisAIConv2D", "Pad")) {
      zendnnInfo(
          ZENDNN_FWKLOG,
          "ZenLayoutRewritePass::ZenOpRewritePass: VitisAIConvPad Successful");
    } else if (ZenFuseConvBatchnormAdd(g, n, "_ZenFusedConv2D", "AddV2",
                                       "Relu")) {
      zendnnInfo(ZENDNN_FWKLOG,
                 "ZenLayoutRewritePass::ZenOpRewritePass: FuseBatchNorm Add "
                 "Successful");
    }
  }
  // Update ZenOP
  result = ZenOpUpdate(g);
  if (!result) {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZenLayoutRewritePass::ZenOpRewritePass: No instance of "
               "FuseBatchNorm found.");
  }

  result = AddReorderAttrs(g);
  if (!result) {
    zendnnInfo(ZENDNN_FWKLOG,
               "ZenLayoutRewritePass::ZenOpRewritePass: No reorder attributes "
               "were updated.");
  }
  DumpGraph("\nAfter ZenRewritePass:\n", &**g);

  // Graph dump to a file after Zenrewrite if FWK is set to 4
  if (zendnnGetLogLevel("FWK") >= 4) {
    Graph *graph_after = g->get();
    DumpGraphToFile("ZENDNN_after_rewrite", *graph_after, nullptr, "./");
  }
  return result;
}

Status ZenLayoutRewritePass::Run(const GraphOptimizationPassOptions &options) {
  if (!IsZenDnnEnabled()) {
    VLOG(2) << "TF-ZENDNN: Disabling ZENDNN INFERENCE";
    return Status::OK();
  }

  if (options.graph == nullptr && options.partition_graphs == nullptr) {
    return Status::OK();
  }

  if (options.graph != nullptr) {
    std::unique_ptr<Graph> *graph = std::move(options.graph);
    ZenOpRewritePass(graph);
    options.graph->reset(graph->release());
  } else {
    for (auto &g : *options.partition_graphs) {
      std::unique_ptr<Graph> *graph = std::move(&g.second);
      ZenOpRewritePass(graph);
      (&g.second)->reset(graph->release());
    }
  }

  return Status::OK();
}

// This function takes thread id(comes from TF threadpool) and coverts
// into integer thread ID using Map.
// Same integer thread ID is used for creating seperate Memory pool for
// inter_op threads.
unsigned int getZenTFthreadId(std::thread::id threadID) {
  static unsigned int numThreads = 0;
  unsigned int intID = -1;
  std::map<std::thread::id, unsigned int>::iterator it;

  it = TFthreadIDmap.find(threadID);
  if (it != TFthreadIDmap.end()) {
    intID = TFthreadIDmap[threadID];
  } else {
    mtx.lock();
    TFthreadIDmap[threadID] = numThreads;
    intID = TFthreadIDmap[threadID];
    numThreads++;
    mtx.unlock();
  }
  return intID;
}

}  // namespace tensorflow

#endif  // AMD_ZENDNN
