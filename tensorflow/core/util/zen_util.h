/*******************************************************************************
 * Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights
 * reserved. Notified per clause 4(b) of the license.
 *******************************************************************************/

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_ZEN_UTIL_H_
#define TENSORFLOW_CORE_UTIL_ZEN_UTIL_H_

#ifdef AMD_ZENDNN

#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/common_runtime/zen_layout_pass.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "zendnn.hpp"

using zendnn::engine;
using zendnn::memory;
using zendnn::primitive;
using zendnn::stream;

namespace tensorflow {

inline void execute_primitives(
    std::vector<zendnn::primitive> &primitives, std::shared_ptr<stream> stream,
    std::vector<std::unordered_map<int, memory>> &net_args) {
  DCHECK_EQ(primitives.size(), net_args.size());
  for (size_t i = 0; i < primitives.size(); ++i) {
    primitives.at(i).execute(*stream, net_args.at(i));
  }
}

//
// LRUCache is a class which implements LRU (Least Recently Used) cache.
// The implementation is taken from
//    tensorflow/core/util/mkl_util.h
//
// The LRU list maintains objects in chronological order based on
// creation time, with the least recently accessed object at the
// tail of LRU list, while the most recently accessed object
// at the head of LRU list.
//
// This class is used to maintain an upper bound on the total number of
// cached items. When the cache reaches its capacity, the LRU item will
// be removed and replaced by a new one from SetOp call.
//
template <typename T>
class LRUCache {
 public:
  explicit LRUCache(size_t capacity) {
    capacity_ = capacity;
    Clear();
  }

  T *GetOp(const string &key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      return nullptr;
    }

    // Move to the front of LRU list as the most recently accessed.
    lru_list_.erase(it->second.lru_iterator);
    lru_list_.push_front(it->first);
    it->second.lru_iterator = lru_list_.begin();
    return it->second.op;
  }

  void SetOp(const string &key, T *op) {
    if (lru_list_.size() >= capacity_) {
      Delete();
    }

    // Insert an entry to the front of the LRU list
    lru_list_.push_front(key);
    Entry entry(op, lru_list_.begin());
    cache_.emplace(std::make_pair(key, std::move(entry)));
  }

  void Clear() {
    if (lru_list_.empty()) {
      return;
    }

    // Clean up the cache
    cache_.clear();
    lru_list_.clear();
  }

 private:
  struct Entry {
    // The entry's value.
    T *op;

    // A list iterator pointing to the entry's position in the LRU list.
    std::list<string>::iterator lru_iterator;

    // Constructor
    Entry(T *op, std::list<string>::iterator it) {
      this->op = op;
      this->lru_iterator = it;
    }

    // Move constructor
    Entry(Entry &&source) noexcept
        : lru_iterator(std::move(source.lru_iterator)) {
      op = std::move(source.op);
      source.op = std::forward<T *>(nullptr);
    }

    // Destructor
    ~Entry() {
      if (op != nullptr) {
        delete op;
      }
    }
  };

  // Remove the least recently accessed entry from LRU list, which
  // is the tail of lru_list_. Update cache_ correspondingly.
  bool Delete() {
    if (lru_list_.empty()) {
      return false;
    }
    string key = lru_list_.back();
    lru_list_.pop_back();
    cache_.erase(key);
    return true;
  }

  // Cache capacity
  size_t capacity_;

  // The cache, a map from string key to a LRU entry.
  std::unordered_map<string, Entry> cache_;

  // The LRU list of entries.
  // The front of the list contains the key of the most recently accessed
  // entry, while the back of the list is the least recently accessed entry.
  std::list<string> lru_list_;
};

class ZenPrimitive {
 public:
  virtual ~ZenPrimitive() {}
  ZenPrimitive() {
    ZenExecutor *ex = ex->getInstance();
    cpu_engine_ = ex->getEngine();
  }
  ZenPrimitive(const engine &cpu_engine) { cpu_engine_ = cpu_engine; }
  unsigned char *DummyData = nullptr;
  engine cpu_engine_;
  const engine &GetEngine() { return cpu_engine_; }
};

class ZenPrimitiveFactory {
 public:
  ZenPrimitiveFactory() {}

  ~ZenPrimitiveFactory() {}

  ZenPrimitive *GetOp(const string &key) {
    auto &lru_cache = ZenPrimitiveFactory::GetLRUCache();
    return lru_cache.GetOp(key);
  }

  void SetOp(const string &key, ZenPrimitive *op) {
    auto &lru_cache = ZenPrimitiveFactory::GetLRUCache();
    lru_cache.SetOp(key, op);
  }

  /// Function to decide whether HW has AVX512 or AVX2
  static inline bool IsLegacyPlatform() {
    return (!port::TestCPUFeature(port::CPUFeature::AVX512F) &&
            !port::TestCPUFeature(port::CPUFeature::AVX2));
  }

  /// Function to check whether primitive reuse optimization is disabled
  static inline bool IsReuseOptDisabled() {
    bool is_reuse_opt_disabled = false;
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ZEN_PRIMITIVE_REUSE_DISABLE", false,
                                   &is_reuse_opt_disabled));
    return is_reuse_opt_disabled;
  }

 private:
  static inline LRUCache<ZenPrimitive> &GetLRUCache() {
    static const int kCapacity = 1024;  // cache capacity
    static thread_local LRUCache<ZenPrimitive> lru_cache_(kCapacity);
    return lru_cache_;
  }
};

// utility class for creating keys of Zen primitive pool.
// The implementation is taken from
//    tensorflow/core/util/mkl_util.h
class FactoryKeyCreator {
 public:
  FactoryKeyCreator() { key_.reserve(kMaxKeyLength); }

  ~FactoryKeyCreator() {}

  void AddAsKey(const string &str) { Append(str); }

  void AddAsKey(const memory::dims &dims) {
    for (unsigned int i = 0; i < dims.size(); i++) {
      AddAsKey<int>(dims[i]);
    }
  }

  template <typename T>
  void AddAsKey(const T data) {
    auto buffer = reinterpret_cast<const char *>(&data);
    Append(StringPiece(buffer, sizeof(T)));
  }

  string GetKey() { return key_; }

 private:
  string key_;
  const char delimiter = 'x';
  const int kMaxKeyLength = 256;
  void Append(StringPiece s) {
    key_.append(string(s));
    key_.append(1, delimiter);
  }
};

}  // namespace tensorflow

#endif  // AMD_ZENDNN
#endif  // TENSORFLOW_CORE_UTIL_ZEN_UTIL_H_
