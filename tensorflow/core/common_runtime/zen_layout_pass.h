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
#ifndef ZEN_LAYOUT_PASS_H_
#define ZEN_LAYOUT_PASS_H_

#ifdef AMD_ZENDNN

#include <sys/types.h>

#include <mutex>   // std::mutex
#include <thread>  // std::thread

#include "tensorflow/core/graph/graph.h"
#include "zendnn.hpp"
#include "zendnn_helper.hpp"

// This is for supporting future integration with streams
// TODO: Test with maximum no of possible streams and tune
//      ZEN_MEM_POOL_LIMIT accordingly
#define ZEN_MEM_POOL_LIMIT 64

// ZEN_TENSOR_POOL_LIMIT define the limit for active tensors inside pool for
//  given Memory pool
// TODO: Test with increased limit and tune it accordingly
#define ZEN_TENSOR_POOL_LIMIT 32

// ZEN_TENSOR_SIZE_FACTOR increased the max size required for storing the max
// o/p
//  of graph. Currently with CNN and NLP(Bert, WideDeep, DLRM) models, this is
//  fixed with 1.
// TODO: Test with other models and see if its required and tune it acordingly.
//#define     ZEN_TENSOR_SIZE_FACTOR      1

zendnn::zendnnEnv readEnv();

extern unsigned int graph_exe_count;

using namespace zendnn;

namespace tensorflow {

// for testing
extern bool RunZenLayoutRewritePass(std::unique_ptr<Graph> *g);
extern std::mutex mtx;
unsigned int getZenTFthreadId(std::thread::id threadID);
unsigned int getNumTFthreads();

// for single engine and stream
// TODO: Need a complete graph manager entity. This can be moved to within that.
class ZenExecutor {
 private:
  static ZenExecutor *instance;
  engine eng;
  std::vector<std::shared_ptr<stream>> engine_stream;

  ZenExecutor() {
    engine temp_eng(zendnn::engine::kind::cpu, 0);
    eng = temp_eng;
    std::shared_ptr<stream> temp_stream = std::make_shared<stream>(eng);
    std::vector<std::shared_ptr<stream>> temp_vec_stream = {temp_stream};
    engine_stream = temp_vec_stream;
  }

 public:
  static ZenExecutor *getInstance() {
    if (!instance) {
      instance = new ZenExecutor();
    }
    return instance;
  }

  engine getEngine() { return eng; }

  stream getStream() {
    std::shared_ptr<stream> s = engine_stream[engine_stream.size() - 1];
    stream res = *s;
    return res;
  }

  std::shared_ptr<stream> getStreamPtr() {
    return engine_stream[engine_stream.size() - 1];
  }

  void addStream() {
    std::shared_ptr<stream> temp_stream = std::make_shared<stream>(eng);
    engine_stream.push_back(temp_stream);
  }
};

enum class zenTensorType { QINT8 = 0, QUINT8 = 1, FLOAT = 2, BFLOAT = 3 };

// zenTensorPool structure holds zenTensorPtr with its state 0, -1 and > 0
//  Possible states -1(not allocated),
//                  0(allocated and free)
//                  >0(occupied, no. of links with other node)
//  zenTensorSize is size of pointed memory(no. of elements inside tensor)
typedef struct persistenTensorState {
  Tensor *zenTensorPtr;
  void *raw_buff;
  int zenTensorPtrStatus;
  unsigned long zenTensorSize;
  zenTensorType zenType;
} zenTensorPool;

// class ZenMemoryPool holds description about memory pool with all tensor
// pointer
//  created inside pool.
class ZenMemoryPool {
  // zenMemPoolArr hold no. of memory pool exist, In case of multiple streams,
  //  each stream will have its own memory pool. Currently ZEN_MEM_POOL_LIMIT
  //  is the limit, can be made dynamic in future for supporting streams >
  //  ZEN_MEM_POOL_LIMIT. Single Memory pool object will be created for each
  //  stream, every call to getZenMemPool( <fixed index>) will return same
  //  object.
  // zenMemPoolCount hold the no of active memory pool
 private:
  static ZenMemoryPool *zenMemPoolArr[ZEN_MEM_POOL_LIMIT];
  static int zenMemPoolCount;

  // Initialize pool object with default values
  ZenMemoryPool() {
    zenTensorPoolSize = 0;
    max_shape = TensorShape();
    zenTensorPoolReset = false;
    zenTensorPoolArr = NULL;

    zendnnEnv zenEnvObj = readEnv();
    zenEnableMemPool = zenEnvObj.zenEnableMemPool;

    //  To enable/disable Reduced memory pool(default) OR Fixed memory
    //  pool tensor from env variable.
    //  Reduced memory pool tensor works with different size tensors in pool,
    //  Usually size of output tensor size go down as we go deeper into model.
    //  Some models are exception to this. For those, in case of Reduced memory
    //  pool, some of the layers will use default memory allocation once we
    //  hit the pool limit with ZEN_TENSOR_POOL_LIMIT
    //  Otherwise works with Fixed memory pool tensor, this will works with
    //  finding max from increasing size as we go deeper into model.
    //  In future as a part of cost function max size will be calculated and
    //  used accordingly.
    max_size_enable = zendnn_getenv_int("ZENDNN_TENSOR_BUF_MAXSIZE_ENABLE");

    // Getting max pool limit from env variable
    // If ZENDNN_TENSOR_POOL_LIMIT env variable is not defined, use default
    // value ZEN_TENSOR_POOL_LIMIT
    zenTensorPoolLimit =
        zendnn_getenv_int("ZENDNN_TENSOR_POOL_LIMIT", ZEN_TENSOR_POOL_LIMIT);
    zenTensorPoolLimit = (zenTensorPoolLimit <= 0) ? 1 : zenTensorPoolLimit;

    zenTensorPoolArr =
        (zenTensorPool *)malloc(zenTensorPoolLimit * sizeof(zenTensorPool));

    for (int i = 0; i < zenTensorPoolLimit; i++) {
      zenTensorPoolArr[i].zenTensorPtr = NULL;
      zenTensorPoolArr[i].raw_buff = NULL;
      zenTensorPoolArr[i].zenTensorPtrStatus = -1;
      zenTensorPoolArr[i].zenTensorSize = 0;
      zenTensorPoolArr[i].zenType = zenTensorType::QINT8;
    }
  }

  // destroy Memory pool once done with usage
  ~ZenMemoryPool() {
    for (int i = 0; i < zenTensorPoolSize; i++) {
      delete zenTensorPoolArr[i].zenTensorPtr;
    }
    free(zenTensorPoolArr);
  }

 public:
  // zenTensorPoolArr will hold all ptr and state of tensors created
  //  in the pool
  // zenTensorPtrStatus will hold the state of those tensor
  //  Possible states -1(not allocated),
  //                0(allocated and free)
  //                >0(tensor links with other node)
  zenTensorPool *zenTensorPoolArr;

  // No. of allocated tensors inside pool
  unsigned int zenTensorPoolSize;

  // Max limit for active tensors inside pool
  unsigned int zenTensorPoolLimit;

  // zenEnableMemPool hold the type of memory pool
  // Possible states 0(disabled),
  //                 1(mempool at graph level, memory footprint > 0 < 2)
  //                 2(mempool at node level, memory footprint > 0 and 1)
  unsigned int zenEnableMemPool;

  // max_size_enable will allocate all tensor in pool of size equal to size
  //  of the o/p first layer or running max with pool array
  // TODO: calulate max_size as part of cost function during graph analysis
  //  phase and then use it accordingly.
  bool max_size_enable;

  // max shape of allocated tensors in the pool
  TensorShape max_shape;

  // Reset TensorPool after every graph execution
  bool zenTensorPoolReset;

  // Get Memory pool pointer from Global array of memory pool based on index
  // Create ZenMemPool object, if not created corresponding to that index
  static ZenMemoryPool *getZenMemPool(int index) {
    mtx.lock();
    // ZEN_MEM_POOL_LIMIT is the hard limit on the total no. of ZenMemoryPool
    // TODO: Need to tune ZEN_MEM_POOL_LIMIT based on the available memory or
    // make it grow dynamically
    if (index >= ZEN_MEM_POOL_LIMIT) {
      mtx.unlock();
      return NULL;
    }
    if (!zenMemPoolArr[index]) {
      zenMemPoolArr[index] = new ZenMemoryPool();
      zenMemPoolCount++;
    }
    mtx.unlock();
    return zenMemPoolArr[index];
  }

  // Free zenMemPoolArr based on index passed
  static void freeZenMemPool(int index) {
    mtx.lock();
    if (index >= ZEN_MEM_POOL_LIMIT) {
      mtx.unlock();
      return;
    }
    if (zenMemPoolArr[index]) {
      delete zenMemPoolArr[index];
      zenMemPoolCount--;
    }
    mtx.unlock();
  }

  // Reset status of all Tensor as free at
  // the start of graph execution.
  void resetPoolStatus() {
    for (int i = 0; i < zenTensorPoolSize; i++) {
      zenTensorPoolArr[i].zenTensorPtrStatus = 0;
    }
  }

  unsigned long get_tensor_size(TensorShape &shape) {
    unsigned long size = 1;
    int num_dimensions = shape.dims();
    for (int i = 0; i < num_dimensions; i++) {
      size *= shape.dim_size(i);
    }
    return size;
  }

  // Acquire Tensor buffer from the given pool object. If pool is not
  //  initialized or buffer is not free, create PERSISTENT tensor and
  //  add to the pool.
  int acquireZenPoolTensor(OpKernelContext *context, Tensor **output,
                           TensorShape out_shape, int outlinks, bool reset,
                           zenTensorType type, int out_index = 0) {
    if (reset && zenTensorPoolSize) {
      zenTensorPoolReset = true;
    }

    /*
    //TODO: Compute max_size as part of cost function during
    //  graph analysis phase
    //Once we supoort above cost function, do early return based on
    //  below condition
    if (max_size_enable) {
        //Fall back to default tensor allocation if required,
        //when out_shape is more than max_shape of pool.
        unsigned long out_size = get_tensor_size(out_shape);
        unsigned long max_size = get_tensor_size(max_shape);
        if (out_size > max_size) {
            zendnnInfo(ZENDNN_FWKLOG,
                       "\nTF-MEM-POOL: Requested Tensor from ZenMemPool, But
    Falling back to default allocation as out_size(", out_size, ") > max_size(",
    max_size, ")of Pool\n"); return 1;
        }
    }
    */
    int acquire_flag = 0;
    int free_flag = 0;

    // Search for free tensor in pool based on zenEnableMemPool,
    // tensor_size and out_size
    for (int i = 0; i < zenTensorPoolSize; i++) {
      if (zenTensorPoolArr[i].zenTensorPtrStatus == 0) {
        free_flag = 1;

        // Go to next free tensor when out_size is more
        //  than tensor_size of pool at given offset
        //  or if type of the node doesn't match.
        unsigned long out_size = get_tensor_size(out_shape);
        unsigned long tensor_size = zenTensorPoolArr[i].zenTensorSize;
        if (out_size > tensor_size ||
            (zenEnableMemPool == 2 && out_size < tensor_size) ||
            (zenTensorPoolArr[i].zenType != type)) {
          continue;
        }
        *output = (zenTensorPoolArr[i].zenTensorPtr);
        if (type == zenTensorType::QINT8) {
          zenTensorPoolArr[i].zenType = type;
          zenTensorPoolArr[i].raw_buff =
              static_cast<qint8 *>((*output)->template flat<qint8>().data());
          zenTensorPoolArr[i].zenTensorPtrStatus = outlinks;
        } else if (type == zenTensorType::QUINT8) {
          zenTensorPoolArr[i].zenType = type;
          zenTensorPoolArr[i].raw_buff =
              static_cast<quint8 *>((*output)->template flat<quint8>().data());
          zenTensorPoolArr[i].zenTensorPtrStatus = outlinks;
        } else if (type == zenTensorType::FLOAT) {
          zenTensorPoolArr[i].zenType = type;
          zenTensorPoolArr[i].raw_buff =
              static_cast<float *>((*output)->template flat<float>().data());
          zenTensorPoolArr[i].zenTensorPtrStatus = outlinks;
        }

        (*output)->set_shape(out_shape);
        if (out_index >= 0) {
          context->set_output(out_index, **output);
        }
        acquire_flag = 1;
        zendnnInfo(ZENDNN_FWKLOG, "\nTF-MEM-POOL: Acquired TensorPool Ptr[", i,
                   "] pointed to size(no. of elements)", tensor_size, "\n");

        break;
      }
    }

    // If requested tensor not found in pool, go ahead and create
    //  new tensor inside pool.
    if (!acquire_flag) {
      if (zenTensorPoolSize == zenTensorPoolLimit) {
        if (free_flag) {
          zendnnInfo(ZENDNN_FWKLOG,
                     "\nTF-MEM-POOL: Requested Tensor from ZenMemPool, But "
                     "Falling back to default allocation as out_size > "
                     "available tensor_size inside Pool\n");
        } else {
          zendnnInfo(ZENDNN_FWKLOG,
                     "\nTF-MEM-POOL: Requested Tensor from ZenMemPool, But "
                     "Falling back to default allocation as zenTensorPoolSize "
                     "== ZEN_TENSOR_POOL_LIMIT\n");
        }
        return 1;
      }

      unsigned int poolOffset = zenTensorPoolSize;
      TensorShape shape;

      // Set max_shape based on current layer output dimension
      // and ZEN_TENSOR_SIZE_FACTOR, Most of the cases Output
      // dimension goes down after first layer. However few
      // models are exception to this.
      // max_size required can be computed during first run
      // for graph execution and same can be used for allocation. But
      // this will not give optimal performance for first graph execution.
      // TODO: Compute max_size as part of cost function during
      // graph analysis phase
      unsigned long out_size = get_tensor_size(out_shape);
      unsigned long max_size = get_tensor_size(max_shape);
      if (out_size > max_size) {
        max_shape = out_shape;
      }

      // max_size_enable will create all tensor with increasing size
      // inside the pool
      if (max_size_enable) {
        unsigned long max_size = get_tensor_size(max_shape);
        zenTensorPoolArr[poolOffset].zenTensorSize = max_size;
        shape = max_shape;
      } else {
        unsigned long size = get_tensor_size(out_shape);
        zenTensorPoolArr[poolOffset].zenTensorSize = size;
        shape = out_shape;
      }

      zenTensorPoolArr[poolOffset].zenTensorPtr = new Tensor();
      if (type == zenTensorType::QINT8) {
        zenTensorPoolArr[poolOffset].zenType = type;
        context->allocate_temp(DT_QINT8, shape,
                               zenTensorPoolArr[poolOffset].zenTensorPtr);

        *output = (zenTensorPoolArr[poolOffset].zenTensorPtr);
        zenTensorPoolArr[poolOffset].raw_buff =
            static_cast<qint8 *>((*output)->template flat<qint8>().data());

        zenTensorPoolArr[poolOffset].zenTensorPtrStatus = outlinks;
      } else if (type == zenTensorType::QUINT8) {
        zenTensorPoolArr[poolOffset].zenType = type;
        context->allocate_temp(DT_QUINT8, shape,
                               zenTensorPoolArr[poolOffset].zenTensorPtr);

        *output = (zenTensorPoolArr[poolOffset].zenTensorPtr);
        zenTensorPoolArr[poolOffset].raw_buff =
            static_cast<quint8 *>((*output)->template flat<quint8>().data());

        zenTensorPoolArr[poolOffset].zenTensorPtrStatus = outlinks;
      } else if (type == zenTensorType::FLOAT) {
        zenTensorPoolArr[poolOffset].zenType = type;
        context->allocate_temp(DT_FLOAT, shape,
                               zenTensorPoolArr[poolOffset].zenTensorPtr);

        *output = (zenTensorPoolArr[poolOffset].zenTensorPtr);
        zenTensorPoolArr[poolOffset].raw_buff =
            static_cast<float *>((*output)->template flat<float>().data());

        zenTensorPoolArr[poolOffset].zenTensorPtrStatus = outlinks;
      }
      (*output)->set_shape(out_shape);
      if (out_index >= 0) {
        context->set_output(out_index, **output);
      }
      acquire_flag = 1;
      zenTensorPoolSize++;
      zendnnInfo(ZENDNN_FWKLOG,
                 "\nTF-MEM-POOL: Allocation done for Tensor in Pool of size = ",
                 (*output)->TotalBytes() / sizeof(float), " elements",
                 " zenTensorPoolCount = ", zenTensorPoolSize - 1, "\n",
                 "TF-MEM-POOL: Acquired TensorPool Ptr[", poolOffset,
                 "] pointed to size(no. of elements)",
                 (*output)->TotalBytes() / sizeof(float), "\n");
    }
    return 0;
  }

  // This will update the state of Memory pool by decrementing
  // zenTensorPoolArrStatus based on the input buffer comparison.
  void zenMemPoolFree(OpKernelContext *context, void *input) {
    if (zenEnableMemPool == 1) {
      // This block optimize the buffer reuse across mempool(each inter op
      // has its own pool memory).
      // Currently this has some performance issues.
      // TODO: Fix this
      mtx.lock();
      for (int i = 0; i < zenMemPoolCount; i++) {
        if (zenMemPoolArr[i]) {
          for (int j = 0; j < zenMemPoolArr[i]->zenTensorPoolSize; j++) {
            void *output_array = zenMemPoolArr[i]->zenTensorPoolArr[j].raw_buff;
            if (input == output_array) {
              zenMemPoolArr[i]->zenTensorPoolArr[j].zenTensorPtrStatus--;
              break;
            }
          }
        }
      }
      mtx.unlock();
    }

    // This will be enabled, when we reset pool after last zen node execution
    if (zenTensorPoolReset) {
      resetPoolStatus();
      zenTensorPoolReset = false;
      graph_exe_count++;
    }
  }

  // Method to update the 'use status' of buffer from tensor pool. Basically
  // it resets the 'use status' with the status value received as argument.
  // Currently this method is used in convolution fused sum optimization
  // where the input buffer is re-used as output buffer.
  void zenMemPoolUpdateTensorPtrStatus(OpKernelContext *context, void *input,
                                       int status, bool reset) {
    if (zenEnableMemPool == 1) {
      // This block optimize the buffer reuse across mempool(each inter op
      // has its own pool memory).
      // Currently this has some performance issues.
      // TODO: Fix this
      mtx.lock();
      for (int i = 0; i < zenMemPoolCount; i++) {
        if (zenMemPoolArr[i]) {
          for (int j = 0; j < zenMemPoolArr[i]->zenTensorPoolSize; j++) {
            void *output_array = zenMemPoolArr[i]->zenTensorPoolArr[j].raw_buff;
            if (input == output_array) {
              zenMemPoolArr[i]->zenTensorPoolArr[j].zenTensorPtrStatus = status;
              break;
            }
          }
        }
      }
      mtx.unlock();
    }
    // This will be enabled, when we reset pool after last zen node execution
    if (reset) {
      resetPoolStatus();
      zenTensorPoolReset = false;
      graph_exe_count++;
    }
  }
};

}  // namespace tensorflow

#endif  // AMD_ZENDNN

#endif
