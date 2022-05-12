/*!
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * \file: /lego.cc
 * \brief:
 * Author: raphael hao
 */

// Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Ns_e_ither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "lego_model.h"
#include "lego_model_instance.h"
#include "lego_schedule.h"
#include "lego_utils.h"
#include "loader.h"
#include "logging.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/common/nvtx.h"

#include <NvInferPlugin.h>
#include <atomic>
#include <chrono>
#include <cuda_runtime_api.h>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <set>
#include <thread>
#include <unordered_map>

//
// Lego backend that implements the TRITONBACKEND API.
//
namespace triton { namespace backend { namespace lego {

namespace {
#define CUDA_CALL(CALL)                                                    \
  do {                                                                     \
    cudaError_t farie_err_ = (CALL);                                       \
    if (farie_err_ != cudaSuccess) {                                       \
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, cudaGetErrorString(farie_err_)); \
      return;                                                              \
    }                                                                      \
  } while (false)

#ifdef TRITON_ENABLE_STATS
#define FAIL_ALL_AND_RETURN_IF_ERROR(REQUESTS, REQUEST_COUNT, RESPONSES, S,    \
                                     LOG_MSG)                                  \
  do {                                                                         \
    TRITONSERVER_Error* farie_err_ = (S);                                      \
    if (farie_err_ != nullptr) {                                               \
      for (uint32_t r = 0; r < REQUEST_COUNT; ++r) {                           \
        if (RESPONSES[r] != nullptr) {                                         \
          LOG_IF_ERROR(TRITONBACKEND_ResponseSend(                             \
                           RESPONSES[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                           farie_err_),                                        \
                       "failed to send Lego backend response");                \
          LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (LOG_MSG));                      \
        }                                                                      \
        LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportStatistics(              \
                         TritonModelInstance(), REQUESTS[r],                   \
                         false /* success */, 0, 0, 0, 0),                     \
                     "failed reporting request statistics");                   \
        LOG_IF_ERROR(TRITONBACKEND_RequestRelease(                             \
                         REQUESTS[r], TRITONSERVER_REQUEST_RELEASE_ALL),       \
                     "failed releasing request");                              \
        REQUESTS[r] = nullptr;                                                 \
      }                                                                        \
      TRITONSERVER_ErrorDelete(farie_err_);                                    \
      return;                                                                  \
    }                                                                          \
  } while (false)

// void CUDART_CB TimestampCaptureCallback(void* data) {
//   SET_TIMESTAMP(*(reinterpret_cast<uint64_t*>(data)));
// }

#else
#define FAIL_ALL_AND_RETURN_IF_ERROR(REQUESTS, REQUEST_COUNT, RESPONSES, S,    \
                                     LOG_MSG)                                  \
  do {                                                                         \
    TRITONSERVER_Error* farie_err_ = (S);                                      \
    if (farie_err_ != nullptr) {                                               \
      for (uint32_t r = 0; r < REQUEST_COUNT; ++r) {                           \
        if (RESPONSES[r] != nullptr) {                                         \
          LOG_IF_ERROR(TRITONBACKEND_ResponseSend(                             \
                           RESPONSES[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                           farie_err_),                                        \
                       "failed to send Lego backend response");                \
          LOG_MESSAGE(TRITONSERVER_LOG_ERROR, (LOG_MSG));                      \
        }                                                                      \
        LOG_IF_ERROR(TRITONBACKEND_RequestRelease(                             \
                         REQUESTS[r], TRITONSERVER_REQUEST_RELEASE_ALL),       \
                     "failed releasing request");                              \
        REQUESTS[r] = nullptr;                                                 \
      }                                                                        \
      TRITONSERVER_ErrorDelete(farie_err_);                                    \
      return;                                                                  \
    }                                                                          \
  } while (false)

#endif  // TRITON_ENABLE_STATS

int GetCudaStreamPriority(LegoModel::Priority priority) {
  // Default priority is 0
  int cuda_stream_priority = 0;

  int min, max;
  cudaError_t cuerr = cudaDeviceGetStreamPriorityRange(&min, &max);
  if ((cuerr != cudaErrorNoDevice) && (cuerr != cudaSuccess)) {
    return 0;
  }

  switch (priority) {
    case LegoModel::Priority::MAX:
      cuda_stream_priority = max;
      break;
    case LegoModel::Priority::MIN:
      cuda_stream_priority = min;
      break;
    default:
      cuda_stream_priority = 0;
      break;
  }

  return cuda_stream_priority;
}

TRITONSERVER_Error* CreateCudaEvent(const std::string& event_name,
                                    unsigned int event_flags,
                                    cudaEvent_t* event) {
  // Not adding 'cudaEventBlockingSync' to reduce gaps between the
  // time of event record and the time of signaling blocking thread.
  // The busy waiting only happens when there is inflight request.
  auto cuerr = cudaEventCreateWithFlags(event, event_flags);
  if (cuerr != cudaSuccess) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL,
        (std::string("unable to create CUDA event for ") + event_name + ": " +
         cudaGetErrorString(cuerr))
            .c_str());
  }
  return nullptr;
}
}  // namespace

//
// BackendConfiguration
//
// Struct to hold value specified via backend config
struct BackendConfiguration {
  BackendConfiguration() : coalesce_request_input_(false) {}
  bool coalesce_request_input_;
};

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState : public LegoModel {
 public:
  static TRITONSERVER_Error* Create(TRITONBACKEND_Model* triton_model,
                                    ModelState** state);
  virtual ~ModelState();

  TRITONSERVER_Error* CreateEngine(
      int gpu_device, const size_t stage_idx, const std::string& model_path,
      std::shared_ptr<nvinfer1::ICudaEngine>* engine);

  void DisableEngineSharing() { stages_engines_sharing_ = false; }
  bool IsEngineSharingEnabled() { return stages_engines_sharing_; }

 private:
  ModelState(TRITONBACKEND_Model* triton_model);

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  // CUDA engine shared across all model instances using the same GPU. The key
  // is the GPU ID.
  std::map<int, std::unordered_map<
                    int, std::pair<std::shared_ptr<nvinfer1::IRuntime>,
                                   std::shared_ptr<nvinfer1::ICudaEngine>>>>
      device_stages_stages_engines_;
  bool stages_engines_sharing_;
};

TRITONSERVER_Error* ModelState::Create(TRITONBACKEND_Model* triton_model,
                                       ModelState** state) {
  try {
    *state = new ModelState(triton_model);
  } catch (const BackendModelException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  RETURN_IF_ERROR((*state)->ValidateModelConfig());

  return nullptr;  // success
}
ModelState::ModelState(TRITONBACKEND_Model* triton_model)
    : LegoModel(triton_model), stages_engines_sharing_(true) {
  // Obtain backend configuration
  TRITONBACKEND_Backend* backend;
  THROW_IF_BACKEND_MODEL_ERROR(
      TRITONBACKEND_ModelBackend(triton_model, &backend));
  // void* vstate;
  // THROW_IF_BACKEND_MODEL_ERROR(TRITONBACKEND_BackendState(backend,
  // &vstate)); backend_config_ =
  // reinterpret_cast<BackendConfiguration*>(vstate);
}

ModelState::~ModelState() {
  for (auto& device_stage_engines : device_stages_stages_engines_) {
    cudaSetDevice(device_stage_engines.first);
    for (auto& stage_engine : device_stage_engines.second) {
      auto& runtime = stage_engine.second.first;
      auto& engine = stage_engine.second.second;
      // Need to reset explicitly to ensure proper destruction order
      if (engine != nullptr) {
        engine.reset();
      }
      if (runtime != nullptr) {
        runtime.reset();
      }
    }
  }
}

TRITONSERVER_Error* ModelState::CreateEngine(
    int gpu_device, const size_t stage_idx, const std::string& model_path,
    std::shared_ptr<nvinfer1::ICudaEngine>* engine) {
  // Lego engine creation is not thread-safe, so multiple creations
  // are serialized with a global lock.
  dbg("creating engine for stage: ", stage_idx);
  static std::mutex global_context_mu;
  std::lock_guard<std::mutex> glock(global_context_mu);

  // Create shared stage engines for the device if haven't tried so.
  auto s_e_it = device_stages_stages_engines_.find(gpu_device);
  if (s_e_it == device_stages_stages_engines_.end()) {
    s_e_it =
        device_stages_stages_engines_
            .emplace(
                gpu_device,
                std::unordered_map<
                    int, std::pair<std::shared_ptr<nvinfer1::IRuntime>,
                                   std::shared_ptr<nvinfer1::ICudaEngine>>>())
            .first;
  }
  // Check if the engine has been created for the stage.
  auto e_it = s_e_it->second.find(stage_idx);
  if (e_it == s_e_it->second.end()) {
    e_it = s_e_it->second.emplace(stage_idx, std::make_pair(nullptr, nullptr))
               .first;
  }

  // We share the engine (for models that don't have dynamic shapes) and
  // runtime across instances that have access to the same GPU.
  if (e_it->second.second == nullptr) {
    auto cuerr = cudaSetDevice(gpu_device);
    if (cuerr != cudaSuccess) {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                   (std::string("unable to set device for ") +
                                    Name() + ": " + cudaGetErrorString(cuerr))
                                       .c_str());
    }

    const bool new_runtime = (e_it->second.first == nullptr);
    RETURN_IF_ERROR(
        LoadPlan(model_path, &e_it->second.first, &e_it->second.second));
    *engine = e_it->second.second;

    if (new_runtime) {
      LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                  (std::string("Created new runtime on GPU device ") +
                   std::to_string(gpu_device) + " for " + Name())
                      .c_str());
    }
    LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                (std::string("Created new engine on GPU device ") +
                 std::to_string(gpu_device) + " for " + Name())
                    .c_str());

    if (IsEngineSharingEnabled()) {
      // This logic runs atleast once to validate whether the engine
      // can be shared.
      bool is_dynamic = false;
      for (int idx = 0; idx < e_it->second.second->getNbBindings(); idx++) {
        auto dims = e_it->second.second->getBindingDimensions(idx);
        // Detect whether dynamic or not
        if (ContainsWildcard(dims)) {
          is_dynamic = true;
          break;
        }
      }
      if (is_dynamic) {
        // Model with dynamic shapes can't share engine
        DisableEngineSharing();
      }
    }

    if (!IsEngineSharingEnabled()) {
      // Set to engine to 'nullptr' as hint, but keeping runtime as it
      // can be used repeatedly
      if (e_it->second.second != nullptr) {
        e_it->second.second.reset();
      }
    }
  } else {
    *engine = e_it->second.second;
  }

  return nullptr;
}

TRITONSERVER_Error* ModelState::ValidateModelConfig() {
  // We have the json DOM for the model configuration...
  triton::common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  return nullptr;  // success
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState : public LegoModelInstance {
 public:
  // GPU device number that indicates that no gpu is available for a
  // context (which is an invalid state since Lego requires a
  // GPU).
  static constexpr int NO_GPU_DEVICE = -1;

  // GPU device number that indicates model will be loaded on GPUs
  // as specified in model graph
  static constexpr int MODEL_DEVICE = -2;

  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state);
  virtual ~ModelInstanceState();

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  std::shared_ptr<nvinfer1::ICudaEngine>* EnginePtr(const size_t stage_idx) {
    return &stage_engines_[stage_idx];
  }
  std::shared_ptr<nvinfer1::ICudaEngine> Engine(const size_t stage_idx) {
    return stage_engines_[stage_idx];
  }

  void ProcessRequests(TRITONBACKEND_Request** requests,
                       const uint32_t request_count);

  void Run(TRITONBACKEND_Request** requests, const uint32_t request_count,
           size_t& context_idx);

 private:
  struct TensorRTContext;

  ModelInstanceState(ModelState* model_state,
                     TRITONBACKEND_ModelInstance* triton_model_instance);

  void RegisterContexts();
  TRITONSERVER_Error* InitStreamsAndEvents();
  TRITONSERVER_Error* InitEventSet(bool busy_wait_events);
  TRITONSERVER_Error* DestroyEventSet();
  TRITONSERVER_Error* InitOptimizationProfiles(const size_t stage_idx);

  TRITONSERVER_Error* ValidateIO(const size_t stage_idx);
  TRITONSERVER_Error* ValidateIOHelper(common::TritonJson::Value& ios,
                                       const bool if_stage,
                                       const bool is_input);

  TRITONSERVER_Error* InitIOBindingBuffers(const size_t stage_idx);
  TRITONSERVER_Error* InitializeConfigStageExecuteInputBindings(
      const size_t stage_idx,
      std::unordered_map<std::string, StageInOut> config_stage_inputs);
  TRITONSERVER_Error* InitializeConfigStageExecuteOutputBindings(
      const size_t stage_idx,
      std::unordered_map<std::string, StageInOut> config_stage_outputs);
  TRITONSERVER_Error* InitializeStageExecuteInputBinding(
      const size_t stage_idx, const std::string& input_name,
      const std::string& input_datatype, std::vector<int64_t>& input_dims,
      size_t& buffer_id, const bool is_control = false);

  void Warmup() {
    // warm up
    for (size_t warmup_idx = 0; warmup_idx < 10; ++warmup_idx) {
      for (size_t stage_idx = 0; stage_idx < total_stages_; ++stage_idx) {
        for (auto& trt_context : stages_trt_contexts_[stage_idx]) {
          trt_context.second.context_->enqueueV2(
              stages_buffer_bindings_[stage_idx][0].data(), nullptr, nullptr);
        }
      }
    }
    CUDA_CALL(cudaDeviceSynchronize());
    return;
  }

  TRITONSERVER_Error* GetProfileDimensions(const size_t stage_idx,
                                           const int io_index,
                                           const int profile_index,
                                           TensorRTContext* context);

  // TRITONSERVER_Error* SetBindingDimensions(const size_t stage_idx,
  //                                          const std::string& input_name,
  //                                          const std::vector<int64_t>& shape,
  //                                          const TensorRTContext&
  //                                          trt_context, const size_t
  //                                          io_index, const size_t
  //                                          binding_index,
  //                                          std::vector<int64_t>* input_dims);

  // TRITONSERVER_Error* GetMostOptimizedProfile(
  //     size_t stage_idx, size_t total_batch_size,
  //     TRITONBACKEND_Request** requests, uint32_t request_count,
  //     std::map<int, TensorRTContext>::iterator* citr);
  // TRITONSERVER_Error* EvaluateTensorRTContext(
  //     size_t stage_idx, std::map<int, TensorRTContext>::iterator& citr,
  //     size_t total_batch_size, TRITONBACKEND_Request** requests,
  //     uint32_t request_count, int64_t* error_distance);

  void ProcessResponse();
  void PreprocessBatch(const size_t& payload_matrix_id,
                       const size_t& batch_size, const size_t& request_id = 0);
  void PostprocessBatch(const size_t& stage_idx, const size_t& matrix_idx,
                        const size_t& vector_idx);
  void StageRun(size_t stage_idx);

  void FlushStatics(std::initializer_list<std::string> request_id_latency) {
    for (const auto& statics : request_id_latency) {
      log_ofs_ << statics << ",";
    }
    log_ofs_ << std::endl;
  }

  void GetConfiguredProfiles(size_t stage_idx, std::string* profiles_desc);
  int CudaStreamPriority() { return cuda_stream_priority_; }
  size_t GetTotalStages() { return total_stages_; };
  int64_t GetBufferSetSize() { return buffer_set_size_; };

  // A struct to hold TensorRT execution context and its meta data, a
  // backend context can have multiple of this struct if multiple
  // optimization profiles is specified.
  struct TensorRTContext {
    TensorRTContext(const std::string& profile_name, const int profile_idx,
                    const int binding_cnts)
        : profile_name_(profile_name),
          profile_idx_(profile_idx),
          context_(nullptr),
          min_dims_(binding_cnts),
          max_dims_(binding_cnts),
          opt_dims_(binding_cnts),
          min_shapes_(binding_cnts),
          max_shapes_(binding_cnts),
          opt_shapes_(binding_cnts),
          is_dynamic_per_binding_(binding_cnts) {}
    std::string profile_name_;
    int profile_idx_;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;

    // Min Dimensions per bindings
    std::vector<nvinfer1::Dims> min_dims_;

    // Max Dimensions per bindings
    std::vector<nvinfer1::Dims> max_dims_;

    // Optimized Dimensions per bindings
    std::vector<nvinfer1::Dims> opt_dims_;

    // Min shape values per bindings
    std::vector<const int32_t*> min_shapes_;

    // Max shape values per bindings
    std::vector<const int32_t*> max_shapes_;

    // Optimized shape values per bindings
    std::vector<const int32_t*> opt_shapes_;

    // The number of shape values
    size_t nb_shape_values_;

    // Whether or not the binding contains a dynamic shape
    std::vector<bool> is_dynamic_per_binding_;
  };

  size_t total_stages_;

  size_t last_stage_idx_;

  int64_t buffer_set_size_;

  size_t max_batch_size_;

  size_t batch_dim_idx_;

  size_t total_alive_batch_size_;

  size_t split_stage_idx_;

  // The engine used for the instance. If the model uses dynamic
  // shape, then the CUDA engine is owned by the instance. Otherwise,
  // the engine is shared across all contexts and it must not be
  // destroyed by the instance. In the future version of TensorRT, the
  // engine may be shared even in the dynamic shape case.
  //[stage] -> engine
  std::vector<std::shared_ptr<nvinfer1::ICudaEngine>> stage_engines_;

  // Map from profile index to the corresponding TensorRT context. Use
  // map to ensure each profile index is mapped to exactly one
  // TensorRT context.
  //[stage, profile_idx] -> TensorRTContext
  std::vector<std::map<int, TensorRTContext>> stages_trt_contexts_;

  // The total number of bindings
  std::vector<int> total_bindings_;

  // The number of expected bindings to the model. In case of dynamic
  // shapes, it is the number of expected bindings to the configured
  // optimization profile.
  std::vector<int> num_expected_bindings_;

  int cuda_stream_priority_;

  // Additional CUDA streams to overlap copy and execution.
  std::vector<cudaStream_t> stage_execution_streams_;
  cudaStream_t input_copy_stream_;

  // A group of CUDA events that signals different stages of the
  // request. One group should be used for one request at any given
  // moment.
  struct CUDAEventSet {
    // CUDA event to signal input buffer availability.
    cudaEvent_t ready_for_input_;
    cudaEvent_t input_ready_;

    // CUDA event for capturing correct timestamp.
    cudaEvent_t ready_for_output_;
    cudaEvent_t output_ready_;

    // CUDA event for synchronizing the order of timestamp capture.
    cudaEvent_t timestamp_signal_;
  };

  // Use two sets of events each for current request and next request.
  std::vector<std::vector<CUDAEventSet>> stage_events_;

  size_t next_event_set_;

  // Completion thread for handling items in the corresponding
  // completion queue. One thread per instance so that the thread
  // logic is simple as this avoids busy-looping on different model
  // executions' event states.
  std::thread completion_thread_;

  std::vector<std::thread> stage_threads_;

  int next_context_idx_;
  std::vector<int> next_stage_contexts_idx_;

  // Assume that the lifetime of composing completion data to extend
  // till the responses are returned.
  triton::common::SyncQueue<std::unique_ptr<Payload>> completion_queue_;

  std::vector<StageQueue> stage_queues_;

  // The maximum possible size of the TensorRT tensor and the
  // corresponding allocated GPU buffer across all optimization
  // profile.
  struct IOBindingInfo {
    IOBindingInfo()
        : byte_size_(0),
          buffer_(nullptr),
          buffer_id_(0),
          device_buffer_(nullptr),
          memory_type_(TRITONSERVER_MEMORY_GPU),
          memory_type_id_(0),
          is_linear_format_(true),
          vectorized_dim_(-1),
          components_per_element_(1) {}
    uint64_t byte_size_;
    void* buffer_;
    size_t buffer_id_;
    void* device_buffer_;
    TRITONSERVER_MemoryType memory_type_;
    int64_t memory_type_id_;
    bool is_linear_format_;
    int vectorized_dim_;
    int components_per_element_;
  };

  // There will be two sets of input/output buffers when
  // separate_output_stream is selected to overlap copy and execution
  // safely.
  size_t next_buffer_binding_set_;

  // There are Context::num_expected_bindings_ number of IOBindingInfo
  // stages_io_binding_infos_;
  // [stage,infoset,binding] -> IOBindingInfo
  std::vector<std::vector<std::vector<IOBindingInfo>>> stages_io_binding_infos_;

  // The pointer to the CUDA buffer for each binding index of the
  // TensorRT engine. This is used to match the TensorRT context
  // execution declaration while minimizing memory allocation. The
  // array size is equal to Context::total_bindings_ One of for each
  // copy stream
  // [stage,infoset,binding] -> cuda pointer
  std::vector<std::vector<std::vector<void*>>> stages_buffer_bindings_;

  // The pointer to the CUDA buffer for the buffers of all stages
  // [buffer_id] -> (cuda pointer, size)
  std::vector<std::map<size_t, std::pair<void*, int64_t>>> all_buffers_;

  // The request details of the ongoing model execution
  std::array<PayloadVector, PAYLOAD_MATRIX_SIZE> payload_matrix_;

  size_t next_payload_matrix_idx_;

  // Whether the input collector will coalesce request inputs as if they
  // form one contiguous buffer when possible
  bool coalesce_request_input_;

  ModelState* model_state_;

  Scheduler* scheduler_;

  std::ofstream log_ofs_;
};

TRITONSERVER_Error* ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state) {
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  } catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }
  auto schedule_type = (*state)->Model()->GetSchedulerType();
  (*state)->batch_dim_idx_ = 0;
  if (schedule_type == LegoModel::SchedulerType::NORMAL) {
    (*state)->scheduler_ = new NomalScheduler(
        (*state)->GetTotalStages(), (*state)->GetBufferSetSize(),
        (*state)->Model()->MaxBatchSize());
  } else if (schedule_type == LegoModel::SchedulerType::NORMAL_INPUT) {
    (*state)->scheduler_ = new NomalInputScheduler(
        (*state)->GetTotalStages(), (*state)->GetBufferSetSize(),
        (*state)->Model()->MaxBatchSize());
    // (*state)->batch_dim_idx_ = 1;
  } else if (schedule_type == LegoModel::SchedulerType::LOAD_DVA) {
    (*state)->scheduler_ = new LoadDVAScheduler(
        (*state)->GetTotalStages(), (*state)->GetBufferSetSize(),
        (*state)->Model()->MaxAllowMergeBatchsize(),
        (*state)->Model()->MaxAllowMergeMicroseconds(),
        (*state)->Model()->MaxAllowMergeStage());
  } else if (schedule_type == LegoModel::SchedulerType::INPUT_DVA) {
    (*state)->scheduler_ = new InputDVAScheduler(
        (*state)->GetTotalStages(), (*state)->GetBufferSetSize(),
        (*state)->Model()->MaxBatchSize(),
        (*state)->Model()->PreferredSeqLen());
    // (*state)->batch_dim_idx_ = 1;
  } else if (schedule_type == LegoModel::SchedulerType::OPERATOR_DVA) {
    (*state)->scheduler_ = new HypbridDVAScheduler(
        (*state)->GetTotalStages(), (*state)->GetBufferSetSize(),
        (*state)->Model()->MaxBatchSize(), (*state)->Model()->FavorPerStage(),
        (*state)->Model()->MaxAllowMergeMicroseconds(),
        (*state)->Model()->MaxAllowMergeStage());
  } else {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_NOT_FOUND,
                                 "unknow scheduler type");
  }
  // If the model configuration doesn't have an explicit model file
  // specified then use the default name.
  std::vector<std::string> cc_model_stage_filenames =
      (*state)->Model()->StageFilenames();
  if ((*state)->GetTotalStages() != cc_model_stage_filenames.size()) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_NOT_FOUND,
                                 "filenames for model stages are not found");
  }
  (*state)->RegisterContexts();
  RETURN_IF_ERROR((*state)->InitStreamsAndEvents());
  auto total_stages = (*state)->GetTotalStages();
  for (size_t stage_idx = 0; stage_idx < total_stages; stage_idx++) {
    auto model_path = JoinPath({model_state->RepositoryPath(),
                                std::to_string(model_state->Version()),
                                cc_model_stage_filenames[stage_idx]});
    {
      bool exists;
      RETURN_IF_ERROR(FileExists(model_path, &exists));
      RETURN_ERROR_IF_FALSE(exists, TRITONSERVER_ERROR_UNAVAILABLE,
                            std::string("unable to find '") + model_path +
                                "' for model instance '" + (*state)->Name() +
                                "'");
    }
    RETURN_IF_ERROR(model_state->CreateEngine((*state)->DeviceId(), stage_idx,
                                              model_path,
                                              (*state)->EnginePtr(stage_idx)));
    RETURN_IF_ERROR((*state)->InitOptimizationProfiles(stage_idx));
    RETURN_IF_ERROR((*state)->ValidateIO(stage_idx));
    RETURN_IF_ERROR((*state)->InitIOBindingBuffers(stage_idx));

    std::string profiles_desc;
    (*state)->GetConfiguredProfiles(stage_idx, &profiles_desc);
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("Created instance ") + (*state)->Name() + " on GPU " +
         std::to_string((*state)->DeviceId()) + " with stream priority " +
         std::to_string((*state)->CudaStreamPriority()) +
         " and optimization profile" + profiles_desc)
            .c_str());
  }
  (*state)->Warmup();
  (*state)->completion_thread_ =
      std::thread(&ModelInstanceState::ProcessResponse, *state);
  for (size_t stage_idx = 0; stage_idx < total_stages; stage_idx++) {
    (*state)->stage_threads_.emplace_back(
        std::thread(&ModelInstanceState::StageRun, *state, stage_idx));
  }
  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance)
    : LegoModelInstance(model_state, triton_model_instance),
      batch_dim_idx_(0),
      model_state_(model_state) {
  // 'coalesce_request_input_' is set at backend level
  {
    TRITONBACKEND_Model* model;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_ModelInstanceModel(triton_model_instance, &model));
    TRITONBACKEND_Backend* backend;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_ModelBackend(model, &backend));
    void* state;
    THROW_IF_BACKEND_INSTANCE_ERROR(
        TRITONBACKEND_BackendState(backend, &state));
    coalesce_request_input_ =
        reinterpret_cast<BackendConfiguration*>(state)->coalesce_request_input_;
  }
  if (Kind() != TRITONSERVER_INSTANCEGROUPKIND_GPU) {
    throw triton::backend::BackendModelInstanceException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unable to load model '") + model_state_->Name() +
         "', Lego backend supports only GPU device")
            .c_str()));
  }
  total_stages_ = Model()->TotalStages();
  buffer_set_size_ = Model()->BufferSetSize();
  last_stage_idx_ = total_stages_ - 1;
  max_batch_size_ = Model()->MaxBatchSize();
  log_ofs_ =
      std::ofstream(Model()->LogFilePath(), std::ios::out | std::ios::trunc);
  total_alive_batch_size_ = 0;
  stage_execution_streams_.resize(total_stages_, nullptr);
  // schedule streams for main thread
  input_copy_stream_ = nullptr;
  // NOTE contorl the binding buffer used for payload
  next_buffer_binding_set_ = 0;
  total_bindings_.resize(total_stages_, 0);
  num_expected_bindings_.resize(total_stages_, 0);
  stage_engines_.resize(total_stages_, nullptr);
  stages_trt_contexts_.resize(total_stages_);
  stages_io_binding_infos_.resize(
      total_stages_, std::vector<std::vector<IOBindingInfo>>(buffer_set_size_));
  stages_buffer_bindings_.resize(
      total_stages_, std::vector<std::vector<void*>>(buffer_set_size_));
  all_buffers_.resize(buffer_set_size_);
  // NOTE control the cuda event set used for the payload
  next_event_set_ = 0;
  stage_events_.resize(total_stages_ + 1,
                       std::vector<CUDAEventSet>(buffer_set_size_));
  stage_queues_ = std::vector<StageQueue>(total_stages_ + 1);
  for (size_t stage_idx = 0; stage_idx <= total_stages_; stage_idx++) {
    for (int idx = 0; idx < buffer_set_size_; idx++) {
      stage_events_[stage_idx][idx].input_ready_ = nullptr;
      stage_events_[stage_idx][idx].ready_for_input_ = nullptr;
      stage_events_[stage_idx][idx].output_ready_ = nullptr;
      stage_events_[stage_idx][idx].ready_for_output_ = nullptr;
      stage_events_[stage_idx][idx].timestamp_signal_ = nullptr;
    }
    // stage_queues_.emplace_back(StageQueue());
    stage_queues_[stage_idx].SetID(stage_idx);
  }
  // NOTE control the current available payload
  next_payload_matrix_idx_ = 0;
  for (size_t matrix_idx = 0; matrix_idx < PAYLOAD_MATRIX_SIZE; matrix_idx++) {
    payload_matrix_[matrix_idx].Init(matrix_idx, total_stages_);
  }
}

ModelInstanceState::~ModelInstanceState() {
  cudaSetDevice(DeviceId());
  for (auto& stage_io_binding_infos : stages_io_binding_infos_) {
    for (auto& io_binding_infos : stage_io_binding_infos) {
      for (auto& io_binding_info : io_binding_infos) {
        auto& buffer =
            all_buffers_[next_buffer_binding_set_][io_binding_info.buffer_id_];
        if (buffer.first != nullptr) {
          cudaError_t err = cudaSuccess;
          if (io_binding_info.memory_type_ == TRITONSERVER_MEMORY_GPU) {
            err = cudaFree(buffer.first);
          } else {
            err = cudaFreeHost(buffer.first);
          }
          buffer.first = nullptr;
          if (err != cudaSuccess) {
            LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                        (std::string("Failed to free allocated memory for '") +
                         Name() + "': " + cudaGetErrorString(err))
                            .c_str());
          }
        }
      }
    }
    next_buffer_binding_set_ =
        (next_buffer_binding_set_ + 1) % buffer_set_size_;
  }

  if (stream_ != nullptr) {
    cudaError_t err = cudaStreamDestroy(stream_);
    if (err != cudaSuccess) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                  (std::string("Failed to destroy cuda stream: ") +
                   +cudaGetErrorString(err))
                      .c_str());
    }
    stream_ = nullptr;
  }

  if (input_copy_stream_ != nullptr) {
    cudaError_t err = cudaStreamDestroy(input_copy_stream_);
    if (err != cudaSuccess) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                  (std::string("Failed to destroy cuda input copy stream: ") +
                   +cudaGetErrorString(err))
                      .c_str());
    }
    input_copy_stream_ = nullptr;
  }

  DestroyEventSet();

  // Notify the stage threads and completion thread to exit
  for (size_t stage_idx = 0; stage_idx < total_stages_; stage_idx++) {
    stage_queues_[stage_idx].Put(std::move(std::unique_ptr<Batch>()));
    if (stage_threads_[stage_idx].joinable()) {
      stage_threads_[stage_idx].join();
    }
  }
  completion_queue_.Put(std::move(std::unique_ptr<Payload>()));
  if (completion_thread_.joinable()) {
    completion_thread_.join();
  }
}

void ModelInstanceState::PreprocessBatch(const size_t& payload_matrix_id,
                                         const size_t& batch_size,
                                         const size_t& request_id) {
  auto& payload_vec = payload_matrix_[payload_matrix_id];
  auto& payload = payload_vec.payloads_[0];
  payload->responses_.reserve(payload_vec.total_request_count_);
  auto request_offset = payload_vec.preprocessed_request_count_;
  auto request_count = payload_vec.total_request_count_ - request_offset;
  for (size_t request_idx = request_offset;
       request_idx < payload_vec.total_request_count_; request_idx++) {
    TRITONBACKEND_Response* response;
    auto err =
        TRITONBACKEND_ResponseNew(&response, payload->requests_[request_idx]);
    if (err == nullptr) {
      payload->responses_.emplace_back(response);
    } else {
      payload->responses_.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }
  payload->collector_.reset(new BackendInputCollector(
      &payload->requests_[request_offset], request_count,
      &payload->responses_[request_offset], model_state_->TritonMemoryManager(),
      model_state_->EnablePinnedInput(), input_copy_stream_,
      stage_events_[0][next_event_set_].input_ready_, nullptr,
      model_state_->GatherKernelBufferThreshold(), HostPolicyName().c_str(),
      false, coalesce_request_input_));
  for (int io_index = 0; io_index < num_expected_bindings_[0]; io_index++) {
    auto& io_binding_info =
        stages_io_binding_infos_[0][next_buffer_binding_set_][io_index];
    if (!stage_engines_[0]->bindingIsInput(io_index)) {
      continue;
    }
    const std::string& name = stage_engines_[0]->getBindingName(io_index);
    if (!stage_engines_[0]->isExecutionBinding(io_index)) {
      continue;
    }
    TRITONBACKEND_Input* repr_input;
    FAIL_ALL_AND_RETURN_IF_ERROR(
        payload->requests_, request_count, payload->responses_,
        TRITONBACKEND_RequestInput(payload->requests_[request_id], name.c_str(),
                                   &repr_input),
        (std::string("failed to obtain the representative input '") + name +
         "'")
            .c_str());
    TRITONSERVER_DataType datatype;
    const int64_t* shape;
    uint32_t dims_count;
    FAIL_ALL_AND_RETURN_IF_ERROR(
        payload->requests_, payload->request_count_, payload->responses_,
        TRITONBACKEND_InputProperties(repr_input, nullptr, &datatype, &shape,
                                      &dims_count, nullptr, nullptr),
        (std::string("failed to obtain the representative input "
                     "properties for '") +
         name + "'")
            .c_str());

    std::vector<int64_t> batchn_shape;
    batchn_shape.reserve(dims_count);
    for (size_t dim_idx = 0; dim_idx < dims_count; dim_idx++) {
      if (dim_idx == 0) {
        batchn_shape.emplace_back(batch_size);
      } else {
        batchn_shape.emplace_back(shape[dim_idx]);
      }
    }
    size_t total_byte_size = 0;
    total_byte_size = GetByteSize(datatype, batchn_shape);
    payload->collector_->ProcessTensor(
        name.c_str(), static_cast<char*>(io_binding_info.buffer_),
        total_byte_size, io_binding_info.memory_type_,
        io_binding_info.memory_type_id_);
  }
  payload->collector_->Finalize();
}

void ModelInstanceState::PostprocessBatch(const size_t& stage_idx,
                                          const size_t& matrix_idx,
                                          const size_t& vector_idx) {
  auto& payload = payload_matrix_[matrix_idx].payloads_[vector_idx];
  auto& stage_engine = stage_engines_[stage_idx];
  auto& stage_trt_context = stages_trt_contexts_[stage_idx].begin()->second;
  payload->responder_.reset(new BackendOutputResponder(
      payload->requests_, payload->request_count_, &payload->responses_,
      model_state_->MaxBatchSize(), model_state_->TritonMemoryManager(),
      model_state_->EnablePinnedOutput(), stage_execution_streams_[stage_idx],
      stage_events_[stage_idx + 1][payload->event_idx_].output_ready_, false));
  for (int io_index = 0; io_index < num_expected_bindings_[stage_idx];
       io_index++) {
    auto& io_binding_info =
        stages_io_binding_infos_[stage_idx][payload->buffer_binding_idx_]
                                [io_index];
    if (stage_engine->bindingIsInput(io_index)) {
      continue;
    }
    const std::string& name = stage_engine->getBindingName(io_index);
    nvinfer1::Dims dims =
        stage_trt_context.context_->getBindingDimensions(io_index);
    std::vector<int64_t> batchn_shape;
    for (int dim_idx = 0; dim_idx < dims.nbDims; dim_idx++) {
      batchn_shape.emplace_back(dims.d[dim_idx]);
    }
    TRITONSERVER_DataType dt =
        ConvertTrtTypeToDataType(stage_engine->getBindingDataType(io_index));
    size_t batch1_byte_size = GetByteSize(dt, batchn_shape);
    batch1_byte_size = batch1_byte_size / payload->request_count_;
    if (io_binding_info.byte_size_ <
        (batch1_byte_size * payload->request_count_)) {
      FAIL_ALL_AND_RETURN_IF_ERROR(
          payload->requests_, payload->request_count_, payload->responses_,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              (std::string("unexpected size for output '") + name +
               "', byte-size " + std::to_string(io_binding_info.byte_size_) +
               " is less than " + std::to_string(payload->request_count_) +
               " * " + std::to_string(batch1_byte_size))
                  .c_str()),
          "failed to run TRT response");
    }
    payload->responder_->ProcessTensor(
        name, dt, batchn_shape,
        static_cast<const char*>(io_binding_info.buffer_),
        io_binding_info.memory_type_, io_binding_info.memory_type_id_,
        batch_dim_idx_);
  }
}

// current use the event set of stage 0 for the whole model instance state
void ModelInstanceState::ProcessRequests(TRITONBACKEND_Request** requests,
                                         const uint32_t request_count) {
  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
              (std::string("TRITONBACKEND_ModelExecute: Issuing ") + Name() +
               " with " + std::to_string(request_count) + " requests")
                  .c_str());

  NVTX_RANGE(nvtx_, "Run " + Name());

  // Need to move the TRITONBACKEND_Request objects as the lifetime
  // must be extended till ProcessResponse completes.
  size_t total_batch_size = 0;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string("null request given to Lego backend for '" + Name() +
                          "'")
                  .c_str()));
      return;
    }
    total_batch_size += 1;
  }

  if (total_batch_size == 0) {
    return;
  }

  if ((total_batch_size != 1) && (total_batch_size > max_batch_size_)) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string("batch size " + std::to_string(total_batch_size) +
                        " for '" + Name() + "', max allowed is " +
                        std::to_string(max_batch_size_))
                .c_str()));
    return;
  }
  // LEGO_DECL_TIMESTAMP(t_now);
  std::vector<std::unique_ptr<Batch>> batches;
  auto target_context_idx = next_context_idx_;
  size_t max_shape_request_id = 0;
  scheduler_->MainSchedule(next_payload_matrix_idx_, next_event_set_,
                           next_buffer_binding_set_, target_context_idx,
                           max_shape_request_id, payload_matrix_, batches,
                           requests, request_count);
  // FIXME not efficient to do this every time

  if (!batches.empty()) {
    // dbg(batches.size());
    for (auto& batch : batches) {
      if (batch != nullptr) {
        PreprocessBatch(batch->payload_matrix_idx_, batch->bs_,
                        max_shape_request_id);
        batch->used_context_idx_ = next_context_idx_;
        stage_queues_[0].Put(std::move(batch));
        scheduler_->GetMainTargetContext(target_context_idx, next_context_idx_);
      } else {
        PreprocessBatch(next_payload_matrix_idx_, total_batch_size,
                        max_shape_request_id);
        scheduler_->GetMainTargetContext(target_context_idx, next_context_idx_);
      }
    }
  } else {
    // dbg("empty batches after schedule");
  }
  scheduler_->UpdateMainContext(target_context_idx);
}

void ModelInstanceState::StageRun(size_t stage_idx) {
  dbg("Initiliazing stage: ", std::to_string(stage_idx));
  auto& stage_trt_context = stages_trt_contexts_[stage_idx].begin()->second;
  // auto& stage_engine = stage_engines_[stage_idx];
  auto& next_stage_context_idx = next_stage_contexts_idx_[stage_idx];
  // auto& stage_io_binding_infos = stages_io_binding_infos_[stage_idx];
  auto& stage_exe_stream = stage_execution_streams_[stage_idx];

  auto& cur_event = stage_events_[stage_idx];

  auto& pos_event = stage_events_[stage_idx + 1];
  auto& cur_batch_queue = stage_queues_[stage_idx];
  auto& post_batch_queue = stage_queues_[stage_idx + 1];
  size_t matrix_idx, vector_idx;
  int32_t used_context_idx;
  while (true) {
    auto batch = std::move(cur_batch_queue.Get());
    if (batch.get() == nullptr) {
      // This is the end of the queue.
      break;
    }
    matrix_idx = batch->payload_matrix_idx_;
    vector_idx = batch->payload_vector_idx_;
    used_context_idx = batch->used_context_idx_;
    auto& event_idx = payload_matrix_[matrix_idx].event_idx_;
    auto& buffer_binding_idx = payload_matrix_[matrix_idx].buffer_binding_idx_;
    // auto& stage_context_idx = payload_matrix_[matrix_idx].context_idx_;

    std::vector<std::unique_ptr<Batch>> batches;
    auto target_stage_context_idx = next_stage_context_idx;
    // NOTE scheduling position
    scheduler_->StageSchedule(stage_idx, matrix_idx, target_stage_context_idx,
                              total_stages_, payload_matrix_, batches, batch);
    // set the binding dimensions according to the buffer id and store the shape
    // inside the payload
    scheduler_->SetBindingDimensions(stage_idx, batch,
                                     stage_trt_context.context_);
    // NOTE now we do not use the ready for input event.
    // Reasons:
    // 1. only needed for the first stage
    // 2. our batch are using different address
    CUDA_CALL(cudaStreamWaitEvent(stage_exe_stream,
                                  cur_event[event_idx].input_ready_, 0));
    // DECL_TIMESTAMP(star_now);
    stage_trt_context.context_->enqueueV2(
        stages_buffer_bindings_[stage_idx][buffer_binding_idx].data(),
        stage_exe_stream, nullptr);
    CUDA_CALL(
        cudaEventRecord(pos_event[event_idx].input_ready_, stage_exe_stream));
    // CUDA_CALL(cudaEventSynchronize(pos_event[event_idx].input_ready_));
    // DECL_TIMESTAMP(end_now);
    // auto elapsed_time = (end_now - star_now) / 1000000.0;
    // dbg(elapsed_time);
    if (!batches.empty()) {
      // dbg(batches.size());
      if (stage_idx == last_stage_idx_) {
        PostprocessBatch(last_stage_idx_, matrix_idx, vector_idx);
        payload_matrix_[matrix_idx].payloads_[vector_idx]->context_idx_ =
            next_stage_context_idx;
        completion_queue_.Put(
            std::move(payload_matrix_[matrix_idx].payloads_[vector_idx]));
        scheduler_->GetStageTargetContext(stage_idx, target_stage_context_idx,
                                          next_stage_context_idx);
      } else {
        for (auto& batch_push : batches) {
          post_batch_queue.Put(std::move(batch_push));

          // to support the operator diveristy
          scheduler_->GetStageTargetContext(stage_idx, target_stage_context_idx,
                                            next_stage_context_idx);
        }
      }
    } else {
      // dbg("empty batch encounterred");
    }
    scheduler_->UpdateStageContext(stage_idx, used_context_idx);
  }
}

void ModelInstanceState::ProcessResponse() {
  auto& complete_event = stage_events_[total_stages_];
  uint64_t compute_end_ns;
  while (true) {
    NVTX_RANGE(nvtx_, "ProcessResponse " + Name());
    auto payload = std::move(completion_queue_.Get());
    if (payload.get() == nullptr) {
      break;
    }
    auto& event_set = complete_event[payload->event_idx_];

    // The model execution associated with the current slot
    // has consumed the inputs. Put the slot back into the available
    // slots so that it can begin enqueuing new memcpys into the input
    // buffers
    // cudaEventSynchronize(event_set.ready_for_output_);

    // Call Finalize() here to defer CUDA synchronization as much as
    // possible

    cudaEventSynchronize(event_set.input_ready_);

    scheduler_->UpdateCompleteContext(payload->context_idx_);

    payload->responder_->Finalize();

    cudaEventSynchronize(event_set.output_ready_);

    // Compute ends when the output data copy is completed
    SET_TIMESTAMP(compute_end_ns);

    // Send all the responses that haven't already been sent because
    // of an earlier error. Note that the responses are not set to
    // nullptr here as we need that indication below to determine if
    // the request we successful or not.
    for (auto& response : payload->responses_) {
      if (response != nullptr) {
        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(
                response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
            "failed to send Lego backend response");
      }
    }

    // Report statistics for each request.
    const char* request_id = "";
    uint64_t start_stamp;
    for (uint32_t r = 0; r < payload->request_count_; ++r) {
      auto& request = payload->requests_[r];
      // LOG_IF_ERROR(
      //     TRITONBACKEND_ModelInstanceReportStatistics(
      //         TritonModelInstance(), request,
      //         (payload->responses_[r] != nullptr) /* success */,
      //         payload->compute_start_ns_, payload->compute_input_end_ns_,
      //         payload->compute_output_start_ns_, compute_end_ns),
      //     "failed reporting request statistics");
      LOG_IF_ERROR(TRITONBACKEND_RequestId(request, &request_id),
                   "failed to get request id");
      LOG_IF_ERROR(TRITONBACKEND_RequestStartNS(request, &start_stamp),
                   "failed to get request start ns");
      FlushStatics({request_id,
                    std::to_string((compute_end_ns - start_stamp) / 1000000),
                    std::to_string(payload->request_count_)});
      LOG_IF_ERROR(TRITONBACKEND_RequestRelease(
                       request, TRITONSERVER_REQUEST_RELEASE_ALL),
                   "failed releasing request");
    }

    // Report the entire batch statistics.
    // LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(
    //                  TritonModelInstance(), payload->request_count_,
    //                  payload->compute_start_ns_,
    //                  payload->compute_input_end_ns_,
    //                  payload->compute_output_start_ns_, compute_end_ns),
    //              "failed reporting batch request statistics");

    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("TRITONBACKEND_ModelExecute: model ") + Name() +
         " released " + std::to_string(payload->request_count_) + " requests")
            .c_str());
  }
}

TRITONSERVER_Error* ModelInstanceState::InitStreamsAndEvents() {
  // Set the device before preparing the context.
  auto cuerr = cudaSetDevice(DeviceId());
  if (cuerr != cudaSuccess) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                 (std::string("unable to set device for ") +
                                  Name() + ": " + cudaGetErrorString(cuerr))
                                     .c_str());
  }

  // Create CUDA streams associated with the instance
  cuda_stream_priority_ = GetCudaStreamPriority(model_state_->ModelPriority());

  // The stream created by default has set priority of 0. Destroy the
  // the default stream and create a new stream with requested
  // priority.
  // FIXME, This should be moved to backend repo to directly build
  // cuda stream with required priority.
  if (cuda_stream_priority_ != 0) {
    if (stream_ != nullptr) {
      cudaError_t err = cudaStreamDestroy(stream_);
      if (err != cudaSuccess) {
        TRITONSERVER_LogMessage(
            TRITONSERVER_LOG_ERROR, __FILE__, __LINE__,
            (std::string("~BackendModelInstance: ") + name_ +
             " failed to destroy cuda stream: " + cudaGetErrorString(err))
                .c_str());
      }
      stream_ = nullptr;
      RETURN_IF_ERROR(
          CreateCudaStream(DeviceId(), cuda_stream_priority_, &stream_));
    }
  }
  RETURN_IF_ERROR(
      CreateCudaStream(DeviceId(), cuda_stream_priority_, &input_copy_stream_));
  // Create cuda streams for each stage
  for (size_t stage_idx = 0; stage_idx < total_stages_; stage_idx++) {
    RETURN_IF_ERROR(CreateCudaStream(DeviceId(), cuda_stream_priority_,
                                     &stage_execution_streams_[stage_idx]));
  }
  // Create CUDA events associated with the execution states
  RETURN_IF_ERROR(InitEventSet(model_state_->BusyWaitEvents()));

  return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::InitEventSet(bool busy_wait_events) {
  unsigned int event_flags =
      (busy_wait_events ? cudaEventDefault : cudaEventBlockingSync) |
      cudaEventDisableTiming;
  for (size_t stage_idx = 0; stage_idx <= total_stages_; stage_idx++) {
    for (int idx = 0; idx < buffer_set_size_; idx++) {
      RETURN_IF_ERROR(CreateCudaEvent(
          "Stage " + std::to_string(stage_idx) + "Set " + std::to_string(idx) +
              " ready for input",
          event_flags, &stage_events_[stage_idx][idx].ready_for_input_));
      RETURN_IF_ERROR(CreateCudaEvent(
          "Stage " + std::to_string(stage_idx) + "Set " + std::to_string(idx) +
              " input ready",
          event_flags, &stage_events_[stage_idx][idx].input_ready_));
      RETURN_IF_ERROR(CreateCudaEvent(
          "Stage " + std::to_string(stage_idx) + "Set " + std::to_string(idx) +
              " ready for output",
          event_flags, &stage_events_[stage_idx][idx].ready_for_output_));
      RETURN_IF_ERROR(CreateCudaEvent(
          "Stage " + std::to_string(stage_idx) + "Set " + std::to_string(idx) +
              " output ready",
          event_flags, &stage_events_[stage_idx][idx].output_ready_));
#ifdef TRITON_ENABLE_STATS
      RETURN_IF_ERROR(CreateCudaEvent(
          "Stage " + std::to_string(stage_idx) + "Set " + std::to_string(idx) +
              " timestamp signal",
          event_flags, &stage_events_[stage_idx][idx].timestamp_signal_));
#endif  // TRITON_ENABLE_STATS
    }
  }
  return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::DestroyEventSet() {
  for (size_t stage_idx = 0; stage_idx <= total_stages_; stage_idx++) {
    for (int idx = 0; idx < buffer_set_size_; idx++) {
      if (stage_events_[stage_idx][idx].ready_for_input_ != nullptr) {
        cudaEventDestroy(stage_events_[stage_idx][idx].ready_for_input_);
      }
      if (stage_events_[stage_idx][idx].input_ready_ != nullptr) {
        cudaEventDestroy(stage_events_[stage_idx][idx].input_ready_);
      }
      if (stage_events_[stage_idx][idx].ready_for_output_ != nullptr) {
        cudaEventDestroy(stage_events_[stage_idx][idx].ready_for_output_);
      }
      if (stage_events_[stage_idx][idx].output_ready_ != nullptr) {
        cudaEventDestroy(stage_events_[stage_idx][idx].output_ready_);
      }
      if (stage_events_[stage_idx][idx].timestamp_signal_ != nullptr) {
        cudaEventDestroy(stage_events_[stage_idx][idx].timestamp_signal_);
      }
    }
  }
  return nullptr;
}

void ModelInstanceState::RegisterContexts() {
  dbg("Registering contexts for the model instance");
  next_stage_contexts_idx_.resize(total_stages_, 0);
  scheduler_->RegisterContexts(next_context_idx_, next_stage_contexts_idx_);
}

TRITONSERVER_Error* ModelInstanceState::InitOptimizationProfiles(
    const size_t stage_idx) {
  total_bindings_[stage_idx] = stage_engines_[stage_idx]->getNbBindings();
  const int total_profiles =
      stage_engines_[stage_idx]->getNbOptimizationProfiles();
  // TRT sets the optimization profile index to be 0 implicitly with
  // the first context creation. As currently triton supports one
  // context per engine, in order to set the specified profile_index,
  // another context is created and the previous context is destroyed.
  std::shared_ptr<nvinfer1::IExecutionContext> default_trt_context(
      stage_engines_[stage_idx]->createExecutionContext());
  if (default_trt_context == nullptr) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                 "unable to create TensorRT context");
  }
  if (total_profiles == 0) {
    num_expected_bindings_[stage_idx] = total_bindings_[stage_idx];
  } else {
    num_expected_bindings_[stage_idx] =
        total_bindings_[stage_idx] / total_profiles;
  }

  // No optimization profile is set for this TensorRT plan
  if ((total_profiles == 0) || ProfileNames().empty()) {
    auto it =
        stages_trt_contexts_[stage_idx]
            .emplace(0, TensorRTContext("default", 0,
                                        num_expected_bindings_[stage_idx]))
            .first;
    it->second.context_ = std::move(default_trt_context);
    // Store the profile dimensions and set binding dimensions to
    // max dims for later initializing the input bindings
    for (int io_index = 0; io_index < num_expected_bindings_[stage_idx];
         io_index++) {
      const auto binding_index = io_index;
      if (stage_engines_[stage_idx]->bindingIsInput(binding_index)) {
        RETURN_IF_ERROR(
            GetProfileDimensions(stage_idx, io_index, 0, &it->second));
        scheduler_->InitilizeStageInputBindings(stage_idx, io_index,
                                                it->second.max_dims_[io_index]);
        if (!it->second.context_->setBindingDimensions(
                binding_index, it->second.max_dims_[io_index])) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              (std::string("01trt failed to set binding dimension to ") +
               DimsDebugString(it->second.max_dims_[io_index]) +
               " for input '" +
               stage_engines_[stage_idx]->getBindingName(binding_index) +
               "' for " + Name())
                  .c_str());
        }
      }
    }
  } else {
    // Create one TRT context for each specified profile
    for (const auto& profile_name : ProfileNames()) {
      int profile_index = 0;
      RETURN_IF_ERROR(GetProfileIndex(profile_name, &profile_index));
      auto res = stages_trt_contexts_[stage_idx].emplace(
          profile_index, TensorRTContext(profile_name, profile_index,
                                         num_expected_bindings_[stage_idx]));
      if (!res.second) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_WARN,
            (profile_name + " maps to profile index " +
             std::to_string(profile_index) + " which has been mapped by " +
             res.first->second.profile_name_ +
             ", existing optimization profile will be reused")
                .c_str());
        continue;
      }
      if (profile_index == 0) {
        res.first->second.context_ = std::move(default_trt_context);
      } else {
        res.first->second.context_.reset(
            stage_engines_[stage_idx]->createExecutionContext());
        if (res.first->second.context_ == nullptr) {
          return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                       "unable to create TensorRT context");
        }
        if (!res.first->second.context_->setOptimizationProfileAsync(
                profile_index, stream_)) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("Can not set the specified optimization "
                           "profile ") +
               profile_name + "[" + std::to_string(profile_index) + "] for " +
               name_ + ". Expected optimization profile index range 0-" +
               std::to_string(
                   stage_engines_[stage_idx]->getNbOptimizationProfiles() - 1))
                  .c_str());
        }
        cudaStreamSynchronize(CudaStream());
      }
      // Store the profile dimensions and set binding dimensions to
      // max dims for later initializing the input bindings
      for (int io_index = 0; io_index < num_expected_bindings_[stage_idx];
           io_index++) {
        const auto binding_index =
            profile_index * num_expected_bindings_[stage_idx] + io_index;
        if (stage_engines_[stage_idx]->bindingIsInput(binding_index)) {
          RETURN_IF_ERROR(GetProfileDimensions(
              stage_idx, io_index, profile_index, &res.first->second));
          if (!res.first->second.context_->setBindingDimensions(
                  binding_index, res.first->second.max_dims_[io_index])) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                (std::string("02trt failed to set binding dimension to ") +
                 DimsDebugString(res.first->second.max_dims_[io_index]) +
                 " for input '" +
                 stage_engines_[stage_idx]->getBindingName(binding_index) +
                 "' for " + Name())
                    .c_str());
          }
        }
      }
    }

    // profile 0 is not specified
    if (default_trt_context != nullptr) {
      default_trt_context.reset();
    }
  }
  return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::ValidateIO(const size_t stage_idx) {
  // Collect all the expected input and allowed output tensor names
  // and validate that the model configuration specifies only those.
  std::set<std::string> allowed_inputs, allowed_outputs;
  for (int i = 0; i < num_expected_bindings_[stage_idx]; ++i) {
    if (stage_engines_[stage_idx]->bindingIsInput(i)) {
      allowed_inputs.emplace(stage_engines_[stage_idx]->getBindingName(i));
    } else {
      allowed_outputs.emplace(stage_engines_[stage_idx]->getBindingName(i));
    }
    if (stage_engines_[stage_idx]->isExecutionBinding(i)) {
      LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                  (std::string("Detected ") +
                   stage_engines_[stage_idx]->getBindingName(i) +
                   " as execution binding for " + Name())
                      .c_str());
    }
  }
  triton::common::TritonJson::Value config_stages_inouts;
  triton::common::TritonJson::Value config_stages_inouts_with_name;
  triton::common::TritonJson::Value config_stage_inputs;
  RETURN_IF_ERROR(model_state_->LegoConfig().MemberAsArray(
      "stages_inputs", &config_stages_inouts));
  RETURN_IF_ERROR(config_stages_inouts.IndexAsObject(
      stage_idx, &config_stages_inouts_with_name));
  RETURN_IF_ERROR(config_stages_inouts_with_name.MemberAsArray(
      "input", &config_stage_inputs));
  if (allowed_inputs.size() < config_stage_inputs.ArraySize()) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("unable to load model '" + model_state_->Name() +
                    "', configuration expects " +
                    std::to_string(config_stage_inputs.ArraySize()) +
                    " inputs, model provides at most " +
                    std::to_string(allowed_inputs.size()))
            .c_str());
  }

  for (size_t i = 0; i < config_stage_inputs.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(config_stage_inputs.IndexAsObject(i, &io));
    RETURN_IF_ERROR(CheckAllowedModelInput(io, allowed_inputs));
  }

  triton::common::TritonJson::Value config_stage_outputs;
  RETURN_IF_ERROR(model_state_->LegoConfig().MemberAsArray(
      "stages_outputs", &config_stages_inouts));
  RETURN_IF_ERROR(config_stages_inouts.IndexAsObject(
      stage_idx, &config_stages_inouts_with_name));
  RETURN_IF_ERROR(config_stages_inouts_with_name.MemberAsArray(
      "output", &config_stage_outputs));

  for (size_t i = 0; i < config_stage_outputs.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(config_stage_outputs.IndexAsObject(i, &io));
    RETURN_IF_ERROR(CheckAllowedModelOutput(io, allowed_outputs));
  }

  RETURN_IF_ERROR(
      ValidateIOHelper(config_stage_inputs, true, true /* is_input */));
  RETURN_IF_ERROR(
      ValidateIOHelper(config_stage_outputs, true, false /* is_input */));

  return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::ValidateIOHelper(
    common::TritonJson::Value& ios, const bool if_stage, const bool is_input) {
  std::string type = is_input ? "input" : "output";
  for (size_t i = 0; i < ios.ArraySize(); i++) {
    triton::common::TritonJson::Value io;
    RETURN_IF_ERROR(ios.IndexAsObject(i, &io));

    std::string io_name;
    RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
    if (if_stage) {
      size_t buffer_id;
      RETURN_IF_ERROR(io.MemberAsUInt("id", &buffer_id));
    } else {
      std::string io_data_type;
      RETURN_IF_ERROR(io.MemberAsString("data_type", &io_data_type));
      if (!ConvertDataTypeToTrtType(
               ModelConfigDataTypeToTritonServerDataType(io_data_type))
               .first) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("unsupported datatype") + io_data_type + " for " +
             type + " '" + io_name + "' for model '" + model_state_->Name() +
             "'")
                .c_str());
      }
    }
  }

  return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::InitIOBindingBuffers(
    const size_t stage_idx) {
  auto config_stage_inputs = model_state_->StagesInputs(stage_idx);
  auto config_stage_outputs = model_state_->StagesOutputs(stage_idx);
  // Initialize the inputs and outputs. Make sure the model matches
  // what is in the configuration. Allocate memory for the maximum
  // possible batch size: min(engine maximum, config maximum)
  for (auto& stage_io_binding_infos : stages_io_binding_infos_[stage_idx]) {
    stage_io_binding_infos.resize(num_expected_bindings_[stage_idx]);
  }
  for (auto& stage_buffer_bindings : stages_buffer_bindings_[stage_idx]) {
    stage_buffer_bindings.resize(total_bindings_[stage_idx], nullptr);
  }

  for (int s = 0; s < buffer_set_size_; s++) {
    next_buffer_binding_set_ = s;
    RETURN_IF_ERROR(InitializeConfigStageExecuteInputBindings(
        stage_idx, config_stage_inputs));
  }

  for (const auto& trt_context : stages_trt_contexts_[stage_idx]) {
    if (!trt_context.second.context_->allInputDimensionsSpecified()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          "failed to specify the dimensions of all input bindings");
    }
  }

  // Validate the batch dimension against the implicit batch dimension
  // if available.
  if (stage_engines_[stage_idx]->hasImplicitBatchDimension() &&
      (model_state_->MaxBatchSize() >
       stage_engines_[stage_idx]->getMaxBatchSize())) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        (std::string("unexpected configuration maximum batch size ") +
         std::to_string(model_state_->MaxBatchSize()) + " for '" + Name() +
         "', model maximum is " +
         std::to_string(stage_engines_[stage_idx]->getMaxBatchSize()))
            .c_str());
  }

  // batch output must be processed before other outputs
  for (int s = 0; s < buffer_set_size_; s++) {
    next_buffer_binding_set_ = s;
    RETURN_IF_ERROR(InitializeConfigStageExecuteOutputBindings(
        stage_idx, config_stage_outputs));
  }
  next_buffer_binding_set_ = 0;
  // Make sure every index which corresponds to an execution binding
  // is initialized.
  for (int s = 0; s < buffer_set_size_; ++s) {
    for (int i = 0; i < num_expected_bindings_[stage_idx]; ++i) {
      if (stages_io_binding_infos_[stage_idx][s][i].buffer_ == nullptr &&
          stage_engines_[stage_idx]->isExecutionBinding(i)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("expected configuration for ") +
             std::string((stage_engines_[stage_idx]->bindingIsInput(i)
                              ? "input"
                              : "output")) +
             " '" + stage_engines_[stage_idx]->getBindingName(i) + "' for " +
             Name())
                .c_str());
      }
    }
  }
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitializeConfigStageExecuteInputBindings(
    const size_t stage_idx,
    std::unordered_map<std::string, StageInOut> config_stage_inputs) {
  for (auto input_it = config_stage_inputs.begin();
       input_it != config_stage_inputs.end(); input_it++) {
    std::string io_name = input_it->first;
    std::string io_datatype = std::get<0>(input_it->second);
    std::vector<int64_t> io_dims = std::get<1>(input_it->second);
    size_t io_buffer_id = std::get<2>(input_it->second);
    RETURN_IF_ERROR(InitializeStageExecuteInputBinding(
        stage_idx, io_name, io_datatype, io_dims, io_buffer_id, false));
  }

  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::InitializeConfigStageExecuteOutputBindings(
    const size_t stage_idx,
    std::unordered_map<std::string, StageInOut> config_stage_outputs) {
  for (auto output_it = config_stage_outputs.begin();
       output_it != config_stage_outputs.end(); output_it++) {
    // the maximum byte sizes across all profiles
    int64_t max_byte_size = 0;

    std::string io_name = output_it->first;
    std::string io_datatype = std::get<0>(output_it->second);
    std::vector<int64_t> io_dims = std::get<1>(output_it->second);
    size_t io_buffer_id = std::get<2>(output_it->second);

    int io_index = stage_engines_[stage_idx]->getBindingIndex(io_name.c_str());

    auto& io_binding_info =
        stages_io_binding_infos_[stage_idx][next_buffer_binding_set_][io_index];
    for (auto& trt_context : stages_trt_contexts_[stage_idx]) {
      auto& profile_index = trt_context.first;
      auto& context = trt_context.second;
      int binding_index =
          num_expected_bindings_[stage_idx] * profile_index + io_index;
      if (binding_index < 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_NOT_FOUND,
            (std::string("output '") + io_name + "' not found for " + Name())
                .c_str());
      }

      if (io_binding_info.buffer_ != nullptr) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("output '") + io_name +
             "'  has already appeared as an input or output for " + Name())
                .c_str());
      }

      if (stage_engines_[stage_idx]->bindingIsInput(binding_index)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("output '") + io_name +
             "' is expected to be an input in model for " + Name())
                .c_str());
      }

      TRITONSERVER_DataType dt = ConvertTrtTypeToDataType(
          stage_engines_[stage_idx]->getBindingDataType(binding_index));
      TRITONSERVER_DataType config_dt =
          ModelConfigDataTypeToTritonServerDataType(io_datatype);
      if ((dt == TRITONSERVER_TYPE_INVALID) || (dt != config_dt)) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("unexpected datatype TYPE_") +
             TRITONSERVER_DataTypeString(dt) + " for inference output '" +
             io_name + "', expecting TYPE_" +
             TRITONSERVER_DataTypeString(config_dt) + " for " + Name())
                .c_str());
      }

      io_binding_info.is_linear_format_ =
          (stage_engines_[stage_idx]->getBindingFormat(binding_index) ==
           nvinfer1::TensorFormat::kLINEAR);
      if (!io_binding_info.is_linear_format_) {
        io_binding_info.vectorized_dim_ =
            stage_engines_[stage_idx]->getBindingVectorizedDim(binding_index);
        io_binding_info.components_per_element_ =
            stage_engines_[stage_idx]->getBindingComponentsPerElement(
                binding_index);
        if (io_binding_info.vectorized_dim_ == -1) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              (std::string("unexpected vectorized dim is -1 for non-linear "
                           "output '") +
               io_name + "' for " + Name())
                  .c_str());
        }
      }

      nvinfer1::Dims stages_engines_dims =
          stage_engines_[stage_idx]->getBindingDimensions(binding_index);
      // Skip 'batch_output' validation as it is not exact match to
      // model dims
      RETURN_IF_ERROR(CompareDimsSupported(
          name_, io_name, stages_engines_dims, io_dims, true,
          (!stage_engines_[stage_idx]->hasImplicitBatchDimension()),
          false /* compare_exact */));

      int64_t byte_size;
      const nvinfer1::Dims output_dim =
          context.context_->getBindingDimensions(binding_index);
      std::vector<int64_t> dim_vec;
      DimsToDimVec(output_dim, &dim_vec);
      byte_size = GetByteSize(dt, dim_vec);

      if (byte_size == -1) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("unable to allocate memory for output '") + io_name +
             "' for " + Name())
                .c_str());
      }
      max_byte_size = std::max(max_byte_size, byte_size);
    }

    // Allocate CUDA memory. Use cudaHostAlloc if zero copy supported.
    // We rely on stages_buffer_bindings_ being non-nullptr to indicate that
    // the buffer has been correctly initalized so even for zero-sized
    // tensors always allocate something.
    void* buffer = nullptr;
    auto buffer_it = all_buffers_[next_buffer_binding_set_].find(io_buffer_id);
    if (buffer_it != all_buffers_[next_buffer_binding_set_].end()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("output '") + io_name + std::string("' buffer id '") +
           std::to_string(io_buffer_id) + "' already allocated for " + Name())
              .c_str());
    } else {
      cudaError_t err = cudaSuccess;
      err = cudaMalloc(&buffer, std::max((int64_t)1, max_byte_size));
      if (err != cudaSuccess) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            (std::string("unable to allocate memory for output '") + io_name +
             "' for " + Name() + ": " + cudaGetErrorString(err))
                .c_str());
      }
      all_buffers_[next_buffer_binding_set_].emplace(
          io_buffer_id, std::make_pair(buffer, max_byte_size));
    }

    io_binding_info.byte_size_ = max_byte_size;
    io_binding_info.buffer_ = buffer;
    io_binding_info.buffer_id_ = io_buffer_id;
    io_binding_info.device_buffer_ = buffer;
    io_binding_info.memory_type_ = TRITONSERVER_MEMORY_GPU;
    io_binding_info.memory_type_id_ = DeviceId();

    // Set buffer bindings of all optimization profile since buffer is
    // allocated
    for (auto& trt_context : stages_trt_contexts_[stage_idx]) {
      auto binding_index =
          num_expected_bindings_[stage_idx] * trt_context.first + io_index;
      stages_buffer_bindings_[stage_idx][next_buffer_binding_set_]
                             [binding_index] = io_binding_info.device_buffer_;
    }
  }

  return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::InitializeStageExecuteInputBinding(
    const size_t stage_idx, const std::string& input_name,
    const std::string& input_datatype, std::vector<int64_t>& input_dims,
    size_t& buffer_id, const bool is_control) {
  // the maximum byte sizes across all profiles
  int64_t max_byte_size = 0;
  int io_index = stage_engines_[stage_idx]->getBindingIndex(input_name.c_str());
  auto& io_binding_info =
      stages_io_binding_infos_[stage_idx][next_buffer_binding_set_][io_index];
  for (auto& trt_context : stages_trt_contexts_[stage_idx]) {
    auto& profile_index = trt_context.first;
    auto& context = trt_context.second;
    int binding_index =
        num_expected_bindings_[stage_idx] * profile_index + io_index;
    if (io_index < 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_NOT_FOUND,
          (std::string("input '") + input_name + "' not found for " + Name())
              .c_str());
    }

    if (io_binding_info.buffer_ != nullptr) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("input '") + input_name +
           "'  has already appeared as an input or output for " + Name())
              .c_str());
    }

    if (!stage_engines_[stage_idx]->bindingIsInput(binding_index)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("input '") + input_name +
           "' is expected to be an input in model for " + Name())
              .c_str());
    }

    TRITONSERVER_DataType dt = ConvertTrtTypeToDataType(
        stage_engines_[stage_idx]->getBindingDataType(binding_index));
    TRITONSERVER_DataType config_dt =
        ModelConfigDataTypeToTritonServerDataType(input_datatype);
    if ((dt == TRITONSERVER_TYPE_INVALID) || (dt != config_dt)) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("unexpected datatype TYPE_") +
           TRITONSERVER_DataTypeString(dt) + " for inference input '" +
           input_name + "', expecting TYPE_" +
           TRITONSERVER_DataTypeString(config_dt) + " for " + Name())
              .c_str());
    }

    io_binding_info.is_linear_format_ =
        (stage_engines_[stage_idx]->getBindingFormat(binding_index) ==
         nvinfer1::TensorFormat::kLINEAR);
    if (!io_binding_info.is_linear_format_) {
      io_binding_info.vectorized_dim_ =
          stage_engines_[stage_idx]->getBindingVectorizedDim(binding_index);
      io_binding_info.components_per_element_ =
          stage_engines_[stage_idx]->getBindingComponentsPerElement(
              binding_index);
      if (io_binding_info.vectorized_dim_ == -1) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("unexpected vectorized dim is -1 for "
                         "non-linear input '") +
             input_name + "' for " + Name())
                .c_str());
      }
    }

    // Detect whether dynamic or not
    nvinfer1::Dims stages_engines_dims =
        stage_engines_[stage_idx]->getBindingDimensions(binding_index);
    if (ContainsWildcard(stages_engines_dims)) {
      context.is_dynamic_per_binding_[io_index] = true;
    }

    if (!(is_control && context.is_dynamic_per_binding_[io_index])) {
      RETURN_IF_ERROR(CompareDimsSupported(
          name_, input_name, stages_engines_dims, input_dims, true,
          (!stage_engines_[stage_idx]->hasImplicitBatchDimension()),
          false /* compare_exact */));
    } else {
      TRITONSERVER_Error* err =
          ValidateControlDimsDynamic(stages_engines_dims, true);
      if (err != nullptr) {
        TRITONSERVER_Error* full_err = TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("unexpected shape ") +
             DimsDebugString(stages_engines_dims) + " for control input '" +
             input_name + "' for model " + model_state_->Name() + ": " +
             TRITONSERVER_ErrorMessage(err))
                .c_str());
        TRITONSERVER_ErrorDelete(err);
        return full_err;
      }
    }

    int64_t byte_size = 0;

    std::vector<int64_t> maximum_dims;
    // TRITONSERVER_Error* err =
    //     ValidateDimension(input_dims, context.min_dims_[io_index],
    //                       context.max_dims_[io_index], true);
    // if (err != nullptr) {
    //   TRITONSERVER_Error* full_err = TRITONSERVER_ErrorNew(
    //       TRITONSERVER_ERROR_INVALID_ARG,
    //       (std::string("model configuration specified invalid shape for "
    //                    "input '") +
    //        input_name + "' for model " + model_state_->Name() +
    //        ". Error details: " + TRITONSERVER_ErrorMessage(err))
    //           .c_str());

    //   TRITONSERVER_ErrorDelete(err);
    //   return full_err;
    // }
    // RETURN_IF_ERROR(MaximumDims(context.max_dims_[io_index], input_dims,
    // true,
    //                             model_state_->MaxBatchSize(),
    //                             &maximum_dims));
    for (auto dim_idx = 0; dim_idx < context.max_dims_[io_index].nbDims;
         ++dim_idx) {
      maximum_dims.push_back(context.max_dims_[io_index].d[dim_idx]);
    }
    byte_size = GetByteSize(dt, maximum_dims);
    // Update the maximum dimension with respect to the allocated
    // buffer
    DimVecToDims(maximum_dims, &context.max_dims_[io_index]);

    if (!context.context_->setBindingDimensions(binding_index,
                                                context.max_dims_[io_index])) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("03trt failed to set binding dimension to ") +
           DimsDebugString(context.max_dims_[io_index]) + " for input '" +
           input_name + "' for " + Name())
              .c_str());
    }
    if (!io_binding_info.is_linear_format_) {
      maximum_dims[io_binding_info.vectorized_dim_] +=
          (io_binding_info.components_per_element_ -
           (maximum_dims[io_binding_info.vectorized_dim_] %
            io_binding_info.components_per_element_));
      byte_size = GetByteSize(dt, maximum_dims);
    }

    if (byte_size == -1) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unable to calculate size for input '") + input_name +
           "' for " + Name())
              .c_str());
    }
    max_byte_size = std::max(max_byte_size, byte_size);
  }

  // Allocate CUDA memory. Use cudaHostAlloc if zero copy supported.
  // We rely on stages_buffer_bindings_ being non-nullptr to indicate that
  // the buffer has been correctly initalized so even for zero-sized
  // tensors always allocate something.
  void* buffer = nullptr;
  auto buffer_it = all_buffers_[next_buffer_binding_set_].find(buffer_id);
  if (buffer_it == all_buffers_[next_buffer_binding_set_].end()) {
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&buffer, std::max((int64_t)1, max_byte_size));
    if (err != cudaSuccess) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unable to allocate memory for input '") + input_name +
           "' for " + Name() + ": " + cudaGetErrorString(err))
              .c_str());
    }
    all_buffers_[next_buffer_binding_set_].emplace(
        buffer_id, std::make_pair(buffer, max_byte_size));
  } else {
    buffer = buffer_it->second.first;
    auto buffer_byte_size = buffer_it->second.second;
    if (max_byte_size > buffer_byte_size) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("unconsistent buffer size for input '") + input_name +
           std::string("' buffer id '") + std::to_string(buffer_it->first) +
           "' for " + Name() + ": buffer size " +
           std::to_string(buffer_byte_size) +
           " is smaller than required size " + std::to_string(max_byte_size))
              .c_str());
    }
  }

  io_binding_info.byte_size_ = max_byte_size;
  io_binding_info.buffer_ = buffer;
  io_binding_info.buffer_id_ = buffer_id;
  io_binding_info.device_buffer_ = buffer;
  io_binding_info.memory_type_ = TRITONSERVER_MEMORY_GPU;
  io_binding_info.memory_type_id_ = DeviceId();

  // Set buffer bindings of all optimization profile since buffer is
  // allocated
  for (auto& trt_context : stages_trt_contexts_[stage_idx]) {
    auto binding_index =
        num_expected_bindings_[stage_idx] * trt_context.first + io_index;
    stages_buffer_bindings_[stage_idx][next_buffer_binding_set_]
                           [binding_index] = io_binding_info.device_buffer_;
  }

  return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::GetProfileDimensions(
    const size_t stage_idx, const int io_index, const int profile_index,
    TensorRTContext* context) {
  int binding_index =
      (profile_index * num_expected_bindings_[stage_idx]) + io_index;
  context->max_dims_[io_index] =
      stage_engines_[stage_idx]->getProfileDimensions(
          binding_index, profile_index, nvinfer1::OptProfileSelector::kMAX);
  context->min_dims_[io_index] =
      stage_engines_[stage_idx]->getProfileDimensions(
          binding_index, profile_index, nvinfer1::OptProfileSelector::kMIN);
  context->opt_dims_[io_index] =
      stage_engines_[stage_idx]->getProfileDimensions(
          binding_index, profile_index, nvinfer1::OptProfileSelector::kOPT);
  return nullptr;
}

void ModelInstanceState::GetConfiguredProfiles(size_t stage_idx,
                                               std::string* profiles_desc) {
  profiles_desc->clear();
  for (const auto& trt_context : stages_trt_contexts_[stage_idx]) {
    (*profiles_desc) += (" " + trt_context.second.profile_name_ + "[" +
                         std::to_string(trt_context.first) + "];");
  }
}

/////////////

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend. But
// here it simply verify the backend API version is compatible
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_Initialize(
    TRITONBACKEND_Backend* backend) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("Triton TRITONBACKEND API version: ") +
               std::to_string(api_version_major) + "." +
               std::to_string(api_version_minor))
                  .c_str());
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("'") + name + "' TRITONBACKEND API version: " +
               std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
               std::to_string(TRITONBACKEND_API_VERSION_MINOR))
                  .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // Set the execution policy as device blocking for the backend.
  RETURN_IF_ERROR(TRITONBACKEND_BackendSetExecutionPolicy(
      backend, TRITONBACKEND_EXECUTION_DEVICE_BLOCKING));

  // Register all the default plugins that come with TensorRT
  bool success = true;
  std::once_flag onceFlag;
  {
    std::call_once(onceFlag, [&success] {
      LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE, "Registering TensorRT Plugins");
      success = initLibNvInferPlugins(&lego_logger, "");
    });
  }
  if (!success) {
    TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                          "unable to register default TensorRT Plugins");
  }

  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(backend_config_message,
                                                      &buffer, &byte_size));
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("backend configuration:\n") + buffer).c_str());

  triton::common::TritonJson::Value backend_config;
  if (byte_size != 0) {
    RETURN_IF_ERROR(backend_config.Parse(buffer, byte_size));
  }

  std::unique_ptr<BackendConfiguration> lconfig(new BackendConfiguration());
  triton::common::TritonJson::Value cmdline;
  if (backend_config.Find("cmdline", &cmdline)) {
    triton::common::TritonJson::Value value;
    std::string value_str;
    if (cmdline.Find("coalesce-request-input", &value)) {
      RETURN_IF_ERROR(value.AsString(&value_str));
      RETURN_IF_ERROR(
          ParseBoolValue(value_str, &lconfig->coalesce_request_input_));
    }
  }
  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(lconfig.get())));

  lconfig.release();
  return nullptr;  // success
}

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_Finalize(
    TRITONBACKEND_Backend* backend) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  delete reinterpret_cast<BackendConfiguration*>(vstate);
  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(
    TRITONBACKEND_Model* model) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_ModelInitialize: ") + name +
               " (version " + std::to_string(version) + ")")
                  .c_str());

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state = nullptr;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(
    TRITONBACKEND_Model* model) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(
    TRITONBACKEND_ModelInstance* instance) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  TRITONSERVER_InstanceGroupKind kind;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceKind(instance, &kind));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
               " (" + TRITONSERVER_InstanceGroupKindString(kind) + " device " +
               std::to_string(device_id) + ")")
                  .c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(
    TRITONBACKEND_ModelInstance* instance) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONBACKEND_ISPEC TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count) {
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
              (std::string("model ") + model_state->Name() + ", instance " +
               instance_state->Name() + ", executing " +
               std::to_string(request_count) + " requests")
                  .c_str());

  // At this point we accept ownership of 'requests', which means that
  // even if something goes wrong we must still return success from
  // this function. If something does go wrong in processing a
  // particular request then we send an error response just for the
  // specific request.
  instance_state->ProcessRequests(requests, request_count);

  return nullptr;  // success
}

}  // extern "C"
}}}  // namespace triton::backend::lego
