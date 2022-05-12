/*!
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * \file: /lego_schedule.cc
 * \brief:
 * Author: raphael hao
 */

#pragma once

#include "dbg.h"
#include "lego_utils.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/common/nvtx.h"

#include <NvInferPlugin.h>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace triton { namespace backend { namespace lego {

// Number of CUDA event set for each instance.
// static constexpr size_t EVENT_SET_SIZE = 2;
// static constexpr size_t BUFFER_BINDING_SET_SIZE = 2;
static constexpr size_t PAYLOAD_MATRIX_SIZE = 2000;
// static constexpr size_t CONTEXT_SIZE = 2;

#define LEGO_SET_TIMESTAMP(TS_NS)                                    \
  {                                                                  \
    TS_NS = std::chrono::duration_cast<std::chrono::nanoseconds>(    \
                std::chrono::steady_clock::now().time_since_epoch()) \
                .count();                                            \
  }

#define LEGO_DECL_TIMESTAMP(TS_NS) \
  uint64_t TS_NS;                  \
  LEGO_SET_TIMESTAMP(TS_NS);

struct Batch {
  size_t payload_matrix_idx_;
  size_t payload_vector_idx_;
  size_t start_;
  size_t end_;
  int used_context_idx_;  // the context id used in previous stage
  size_t event_idx_;
  // std::unordered_map<int, nvinfer1::Dims> bindings_shape_;
  size_t len_;
  int bs_;
  // FIXME remove some parameters used for initalization
  explicit Batch(const size_t& payload_matrix_idx = 0,
                 const size_t& payload_vector_id = 0, const size_t& start = 0,
                 const size_t& end = 0, const size_t& event_idx = 0,
                 const int& used_context_idx = 0, const size_t& len = 0)
      : payload_matrix_idx_(payload_matrix_idx),
        payload_vector_idx_(payload_vector_id),
        start_(start),
        end_(end),
        used_context_idx_(used_context_idx),
        event_idx_(event_idx),
        len_(len) {
    bs_ = end_ - start_;
  }
};

class StageQueue {
 public:
  StageQueue() {}
  void SetID(size_t stage_idx) { stage_idx_ = stage_idx; }
  bool Empty() {
    std::lock_guard<std::mutex> lk(mu_);
    return queue_.empty();
  }

  std::unique_ptr<Batch> Get() {
    std::unique_lock<std::mutex> lk(mu_);
    if (queue_.empty()) {
      cv_.wait(lk, [this] { return !queue_.empty(); });
    }

    auto res = std::move(queue_.front());
    // std::cout << "Stage Queue: " << stage_idx_ << " get Batch id: ("
    //           << res->payload_matrix_idx_ << ", " << res->payload_vector_idx_
    //           << "), (bs, start, end): (" << res->bs_ << ", " << res->start_
    //           << ", " << res->end_ << ")" << std::endl;
    queue_.pop_front();
    return res;
  }

  // void Put(const std::unique_ptr<Batch>& value) {
  //   {
  //     std::lock_guard<std::mutex> lk(mu_);
  //     queue_.push_back(std::move(value));
  //   }
  //   cv_.notify_all();
  // }

  void Put(std::unique_ptr<Batch>&& value) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      // std::cout << "Stage Queue: " << stage_idx_ << " putting Batch id: ("
      //           << value->payload_matrix_idx_ << ", "
      //           << value->payload_vector_idx_ << "), (bs, start, end): ("
      //           << value->bs_ << ", " << value->start_ << ", " << value->end_
      //           << ")" << std::endl;
      queue_.push_back(std::move(value));
    }
    cv_.notify_all();
  }

 private:
  size_t stage_idx_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::unique_ptr<Batch>> queue_;
};

// The details needed by the completion thread to finalize the
// response for a model execution.
struct Payload {
  explicit Payload(const size_t& matrix_idx, const size_t& vec_idx,
                   const size_t& event_idx, const size_t& buffer_binding_idx,
                   TRITONBACKEND_Request** requests,
                   const uint32_t& request_count, const int& context_idx,
                   const uint64_t& compute_start_ns = 0)
      : matrix_idx_(matrix_idx),
        vec_idx_(vec_idx),
        event_idx_(event_idx),
        buffer_binding_idx_(buffer_binding_idx),
        compute_start_ns_(compute_start_ns),
        compute_input_end_ns_(0),
        compute_output_start_ns_(0),
        requests_(requests),
        request_count_(request_count),
        context_idx_(context_idx) {
    requests_list_.reserve(request_count_);
    for (size_t i = 0; i < request_count; ++i) {
      requests_list_.emplace_back(requests[i]);
    }
    requests_ = requests_list_.data();
  }

  size_t matrix_idx_;

  size_t vec_idx_;
  // The index to the event set handling the request
  size_t event_idx_;
  // The index to the buffer binding set handling the request
  size_t buffer_binding_idx_;

  // The total batch size for the request
  // size_t total_batch_size_;

  // The timestamps for reporting stats
  uint64_t compute_start_ns_;
  uint64_t compute_input_end_ns_;
  uint64_t compute_output_start_ns_;

  // All the composing InferenceRequest objects
  std::vector<TRITONBACKEND_Request*> requests_list_;
  TRITONBACKEND_Request** requests_;
  uint32_t request_count_;
  int context_idx_;

  // All the generated InferenceResponse objects
  std::vector<TRITONBACKEND_Response*> responses_;
  // The collector and responder of the payload, need to extend
  // their lifetime to match the payload to ensure content is intact
  // until the end of execution.
  std::unique_ptr<BackendInputCollector> collector_;
  std::unique_ptr<BackendOutputResponder> responder_;
};

struct PayloadVector {
  explicit PayloadVector()
      : matrix_idx_(0),
        total_payloads_(0),
        total_stages_(0),
        disable_merge_(false) {}

  void Init(const size_t& matrix_idx, const size_t& total_stages) {
    matrix_idx_ = matrix_idx;
    total_stages_ = total_stages;
    handled_batch_size_.resize(total_stages, 0);
    total_payloads_ = 0;
    disable_merge_ = false;
  }
  void Reset() {
    total_payloads_ = 0;
    total_request_count_ = 0;
    handled_stages_ = 0;
    payloads_.clear();
    std::fill(handled_batch_size_.begin(), handled_batch_size_.end(), 0);
    preprocessed_request_count_ = 0;
    sequence_length_ = 0;
    disable_merge_ = false;
  }

  size_t New(const size_t& event_idx, const size_t& buffer_binding_idx,
             TRITONBACKEND_Request** requests, const uint32_t& request_count,
             const int& context_idx, bool disable_merge = false) {
    // dbg("New Payload at Matrix id: " + std::to_string(matrix_idx_));
    total_payloads_ = 0;
    handled_stages_ = 0;
    preprocessed_request_count_ = 0;
    std::fill(handled_batch_size_.begin(), handled_batch_size_.end(), 0);
    payloads_.clear();
    disable_merge_ = disable_merge;
    event_idx_ = event_idx;
    buffer_binding_idx_ = buffer_binding_idx;
    context_idx_ = context_idx;
    total_request_count_ = request_count;
    LEGO_SET_TIMESTAMP(compute_start_ns_);
    payloads_.emplace_back(new Payload(
        matrix_idx_, total_payloads_++, event_idx_, buffer_binding_idx_,
        requests, request_count, context_idx_, compute_start_ns_));
    return total_request_count_;
  }
  // only support strectch while one payload resists in the vector
  size_t Stretch(TRITONBACKEND_Request** requests, uint32_t request_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!disable_merge_) {
      // dbg("Payload at Matrix id: " + std::to_string(matrix_idx_) +
      // " is strectched");
      preprocessed_request_count_ = total_request_count_;
      total_request_count_ += request_count;
      auto& payload = payloads_.back();
      payload->request_count_ += request_count;
      payload->requests_list_.reserve(total_request_count_);
      for (uint32_t i = 0; i < request_count; ++i) {
        payload->requests_list_.push_back(requests[i]);
      }
      payload->requests_ = payload->requests_list_.data();
      return total_request_count_;
    }
    return 0;
  }

  // FIXME only support split from the payload with index 0. Now, we only reset
  // the request count, we did not release the resource in the split payload,
  // when split is enabled, we have already made sure that merge is disabled.
  std::vector<std::unique_ptr<Batch>> Split(const size_t& stage_idx,
                                            const size_t& vector_idx,
                                            const size_t& split_batch_size,
                                            const int& stage_context_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    // dbg("Payload at Matrix id: " + std::to_string(matrix_idx_) +
    // " is splitted");
    if (!disable_merge_) {
      LOG_MESSAGE(TRITONSERVER_LOG_WARN, "merge is still enabled");
      disable_merge_ = true;
    }
    auto& split_payload = payloads_[vector_idx];
    std::vector<std::unique_ptr<Batch>> ret_batch;
    ret_batch.emplace_back(new Batch(matrix_idx_, vector_idx, 0,
                                     split_batch_size, event_idx_,
                                     stage_context_idx, sequence_length_));

    size_t end_request_idx = split_payload->request_count_;
    payloads_.emplace_back(new Payload(
        matrix_idx_, total_payloads_, event_idx_, buffer_binding_idx_,
        &payloads_[vector_idx]->requests_[split_batch_size],
        end_request_idx - split_batch_size, context_idx_));
    payloads_[total_payloads_]->responses_ =
        std::vector<TRITONBACKEND_Response*>(
            split_payload->responses_.begin() + split_batch_size,
            split_payload->responses_.begin() + end_request_idx);
    ret_batch.emplace_back(new Batch(
        matrix_idx_, total_payloads_, split_batch_size, end_request_idx,
        event_idx_, stage_context_idx, sequence_length_));
    total_payloads_++;
    payloads_[vector_idx]->request_count_ = split_batch_size;
    return std::move(ret_batch);
  }

  int Forward(const size_t& stage_idx, const size_t& end) {
    if (end == total_request_count_) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (end == total_request_count_) {
        size_t start_request_idx = handled_batch_size_[stage_idx];
        handled_batch_size_[stage_idx] = end;
        handled_stages_ = std::max(handled_stages_, stage_idx);
        return start_request_idx;
      }
    }
    return -1;  // indicate fail to forward
  }

  // id
  size_t matrix_idx_;
  // total number of payloads
  size_t total_payloads_;
  // total stages
  size_t total_stages_;
  // event set index
  size_t event_idx_;
  // context index
  int context_idx_;
  // buffer binding set index
  size_t buffer_binding_idx_;
  // total request count
  size_t total_request_count_;
  // preprocessed requests
  size_t preprocessed_request_count_;
  // handled stages
  size_t handled_stages_;
  // start time of the whole payload vector
  uint64_t compute_start_ns_;
  // NOTE no use of total batch size
  // size_t total_batch_size_;
  // control status
  // mutex for update the payload status
  std::mutex mutex_;
  // payloads store in a payload vector
  std::vector<std::unique_ptr<Payload>> payloads_;
  // all requests sotred in payloads
  // std::vector<TRITONBACKEND_Request*> requests_list_;
  // the handled request status for each stage
  std::vector<size_t> handled_batch_size_;
  // request length for current payload
  size_t sequence_length_;
  // disable merge
  bool disable_merge_;
};

class Scheduler {
 public:
  Scheduler(const size_t& total_stages, const int64_t& buffer_set_size,
            const size_t& max_allowed_batch_size)
      : total_stages_(total_stages),
        buffer_set_size_(buffer_set_size),
        max_allowed_batch_size_(max_allowed_batch_size) {
    stage_input_bindings_shape_.resize(total_stages_);
  }
  virtual ~Scheduler() {}

  void InitilizeStageInputBindings(size_t stage_idx, int input_binding_idx,
                                   nvinfer1::Dims shape) {
    stage_input_bindings_shape_[stage_idx][input_binding_idx] = shape;
  }

  void RegisterContexts(int& main_context_idx,
                        std::vector<int>& stage_contexts_idx) {
    main_ctx_queue_ = new triton::common::SyncQueue<int>();
    for (size_t stage_idx = 0; stage_idx < total_stages_; stage_idx++) {
      stage_cur_ctx_queues_.emplace_back(new triton::common::SyncQueue<int>());
    }
    stage_prev_ctx_queues_.resize(total_stages_, nullptr);
    for (size_t stage_idx = 0; stage_idx < total_stages_; stage_idx++) {
      if (stage_idx == 0) {
        stage_prev_ctx_queues_[stage_idx] = main_ctx_queue_;
      } else {
        stage_prev_ctx_queues_[stage_idx] =
            stage_cur_ctx_queues_[stage_idx - 1];
      }
    }
    complete_ctx_queue_ = stage_cur_ctx_queues_[total_stages_ - 1];

    for (int context_idx = 0; context_idx < buffer_set_size_; context_idx++) {
      main_ctx_queue_->Put(context_idx);
      for (size_t stage_idx = 0; stage_idx < total_stages_; stage_idx++) {
        stage_cur_ctx_queues_[stage_idx]->Put(context_idx);
      }
    }

    main_context_idx = main_ctx_queue_->Get();
    for (size_t stage_idx = 0; stage_idx < total_stages_; stage_idx++) {
      stage_contexts_idx[stage_idx] = stage_cur_ctx_queues_[stage_idx]->Get();
    }
  }

  virtual void MainSchedule(
      size_t& payload_matrix_idx, size_t& event_idx, size_t& buffer_binding_idx,
      int& context_idx, size_t& max_shape_request_id,
      std::array<PayloadVector, PAYLOAD_MATRIX_SIZE>& payload_matrix,
      std::vector<std::unique_ptr<Batch>>& batches,
      TRITONBACKEND_Request** requests, const size_t& requests_count) = 0;

  virtual void StageSchedule(
      const size_t& stage_idx, const size_t& payload_matrix_idx,
      int& stage_context_idx, const size_t& total_stages,
      std::array<PayloadVector, PAYLOAD_MATRIX_SIZE>& payload_matrix,
      std::vector<std::unique_ptr<Batch>>& batches,
      std::unique_ptr<Batch>& batch) = 0;

  virtual void UpdateMainContext(const int& context_idx) {}
  virtual void UpdateStageContext(const size_t& stage_idx,
                                  const int& used_context_idx) {
    // dbg("Put context: " + std::to_string(used_context_idx) +
    //     " for stage: " + std::to_string(stage_idx - 1));
    stage_prev_ctx_queues_[stage_idx]->Put(used_context_idx);
  }

  virtual void UpdateCompleteContext(const int& context_idx) {
    // dbg("Put context: " + std::to_string(context_idx) +
    //     " for stage: " + std::to_string(total_stages_ - 1));
    complete_ctx_queue_->Put(context_idx);
  }

  virtual void GetMainTargetContext(const int& target_context_idx,
                                    int& available_context_idx) {
    // dbg("Fetching context: " + std::to_string(target_context_idx) +
    //     " for Main thread");
    available_context_idx = main_ctx_queue_->Get();
    if (target_context_idx == -1) {
      // dbg("Got context: " + std::to_string(available_context_idx) +
      //     " for Main context");
      return;
    }
    while (target_context_idx != available_context_idx) {
      main_ctx_queue_->Put(available_context_idx);
      available_context_idx = main_ctx_queue_->Get();
    }
    // dbg("Got context: " + std::to_string(available_context_idx) +
    //     " for Main context");
  }

  virtual void GetStageTargetContext(const size_t& stage_idx,
                                     const int& target_context_idx,
                                     int& available_context_idx) {
    // dbg("Fetching context: " + std::to_string(target_context_idx) +
    //     " for stage: " + std::to_string(stage_idx));
    available_context_idx = stage_cur_ctx_queues_[stage_idx]->Get();
    if (target_context_idx == -1) {
      // dbg("Got context: " + std::to_string(available_context_idx) +
      //     " for stage: " + std::to_string(stage_idx));
      return;
    }
    while (target_context_idx != available_context_idx) {
      stage_cur_ctx_queues_[stage_idx]->Put(available_context_idx);
      available_context_idx = stage_cur_ctx_queues_[stage_idx]->Get();
    }
    // dbg("Got context: " + std::to_string(available_context_idx) +
    //     " for stage: " + std::to_string(stage_idx));
  }

  const size_t& MaxAllowedBatchSize() { return max_allowed_batch_size_; }
  const size_t& TotalStages() { return total_stages_; }

  virtual void SetBindingDimensions(
      const size_t& stage_idx, std::unique_ptr<Batch>& batch,
      std::shared_ptr<nvinfer1::IExecutionContext>& context) = 0;

  void Set1DBindingDimensions(
      const size_t& stage_idx,
      std::shared_ptr<nvinfer1::IExecutionContext>& context, const int& dim_0) {
    auto& stage_input_bindings_shape = stage_input_bindings_shape_[stage_idx];
    for (auto& shape : stage_input_bindings_shape) {
      shape.second.d[0] = dim_0;
      context->setBindingDimensions(shape.first, shape.second);
    }
  }
  void Set2DBindingDimensions(
      const size_t& stage_idx,
      std::shared_ptr<nvinfer1::IExecutionContext>& context,
      const int& dim_0 = 0, const int& dim_1 = 0) {
    auto& stage_input_bindings_shape = stage_input_bindings_shape_[stage_idx];
    for (auto& shape : stage_input_bindings_shape) {
      shape.second.d[0] = dim_0;
      shape.second.d[1] = dim_1;
      context->setBindingDimensions(shape.first, shape.second);
    }
  }
  void Set3DBindingDimensions(
      const size_t& stage_idx,
      std::shared_ptr<nvinfer1::IExecutionContext>& context,
      const int& dim_0 = 0, const int& dim_1 = 0, const int& dim_2 = 0) {
    auto& stage_input_bindings_shape = stage_input_bindings_shape_[stage_idx];
    for (auto& shape : stage_input_bindings_shape) {
      shape.second.d[0] = dim_0;
      shape.second.d[1] = dim_1;
      shape.second.d[2] = dim_2;
      context->setBindingDimensions(shape.first, shape.second);
    }
  }
  std::unordered_map<int, nvinfer1::Dims>& GetStageInputBindingsShape(
      const size_t& stage_idx) {
    return stage_input_bindings_shape_[stage_idx];
  }

 protected:
  size_t total_stages_;
  int64_t buffer_set_size_;
  triton::common::SyncQueue<int>* main_ctx_queue_;
  std::vector<triton::common::SyncQueue<int>*> stage_cur_ctx_queues_;
  std::vector<triton::common::SyncQueue<int>*> stage_prev_ctx_queues_;
  triton::common::SyncQueue<int>* complete_ctx_queue_;

  size_t max_allowed_batch_size_;
  std::vector<std::unordered_map<int, nvinfer1::Dims>>
      stage_input_bindings_shape_;
};

// the normal scheduler
class NomalScheduler : public Scheduler {
 public:
  NomalScheduler(const size_t& total_stages, const size_t& buffer_set_size,
                 const size_t& max_allowed_batch_size)
      : Scheduler(total_stages, buffer_set_size, max_allowed_batch_size) {}

  void MainSchedule(
      size_t& payload_matrix_idx, size_t& event_idx, size_t& buffer_binding_idx,
      int& context_idx, size_t& max_shape_request_id,
      std::array<PayloadVector, PAYLOAD_MATRIX_SIZE>& payload_matrix,
      std::vector<std::unique_ptr<Batch>>& batches,
      TRITONBACKEND_Request** requests, const size_t& requests_count) override {
    // dbg("Main schedule of Normal Schedule");
    size_t start_request_idx = 0, end_request_idx = 0;
    if (payload_matrix[payload_matrix_idx].total_request_count_ > 0) {
      payload_matrix_idx = (payload_matrix_idx + 1) % PAYLOAD_MATRIX_SIZE;
      event_idx = (event_idx + 1) % buffer_set_size_;
      buffer_binding_idx = (buffer_binding_idx + 1) % buffer_set_size_;
    }
    end_request_idx = payload_matrix[payload_matrix_idx].New(
        event_idx, buffer_binding_idx, requests, requests_count, context_idx);
    batches.emplace_back(new Batch(payload_matrix_idx, 0, start_request_idx,
                                   end_request_idx, event_idx, context_idx,
                                   -1));
    context_idx = -1;
    return;
  }

  void StageSchedule(
      const size_t& stage_idx, const size_t& payload_matrix_idx,
      int& stage_context_idx, const size_t& total_stages,
      std::array<PayloadVector, PAYLOAD_MATRIX_SIZE>& payload_matrix,
      std::vector<std::unique_ptr<Batch>>& batches,
      std::unique_ptr<Batch>& batch) override {
    // dbg("Normal Stage Schedule at stage: " + std::to_string(stage_idx));
    batches.emplace_back(new Batch(payload_matrix_idx, 0, batch->start_,
                                   batch->end_, batch->event_idx_,
                                   stage_context_idx, -1));
    stage_context_idx = -1;
  }

  void SetBindingDimensions(
      const size_t& stage_idx, std::unique_ptr<Batch>& batch,
      std::shared_ptr<nvinfer1::IExecutionContext>& context) override {
    Set1DBindingDimensions(stage_idx, context, batch->end_ - batch->start_);
  }
};

class NomalInputScheduler : public Scheduler {
 public:
  NomalInputScheduler(const size_t& total_stages, const size_t& buffer_set_size,
                      const size_t& max_allowed_batch_size)
      : Scheduler(total_stages, buffer_set_size, max_allowed_batch_size) {}

  void MainSchedule(
      size_t& payload_matrix_idx, size_t& event_idx, size_t& buffer_binding_idx,
      int& context_idx, size_t& max_shape_request_id,
      std::array<PayloadVector, PAYLOAD_MATRIX_SIZE>& payload_matrix,
      std::vector<std::unique_ptr<Batch>>& batches,
      TRITONBACKEND_Request** requests, const size_t& requests_count) override {
    // dbg("Main schedule of Normal Input Schedule");
    size_t start_request_idx = 0, end_request_idx = 0;
    if (payload_matrix[payload_matrix_idx].total_request_count_ > 0) {
      payload_matrix_idx = (payload_matrix_idx + 1) % PAYLOAD_MATRIX_SIZE;
      event_idx = (event_idx + 1) % buffer_set_size_;
      buffer_binding_idx = (buffer_binding_idx + 1) % buffer_set_size_;
    }
    end_request_idx = payload_matrix[payload_matrix_idx].New(
        event_idx, buffer_binding_idx, requests, requests_count, context_idx);
    int64_t seq_len = 2;
    for (size_t request_idx = 0; request_idx < requests_count; request_idx++) {
      TRITONBACKEND_Input* input;
      const int64_t* shape;
      TRITONBACKEND_RequestInputByIndex(requests[request_idx], 0 /* index */,
                                        &input);
      TRITONBACKEND_InputProperties(input, nullptr, nullptr, &shape, nullptr,
                                    nullptr, nullptr);
      if (shape[1] > seq_len) {
        seq_len = shape[1];
        max_shape_request_id = request_idx;
      }
      dbg(seq_len);
      // dbg(requests[request_idx]);
    }
    batches.emplace_back(new Batch(payload_matrix_idx, 0, start_request_idx,
                                   end_request_idx, event_idx, context_idx,
                                   seq_len));
    context_idx = -1;
    return;
  }

  void StageSchedule(
      const size_t& stage_idx, const size_t& payload_matrix_idx,
      int& stage_context_idx, const size_t& total_stages,
      std::array<PayloadVector, PAYLOAD_MATRIX_SIZE>& payload_matrix,
      std::vector<std::unique_ptr<Batch>>& batches,
      std::unique_ptr<Batch>& batch) override {
    // dbg("Normal Input Stage Schedule");
    batches.emplace_back(new Batch(payload_matrix_idx, 0, batch->start_,
                                   batch->end_, batch->event_idx_,
                                   stage_context_idx, batch->len_));
    stage_context_idx = -1;
  }

  void SetBindingDimensions(
      const size_t& stage_idx, std::unique_ptr<Batch>& batch,
      std::shared_ptr<nvinfer1::IExecutionContext>& context) override {
    Set2DBindingDimensions(stage_idx, context, batch->len_,
                           batch->end_ - batch->start_);
  }
};

// the load diversity scheduler
class LoadDVAScheduler : public Scheduler {
 public:
  LoadDVAScheduler(const size_t& total_stages, const size_t& buffer_set_size,
                   const size_t& max_allowed_batch_size,
                   const size_t& max_wait_time, const size_t& max_stage_num)
      : Scheduler(total_stages, buffer_set_size, max_allowed_batch_size),
        max_wait_time_(max_wait_time),
        max_stage_num_(max_stage_num) {
    dbg(max_wait_time_);
  }

  void MainSchedule(
      size_t& payload_matrix_idx, size_t& event_idx, size_t& buffer_binding_idx,
      int& context_idx, size_t& max_shape_request_id,
      std::array<PayloadVector, PAYLOAD_MATRIX_SIZE>& payload_matrix,
      std::vector<std::unique_ptr<Batch>>& batches,
      TRITONBACKEND_Request** requests, const size_t& requests_count) override {
    // dbg("Load Diversity Schedule");
    LEGO_DECL_TIMESTAMP(t_now);
    size_t start_request_idx = 0, end_request_idx = 0;
    auto& cur_payload_vector = payload_matrix[payload_matrix_idx];
    if (cur_payload_vector.total_request_count_ > 0) {  // check current vector
      if (cur_payload_vector.total_request_count_ + requests_count <
          MaxAllowedBatchSize()) {  // check the max batch size
        if (cur_payload_vector.handled_stages_ <
            max_stage_num_) {  // check the max stage
          if (t_now - cur_payload_vector.compute_start_ns_ <
              max_wait_time_) {  // check the max wait time
            start_request_idx = cur_payload_vector.total_request_count_;
            end_request_idx =
                cur_payload_vector.Stretch(requests, requests_count);
            if (end_request_idx > start_request_idx) {
              batches.emplace_back(new Batch(payload_matrix_idx, 0,
                                             start_request_idx, end_request_idx,
                                             event_idx, context_idx, -1));
              context_idx = -1;
              return;
            }
          }
        }
      }
      payload_matrix_idx = (payload_matrix_idx + 1) % PAYLOAD_MATRIX_SIZE;
      event_idx = (event_idx + 1) % buffer_set_size_;
      buffer_binding_idx = (buffer_binding_idx + 1) % buffer_set_size_;
      end_request_idx = payload_matrix[payload_matrix_idx].New(
          event_idx, buffer_binding_idx, requests, requests_count, context_idx);
      batches.emplace_back(new Batch(payload_matrix_idx, 0, 0, end_request_idx,
                                     event_idx, context_idx, -1));
      context_idx = -1;
      return;
    }
    // event_idx = (event_idx + 1) % EVENT_SET_SIZE;
    // buffer_binding_idx = (buffer_binding_idx + 1) % BUFFER_BINDING_SET_SIZE;
    end_request_idx = payload_matrix[payload_matrix_idx].New(
        event_idx, buffer_binding_idx, requests, requests_count, context_idx);
    batches.emplace_back(new Batch(payload_matrix_idx, 0, 0, end_request_idx,
                                   event_idx, context_idx, -1));
    context_idx = -1;
    return;
  }

  void StageSchedule(
      const size_t& stage_idx, const size_t& payload_matrix_idx,
      int& stage_context_idx, const size_t& total_stages,
      std::array<PayloadVector, PAYLOAD_MATRIX_SIZE>& payload_matrix,
      std::vector<std::unique_ptr<Batch>>& batches,
      std::unique_ptr<Batch>& batch) override {
    // dbg("Load Diversity Stage Schedule");
    auto start_request_idx =
        payload_matrix[payload_matrix_idx].Forward(stage_idx + 1, batch->end_);
    if (start_request_idx != -1) {
      // std::cout << "start_request_idx: " << start_request_idx
      //           << "end: " << batch->end_ << std::endl;
      batches.emplace_back(new Batch(payload_matrix_idx, 0, start_request_idx,
                                     batch->end_, batch->event_idx_,
                                     stage_context_idx, -1));
      stage_context_idx = -1;
      return;
    }
    return;
  }

  void SetBindingDimensions(
      const size_t& stage_idx, std::unique_ptr<Batch>& batch,
      std::shared_ptr<nvinfer1::IExecutionContext>& context) override {
    Set1DBindingDimensions(stage_idx, context, batch->end_ - batch->start_);
  }

 protected:
  size_t max_wait_time_;
  size_t max_stage_num_;
};

// the input diversity scheduler
class InputDVAScheduler final : public Scheduler {
 public:
  InputDVAScheduler(const size_t& total_stages, const size_t& buffer_set_size,
                    const size_t& max_allowed_batch_size,
                    const std::vector<size_t>& allowed_sequence_lengths)
      : Scheduler(total_stages, buffer_set_size, max_allowed_batch_size),
        allowed_sequence_lengths_(allowed_sequence_lengths) {}

  void MainSchedule(
      size_t& payload_matrix_idx, size_t& event_idx, size_t& buffer_binding_idx,
      int& context_idx, size_t& max_shape_request_id,
      std::array<PayloadVector, PAYLOAD_MATRIX_SIZE>& payload_matrix,
      std::vector<std::unique_ptr<Batch>>& batches,
      TRITONBACKEND_Request** requests, const size_t& requests_count) override {
    SortRequests(requests, requests_count);
    for (auto& s_requests : sorted_requests_) {
      dbg(s_requests.first);
      if (!s_requests.second.empty()) {
        if (payload_matrix[payload_matrix_idx].total_request_count_ > 0) {
          payload_matrix_idx = (payload_matrix_idx + 1) % PAYLOAD_MATRIX_SIZE;
          event_idx = (event_idx + 1) % buffer_set_size_;
          buffer_binding_idx = (buffer_binding_idx + 1) % buffer_set_size_;
        }
        payload_matrix[payload_matrix_idx].New(
            event_idx, buffer_binding_idx, s_requests.second.data(),
            s_requests.second.size(), context_idx);
        batches.emplace_back(new Batch(payload_matrix_idx, 0, 0,
                                       s_requests.second.size(), event_idx,
                                       context_idx, s_requests.first));
      }
    }
    context_idx = -1;
    return;
  }

  void StageSchedule(
      const size_t& stage_idx, const size_t& payload_matrix_idx,
      int& stage_context_idx, const size_t& total_stages,
      std::array<PayloadVector, PAYLOAD_MATRIX_SIZE>& payload_matrix,
      std::vector<std::unique_ptr<Batch>>& batches,
      std::unique_ptr<Batch>& batch) override {
    batches.emplace_back(new Batch(payload_matrix_idx, 0, 0, batch->end_,
                                   batch->event_idx_, stage_context_idx,
                                   batch->len_));
    stage_context_idx = -1;
    return;
  }

  void SetBindingDimensions(
      const size_t& stage_idx, std::unique_ptr<Batch>& batch,
      std::shared_ptr<nvinfer1::IExecutionContext>& context) override {
    Set2DBindingDimensions(stage_idx, context, batch->len_,
                           batch->end_ - batch->start_);
  }

  size_t SequenceLengthMap(const size_t& sequence_length) {
    for (auto seq_len_map : allowed_sequence_lengths_) {
      if (sequence_length <= seq_len_map) {
        return seq_len_map;
      }
    }
    return 0;
  }

  void SortRequests(TRITONBACKEND_Request** requests,
                    const size_t& requests_count) {
    sorted_requests_.clear();
    for (size_t request_idx = 0; request_idx < requests_count; request_idx++) {
      auto& request = requests[request_idx];
      TRITONBACKEND_Input* input;
      const int64_t* shape;
      TRITONBACKEND_RequestInputByIndex(request, 0 /* index */, &input);
      TRITONBACKEND_InputProperties(input, nullptr, nullptr, &shape, nullptr,
                                    nullptr, nullptr);
      auto seq_len_map = SequenceLengthMap(shape[1]);
      sorted_requests_[seq_len_map].emplace_back(request);
    }
  }

 protected:
  std::map<size_t, std::vector<TRITONBACKEND_Request*>> sorted_requests_;
  std::vector<size_t> allowed_sequence_lengths_;
};

// the operator diversity scheduler
class HypbridDVAScheduler final : public LoadDVAScheduler {
 public:
  HypbridDVAScheduler(const size_t& total_stages, const size_t& buffer_set_size,
                      const size_t& max_allowed_batch_size,
                      const std::vector<size_t>& stage_favored_batch_sizes,
                      const size_t& max_wait_time, const size_t& max_stage_num)
      : LoadDVAScheduler(total_stages, buffer_set_size, max_allowed_batch_size,
                         max_wait_time, max_stage_num),
        stage_favored_batch_sizes_(stage_favored_batch_sizes) {
    stage_cond_ = TotalStages() - 2;
    split_stage_idx_ = TotalStages();
  }

  void MainSchedule(
      size_t& payload_matrix_idx, size_t& event_idx, size_t& buffer_binding_idx,
      int& context_idx, size_t& max_shape_request_id,
      std::array<PayloadVector, PAYLOAD_MATRIX_SIZE>& payload_matrix,

      std::vector<std::unique_ptr<Batch>>& batches,
      TRITONBACKEND_Request** requests, const size_t& requests_count) override {
    dbg("Hybrid Diversity Scheduler");
    LEGO_DECL_TIMESTAMP(t_now);
    size_t start_request_idx = 0, end_request_idx = 0;
    auto& cur_payload_vector = payload_matrix[payload_matrix_idx];
    if (cur_payload_vector.total_request_count_ > 0) {  // check current vector
      if (!cur_payload_vector.disable_merge_) {  // check allow requests merge
        if (cur_payload_vector.total_request_count_ + requests_count <
            MaxAllowedBatchSize()) {  // check the max batch size
          if (cur_payload_vector.handled_stages_ <
              max_stage_num_) {  // check the max stage
            if (t_now - cur_payload_vector.compute_start_ns_ <
                max_wait_time_) {  // check the max wait time
              start_request_idx = cur_payload_vector.total_request_count_;
              end_request_idx =
                  cur_payload_vector.Stretch(requests, requests_count);
              if (end_request_idx > start_request_idx) {
                batches.emplace_back(
                    new Batch(payload_matrix_idx, 0, start_request_idx,
                              end_request_idx, event_idx, context_idx, -1));
                return;
              }
            }
          }
        }
      } else {
        payload_matrix_idx = (payload_matrix_idx + 1) % PAYLOAD_MATRIX_SIZE;
        event_idx = (event_idx + 1) % buffer_set_size_;
        buffer_binding_idx = (buffer_binding_idx + 1) % buffer_set_size_;
        end_request_idx = payload_matrix[payload_matrix_idx].New(
            event_idx, buffer_binding_idx, requests, requests_count,
            context_idx);
        batches.emplace_back(nullptr);
        return;
      }
      payload_matrix_idx = (payload_matrix_idx + 1) % PAYLOAD_MATRIX_SIZE;
      event_idx = (event_idx + 1) % buffer_set_size_;
      buffer_binding_idx = (buffer_binding_idx + 1) % buffer_set_size_;
      end_request_idx = payload_matrix[payload_matrix_idx].New(
          event_idx, buffer_binding_idx, requests, requests_count, context_idx);
      batches.emplace_back(new Batch(payload_matrix_idx, 0, start_request_idx,
                                     end_request_idx, event_idx, context_idx,
                                     -1));
      return;
    }
    end_request_idx = payload_matrix[payload_matrix_idx].New(
        event_idx, buffer_binding_idx, requests, requests_count, context_idx);
    batches.emplace_back(new Batch(payload_matrix_idx, 0, start_request_idx,
                                   end_request_idx, event_idx, -1));
    return;
  }

  void StageSchedule(
      const size_t& stage_idx, const size_t& payload_matrix_idx,
      int& stage_context_idx, const size_t& total_stages,
      std::array<PayloadVector, PAYLOAD_MATRIX_SIZE>& payload_matrix,
      std::vector<std::unique_ptr<Batch>>& batches,
      std::unique_ptr<Batch>& batch) override {
    dbg("Hybrid Diversity Stage Scheduler");
    auto start_request_idx =
        payload_matrix[payload_matrix_idx].Forward(stage_idx + 1, batch->end_);
    if (start_request_idx != -1) {
      auto bs_favor =
          2 * stage_favored_batch_sizes_[stage_idx + 1];  // apply to -2 stage
      if (start_request_idx == 0 && batch->end_ >= 2 * bs_favor &&
          stage_idx < stage_cond_) {
        batches = payload_matrix[payload_matrix_idx].Split(
            stage_idx, batch->payload_vector_idx_, bs_favor, stage_context_idx);
        if (split_stage_idx_ == TotalStages()) {
          split_stage_idx_ = stage_idx;
        }
        return;
      }
      batches.emplace_back(new Batch(payload_matrix_idx, 0, start_request_idx,
                                     batch->end_, batch->event_idx_,
                                     stage_context_idx, -1));
    }
  }

  void SetBindingDimensions(
      const size_t& stage_idx, std::unique_ptr<Batch>& batch,
      std::shared_ptr<nvinfer1::IExecutionContext>& context) override {
    Set1DBindingDimensions(stage_idx, context, batch->end_ - batch->start_);
  }

 private:
  std::vector<size_t> stage_favored_batch_sizes_;
  size_t split_stage_idx_;
  size_t stage_cond_;
};

class OperatorDVAScheduler final : public Scheduler {
 public:
  OperatorDVAScheduler(const size_t& total_stages,
                       const size_t& buffer_set_size,
                       const size_t& max_allowed_batch_size,
                       const std::vector<size_t>& stage_favored_batch_sizes,
                       const size_t& max_wait_time, const size_t& max_stage_num)
      : Scheduler(total_stages, buffer_set_size, max_allowed_batch_size),
        stage_favored_batch_sizes_(stage_favored_batch_sizes) {
    stage_cond_ = TotalStages() - 2;
    split_stage_idx_ = TotalStages();
  }

  void MainSchedule(
      size_t& payload_matrix_idx, size_t& event_idx, size_t& buffer_binding_idx,
      int& context_idx, size_t& max_shape_request_id,
      std::array<PayloadVector, PAYLOAD_MATRIX_SIZE>& payload_matrix,
      std::vector<std::unique_ptr<Batch>>& batches,
      TRITONBACKEND_Request** requests, const size_t& requests_count) override {
    dbg("Operator Diversity Scheduler");
    size_t start_request_idx = 0, end_request_idx = 0;
    if (payload_matrix[payload_matrix_idx].total_request_count_ > 0) {
      payload_matrix_idx = (payload_matrix_idx + 1) % PAYLOAD_MATRIX_SIZE;
      event_idx = (event_idx + 1) % buffer_set_size_;
      buffer_binding_idx = (buffer_binding_idx + 1) % buffer_set_size_;
    }
    end_request_idx = payload_matrix[payload_matrix_idx].New(
        event_idx, buffer_binding_idx, requests, requests_count, context_idx);
    batches.emplace_back(new Batch(payload_matrix_idx, 0, start_request_idx,
                                   end_request_idx, event_idx, context_idx,
                                   -1));
    return;
  }

  void StageSchedule(
      const size_t& stage_idx, const size_t& payload_matrix_idx,
      int& stage_context_idx, const size_t& total_stages,
      std::array<PayloadVector, PAYLOAD_MATRIX_SIZE>& payload_matrix,
      std::vector<std::unique_ptr<Batch>>& batches,
      std::unique_ptr<Batch>& batch) override {
    dbg("Operator Diversity Stage Scheduler");
    auto bs_favor =
        2 * stage_favored_batch_sizes_[stage_idx + 1];  // apply to -2 stage
    if (batch->end_ >= 2 * bs_favor && stage_idx < stage_cond_) {
      batches = payload_matrix[payload_matrix_idx].Split(
          stage_idx, batch->payload_vector_idx_, bs_favor, stage_context_idx);
      if (split_stage_idx_ == TotalStages()) {
        split_stage_idx_ = stage_idx;
      }
      return;
    }
    batches.emplace_back(new Batch(payload_matrix_idx, 0, 0, batch->end_,
                                   batch->event_idx_, stage_context_idx, -1));
    return;
  }

  void SetBindingDimensions(
      const size_t& stage_idx, std::unique_ptr<Batch>& batch,
      std::shared_ptr<nvinfer1::IExecutionContext>& context) override {
    Set1DBindingDimensions(stage_idx, context, batch->end_ - batch->start_);
  }

 private:
  std::vector<size_t> stage_favored_batch_sizes_;
  size_t split_stage_idx_;
  size_t stage_cond_;
};
}}}  // namespace triton::backend::lego
