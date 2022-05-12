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
//  * Neither the name of NVIDIA CORPORATION nor the names of its
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
#pragma once

#include "triton/backend/backend_model.h"

namespace triton { namespace backend { namespace lego {
using StageInOut = std::tuple<std::string, std::vector<int64_t>, size_t>;

class LegoModel : public BackendModel {
 public:
  LegoModel(TRITONBACKEND_Model* triton_model);
  virtual ~LegoModel() = default;

  void ParseModelConfig();

  // The model configuration.
  enum class Priority { DEFAULT = 0, MIN = 1, MAX = 2 };
  enum class SchedulerType {
    NORMAL = 0,
    NORMAL_INPUT = 1,
    LOAD_DVA = 2,
    INPUT_DVA = 3,
    OPERATOR_DVA = 4,
    UNKNOWN = 5
  };

  Priority ModelPriority() { return priority_; }
  size_t GatherKernelBufferThreshold() {
    return gather_kernel_buffer_threshold_;
  }
  bool BusyWaitEvents() { return busy_wait_events_; }

  triton::common::TritonJson::Value& LegoConfig() { return lego_config_; }
  size_t TotalStages() { return total_stages_; }
  size_t TotalBuffers() { return total_buffers_; }
  int64_t BufferSetSize() { return buffer_set_size_; }
  std::string LogFilePath() { return log_filepath_; }
  std::string LogDirectory() { return log_dir_; }
  // std::vector<size_t> EnginesPerStage() { return engines_per_stage_; }
  std::vector<std::string> StageFilenames() { return stage_filenames_; }
  std::unordered_map<std::string, StageInOut> StagesInputs(size_t stage_id) {
    return stages_inputs_[stage_id];
  }
  std::unordered_map<std::string, StageInOut> StagesOutputs(size_t stage_id) {
    return stages_outputs_[stage_id];
  }
  SchedulerType GetSchedulerType() { return scheduler_type_; }
  std::vector<size_t> FavorPerStage() { return favor_per_stage_; }
  std::vector<size_t> PreferredSeqLen() { return preferred_seq_len_; }
  size_t MaxAllowMergeStage() { return max_allow_merge_stage_; }
  size_t MaxAllowMergeBatchsize() { return max_allow_merge_batchsize_; }
  size_t MaxAllowMergeMicroseconds() { return max_allow_merge_microseconds_; }

 protected:
  Priority priority_;
  size_t gather_kernel_buffer_threshold_;
  bool separate_output_stream_;
  bool eager_batching_;
  bool busy_wait_events_;

  // The lego configuration.
  size_t total_stages_;
  size_t total_buffers_;
  std::vector<std::string> stage_filenames_;
  //< <stage_id>, <buffer id>, dims>
  std::unordered_map<size_t, std::unordered_map<std::string, StageInOut>>
      stages_inputs_;
  std::unordered_map<size_t, std::unordered_map<std::string, StageInOut>>
      stages_outputs_;

  // The lego schedule config.
  SchedulerType scheduler_type_;

  int64_t buffer_set_size_;

  std::vector<size_t> favor_per_stage_;

  std::vector<size_t> preferred_seq_len_;

  size_t max_allow_merge_stage_;
  size_t max_allow_merge_batchsize_;
  size_t max_allow_merge_microseconds_;

  std::string log_dir_;
  std::string log_filepath_;

  common::TritonJson::Value lego_config_;
};

}}}  // namespace triton::backend::lego
