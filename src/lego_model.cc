/*!
 * Motto: Were It to Benefit My Country, I Would Lay Down My Life!
 * \file: /lego_model.cc
 * \brief:
 * Author: raphael hao
 */
#include "lego_model.h"

#include "dbg.h"

namespace triton { namespace backend { namespace lego {

LegoModel::Priority ParsePriority(const std::string& priority) {
  LegoModel::Priority lego_priority = LegoModel::Priority::DEFAULT;

  if (priority.compare("PRIORITY_MAX") == 0) {
    lego_priority = LegoModel::Priority::MAX;
  } else if (priority.compare("PRIORITY_MIN") == 0) {
    lego_priority = LegoModel::Priority::MIN;
  } else if (priority.compare("PRIORITY_DEFAULT") != 0) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_WARN,
        (std::string(
             "TRT backend does not support the provided stream priority '") +
         priority + "', using 'PRIORITY_DEFAULT'.")
            .c_str());
  }

  return lego_priority;
}

LegoModel::LegoModel(TRITONBACKEND_Model* triton_model)
    : BackendModel(triton_model),
      priority_(Priority::DEFAULT),
      gather_kernel_buffer_threshold_(0),
      separate_output_stream_(false),
      eager_batching_(false),
      busy_wait_events_(false) {
  ParseModelConfig();
}

void LegoModel::ParseModelConfig() {
  if (model_config_.Find("lego_config", &lego_config_)) {
    lego_config_.MemberAsUInt("total_stages", &total_stages_);
    lego_config_.MemberAsUInt("total_buffers", &total_buffers_);
    lego_config_.MemberAsString("log_dir", &log_dir_);
    bool log_dir_exists = false;
    IsDirectory(log_dir_, &log_dir_exists);
    if (!log_dir_exists) {
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR,
                  ("Log directory: " + log_dir_ + "does not exist.").c_str());
    }
    triton::common::TritonJson::Value stage_filenames;
    lego_config_.MemberAsArray("stage_filenames", &stage_filenames);
    stage_filenames_.resize(stage_filenames.ArraySize());
    for (size_t i = 0; i < stage_filenames.ArraySize(); i++) {
      stage_filenames.IndexAsString(i, &stage_filenames_[i]);
    }

    stages_inputs_.reserve(total_stages_);
    triton::common::TritonJson::Value stages_inputs;
    lego_config_.MemberAsArray("stages_inputs", &stages_inputs);
    assert(stages_inputs.ArraySize() == total_stages_);
    printf("parsing the stages_inputs\n");
    for (size_t i = 0; i < stages_inputs.ArraySize(); i++) {
      triton::common::TritonJson::Value stage_inputs_with_name;
      stages_inputs.IndexAsObject(i, &stage_inputs_with_name);
      triton::common::TritonJson::Value stage_inputs;
      stage_inputs_with_name.MemberAsArray("input", &stage_inputs);
      std::unordered_map<std::string, StageInOut> inputs;
      for (size_t j = 0; j < stage_inputs.ArraySize(); j++) {
        std::string input_name;
        std::string input_data_type;
        std::vector<int64_t> input_dims;
        size_t input_id;
        triton::common::TritonJson::Value stage_input;
        stage_inputs.IndexAsObject(j, &stage_input);
        // get stage input name
        stage_input.MemberAsString("name", &input_name);
        // get stage input data type
        stage_input.MemberAsString("data_type", &input_data_type);
        // get stage input dims
        triton::common::TritonJson::Value stage_input_dims;
        stage_input.MemberAsArray("dims", &stage_input_dims);
        input_dims.resize(stage_input_dims.ArraySize());
        for (size_t k = 0; k < stage_input_dims.ArraySize(); k++) {
          stage_input_dims.IndexAsInt(k, &input_dims[k]);
        }
        // get stage input buffer id
        stage_input.MemberAsUInt("id", &input_id);
        inputs.emplace(input_name,
                       std::make_tuple(input_data_type, input_dims, input_id));
      }
      stages_inputs_.emplace(i, inputs);
    }
    stages_outputs_.reserve(total_stages_);
    triton::common::TritonJson::Value stages_outputs;
    lego_config_.MemberAsArray("stages_outputs", &stages_outputs);
    assert(stages_outputs.ArraySize() == total_stages_);
    for (size_t i = 0; i < stages_outputs.ArraySize(); i++) {
      triton::common::TritonJson::Value stage_outputs_with_name;
      stages_outputs.IndexAsObject(i, &stage_outputs_with_name);
      triton::common::TritonJson::Value stage_outputs;
      stage_outputs_with_name.MemberAsArray("output", &stage_outputs);
      std::unordered_map<std::string, StageInOut> outputs;
      for (size_t j = 0; j < stage_outputs.ArraySize(); j++) {
        std::string output_name;
        std::string output_data_type;
        std::vector<int64_t> output_dims;
        size_t output_id;
        triton::common::TritonJson::Value stage_output;
        stage_outputs.IndexAsObject(j, &stage_output);
        // get stage output name
        stage_output.MemberAsString("name", &output_name);
        // get stage output data type
        stage_output.MemberAsString("data_type", &output_data_type);
        // get stage output dims
        triton::common::TritonJson::Value stage_output_dims;
        stage_output.MemberAsArray("dims", &stage_output_dims);
        output_dims.resize(stage_output_dims.ArraySize());
        for (size_t k = 0; k < stage_output_dims.ArraySize(); k++) {
          stage_output_dims.IndexAsInt(k, &output_dims[k]);
        }
        // get stage output buffer id
        stage_output.MemberAsUInt("id", &output_id);
        outputs.emplace(output_name, std::make_tuple(output_data_type,
                                                     output_dims, output_id));
      }
      stages_outputs_.emplace(i, outputs);
    }
    dbg("parsing the lego scheduling information\n");
    triton::common::TritonJson::Value schedule_option;
    if (lego_config_.Find("schedule_option", &schedule_option)) {
      std::string scheduler_type;
      schedule_option.MemberAsString("scheduler_type", &scheduler_type);
      // generate the log filie path
      dbg(scheduler_type);
      log_filepath_ =
          JoinPath({log_dir_, Name() + "_" + scheduler_type + ".csv"});
      dbg(log_filepath_);
      if (scheduler_type == "NORMAL") {
        scheduler_type_ = SchedulerType::NORMAL;
      } else if (scheduler_type == "NORMAL_INPUT") {
        scheduler_type_ = SchedulerType::NORMAL_INPUT;
      } else if (scheduler_type == "LOAD_DVA") {
        scheduler_type_ = SchedulerType::LOAD_DVA;
      } else if (scheduler_type == "INPUT_DVA") {
        scheduler_type_ = SchedulerType::INPUT_DVA;
      } else if (scheduler_type == "OPERATOR_DVA") {
        scheduler_type_ = SchedulerType::OPERATOR_DVA;
      } else {
        scheduler_type_ = SchedulerType::UNKNOWN;
      }
      schedule_option.MemberAsInt("buffer_set_size", &buffer_set_size_);
      schedule_option.MemberAsBool("busy_wait_events", &busy_wait_events_);
      if (scheduler_type_ == SchedulerType::OPERATOR_DVA) {
        triton::common::TritonJson::Value favor_per_stage;
        schedule_option.MemberAsArray("favor_per_stage", &favor_per_stage);
        assert(favor_per_stage.ArraySize() == total_stages_);
        favor_per_stage_.resize(favor_per_stage.ArraySize());
        for (size_t i = 0; i < favor_per_stage.ArraySize(); i++) {
          favor_per_stage.IndexAsUInt(i, &favor_per_stage_[i]);
        }
      }
      if (scheduler_type_ == SchedulerType::INPUT_DVA) {
        triton::common::TritonJson::Value preferred_seq_len;
        schedule_option.MemberAsArray("preferred_seq_len", &preferred_seq_len);
        preferred_seq_len_.resize(preferred_seq_len.ArraySize());
        for (size_t i = 0; i < preferred_seq_len.ArraySize(); i++) {
          preferred_seq_len.IndexAsUInt(i, &preferred_seq_len_[i]);
        }
      }
      if (scheduler_type_ == SchedulerType::LOAD_DVA ||
          scheduler_type_ == SchedulerType::OPERATOR_DVA) {
        schedule_option.MemberAsUInt("max_allow_merge_stage",
                                     &max_allow_merge_stage_);
        schedule_option.MemberAsUInt("max_allow_merge_batchsize",
                                     &max_allow_merge_batchsize_);
        schedule_option.MemberAsUInt("max_allow_merge_microseconds",
                                     &max_allow_merge_microseconds_);
      }
    }
  } else {
    LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "No lego configuration is specified");
  }
}

}}}  // namespace triton::backend::lego
