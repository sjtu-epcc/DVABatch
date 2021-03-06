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

#include "lego_model_instance.h"

namespace triton { namespace backend { namespace lego {

LegoModelInstance::LegoModelInstance(
    LegoModel* lego_model, TRITONBACKEND_ModelInstance* triton_model_instance)
    : BackendModelInstance(lego_model, triton_model_instance),
      lego_model_(lego_model) {
  uint32_t profile_count;
  THROW_IF_BACKEND_INSTANCE_ERROR(TRITONBACKEND_ModelInstanceProfileCount(
      triton_model_instance, &profile_count));
  for (uint32_t index = 0; index < profile_count; index++) {
    const char* profile_name;
    THROW_IF_BACKEND_INSTANCE_ERROR(TRITONBACKEND_ModelInstanceProfileName(
        triton_model_instance, index, &profile_name));
    profile_names_.insert(profile_name);
  }
}

}}}  // namespace triton::backend::lego
