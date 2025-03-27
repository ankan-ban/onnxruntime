// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "nv_provider_factory.h"
#include <atomic>
#include "nv_execution_provider.h"
#include "nv_provider_factory_creator.h"
#include "core/framework/provider_options.h"
#include "core/providers/nv/nv_provider_options.h"
#include "core/providers/nv/nv_execution_provider_custom_ops.h"
#include <string.h>

using namespace onnxruntime;

namespace onnxruntime {

void InitializeRegistry();
void DeleteRegistry();

struct ProviderInfo_Nv_Impl final : ProviderInfo_Nv {
  OrtStatus* GetCurrentGpuDeviceId(_In_ int* device_id) override {
    auto cuda_err = cudaGetDevice(device_id);
    if (cuda_err != cudaSuccess) {
      return CreateStatus(ORT_FAIL, "Failed to get device id.");
    }
    return nullptr;
  }

  OrtStatus* UpdateProviderOptions(void* provider_options, const ProviderOptions& options, bool string_copy) override {
    NvExecutionProviderInfo::UpdateProviderOptions(provider_options, options, string_copy);
    return nullptr;
  }

  OrtStatus* GetTensorRTCustomOpDomainList(std::vector<OrtCustomOpDomain*>& domain_list, const std::string extra_plugin_lib_paths) override {
    common::Status status = CreateTensorRTCustomOpDomainList(domain_list, extra_plugin_lib_paths);
    if (!status.IsOK()) {
      return CreateStatus(ORT_FAIL, "[Nv EP] Can't create custom ops for TRT plugins.");
    }
    return nullptr;
  }

  OrtStatus* ReleaseCustomOpDomainList(std::vector<OrtCustomOpDomain*>& domain_list) override {
    ReleaseTensorRTCustomOpDomainList(domain_list);
    return nullptr;
  }

} g_info;

struct NvProviderFactory : IExecutionProviderFactory {
  NvProviderFactory(const NvExecutionProviderInfo& info) : info_{info} {}
  ~NvProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  NvExecutionProviderInfo info_;
};

std::unique_ptr<IExecutionProvider> NvProviderFactory::CreateProvider() {
  return std::make_unique<NvExecutionProvider>(info_);
}

struct Nv_Provider : Provider {
  void* GetInfo() override { return &g_info; }
  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(int device_id) override {
    NvExecutionProviderInfo info;
    info.device_id = device_id;
    info.has_trt_options = false;

    return std::make_shared<NvProviderFactory>(info);
  }

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtNvProviderOptionsV2*>(provider_options);
    NvExecutionProviderInfo info;
    info.device_id = options.device_id;
    info.has_user_compute_stream = options.has_user_compute_stream != 0;
    info.user_compute_stream = options.user_compute_stream;
    info.has_trt_options = true;
    info.max_partition_iterations = options.nv_max_partition_iterations;
    info.min_subgraph_size = options.nv_min_subgraph_size;
    info.max_workspace_size = options.nv_max_workspace_size;
    info.fp16_enable = options.nv_fp16_enable != 0;
    info.int8_enable = options.nv_int8_enable != 0;
    info.int8_calibration_table_name = options.nv_int8_calibration_table_name == nullptr ? "" : options.nv_int8_calibration_table_name;
    info.int8_use_native_calibration_table = options.nv_int8_use_native_calibration_table != 0;
    info.dla_enable = options.nv_dla_enable != 0;
    info.dla_core = options.nv_dla_core;
    info.dump_subgraphs = options.nv_dump_subgraphs != 0;
    info.engine_cache_enable = options.nv_engine_cache_enable != 0;
    info.engine_cache_path = options.nv_engine_cache_path == nullptr ? "" : options.nv_engine_cache_path;
    info.weight_stripped_engine_enable = options.nv_weight_stripped_engine_enable != 0;
    info.onnx_model_folder_path = options.nv_onnx_model_folder_path == nullptr ? "" : options.nv_onnx_model_folder_path;
    info.engine_decryption_enable = options.nv_engine_decryption_enable != 0;
    info.engine_decryption_lib_path = options.nv_engine_decryption_lib_path == nullptr ? "" : options.nv_engine_decryption_lib_path;
    info.force_sequential_engine_build = options.nv_force_sequential_engine_build != 0;
    info.context_memory_sharing_enable = options.nv_context_memory_sharing_enable != 0;
    info.layer_norm_fp32_fallback = options.nv_layer_norm_fp32_fallback != 0;
    info.timing_cache_enable = options.nv_timing_cache_enable != 0;
    info.timing_cache_path = options.nv_timing_cache_path == nullptr ? "" : options.nv_timing_cache_path;
    info.force_timing_cache = options.nv_force_timing_cache != 0;
    info.detailed_build_log = options.nv_detailed_build_log != 0;
    info.build_heuristics_enable = options.nv_build_heuristics_enable != 0;
    info.sparsity_enable = options.nv_sparsity_enable;
    info.builder_optimization_level = options.nv_builder_optimization_level;
    info.auxiliary_streams = options.nv_auxiliary_streams;
    info.tactic_sources = options.nv_tactic_sources == nullptr ? "" : options.nv_tactic_sources;
    info.extra_plugin_lib_paths = options.nv_extra_plugin_lib_paths == nullptr ? "" : options.nv_extra_plugin_lib_paths;
    info.profile_min_shapes = options.nv_profile_min_shapes == nullptr ? "" : options.nv_profile_min_shapes;
    info.profile_max_shapes = options.nv_profile_max_shapes == nullptr ? "" : options.nv_profile_max_shapes;
    info.profile_opt_shapes = options.nv_profile_opt_shapes == nullptr ? "" : options.nv_profile_opt_shapes;
    info.cuda_graph_enable = options.nv_cuda_graph_enable != 0;
    info.dump_ep_context_model = options.nv_dump_ep_context_model != 0;
    info.ep_context_file_path = options.nv_ep_context_file_path == nullptr ? "" : options.nv_ep_context_file_path;
    info.ep_context_embed_mode = options.nv_ep_context_embed_mode;
    info.engine_cache_prefix = options.nv_engine_cache_prefix == nullptr ? "" : options.nv_engine_cache_prefix;
    info.engine_hw_compatible = options.nv_engine_hw_compatible != 0;
    info.onnx_bytestream = options.nv_onnx_bytestream;
    info.onnx_bytestream_size = options.nv_onnx_bytestream_size;
    info.op_types_to_exclude = options.nv_op_types_to_exclude == nullptr ? "" : options.nv_op_types_to_exclude;

    return std::make_shared<NvProviderFactory>(info);
  }

  void UpdateProviderOptions(void* provider_options, const ProviderOptions& options) override {
    NvExecutionProviderInfo::UpdateProviderOptions(provider_options, options, true);
  }

  ProviderOptions GetProviderOptions(const void* provider_options) override {
    auto& options = *reinterpret_cast<const OrtNvProviderOptionsV2*>(provider_options);
    return onnxruntime::NvExecutionProviderInfo::ToProviderOptions(options);
  }

  void Initialize() override {
    InitializeRegistry();
  }

  void Shutdown() override {
    DeleteRegistry();
  }

} g_provider;

}  // namespace onnxruntime

extern "C" {

ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_provider;
}
}
