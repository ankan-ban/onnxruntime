// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nv/nv_execution_provider_info.h"
#include "core/providers/nv/nv_provider_options.h"

#include "core/common/make_string.h"
#include "core/common/parse_string.h"
#include "core/framework/provider_options_utils.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace nv {
namespace provider_option_names {
constexpr const char* kDeviceId = "device_id";
constexpr const char* kHasUserComputeStream = "has_user_compute_stream";
constexpr const char* kUserComputeStream = "user_compute_stream";
constexpr const char* kMaxPartitionIterations = "nv_max_partition_iterations";
constexpr const char* kMinSubgraphSize = "nv_min_subgraph_size";
constexpr const char* kMaxWorkspaceSize = "nv_max_workspace_size";
constexpr const char* kFp16Enable = "nv_fp16_enable";
constexpr const char* kInt8Enable = "nv_int8_enable";
constexpr const char* kInt8CalibTable = "nv_int8_calibration_table_name";
constexpr const char* kInt8UseNativeCalibTable = "nv_int8_use_native_calibration_table";
constexpr const char* kDLAEnable = "nv_dla_enable";
constexpr const char* kDLACore = "nv_dla_core";
constexpr const char* kDumpSubgraphs = "nv_dump_subgraphs";
constexpr const char* kEngineCacheEnable = "nv_engine_cache_enable";
constexpr const char* kEngineCachePath = "nv_engine_cache_path";
constexpr const char* kWeightStrippedEngineEnable = "nv_weight_stripped_engine_enable";
constexpr const char* kOnnxModelFolderPath = "nv_onnx_model_folder_path";
constexpr const char* kEngineCachePrefix = "nv_engine_cache_prefix";
constexpr const char* kDecryptionEnable = "nv_engine_decryption_enable";
constexpr const char* kDecryptionLibPath = "nv_engine_decryption_lib_path";
constexpr const char* kForceSequentialEngineBuild = "nv_force_sequential_engine_build";
// add new provider option name here.
constexpr const char* kContextMemorySharingEnable = "nv_context_memory_sharing_enable";
constexpr const char* kLayerNormFP32Fallback = "nv_layer_norm_fp32_fallback";
constexpr const char* kTimingCacheEnable = "nv_timing_cache_enable";
constexpr const char* kTimingCachePath = "nv_timing_cache_path";
constexpr const char* kForceTimingCacheMatch = "nv_force_timing_cache";
constexpr const char* kDetailedBuildLog = "nv_detailed_build_log";
constexpr const char* kBuildHeuristics = "nv_build_heuristics_enable";
constexpr const char* kSparsityEnable = "nv_sparsity_enable";
constexpr const char* kBuilderOptimizationLevel = "nv_builder_optimization_level";
constexpr const char* kAuxiliaryStreams = "nv_auxiliary_streams";
constexpr const char* kTacticSources = "nv_tactic_sources";
constexpr const char* kExtraPluginLibPaths = "nv_extra_plugin_lib_paths";
constexpr const char* kProfilesMinShapes = "nv_profile_min_shapes";
constexpr const char* kProfilesMaxShapes = "nv_profile_max_shapes";
constexpr const char* kProfilesOptShapes = "nv_profile_opt_shapes";
constexpr const char* kCudaGraphEnable = "nv_cuda_graph_enable";
constexpr const char* kEpContextEmbedMode = "nv_ep_context_embed_mode";
constexpr const char* kEpContextFilePath = "nv_ep_context_file_path";
constexpr const char* kDumpEpContextModel = "nv_dump_ep_context_model";
constexpr const char* kEngineHwCompatible = "nv_engine_hw_compatible";
constexpr const char* kONNXBytestream = "nv_onnx_bytestream";
constexpr const char* kONNXBytestreamSize = "nv_onnx_bytestream_size";
constexpr const char* kOpTypesToExclude = "nv_op_types_to_exclude";

}  // namespace provider_option_names
}  // namespace nv

NvExecutionProviderInfo NvExecutionProviderInfo::FromProviderOptions(const ProviderOptions& options) {
  NvExecutionProviderInfo info{};
  void* user_compute_stream = nullptr;
  void* onnx_bytestream = nullptr;
  ORT_THROW_IF_ERROR(
      ProviderOptionsParser{}
          .AddValueParser(
              nv::provider_option_names::kDeviceId,
              [&info](const std::string& value_str) -> Status {
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, info.device_id));
                int num_devices{};
                CUDA_RETURN_IF_ERROR(cudaGetDeviceCount(&num_devices));
                ORT_RETURN_IF_NOT(
                    0 <= info.device_id && info.device_id < num_devices,
                    "Invalid device ID: ", info.device_id,
                    ", must be between 0 (inclusive) and ", num_devices, " (exclusive).");
                return Status::OK();
              })
          .AddAssignmentToReference(nv::provider_option_names::kMaxPartitionIterations, info.max_partition_iterations)
          .AddAssignmentToReference(nv::provider_option_names::kHasUserComputeStream, info.has_user_compute_stream)
          .AddValueParser(
              nv::provider_option_names::kUserComputeStream,
              [&user_compute_stream](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                user_compute_stream = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddAssignmentToReference(nv::provider_option_names::kMinSubgraphSize, info.min_subgraph_size)
          .AddAssignmentToReference(nv::provider_option_names::kMaxWorkspaceSize, info.max_workspace_size)
          .AddAssignmentToReference(nv::provider_option_names::kFp16Enable, info.fp16_enable)
          .AddAssignmentToReference(nv::provider_option_names::kInt8Enable, info.int8_enable)
          .AddAssignmentToReference(nv::provider_option_names::kInt8CalibTable, info.int8_calibration_table_name)
          .AddAssignmentToReference(nv::provider_option_names::kInt8UseNativeCalibTable, info.int8_use_native_calibration_table)
          .AddAssignmentToReference(nv::provider_option_names::kDLAEnable, info.dla_enable)
          .AddAssignmentToReference(nv::provider_option_names::kDLACore, info.dla_core)
          .AddAssignmentToReference(nv::provider_option_names::kDumpSubgraphs, info.dump_subgraphs)
          .AddAssignmentToReference(nv::provider_option_names::kEngineCacheEnable, info.engine_cache_enable)
          .AddAssignmentToReference(nv::provider_option_names::kEngineCachePath, info.engine_cache_path)
          .AddAssignmentToReference(nv::provider_option_names::kWeightStrippedEngineEnable, info.weight_stripped_engine_enable)
          .AddAssignmentToReference(nv::provider_option_names::kOnnxModelFolderPath, info.onnx_model_folder_path)
          .AddAssignmentToReference(nv::provider_option_names::kEngineCachePrefix, info.engine_cache_prefix)
          .AddAssignmentToReference(nv::provider_option_names::kDecryptionEnable, info.engine_decryption_enable)
          .AddAssignmentToReference(nv::provider_option_names::kDecryptionLibPath, info.engine_decryption_lib_path)
          .AddAssignmentToReference(nv::provider_option_names::kForceSequentialEngineBuild, info.force_sequential_engine_build)
          .AddAssignmentToReference(nv::provider_option_names::kContextMemorySharingEnable, info.context_memory_sharing_enable)
          .AddAssignmentToReference(nv::provider_option_names::kLayerNormFP32Fallback, info.layer_norm_fp32_fallback)
          .AddAssignmentToReference(nv::provider_option_names::kTimingCacheEnable, info.timing_cache_enable)
          .AddAssignmentToReference(nv::provider_option_names::kTimingCachePath, info.timing_cache_path)
          .AddAssignmentToReference(nv::provider_option_names::kForceTimingCacheMatch, info.force_timing_cache)
          .AddAssignmentToReference(nv::provider_option_names::kDetailedBuildLog, info.detailed_build_log)
          .AddAssignmentToReference(nv::provider_option_names::kBuildHeuristics, info.build_heuristics_enable)
          .AddAssignmentToReference(nv::provider_option_names::kSparsityEnable, info.sparsity_enable)
          .AddAssignmentToReference(nv::provider_option_names::kBuilderOptimizationLevel, info.builder_optimization_level)
          .AddAssignmentToReference(nv::provider_option_names::kAuxiliaryStreams, info.auxiliary_streams)
          .AddAssignmentToReference(nv::provider_option_names::kTacticSources, info.tactic_sources)
          .AddAssignmentToReference(nv::provider_option_names::kExtraPluginLibPaths, info.extra_plugin_lib_paths)
          .AddAssignmentToReference(nv::provider_option_names::kProfilesMinShapes, info.profile_min_shapes)
          .AddAssignmentToReference(nv::provider_option_names::kProfilesMaxShapes, info.profile_max_shapes)
          .AddAssignmentToReference(nv::provider_option_names::kProfilesOptShapes, info.profile_opt_shapes)
          .AddAssignmentToReference(nv::provider_option_names::kCudaGraphEnable, info.cuda_graph_enable)
          .AddAssignmentToReference(nv::provider_option_names::kDumpEpContextModel, info.dump_ep_context_model)
          .AddAssignmentToReference(nv::provider_option_names::kEpContextFilePath, info.ep_context_file_path)
          .AddAssignmentToReference(nv::provider_option_names::kEpContextEmbedMode, info.ep_context_embed_mode)
          .AddAssignmentToReference(nv::provider_option_names::kEngineHwCompatible, info.engine_hw_compatible)
          .AddValueParser(
              nv::provider_option_names::kONNXBytestream,
              [&onnx_bytestream](const std::string& value_str) -> Status {
                size_t address;
                ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(value_str, address));
                onnx_bytestream = reinterpret_cast<void*>(address);
                return Status::OK();
              })
          .AddAssignmentToReference(nv::provider_option_names::kONNXBytestreamSize, info.onnx_bytestream_size)
          .AddAssignmentToReference(nv::provider_option_names::kOpTypesToExclude, info.op_types_to_exclude)
          .Parse(options));  // add new provider option here.

  info.user_compute_stream = user_compute_stream;
  info.has_user_compute_stream = (user_compute_stream != nullptr);
  info.onnx_bytestream = onnx_bytestream;
  return info;
}

ProviderOptions NvExecutionProviderInfo::ToProviderOptions(const NvExecutionProviderInfo& info) {
  const ProviderOptions options{
      {nv::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {nv::provider_option_names::kMaxPartitionIterations, MakeStringWithClassicLocale(info.max_partition_iterations)},
      {nv::provider_option_names::kHasUserComputeStream, MakeStringWithClassicLocale(info.has_user_compute_stream)},
      {nv::provider_option_names::kUserComputeStream, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.user_compute_stream))},
      {nv::provider_option_names::kMinSubgraphSize, MakeStringWithClassicLocale(info.min_subgraph_size)},
      {nv::provider_option_names::kMaxWorkspaceSize, MakeStringWithClassicLocale(info.max_workspace_size)},
      {nv::provider_option_names::kFp16Enable, MakeStringWithClassicLocale(info.fp16_enable)},
      {nv::provider_option_names::kInt8Enable, MakeStringWithClassicLocale(info.int8_enable)},
      {nv::provider_option_names::kInt8CalibTable, MakeStringWithClassicLocale(info.int8_calibration_table_name)},
      {nv::provider_option_names::kInt8UseNativeCalibTable, MakeStringWithClassicLocale(info.int8_use_native_calibration_table)},
      {nv::provider_option_names::kDLAEnable, MakeStringWithClassicLocale(info.dla_enable)},
      {nv::provider_option_names::kDLACore, MakeStringWithClassicLocale(info.dla_core)},
      {nv::provider_option_names::kDumpSubgraphs, MakeStringWithClassicLocale(info.dump_subgraphs)},
      {nv::provider_option_names::kEngineCacheEnable, MakeStringWithClassicLocale(info.engine_cache_enable)},
      {nv::provider_option_names::kEngineCachePath, MakeStringWithClassicLocale(info.engine_cache_path)},
      {nv::provider_option_names::kWeightStrippedEngineEnable, MakeStringWithClassicLocale(info.weight_stripped_engine_enable)},
      {nv::provider_option_names::kOnnxModelFolderPath, MakeStringWithClassicLocale(info.onnx_model_folder_path)},
      {nv::provider_option_names::kEngineCachePrefix, MakeStringWithClassicLocale(info.engine_cache_prefix)},
      {nv::provider_option_names::kDecryptionEnable, MakeStringWithClassicLocale(info.engine_decryption_enable)},
      {nv::provider_option_names::kDecryptionLibPath, MakeStringWithClassicLocale(info.engine_decryption_lib_path)},
      {nv::provider_option_names::kForceSequentialEngineBuild, MakeStringWithClassicLocale(info.force_sequential_engine_build)},
      // add new provider option here.
      {nv::provider_option_names::kContextMemorySharingEnable, MakeStringWithClassicLocale(info.context_memory_sharing_enable)},
      {nv::provider_option_names::kLayerNormFP32Fallback, MakeStringWithClassicLocale(info.layer_norm_fp32_fallback)},
      {nv::provider_option_names::kTimingCacheEnable, MakeStringWithClassicLocale(info.timing_cache_enable)},
      {nv::provider_option_names::kTimingCachePath, MakeStringWithClassicLocale(info.timing_cache_path)},
      {nv::provider_option_names::kForceTimingCacheMatch, MakeStringWithClassicLocale(info.force_timing_cache)},
      {nv::provider_option_names::kDetailedBuildLog, MakeStringWithClassicLocale(info.detailed_build_log)},
      {nv::provider_option_names::kBuildHeuristics, MakeStringWithClassicLocale(info.build_heuristics_enable)},
      {nv::provider_option_names::kSparsityEnable, MakeStringWithClassicLocale(info.sparsity_enable)},
      {nv::provider_option_names::kBuilderOptimizationLevel, MakeStringWithClassicLocale(info.builder_optimization_level)},
      {nv::provider_option_names::kAuxiliaryStreams, MakeStringWithClassicLocale(info.auxiliary_streams)},
      {nv::provider_option_names::kTacticSources, MakeStringWithClassicLocale(info.tactic_sources)},
      {nv::provider_option_names::kExtraPluginLibPaths, MakeStringWithClassicLocale(info.extra_plugin_lib_paths)},
      {nv::provider_option_names::kProfilesMinShapes, MakeStringWithClassicLocale(info.profile_min_shapes)},
      {nv::provider_option_names::kProfilesMaxShapes, MakeStringWithClassicLocale(info.profile_max_shapes)},
      {nv::provider_option_names::kProfilesOptShapes, MakeStringWithClassicLocale(info.profile_opt_shapes)},
      {nv::provider_option_names::kCudaGraphEnable, MakeStringWithClassicLocale(info.cuda_graph_enable)},
      {nv::provider_option_names::kDumpEpContextModel, MakeStringWithClassicLocale(info.dump_ep_context_model)},
      {nv::provider_option_names::kEpContextFilePath, MakeStringWithClassicLocale(info.ep_context_file_path)},
      {nv::provider_option_names::kEpContextEmbedMode, MakeStringWithClassicLocale(info.ep_context_embed_mode)},
      {nv::provider_option_names::kEngineHwCompatible, MakeStringWithClassicLocale(info.engine_hw_compatible)},
      {nv::provider_option_names::kONNXBytestream, MakeStringWithClassicLocale(info.onnx_bytestream)},
      {nv::provider_option_names::kONNXBytestreamSize, MakeStringWithClassicLocale(info.onnx_bytestream_size)},
      {nv::provider_option_names::kOpTypesToExclude, MakeStringWithClassicLocale(info.op_types_to_exclude)},
  };
  return options;
}

ProviderOptions NvExecutionProviderInfo::ToProviderOptions(const OrtNvProviderOptionsV2& info) {
  auto empty_if_null = [](const char* s) { return s != nullptr ? std::string{s} : std::string{}; };
  const std::string kInt8CalibTable_ = empty_if_null(info.nv_int8_calibration_table_name);
  const std::string kEngineCachePath_ = empty_if_null(info.nv_engine_cache_path);
  const std::string kEngineCachePrefix_ = empty_if_null(info.nv_engine_cache_prefix);
  const std::string kTimingCachePath_ = empty_if_null(info.nv_timing_cache_path);
  const std::string kTacticSources_ = empty_if_null(info.nv_tactic_sources);
  const std::string kDecryptionLibPath_ = empty_if_null(info.nv_engine_decryption_lib_path);
  const std::string kExtraPluginLibPaths_ = empty_if_null(info.nv_extra_plugin_lib_paths);
  const std::string kProfilesMinShapes_ = empty_if_null(info.nv_profile_min_shapes);
  const std::string kProfilesMaxShapes_ = empty_if_null(info.nv_profile_max_shapes);
  const std::string kProfilesOptShapes_ = empty_if_null(info.nv_profile_opt_shapes);
  const std::string kEpContextFilePath_ = empty_if_null(info.nv_ep_context_file_path);
  const std::string kOnnxModelFolderPath_ = empty_if_null(info.nv_onnx_model_folder_path);
  const std::string kOpTypesToExclude_ = empty_if_null(info.nv_op_types_to_exclude);

  const ProviderOptions options{
      {nv::provider_option_names::kDeviceId, MakeStringWithClassicLocale(info.device_id)},
      {nv::provider_option_names::kHasUserComputeStream, MakeStringWithClassicLocale(info.has_user_compute_stream)},
      {nv::provider_option_names::kUserComputeStream, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.user_compute_stream))},
      {nv::provider_option_names::kMaxPartitionIterations, MakeStringWithClassicLocale(info.nv_max_partition_iterations)},
      {nv::provider_option_names::kMinSubgraphSize, MakeStringWithClassicLocale(info.nv_min_subgraph_size)},
      {nv::provider_option_names::kMaxWorkspaceSize, MakeStringWithClassicLocale(info.nv_max_workspace_size)},
      {nv::provider_option_names::kFp16Enable, MakeStringWithClassicLocale(info.nv_fp16_enable)},
      {nv::provider_option_names::kInt8Enable, MakeStringWithClassicLocale(info.nv_int8_enable)},
      {nv::provider_option_names::kInt8CalibTable, kInt8CalibTable_},
      {nv::provider_option_names::kInt8UseNativeCalibTable, MakeStringWithClassicLocale(info.nv_int8_use_native_calibration_table)},
      {nv::provider_option_names::kDLAEnable, MakeStringWithClassicLocale(info.nv_dla_enable)},
      {nv::provider_option_names::kDLACore, MakeStringWithClassicLocale(info.nv_dla_core)},
      {nv::provider_option_names::kDumpSubgraphs, MakeStringWithClassicLocale(info.nv_dump_subgraphs)},
      {nv::provider_option_names::kEngineCacheEnable, MakeStringWithClassicLocale(info.nv_engine_cache_enable)},
      {nv::provider_option_names::kEngineCachePath, kEngineCachePath_},
      {nv::provider_option_names::kEngineCachePrefix, kEngineCachePrefix_},
      {nv::provider_option_names::kWeightStrippedEngineEnable, MakeStringWithClassicLocale(info.nv_weight_stripped_engine_enable)},
      {nv::provider_option_names::kOnnxModelFolderPath, kOnnxModelFolderPath_},
      {nv::provider_option_names::kDecryptionEnable, MakeStringWithClassicLocale(info.nv_engine_decryption_enable)},
      {nv::provider_option_names::kDecryptionLibPath, kDecryptionLibPath_},
      {nv::provider_option_names::kForceSequentialEngineBuild, MakeStringWithClassicLocale(info.nv_force_sequential_engine_build)},
      {nv::provider_option_names::kContextMemorySharingEnable, MakeStringWithClassicLocale(info.nv_context_memory_sharing_enable)},
      {nv::provider_option_names::kLayerNormFP32Fallback, MakeStringWithClassicLocale(info.nv_layer_norm_fp32_fallback)},
      {nv::provider_option_names::kTimingCacheEnable, MakeStringWithClassicLocale(info.nv_timing_cache_enable)},
      {nv::provider_option_names::kTimingCachePath, kTimingCachePath_},
      {nv::provider_option_names::kForceTimingCacheMatch, MakeStringWithClassicLocale(info.nv_force_timing_cache)},
      {nv::provider_option_names::kDetailedBuildLog, MakeStringWithClassicLocale(info.nv_detailed_build_log)},
      {nv::provider_option_names::kBuildHeuristics, MakeStringWithClassicLocale(info.nv_build_heuristics_enable)},
      {nv::provider_option_names::kSparsityEnable, MakeStringWithClassicLocale(info.nv_sparsity_enable)},
      {nv::provider_option_names::kBuilderOptimizationLevel, MakeStringWithClassicLocale(info.nv_builder_optimization_level)},
      {nv::provider_option_names::kAuxiliaryStreams, MakeStringWithClassicLocale(info.nv_auxiliary_streams)},
      {nv::provider_option_names::kTacticSources, kTacticSources_},
      {nv::provider_option_names::kExtraPluginLibPaths, kExtraPluginLibPaths_},
      {nv::provider_option_names::kProfilesMinShapes, kProfilesMinShapes_},
      {nv::provider_option_names::kProfilesMaxShapes, kProfilesMaxShapes_},
      {nv::provider_option_names::kProfilesOptShapes, kProfilesOptShapes_},
      {nv::provider_option_names::kCudaGraphEnable, MakeStringWithClassicLocale(info.nv_cuda_graph_enable)},
      {nv::provider_option_names::kEpContextFilePath, kEpContextFilePath_},
      {nv::provider_option_names::kDumpEpContextModel, MakeStringWithClassicLocale(info.nv_dump_ep_context_model)},
      {nv::provider_option_names::kEpContextEmbedMode, MakeStringWithClassicLocale(info.nv_ep_context_embed_mode)},
      {nv::provider_option_names::kEngineHwCompatible, MakeStringWithClassicLocale(info.nv_engine_hw_compatible)},
      {nv::provider_option_names::kONNXBytestream, MakeStringWithClassicLocale(reinterpret_cast<size_t>(info.nv_onnx_bytestream))},
      {nv::provider_option_names::kONNXBytestreamSize, MakeStringWithClassicLocale(info.nv_onnx_bytestream_size)},
      {nv::provider_option_names::kOpTypesToExclude, kOpTypesToExclude_},
  };
  return options;
}

/**
 * Update OrtNvProviderOptionsV2 instance with ProviderOptions (map of string-based key-value pairs)
 *
 * Please note that it will reset the OrtNvProviderOptionsV2 instance first and then set up the provided provider options
 * See NvExecutionProviderInfo::FromProviderOptions() for more details. This function will be called by the C API UpdateTensorRTProviderOptions() also.
 *
 * \param provider_options - a pointer to OrtNvProviderOptionsV2 instance
 * \param options - a reference to ProviderOptions instance
 * \param string_copy - if it's true, it uses strncpy() to copy 'provider option' string from ProviderOptions instance to where the 'provider option' const char pointer in OrtNvProviderOptionsV2 instance points to.
 *                      it it's false, it only saves the pointer and no strncpy().
 *
 * Note: If there is strncpy involved, please remember to deallocate or simply call C API ReleaseTensorRTProviderOptions.
 */
void NvExecutionProviderInfo::UpdateProviderOptions(void* provider_options, const ProviderOptions& options, bool string_copy) {
  if (provider_options == nullptr) {
    return;
  }
  auto copy_string_if_needed = [&](std::string& s_in) {
    if (string_copy) {
      char* dest = nullptr;
      auto str_size = s_in.size();
      if (str_size == 0) {
        return (const char*)nullptr;
      } else {
        dest = new char[str_size + 1];
#ifdef _MSC_VER
        strncpy_s(dest, str_size + 1, s_in.c_str(), str_size);
#else
        strncpy(dest, s_in.c_str(), str_size);
#endif
        dest[str_size] = '\0';
        return (const char*)dest;
      }
    } else {
      return s_in.c_str();
    }
  };

  NvExecutionProviderInfo internal_options = onnxruntime::NvExecutionProviderInfo::FromProviderOptions(options);
  auto& nv_provider_options_v2 = *reinterpret_cast<OrtNvProviderOptionsV2*>(provider_options);
  nv_provider_options_v2.device_id = internal_options.device_id;

  // The 'has_user_compute_stream' of the OrtNvProviderOptionsV2 instance can be set by C API UpdateTensorRTProviderOptionsWithValue() as well
  // We only set the 'has_user_compute_stream' of the OrtNvProviderOptionsV2 instance if it is provided in options or user_compute_stream is provided
  if (options.find("has_user_compute_stream") != options.end()) {
    nv_provider_options_v2.has_user_compute_stream = internal_options.has_user_compute_stream;
  }
  if (options.find("user_compute_stream") != options.end() && internal_options.user_compute_stream != nullptr) {
    nv_provider_options_v2.user_compute_stream = internal_options.user_compute_stream;
    nv_provider_options_v2.has_user_compute_stream = true;
  }

  nv_provider_options_v2.nv_max_partition_iterations = internal_options.max_partition_iterations;
  nv_provider_options_v2.nv_min_subgraph_size = internal_options.min_subgraph_size;
  nv_provider_options_v2.nv_max_workspace_size = internal_options.max_workspace_size;
  nv_provider_options_v2.nv_fp16_enable = internal_options.fp16_enable;
  nv_provider_options_v2.nv_int8_enable = internal_options.int8_enable;

  nv_provider_options_v2.nv_int8_calibration_table_name = copy_string_if_needed(internal_options.int8_calibration_table_name);

  nv_provider_options_v2.nv_int8_use_native_calibration_table = internal_options.int8_use_native_calibration_table;
  nv_provider_options_v2.nv_dla_enable = internal_options.dla_enable;
  nv_provider_options_v2.nv_dla_core = internal_options.dla_core;
  nv_provider_options_v2.nv_dump_subgraphs = internal_options.dump_subgraphs;
  nv_provider_options_v2.nv_engine_cache_enable = internal_options.engine_cache_enable;
  nv_provider_options_v2.nv_weight_stripped_engine_enable = internal_options.weight_stripped_engine_enable;
  nv_provider_options_v2.nv_onnx_model_folder_path = copy_string_if_needed(internal_options.onnx_model_folder_path);

  nv_provider_options_v2.nv_engine_cache_path = copy_string_if_needed(internal_options.engine_cache_path);
  nv_provider_options_v2.nv_engine_cache_prefix = copy_string_if_needed(internal_options.engine_cache_prefix);
  nv_provider_options_v2.nv_timing_cache_path = copy_string_if_needed(internal_options.timing_cache_path);

  nv_provider_options_v2.nv_engine_decryption_enable = internal_options.engine_decryption_enable;

  nv_provider_options_v2.nv_engine_decryption_lib_path = copy_string_if_needed(internal_options.engine_decryption_lib_path);

  nv_provider_options_v2.nv_force_sequential_engine_build = internal_options.force_sequential_engine_build;
  nv_provider_options_v2.nv_context_memory_sharing_enable = internal_options.context_memory_sharing_enable;
  nv_provider_options_v2.nv_layer_norm_fp32_fallback = internal_options.layer_norm_fp32_fallback;
  nv_provider_options_v2.nv_timing_cache_enable = internal_options.timing_cache_enable;
  nv_provider_options_v2.nv_force_timing_cache = internal_options.force_timing_cache;
  nv_provider_options_v2.nv_detailed_build_log = internal_options.detailed_build_log;
  nv_provider_options_v2.nv_build_heuristics_enable = internal_options.build_heuristics_enable;
  nv_provider_options_v2.nv_sparsity_enable = internal_options.sparsity_enable;
  nv_provider_options_v2.nv_builder_optimization_level = internal_options.builder_optimization_level;
  nv_provider_options_v2.nv_auxiliary_streams = internal_options.auxiliary_streams;

  nv_provider_options_v2.nv_tactic_sources = copy_string_if_needed(internal_options.tactic_sources);
  nv_provider_options_v2.nv_extra_plugin_lib_paths = copy_string_if_needed(internal_options.extra_plugin_lib_paths);
  nv_provider_options_v2.nv_profile_min_shapes = copy_string_if_needed(internal_options.profile_min_shapes);
  nv_provider_options_v2.nv_profile_max_shapes = copy_string_if_needed(internal_options.profile_max_shapes);
  nv_provider_options_v2.nv_profile_opt_shapes = copy_string_if_needed(internal_options.profile_opt_shapes);

  nv_provider_options_v2.nv_cuda_graph_enable = internal_options.cuda_graph_enable;
  nv_provider_options_v2.nv_dump_ep_context_model = internal_options.dump_ep_context_model;
  nv_provider_options_v2.nv_ep_context_embed_mode = internal_options.ep_context_embed_mode;
  nv_provider_options_v2.nv_ep_context_file_path = copy_string_if_needed(internal_options.ep_context_file_path);
  nv_provider_options_v2.nv_engine_hw_compatible = internal_options.engine_hw_compatible;
  nv_provider_options_v2.nv_onnx_bytestream = internal_options.onnx_bytestream;
  nv_provider_options_v2.nv_onnx_bytestream_size = internal_options.onnx_bytestream_size;
  nv_provider_options_v2.nv_op_types_to_exclude = copy_string_if_needed(internal_options.op_types_to_exclude);
}
}  // namespace onnxruntime
