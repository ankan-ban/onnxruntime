// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/// <summary>
/// Options for the  provider that are passed to SessionOptionsAppendExecutionProvider_Nv.
/// </summary>
struct OrtNvProviderOptionsV2 {
  OrtNvProviderOptionsV2& operator=(const OrtNvProviderOptionsV2& other);  // copy assignment operator

  int device_id{0};                                      // cuda device id.
  int has_user_compute_stream{0};                        // indicator of user specified CUDA compute stream.
  void* user_compute_stream{nullptr};                    // user specified CUDA compute stream.
  int nv_max_partition_iterations{1000};                // maximum iterations for TensorRT parser to get capability
  int nv_min_subgraph_size{1};                          // minimum size of TensorRT subgraphs
  size_t nv_max_workspace_size{0};                      // maximum workspace size for TensorRT. Default is 0 means max device memory size
  int nv_fp16_enable{0};                                // enable TensorRT FP16 precision. Default 0 = false, nonzero = true
  int nv_int8_enable{0};                                // enable TensorRT INT8 precision. Default 0 = false, nonzero = true
  const char* nv_int8_calibration_table_name{nullptr};  // TensorRT INT8 calibration table name.
  int nv_int8_use_native_calibration_table{0};          // use native TensorRT generated calibration table. Default 0 = false, nonzero = true
  int nv_dla_enable{0};                                 // enable DLA. Default 0 = false, nonzero = true
  int nv_dla_core{0};                                   // DLA core number. Default 0
  int nv_dump_subgraphs{0};                             // dump  subgraph. Default 0 = false, nonzero = true
  int nv_engine_cache_enable{0};                        // enable engine caching. Default 0 = false, nonzero = true
  const char* nv_engine_cache_path{nullptr};            // specify engine cache path, defaults to the working directory
  int nv_engine_decryption_enable{0};                   // enable engine decryption. Default 0 = false, nonzero = true
  const char* nv_engine_decryption_lib_path{nullptr};   // specify engine decryption library path
  int nv_force_sequential_engine_build{0};              // force building TensorRT engine sequentially. Default 0 = false, nonzero = true
  int nv_context_memory_sharing_enable{0};              // enable context memory sharing between subgraphs. Default 0 = false, nonzero = true
  int nv_layer_norm_fp32_fallback{0};                   // force Pow + Reduce ops in layer norm to FP32. Default 0 = false, nonzero = true
  int nv_timing_cache_enable{0};                        // enable TensorRT timing cache. Default 0 = false, nonzero = true
  const char* nv_timing_cache_path{nullptr};            // specify timing cache path, if none is provided the nv_engine_cache_path is used
  int nv_force_timing_cache{0};                         // force the TensorRT cache to be used even if device profile does not match. Default 0 = false, nonzero = true
  int nv_detailed_build_log{0};                         // Enable detailed build step logging on TensorRT  with timing for each engine build. Default 0 = false, nonzero = true
  int nv_build_heuristics_enable{0};                    // Build engine using heuristics to reduce build time. Default 0 = false, nonzero = true
  int nv_sparsity_enable{0};                            // Control if sparsity can be used. Default 0 = false, 1 = true
  int nv_builder_optimization_level{3};                 // Set the builder optimization level. WARNING: levels below 3 do not guarantee good engine performance, but greatly improve build time.  Default 3, valid range [0-5]
  int nv_auxiliary_streams{-1};                         // Set maximum number of auxiliary streams per inference stream. Setting this value to 0 will lead to optimal memory usage. Default -1 = heuristics
  const char* nv_tactic_sources{nullptr};               // pecify the tactics to be used by adding (+) or removing (-) tactics from the default
                                                         // tactic sources (default = all available tactics) e.g. "-CUDNN,+CUBLAS" available keys: "CUBLAS"|"CUBLAS_LT"|"CUDNN"|"EDGE_MASK_CONVOLUTIONS"
  const char* nv_extra_plugin_lib_paths{nullptr};       // specify extra TensorRT plugin library paths
  const char* nv_profile_min_shapes{nullptr};           // Specify the range of the input shapes to build the engine with
  const char* nv_profile_max_shapes{nullptr};           // Specify the range of the input shapes to build the engine with
  const char* nv_profile_opt_shapes{nullptr};           // Specify the range of the input shapes to build the engine with
  int nv_cuda_graph_enable{0};                          // Enable CUDA graph in ORT NV

  /*
   * Please note that there are rules for using following context model related provider options:
   *
   * 1. In the case of dumping the context model and loading the context model,
   *    for security reason,  EP doesn't allow the "ep_cache_context" node attribute of EP context node to be
   *    the absolute path or relative path that is outside of context model directory.
   *    It means engine cache needs to be in the same directory or sub-directory of context model.
   *
   * 2. In the case of dumping the context model, the engine cache path will be changed to the relative path of context model directory.
   *    For example:
   *    If "nv_dump_ep_context_model" is enabled and "nv_engine_cache_enable" is enabled,
   *       if "nv_ep_context_file_path" is "./context_model_dir",
   *       - if "nv_engine_cache_path" is "" -> the engine cache will be saved to "./context_model_dir"
   *       - if "nv_engine_cache_path" is "engine_dir" -> the engine cache will be saved to "./context_model_dir/engine_dir"
   *
   * 3. In the case of building weight-stripped engines, the same security reasons as listed in 1) apply to the
   *    "onnx_model_filename" node attribute of EP context node, which contains a filename of the ONNX model with the
   *    weights needed for the refit process. User can specify a folder path relative to the current working
   *    directory by means of the "nv_onnx_model_folder_path" option.
   *
   */
  int nv_dump_ep_context_model{0};                 // Dump EP context node model
  const char* nv_ep_context_file_path{nullptr};    // Specify file name to dump EP context node model. Can be a path or a file name or a file name with path.
  int nv_ep_context_embed_mode{0};                 // Specify EP context embed mode. Default 0 = context is engine cache path, 1 = context is engine binary data
  int nv_weight_stripped_engine_enable{0};         // Enable weight-stripped engine build. Default 0 = false,
                                                    // nonzero = true
  const char* nv_onnx_model_folder_path{nullptr};  // Folder path relative to the current working directory for
                                                    // the ONNX model containing the weights (applicable only when
                                                    // the "nv_weight_stripped_engine_enable" option is enabled)
  const void* nv_onnx_bytestream{nullptr};         // The byte stream of th original ONNX model containing the weights
                                                    // (applicable only when the "nv_weight_stripped_engine_enable"
                                                    // option is enabled)
                                                    // can be updated using: UpdateTensorRTProviderOptionsWithValue
  size_t nv_onnx_bytestream_size{0};               // size of the byte stream provided as "nv_onnx_bytestream"
                                                    // can be updated using: UpdateTensorRTProviderOptionsWithValue

  const char* nv_engine_cache_prefix{nullptr};  // specify engine cache prefix
  int nv_engine_hw_compatible{0};               // Enable hardware compatibility. Default 0 = false, nonzero = true
  const char* nv_op_types_to_exclude{};         // Exclude specific ops from running on .
};
