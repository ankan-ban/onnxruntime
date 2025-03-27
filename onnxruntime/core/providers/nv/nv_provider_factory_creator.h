// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/providers/providers.h"

struct OrtNvProviderOptions;
struct OrtNvProviderOptionsV2;

namespace onnxruntime {
// defined in provider_bridge_ort.cc
struct NvProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(int device_id);
  static std::shared_ptr<IExecutionProviderFactory> Create(const OrtNvProviderOptions* provider_options);
  static std::shared_ptr<IExecutionProviderFactory> Create(const OrtNvProviderOptionsV2* provider_options);
};
}  // namespace onnxruntime
