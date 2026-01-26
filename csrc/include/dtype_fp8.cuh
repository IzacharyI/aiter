#pragma once
/*
 * Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
 * Copyright (C) 2024-2025, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "attention_generic.cuh"

#include <stdint.h>
#ifdef ENABLE_FP8
#ifndef USE_ROCM
#include <cuda_fp8.h>
#endif // USE_ROCM
#endif // ENABLE_FP8

namespace vllm {

enum class Fp8KVCacheDataType
{
    kAuto    = 0,
    kFp8E4M3 = 1,
    kFp8E5M2 = 2,
    kFp4E2M1 = 3,  // FP4 E2M1 (MXFP4) format, supported on gfx950+
};

enum class Fp8QuantMethod
{
    kPerTensor = 0,
    kPerHead   = 1,
    kPerToken  = 2,
};

// fp8 vector types for quantization of kv cache
template <>
struct Vec<uint8_t, 1>
{
    using Type = uint8_t;
};

template <>
struct Vec<uint8_t, 2>
{
    using Type = uint16_t;
};

template <>
struct Vec<uint8_t, 4>
{
    using Type = uint32_t;
};

template <>
struct Vec<uint8_t, 8>
{
    using Type = uint64_t;
};

// FP4 packed type: 2 FP4 values packed in 1 byte (uint8_t)
// For FP4, we use the same underlying storage as FP8 (uint8_t)
// but interpret it differently: each byte contains 2 FP4 values
// Lower 4 bits = first value, Upper 4 bits = second value

} // namespace vllm
