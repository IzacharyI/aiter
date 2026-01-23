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
#pragma once

#include "attention_generic.cuh"
#include <stdint.h>

namespace vllm {

constexpr float FP4_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

constexpr float FP4_E2M1_MAX = 6.0f;

// Packed FP4x2: 2 FP4 values in 1 byte
struct fp4x2_t {
    uint8_t data;
    
    __device__ __forceinline__ uint8_t low() const { return data & 0x0F; }
    __device__ __forceinline__ uint8_t high() const { return (data >> 4) & 0x0F; }
};

// Packed FP4x4: 4 FP4 values in 2 bytes
struct fp4x4_t {
    fp4x2_t xy[2];
};

// Packed FP4x8: 8 FP4 values in 4 bytes
struct fp4x8_t {
    fp4x2_t xy[4];
};

// Packed FP4x16: 16 FP4 values in 8 bytes
struct fp4x16_t {
    fp4x8_t xy[2];
};

// Packed FP4x32: 32 FP4 values in 16 bytes
struct fp4x32_t {
    fp4x16_t xy[2];
};

template <>
struct Vec<uint8_t, 1> {
    using Type = fp4x2_t;
};

template <>
struct Vec<uint8_t, 4> {
    using Type = fp4x8_t;
};

template <>
struct Vec<uint8_t, 8> {
    using Type = fp4x16_t;
};

template <>
struct Vec<uint8_t, 16> {
    using Type = fp4x32_t;
};


__device__ __forceinline__ float fp4_to_float(const uint8_t index)
{
    constexpr float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return lut[index & 0x0F];
}

__device__ __forceinline__ float2 fp4x2_to_float2(const uint8_t packed)
{
    constexpr float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    float2 ret;
    ret.x = lut[packed & 0x0F];
    ret.y = lut[(packed >> 4) & 0x0F];
    return ret;
}

__device__ __forceinline__ float2 fp4x2_to_float2_scaled(const uint8_t packed, const float scale)
{
    float2 ret = fp4x2_to_float2(packed);
    ret.x *= scale;
    ret.y *= scale;
    return ret;
}

__device__ __forceinline__ uint8_t float_to_fp4(const float x, const float scale)
{
    constexpr float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    
    float scaled = x / scale;
    float abs_val = fabsf(scaled);
    uint8_t sign = (scaled < 0) ? 8 : 0;
    
    uint8_t idx = 0;
    if (abs_val >= 0.25f) idx = 1;
    if (abs_val >= 0.75f) idx = 2;
    if (abs_val >= 1.25f) idx = 3;
    if (abs_val >= 1.75f) idx = 4;
    if (abs_val >= 2.5f) idx = 5;
    if (abs_val >= 3.5f) idx = 6;
    if (abs_val >= 5.0f) idx = 7;
    
    return sign | idx;
}

__device__ __forceinline__ uint8_t pack_fp4x2(const uint8_t low, const uint8_t high)
{
    return (low & 0x0F) | ((high & 0x0F) << 4);
}

__device__ __forceinline__ void unpack_fp4x2(const uint8_t packed, uint8_t& low, uint8_t& high)
{
    low = packed & 0x0F;
    high = (packed >> 4) & 0x0F;
}

}

