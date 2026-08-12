# SPDX-License-Identifier: MIT
"""Focused tests for measured MegaMoEV2 configuration overrides."""

from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    select_mega_moe_config,
)


def test_large_mtpr_512_uses_measured_payload_chunk():
    config = select_mega_moe_config(512, 8192)
    assert config.stage1.payload_chunk_rows == 256


def test_payload_chunk_override_is_bucket_and_mtpr_specific():
    assert select_mega_moe_config(256, 8192).stage1.payload_chunk_rows == 384
    assert select_mega_moe_config(1024, 8192).stage1.payload_chunk_rows == 384
    assert select_mega_moe_config(8192, 8192).stage1.payload_chunk_rows == 384
    assert select_mega_moe_config(512, 512).stage1.payload_chunk_rows == 0
