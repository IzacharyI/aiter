# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest


def test_flydsl_legacy_moe_import_still_available():
    flydsl = pytest.importorskip("aiter.ops.flydsl")

    assert hasattr(flydsl, "flydsl_moe_stage1")
    assert hasattr(flydsl, "flydsl_moe_stage2")


def test_mega_moe_import_when_mori_available():
    pytest.importorskip("mori")
    pytest.importorskip("mori.ir.flydsl")

    from aiter.ops.flydsl.mega_moe import MegaMoE, MegaMoeStage1, MegaMoeStage2

    assert MegaMoE is not None
    assert MegaMoeStage1 is not None
    assert MegaMoeStage2 is not None
