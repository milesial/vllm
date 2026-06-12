# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.cpu.spec import (
    is_mla_tp_dedup_supported,
    should_save_only_first_rank,
)


def _kv_cache_config(*specs: KVCacheSpec) -> KVCacheConfig:
    return KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[f"layer_{idx}"],
                kv_cache_spec=spec,
            )
            for idx, spec in enumerate(specs)
        ],
    )


def _mla_spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
    )


def _sliding_mla_spec() -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
        sliding_window=64,
    )


def _full_attention_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=128,
        dtype=torch.float16,
    )


def test_mla_tp_dedup_supported_accepts_mla_only_groups():
    kv_cache_config = _kv_cache_config(_mla_spec(), _sliding_mla_spec())

    assert is_mla_tp_dedup_supported(kv_cache_config)


def test_mla_tp_dedup_supported_rejects_mixed_groups():
    kv_cache_config = _kv_cache_config(_mla_spec(), _full_attention_spec())

    assert not is_mla_tp_dedup_supported(kv_cache_config)


def test_mla_tp_dedup_supported_rejects_empty_groups():
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )

    assert not is_mla_tp_dedup_supported(kv_cache_config)


def test_mla_tp_dedup_supported_accepts_uniform_type_mla_group():
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer_0": _mla_spec(),
            "layer_1": _mla_spec(),
        },
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer_0", "layer_1"],
                kv_cache_spec=uniform_spec,
            )
        ],
    )

    assert is_mla_tp_dedup_supported(kv_cache_config)


def test_mla_tp_dedup_supported_rejects_mixed_uniform_type_group():
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={
            "layer_0": _mla_spec(),
            "layer_1": _full_attention_spec(),
        },
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["layer_0", "layer_1"],
                kv_cache_spec=uniform_spec,
            )
        ],
    )

    assert not is_mla_tp_dedup_supported(kv_cache_config)


@pytest.mark.parametrize(
    ("extra_config", "kv_cache_config", "tensor_parallel_size", "expected"),
    [
        ({}, _kv_cache_config(_mla_spec()), 2, True),
        ({"save_only_first_rank": True}, _kv_cache_config(_mla_spec()), 2, True),
        ({"save_only_first_rank": False}, _kv_cache_config(_mla_spec()), 2, False),
        ({}, _kv_cache_config(_mla_spec()), 1, False),
        ({"save_only_first_rank": True}, _kv_cache_config(_mla_spec()), 1, False),
        ({}, _kv_cache_config(_full_attention_spec()), 2, False),
        (
            {"save_only_first_rank": True},
            _kv_cache_config(_full_attention_spec()),
            2,
            False,
        ),
    ],
)
def test_should_save_only_first_rank(
    extra_config: dict[str, bool],
    kv_cache_config: KVCacheConfig,
    tensor_parallel_size: int,
    expected: bool,
):
    assert (
        should_save_only_first_rank(
            extra_config,
            kv_cache_config,
            tensor_parallel_size=tensor_parallel_size,
        )
        == expected
    )


def test_should_save_only_first_rank_rejects_string_value():
    kv_cache_config = _kv_cache_config(_mla_spec())

    with pytest.raises(ValueError, match="save_only_first_rank must be a bool"):
        should_save_only_first_rank(
            {"save_only_first_rank": "false"},
            kv_cache_config,
            tensor_parallel_size=2,
        )
