# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterator
from typing import Any

from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheSpecKind,
    get_kv_cache_spec_kind,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingCounterMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
)
from vllm.v1.kv_offload.cpu.common import METRIC_STORES_SKIPPED, CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CpuGpuOffloadingHandlers
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.worker.worker import OffloadingHandler

logger = init_logger(__name__)

_TP_DEDUP_MLA_KINDS = {
    KVCacheSpecKind.MLA_ATTENTION,
    KVCacheSpecKind.SLIDING_WINDOW_MLA,
}


def is_mla_tp_dedup_supported(kv_cache_config: KVCacheConfig) -> bool:
    """Return True if every KV cache group stores replicated MLA latent KV.

    Hybrid layouts are rejected because source-rank-only host I/O can
    partially populate physical pages shared across canonical tensors.
    """
    if not kv_cache_config.kv_cache_groups:
        return False
    for group in kv_cache_config.kv_cache_groups:
        if get_kv_cache_spec_kind(group.kv_cache_spec) not in _TP_DEDUP_MLA_KINDS:
            return False
    return True


def should_save_only_first_rank(
    extra_config: dict[str, Any],
    kv_cache_config: KVCacheConfig,
    tensor_parallel_size: int,
) -> bool:
    """Resolve the LMCache-compatible ``save_only_first_rank`` knob.

    Args:
        extra_config: KV connector extra config.
        kv_cache_config: KV cache layout to check for replicated MLA KV.
        tensor_parallel_size: Tensor-parallel group size.

    Returns:
        True when native CPU offload can use source-rank-only host I/O,
        implemented by the MLA TP-dedup broadcast path. The default is True
        for supported MLA TP layouts and False otherwise. Explicit True is
        still rejected for TP=1 or non-MLA layouts.
    """
    raw_save_only_first_rank = extra_config.get("save_only_first_rank")
    if raw_save_only_first_rank is not None and not isinstance(
        raw_save_only_first_rank, bool
    ):
        raise ValueError(
            "save_only_first_rank must be a bool, got "
            f"{type(raw_save_only_first_rank).__name__}="
            f"{raw_save_only_first_rank!r}"
        )

    mla_tp_dedup_supported = is_mla_tp_dedup_supported(kv_cache_config)
    supported = tensor_parallel_size > 1 and mla_tp_dedup_supported
    if raw_save_only_first_rank is True and not supported:
        logger.warning(
            "save_only_first_rank=True ignored: requires tp_size>1 and "
            "MLA-only KV cache layout (tp_size=%d, mla_supported=%s).",
            tensor_parallel_size,
            mla_tp_dedup_supported,
        )

    if not supported:
        return False
    return True if raw_save_only_first_rank is None else raw_save_only_first_rank


class CPUOffloadingSpec(OffloadingSpec):
    BLOCK_SIZE_ALIGNMENT = 1

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold < 2:
            return {}
        return {
            METRIC_STORES_SKIPPED: OffloadingCounterMetadata(
                documentation=(
                    "Number of KV offload stores skipped because the reuse "
                    "threshold was not reached."
                ),
            )
        }

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig):
        super().__init__(vllm_config, kv_cache_config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        parallel_config = vllm_config.parallel_config
        world_size = parallel_config.world_size
        self.save_only_first_rank = should_save_only_first_rank(
            self.extra_config,
            kv_cache_config,
            parallel_config.tensor_parallel_size,
        )
        if self.save_only_first_rank:
            logger.info(
                "CPU offload save_only_first_rank=True (MLA, tp=%d)",
                parallel_config.tensor_parallel_size,
            )
        self.num_blocks = 0
        self.kv_bytes_per_offloaded_block = 0
        self.cpu_page_size_per_worker = 0
        assert kv_cache_config is not None
        if kv_cache_config.num_blocks > 0 and world_size > 0:
            total_gpu_kv_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
            kv_bytes_per_block = (
                total_gpu_kv_bytes // kv_cache_config.num_blocks
            ) * world_size
            kv_bytes_per_offloaded_block = kv_bytes_per_block * self.block_size_factor

            # calculate cpu_page_size_per_worker
            self.cpu_page_size_per_worker = kv_bytes_per_offloaded_block // world_size

            # calculate num_blocks
            aligned_kv_bytes_per_offloaded_block = round_up(
                kv_bytes_per_offloaded_block, self.BLOCK_SIZE_ALIGNMENT
            )
            self.num_blocks = (
                int(cpu_bytes_to_use) // aligned_kv_bytes_per_offloaded_block
            )

            # Expose aligned_kv_bytes_per_offloaded_block as
            # kv_bytes_per_offloaded_block. Note that this might contain
            # some padding. i.e. each offloaded block is of the form,
            # |--- W0-B0---|---- W1-B0---| ... |---- Wn-B0---| *** maybe-pad *** |
            self.kv_bytes_per_offloaded_block = aligned_kv_bytes_per_offloaded_block

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._handlers: CpuGpuOffloadingHandlers | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            kv_events_config = self.vllm_config.kv_events_config
            enable_events = (
                kv_events_config is not None and kv_events_config.enable_kv_cache_events
            )

            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for CPU offloading.  Values < 2 disable
            # filtering (a threshold of 1 equals no filter; 0 is the default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            self._manager = CPUOffloadingManager(
                num_blocks=self.num_blocks,
                cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                enable_events=enable_events,
                store_threshold=store_threshold,
                max_tracker_size=max_tracker_size,
            )
        return self._manager

    def create_handlers(self, kv_caches: CanonicalKVCaches) -> CpuGpuOffloadingHandlers:
        return CpuGpuOffloadingHandlers(
            kv_caches=kv_caches,
            block_size_factor=self.block_size_factor,
            num_cpu_blocks=self.num_blocks,
            tp_dedup_enabled=self.save_only_first_rank,
        )

    @override
    def get_handlers(
        self, kv_caches: CanonicalKVCaches
    ) -> Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]:
        if not self._handlers:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike "
                    "and XPU GPUs"
                )
            self._handlers = self.create_handlers(kv_caches)

        assert self._handlers is not None
        yield GPULoadStoreSpec, CPULoadStoreSpec, self._handlers.gpu_to_cpu_handler
        yield CPULoadStoreSpec, GPULoadStoreSpec, self._handlers.cpu_to_gpu_handler
