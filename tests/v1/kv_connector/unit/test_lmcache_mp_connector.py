# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import enum
import importlib
import logging
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

MODULE_NAME = "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_mp_connector"


def _make_fake_lmcache_modules() -> dict[str, types.ModuleType]:
    class RequestType(enum.Enum):
        CLEAR = enum.auto()

    class ExternalLMCacheMPConnector:
        @property
        def role(self):
            return self._role

    def send_lmcache_request(_mq_client, _request_type, _payloads):
        raise AssertionError("send_lmcache_request should be monkeypatched")

    lmcache_mod = types.ModuleType("lmcache")
    utils_mod = types.ModuleType("lmcache.utils")
    utils_mod.init_logger = logging.getLogger  # type: ignore[attr-defined]

    integration_mod = types.ModuleType("lmcache.integration")
    vllm_mod = types.ModuleType("lmcache.integration.vllm")
    vllm_utils_mod = types.ModuleType("lmcache.integration.vllm.utils")
    vllm_utils_mod.mla_enabled = (  # type: ignore[attr-defined]
        lambda _model_config: False
    )

    mp_adapter_mod = types.ModuleType(
        "lmcache.integration.vllm.vllm_multi_process_adapter"
    )
    mp_adapter_mod.LMCacheMPSchedulerAdapter = object  # type: ignore[attr-defined]
    mp_adapter_mod.LMCacheMPWorkerAdapter = object  # type: ignore[attr-defined]
    mp_adapter_mod.LoadStoreOp = object  # type: ignore[attr-defined]
    mp_adapter_mod.ParallelStrategy = object  # type: ignore[attr-defined]
    mp_adapter_mod.RequestType = RequestType  # type: ignore[attr-defined]
    mp_adapter_mod.send_lmcache_request = (  # type: ignore[attr-defined]
        send_lmcache_request
    )

    external_connector_mod = types.ModuleType(
        "lmcache.integration.vllm.lmcache_mp_connector"
    )
    external_connector_mod.LMCacheMPConnector = (  # type: ignore[attr-defined]
        ExternalLMCacheMPConnector
    )

    v1_mod = types.ModuleType("lmcache.v1")
    multiprocess_mod = types.ModuleType("lmcache.v1.multiprocess")
    custom_types_mod = types.ModuleType("lmcache.v1.multiprocess.custom_types")
    custom_types_mod.RequestAllocationRecord = object  # type: ignore[attr-defined]

    return {
        "lmcache": lmcache_mod,
        "lmcache.utils": utils_mod,
        "lmcache.integration": integration_mod,
        "lmcache.integration.vllm": vllm_mod,
        "lmcache.integration.vllm.utils": vllm_utils_mod,
        "lmcache.integration.vllm.vllm_multi_process_adapter": mp_adapter_mod,
        "lmcache.integration.vllm.lmcache_mp_connector": external_connector_mod,
        "lmcache.v1": v1_mod,
        "lmcache.v1.multiprocess": multiprocess_mod,
        "lmcache.v1.multiprocess.custom_types": custom_types_mod,
    }


@pytest.fixture()
def lmcache_mp_module(monkeypatch):
    old_module = sys.modules.pop(MODULE_NAME, None)
    for name, module in _make_fake_lmcache_modules().items():
        monkeypatch.setitem(sys.modules, name, module)

    module = importlib.import_module(MODULE_NAME)
    yield module

    sys.modules.pop(MODULE_NAME, None)
    if old_module is not None:
        sys.modules[MODULE_NAME] = old_module


def test_reset_cache_scheduler_delegates_to_adapter(lmcache_mp_module):
    adapter = SimpleNamespace(reset_cache=MagicMock(return_value=True))
    connector = SimpleNamespace(
        role=lmcache_mp_module.KVConnectorRole.SCHEDULER,
        request_trackers={"req": object()},
        scheduler_adapter=adapter,
    )

    assert lmcache_mp_module._reset_lmcache_mp_connector(connector) is True

    adapter.reset_cache.assert_called_once_with()
    assert connector.request_trackers == {}


def test_reset_cache_scheduler_propagates_adapter_failure(lmcache_mp_module):
    adapter = SimpleNamespace(reset_cache=MagicMock(return_value=False))
    connector = SimpleNamespace(
        role=lmcache_mp_module.KVConnectorRole.SCHEDULER,
        request_trackers={},
        scheduler_adapter=adapter,
    )

    assert lmcache_mp_module._reset_lmcache_mp_connector(connector) is False


def test_reset_cache_scheduler_fallback_sends_clear_request(
    lmcache_mp_module, monkeypatch
):
    future = MagicMock()
    mq_client = object()
    send_request = MagicMock(return_value=future)
    monkeypatch.setattr(lmcache_mp_module, "send_lmcache_request", send_request)

    scheduler_adapter = SimpleNamespace(
        mq_client=mq_client,
        _mq_timeout=1.5,
        lookup_futures={"req": object()},
        _pending_lookups={"req": object()},
        _finished_lookup_results={"req": object()},
    )
    connector = SimpleNamespace(
        role=lmcache_mp_module.KVConnectorRole.SCHEDULER,
        request_trackers={"req": object()},
        scheduler_adapter=scheduler_adapter,
    )

    assert lmcache_mp_module._reset_lmcache_mp_connector(connector) is True

    send_request.assert_called_once_with(
        mq_client, lmcache_mp_module.RequestType.CLEAR, []
    )
    future.result.assert_called_once_with(timeout=1.5)
    assert connector.request_trackers == {}
    assert scheduler_adapter.lookup_futures == {}
    assert scheduler_adapter._pending_lookups == {}
    assert scheduler_adapter._finished_lookup_results == {}


def test_reset_cache_worker_is_noop(lmcache_mp_module):
    connector = SimpleNamespace(role=lmcache_mp_module.KVConnectorRole.WORKER)

    assert lmcache_mp_module._reset_lmcache_mp_connector(connector) is None


def test_external_mp_connector_without_reset_cache_is_wrapped(lmcache_mp_module):
    adapter = SimpleNamespace(reset_cache=MagicMock(return_value=True))
    connector = lmcache_mp_module.LMCacheMPConnector()
    connector._role = lmcache_mp_module.KVConnectorRole.SCHEDULER
    connector.request_trackers = {"req": object()}
    connector.scheduler_adapter = adapter

    assert lmcache_mp_module.LMCacheMPConnector is not (
        lmcache_mp_module.LMCacheMPConnectorUpstream
    )
    assert connector.reset_cache() is True
    adapter.reset_cache.assert_called_once_with()
    assert connector.request_trackers == {}
