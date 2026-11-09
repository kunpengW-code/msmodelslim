#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
FusedTensorCache 单元测试
"""

import threading

import torch

from msmodelslim.core.quant_service.modelslim_convert.weight_mapping.fused_cache import FusedTensorCache


class TestFusedTensorCache:
    """测试 FusedTensorCache 类"""

    def test_get_or_load_call_loader_once_when_same_key_requested_twice(self):
        cache = FusedTensorCache()
        calls = {"n": 0}

        def loader():
            calls["n"] += 1
            return torch.ones(2, 2)

        t1 = cache.get_or_load("fused", "cpu", loader)
        t2 = cache.get_or_load("fused", "cpu", loader)
        assert calls["n"] == 1
        assert torch.equal(t1, t2)

    def test_get_or_load_call_loader_per_device_when_device_differs(self):
        cache = FusedTensorCache()
        calls = {"n": 0}

        def loader():
            calls["n"] += 1
            return torch.ones(2, 2)

        cache.get_or_load("fused", "cpu", loader)
        cache.get_or_load("fused", "npu", loader)
        assert calls["n"] == 2

    def test_get_or_load_serve_same_tensor_when_concurrent_same_key(self):
        cache = FusedTensorCache()

        def loader():
            return torch.arange(4, dtype=torch.float32).reshape(2, 2)

        barrier = threading.Barrier(4)
        outputs: list[torch.Tensor] = []

        def worker():
            barrier.wait()
            outputs.append(cache.get_or_load("fused", "cpu", loader))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(outputs) == 4
        for out in outputs[1:]:
            assert torch.equal(outputs[0], out)
        assert cache.misses >= 1
        assert cache.hits + cache.misses == 4
