# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import pytest
import torch

from rl_engine.platforms import device as device_module
from rl_engine.runtime import registry as registry_module
from rl_engine.runtime.registry import KernelRegistry, OpBackend


@pytest.mark.parametrize(
    "hip,maca,platform,version",
    [
        (None, "3.8.2.2", "maca", "3.8.2.2"),
        ("6.4", None, "rocm", "6.4"),
        (None, None, "cuda", "12.8"),
    ],
)
def test_cuda_api_vendor_identity(monkeypatch, hip, maca, platform, version):
    monkeypatch.setattr(device_module, "is_musa_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.version, "maca", maca, raising=False)
    monkeypatch.setattr(torch.version, "cuda", "12.8")

    context = device_module.DeviceContext()
    assert context.device.type == "cuda"
    assert context.device_type == platform
    assert context.backend_version == version
    assert context.is_maca == (platform == "maca")
    monkeypatch.setattr(registry_module, "device_ctx", context)
    registry = KernelRegistry()
    assert registry._platform() == platform
    assert registry._platform_for_device("cuda:0") == platform
    assert registry._platform_for_device("cpu") == "cpu"


def test_maca_torch_without_visible_gpu_uses_cpu(monkeypatch):
    monkeypatch.setattr(device_module, "is_musa_available", lambda: False)
    monkeypatch.setattr(device_module, "_npu_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "maca", "3.8.2.2", raising=False)
    context = device_module.DeviceContext()
    assert context.device.type == "cpu"
    assert context.device_type == "cpu"


def test_maca_dispatch_uses_operator_specific_pytorch_implementations(monkeypatch):
    monkeypatch.setattr(registry_module.device_ctx, "device_type", "maca")
    monkeypatch.setattr(registry_module.device_ctx, "is_rocm", False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "maca", "3.8.2.2", raising=False)
    registry = KernelRegistry()
    for operator, expected in (
        ("linear_logp", OpBackend.PYTORCH_LINEAR_LOGP),
        ("grpo_loss", OpBackend.PYTORCH_GRPO_LOSS),
        ("attn", OpBackend.PYTORCH_ATTN),
    ):
        chosen = registry.get_op(operator, device="cuda:0")
        assert type(chosen) is registry._load_backend(expected)
    with pytest.raises(RuntimeError, match="No functional backend found for det_gemm on maca"):
        registry.get_op("det_gemm", device="cuda:0")
