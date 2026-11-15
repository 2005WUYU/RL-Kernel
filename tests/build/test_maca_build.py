# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import torch

from build_tools import extensions


def test_maca_build_selects_generic_logp_without_nvidia_dependencies(monkeypatch):
    monkeypatch.setattr(torch.version, "maca", "3.8.2.2", raising=False)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("KERNEL_ALIGN_FORCE_SM90", "1")
    monkeypatch.setenv("KERNEL_ALIGN_DET_GEMM_SM90", "1")
    monkeypatch.setattr(
        extensions,
        "_load_torch_extension_tools",
        lambda: (torch, None, lambda **kwargs: kwargs),
    )
    (extension,) = extensions.get_extensions()
    assert extension["sources"] == [
        "csrc/maca/ops.cpp",
        "csrc/cuda/logprob/fused_logp_kernel.cu",
    ]
    assert "-lcuda" not in extension["extra_link_args"]
    flags = extension["extra_compile_args"]["nvcc"]
    assert "-DKERNEL_ALIGN_WITH_MACA" in flags
    assert not any(flag.startswith(("-gencode", "--expt-", "-Xfatbin")) for flag in flags)
