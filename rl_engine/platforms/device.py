# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 RL-Kernel Contributors

import importlib

import torch

from rl_engine.platforms.constants import DeviceType
from rl_engine.utils.logger import logger


def _npu_available() -> bool:
    """torch.npu only exists after torch_npu is imported; probe defensively."""
    try:
        import torch_npu  # noqa: F401

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def is_musa_available() -> bool:
    """Return whether torch_musa is installed and a MUSA device is usable."""
    try:
        importlib.import_module("torch_musa")
        return bool(hasattr(torch, "musa") and torch.musa.is_available())
    except Exception as exc:
        logger.warning("MUSA availability check failed: %s", exc)
        return False


def cuda_platform() -> str:
    """Identify the vendor behind PyTorch's CUDA-facing device API."""
    if getattr(torch.version, "hip", None) is not None:
        return DeviceType.ROCM.value
    if getattr(torch.version, "maca", None) is not None:
        return DeviceType.MACA.value
    return DeviceType.CUDA.value


class DeviceContext:
    """
    Hardware-aware context manager for high-performance RL tasks.

    Identifies the vendor runtime separately from the PyTorch device interface
    for CUDA, ROCm, MACA, MUSA, and Ascend devices.
    """

    def __init__(self):
        if is_musa_available():
            self.device = torch.device(DeviceType.MUSA.value)
        elif torch.cuda.is_available():
            self.device = torch.device(DeviceType.CUDA.value)
        else:
            self.device = torch.device(DeviceType.CPU.value)
        self.is_rocm = False
        self.is_musa = False
        self.is_maca = False
        self.backend_version = "N/A"
        self.device_type = DeviceType.CPU.value

        if self.device.type == DeviceType.MUSA.value:
            self.is_musa = True
            self.device_type = DeviceType.MUSA.value
            self.backend_version = str(getattr(torch.version, "musa", "N/A"))
            logger.info_once(
                "RL-Engine initialized with Moore Threads MUSA backend"
                f" (Version: {self.backend_version})"
            )
        elif self.device.type == DeviceType.CUDA.value:
            # Tensor device strings describe the API, not the GPU vendor.
            platform = cuda_platform()
            if platform == DeviceType.ROCM.value:
                self.is_rocm = True
                self.device_type = DeviceType.ROCM.value
                self.backend_version = torch.version.hip
                logger.info_once(
                    f"RL-Engine initialized with AMD ROCm backend (Version: {self.backend_version})"
                )
            elif platform == DeviceType.MACA.value:
                self.is_maca = True
                self.device_type = DeviceType.MACA.value
                self.backend_version = torch.version.maca
                logger.info_once(
                    f"RL-Engine initialized with MetaX MACA backend"
                    f" (Version: {self.backend_version})"
                )
            else:
                self.is_rocm = False
                self.device_type = DeviceType.CUDA.value
                self.backend_version = torch.version.cuda
                logger.info_once(
                    f"RL-Engine initialized with NVIDIA CUDA backend"
                    f" (Version: {self.backend_version})"
                )
        else:
            if _npu_available():
                self.device = torch.device(DeviceType.NPU.value)
                self.device_type = DeviceType.NPU.value
                self.backend_version = getattr(torch.version, "cann", None) or "N/A"
                logger.info_once(
                    f"RL-Engine initialized with Huawei Ascend NPU backend"
                    f" (CANN Version: {self.backend_version})"
                )
            else:
                self.device_type = DeviceType.CPU.value
                logger.warning("No GPU detected. RL-Engine is falling back to CPU mode.")

    def get_preferred_dtype(self):
        """
        Returns the optimal data type for the current hardware.
        AMD ROCm and Moore Threads MUSA typically use bfloat16 for RL workloads.
        """
        return torch.bfloat16 if self.is_rocm or self.is_musa else torch.float16


device_ctx = DeviceContext()
