"""
Unit tests for jvd.core.device.DeviceManager.

Tests cover:
    - Singleton guarantee
    - GPU unavailability raises HardwareConstraintError
    - CPU initialization succeeds without VRAM constraints
    - Unrecognized device flag raises HardwareConstraintError
    - Double initialization is a no-op
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jvd.core.device import DeviceManager
from jvd.core.exceptions import HardwareConstraintError


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the DeviceManager singleton before each test."""
    DeviceManager._reset()
    yield
    DeviceManager._reset()


class TestSingleton:
    """Verify the Singleton pattern."""

    def test_same_instance(self):
        a = DeviceManager.get_instance()
        b = DeviceManager.get_instance()
        assert a is b

    def test_new_returns_same_object(self):
        a = DeviceManager()
        b = DeviceManager()
        assert a is b


class TestCPUInitialization:
    """Verify CPU mode works without CUDA."""

    def test_cpu_device(self):
        dm = DeviceManager.get_instance()
        dm.initialize(device_flag="cpu")
        assert dm.device.type == "cpu"
        assert not dm.is_gpu

    def test_double_init_is_noop(self):
        dm = DeviceManager.get_instance()
        dm.initialize(device_flag="cpu")
        # Should log a warning but not raise
        dm.initialize(device_flag="gpu")
        assert dm.device.type == "cpu"


class TestGPUValidation:
    """Verify strict GPU enforcement."""

    @patch("jvd.core.device.torch.cuda.is_available", return_value=False)
    def test_gpu_unavailable_raises(self, mock_cuda):
        dm = DeviceManager.get_instance()
        with pytest.raises(HardwareConstraintError, match="GPU was explicitly requested"):
            dm.initialize(device_flag="gpu")

    def test_bad_device_flag_raises(self):
        dm = DeviceManager.get_instance()
        with pytest.raises(HardwareConstraintError, match="Unrecognized device flag"):
            dm.initialize(device_flag="tpu")


class TestDeviceNotInitialized:
    """Verify accessing .device before init raises."""

    def test_device_access_before_init(self):
        dm = DeviceManager.get_instance()
        with pytest.raises(HardwareConstraintError, match="not been initialized"):
            _ = dm.device


class TestVRAMFraction:
    """Verify VRAM fraction computation."""

    def test_fraction_clamped_low(self):
        # 0.1 GB out of 4 GB = 0.025 → clamped to 0.10
        result = DeviceManager._compute_fraction(0.1, 4.0)
        assert result == 0.10

    def test_fraction_clamped_high(self):
        # 3.9 GB out of 4.0 GB = 0.975 → clamped to 0.95
        result = DeviceManager._compute_fraction(3.9, 4.0)
        assert result == 0.95

    def test_fraction_normal(self):
        # 2.8 GB out of 4.0 GB = 0.70 → within bounds
        result = DeviceManager._compute_fraction(2.8, 4.0)
        assert abs(result - 0.70) < 1e-6
