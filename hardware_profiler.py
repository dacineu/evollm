"""
Hardware profiling for EvoLLM.
Auto-detects system capabilities and provides configuration recommendations.
"""

import psutil
import torch
import time
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import subprocess


@dataclass
class HardwareProfile:
    """Stores detected hardware capabilities"""
    # CPU
    cpu_cores: int = 0
    cpu_ram_gb: float = 0.0

    # GPU
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    gpu_compute_capability: Optional[str] = None

    # PCIe
    pcie_bandwidth_gb_s: float = 0.0

    # Disk
    disk_speed_mb_s: float = 0.0
    disk_type: str = "unknown"  # 'nvme', 'sata', 'hdd'

    # System
    platform: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'cpu_cores': self.cpu_cores,
            'cpu_ram_gb': self.cpu_ram_gb,
            'gpu_available': self.gpu_available,
            'gpu_name': self.gpu_name,
            'gpu_vram_gb': self.gpu_vram_gb,
            'gpu_compute_capability': self.gpu_compute_capability,
            'pcie_bandwidth_gb_s': self.pcie_bandwidth_gb_s,
            'disk_speed_mb_s': self.disk_speed_mb_s,
            'disk_type': self.disk_type,
            'platform': self.platform,
        }

    def __str__(self) -> str:
        lines = [
            "=== Hardware Profile ===",
            f"Platform: {self.platform}",
            f"CPU: {self.cpu_cores} cores",
            f"RAM: {self.cpu_ram_gb:.1f} GB",
        ]

        if self.gpu_available:
            lines.extend([
                f"GPU: {self.gpu_name}",
                f"GPU VRAM: {self.gpu_vram_gb:.1f} GB",
                f"Compute Capability: {self.gpu_compute_capability}",
                f"PCIe Bandwidth: {self.pcie_bandwidth_gb_s:.1f} GB/s",
            ])
        else:
            lines.append("GPU: Not available (CPU-only mode)")

        lines.extend([
            f"Disk: {self.disk_type.upper()} ({self.disk_speed_mb_s:.0f} MB/s)",
            "========================="
        ])

        return "\n".join(lines)


class HardwareProfiler:
    """
    Profiles system hardware capabilities.

    Methods
    -------
    profile() -> HardwareProfile
        Run all detection and profiling tests
    recommend_config(profile, model_size_b) -> EvoLLMConfig
        Generate optimal configuration based on profile
    """

    def __init__(self, quick: bool = False):
        """
        Parameters
        ----------
        quick : bool
            If True, skip slow benchmarks (disk speed test)
        """
        self.quick = quick

    def profile(self) -> HardwareProfile:
        """
        Perform comprehensive hardware profiling.

        Returns
        -------
        HardwareProfile
            Detected hardware capabilities
        """
        profile = HardwareProfile()

        # Platform
        profile.platform = os.uname().sysname if hasattr(os, 'uname') else "Unknown"

        # CPU info
        profile.cpu_cores = os.cpu_count() or 1

        # RAM
        ram = psutil.virtual_memory()
        profile.cpu_ram_gb = ram.total / 1e9

        # GPU
        if torch.cuda.is_available():
            self._profile_gpu(profile)
        else:
            profile.gpu_available = False

        # Disk (maybe slow)
        if not self.quick:
            self._profile_disk(profile)
        else:
            profile.disk_speed_mb_s = 500  # Assumption for quick mode
            profile.disk_type = "nvme"

        # PCIe bandwidth (estimate from GPU specs or test)
        if profile.gpu_available:
            self._estimate_pcie_bandwidth(profile)

        return profile

    def _profile_gpu(self, profile: HardwareProfile):
        """Detect GPU capabilities"""
        try:
            props = torch.cuda.get_device_properties(0)
            profile.gpu_name = props.name
            profile.gpu_vram_gb = props.total_memory / 1e9
            profile.gpu_compute_capability = f"{props.major}.{props.minor}"

            # Try to get PCIe bandwidth from nvidia-smi if available
            if profile.gpu_name.lower().startswith('nvidia'):
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=pcie.link.gen.current,pcie.link.width.current', '--format=csv,noheader'],
                        capture_output=True, text=True, timeout=2
                    )
                    if result.returncode == 0:
                        gen, width = result.stdout.strip().split(',')
                        gen = int(gen.strip())
                        width = int(width.strip())

                        # PCIe bandwidth per lane (GB/s)
                        pcie_bandwidth_per_lane = {
                            1: 0.984,
                            2: 1.969,
                            3: 3.938,
                            4: 7.877,
                            5: 15.754,
                        }.get(gen, 0.984)

                        profile.pcie_bandwidth_gb_s = pcie_bandwidth_per_lane * width
                except Exception:
                    # Fallback: assume PCIe 3.0 x16
                    profile.pcie_bandwidth_gb_s = 15.75
        except Exception as e:
            print(f"Warning: GPU profiling error: {e}")
            profile.gpu_available = False

    def _estimate_pcie_bandwidth(self, profile: HardwareProfile):
        """Estimate PCIe bandwidth if not already set"""
        if profile.pcie_bandwidth_gb_s <= 0:
            # Conservative estimate based on GPU age
            if profile.gpu_vram_gb >= 24:
                # High-end card, likely PCIe 4.0+
                profile.pcie_bandwidth_gb_s = 16.0
            else:
                # Assume PCIe 3.0 x16
                profile.pcie_bandwidth_gb_s = 15.75

    def _profile_disk(self, profile: HardwareProfile):
        """
        Test disk speed by writing/reading a temporary file.
        Not super accurate but gives ballpark.
        """
        import tempfile
        import shutil

        # Find a temp location on same disk as model storage
        # Use current directory for test
        test_dir = "."
        test_file = os.path.join(test_dir, ".fitllm_disk_test.tmp")

        test_size = 100 * 1024 * 1024  # 100MB

        try:
            # Write test
            data = os.urandom(1024 * 1024)  # 1MB random
            start = time.time()
            with open(test_file, 'wb') as f:
                for _ in range(100):
                    f.write(data)
            write_time = time.time() - start

            # Read test
            start = time.time()
            with open(test_file, 'rb') as f:
                while f.read(1024 * 1024):
                    pass
            read_time = time.time() - start

            os.remove(test_file)

            # Calculate throughput (MB/s)
            write_mb_s = test_size / 1024 / 1024 / write_time if write_time > 0 else 0
            read_mb_s = test_size / 1024 / 1024 / read_time if read_time > 0 else 0
            avg_mb_s = (write_mb_s + read_mb_s) / 2

            profile.disk_speed_mb_s = avg_mb_s

            # Classify disk type
            if avg_mb_s >= 2000:
                profile.disk_type = "nvme"
            elif avg_mb_s >= 500:
                profile.disk_type = "sata ssd"
            else:
                profile.disk_type = "hdd"

        except Exception as e:
            print(f"Warning: Disk profiling failed: {e}")
            profile.disk_speed_mb_s = 500  # Default assumption
            profile.disk_type = "unknown"

    def recommend_config(self,
                         profile: HardwareProfile,
                         model_size_b: float,
                         safety_factor: float = 0.8) -> 'EvoLLMConfig':
        """
        Generate optimal EvoLLM configuration based on hardware profile.

        Parameters
        ----------
        profile : HardwareProfile
            Hardware profile
        model_size_b : float
            Model size in bytes
        safety_factor : float
            Safety margin for memory (0.0-1.0)

        Returns
        -------
        EvoLLMConfig
            Recommended configuration
        """
        from .config import EvoLLMConfig

        config = EvoLLMConfig()

        # Calculate layer size (rough estimate)
        # Most models have ~80 transformer layers
        est_layer_size_gb = model_size_b / 1e9 / 80

        # Configure CPU cache based on RAM
        if profile.cpu_ram_gb >= 64 and est_layer_size_gb * 20 < profile.cpu_ram_gb * safety_factor:
            config.cpu_cache_gb = 48.0
        elif profile.cpu_ram_gb >= 32 and est_layer_size_gb * 10 < profile.cpu_ram_gb * safety_factor:
            config.cpu_cache_gb = 24.0
        elif profile.cpu_ram_gb >= 16 and est_layer_size_gb * 5 < profile.cpu_ram_gb * safety_factor:
            config.cpu_cache_gb = 8.0
        else:
            config.cpu_cache_gb = 0.0  # Not enough RAM for useful cache

        # Configure GPU caching based on VRAM
        if profile.gpu_available:
            max_gpu_layers = int(profile.gpu_vram_gb / est_layer_size_gb * safety_factor)
            config.gpu_layers = min(4, max(1, max_gpu_layers))
            config.gpu_multi_layer_caching = config.gpu_layers > 1
        else:
            config.gpu_layers = 0
            config.gpu_multi_layer_caching = False

        # Configure prefetch depth based on disk speed and PCIe
        if profile.disk_speed_mb_s >= 2000:
            # NVMe: can afford deeper prefetch
            config.prefetch_depth = 3
        elif profile.disk_speed_mb_s >= 500:
            # SATA SSD: moderate prefetch
            config.prefetch_depth = 2
        else:
            # Slow disk: minimize speculative prefetch
            config.prefetch_depth = 1

        # Adjust for PCIe bandwidth if available
        if profile.pcie_bandwidth_gb_s > 0:
            if profile.pcie_bandwidth_gb_s >= 16:
                config.prefetch_depth = max(config.prefetch_depth, 3)
            elif profile.pcie_bandwidth_gb_s >= 8:
                config.prefetch_depth = max(config.prefetch_depth, 2)

        return config


def profile_and_recommend(model_size_b: Optional[float] = None,
                         quick: bool = False) -> tuple[HardwareProfile, 'EvoLLMConfig']:
    """
    Convenience function: profile hardware and recommend config.

    Parameters
    ----------
    model_size_b : float, optional
        Model size in bytes (for better recommendations)
    quick : bool
        Skip slow benchmarks if True

    Returns
    -------
    profile : HardwareProfile
        Detected hardware
    config : EvoLLMConfig
        Recommended configuration
    """
    profiler = HardwareProfiler(quick=quick)
    profile = profiler.profile()

    if model_size_b is None:
        # Use a default 70B model size (fp16: ~140GB)
        model_size_b = 140e9

    from .config import EvoLLMConfig
    config = profiler.recommend_config(profile, model_size_b)

    return profile, config
