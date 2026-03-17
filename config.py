"""
Configuration system for EvoLLM with automatic resource discovery.
Provides dataclass for all configuration options and auto-detection logic.
"""

from dataclasses import dataclass
from typing import Optional, Literal
import psutil
import torch


@dataclass
class EvoLLMConfig:
    """
    Configuration for EvoLLM inference system with automatic discovery.

    By default, EvoLLM automatically detects hardware and configures itself.
    Override any setting by explicitly providing it.

    Memory Budgets
    --------------
    gpu_layers : int (default: 0 = auto)
        Number of layers to keep in GPU VRAM. 0 = auto-detect based on VRAM.
    cpu_cache_gb : float (default: 0.0 = auto)
        Maximum RAM for layer cache in GB. 0.0 = auto-detect from available RAM.
    max_ram_percent : float (default: 0.7)
        When auto-detecting CPU cache, use this % of available RAM.

    Performance Tuning
    -----------------
    prefetch_depth : int (default: 0 = auto)
        Number of layers to prefetch ahead. 0 = auto-detect based on disk speed.
    prefetch_async : bool (default: True)
        Use async loading with ThreadPoolExecutor.
    prefetch_batches : int (default: 1)
        Number of prefetch batches to run in parallel.
    cache_policy : Literal['lru', 'freq', 'adaptive'] (default: 'lru')
        Cache eviction policy.
    cache_warmup : bool (default: True)
        Warm cache on first generation pass.
    gpu_multi_layer_caching : bool (default: auto)
        Keep multiple layers in GPU. Auto=True when gpu_layers>1.

    Compression
    -----------
    compression : Optional[Literal['4bit', '8bit']] (default: None)
        Enable weight compression.

    Hardware
    --------
    device : str (default: 'auto')
        Device for computation. 'auto' = cuda if available, else cpu.
    dtype : torch.dtype (default: torch.float16)
        Data type for computation.

    Advanced
    --------
    enable_profiling : bool (default: False)
        Enable detailed performance profiling.
    prefetch_stream : bool (default: True)
        Use separate CUDA stream for prefetching (GPU only).
    evict_after_use : bool (default: True)
        Evict layers from GPU after use unless caching enabled.
    """

    # Memory budgets (0 = auto-detect)
    gpu_layers: int = 0
    cpu_cache_gb: float = 0.0
    max_ram_percent: float = 0.7

    # Performance tuning (0 = auto-detect except where noted)
    prefetch_depth: int = 0
    prefetch_async: bool = True
    prefetch_batches: int = 1
    cache_policy: Literal['lru', 'freq', 'adaptive'] = 'lru'
    cache_warmup: bool = True
    gpu_multi_layer_caching: Optional[bool] = None

    # Compression
    compression: Optional[Literal['4bit', '8bit']] = None

    # Hardware
    device: str = 'auto'
    dtype: torch.dtype = torch.float16

    # Advanced
    enable_profiling: bool = False
    prefetch_stream: bool = True
    evict_after_use: bool = True

    def __post_init__(self):
        """Validate configuration"""
        if self.gpu_layers < 0:
            raise ValueError("gpu_layers must be >= 0")
        if self.cpu_cache_gb < 0:
            raise ValueError("cpu_cache_gb must be >= 0")
        if not (0.0 < self.max_ram_percent <= 1.0):
            raise ValueError("max_ram_percent must be in (0.0, 1.0]")
        if self.prefetch_depth < 0:
            raise ValueError("prefetch_depth must be >= 0")
        if self.prefetch_batches < 1:
            raise ValueError("prefetch_batches must be >= 1")


def _quick_disk_speed_test(file_size_mb: int = 50) -> float:
    """
    Quick test of disk read/write speed in MB/s.
    Returns average of write/read speed.
    """
    import tempfile
    import time
    import os

    test_dir = tempfile.gettempdir()
    test_file = os.path.join(test_dir, ".fitllm_disk_speed_test.tmp")

    test_bytes = file_size_mb * 1024 * 1024
    chunk_size = 1024 * 1024  # 1MB chunks

    try:
        # Write test
        data = os.urandom(chunk_size)
        start = time.time()
        with open(test_file, 'wb') as f:
            for _ in range(file_size_mb):
                f.write(data)
        write_time = time.time() - start

        # Read test
        start = time.time()
        with open(test_file, 'rb') as f:
            while f.read(chunk_size):
                pass
        read_time = time.time() - start

        os.remove(test_file)

        # Average MB/s
        mb_written = file_size_mb
        mb_read = file_size_mb
        total_mb = mb_written + mb_read
        total_time = write_time + read_time

        return total_mb / total_time if total_time > 0 else 0.0

    except Exception as e:
        raise RuntimeError(f"Disk speed test failed: {e}")


def auto_config(model_size_b: Optional[float] = None,
                checkpoint_path: Optional[str] = None,
                verbose: bool = True) -> EvoLLMConfig:
    """
    Auto-detect hardware and configure EvoLLM optimally.

    This is the core automatic resource discovery system. It detects:

    1. System RAM → configures CPU cache size
    2. GPU VRAM → configures layers to keep in GPU
    3. Disk speed → configures prefetch depth
    4. Model size → optimizes all of the above

    All detections can be overridden by setting fields explicitly in the returned config.

    Parameters
    ----------
    model_size_b : float, optional
        Model size in bytes. If None, estimated from checkpoint_path.
    checkpoint_path : str, optional
        Path to split model directory (for model size estimation).
    verbose : bool
        Print detection summary if True.

    Returns
    -------
    EvoLLMConfig
        Automatically configured instance

    Detection Logic
    ---------------
    CPU cache:
      - < 8GB usable RAM → no cache
      - 8-16GB usable → cache 4-8GB
      - 16-32GB usable → cache 8-16GB
      - 32-64GB usable → cache 16-32GB
      - > 64GB usable → cache 32-48GB

    GPU layers:
      - < 4GB VRAM → 0 layers (CPU only)
      - 4-8GB VRAM → 1 layer
      - 8-16GB VRAM → 2-3 layers
      - > 16GB VRAM → 3-4 layers

    Prefetch depth:
      - NVMe (2000+ MB/s) → 3-4
      - SATA SSD (500-2000 MB/s) → 2-3
      - HDD (<500 MB/s) → 1-2
    """
    config = EvoLLMConfig()

    # ========== STEP 1: Detect System RAM ==========
    try:
        ram_gb = psutil.virtual_memory().total / 1e9
        available_ram_gb = psutil.virtual_memory().available / 1e9

        # Reserve for OS + other processes
        os_reserve_gb = max(8.0, ram_gb * 0.20)
        usable_ram_gb = max(0.0, available_ram_gb - os_reserve_gb)

        if verbose:
            print(f"[EvoLLM] Detected RAM: {ram_gb:.1f}GB total, {available_ram_gb:.1f}GB available, {usable_ram_gb:.1f}GB usable")

    except Exception as e:
        print(f"Warning: Could not detect RAM: {e}")
        ram_gb = 0
        usable_ram_gb = 0

    # ========== STEP 2: Estimate Model Size ==========
    est_layer_size_gb = 2.0  # Conservative default
    total_layers_estimate = 80

    if model_size_b is not None:
        model_gb = model_size_b / 1e9
        est_layer_size_gb = model_gb / total_layers_estimate
        if verbose:
            print(f"[EvoLLM] Provided model size: {model_gb:.1f}GB → ~{est_layer_size_gb:.2f}GB per layer")
    elif checkpoint_path:
        try:
            from .config import estimate_model_size
            estimated_size = estimate_model_size(checkpoint_path)
            if estimated_size:
                model_gb = estimated_size / 1e9
                est_layer_size_gb = model_gb / total_layers_estimate
                if verbose:
                    print(f"[EvoLLM] Estimated model size: {model_gb:.1f}GB → ~{est_layer_size_gb:.2f}GB per layer")
        except Exception as e:
            if verbose:
                print(f"[EvoLLM] Could not estimate model size: {e}, using default {est_layer_size_gb}GB/layer")

    # ========== STEP 3: Configure CPU Cache ==========
    if config.cpu_cache_gb <= 0:  # Auto-detection
        if usable_ram_gb < 8:
            config.cpu_cache_gb = 0.0
        else:
            min_useful_layers = 5
            min_cache_gb = min_useful_layers * est_layer_size_gb
            max_possible_gb = usable_ram_gb * config.max_ram_percent

            if max_possible_gb < min_cache_gb:
                config.cpu_cache_gb = 0.0
            else:
                target_layers = min(20, int(max_possible_gb / est_layer_size_gb * 0.9))
                config.cpu_cache_gb = min(target_layers * est_layer_size_gb, max_possible_gb * 0.9)

        if config.cpu_cache_gb > 0 and config.cpu_cache_gb < 4:
            config.cpu_cache_gb = 0.0

    # ========== STEP 4: Configure GPU Caching ==========
    if config.gpu_layers <= 0:  # Auto-detection
        try:
            if torch.cuda.is_available():
                gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                gpu_name = torch.cuda.get_device_name(0)

                available_vram_gb = gpu_vram_gb * 0.8
                layers_fit = int(available_vram_gb / est_layer_size_gb)

                if gpu_vram_gb < 4:
                    config.gpu_layers = 0
                elif gpu_vram_gb < 8:
                    config.gpu_layers = min(1, layers_fit)
                elif gpu_vram_gb < 16:
                    config.gpu_layers = min(2, layers_fit)
                else:
                    config.gpu_layers = min(4, layers_fit)

                if config.gpu_layers > 1:
                    config.gpu_multi_layer_caching = True if config.gpu_multi_layer_caching is None else config.gpu_multi_layer_caching
                else:
                    config.gpu_layers = 0
                    config.gpu_multi_layer_caching = False

                if verbose:
                    print(f"[EvoLLM] Detected GPU: {gpu_name} ({gpu_vram_gb:.1f}GB)")
            else:
                config.gpu_layers = 0
                config.gpu_multi_layer_caching = False
        except Exception as e:
            if verbose:
                print(f"Warning: GPU detection failed: {e}")
            config.gpu_layers = 0
            config.gpu_multi_layer_caching = False

    # ========== STEP 5: Configure Prefetch Depth ==========
    if config.prefetch_depth <= 0:  # Auto-detection
        try:
            disk_speed_mb_s = _quick_disk_speed_test()
            if verbose:
                print(f"[EvoLLM] Disk speed: {disk_speed_mb_s:.0f} MB/s")

            if disk_speed_mb_s >= 2000:
                config.prefetch_depth = 3 if usable_ram_gb > 16 else 2
            elif disk_speed_mb_s >= 800:
                config.prefetch_depth = 2
            elif disk_speed_mb_s >= 200:
                config.prefetch_depth = 1
            else:
                config.prefetch_depth = 1
        except Exception:
            config.prefetch_depth = 2  # Safe default

    # ========== STEP 6: Set Device ==========
    if config.device == 'auto':
        config.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # ========== STEP 7: Validation ==========
    if config.cpu_cache_gb > 0 or config.gpu_layers > 0:
        if verbose:
            print(f"\n[EvoLLM] Auto-configured for optimal performance:")
            if config.cpu_cache_gb > 0:
                print(f"  CPU cache: {config.cpu_cache_gb:.1f}GB")
            if config.gpu_layers > 0:
                print(f"  GPU cache: {config.gpu_layers} layers")
            print(f"  Prefetch depth: {config.prefetch_depth}")
            print("")

    return config


def estimate_model_size(model_path: str) -> Optional[float]:
    """
    Estimate model size in bytes from a HuggingFace model path.

    Parameters
    ----------
    model_path : str
        Path to HuggingFace model directory

    Returns
    -------
    float or None
        Total size in bytes of all model shard files
    """
    try:
        import json
        from pathlib import Path

        index_path = Path(model_path) / "pytorch_model.bin.index.json"
        if not index_path.exists():
            index_path = Path(model_path) / "model.safetensors.index.json"

        if not index_path.exists():
            return None

        with open(index_path, 'r') as f:
            index = json.load(f)

        weight_map = index.get('weight_map', {})
        shard_files = set(weight_map.values())

        total_size = 0
        for shard in shard_files:
            shard_path = Path(model_path) / shard
            if shard_path.exists():
                total_size += shard_path.stat().st_size

        return total_size if total_size > 0 else None

    except Exception as e:
        print(f"Warning: Could not estimate model size: {e}")
        return None


def validate_config(config: 'EvoLLMConfig', model_size_b: float) -> 'EvoLLMConfig':
    """
    Validate configuration against available hardware and model size.
    Prints warnings if resources are insufficient.
    """
    ram_gb = psutil.virtual_memory().total / 1e9
    requested_cache_gb = config.cpu_cache_gb if config.cpu_cache_gb > 0 else config.max_ram_percent * ram_gb

    if requested_cache_gb > 0:
        model_gb = model_size_b / 1e9
        min_required = model_gb + requested_cache_gb + 8
        if min_required > ram_gb:
            print(f"WARNING: Requested cache ({requested_cache_gb:.1f}GB) + model ({model_gb:.1f}GB) "
                  f"exceeds available RAM ({ram_gb:.1f}GB). Consider reducing cpu_cache_gb.")

    if config.gpu_layers > 0 and torch.cuda.is_available():
        gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        est_layer_gb = model_size_b / 1e9 / 80
        if config.gpu_layers * est_layer_gb > gpu_vram_gb * 0.8:
            print(f"WARNING: Requested gpu_layers={config.gpu_layers} may exceed GPU VRAM "
                  f"({gpu_vram_gb:.1f}GB). Consider reducing gpu_layers.")

    return config
