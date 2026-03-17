"""
Utility functions for EvoLLM.
Re-export common functions from AirLLM for convenience.
"""

from typing import Dict, List, Optional
import torch


def print_hardware_summary():
    """Print a summary of detected hardware"""
    print("\n" + "="*60)
    print("EvoLLM Hardware Resource Discovery")
    print("="*60)

    # CPU / RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"\n🖥️  CPU / RAM:")
        print(f"   Total RAM: {ram.total / 1e9:.1f} GB")
        print(f"   Available: {ram.available / 1e9:.1f} GB")
        print(f"   Cores: {psutil.cpu_count()} logical")
    except ImportError:
        print("\n❌ psutil not installed: cannot detect RAM")

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"\n🎮 GPU:")
            print(f"   Name: {props.name}")
            print(f"   VRAM: {props.total_memory / 1e9:.1f} GB")
            print(f"   Compute: {props.major}.{props.minor}")
        else:
            print("\n❌ No CUDA GPU detected")
    except Exception as e:
        print(f"\n❌ GPU detection failed: {e}")

    # Disk
    try:
        import tempfile
        import time
        test_file = tempfile.NamedTemporaryFile(delete=False).name
        size_mb = 50
        chunk = b'\x00' * (1024 * 1024)

        start = time.time()
        with open(test_file, 'wb') as f:
            for _ in range(size_mb):
                f.write(chunk)
        write_time = time.time() - start

        start = time.time()
        with open(test_file, 'rb') as f:
            while f.read(1024 * 1024):
                pass
        read_time = time.time() - start

        import os
        os.remove(test_file)

        speed = (size_mb * 2) / (write_time + read_time)
        disk_type = "NVMe SSD" if speed >= 2000 else "SATA SSD" if speed >= 500 else "HDD"
        print(f"\n💾 Disk:")
        print(f"   Speed: {speed:.0f} MB/s ({disk_type})")
    except Exception as e:
        print(f"\n⚠️  Disk speed test failed: {e}")

    print("\n" + "="*60)


def get_recommended_config_for_model(model_name: str) -> Dict:
    """
    Get recommended configuration for a known model.

    This is a convenience function that returns typical values for
    common models without needing to download them.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or path

    Returns
    -------
    dict
        Recommended configuration parameters

    Examples
    --------
    >>> from fitllm.utils import get_recommended_config_for_model
    >>> config = get_recommended_config_for_model("meta-llama/Llama-2-70b-hf")
    >>> print(config)
    {'cpu_cache_gb': 32, 'gpu_layers': 1, 'prefetch_depth': 3}
    """
    # Known model sizes (approximate, fp16)
    model_sizes = {
        'meta-llama/Llama-2-7b-hf': 13e9,
        'meta-llama/Llama-2-13b-hf': 25e9,
        'meta-llama/Llama-2-70b-hf': 140e9,
        'meta-llama/Llama-3-8b': 16e9,
        'meta-llama/Llama-3-70b': 140e9,
        'mistralai/Mistral-7B-v0.1': 14e9,
        'mistralai/Mixtral-8x7B-v0.1': 80e9,  # Sparse, actually larger
    }

    size = model_sizes.get(model_name)
    if not size:
        return {
            'note': f'Unknown model: {model_name}. Use auto_config() after initialization for best results.',
            'suggestion': 'EvoLLMConfig(cpu_cache_gb="auto", gpu_layers="auto", prefetch_depth="auto")'
        }

    size_gb = size / 1e9

    return {
        'model_size_gb': size_gb,
        'estimate': f"For {model_name} (~{size_gb:.0f}GB fp16)",
        'layer_size_gb_estimate': size_gb / 80,
        'recommendation': 'Use EvoLLM.from_pretrained(..., auto_config=True) for automatic tuning',
    }


def check_fitllm_readiness() -> Dict:
    """
    Check if system meets minimum requirements for EvoLLM.

    Returns
    -------
    dict
        Readiness report with warnings and status
    """
    report = {
        'ready': True,
        'warnings': [],
        'recommendations': []
    }

    # Check RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        if ram_gb < 16:
            report['warnings'].append(f"Only {ram_gb:.1f}GB RAM - very limited for large models")
            report['recommendations'].append("Consider using a smaller model (7B-13B)")
        elif ram_gb < 32:
            report['recommendations'].append("16-32GB RAM: use CPU cache for modest speedup")
    except ImportError:
        report['warnings'].append("Cannot detect RAM (psutil not installed)")
        report['ready'] = False

    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb < 4:
                report['warnings'].append(f"GPU has only {vram_gb:.1f}GB VRAM")
                report['recommendations'].append("Will run in CPU-only mode")
            elif vram_gb < 8:
                report['recommendations'].append("4-8GB GPU: expect 1 layer in VRAM")
        else:
            report['warnings'].append("No CUDA GPU - will use CPU only (much slower)")
    except Exception:
        report['warnings'].append("Cannot detect GPU")

    # Check disk
    try:
        import tempfile
        test_file = tempfile.NamedTemporaryFile(delete=False).name
        with open(test_file, 'wb') as f:
            f.write(b'\x00' * (50 * 1024 * 1024))
        import os
        os.remove(test_file)
    except Exception:
        report['warnings'].append("Cannot write to disk - may affect model loading")
        report['recommendations'].append("Ensure sufficient disk space for model (~150GB for 70B)")

    return report
