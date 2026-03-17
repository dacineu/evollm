"""
EvoLLM: Evolving Large Language Model inference
Core module for the EvoOS distributed AI operating system.
"""

__version__ = "0.1.0"
__author__ = "EvoLLM Team"

from .config import EvoLLMConfig, auto_config
from .evollm_base import EvoLLMModel, EvoLLMAutoModel

# Re-export key classes for convenience
__all__ = [
    "EvoLLMConfig",
    "auto_config",
    "EvoLLMModel",
    "EvoLLMAutoModel",
]
