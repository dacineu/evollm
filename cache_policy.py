"""
Cache policy implementations for EvoLLM.
Provides layered caching strategies with memory bounds.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import time


@dataclass
class CacheEntry:
    """Represents a cached layer entry"""
    state_dict: Dict
    size_bytes: int
    last_access: float = field(default_factory=time.time)
    access_count: int = 0

    def touch(self):
        """Update access metadata"""
        self.last_access = time.time()
        self.access_count += 1


class CachePolicy(ABC):
    """Abstract base class for cache eviction policies"""

    @abstractmethod
    def should_evict(self, cache: 'LayerCache', new_entry: CacheEntry) -> bool:
        """Determine if eviction is needed before adding new entry"""
        pass

    @abstractmethod
    def evict(self, cache: 'LayerCache') -> Optional[str]:
        """Select and remove an entry from cache. Returns evicted key or None."""
        pass


class LRUPolicy(CachePolicy):
    """Least Recently Used eviction policy"""

    def should_evict(self, cache: 'LayerCache', new_entry: CacheEntry) -> bool:
        """Check if adding new entry would exceed capacity"""
        return cache.current_size + new_entry.size_bytes > cache.max_size_bytes

    def evict(self, cache: 'LayerCache') -> Optional[str]:
        """Evict the least recently used entry"""
        if not cache.cache:
            return None

        # Find the LRU entry (oldest last_access)
        oldest_key = min(
            cache.cache.keys(),
            key=lambda k: cache.cache[k].last_access
        )

        evicted = cache.cache.pop(oldest_key)
        cache.current_size -= evicted.size_bytes
        return oldest_key


class FREQPolicy(CachePolicy):
    """Frequency-based eviction (LFU: Least Frequently Used)"""

    def should_evict(self, cache: 'LayerCache', new_entry: CacheEntry) -> bool:
        """Check if adding new entry would exceed capacity"""
        return cache.current_size + new_entry.size_bytes > cache.max_size_bytes

    def evict(self, cache: 'LayerCache') -> Optional[str]:
        """Evict the least frequently used entry"""
        if not cache.cache:
            return None

        # Find the LFU entry (lowest access_count, tie-breaker: LRU)
        min_key = min(
            cache.cache.keys(),
            key=lambda k: (cache.cache[k].access_count, cache.cache[k].last_access)
        )

        evicted = cache.cache.pop(min_key)
        cache.current_size -= evicted.size_bytes
        return min_key


class AdaptivePolicy(CachePolicy):
    """
    Adaptive policy that combines LRU and frequency-based eviction.
    Switches between policies based on access pattern detected.
    """

    def __init__(self):
        self.use_lru = True
        self.lru_hits = 0
        self.freq_hits = 0
        self.switch_threshold = 10

    def should_evict(self, cache: 'LayerCache', new_entry: CacheEntry) -> bool:
        """Check if adding new entry would exceed capacity"""
        return cache.current_size + new_entry.size_bytes > cache.max_size_bytes

    def evict(self, cache: 'LayerCache') -> Optional[str]:
        """Evict based on current policy"""
        if self.use_lru:
            return self._evict_lru(cache)
        else:
            return self._evict_lfu(cache)

    def _evict_lru(self, cache: 'LayerCache') -> Optional[str]:
        """Evict LRU and track hits"""
        if not cache.cache:
            return None

        oldest_key = min(
            cache.cache.keys(),
            key=lambda k: cache.cache[k].last_access
        )

        evicted = cache.cache.pop(oldest_key)
        cache.current_size -= evicted.size_bytes

        # Track if high-access items are being evicted
        if evicted.access_count > 5:
            self.freq_hits += 1
            if self.freq_hits > self.switch_threshold:
                self.use_lru = False
                print(f"[EvoLLM] Switching to LFU policy (freq hits: {self.freq_hits})")

        return oldest_key

    def _evict_lfu(self, cache: 'LayerCache') -> Optional[str]:
        """Evict LFU and track hits"""
        if not cache.cache:
            return None

        min_key = min(
            cache.cache.keys(),
            key=lambda k: (cache.cache[k].access_count, cache.cache[k].last_access)
        )

        evicted = cache.cache.pop(min_key)
        cache.current_size -= evicted.size_bytes

        # Track if recently-used items are being evicted
        age = time.time() - cache.cache.get(min_key, CacheEntry({}, 0)).last_access if min_key in cache.cache else 0
        if evicted.access_count == 1 and time.time() - evicted.last_access < 1.0:
            self.lru_hits += 1
            if self.lru_hits > self.switch_threshold:
                self.use_lru = True
                print(f"[EvoLLM] Switching to LRU policy (lru hits: {self.lru_hits})")

        return min_key


class LayerCache:
    """
    Bounded LRU cache for layer state dictionaries.

    Features
    --------
    - Memory-bounded: never exceeds max_size_bytes
    - Thread-safe for concurrent access (with external lock)
    - Tracks hits/misses for profiling
    - Supports multiple eviction policies

    Parameters
    ----------
    max_size_bytes : int
        Maximum cache size in bytes
    policy : CachePolicy, optional
        Eviction policy (default: LRUPolicy)

    Usage
    -----
    cache = LayerCache(max_size_bytes=32 * 1e9)
    state = cache.get('model.layers.0')
    if state is None:
        state = load_layer_from_disk(...)
        cache.put('model.layers.0', state)
    """

    def __init__(self, max_size_bytes: int, policy: Optional[CachePolicy] = None):
        self.max_size_bytes = max_size_bytes
        self.current_size = 0
        self.cache: Dict[str, CacheEntry] = OrderedDict()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

        # Policy
        self.policy = policy or LRUPolicy()

    def get(self, key: str) -> Optional[Dict]:
        """
        Retrieve a layer from cache.

        Returns
        -------
        state_dict or None
            Cached state dictionary or None if not found
        """
        if key in self.cache:
            entry = self.cache[key]
            entry.touch()

            # Move to end (for LRU ordering)
            self.cache.move_to_end(key)

            self.hits += 1
            return entry.state_dict

        self.misses += 1
        return None

    def put(self, key: str, state_dict: Dict, size_bytes: Optional[int] = None):
        """
        Add a layer to cache.

        Parameters
        ----------
        key : str
            Layer name (e.g., 'model.layers.0')
        state_dict : dict
            Layer state dictionary
        size_bytes : int, optional
            Size in bytes. If None, estimated from tensors.
        """
        if size_bytes is None:
            size_bytes = self._estimate_size(state_dict)

        # Check if eviction needed
        while self.policy.should_evict(self, CacheEntry(state_dict, size_bytes)) and self.cache:
            evicted_key = self.policy.evict(self)
            if evicted_key:
                self.evictions += 1

        # Add to cache
        entry = CacheEntry(state_dict, size_bytes)
        self.cache[key] = entry
        self.current_size += size_bytes

    def _estimate_size(self, state_dict: Dict) -> int:
        """Estimate memory size of state_dict in bytes"""
        total = 0
        for tensor in state_dict.values():
            if hasattr(tensor, 'numel') and hasattr(tensor, 'element_size'):
                total += tensor.numel() * tensor.element_size()
        return total

    def remove(self, key: str):
        """Manually remove an entry from cache"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size -= entry.size_bytes

    def clear(self):
        """Clear all entries"""
        self.cache.clear()
        self.current_size = 0

    def hit_rate(self) -> float:
        """Compute cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate(),
            'current_size_gb': self.current_size / 1e9,
            'max_size_gb': self.max_size_bytes / 1e9,
            'num_entries': len(self.cache),
            'evictions': self.evictions,
        }

    def __contains__(self, key: str) -> bool:
        return key in self.cache

    def __len__(self) -> int:
        return len(self.cache)

    def __repr__(self) -> str:
        return (f"LayerCache(entries={len(self)}, size={self.current_size/1e9:.2f}GB, "
                f"hit_rate={self.hit_rate():.2%})")


class GPUCache:
    """
    Simple cache for keeping layers in GPU VRAM.

    Unlike LayerCache, this is not memory-bounded by default.
    Used for keeping a few hot layers in VRAM for fast access.

    Parameters
    ----------
    max_layers : int
        Maximum number of layers to keep in GPU
    """

    def __init__(self, max_layers: int = 3):
        self.max_layers = max_layers
        self.cache: Dict[str, bool] = {}  # Just track which layers are in GPU
        self.order: list = []  # LRU order

    def should_keep(self, layer_idx: int, total_layers: int) -> bool:
        """
        Determine if a layer should stay in GPU based on position.

        Strategy: Keep early layers (0-2) as they're frequently reused.
        Or keep most recent N layers if gpu_layers is small.
        """
        if self.max_layers <= 0:
            return False

        # Simple strategy: keep first N layers (embed + early layers)
        # These are used in every generation
        return layer_idx < self.max_layers


class TensorCacheManager:
    """
    Manages hierarchical caching: GPU -> CPU RAM -> Disk

    Coordinates between different cache levels and provides
    unified interface for layer loading.
    """

    def __init__(self,
                 cpu_cache: Optional[LayerCache] = None,
                 gpu_cache: Optional[GPUCache] = None,
                 prefetch_depth: int = 1):
        self.cpu_cache = cpu_cache
        self.gpu_cache = gpu_cache
        self.prefetch_depth = prefetch_depth

        # Statistics
        self.gpu_hits = 0
        self.cpu_hits = 0
        self.disk_loads = 0

    def get_layer(self,
                  layer_name: str,
                  layer_idx: int,
                  load_from_disk_fn,
                  move_to_gpu_fn) -> Tuple[Dict, str]:
        """
        Get a layer, trying cache hierarchy.

        Parameters
        ----------
        layer_name : str
            Name of layer to load
        layer_idx : int
            Index of layer in model (for GPU caching decisions)
        load_from_disk_fn : callable
            Function to load layer from disk
        move_to_gpu_fn : callable
            Function to move layer tensors to GPU

        Returns
        -------
        state_dict : dict
            Layer state dictionary
        source : str
            'gpu', 'cpu', or 'disk' indicating where layer came from
        """
        # Check if should be kept in GPU (policy decision, happens after load)
        should_keep_in_gpu = (self.gpu_cache and
                              self.gpu_cache.should_keep(layer_idx, 999))

        # If already in CPU cache, use it
        if self.cpu_cache and layer_name in self.cpu_cache:
            state_dict = self.cpu_cache.get(layer_name)
            self.cpu_hits += 1
            source = 'cpu'
        else:
            # Load from disk
            state_dict = load_from_disk_fn(layer_name)
            self.disk_loads += 1
            source = 'disk'

            # Add to CPU cache if available
            if self.cpu_cache:
                self.cpu_cache.put(layer_name, state_dict)

        # Note: Actual GPU movement handled by caller after this returns
        # We just inform whether it should stay in GPU
        return state_dict, source

    def get_stats(self) -> Dict:
        """Get hierarchical cache statistics"""
        total_accesses = self.gpu_hits + self.cpu_hits + self.disk_loads

        stats = {
            'total_accesses': total_accesses,
            'gpu_hits': self.gpu_hits,
            'cpu_hits': self.cpu_hits,
            'disk_loads': self.disk_loads,
            'overall_hit_rate': (self.gpu_hits + self.cpu_hits) / total_accesses if total_accesses > 0 else 0.0,
        }

        if self.cpu_cache:
            stats['cpu_cache'] = self.cpu_cache.get_stats()

        return stats


def create_cache(config, estimated_layer_size_gb: float = 2.0) -> Optional[TensorCacheManager]:
    """
    Factory function to create appropriate cache manager from config.

    Parameters
    ----------
    config : EvoLLMConfig
        Configuration
    estimated_layer_size_gb : float
        Estimated size per layer in GB (for capacity planning)

    Returns
    -------
    TensorCacheManager or None
        Configured cache manager, or None if caching disabled
    """
    # Check if any caching is enabled
    if config.cpu_cache_gb <= 0 and config.gpu_layers <= 0:
        return None

    # Create CPU cache if requested
    cpu_cache = None
    if config.cpu_cache_gb > 0:
        max_bytes = int(config.cpu_cache_gb * 1e9)

        # Select policy
        if config.cache_policy == 'lru':
            policy = LRUPolicy()
        elif config.cache_policy == 'freq':
            policy = FREQPolicy()
        elif config.cache_policy == 'adaptive':
            policy = AdaptivePolicy()
        else:
            raise ValueError(f"Unknown cache_policy: {config.cache_policy}")

        cpu_cache = LayerCache(max_size_bytes=max_bytes, policy=policy)
        print(f"[EvoLLM] CPU cache initialized: {config.cpu_cache_gb:.1f}GB, policy={config.cache_policy}")

    # Create GPU cache if requested
    gpu_cache = None
    if config.gpu_layers > 0:
        gpu_cache = GPUCache(max_layers=config.gpu_layers)
        print(f"[EvoLLM] GPU cache initialized: {config.gpu_layers} layers")

    return TensorCacheManager(
        cpu_cache=cpu_cache,
        gpu_cache=gpu_cache,
        prefetch_depth=config.prefetch_depth
    )
