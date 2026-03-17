"""
Hierarchical Tensor Loader for EvoLLM.

Manages loading of model layers across three levels:
  Level 0: GPU VRAM (fastest)
  Level 1: CPU RAM cache (medium)
  Level 2: SSD/NVMe (slowest)

Provides unified interface for layer loading with caching.
"""

from typing import Dict, Optional, Tuple, Callable, List
import time
from concurrent.futures import ThreadPoolExecutor, Future
import torch

from .cache_policy import TensorCacheManager, CacheEntry


class HierarchicalTensorLoader:
    """
    Manages hierarchical layer loading with configurable caching.

    Parameters
    ----------
    checkpoint_path : str
        Path to split model directory
    cache_manager : TensorCacheManager, optional
        Cache manager (None = no caching)
    device : str
        Device for computation (e.g., 'cuda:0')
    prefetch_depth : int
        Number of layers to prefetch ahead
    prefetch_async : bool
        Use async loading with thread pool
    """

    def __init__(self,
                 checkpoint_path: str,
                 cache_manager: Optional[TensorCacheManager] = None,
                 device: str = 'cuda:0',
                 prefetch_depth: int = 1,
                 prefetch_async: bool = True,
                 prefetch_batches: int = 1):
        self.checkpoint_path = checkpoint_path
        self.cache_manager = cache_manager
        self.device = device
        self.prefetch_depth = prefetch_depth
        self.prefetch_async = prefetch_async
        self.prefetch_batches = prefetch_batches

        # GPU cache tracking (which layers are currently in GPU)
        self.gpu_resident_layers: List[str] = []
        self.max_gpu_layers = 3  # Default, can be overridden

        # Prefetch thread pool
        self.executor = ThreadPoolExecutor(max_workers=prefetch_batches) if prefetch_async else None
        self.prefetch_futures: List[Future] = []

        # Profiling
        self.load_times = []
        self.cache_hits = 0

    def set_gpu_cache_capacity(self, max_layers: int):
        """Set maximum layers to keep in GPU"""
        self.max_gpu_layers = max_layers

    def load_layer(self,
                   layer_name: str,
                   layer_idx: int,
                   load_fn: Callable[[str], Dict],
                   move_fn: Callable[[Dict], List[str]]) -> Tuple[Dict, str]:
        """
        Load a layer using cache hierarchy.

        Parameters
        ----------
        layer_name : str
            Name of layer to load
        layer_idx : int
            Index in layer sequence (for caching decisions)
        load_fn : callable
            Function to load from disk: load_fn(layer_name) -> state_dict
        move_fn : callable
            Function to move to device: move_fn(state_dict) -> list of param names

        Returns
        -------
        state_dict : dict
            Loaded layer state
        source : str
            'gpu_cache', 'cpu_cache', or 'disk'
        """
        start_time = time.time()

        source = 'disk'

        if self.cache_manager:
            # Use cache manager's logic
            state_dict, source = self.cache_manager.get_layer(
                layer_name=layer_name,
                layer_idx=layer_idx,
                load_from_disk_fn=load_fn,
                move_to_gpu_fn=move_fn
            )
        else:
            # No caching: load directly from disk
            state_dict = load_fn(layer_name)
            source = 'disk'

        load_time = time.time() - start_time
        self.load_times.append(load_time)

        return state_dict, source

    def should_evict_from_gpu(self, layer_name: str) -> bool:
        """
        Decide if a layer should be evicted from GPU after use.
        Based on GPU cache capacity and layer position.
        """
        # Keep only first N layers in GPU
        if self.cache_manager and self.cache_manager.gpu_cache:
            layer_idx = self._get_layer_index(layer_name)
            return not self.cache_manager.gpu_cache.should_keep(layer_idx, 999)
        else:
            # Default: evict if not in first few layers
            layer_idx = self._get_layer_index(layer_name)
            return layer_idx >= self.max_gpu_layers

    def _get_layer_index(self, layer_name: str) -> int:
        """Extract layer index from name like 'model.layers.23'"""
        try:
            if 'layers' in layer_name:
                parts = layer_name.split('.')
                idx = parts.index('layers') + 1
                if idx < len(parts):
                    return int(parts[idx])
        except (ValueError, IndexError):
            pass
        return 999  # Special layers (embedding, norm, head) get high index

    def prefetch_layers(self,
                        upcoming_layer_names: List[str],
                        layer_indices: List[int],
                        load_fn: Callable[[str], Dict]):
        """
        Prefetch upcoming layers asynchronously.

        Parameters
        ----------
        upcoming_layer_names : list of str
            Names of layers to prefetch
        layer_indices : list of int
            Indices of these layers
        load_fn : callable
            Function to load layer from disk
        """
        if not self.prefetch_async or not upcoming_layer_names:
            return

        # Cancel any pending prefetches that are no longer needed
        # (This is simplified; in production we'd track and cancel if needed)

        # Start new prefetches
        for layer_name, layer_idx in zip(upcoming_layer_names[:self.prefetch_depth],
                                         layer_indices[:self.prefetch_depth]):

            # Skip if already in CPU cache
            if self.cache_manager and self.cache_manager.cpu_cache:
                if layer_name in self.cache_manager.cpu_cache:
                    continue

            # Submit async load
            future = self.executor.submit(load_fn, layer_name)

            # We don't store the future because we're not waiting for it
            # The result will be added to cache when complete in the async flow
            # But for simplicity, we just fire-and-forget; cache population happens
            # when the main thread requests the layer and it's already loaded elsewhere
            # This is a limitation - a more sophisticated approach would add to cache
            # in the background thread. For now, rely on main thread to benefit.
            # TODO: Proper async cache population

    def get_stats(self) -> Dict:
        """Get loader statistics"""
        avg_load_time = sum(self.load_times) / len(self.load_times) if self.load_times else 0

        stats = {
            'avg_load_time_s': avg_load_time,
            'total_loads': len(self.load_times),
            'gpu_resident_layers': len(self.gpu_resident_layers),
        }

        if self.cache_manager:
            stats['cache'] = self.cache_manager.get_stats()

        return stats

    def shutdown(self):
        """Cleanup resources (call on shutdown)"""
        if self.executor:
            self.executor.shutdown(wait=True)


class LayerLoadTracker:
    """
    Tracks layer loading patterns for adaptive prefetching.

    Can be used to build a profile of which layers are accessed
    most frequently and adjust prefetch strategy accordingly.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.access_counts = [0] * num_layers
        self.last_access_time = [0.0] * num_layers
        self.inter_access_intervals = [[] for _ in range(num_layers)]

    def record_access(self, layer_idx: int):
        """Record that layer was accessed"""
        now = time.time()
        last = self.last_access_time[layer_idx]
        if last > 0:
            interval = now - last
            self.inter_access_intervals[layer_idx].append(interval)

        self.access_counts[layer_idx] += 1
        self.last_access_time[layer_idx] = now

    def get_hot_layers(self, top_k: int = 10) -> List[int]:
        """Get indices of most frequently accessed layers"""
        indexed = list(enumerate(self.access_counts))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in indexed[:top_k]]

    def get_avg_interval(self, layer_idx: int) -> float:
        """Get average inter-access interval for layer"""
        intervals = self.inter_access_intervals[layer_idx]
        return sum(intervals) / len(intervals) if intervals else 0.0

    def suggest_prefetch_pattern(self) -> List[int]:
        """
        Suggest optimal prefetch pattern based on access history.

        Returns list of layer indices to prefetch.
        """
        # Simple heuristic: prefetch top accessed layers
        # For sequential generation, this would be next few layers
        # For batching, could be hottest layers regardless of position
        hot = self.get_hot_layers(top_k=self.num_layers // 4)
        return sorted(hot)[:self.prefetch_depth if hasattr(self, 'prefetch_depth') else 3]
