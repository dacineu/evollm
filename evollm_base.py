"""
EvoLLM Model: Extends AirLLMBaseModel with hierarchical caching.

Key additions:
  - CPU RAM cache for layers (LayerCache)
  - GPU multi-layer caching (keep N layers in VRAM)
  - Prefetch multiple layers ahead
  - Hardware auto-detection and config
"""

from typing import List, Optional, Tuple, Union, Dict
import time
from dataclasses import dataclass
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from transformers.modeling_outputs import CausalLMOutputWithPast

# Import from AirLLM with fallback for different package structures
try:
    from air_llm.airllm.airllm_base import AirLLMBaseModel
    from air_llm.airllm.utils import load_layer as airllm_load_layer, clean_memory
    from air_llm.airllm.profiler import LayeredProfiler
except ImportError:
    try:
        from airllm.airllm_base import AirLLMBaseModel
        from airllm.utils import load_layer as airllm_load_layer, clean_memory
        from airllm.profiler import LayeredProfiler
    except ImportError:
        raise ImportError("Could not import AirLLM modules. Ensure EvoLLM is run within AirLLM repository.")

from .config import EvoLLMConfig
from .cache_policy import create_cache, TensorCacheManager, LayerCache
from .hardware_profiler import HardwareProfiler, profile_and_recommend
from .tensor_loader import HierarchicalTensorLoader


class EvoLLMModel(AirLLMBaseModel):
    """
    EvoLLM: Hybrid layer-cached LLM inference model.

    Extends AirLLMBaseModel with:
      - CPU RAM layer cache (configurable size)
      - GPU multi-layer caching (keep N layers in VRAM)
      - Async prefetching with configurable depth
      - Hardware auto-detection

    Backward compatible: Set cpu_cache_gb=0 and gpu_layers=0 for AirLLM behavior.

    Parameters
    ----------
    Same as AirLLMBaseModel plus:
    evolllm_config : EvoLLMConfig, optional
        EvoLLM-specific configuration
    auto_config : bool
        If True, auto-detect hardware and configure
    """

    def __init__(self,
                 model_local_path_or_repo_id,
                 device="cuda:0",
                 dtype=torch.float16,
                 max_seq_len=512,
                 layer_shards_saving_path=None,
                 profiling_mode=False,
                 compression=None,
                 hf_token=None,
                 prefetching=True,
                 delete_original=False,
                 evolllm_config: Optional[EvoLLMConfig] = None,
                 auto_config: bool = False):
        """
        Initialize EvoLLM model.

        Notes
        -----
        Pass evolllm_config to customize caching behavior.
        Use auto_config=True to auto-detect optimal settings.
        """
        # Store EvoLLM config before super().__init__
        self.evolllm_config = evolllm_config
        self.auto_config = auto_config
        self.layer_cache: Optional[LayerCache] = None
        self.cache_manager: Optional[TensorCacheManager] = None
        self.tensor_loader: Optional[HierarchicalTensorLoader] = None

        # Profiling
        self.evolllm_profiling = profiling_mode
        self.layer_access_tracker = None

        # Call parent init (sets up model, layers, etc.)
        super().__init__(
            model_local_path_or_repo_id=model_local_path_or_repo_id,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
            layer_shards_saving_path=layer_shards_saving_path,
            profiling_mode=profiling_mode,
            compression=compression,
            hf_token=hf_token,
            prefetching=prefetching,
            delete_original=delete_original
        )

        # After parent init, we have:
        # - self.checkpoint_path: path to split model
        # - self.layer_names: list of layer names
        # - self.layers: list of layer modules (initially on 'meta')

        # Initialize EvoLLM components
        self._init_evolllm()

    def _init_evolllm(self):
        """Initialize EvoLLM-specific components after AirLLM init"""
        from .config import auto_config as auto_config_fn

        # Determine configuration
        if self.auto_config and not self.evolllm_config:
            # Auto-detect hardware
            print("[EvoLLM] Auto-detecting hardware configuration...")
            model_size_estimate = None
            try:
                # Try to get actual model size
                from .config import estimate_model_size
                model_size_estimate = estimate_model_size(str(self.model_local_path))
            except Exception:
                pass

            profile, self.evolllm_config = profile_and_recommend(
                model_size_b=model_size_estimate,
                quick=False
            )
            print(f"[EvoLLM] Hardware profile:\n{profile}")
            print(f"[EvoLLM] Auto-config: {self.evolllm_config}")

        elif not self.evolllm_config:
            # Default: AirLLM compatibility mode (no caching)
            self.evolllm_config = EvoLLMConfig(
                gpu_layers=0,
                cpu_cache_gb=0.0,
                prefetch_depth=1 if self.prefetching else 0
            )

        # Override compression setting from config
        if self.evolllm_config.compression and self.compression is None:
            print(f"[EvoLLM] Warning: compression in config but not in AirLLM init")
        elif self.compression and not self.evolllm_config.compression:
            self.evolllm_config.compression = self.compression

        # Initialize cache manager
        if self.evolllm_config.cpu_cache_gb > 0 or self.evolllm_config.gpu_layers > 0:
            # Estimate layer size from model
            est_layer_size_gb = self._estimate_layer_size_gb()

            self.cache_manager = create_cache(
                self.evolllm_config,
                estimated_layer_size_gb=est_layer_size_gb
            )

            if self.cache_manager and self.cache_manager.cpu_cache:
                self.layer_cache = self.cache_manager.cpu_cache
        else:
            print("[EvoLLM] Caching disabled (cpu_cache_gb=0, gpu_layers=0)")
            self.cache_manager = None

        # Initialize hierarchical tensor loader
        if self.cache_manager:
            self.tensor_loader = HierarchicalTensorLoader(
                checkpoint_path=str(self.checkpoint_path),
                cache_manager=self.cache_manager,
                device=self.device,
                prefetch_depth=self.evolllm_config.prefetch_depth,
                prefetch_async=self.evolllm_config.prefetch_async,
                prefetch_batches=self.evolllm_config.prefetch_batches
            )
            self.tensor_loader.set_gpu_cache_capacity(self.evolllm_config.gpu_layers)
        else:
            self.tensor_loader = None

        print(f"[EvoLLM] Initialized with config: {self.evolllm_config}")

    def _estimate_layer_size_gb(self) -> float:
        """Estimate average layer size in GB from model checkpoint"""
        try:
            import os
            from pathlib import Path

            checkpoint_path = Path(self.checkpoint_path)

            # Get total size of split layers
            total_size = 0
            layer_count = 0

            for layer_file in checkpoint_path.glob("*"):
                if layer_file.is_file() and not layer_file.name.endswith('.json'):
                    total_size += layer_file.stat().st_size
                    layer_count += 1

            if layer_count > 0:
                # Exclude embedding, norm, lm_head from count
                transformer_layers = max(0, layer_count - 3)
                if transformer_layers > 0:
                    return (total_size / transformer_layers) / 1e9

            # Fallback: estimate from model config
            if hasattr(self.config, 'hidden_size'):
                # Rough: param_count * 2 bytes (fp16)
                param_count = sum(p.numel() for p in self.model.parameters() if p.is_meta)
                return (param_count * 2) / 1e9

        except Exception as e:
            print(f"[EvoLLM] Warning: Could not estimate layer size: {e}")

        # Default for 70B models
        return 2.0  # ~2GB per layer

    def _should_use_cached_load_layer(self) -> bool:
        """Check if we should use the cache-enabled layer loader"""
        return self.cache_manager is not None

    def _load_layer_with_cache(self, layer_name: str) -> Dict:
        """
        Load layer with hierarchical caching.

        Wraps airllm's load_layer_to_cpu with cache logic.
        """
        # Get layer index for GPU caching decision
        layer_idx = self._get_layer_index(layer_name)

        # Use tensor loader if available
        if self.tensor_loader:
            state_dict, source = self.tensor_loader.load_layer(
                layer_name=layer_name,
                layer_idx=layer_idx,
                load_fn=lambda name: airllm_load_layer(str(self.checkpoint_path), name, self.profiling_mode),
                move_fn=lambda state: self.move_layer_to_device(state)  # We'll move after
            )

            if self.evolllm_config.enable_profiling:
                print(f"[EvoLLM] Layer {layer_name} loaded from {source}")

            return state_dict
        else:
            # Fallback to direct load (AirLLM style)
            return airllm_load_layer(str(self.checkpoint_path), layer_name, self.profiling_mode)

    def _get_layer_index(self, layer_name: str) -> int:
        """Extract numeric layer index from layer name"""
        try:
            if '.layers.' in layer_name:
                parts = layer_name.split('.')
                idx = parts.index('layers') + 1
                return int(parts[idx])
        except (ValueError, IndexError):
            pass
        return 999  # Special layers

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass with hierarchical caching.

        Overrides AirLLMBaseModel.forward to add cache-friendly layer loading.
        """
        if self.profiling_mode or self.evolllm_config.enable_profiling:
            forward_start_wall = time.time()
            total_layer_load_time = 0.0
            total_move_time = 0.0

        # Reboot model (AirLLM pattern)
        del self.model
        clean_memory()
        self.init_model()

        # Prepare inputs (AirLLM pattern)
        batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]

        # Create attention mask and position ids
        attention_mask = torch.ones(self.max_seq_len, self.max_seq_len)
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...] == 0
        attention_mask = attention_mask.to(self.running_device)
        position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=self.running_device)[None, :]

        kv_cache_list = [] if use_cache else None
        if use_cache:
            for x in self.layers:
                kv_cache_list.append(([], []))

        all_hidden_states = [] * len(self.layers) if output_hidden_states else None
        all_self_attns = [] * len(self.layers) if output_attentions else None

        # Main layer loop with cache-enabled loading
        with torch.inference_mode():
            if self.evolllm_config.prefetch_async and self.tensor_loader and self.tensor_loader.executor:
                executor = self.tensor_loader.executor
            else:
                executor = ThreadPoolExecutor(max_workers=1)

            # Prefetch first layer if async enabled
            if self.evolllm_config.prefetch_async and self.tensor_loader and self.evolllm_config.prefetch_depth > 1:
                self._prefetch_layers_ahead(executor, 0)

            for i, (layer_name, layer) in tqdm(
                enumerate(zip(self.layer_names, self.layers)),
                desc=f'EvoLLM layers ({self.running_device})',
                total=len(self.layers)
            ):
                # Load layer (with cache if enabled)
                if self.profiling_mode or self.evolllm_config.enable_profiling:
                    load_start = time.time()

                state_dict = self._load_layer_with_cache(layer_name)

                if self.profiling_mode or self.evolllm_config.enable_profiling:
                    total_layer_load_time += time.time() - load_start

                # Move layer to GPU
                if self.profiling_mode or self.evolllm_config.enable_profiling:
                    move_start = time.time()

                moved_layers = self.move_layer_to_device(state_dict)

                if self.profiling_mode or self.evolllm_config.enable_profiling:
                    total_move_time += time.time() - move_start

                # Run computation on this layer
                for j, seq in enumerate(batch):
                    if layer_name == self.layer_names_dict['embed']:
                        batch[j] = layer(seq)
                    elif layer_name == self.layer_names_dict['norm']:
                        batch[j] = self.run_norm(layer, seq)
                    elif layer_name == self.layer_names_dict['lm_head']:
                        batch[j] = self.run_lm_head(layer, seq)
                    else:
                        # Standard transformer layer with KV cache support
                        batch[j] = self._run_transformer_layer(
                            layer, seq, layer_name, i,
                            past_key_values, use_cache,
                            kv_cache_list, attention_mask,
                            position_ids, output_attentions, all_self_attns, j
                        )

                if output_hidden_states and i < len(all_hidden_states):
                    all_hidden_states[i] = batch[j]

                # Eviction logic: Remove from GPU if not in multi-layer cache
                if not self.evolllm_config.gpu_multi_layer_caching or self.evolllm_config.evict_after_use:
                    # Don't keep this layer in GPU unless it's in the first gpu_layers
                    if not self._should_keep_in_gpu(layer_name, i):
                        if self.hf_quantizer is not None:
                            for param_name in moved_layers:
                                set_module_tensor_to_device(self.model, param_name, 'meta')
                        else:
                            layer.to("meta")
                        clean_memory()

                # Prefetch next layers (lookahead)
                if (self.evolllm_config.prefetch_async and self.tensor_loader and
                    self.evolllm_config.prefetch_depth > 1):
                    self._prefetch_layers_ahead(executor, i)

            if executor != self.tensor_loader.executor if self.tensor_loader else None:
                executor.shutdown(wait=False)

        # Return output (similar to AirLLM)
        logits = torch.cat(batch, 0)

        # KV cache post-processing
        if use_cache:
            kv_cache_list = kv_cache_list[1:-2]
            for i in range(len(kv_cache_list)):
                kv_cache_list[i] = (torch.cat(kv_cache_list[i][0], 0), torch.cat(kv_cache_list[i][1], 0))

        if output_attentions:
            all_self_attns = all_self_attns[0:-2]
            for i in range(len(all_self_attns)):
                all_self_attns[i] = torch.cat(all_self_attns[i], 0)

        if output_hidden_states:
            all_hidden_states = all_hidden_states[0:-2]
            for i in range(len(all_hidden_states)):
                all_hidden_states[i] = torch.cat(all_hidden_states[i], 0)

        # Print profiling summary
        if self.profiling_mode or self.evolllm_config.enable_profiling:
            forward_elapsed_wall = time.time() - forward_start_wall

            print("\n=== EvoLLM Performance Summary ===")
            print(f"Total wall time: {forward_elapsed_wall:.3f}s")
            if total_layer_load_time > 0:
                print(f"Layer load time: {total_layer_load_time:.3f}s")
            if total_move_time > 0:
                print(f"Move to GPU time: {total_move_time:.3f}s")

            if self.cache_manager:
                stats = self.cache_manager.get_stats()
                print("\nCache Statistics:")
                print(f"  Overall hit rate: {stats.get('overall_hit_rate', 0):.1%}")
                if 'cpu_cache' in stats:
                    cpu_stats = stats['cpu_cache']
                    print(f"  CPU cache hit rate: {cpu_stats.get('hit_rate', 0):.1%}")
                    print(f"  CPU cache size: {cpu_stats.get('current_size_gb', 0):.1f}/{cpu_stats.get('max_size_gb', 0):.1f} GB")
                    print(f"  CPU cache entries: {cpu_stats.get('num_entries', 0)}")
                    print(f"  CPU cache evictions: {cpu_stats.get('evictions', 0)}")
            print("==================================\n")

        if not return_dict:
            return tuple(v for v in [
                logits,
                tuple(kv_cache_list) if kv_cache_list else None,
                tuple(all_hidden_states) if all_hidden_states else None,
                tuple(all_self_attns) if all_self_attns else None
            ] if v is not None)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=tuple(kv_cache_list) if kv_cache_list else None,
            hidden_states=tuple(all_hidden_states) if all_hidden_states else None,
            attentions=tuple(all_self_attns) if all_self_attns else None,
        )

    def _should_keep_in_gpu(self, layer_name: str, layer_idx: int) -> bool:
        """Determine if layer should remain in GPU after use"""
        if not self.evolllm_config.gpu_multi_layer_caching:
            return False

        if self.cache_manager and self.cache_manager.gpu_cache:
            return self.cache_manager.gpu_cache.should_keep(layer_idx, 999)

        # Fallback: keep first N layers
        return layer_idx < self.evolllm_config.gpu_layers

    def _prefetch_layers_ahead(self, executor: ThreadPoolExecutor, current_idx: int):
        """Trigger async prefetch for upcoming layers"""
        if not self.tensor_loader:
            return

        # Determine which layers to prefetch
        start_idx = current_idx + 1
        end_idx = min(current_idx + 1 + self.evolllm_config.prefetch_depth, len(self.layer_names))

        upcoming_names = self.layer_names[start_idx:end_idx]
        upcoming_indices = list(range(start_idx, end_idx))

        self.tensor_loader.prefetch_layers(
            upcoming_layer_names=upcoming_names,
            layer_indices=upcoming_indices,
            load_fn=lambda name: airllm_load_layer(str(self.checkpoint_path), name, self.profiling_mode)
        )

    def _run_transformer_layer(self,
                               layer,
                               seq,
                               layer_name: str,
                               layer_idx: int,
                               past_key_values,
                               use_cache,
                               kv_cache_list,
                               attention_mask,
                               position_ids,
                               output_attentions,
                               all_self_attns,
                               batch_idx: int):
        """Run a single transformer layer (extracted for clarity)"""
        # This is essentially the same logic from AirLLMBaseModel.forward
        if past_key_values is not None:
            k_cache, v_cache = past_key_values[layer_idx - 1]
            len_p = self.get_past_key_values_cache_seq_len(past_key_values)
            len_s = self.get_sequence_len(seq)

            position_ids_args = self.get_position_ids_args(position_ids, len_p, len_s)
            attention_mask_args = self.get_attention_mask_args(attention_mask, len_p, len_s)
            past_key_value_args = self.get_past_key_value_args(k_cache, v_cache)

            kwargs = {'use_cache': True}
            pos_embed_args = self.get_pos_emb_args(len_p, len_s)
            kwargs = {**kwargs, **past_key_value_args, **pos_embed_args,
                     **attention_mask_args, **position_ids_args}

            layer_outputs = layer(seq, **kwargs)
            new_seq = layer_outputs[0]

            if output_attentions:
                all_self_attns[layer_idx].append(layer_outputs[1])

            if use_cache:
                (k_cache, v_cache) = layer_outputs[2 if output_attentions else 1]
                kv_cache_list[layer_idx][0].append(k_cache)
                kv_cache_list[layer_idx][1].append(v_cache)

        else:
            len_seq = self.get_sequence_len(seq)
            pos_embed_args = self.get_pos_emb_args(0, len_seq)
            attention_mask_args = self.get_attention_mask_args(attention_mask, 0, len_seq)
            position_ids_args = self.get_position_ids_args(position_ids, 0, len_seq)

            if not use_cache:
                kwargs = {
                    'use_cache': False,
                    'attention_mask': attention_mask[:, :, -len_seq:, -len_seq:],
                }
                kwargs = {**kwargs, **pos_embed_args, **attention_mask_args, **position_ids_args}
                new_seq = layer(seq, **kwargs)[0]
            else:
                kwargs = {
                    'use_cache': True,
                    'attention_mask': attention_mask[:, :, -len_seq:, -len_seq:],
                }
                kwargs = {**kwargs, **pos_embed_args, **attention_mask_args, **position_ids_args}
                layer_out = layer(seq, **kwargs)
                new_seq, (k_cache, v_cache) = layer_out
                kv_cache_list[layer_idx][0].append(k_cache)
                kv_cache_list[layer_idx][1].append(v_cache)

        return new_seq

    def get_cache_stats(self) -> Dict:
        """Get EvoLLM-specific cache statistics"""
        stats = {}

        if self.cache_manager:
            stats.update(self.cache_manager.get_stats())

        if self.tensor_loader:
            stats['loader'] = self.tensor_loader.get_stats()

        return stats

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'tensor_loader') and self.tensor_loader:
            try:
                self.tensor_loader.shutdown()
            except:
                pass


class EvoLLMAutoModel:
    """
    AutoModel wrapper matching AirLLM's AutoModel interface.

    Usage
    -----
    from evollm import AutoModel

    model = AutoModel.from_pretrained(
        "meta-llama/Llama-2-70b-hf",
        auto_config=True
    )
    """

    @classmethod
    def from_pretrained(cls,
                        model_local_path_or_repo_id,
                        device="cuda:0",
                        dtype=torch.float16,
                        max_seq_len=512,
                        layer_shards_saving_path=None,
                        profiling_mode=False,
                        compression=None,
                        hf_token=None,
                        prefetching=True,
                        delete_original=False,
                        evolllm_config: Optional[EvoLLMConfig] = None,
                        auto_config: bool = False,
                        **kwargs):
        """
        Load EvoLLM model from pretrained checkpoint.

        Parameters
        ----------
        Same as AirLLMBaseModel plus EvoLLM-specific options.
        """
        # Determine base class based on model type
        # For now, use generic EvoLLMModel (works for Llama, Mistral, etc.)
        model = EvoLLMModel(
            model_local_path_or_repo_id=model_local_path_or_repo_id,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
            layer_shards_saving_path=layer_shards_saving_path,
            profiling_mode=profiling_mode,
            compression=compression,
            hf_token=hf_token,
            prefetching=prefetching,
            delete_original=delete_original,
            evolllm_config=evolllm_config,
            auto_config=auto_config,
            **kwargs
        )

        return model


# Alias for convenience
AutoModel = EvoLLMAutoModel
