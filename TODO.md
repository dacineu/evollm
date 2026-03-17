# EvoLLM Project TODO List

**Vision**: Evolving Large Language Model inference → EvoOS (distributed AI operating system)

**Current Phase**: Phase 1 (Single-node hierarchical caching) → Phase 2 (P2P coordination)

---

## 📋 Legend

- [ ] Not started
- [x] Completed
- [~] In progress / blocked
- 🔴 High priority
- 🟡 Medium priority
- 🟢 Low priority / polish

---

## Phase 1: Foundation & Single-Node Optimization (CURRENT)

### Core Modules

#### cache_policy.py - Hierarchical Caching
- [x] `CacheEntry` dataclass
- [x] `CachePolicy` abstract base class
- [x] `LRUPolicy` implementation
- [x] `FREQPolicy` (LFU) implementation
- [x] `AdaptivePolicy` (auto-switch LRU/LFU)
- [x] `LayerCache` (bounded CPU RAM cache with OrderedDict)
- [x] `GPUCache` (simple GPU layer cache)
- [x] `TensorCacheManager` (coordinates GPU→CPU→SSD hierarchy)
- [x] `create_cache()` factory function
- [~] **Add cache warming strategy** - Pre-populate cache based on layer access patterns from previous runs
- [ ] **Add cache compression** - Optional LZ4/Zstd compression for cached layers to increase effective capacity
- [ ] **Add cache persistence** - Save/load cache state to disk between runs (reuse warm cache)
- [ ] **Add cache partitioning** - Allow reserving cache segments for different models or priorities

#### config.py - Configuration System
- [x] `EvoLLMConfig` dataclass
- [x] `auto_config()` hardware detection
- [x] `estimate_model_size()` from checkpoint
- [x] `validate_config()` resource checks
- [ ] **Add config validation schemas** - Use Pydantic for stricter validation
- [ ] **Add config presets** - Pre-defined configs: `conservative`, `performance`, `balanced`, `memory_saver`
- [ ] **Add config migration** - Handle config version changes gracefully
- [ ] **Environment variable overrides** - Allow `EVOLLM_CPU_CACHE_GB=32` to override
- [ ] **Config file formats** - Support YAML/JSON config files in addition to dataclass

#### hardware_profiler.py - Hardware Detection
- [x] `HardwareProfile` dataclass
- [x] `HardwareProfiler` class
- [x] CPU/RAM detection (psutil)
- [x] GPU detection (torch.cuda + nvidia-smi)
- [x] Disk speed benchmark
- [x] PCIe bandwidth estimation
- [x] `recommend_config()` optimal settings
- [x] `profile_and_recommend()` convenience function
- [ ] **Add CPU affinity detection** - NUMA nodes, cache hierarchy
- [ ] **Add power/thermal profiling** - Detect thermal throttling, battery status (mobile)
- [ ] **Add network bandwidth detection** - For future P2P planning
- [ ] **Add GPU memory fragmentation detection** - Estimate usable VRAM vs total
- [ ] **Add multi-GPU detection** - Support for systems with multiple GPUs

#### tensor_loader.py - Async Prefetching
- [x] `HierarchicalTensorLoader` class
- [x] ThreadPoolExecutor for async loading
- [x] `load_layer()` delegates to cache_manager
- [x] `prefetch_layers()` fire-and-forget
- [x] GPU capacity tracking
- [ ] **Implement proper async cache population** - Background threads should add to cache, not just load
- [ ] **Add prefetch throttling** - Don't prefetch if cache pressure high
- [ ] **Add intelligent prefetch planning** - Use access history to predict which layers needed next
- [ ] **Add stream-based prefetch** - Use CUDA streams for async CPU→GPU transfers
- [ ] **Add prefetch metrics** - Track prefetch hit rate, waste (unused prefetches)

#### evollm_base.py - Main Model
- [x] `EvoLLMModel` extends AirLLMBaseModel
- [x] `_init_evolllm()` setup
- [x] Auto-config integration
- [x] Cache manager creation
- [x] Tensor loader initialization
- [x] `_load_layer_with_cache()` wrapper
- [x] `_estimate_layer_size_gb()`
- [x] `forward()` with cache-friendly loading
- [x] Async prefetch in forward loop
- [x] GPU eviction logic
- [x] Profiling output
- [x] `EvoLLMAutoModel` factory
- [x] `get_cache_stats()` method
- [ ] **Add checkpoint/resume** - Save state mid-generation for long-running inferences
- [ ] **Add batch inference optimization** - Better KV cache handling for multiple sequences
- [ ] **Add streaming output** - Yield tokens as generated (currently returns full sequence)
- [ ] **Add attention optimization** - FlashAttention, sliding window, etc.
- [ ] **Add quantization support** - 4bit/8bit quantization integration
- [ ] **Add gradient checkpointing** - For fine-tuning scenarios

#### utils.py - Utilities
- [x] `print_hardware_summary()`
- [x] `get_recommended_config_for_model()` known presets
- [x] `check_fitllm_readiness()` system checks
- [ ] **Add model download helper** - Automatically download from HuggingFace if not present
- [ ] **Add benchmark suite** - Standardized throughput/latency tests for different configs
- [ ] **Add log configurator** - Centralized logging with levels, file output
- [ ] **Add telemetry opt-in** - Anonymous usage stats to improve defaults

### Testing (Phase 1)

- [ ] **Unit tests for cache_policy.py**
  - [ ] LRU eviction order
  - [ ] LFU eviction with tie-breakers
  - [ ] Adaptive policy switching
  - [ ] LayerCache hit/miss tracking
  - [ ] GPUCache.should_keep() logic
  - [ ] TensorCacheManager.get_layer() flow

- [ ] **Unit tests for config.py**
  - [ ] auto_config RAM detection logic
  - [ ] auto_config GPU detection logic
  - [ ] auto_config prefetch depth logic
  - [ ] config validation edge cases

- [ ] **Unit tests for hardware_profiler.py**
  - [ ] Mock psutil for RAM detection
  - [ ] Mock torch.cuda for GPU detection
  - [ ] Disk speed test with temp files
  - [ ] recommend_config() outputs

- [ ] **Unit tests for tensor_loader.py**
  - [ ] Prefetch depth handling
  - [ ] ThreadPoolExecutor lifecycle
  - [ ] GPU capacity tracking

- [ ] **Integration tests**
  - [ ] End-to-end small model (7B) inference with caching
  - [ ] Cache hit rate verification
  - [ ] Prefetch effectiveness measurement
  - [ ] Memory usage stays within bounds

- [ ] **Performance benchmarks**
  - [ ] Benchmark suite: throughput vs cache sizes
  - [ ] Compare AirLLM baseline vs EvoLLM with cache
  - [ ] Measure prefetch impact
  - [ ] Profile memory bandwidth usage

---

## Phase 2: Peer-to-Peer Coordination (NEXT MAJOR PHASE)

### New Module: resource_provider.py (Abstraction Layer)
- [ ] **Define `ResourceProvider` ABC**
  - [x] Abstract methods: `get_layer()`, `cache_layer()`, `get_stats()`, `shutdown()`
  - [ ] Define return types and error handling contract
  - [ ] Document interface with examples

- [ ] **Implement `LocalResourceManager`**
  - [ ] Wrap TensorCacheManager
  - [ ] Adapt load_fn/move_fn parameters
  - [ ] Translate stats format
  - [ ] Unit tests with mock cache_manager

- [ ] **Implement `PeerClientResourceManager`**
  - [ ] Accept registry and fetcher dependencies
  - [ ] Implement fallback to disk
  - [ ] Stats aggregation
  - [ ] Unit tests

- [ ] **Implement `PeerProviderResourceManager`**
  - [ ] Expose PeerServer's cache as read-only provider
  - [ ] Handle concurrent access safely
  - [ ] Unit tests

- [ ] **Implement `HybridResourceManager`**
  - [ ] Support 'local_first' policy
  - [ ] Support 'parallel' policy (ThreadPoolExecutor)
  - [ ] Support 'cheapest' policy (credit-cost minimization)
  - [ ] Stats per-provider tracking
  - [ ] Graceful degradation if provider fails
  - [ ] Unit tests for each policy

- [ ] **Integration tests**
  - [ ] Test composition: Local + Peer client
  - [ ] Test fallback chains
  - [ ] Verify stats correctly aggregated

### New Module: peer/client.py (Peer Client)

- [ ] **Define gRPC protocol** (`protocol.proto`)
  - [ ] `PeerService` service definition
  - [ ] `FetchLayer` RPC (server-side streaming for large layers)
  - [ ] `HasLayer` RPC (quick existence check)
  - [ ] `AdvertiseLayers` RPC (periodic announcements)
  - [ ] `Heartbeat` RPC (health monitoring)
  - [ ] Generate Python stubs with `grpcio-tools`
  - [ ] Document message formats

- [ ] **Implement `PeerInfo` dataclass**
  - [ ] Fields: peer_id, address, layers, latency, reputation, price, load
  - [ ] Serialization/deserialization
  - [ ] Update methods for metrics

- [ ] **Implement `PeerRegistry`**
  - [ ] Mode: 'static' (bootstrap_peers list)
  - [ ] Mode: 'http' (central registry server) - future
  - [ ] Mode: 'dht' (distributed hash table) - future
  - [ ] `add_peer()`, `remove_peer()`
  - [ ] `find_peers_for_layer()` lookup
  - [ ] Background refresh task
  - [ ] Unit tests with in-memory registry

- [ ] **Implement `PeerLayerFetcher`**
  - [ ] Connection pooling (peer_id → gRPC stub)
  - [ ] `fetch(layer_name)` main method
  - [ ] `_rank_peers()` scoring algorithm
  - [ ] Ranking factors: latency, reputation, price, load
  - [ ] Retry logic with exponential backoff
  - [ ] Circuit breaker pattern (blacklist after N failures)
  - [ ] Timeout handling
  - [ ] Stats collection (success/failure counts, latencies)
  - [ ] `_fetch_from_peer()` gRPC call with streaming
  - [ ] Response validation (checksum)
  - [ ] Update peer metrics post-fetch
  - [ ] Unit tests with mock gRPC server

- [ ] **Implement exceptions**
  - [ ] `PeerUnavailableError`
  - [ ] `PeerFetchError`
  - [ ] `LayerChecksumMismatchError`

- [ ] **Integration tests**
  - [ ] Mock gRPC server simulating real peer
  - [ ] Test fetch success path
  - [ ] Test retry on timeout
  - [ ] Test circuit breaker after failures
  - [ ] Test peer ranking logic

### New Module: peer/server.py (Peer Server)

- [ ] **Implement `PeerServerConfig`**
  - [ ] Fields: port, advertise_interval, max_connections, sharing_policy
  - [ ] Validation

- [ ] **Implement `PeerServer`**
  - [ ] `__init__` with config and local cache
  - [ ] `start()` async method
    - [ ] Start gRPC server on port
    - [ ] Start periodic advertisement task
    - [ ] Register with central registry (if configured)
  - [ ] `stop()` graceful shutdown
  - [ ] `_make_servicer()` create gRPC servicer
  - [ ] `_periodic_advertise()` broadcast to registry
  - [ ] `_generate_peer_id()` unique ID
  - [ ] Lifecycle management (start/stop)

- [ ] **Implement gRPC Servicer**
  - [ ] `HasLayer` handler - quick existence check
  - [ ] `FetchLayer` handler - streaming response
  - [ ] `AdvertiseLayers` handler - registry updates
  - [ ] `Heartbeat` handler - health tracking
  - [ ] Authentication middleware (mTLS cert extraction)
  - [ ] Authorization checks (whitelist, rate limits)
  - [ ] Sharing policy enforcement
  - [ ] Streaming serialization (chunk large layers)
  - [ ] Checksum calculation for outgoing data

- [ ] **Implement serialization**
  - [ ] `_serialize_state_dict()` to bytes (maybe msgpack + numpy)
  - [ ] `_deserialize_state_dict()` from bytes
  - [ ] Chunking for memory efficiency (stream 10MB chunks)
  - [ ] Compression option (lz4)

- [ ] **Integration tests**
  - [ ] In-memory gRPC server with test layers
  - [ ] Test HasLayer returns correct bool
  - [ ] Test FetchLayer streams correctly
  - [ ] Test sharing policy blocks unauthorized layers
  - [ ] Test rate limiting
  - [ ] Test concurrent requests

### New Module: peer/ledger.py (Credits & Reputation)

- [ ] **Implement `Ledger`**
  - [ ] Mode: 'local' (in-memory dict)
  - [ ] Mode: 'http' (central server API)
  - [ ] Mode: 'sqlite' (local file-based)
  - [ ] `credit(peer_id, amount, desc)` - add credits
  - [ ] `debit(peer_id, amount, desc)` - subtract with balance check
  - [ ] `get_balance(peer_id)` - query balance
  - [ ] `enforce_rate_limit(peer_id, max_debt)` - can borrow?
  - [ ] Transaction logging (for audit)
  - [ ] Double-entry accounting consistency
  - [ ] Thread-safety (locks for concurrent access)

- [ ] **Implement `ReputationManager`**
  - [ ] `PeerReputation` dataclass: success_rate, avg_latency, total_volume, last_seen
  - [ ] `record_transaction()` update metrics with EMA
  - [ ] `get_reputation_score()` normalized 0-1 score
  - [ ] Persistence across restarts (optional)

- [ ] **Implement `Account`** (if using ledger-style)
  - [ ] Balance, transaction history
  - [ ] Credit limit / overdraft settings

- [ ] **Unit tests**
  - [ ] Credit/debit operations
  - [ ] Balance checks prevent overdraft
  - [ ] Transaction logging
  - [ ] Reputation EMA updates correctly
  - [ ] Score calculation edge cases

- [ ] **Integration tests**
  - [ ] Two peers: one serves, one borrows; ledger updates
  - [ ] Insufficient balance rejection
  - [ ] Reputation changes after multiple transactions

### New Module: peer/security.py (Auth & Encryption)

- [ ] **Implement certificate utilities**
  - [ ] Generate self-signed certs for development
  - [ ] Load CA certificates for mTLS
  - [ ] Extract peer_id from certificate SAN
  - [ ] Certificate validation (expiry, signature)

- [ ] **Implement mTLS for gRPC**
  - [ ] Server credentials: `grpc.ssl_channel_credentials()`
  - [ ] Client credentials with client_cert/client_key
  - [ ] Mutual authentication handshake
  - [ ] Optional: disable for development (insecure mode)

- [ ] **Implement whitelist enforcement**
  - [ ] IP-based whitelist (allowed_peer_cidrs)
  - [ ] Peer ID whitelist (from ledger/registry)
  - [ ] Reject unauthorized connections at gRPC interceptor

- [ ] **Implement request signing** (optional advanced)
  - [ ] Each request signed with peer's private key
  - [ ] Server verifies signature
  - [ ] Prevent replay attacks (nonce/timestamp)

- [ ] **Unit tests**
  - [ ] Cert generation and loading
  - [ ] mTLS connection establishment
  - [ ] Whitelist allow/deny logic
  - [ ] Rejection of unauthenticated connections

- [ ] **Integration tests**
  - [ ] gRPC server with mTLS requiring client certs
  - [ ] Client with valid cert connects successfully
  - [ ] Client without cert rejected

### Config Extensions: config.py

- [ ] **Add `PeerBackendConfig` dataclass**
  - [x] enabled: bool
  - [x] mode: Literal['client', 'server', 'hybrid']
  - [x] registry_mode: Literal['static', 'http', 'dht']
  - [x] bootstrap_peers: List[str]
  - [x] registry_url: Optional[str]
  - [x] max_peers_to_query: int
  - [x] fetch_timeout_ms: float
  - [x] fallback_to_local: bool
  - [x] server_port: int
  - [x] advertise_enabled: bool
  - [x] advertise_interval_s: int
  - [x] max_connections: int
  - [x] sharing_enabled: bool
  - [x] max_share_cache_percent: float
  - [x] min_reputation_to_serve: float
  - [x] price_per_gb: float
  - [x] require_tls: bool
  - [x] ca_cert_path, client_cert_path, client_key_path
  - [x] allowed_peer_cidrs: List[str]
  - [x] ledger_mode: str
  - [x] ledger_server_url: Optional[str]
  - [x] initial_credits: float
  - [ ] Add field: `checksum_verification` (bool, default True)
  - [ ] Add field: `max_debt_gb` (float, overdraft limit)
  - [ ] Add field: `price_dynamic` (bool, use dynamic pricing)
  - [ ] Add field: `share_exclude_layers` (List[str], regex patterns)
  - [ ] Add field: `share_only_layers` (List[str], whitelist)
  - [ ] Add field: `prefetch_from_peers` (bool, enable peer prefetch)
  - [ ] Add field: `max_peer_fetches` (int, concurrent fetch limit)

- [ ] **Extend `EvoLLMConfig`**
  - [x] Add field: `peer: PeerBackendConfig`
  - [ ] Update `__post_init__` to validate peer config
  - [ ] Update `__str__` to include peer settings
  - [ ] Ensure backward compatibility (peer config optional)

- [ ] **Add config helpers**
  - [ ] `merge_configs(base, overrides)` - layer config merging
  - [ ] `config_from_yaml(path)` - load from file
  - [ ] `config_to_yaml(config)` - save to file
  - [ ] `config_from_env()` - override from environment variables

### Integration: evollm_base.py

- [ ] **Modify `_init_evolllm()`**
  - [ ] Replace `create_cache()` with `create_resource_manager()`
  - [ ] Pass load_fn and move_fn to factory
  - [ ] Handle peer server initialization if mode in ('server', 'hybrid')
  - [ ] Start peer server async task
  - [ ] Store `self.resource_manager` instead of `self.cache_manager`
  - [ ] Update `_load_layer_with_cache()` to use `self.resource_manager`
  - [ ] Update `get_cache_stats()` to aggregate from resource_manager
  - [ ] Add `self.peer_server` attribute and shutdown in `__del__`
  - [ ] Handle config.peer.enabled conditional logic

- [ ] **Add factory function** `create_resource_manager(config, load_fn, move_fn)`
  - [ ] Build local_manager if cpu_cache_gb>0 or gpu_layers>0
  - [ ] If peer.enabled: initialize ledger, registry, server (if needed)
  - [ ] Build appropriate composite (client/server/hybrid)
  - [ ] Return ResourceProvider (or None if no caching)

- [ ] **Update `EvoLLMAutoModel.from_pretrained()`**
  - [ ] Pass through peer config
  - [ ] No changes needed if peer config in EvoLLMConfig

### Integration: tensor_loader.py

- [ ] **Modify `HierarchicalTensorLoader.__init__()`**
  - [ ] Replace `cache_manager` param with `resource_manager: ResourceProvider`
  - [ ] Remove direct cache_manager references
  - [ ] Update `load_layer()` to call `self.resource_manager.get_layer()`
  - [ ] Update `get_stats()` to query `self.resource_manager.get_stats()`
  - [ ] Update `shutdown()` to call `self.resource_manager.shutdown()`
  - [ ] Keep prefetch logic unchanged (still async)

### gRPC Infrastructure

- [ ] **Create `evollm/peer/protocol.proto`**
  - [ ] Complete service definition (see plan for details)
  - [ ] Import google/protobuf/empty.proto if needed
  - [ ] Use appropriate types (int64 for sizes, string for checksums)

- [ ] **Generate Python stubs**
  - [ ] Add to setup.py or pyproject.toml: `grpcio-tools` build dependency
  - [ ] Script to regenerate: `python -m grpc_tools.protoc ...`
  - [ ] Commit generated files: `evollm/peer/protocol_pb2.py`, `evollm/peer/protocol_pb2_grpc.py`

- [ ] **Implement connection pooling**
  - [ ] `PeerConnectionPool` class (optional optimization)
  - [ ] Reuse gRPC channels per peer
  - [ ] Health checking and reconnection logic

### Testing Infrastructure

- [ ] **Create test directory structure**
  ```
  tests/
  ├── unit/
  │   ├── test_cache_policy.py
  │   ├── test_config.py
  │   ├── test_hardware_profiler.py
  │   ├── test_tensor_loader.py
  │   └── peer/
  │       ├── test_peer_registry.py
  │       ├── test_peer_fetcher.py
  │       ├── test_peer_server.py
  │       ├── test_ledger.py
  │       ├── test_hybrid_manager.py
  │       └── test_security.py
  ├── integration/
  │   └── peer/
  │       ├── test_two_peer_fetch.py
  │       ├── test_hybrid_local_peer.py
  │       ├── test_checksum_verification.py
  │       └── test_cache_invalidation.py
  └── benchmarks/
      ├── benchmark_throughput.py
      ├── benchmark_cache_hit_rate.py
      └── benchmark_peer_overhead.py
  ```

- [ ] **Add pytest fixtures**
  - [ ] `mock_cache_manager()` - fake cache for testing
  - [ ] `mock_grpc_server()` - in-memory gRPC test server
  - [ ] `temp_model_path()` - create fake checkpoint with layers
  - [ ] `evollm_config()` - common test configs

- [ ] **Add test utilities**
  - [ ] `tests/utils.py` with helper functions
  - [ ] Layer data generators (random tensors)
  - [ ] Checksum calculators

- [ ] **CI/CD configuration** (if using GitHub Actions)
  - [ ] .github/workflows/test.yml
  - [ ] Matrix: Python 3.9, 3.10, 3.11
  - [ ] Install dependencies, run pytest
  - [ ] Lint: black, isort, flake8, mypy

### Documentation

- [ ] **README.md updates**
  - [ ] Add peer-to-peer section
  - [ ] Installation with peer dependencies (grpcio)
  - [ ] Quick start examples (client, server, hybrid)
  - [ ] Configuration reference
  - [ ] Troubleshooting

- [ ] **Add PEER_GUIDE.md**
  - [ ] Overview of P2P architecture
  - [ ] Deployment scenarios (cluster, edge, cloud)
  - [ ] Security considerations
  - [ ] Performance tuning
  - [ ] Monitoring and metrics
  - [ ] FAQ

- [ ] **API documentation** (generate with Sphinx)
  - [ ] conf.py, index.rst
  - [ ] Autodoc for all public modules
  - [ ] Examples in docstrings

- [ ] **Add architecture diagrams**
  - [ ] resource_provider.png (class diagram)
  - [ ] peer_communication.png (sequence diagram)
  - [ ] cluster_deployment.png (infrastructure diagram)

### CLI / Examples

- [ ] **Create `evollm/cli.py`** (if not exists)
  - [ ] `main()` with argparse
  - [ ] Subcommands: `serve`, `infer`, `benchmark`
  - [ ] Config loading from file or flags
  - [ ] Logging setup

- [ ] **Add example scripts**
  - [ ] `examples/01_basic_inference.py` - Simple inference
  - [ ] `examples/02_peer_server.py` - Start a peer server
  - [ ] `examples/03_peer_client.py` - Connect to peers
  - [ ] `examples/04_hybrid_cluster.py` - 2-node cluster demo
  - [ ] `examples/05_benchmark.py` - Compare configs

- [ ] **Add Jupyter notebook examples**
  - [ ] `examples/notebooks/01_EvoLLM_Tutorial.ipynb`
  - [ ] `examples/notebooks/02_Peer_Clustering.ipynb`

### Build & Packaging

- [ ] **setup.py or pyproject.toml**
  - [ ] Package name, version, description
  - [ ] Authors, license
  - [ ] Install requires: torch, transformers, psutil, tqdm, grpcio, etc.
  - [ ] Extras: `[peer]` for gRPC dependencies
  - [ ] Entry points: `evollm=evollm.cli:main`

- [ ] **Manifest**
  - [ ] Include non-Python files (protocol.proto, configs)
  - [ ] Exclude test files, .git, __pycache__

- [ ] **Docker support** (optional)
  - [ ] Dockerfile for CPU-only
  - [ ] Dockerfile for GPU (CUDA)
  - [ ] docker-compose.yml for 2-node cluster test

### Performance Optimization

- [ ] **Profile forward pass** with cProfile
  - [ ] Identify hotspots (layer loading, movement, computation)
  - [ ] Optimize serialization/deserialization
  - [ ] Reduce allocations

- [ ] **Memory optimization**
  - [ ] Use memory-mapped files for layer loading (torch.load with mmap)
  - [ ] Pin memory for async transfers (pin_memory=True)
  - [ ] Optimize cache entry metadata (use __slots__ if beneficial)

- [ ] **Async optimization**
  - [ ] Use asyncio instead of ThreadPoolExecutor? (evaluate tradeoffs)
  - [ ] Overlap disk I/O with GPU compute (pipeline better)
  - [ ] Batch prefetch requests

- [ ] **Serialization optimization**
  - [ ] Use `torch.save`/`torch.load` with pickle protocol 4/5
  - [ ] Consider `safetensors` format for faster loading
  - [ ] Add compression (lz4, zstd) with configurable level

### Monitoring & Observability

- [ ] **Add metrics export**
  - [ ] Prometheus metrics endpoint (/metrics)
  - [ ] Cache hit rates, load times, peer fetch latencies
  - [ ] Ledger balances, reputation scores
  - [ ] System metrics (RAM, VRAM, disk usage)

- [ ] **Add structured logging**
  - [ ] JSON logs with fields: timestamp, level, component, message, extra
  - [ ] Log levels: DEBUG, INFO, WARNING, ERROR
  - [ ] Rotate log files

- [ ] **Add tracing** (OpenTelemetry)
  - [ ] Trace a full layer fetch request
  - [ ] Propagate trace context across gRPC
  - [ ] Export to Jaeger/Zipkin

- [ ] **Add health check endpoint**
  - [ ] /healthz returns 200 if model loaded and operational
  - [ ] /readyz returns 200 if model ready to serve
  - [ ] Include subsystem status (cache, peers, ledger)

---

## Phase 2½: Energy-Aware Resource Management (NEXT AFTER P2P)

**Goal**: Optimize energy consumption and costs through dynamic resource state management and crowd-sourced wake requests.

**Timeline**: ~10 weeks with 1 engineer

### Energy Profiling & Accounting

- [ ] **Create `evollm/energy.py` module**
  - [ ] `ResourceEnergyProfile` dataclass with power specs (p_max_w, p_idle_w, p_hibernate_w, p_sleep_w)
  - [ ] `ResourceEnergyLease` for tracking energy consumption and cost
  - [ ] `EnergyPriceModel` with time-of-day pricing and carbon cost support
  - [ ] Default hardware profiles for common GPUs (NVIDIA) and CPUs (Intel/AMD)
  - [ ] Unit tests for energy calculations (Joules → Wh → cost)

- [ ] **Extend `ResourceLease` to include energy tracking**
  - [ ] Add fields: `energy_estimate_wh`, `energy_actual_wh`, `energy_cost_credits`
  - [ ] Instrument `lease.accumulate()` to track energy during lease duration
  - [ ] Finalize energy accounting on lease release

- [ ] **Instrument resource allocation with energy costs**
  - [ ] Before allocation: estimate energy cost (including wake cost if sleeping)
  - [ ] Check against requester's `max_price_per_hour` constraint
  - [ ] Provider profitability check: expected revenue > total cost
  - [ ] Periodic energy sampling during lease (read from hardware sensors or estimate)

- [ ] **Add CLI commands for energy management** (`evollm/cli/energy.py`)
  - [ ] `evosh energy profile` - Show current energy profile for resources
  - [ ] `evosh energy estimate --resource gpu --duration 2h` - Estimate cost
  - [ ] `evosh ledger energy --last 24h` - Show energy consumption breakdown
  - [ ] `evosh energy watch` - Live monitoring of states and consumption

### Resource State Management

- [ ] **Create `evollm/resource_state.py`**
  - [ ] `ResourceState` enum: FUNCTIONAL, HIBERNATE, SLEEP, OFFLINE
  - [ ] `ResourceStateManager` class
    - [ ] Track current state and `state_since` timestamp
    - [ ] `update()` method for periodic auto-transition checks
    - [ ] `transition_to(new_state)` async method with proper power management
    - [ ] Wake energy amortization tracking
    - [ ] State change event notifications (hooks for logging/metrics)
  - [ ] Unit tests for state transition logic and timing

- [ ] **Integrate state manager with LocalResourceManager**
  - [ ] Wrap each resource (GPU, CPU cores) with `ResourceStateManager`
  - [ ] On `allocate()`: ensure resource is FUNCTIONAL (trigger wake if needed)
  - [ ] On `release()`: start idle timer, potentially transition to HIBERNATE/SLEEP
  - [ ] Add `get_resource_state()` to ResourceProvider interface
  - [ ] Handle state transitions gracefully (block new allocations during transition)

- [ ] **Implement power measurement abstraction**
  - [ ] GPU: Use NVML (NVIDIA) or ROCm-SMI (AMD) to read real-time power
  - [ ] CPU: Use Intel RAPL (`/sys/class/powercap/`) or AMD equivalent
  - [ ] Fallback: estimate from utilization and known profile
  - [ ] Unit tests with mock power sensors
  - [ ] Config: `power_measurement: auto|nvml|rapl|manual`

### Crowd Wake Requests

- [ ] **Extend gRPC protocol** (`peer/protocol.proto`)
  - [ ] Add `WakeService` with `RequestWake` RPC
    ```protobuf
    message WakeRequest {
      string requesting_peer_id = 1;
      ResourceType resource_type = 2;
      float quantity = 3;
      int64 duration_s = 4;
      float max_price_per_hour = 5;
    }
    message WakeResponse {
      bool accepted = 1;
      string reason = 2;
      Lease lease = 3;
    }
    ```
  - [ ] Regenerate Python stubs

- [ ] **Implement `PeerServer.handle_wake_request()`**
  - [ ] Validate requester (mTLS, whitelist check)
  - [ ] Query `ResourceStateManager`: is resource SLEEP/HIBERNATE?
  - [ ] Evaluate profitability: bid price > estimated cost (operational + wake)
  - [ ] Check owner policy (min bid, daily wakes, allowed hours)
  - [ ] If yes: transition to FUNCTIONAL, allocate lease, return
  - [ ] If no: return rejection with reason code
  - [ ] Log all wake requests for audit
  - [ ] Unit tests with mock state manager and policy

- [ ] **Implement `PeerClient.request_remote_wake()`**
  - [ ] Try to find peers with needed resource in SLEEP/HIBERNATE state
  - [ ] Send `WakeRequest` with bid (based on user's urgency/budget)
  - [ ] On success: receive Lease and use normally
  - [ ] On failure: fallback to next cheapest peer or local-only
  - [ ] Retry logic with exponential backoff
  - [ ] Unit tests with mock gRPC server

- [ ] **Update `PeerRegistry` to track and advertise energy states**
  - [ ] Peers periodically advertise: `{"gpu": "hibernate", "cpu": "functional", "load": 0.3}`
  - [ ] `query_available_with_energy()` filters by state preference
  - [ ] Adjust prices based on state (HIBERNATE: +wake premium, SLEEP: +larger premium)
  - [ ] Sort by (state priority, adjusted price)

### Owner Policy & Control

- [ ] **Create `evollm/energy_policy.py`**
  - [ ] `EnergyPolicy` dataclass with fields:
    - `auto_hibernate_idle_timeout_s: int = 300`
    - `auto_sleep_idle_timeout_s: int = 1800`
    - `min_bid_price_per_kwh: float = 0.03`
    - `max_daily_wakes: int = 50`
    - `allowed_wake_hours: Tuple[int, int] = (0, 24)`
    - `max_daily_energy_kwh: float = 100.0`
    - `resource_overrides: Dict[ResourceType, ResourcePolicy]`
  - [ ] `can_wake_for_bid(resource_type, bid_price_per_kwh) -> (bool, reason)`
  - [ ] `record_wake()` to increment daily counters
  - [ ] `_compute_daily_energy()` aggregation
  - [ ] Daily reset logic (midnight or rolling 24h)
  - [ ] Unit tests for policy checks

- [ ] **Add configuration file support**
  - [ ] Load from `~/.evollm/energy_policy.yaml` with sensible defaults
  - [ ] Merge with config file if present
  - [ ] CLI overrides: `evosh energy policy set --key value`
  - [ ] Config validation: warn on invalid values

- [ ] **Add CLI for policy management** (`evollm/cli/energy.py`)
  - [ ] `evosh energy policy show` - Display current policy
  - [ ] `evosh energy policy set --min-bid-price 0.06`
  - [ ] `evosh energy policy set --max-daily-wakes 10`
  - [ ] `evosh energy policy enable-auto-hibernate --idle-timeout 600`
  - [ ] `evosh energy policy reset-counters` - Clear daily wake/energy counters

- [ ] **Real-time monitoring command**
  - [ ] `evosh energy watch` - Live dashboard showing:
    - States: FUNCTIONAL (2 GPU, 8 CPU), HIBERNATE (1 GPU)
    - Energy consumption today: 3.2 kWh ($0.48)
    - Wakes today: 5/50
    - Current price: $0.15/kWh
  - [ ] Auto-refresh every 2 seconds
  - [ ] Color-coded output (green=good, red=limit approaching)

### Ledger Integration

- [ ] **Extend `ResourceLedger` for energy accounting**
  - [ ] Add separate energy balance fields: `energy_earned_wh`, `energy_consumed_wh`
  - [ ] `debit_energy(lease_id, cost_credits)` method
  - [ ] `credit_energy(peer_id, cost_credits)` method
  - [ ] Track energy separately from data transfer credits (or unified)
  - [ ] Transaction logging includes energy vs regular credit type
  - [ ] Unit tests for energy accounting

- [ ] **Integrate energy costs into allocation flow**
  - [ ] Provider: add `energy_cost_estimate` to lease metadata when allocating to requester
  - [ ] Requester: energy cost debited from ledger along with data transfer cost
  - [ ] On lease release: adjust final actual cost (if different from estimate)
  - [ ] Provider earns energy credits (can be used to offset own consumption or sold)

### Config Extensions

- [ ] **Update `config.py`**
  - [ ] Add `EvoLLMConfig.energy_aware: bool = True` (opt-in, default False for v1)
  - [ ] Add `EvoLLMConfig.energy_policy_file: Optional[str]`
  - [ ] Add `EvoLLMConfig.auto_hibernate_idle_s: int = 300`
  - [ ] Add `EvoLLMConfig.bid_price_per_kwh: float = 0.10` (what I'm willing to pay)
  - [ ] Add `EvoLLMConfig.energy_measurement: Literal['auto','nvml','rapl','manual']`
  - [ ] Add `EvoLLMConfig.enable_crowd_wake: bool = True`
  - [ ] Update validation logic for energy fields
  - [ ] Update `__str__` to include energy settings

### Advanced Features (Stretch Goals)

- [ ] **Predictive warming**
  - [ ] Observe access patterns (time of day, day of week)
  - [ ] Auto-wake 5 minutes before expected workload
  - [ ] Simple: cron-like schedule from config
  - [ ] Advanced: ML model (prophet, LSTM) for prediction

- [ ] **Carbon-aware scheduling**
  - [ ] Integrate with carbon intensity API (WattTime, ElectricityMaps)
  - [ ] Prefer to do heavy compute when grid is green
  - [ ] May voluntarily hibernate during dirty periods
  - [ ] Config: `max_carbon_intensity_kg_co2_per_kwh`
  - [ ] Display carbon impact in CLI

- [ ] **Energy market integration**
  - [ ] Connect to real-time spot markets (if available in region)
  - [ ] Dynamic pricing based on grid load
  - [ ] Auto-hibernate during price spikes
  - [ ] Aggressively wake during negative price periods (over-supply)

- [ ] **Hardware degradation modeling**
  - [ ] Factor wake cycles into hardware lifespan estimates
  - [ ] NVMe: limit spin-up cycles
  - [ ] GPU: memory wear from thermal cycles
  - [ ] Policy: `max_wakes_per_day` with smart grouping

### Documentation

- [ ] **Add `docs/energy_aware_resource_management.md`**
  - [ ] Full design doc (from PLAN.md)
  - [ ] Configuration reference
  - [ ] CLI usage examples
  - [ ] Deployment scenarios (single node, cluster, edge)
  - [ ] Troubleshooting guide

- [ ] **Update README.md**
  - [ ] Section on energy management features
  - [ ] Benefits: cost savings, carbon reduction
  - [ ] Quick start: enable with `--energy-aware`
  - [ ] Policy configuration basics

- [ ] **API documentation** (Sphinx autodoc)
  - [ ] Document all energy-related classes and functions
  - [ ] Add examples in docstrings

### Testing & Validation

- [ ] **Unit tests**
  - [ ] State transitions (FUNCTIONAL → HIBERNATE → SLEEP → FUNCTIONAL)
  - [ ] Energy accounting accuracy (power × duration with price)
  - [ ] Policy rejection (below min bid, daily limit exceeded, outside hours)
  - [ ] Wake cost amortization correct

- [ ] **Integration tests**
  - [ ] Multi-peer wake scenario: Peer A requests B to wake GPU, B complies, ledger updates
  - [ ] Policy enforcement: daily wake limit, allowed hours, min bid
  - [ ] Energy cost flow: charged to requester, credited to provider
  - [ ] State advertisement and discovery
  - [ ] Automatic state transitions after idle timeouts

- [ ] **Performance validation**
  - [ ] Measure wake latency: <2s from HIBERNATE, <10s from SLEEP
  - [ ] Benchmark overhead: <1% performance impact when FUNCTIONAL
  - [ ] Measure actual power savings vs theoretical (aim 30-50% idle power reduction)
  - [ ] Validate energy meter accuracy against hardware sensors (if available)

- [ ] **Simulations**
  - [ ] Simulate 100 peers with varied workloads (trace-driven)
  - [ ] Validate profitability: waking resources earns > wake cost
  - [ ] Validate grid stability: can hibernate during peak load
  - [ ] Measure community-wide energy savings

### Backward Compatibility & Risk Mitigation

- [ ] **Default OFF**: Energy-aware features opt-in, not default in v1
- [ ] **API versioning**: PeerServer API v2 (energy features), v1 (no energy)
- [ ] **Protocol negotiation**: During handshake, peers exchange supported versions
- [ ] **Feature flags**: All energy features behind config flags
- [ ] **Document risks**: Hardware incompatibility, wear from wakes, policy complexity

---

## Phase 3: EvoOS - Distributed Operating System (FUTURE)

### Generalized Resource Sharing (ANY System Object)

> **Vision**: Share ANY system resource - not just model layers, but files, processes, displays, GPUs, keyboards, etc.
> **Architecture**: Extend `ResourceProvider` to support multiple resource categories (see `docs/generalized_resource_sharing.md`)

- [ ] **Extend ResourceProvider abstraction**
  - [ ] Rename `get_layer()` → `allocate(spec: ResourceSpec) → ResourceLease`
  - [ ] Keep `get_layer()` as convenience wrapper for backward compatibility
  - [ ] Define `ResourceType` enum covering all categories:
    - [ ] STORAGE: FILE, DIRECTORY, BLOB, DATABASE, CHECKPOINT
    - [ ] COMPUTE: PROCESS, CONTAINER, FUNCTION, GPU_KERNEL, JOB
    - [ ] INTERACTIVE: KEYBOARD, DISPLAY, FRAMEBUFFER, AUDIO_*, CONSOLE
    - [ ] DEVICE: GPU, CPU_CORE, NPU, TPU, NIC, CAMERA, MICROPHONE, DISK_IO
    - [ ] VIRTUAL: PID_NAMESPACE, NET_NAMESPACE, USER_NAMESPACE, IPC_NAMESPACE, MOUNT_NAMESPACE
    - [ ] SPECIAL: MODEL_LAYER (existing), LICENSE
  - [ ] Define `ResourceSpec`, `ResourceOffer`, `ResourceLease` dataclasses
  - [ ] Add new abstract methods: `snapshot()`, `restore()`, `transfer()`, `can_preempt()`

- [ ] **Implement StorageResourceProvider**
  - [ ] `FileResourceProvider` - Share files/directories
  - [ ] Stream download/upload with chunking
  - [ ] Access control (read-only vs read-write)
  - [ ] Tests: allocate file, stream to local

- [ ] **Implement ComputeResourceProvider**
  - [ ] `ProcessResourceProvider` - Execute remote processes (xterm/SSH use case!)
  - [ ] Spawn subprocess with stdin/stdout/stderr pipes
  - [ ] lifecycle management (monitor, kill on expiry)
  - [ ] Container support (Docker/Podman) for isolation
  - [ ] Tests: allocate shell, send commands, read output

- [ ] **Implement InteractiveResourceProvider**
  - [ ] `DisplayResourceProvider` - Remote framebuffer (VNC/RDP)
  - [ ] `KeyboardResourceProvider` - Input injection (keystrokes, mouse)
  - [ ] `AudioResourceProvider` - Audio streaming
  - [ ] Bidirectional streaming (I/O events)
  - [ ] Compression for video (H.264, AV1)
  - [ ] Tests: allocate display, view Peer A's desktop

- [ ] **Implement DeviceResourceProvider**
  - [ ] `GPUResourceProvider` - Time-share GPU (CUDA context)
  - [ ] `CPUResourceProvider` - CPU core allocation with cgroups
  - [ ] Resource scheduling (time-slicing, preemption)
  - [ ] QoS guarantees (latency, bandwidth)
  - [ ] Isolation (CUDA MPS/MIG, cgroups)
  - [ ] Tests: allocate GPU, launch kernel remotely

- [ ] **Extend Registry for all resource types**
  - [ ] Index: `(resource_type, constraints) → [offers]`
  - [ ] Query API: `registry.query(type=GPU, min_vram=4, max_price=0.5)`
  - [ ] Advertisements include `resource_type` field
  - [ ] Mixed-type queries ("who has resources for training job?")

- [ ] **evosh CLI** (unified resource management)
  - [ ] `evosh resources list` - Show all available offers
  - [ ] `evosh resources allocate --type <type> --spec <json>` - Generic allocate
  - [ ] `evosh file allocate --peer <id> --path <path>` ( shortcuts )
  - [ ] `evosh process allocate --peer <id> --cmd "python train.py"`
  - [ ] `evosh shell allocate --peer <id>` (xterm)
  - [ ] `evosh display allocate --peer <id>` (remote desktop)
  - [ ] `evosh gpu allocate --peer <id> --quantity 4`
  - [ ] `evosh resources release --lease <id>`
  - [ ] `evosh resources watch` - Real-time monitoring
  - [ ] `evosh ledger balance` - Show credits

- [ ] **Security & Ethics for high-risk resources**
  - [ ] Keyboard/display/camera require explicit consent UI
  - [ ] Visible indicators (LED, on-screen warning)
  - [ ] Time-limited auto-revoke (max 1h for interactive)
  - [ ] Audit logging of all allocations
  - [ ] Capability-based access control (fine-grained ACLs)
  - [ ] Sandboxing: process allocation uses containers/seccomp
  - [ ] Ethical use policy documentation

- [ ] **Resource lifecycle & control**
  - [ ] `lease.renew()` - Extend duration
  - [ ] `lease.transfer()` - Migrate to another peer
  - [ ] `lease.snapshot()` - Checkpoint state (for long-running jobs)
  - [ ] `lease.restore()` - Resume from checkpoint
  - [ ] Auto-renewal for long jobs (if capacity available)

- [ ] **Integration with EvoLLM model**
  - [ ] Config: `evollm_config.resource_manager = HybridResourceManager([...])`
  - [ ] Can use file provider for model loading (instead of local disk)
  - [ ] Can use GPU provider for compute offload (Phase 3b)
  - [ ] Transparent: model code unchanged, just different resource source

- [ ] **Tests & Examples**
  - [ ] End-to-end: allocate file, stream, verify checksum
  - [ ] End-to-end: allocate process, execute command, capture output
  - [ ] End-to-end: allocate display, view via VNC
  - [ ] Example: `examples/generalized_resources.py` - Demo all types
  - [ ] Example: `examples/xterm_via_peer.py` - Your use case

- [ ] **Documentation**
  - [ ] `docs/generalized_resource_sharing.md` (complete ✅)
  - [ ] API reference for all ResourceProvider implementations
  - [ ] Security considerations guide
  - [ ] Performance tuning guide per resource type
  - [ ] Troubleshooting common errors

### Multi-Node Orchestration

- [ ] **Implement DHT-based peer discovery** (replace static bootstrap)
  - [ ] Kademlia protocol implementation or library
  - [ ] Node join/leave handling
  - [ ] Replication of layer→peer mappings
  - [ ] Bootstrap nodes configuration

- [ ] **Implement load balancer**
  - [ ] Central scheduler or distributed consensus
  - [ ] Work stealing between overloaded/underloaded peers
  - [ ] Queue management for requests

- [ ] **Add live migration**
  - [ ] Move cache entries between peers
  - [ ] Checkpoint and restore model state
  - [ ] Network-level handoff

- [ ] **Fault tolerance**
  - [ ] Raft/Paxos for leader election (if using central coordinator)
  - [ ] Replicate critical metadata
  - [ ] Automatic failover

### Crypto-Economic System

- [ ] **Design token economics**
  - [ ] Token supply schedule
  - [ ] Inflation/deflation mechanics
  - [ ] Staking and slashing

- [ ] **Implement blockchain or accounting layer**
  - [ ] Smart contracts for resource leasing
  - [ ] Proof-of-work/pos for decentralized consensus
  - [ ] Distributed ledger (maybe use existing like Ethereum?)

- [ ] **Add marketplace**
  - [ ] Order book for layer rentals
  - [ ] Price discovery mechanism
  - [ ] Futures contracts for reserved capacity

- [ ] **Reputation system v2**
  - [ ] Sybil attack resistance
  - [ ] Collusion detection
  - [ ] Long-term reputation building

### Mobile & Edge Support

- [ ] **Mobile-specific config** (from PLAN.md)
  - [ ] `EvoLLMMobileConfig` subclass
  - [ ] Thermal throttling detection
  - [ ] Battery monitoring
  - [ ] Adaptive power management

- [ ] **Cross-platform support**
  - [ ] iOS (Core ML, Metal)
  - [ ] Android (NNAPI, Qualcomm NN)
  - [ ] Raspberry Pi (ARM64)

- [ ] **Intermittent connectivity**
  - [ ] Offline-first operation
  - [ ] Opportunistic sync when online
  - [ ] Compressed model distribution

### Advanced Features

- [ ] **Multi-model support**
  - [ ] Concurrent serving of multiple models
  - [ ] Model version management
  - [ ] A/B testing canary deployments

- [ ] **Training support**
  - [ ] Gradient accumulation across peers
  - [ ] Federated learning
  - [ ] Differential privacy

- [ ] **Plugin system**
  - [ ] Hook points for custom resource managers
  - [ ] Python plugin API
  - [ ] Configuration plugins

- [ ] **Web UI / Dashboard**
  - [ ] Cluster topology visualization
  - [ ] Real-time metrics
  - [ ] Peer management console
  - [ ] Cost/revenue reporting

---

## Research & Experimentation

- [ ] **Benchmark existing solutions**
  - [ ] Compare with FlexGen, HuggingFace accelerate, vLLM
  - [ ] Compare with Petals, BLOOM
  - [ ] Publish benchmark results

- [ ] **Model specific optimizations**
  - [ ] Llama-specific: KV cache optimizations
  - [ ] Mixtral: MoE expert routing with cache
  - [ ] GPT-style: attention optimization

- [ ] **Academic paper**
  - [ ] Write paper on hierarchical caching + P2P
  - [ ] Submit to conference (OSDI, MLSys, etc.)
  - [ ] Open source release with paper

---

## Project Management

### Code Quality

- [ ] **Linting & Formatting**
  - [ ] Configure black (line length 100)
  - [ ] Configure isort
  - [ ] Configure flake8
  - [ ] Pre-commit hooks to auto-format
  - [ ] CI enforces linting

- [ ] **Type checking**
  - [ ] Add mypy configuration
  - [ ] Type annotate all public functions
  - [ ] CI runs mypy in strict mode

- [ ] **Testing coverage**
  - [ ] Aim for 80%+ coverage
  - [ ] Track coverage with pytest-cov
  - [ ] CI badge for coverage %

- [ ] **Security scanning**
  - [ ] Bandit for security issues
  - [ ] Safety for dependency vulnerabilities
  - [ ] Snyk or similar

### Dependency Management

- [ ] **Pin dependencies**
  - [ ] requirements.txt or pyproject.toml
  - [ ] Separate dev/prod dependencies
  - [ ] Use Dependabot or Renovate for updates

- [ ] **Minimize dependencies**
  - [ ] Avoid unnecessary packages
  - [ ] Use standard library when possible
  - [ ] Optional imports for peer features

### Release Management

- [ ] **Version scheme** - Semantic Versioning (MAJOR.MINOR.PATCH)
- [ ] **CHANGELOG.md** - Track changes per release
- [ ] **GitHub Releases** - Package wheels, source dist
- [ ] **Docker images** - Publish to Docker Hub/ghcr.io
- [ ] **Release checklist**
  - [ ] Run full test suite
  - [ ] Update documentation
  - [ ] Bump version in __init__.py
  - [ ] Build and upload to PyPI

### Community

- [ ] **Create website** - evolllm.ai or GitHub Pages
- [ ] **Discord/Slack** - Community chat
- [ ] **Contributing guide** - CONTRIBUTING.md
- [ ] **Code of conduct** - CODE_OF_CONDUCT.md
- [ ] **Governance model** - How decisions are made
- [ ] **Roadmap public** - Keep this TODO.md updated and public

---

## Immediate Next Steps (Priority Order)

1. 🔴 **Complete Phase 1 Testing** - Before adding P2P, ensure single-node is solid
   - [ ] Write unit tests for cache_policy.py
   - [ ] Write integration test for EvoLLMModel with small model
   - [ ] Benchmark baseline performance

2. 🔴 **Implement ResourceProvider abstraction** - Foundation for P2P
   - [ ] Create resource_provider.py with ABC
   - [ ] Implement LocalResourceManager
   - [ ] Refactor create_cache() to return ResourceProvider
   - [ ] Update HierarchicalTensorLoader to accept ResourceProvider

3. 🔴 **Basic Peer Client** - Fetch from single peer
   - [ ] Define gRPC protocol
   - [ ] Generate stubs
   - [ ] Implement PeerRegistry (static mode)
   - [ ] Implement PeerLayerFetcher basic fetch
   - [ ] Integrate with HybridResourceManager

4. 🟡 **Basic Peer Server** - Serve local cache
   - [ ] Implement PeerServer with gRPC
   - [ ] Implement sharing policy checks
   - [ ] Advertise to registry
   - [ ] Test with client

5. 🟡 **Ledger & Reputation**
   - [ ] Implement Ledger (local mode)
   - [ ] Implement ReputationManager
   - [ ] Wire into PeerLayerFetcher ranking
   - [ ] Test economic transactions

6. 🟢 **Polish & Optimize**
   - [ ] Add comprehensive error handling
   - [ ] Add retries and timeouts
   - [ ] Tune default parameters
   - [ ] Performance tuning

7. 🟢 **Documentation & Release**
   - [ ] Write PEER_GUIDE.md
   - [ ] Create example scripts
   - [ ] Update README
   - [ ] Release v0.2.0 with P2P support

---

## Open Questions / Decisions Needed

1. **Discovery mechanism**: Central registry vs DHT?
   - Recommendation: Start with static bootstrap, add central registry in v0.3

2. **Ledger persistence**: Should ledger survive restarts?
   - If yes: Use SQLite; if no: in-memory dict (simpler)

3. **Security**: Enforce mTLS or make optional?
   - Recommendation: Optional (default False), required for production

4. **Pricing**: Free sharing or credits by default?
   - Recommendation: Free by default, credits opt-in

5. **Cache consistency**: Strict (fail on checksum mismatch) or permissive?
   - Recommendation: Strict (fail), with `--skip-checks` override

6. **Transport**: gRPC vs HTTP/JSON?
   - Recommendation: gRPC (performance, streaming, type safety)

7. **Serialization**: torch.save, msgpack, safetensors?
   - Recommendation: torch.save for simplicity, optimize later

8. **Registry implementation**: Central HTTP server or embed in peers?
   - Recommendation: Simple HTTP server (exp/horseradish), participants run it

---

## Success Metrics (Definition of Done)

### Phase 1 Complete
- [ ] All unit tests passing (80%+ coverage)
- [ ] Integration test: 70B model runs with 32GB cache
- [ ] Throughput benchmark: >1 token/sec (vs AirLLM 0.05)
- [ ] Memory usage within config bounds
- [ ] Documentation complete

### Phase 2 Complete
- [ ] 2-node cluster: Peer A borrows from Peer B
- [ ] End-to-end: layer fetched, verified, cached locally
- [ ] Ledger debits/credits tracked correctly
- [ ] Reputation updates after transactions
- [ ] Throughput increases (collective cache > individual)
- [ ] Graceful degradation: peer failure → fallback to disk
- [ ] Security: mTLS connection validated
- [ ] Documentation: PEER_GUIDE.md complete

### Phase 3 Vision
- [ ] 4+ node cluster with dynamic membership
- [ ] Load balancing across nodes
- [ ] Economic transactions with crypto ledger
- [ ] Mobile device integration
- [ ] 1TB+ model inference across cluster

---

**Maintainer**: @dacineu
**Last Updated**: 2026-03-17
**Related Plans**: See `.claude/plans/hashed-brewing-candle.md` for detailed peer architecture design
