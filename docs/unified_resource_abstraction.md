# Unified Resource Abstraction for EvoOS

**Status**: Conceptual Design
**Date**: 2026-03-17
**Context**: Extends the energy-aware resource management plan to create a full operating system abstraction

---

## Vision: EvoOS - A Distributed Operating System for AI

EvoOS reimagines resource management for AI workloads, treating **all resources** (compute, memory, storage, network, energy, models, data) as first-class, discoverable, and allocatable entities across a peer-to-peer network. It's an **operating system** that spans multiple physical machines, providing a unified resource plane.

### Traditional OS vs EvoOS

```
┌─────────────────────────────────────────────────────────────┐
│                    Traditional OS                         │
│  Single machine, single kernel, single address space      │
├─────────────────────────────────────────────────────────────┤
│  Processes  │  Files  │  Memory  │  Devices  │  Network    │
│  (PID)      │  (FD)   │  (VMA)   │  (driver)  │  (socket)   │
└─────────────────────────────────────────────────────────────┘
            Manages ONE machine's resources

┌─────────────────────────────────────────────────────────────┐
│                      EvoOS                                │
│  Distributed across peers, consensus-based, global view   │
├─────────────────────────────────────────────────────────────┤
│  Resources  │  Leases  │  Locality  │  Pricing  │  Energy  │
│  (spec)     │  (agmt)  │  (affinity)│  (market) │  (cost)  │
└─────────────────────────────────────────────────────────────┘
            Manages MANY machines as ONE system
```

---

## Core Abstraction: The Resource Model

### 1. Resource Types (RTypes)

All resources are typed. Types are hierarchical and extensible:

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

class ResourceCategory(Enum):
    """Top-level resource categories"""
    COMPUTE = "compute"          # CPU, GPU, TPU, NPU, FPGA
    MEMORY = "memory"            # RAM, VRAM, cache
    STORAGE = "storage"          # Disk, SSD, NVMe, blob store
    NETWORK = "network"          # Bandwidth, ports, sockets
    MODEL = "model"              # LLM layers, embeddings, checkpoints
    DATA = "data"                # Datasets, files, databases
    INTERACTIVE = "interactive"  # Display, keyboard, audio
    SPECIAL = "special"          # Licenses, certificates, API keys

class ResourceType:
    """
    Hierarchical resource type identifier.
    Format: category.subcategory.variant
    Examples:
      - compute.cpu.core
      - compute.gpu.sm
      - memory.ram.byte
      - model.layer.transformer
      - storage.disk.iops
    """
    def __init__(self, category: str, subcategory: str, variant: str = "default"):
        self.category = category
        self.subcategory = subcategory
        self.variant = variant

    def __str__(self):
        return f"{self.category}.{self.subcategory}.{self.variant}"

    def matches(self, pattern: str) -> bool:
        """Check if this type matches a pattern (with wildcards)"""
        # Implement wildcard matching
        pass

# Predefined resource types
R = {
    'CPU_CORE': ResourceType('compute', 'cpu', 'core'),
    'GPU_SM': ResourceType('compute', 'gpu', 'sm'),  # Streaming multiprocessor
    'GPU_MEMORY': ResourceType('memory', 'vram', 'byte'),
    'RAM': ResourceType('memory', 'ram', 'byte'),
    'DISK_IOPS': ResourceType('storage', 'disk', 'iops'),
    'NET_BANDWIDTH': ResourceType('network', 'bandwidth', 'bps'),
    'MODEL_LAYER': ResourceType('model', 'layer', 'transformer'),
    'ATTENTION_KV': ResourceType('model', 'kv_cache', 'token'),
    'FILE': ResourceType('data', 'file', 'read'),
    'PROCESS': ResourceType('compute', 'process', 'slot'),
    'DISPLAY': ResourceType('interactive', 'display', 'framebuffer'),
}
```

### 2. Resource Specification (ResourceSpec)

A request for resources, analogous to `open()` or `fork()` syscalls:

```python
@dataclass
class ResourceSpec:
    """
    What resources do you need? This is the REQUEST.
    Analogous to: process control block, open file table entry, etc.
    """
    resource_type: ResourceType
    quantity: float  # How many units (can be fractional for shared resources)
    constraints: Dict[str, Any] = field(default_factory=dict)
    duration_s: Optional[float] = None  # How long needed (None = until release)
    priority: int = 0  # Lower = higher priority
    affinity: Optional[str] = None  # Preferred peer_id or locality hint
    energy_budget_wh: Optional[float] = None  # Max energy willing to consume
    max_price_credits: Optional[float] = None  # Max cost willing to pay

    # Common constraints:
    # - min_performance: float (e.g., tokens/sec for GPU)
    # - max_latency_ms: float (e.g., for network)
    # - required_features: List[str] (e.g., ['fp16', 'tensor cores'])
    # - isolation_level: Literal['shared', 'dedicated', 'exclusive']
```

**Example**:
```python
# Need 4GB of GPU memory, exclusive access, willing to pay $0.50
spec = ResourceSpec(
    resource_type=R['GPU_MEMORY'],
    quantity=4.0,  # GB
    constraints={
        'isolation_level': 'dedicated',
        'min_performance': 100.0,  # GB/s
        'required_features': ['fp16'],
    },
    duration_s=3600,
    max_price_credits=0.50
)

# Need 1 CPU core, shared, with energy budget
spec = ResourceSpec(
    resource_type=R['CPU_CORE'],
    quantity=1.0,
    constraints={'isolation_level': 'shared'},
    energy_budget_wh=10.0  # Max 10 watt-hours
)
```

### 3. Resource Offer (ResourceOffer)

A provider's advertisement of available resources, analogous to `/proc` or system info:

```python
@dataclass
class ResourceOffer:
    """
    What resources are available? This is the ADVERTISEMENT.
    Analogous to: system capacity, free memory, etc.
    """
    resource_type: ResourceType
    available: float  # Units available
    total: float  # Total capacity
    price_per_unit: float  # Credits per unit (or energy if energy-aware)
    peer_id: str  # Who provides this
    locality_hints: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Dynamic metrics:
    current_load: float = 0.0  # 0-1 utilization
    avg_latency_ms: float = 0.0
    reliability_score: float = 1.0  # 0-1 uptime/quality
    energy_state: Optional[str] = None  # functional/hibernate/sleep
    features: List[str] = field(default_factory=list)

    def can_fulfill(self, spec: ResourceSpec) -> bool:
        """Check if this offer satisfies the spec"""
        if self.available < spec.quantity:
            return False
        # Check constraints...
        return True

    def estimated_cost(self, spec: ResourceSpec) -> float:
        """Estimate total cost for the spec"""
        return spec.quantity * self.price_per_unit
```

**Example**:
```python
# Peer A advertises:
offer = ResourceOffer(
    resource_type=R['GPU_MEMORY'],
    available=24.0,  # GB free
    total=32.0,
    price_per_unit=0.01,  # $0.01/GB
    peer_id='peer-a-123',
    current_load=0.3,
    energy_state='hibernate',  # Will need wake cost
    features=['fp16', 'tensor_cores', 'nvml'],
    locality_hints={'region': 'us-west', 'latency_ms': 15}
)
```

### 4. Resource Lease (ResourceLease)

A granted allocation, analogous to a file descriptor or process handle:

```python
@dataclass
class ResourceLease:
    """
    A granted resource allocation with lifecycle.
    Analogous to: file descriptor (has ref count, can be released)
    """
    lease_id: str
    spec: ResourceSpec
    offer: ResourceOffer
    provider_id: str
    state: str = 'active'  # active | suspended | expired | released

    # Lifecycle
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    last_used: float = field(default_factory=time.time)

    # Accounting
    energy_consumed_wh: float = 0.0
    energy_cost_credits: float = 0.0
    data_transfer_gb: float = 0.0
    total_cost_credits: float = 0.0

    # Metadata
    parent_lease_id: Optional[str] = None  # For hierarchical leases
    tags: Dict[str, str] = field(default_factory=dict)  # e.g., {'model': 'llama-70b'}

    async def renew(self, duration_s: Optional[float] = None):
        """Extend the lease (like kevent or fcntl)"""
        pass

    async def transfer(self, new_provider_id: str):
        """Migrate lease to another provider (like process migration)"""
        pass

    async def snapshot(self) -> bytes:
        """Save lease state for checkpoint/restore"""
        pass

    async def release(self):
        """Release the resource (like close())"""
        pass

    def touch(self):
        """Mark as recently used (for LRU eviction)"""
        self.last_used = time.time()
```

---

## OS-like Resource Manager

### IResourceManager Interface

The **kernel** of EvoOS - analogous to VFS, scheduler, memory manager:

```python
class IResourceManager(ABC):
    """
    Core abstraction for ALL resource management.
    This is the "system call" interface to EvoOS.
    Analogous to: open(), read(), write(), fork(), exec(), mmap()
    """

    @abstractmethod
    async def allocate(self, spec: ResourceSpec) -> ResourceLease:
        """
        Request resources. Blocks until resources are available or fails.
        Analogous to: open() + fork() + mmap() combined
        """
        pass

    @abstractmethod
    async def release(self, lease_id: str):
        """
        Release a lease. Resources may be reclaimed or returned to pool.
        Analogous to: close()
        """
        pass

    @abstractmethod
    async def renew(self, lease_id: str, duration_s: Optional[float] = None):
        """
        Extend lease duration. Prevents expiration.
        Analogous to: kevent() to update timer
        """
        pass

    @abstractmethod
    async def query(self, resource_type: ResourceType,
                   constraints: Dict[str, Any]) -> List[ResourceOffer]:
        """
        Discover available resources without allocating.
        Analogous to: reading /proc, sysctl, df, etc.
        """
        pass

    @abstractmethod
    async def transfer(self, lease_id: str, target_peer_id: str):
        """
        Migrate a lease to another peer.
        Analogous to: process migration, live VM migration
        """
        pass

    @abstractmethod
    async def snapshot(self, lease_id: str) -> bytes:
        """
        Checkpoint lease state (for long-running jobs).
        Analogous to: fork() + exec() checkpoint, CRIU
        """
        pass

    @abstractmethod
    async def restore(self, snapshot: bytes, new_lease: bool) -> ResourceLease:
        """
        Restore from snapshot.
        Analogous to: restoring from checkpoint
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Global resource statistics.
        Analogous to: vmstat, iostat, top
        """
        pass

    @abstractmethod
    def list_active_leases(self) -> List[ResourceLease]:
        """
        List all active leases (like ps for processes).
        Analogous to: ps, lsof
        """
        pass

    @abstractmethod
    async def suppress(self, lease_id: str, duration_s: float):
        """
        Temporarily reduce resource usage (for energy saving).
        Analogous to: cgroups set throttle
        """
        pass
```

---

## Resource Provider Hierarchy (Device Drivers)

Each physical/virtual resource type has a provider (like a kernel driver). Providers are pluggable:

```
IResourceProvider (Abstract Base)
│
├── LocalResourceProvider
│   ├── CPUResourceProvider
│   │   ├── CPUCoreProvider
│   │   ├── CPUMemoryProvider
│   │   └── CPUEnergyProvider
│   │
│   ├── GPUResourceProvider
│   │   ├── GPUMemoryProvider
│   │   ├── GPUComputeProvider
│   │   └── GPUEnergyProvider
│   │
│   ├── StorageResourceProvider
│   │   ├── DiskIOPSProvider
│   │   ├── DiskSpaceProvider
│   │   └── NVMeControllerProvider
│   │
│   ├── NetworkResourceProvider
│   │   ├── BandwidthProvider
│   │   └── PortProvider
│   │
│   └── ModelResourceProvider
│       ├── LayerCacheProvider
│       └── KVCacheProvider
│
├── PeerResourceProvider (Remote)
│   ├── PeerLayerProvider (borrow model layers)
│   ├── PeerComputeProvider (borrow CPU/GPU)
│   └── PeerStorageProvider (borrow disk)
│
├── CloudResourceProvider
│   ├── AWSS3Provider
│   ├── GCPStorageProvider
│   └── AzureBlobProvider
│
└── SpecialResourceProvider
    ├── DisplayProvider (VNC/RDP)
    ├── AudioProvider
    ├── ProcessProvider (remote execution)
    └── LicenseProvider (software licenses)
```

### Example: CPUResourceProvider

```python
class CPUResourceProvider(IResourceProvider):
    """
    Manages CPU resources via cgroups, nice, affinity, frequency scaling.
    Like a CPU scheduler/driver.
    """

    def __init__(self, hardware_profiler: HardwareProfiler):
        self.hw = hardware_profiler
        self.cgroups = CGroupManager()  # Linux cgroups v2
        self.affinity = CPUAffinityManager()
        self.dvfs = DVFSManager()  # Dynamic voltage/frequency scaling
        self.available_cores = self.hw.detect_cores()
        self.core_allocation = {}  # lease_id → set of cores

    async def allocate(self, spec: ResourceSpec) -> ResourceLease:
        if spec.resource_type != R['CPU_CORE']:
            raise ResourceUnavailable("Not a CPU resource")

        # Check availability
        requested_cores = int(spec.quantity)
        free_cores = self._get_free_cores()
        if len(free_cores) < requested_cores:
            raise ResourceUnavailable(f"Need {requested_cores}, have {len(free_cores)}")

        # Allocate cores
        allocated = free_cores[:requested_cores]
        lease_id = generate_lease_id()
        self.core_allocation[lease_id] = allocated

        # Apply cgroup restrictions
        self.cgroups.create_lease(lease_id)
        self.cgroups.set_cpu_quota(lease_id, allocated)
        self.affinity.set_affinity(lease_id, allocated)

        # Energy policy: if lease has energy_budget, apply DVFS
        if spec.energy_budget_wh:
            self.dvfs.set_frequency_ ceiling(lease_id, 'powersave')

        lease = ResourceLease(
            lease_id=lease_id,
            spec=spec,
            offer=ResourceOffer(...),  # from query()
            provider_id='local-cpu',
            metadata={'allocated_cores': allocated}
        )
        return lease

    async def release(self, lease_id: str):
        if lease_id in self.core_allocation:
            cores = self.core_allocation[lease_id]
            self.cgroups.delete_lease(lease_id)
            self.affinity.clear_affinity(lease_id, cores)
            del self.core_allocation[lease_id]

    async def suppress(self, lease_id: str, duration_s: float):
        """Throttle CPU for lease (energy saving)"""
        # Reduce frequency, limit cgroup quota
        pass
```

### Example: GPUMemoryProvider (Energy-Aware)

```python
class GPUMemoryProvider(IResourceProvider):
    """
    Manages GPU memory via CUDA, NVML, and EvoLLM cache.
    Integrates with energy state management.
    """

    def __init__(self, gpu_id: int, energy_manager: Optional['ResourceStateManager']):
        self.gpu_id = gpu_id
        self.energy_manager = energy_manager
        self.nvml = NVMLClient()
        self.cache = GPUCache()  # From EvoLLM
        self.total_memory = self.nvml.get_memory_total(gpu_id)
        self.allocations = {}  # lease_id → {'size_gb': float, 'layers': []}

    async def allocate(self, spec: ResourceSpec) -> ResourceLease:
        # Check energy state - if sleeping, wake up first
        if self.energy_manager and self.energy_manager.state != ResourceState.FUNCTIONAL:
            await self.energy_manager.transition_to(ResourceState.FUNCTIONAL)

        # Check available memory
        free_mem = self.nvml.get_memory_free(self.gpu_id)
        if spec.quantity > free_mem / 1e9:  # Convert bytes to GB
            raise ResourceUnavailable("Insufficient GPU memory")

        lease_id = generate_lease_id()

        # Allocate (track in cache manager)
        success = self.cache.allocate_memory(spec.quantity * 1e9, lease_id)
        if not success:
            raise ResourceUnavailable("Cache allocation failed")

        self.allocations[lease_id] = {
            'size_gb': spec.quantity,
            'allocated_at': time.time()
        }

        lease = ResourceLease(
            lease_id=lease_id,
            spec=spec,
            offer=ResourceOffer(...),
            provider_id=f'gpu-{self.gpu_id}',
            metadata={'gpu_id': self.gpu_id}
        )
        return lease

    async def release(self, lease_id: str):
        if lease_id in self.allocations:
            alloc = self.allocations[lease_id]
            self.cache.free_memory(alloc['size_gb'] * 1e9, lease_id)
            del self.allocations[lease_id]
```

---

## HybridResourceManager: The " initProcess" of EvoOS

The top-level resource manager that composes multiple providers. This is analogous to the OS kernel's resource scheduler:

```python
class HybridResourceManager(IResourceManager):
    """
    Composes multiple resource providers into a unified system.
    Analogous to: Linux kernel combining CPU, memory, I/O schedulers
    """

    def __init__(self, config: 'EvoLLMConfig'):
        self.config = config
        self.providers: Dict[str, IResourceProvider] = {}
        self.local_providers: List[IResourceProvider] = []
        self.peer_provider: Optional[PeerResourceProvider] = None
        self.cloud_providers: List[CloudResourceProvider] = []

        # Energy integration
        self.energy_model = EnergyPriceModel()
        self.energy_policy = EnergyPolicy()
        if config.energy.enabled:
            self.energy_policy = load_energy_policy(config.energy.policy_file)

        # Registry for peer discovery
        self.registry = PeerRegistry()

        # Active leases
        self.leases: Dict[str, ResourceLease] = {}
        self.lease_lock = asyncio.Lock()

        # Monitoring
        self.stats = ResourceStats()

    async def initialize(self):
        """Start all providers (like booting OS)"""
        # Initialize local providers based on hardware
        hw_profile = HardwareProfiler.profile()

        if hw_profile.cpu_cores > 0:
            cpu_provider = CPUResourceProvider(hw_profile)
            await cpu_provider.initialize()
            self.providers['cpu'] = cpu_provider
            self.local_providers.append(cpu_provider)

        if hw_profile.gpu_count > 0:
            gpu_provider = GPUResourceProvider(hw_profile)
            await gpu_provider.initialize()
            self.providers['gpu'] = gpu_provider
            self.local_providers.append(gpu_provider)

        # Initialize peer provider if enabled
        if self.config.peer.enabled:
            self.peer_provider = PeerResourceProvider(
                registry=self.registry,
                config=self.config.peer
            )
            await self.peer_provider.initialize()

        # Start background tasks
        asyncio.create_task(self._lease_reaper())
        asyncio.create_task(self._energy_monitor())

    async def allocate(self, spec: ResourceSpec) -> ResourceLease:
        """
        Smart allocation across all providers.
        Strategy:
          1. Check local providers first (if policy prefers local)
          2. Check peers for borrowable resources
          3. Check cloud providers (if configured)
          4. Select cheapest/best offer based on price, latency, energy
          5. Allocate from chosen provider
          6. Track lease centrally
        """
        async with self.lease_lock:
            # Step 1: Query all providers for offers
            offers = await self._query_offers(spec)

            if not offers:
                raise ResourceUnavailable(f"No resources available for {spec.resource_type}")

            # Step 2: Score and rank offers
            scored_offers = []
            for offer in offers:
                score = await self._score_offer(offer, spec)
                scored_offers.append((score, offer))

            scored_offers.sort(key=lambda x: x[0])  # Lower score = better

            # Step 3: Select best offer
            best_score, best_offer = scored_offers[0]

            # Step 4: Allocate from provider
            provider = self.providers[best_offer.provider_id]
            lease = await provider.allocate(spec)
            lease.offer = best_offer  # Attach the offer we used

            # Step 5: Track lease centrally
            self.leases[lease.lease_id] = lease

            # Step 6: Log allocation
            self.stats.record_allocation(lease)

            return lease

    async def _query_offers(self, spec: ResourceSpec) -> List[ResourceOffer]:
        """Query all providers for offers matching spec"""
        tasks = []
        for provider in self.local_providers:
            tasks.append(provider.query(spec.resource_type, spec.constraints))

        if self.peer_provider:
            tasks.append(self.peer_provider.query(spec.resource_type, spec.constraints))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_offers = []
        for result in results:
            if isinstance(result, Exception):
                continue
            all_offers.extend(result)

        return all_offers

    async def _score_offer(self, offer: ResourceOffer, spec: ResourceSpec) -> float:
        """
        Compute a score for an offer (lower is better).
        Factors:
          - Price per unit (weight: 0.4)
          - Latency (weight: 0.3)
          - Energy state (wake penalty) (weight: 0.2)
          - Reliability (weight: 0.1)
        """
        score = 0.0

        # Price component (normalize by max expected price)
        price_score = offer.price_per_unit / 1.0  # Assume $1 max per unit
        score += price_score * 0.4

        # Latency component
        latency_ms = offer.avg_latency_ms
        if offer.energy_state == 'hibernate':
            latency_ms += 1000  # +1s wake penalty
        elif offer.energy_state == 'sleep':
            latency_ms += 5000  # +5s wake penalty
        latency_score = latency_ms / 10000  # Normalize to 10s max
        score += latency_score * 0.3

        # Energy cost component
        if spec.energy_budget_wh:
            # Estimate energy cost for lease duration
            duration = spec.duration_s or 3600
            estimated_energy_wh = offer.energy_consumption_rate_w * duration / 3600
            if estimated_energy_wh > spec.energy_budget_wh:
                score += 1000  # Very high penalty for exceeding budget

        # Reliability component (uptime score)
        reliability_score = (1.0 - offer.reliability_score) * 0.1
        score += reliability_score

        return score

    async def release(self, lease_id: str):
        async with self.lease_lock:
            if lease_id not in self.leases:
                raise LeaseNotFound(f"Lease {lease_id} not found")

            lease = self.leases[lease_id]

            # Release from provider
            provider = self.providers[lease.provider_id]
            await provider.release(lease_id)

            # Clean up
            del self.leases[lease_id]
            self.stats.record_release(lease)

    async def list_active_leases(self) -> List[ResourceLease]:
        return list(self.leases.values())

    def get_stats(self) -> Dict[str, Any]:
        return self.stats.to_dict()

    async def _lease_reaper(self):
        """Background task: expire old leases"""
        while True:
            await asyncio.sleep(10)
            now = time.time()
            expired = []
            async with self.lease_lock:
                for lease_id, lease in self.leases.items():
                    if lease.expires_at and lease.expires_at < now:
                        expired.append(lease_id)

            for lease_id in expired:
                try:
                    await self.release(lease_id)
                except Exception as e:
                    logger.error(f"Failed to reaper lease {lease_id}: {e}")

    async def _energy_monitor(self):
        """Background task: update energy prices, check policies"""
        while True:
            await asyncio.sleep(60)  # Every minute
            # Update spot prices
            self.energy_model.update_prices()
```

---

## EvoOS System Call Interface (CLI / API)

Unified command-line tool: `evosh` (EvoOS Shell)

```bash
# Resource queries (like cat /proc/meminfo)
evosh resources list
evosh resources list --type gpu.memory
evosh resources list --peer <peer_id>

# Request resources (like fork + mmap)
evosh resources allocate \
  --type gpu.memory \
  --quantity 8GB \
  --duration 1h \
  --max-price 0.50 \
  --energy-budget 100Wh

# Monitor (like top, vmstat)
evosh resources watch
evosh energy watch
evosh ledger balance

# Lease management (like ps, kill)
evosh leases list
evosh leases release <lease_id>
evosh leases renew <lease_id> --duration 30m

# Peer management (like ifconfig, route)
evosh peers list
evosh peers advertise --gpu 24GB --price 0.02/GB
evosh peers discover
evosh peers trust <peer_id>  # Like adding SSH key

# Energy policy (like sysctl)
evosh energy policy show
evosh energy policy set --auto-hibernate 300
evosh energy state  # Show current power states

# Shell (like xterm but remote)
evosh shell allocate --peer <peer_id>  # Get remote shell
evosh display allocate --peer <peer_id>  # Remote desktop

# File operations (like ls, cp but across peers)
evosh files cp model.layers.0 /local/path --from-peer <peer_id>
```

---

## Integration with EvoLLM

EvoLLM becomes a **user-space service** that uses EvoOS resource APIs:

```python
# Before: EvoLLM directly manages cache
class EvoLLMModel:
    def __init__(self, config):
        self.cache_manager = create_cache(config)  # Tightly coupled

# After: EvoLLM uses EvoOS resource manager
class EvoLLMModel:
    def __init__(self, config):
        self.resource_manager = HybridResourceManager(config)
        # Request resources at startup
        asyncio.create_task(self._request_resources())

    async def _request_resources(self):
        """Request necessary resources from EvoOS"""
        # Request GPU memory for layers
        gpu_spec = ResourceSpec(
            resource_type=R['GPU_MEMORY'],
            quantity=config.gpu_layers * 2.0,  # Estimate 2GB per layer
            constraints={
                'required_features': ['fp16'],
                'min_performance': 200.0  # GB/s
            },
            energy_budget_wh=config.max_energy_wh
        )
        self.gpu_lease = await self.resource_manager.allocate(gpu_spec)

        # Request CPU RAM for cache
        ram_spec = ResourceSpec(
            resource_type=R['RAM'],
            quantity=config.cpu_cache_gb,
            constraints={'isolation_level': 'shared'}
        )
        self.ram_lease = await self.resource_manager.allocate(ram_spec)

    async def forward(self, x):
        # Use allocated resources via leases
        gpu_mem = self.gpu_lease.metadata['allocation_handle']
        # ... inference using allocated GPU memory
        pass
```

---

## System Architecture: Layered View

```
┌─────────────────────────────────────────────────────────────┐
│                    User Space                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐         │
│  │   EvoLLM   │  │   Trainer   │  │   Inference │         │
│  │   Agent    │  │   Service   │  │   Service   │         │
│  └────────────┘  └────────────┘  └────────────┘         │
│         │              │              │                   │
│         └──────────────┼──────────────┘                   │
│                        │                                   │
│              ┌─────────▼─────────┐                        │
│              │  evosh CLI / API  │  (System calls)       │
│              └─────────┬─────────┘                        │
│                        │                                   │
└────────────────────────┼───────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────┐
│                 EvoOS Kernel                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │        HybridResourceManager (Global Scheduler)    │  │
│  │  - Queries all providers                            │  │
│  │  - Scores offers                                    │  │
│  │  - Allocates leases                                 │  │
│  │  - Tracks global state                              │  │
│  └───────────────────┬─────────────────────────────────┘  │
│                      │                                    │
│  ┌───────────────────┼─────────────────────────────────┐  │
│  │                   │                                 │  │
│  ▼                   ▼                                 ▼  │
│ ┌────────────┐  ┌────────────┐                ┌────────────┐│
│ │CPU Provider│  │GPU Provider│                │Peer Provider││
│ │(cgroups)   │  │(NVML+CUDA) │                │(gRPC)      ││
│ └────────────┘  └────────────┘                └────────────┘│
│                                                             │
│  ┌────────────┐  ┌────────────┐                ┌────────────┐│
│  │Storage     │  │Network     │                │Energy      ││
│  │Provider    │  │Provider    │                │Manager     ││
│  └────────────┘  └────────────┘                └────────────┘│
└─────────────────────────────────────────────────────────────┘
                         │
┌────────────────────────▼───────────────────────────────────┐
│                 Hardware / Network                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐        │
│  │   CPU      │  │   GPU      │  │   Network  │        │
│  │  (8 cores) │  │ (RTX 4090) │  │  (10 GbE)  │        │
│  └────────────┘  └────────────┘  └────────────┘        │
│  ┌────────────┐  ┌────────────┐                         │
│  │   RAM      │  │   Disk     │                         │
│  │  (64 GB)   │  │  (2 TB)    │                         │
│  └────────────┘  └────────────┘                         │
└───────────────────────────────────────────────────────────┘

                          │
                    ┌─────┴─────┐
                    ▼           ▼
          ┌─────────────────────────────┐
          │   Other Peers in Network    │
          │  (EvoOS on other machines) │
          └─────────────────────────────┘
```

---

## Resource State Machine with Energy

Each resource provider maintains energy state, integrated with allocation:

```python
class EnergyAwareProvider(IResourceProvider):
    """
    Mixin for providers that support power states.
    Like ACPI driver in OS.
    """

    def __init__(self, profile: ResourceEnergyProfile, policy: EnergyPolicy):
        self.energy_profile = profile
        self.energy_policy = policy
        self.state = ResourceState.FUNCTIONAL
        self.state_since = time.time()
        self.lease_energy_usage: Dict[str, float] = {}  # lease_id → Wh

    async def allocate(self, spec: ResourceSpec) -> ResourceLease:
        # Check if need to wake
        if self.state != ResourceState.FUNCTIONAL:
            await self._ensure_functional()

        # Perform allocation (subclass implements)
        lease = await self._do_allocate(spec)

        # Start energy accounting for lease
        self.lease_energy_usage[lease.lease_id] = 0.0
        asyncio.create_task(self._track_lease_energy(lease.lease_id))

        return lease

    async def _ensure_functional(self):
        """Wake resource if needed"""
        current_state = self.state
        if current_state == ResourceState.SLEEP:
            await self._transition_to(ResourceState.HIBERNATE)  # Sleep→Hibernate first
            await self._transition_to(ResourceState.FUNCTIONAL)
        elif current_state == ResourceState.HIBERNATE:
            await self._transition_to(ResourceState.FUNCTIONAL)

    async def _transition_to(self, new_state: ResourceState):
        """Execute state transition"""
        old_state = self.state
        if new_state == old_state:
            return

        # Record wake energy cost
        if new_state == ResourceState.FUNCTIONAL:
            if old_state == ResourceState.HIBERNATE:
                wake_energy_j = self.energy_profile.e_wake_from_hibernate_j
            elif old_state == ResourceState.SLEEP:
                wake_energy_j = self.energy_profile.e_wake_from_sleep_j
            # Allocate wake cost to active leases proportionally
            await self._allocate_wake_cost(wake_energy_j)

        self.state = new_state
        self.state_since = time.time()
        logger.info(f"Resource {self.energy_profile.resource_id} "
                   f"transitioned {old_state.value}→{new_state.value}")

    async def _track_lease_energy(self, lease_id: str):
        """Background task: sample power and accumulate"""
        while lease_id in self.lease_energy_usage:
            power_w = self._read_power()  # From NVML/RAPL
            duration_s = 1.0  # Sample interval
            energy_wh = (power_w * duration_s) / 3600
            self.lease_energy_usage[lease_id] += energy_wh

            # Check lease energy budget
            lease = self.leases.get(lease_id)
            if lease and lease.spec.energy_budget_wh:
                if self.lease_energy_usage[lease_id] > lease.spec.energy_budget_wh:
                    logger.warning(f"Lease {lease_id} exceeded energy budget")
                    # Could throttle or terminate

            await asyncio.sleep(1.0)

    def get_energy_report(self) -> Dict:
        """Report energy usage by lease and resource"""
        return {
            'resource_id': self.energy_profile.resource_id,
            'state': self.state.value,
            'time_in_state_s': time.time() - self.state_since,
            'leases': self.lease_energy_usage.copy(),
            'total_energy_wh': sum(self.lease_energy_usage.values()),
        }
```

---

## Peer-to-Peer as a Cluster File System

The peer network is like a **distributed file system** but for all resources:

```
/dev/evos/
├── peers/
│   ├── peer-a-123/
│   │   ├── compute/cpu/core/ (4 available)
│   │   ├── memory/gpu/ (16GB free)
│   │   ├── storage/disk/ (1TB free)
│   │   └── model/layers/ (has Llama-70B layers 0-30)
│   ├── peer-b-456/
│   │   ├── compute/cpu/core/ (8 available)
│   │   ├── memory/gpu/ (0GB free - SLEEP)
│   │   └── energy/state: sleep
│   └── peer-c-789/
│       ├── compute/gpu/sm/ (80 SMs available)
│       ├── energy/price: $0.12/kWh
│       └── locality: us-west-2
│
├── local/
│   ├── compute/cpu/core/ (8 total, 2 free)
│   ├── memory/ram/ (32GB free)
│   └── storage/nvme/ (500GB free)
│
└── system/
    ├── energy/price: $0.15/kWh
    ├── ledger/balance: 123.45 credits
    └── leases/  # Currently active leases
        ├── lease-abc123
        └── lease-def456
```

You `open()` a resource by calling `allocate()`, you `read()` by using it, and you `close()` by calling `release()`.

---

## Compatibility with Global Plan

This unified abstraction **integrates seamlessly** with the energy-aware plan:

### 1. ResourceState is Built-In

Every `IResourceProvider` can optionally implement `EnergyAwareProvider` mixin:
- Local GPU provider: transitions between FUNCTIONAL/HIBERNATE/SLEEP
- Peer provider: advertises remote peer's energy state via `ResourceOffer.energy_state`
- Cloud provider: integrates with cloud energy APIs (AWS Spot, price spikes)

### 2. Ledger Extends to All Resource Types

Current plan: Ledger tracks energy credits for GPU time.

Extended: Ledger tracks all resource types:
```python
class ResourceLedger:
    def debit(resource_type: ResourceType, amount: float, reason: str) -> bool
    def credit(resource_type: ResourceType, peer_id: str, amount: float, reason: str)
    def get_balance(resource_type: ResourceType, peer_id: str) -> float
```

Example entries:
```
2024-01-15 10:30:00 DEBIT  peer-b  compute.gpu.core    1.0hr  $0.50  inference
2024-01-15 10:30:00 CREDIT peer-a  compute.gpu.core    1.0hr  $0.45  lending
2024-01-15 10:31:00 DEBIT  peer-b  energy.kwh         50.0Wh  $0.01  gpu_wake
```

### 3. Registry Tracks All Offer Types

`PeerRegistry` currently tracks layers and compute offers.

Extended: Track all `ResourceOffer`s with resource_type field:
```python
class PeerRegistry:
    def advertise(self, peer_id: str, offers: List[ResourceOffer]):
        # Store by (peer_id, resource_type)
        pass

    def query(self, resource_type: ResourceType, min_qty: float) -> List[ResourceOffer]:
        # Filter by type
        pass
```

### 4. Energy Policy Applies to All Resources

`EnergyPolicy.can_wake_for_bid()` currently checks GPU bids.

Extended: Generalize to any resource type:
```python
def can_allocate(self, spec: ResourceSpec, bid_price: float) -> Tuple[bool, str]:
    # Check min bid, daily limits, energy caps
    # Resource-specific rules from policy.resource_overrides[spec.resource_type.category]
    pass
```

---

## Implementation Phases (Reconciled with Existing Plan)

### Phase 1: Core Abstraction (Parallel with P2P)

While implementing PeerServer/Client (PLAN Phase 2), also define:

1. `evollm/resource/datatypes.py`:
   - `ResourceType`, `ResourceSpec`, `ResourceOffer`, `ResourceLease`

2. `evollm/resource/manager.py`:
   - `IResourceManager` ABC
   - `HybridResourceManager` (composition of providers)

3. `evollm/resource/providers/local/`:
   - `base.py` - Local provider base
   - `cpu.py` - CPU core allocator (cgroups)
   - `gpu.py` - GPU memory allocator (CUDA)
   - `memory.py` - RAM allocator (malloc/mmap)

4. `evollm/resource/providers/peer/`:
   - `peer_provider.py` - Wraps PeerClient to fetch remote resources

These create the foundation for both EvoLLM caching and energy management.

### Phase 2: EvoLLM Integration

Refactor existing EvoLLM code to use new abstraction:

1. `evollm/evollm_base.py`:
   ```python
   # Old:
   self.cache_manager = TensorCacheManager(...)

   # New:
   self.resource_manager = HybridResourceManager(config)
   self.gpu_lease = await self.resource_manager.allocate(gpu_spec)
   self.ram_lease = await self.resource_manager.allocate(ram_spec)
   ```

2. `evollm/cache_policy.py` becomes `resource/providers/local/cache_provider.py`

3. `evollm/tensor_loader.py` becomes `resource/loader.py` (uses leases)

### Phase 3: Energy-Aware Resources

Extend providers with energy:

1. `evollm/resource/energy.py` (already planned)
   - `ResourceStateManager`
   - `EnergyAwareProvider` mixin
   - Power measurement (NVML, RAPL)

2. Annotate `ResourceSpec` with energy fields:
   ```python
   @dataclass
   class ResourceSpec:
       ...
       energy_budget_wh: Optional[float] = None
       prefer_low_carbon: bool = False
   ```

3. `HybridResourceManager` considers energy in scoring:
   - Prefer providers with lower carbon intensity
   - Respect energy budget constraints
   - Account wake energy costs

### Phase 4: Distributed OS Features

Add advanced OS features:

1. **Resource namespaces** (like Linux namespaces):
   ```python
   namespace = ResourceNamespace(
       name='inference-job-123',
       resources=[lease1, lease2],
       quota={'gpu.memory': 16, 'cpu.core': 4}
   )
   ```

2. **Quality of Service (QoS)**:
   ```python
   lease = await manager.allocate_with_qos(
       spec=spec,
       min_guarantee=True,  # Reserve capacity
       priority=10,  # Higher priority gets preempted?
   )
   ```

3. **Live migration**:
   ```python
   await lease.transfer(target_peer='peer-b-456')
   ```

4. **Checkpoint/Restore**:
   ```python
   snapshot = await lease.snapshot()
   # Later...
   restored_lease = await manager.restore(snapshot)
   ```

5. **Resource accounting & limits** (like ulimit):
   ```python
   manager.set_user_limit(user='alice', max_gpu_hours=10.0/week)
   ```

---

## Example: Complete Workflow

```python
# 1. Start EvoOS
config = EvoLLMConfig(peer={'enabled': True, 'energy_aware': True})
manager = HybridResourceManager(config)
await manager.initialize()

# 2. Peers advertise (background)
# Peer A: has GPU with 16GB free, state=HIBERNATE, price=$0.02/GB
# Peer B: has CPU with 4 cores free, state=FUNCTIONAL, price=$0.01/core

# 3. Running inference job needs resources
model_spec = ResourceSpec(
    resource_type=R['MODEL_LAYER'],
    quantity=30.0,  # Need 30 layers cached
    duration_s=3600,
    energy_budget_wh=50.0
)

lease = await manager.allocate(model_spec)
# Manager:
#   - Queries: local GPU cache provider? No space
#   - Queries: Peer A has layers in cache, HIBERNATE state
#   - Score: price=0.02 + wake_premium(1s)=0.005 = 0.025/GB
#   - Allocates: sends WakeRequest to Peer A
#   - Peer A: wakes GPU, transfers 30 layers (≈60GB)
#   - Manager: leases 60GB from Peer A for 1 hour
#   - Energy: Levy includes wake cost + transfer energy

# 4. During lease, energy monitored
#   - Peer A's GPU is FUNCTIONAL, consuming 250W
#   - Energy tracked: 250W * 1h = 250Wh = $0.0375 (at $0.15/kWh)
#   - Wake cost amortized: 1000J = 0.0003kWh = $0.000045

# 5. Lease expires or early release
await manager.release(lease.lease_id)
#   - Peer A's GPU returns to HIBERNATE after 5min idle (per policy)
#   - Ledger: Consumer debited $0.025*60 = $1.50, Provider credited $1.35 (10% fee)
```

---

## Key Insights: Why This is an OS

1. **Virtualization**: Resources are virtualized (leases), not directly allocated
2. **System Calls**: `allocate()`, `release()`, `query()` are like syscalls
3. **Kernel**: `HybridResourceManager` is the kernel - global authority
4. **Drivers**: `IResourceProvider` implementations are drivers
5. **Process**: `ResourceLease` is like a process descriptor (has ID, resources, lifetime)
6. **Address Space**: Resources have handles (lease_id) instead of physical addresses
7. **Scheduling**: `_score_offer()` is the scheduler picking best provider
8. **File Descriptor**: `lease_id` is like FD - you use it to access resource
9. **Namespace**: Each application can have its own set of leases isolated
10. **Accounting**: Ledger tracks resource usage (like process accounting in Unix)

---

## Next Steps

1. **Implement core datatypes** (`ResourceType`, `ResourceSpec`, `ResourceOffer`, `ResourceLease`)
2. **Create test mocks** for unit testing
3. **Build HybridResourceManager skeleton** with scoring logic
4. **Integrate with EvoLLM**: Refactor cache_manager to use new abstraction
5. **Add energy layer**: `EnergyAwareProvider` mixin, state transitions
6. **Build CLI**: `evosh` frontend to resource manager
7. **Extend peer protocol**: Add resource query/advertise beyond layers
8. **Implement ledger extension**: Track all resource types, not just energy

---

**See Also**:
- [PLAN.md](../PLAN.md) - Overall project plan
- [energy_aware_resource_management.md](energy_aware_resource_management.md) - Energy management details
- [generalized_resource_sharing.md](../docs/generalized_resource_sharing.md) - Generalized resources
