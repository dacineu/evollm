# Generalized Resource Forwarding in EvoOS

**Extends**: Peer-to-peer layer sharing → General compute resource sharing
**Date**: 2026-03-17
**Related**: `peer_analytics.md`, `.claude/plans/hashed-brewing-candle.md`

---

## Vision: "Everything is a Resource"

The core insight: Just as we abstract model layers as `ResourceProvider`, we can abstract **any compute resource**:

- **GPU memory** (VRAM allocation)
- **CPU cores** (compute slices)
- **RAM** (memory leases)
- **Disk space** (storage buckets)
- **Network bandwidth** (QoS guarantees)
- **Interactive sessions** (SSH/terminal, Jupyter notebooks)
- **Specialized hardware** (TPUs, NPUs, FPGAs)
- **Software licenses** (proprietary model access)

Each becomes a pluggable `ResourceProvider` with standard lifecycle:
```
allocate(requirements) → ResourceLease (handle, endpoint, credentials)
release(lease_id) → free resources
get_available() → List[ResourceOffer]
```

**Same pattern** as layer fetching:
```python
# Layer sharing (Phase 2)
layer = provider.get_layer("layer.20", ...)

# Generalized (Phase 3+)
lease = provider.allocate(ResourceType.GPU, {
    'memory_gb': 4.0,
    'compute_flops': 1e15,
    'duration_s': 3600
})
# lease.endpoint = "grpc://10.0.0.5:50052"
# Use remote GPU via the endpoint
```

---

## Architectural Extension: Resource Types

### Resource Taxonomy

```
Resource
├── StorageResource
│   ├── ModelLayer (current Phase 2)
│   ├── Checkpoint (full model snapshots)
│   ├── DatasetShard (training data)
│   └── CacheBlob (key-value store)
│
├── ComputeResource
│   ├── GPU (VRAM + compute cores)
│   ├── CPU (cores + cache)
│   ├── NPU (neural processing unit)
│   ├── TPU (tensor processing unit)
│   └── FPGA (reconfigurable logic)
│
├── MemoryResource
│   ├── RAM (volatile memory)
│   ├── PersistentMemory (NVDIMM)
│   └── SwapSpace (disk-backed)
│
├── NetworkResource
│   ├── BandwidthSlice (guaranteed throughput)
│   ├── LatencyGuarantee (max RTT)
│   └── PortForward (TCP/UDP tunnel)
│
├── InteractiveResource
│   ├── ShellSession (bash, zsh)
│   ├── JupyterKernel (Python/R/Julia)
│   ├── RemoteDesktop (VNC, RDP)
│   └── APIServer (REST/GraphQL endpoint)
│
└── SoftwareLicense
    ├── ModelAccess (closed-weight models)
    ├── CommercialTool (proprietary software)
    └── APIToken (rate-limited access)
```

Each resource type defines:
- **Allocation semantics** (exclusive vs shared)
- **Billing unit** (per-second, per-GB, per-request)
- **QoS parameters** (throughput, latency, priority)
- **Lifecycle hooks** (setup, teardown, suspend, resume)

---

## Core Abstraction: ResourceProvider (Generalized)

### Existing: LayerProvider (Phase 2)

```python
class ResourceProvider(ABC):
    @abstractmethod
    def get_layer(self, layer_name: str, layer_idx: int, load_fn, move_fn) -> Tuple[Dict, str]:
        """Fetch model layer weights"""
        pass

    @abstractmethod
    def cache_layer(self, layer_name: str, state_dict: Dict) -> bool:
        """Offer layer to cache for others"""
        pass
```

### Extended: GeneralResourceProvider (Phase 3+)

```python
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List

class ResourceType(Enum):
    GPU = "gpu"
    CPU = "cpu"
    RAM = "ram"
    DISK = "disk"
    SHELL = "shell"
    JUPYTER = "jupyter"
    MODEL_LAYER = "model_layer"
    BANDWIDTH = "bandwidth"

class AllocationMode(Enum):
    EXCLUSIVE = "exclusive"  # Only one consumer
    SHARED = "shared"        # Multiple consumers (read-only)
    BURSTABLE = "burstable"  # Shared with burst credits

@dataclass
class ResourceSpec:
    """What the requester needs"""
    resource_type: ResourceType
    quantity: float  # GB, cores, MB/s, etc.
    constraints: Dict[str, Any]  # e.g., {'min_compute_capability': '7.5'}
    duration_s: Optional[float] = None  # Lease duration
    mode: AllocationMode = AllocationMode.EXCLUSIVE

@dataclass
class ResourceOffer:
    """What a provider can offer"""
    provider_id: str
    resource_type: ResourceType
    available: float  # GB, cores, etc.
    price_per_unit: float  # credits/GB-hour, etc.
    capabilities: Dict[str, Any]  # GPU model, CPU arch, etc.
    load_score: float  # 0-1, provider's current utilization
    quality: float  # 0-1, reliability, reputation-weighted

@dataclass
class ResourceLease:
    """Granted allocation (implies permission)"""
    lease_id: str  # UUID
    provider_id: str
    resource_type: ResourceType
    quantity: float
    endpoint: str  # Connection info: "grpc://...", "ssh://...", "local"
    credentials: Optional[Dict]  # Auth tokens, certs
    expires_at: float  # Unix timestamp
    metadata: Dict[str, Any]  # Usage limits, QoS guarantees

class GeneralResourceProvider(ABC):
    """Extended abstraction for any resource"""

    @abstractmethod
    def advertise_capabilities(self) -> List[ResourceOffer]:
        """Broadcast what resources I can provide"""
        pass

    @abstractmethod
    def allocate(self, spec: ResourceSpec) -> ResourceLease:
        """
        Request resource allocation.
        Returns lease with access endpoint.
        Raises ResourceUnavailable if can't satisfy.
        """
        pass

    @abstractmethod
    def release(self, lease_id: str):
        """Free allocated resource (early termination)"""
        pass

    @abstractmethod
    def get_available(self, resource_type: ResourceType) -> List[ResourceOffer]:
        """List currently available resources of this type"""
        pass

    @abstractmethod
    def get_utilization(self) -> Dict[ResourceType, float]:
        """Current utilization per resource type (0-1)"""
        pass
```

**Key difference**: Instead of `get_layer()` (pull model), we have `allocate()` → `lease` with an **endpoint** where the resource can be accessed. This enables:
- Remote GPU compute: lease gives gRPC endpoint to execute kernels
- Remote shell: lease gives SSH host/port + credentials
- Remote storage: lease gives NFS mount or S3 bucket credentials

---

## Example: Forwarding an XTerm Session

### Scenario
User Alice on **Peer A** wants an interactive shell on **Peer B** (which has better resources or specific software).

### Current (Classic SSH)
```bash
# Alice on Peer A:
ssh alice@peer-b  # Direct connection, Peer B must be publicly reachable
```

Issues:
- Peer B needs public IP or port forwarding
- Alice needs account on Peer B
- No accounting/quotas
- No resource reservation ( Peer B may be overloaded)

### EvoOS-Enabled (Permission + Resource Control)

```bash
# Alice on Peer A:
evosh --peer peer-b --resource shell --duration 2h --priority high
```

**Under the hood**:

1. **Query registry**:
   ```
   A → Registry: "Who offers shell sessions?"
   Registry → A: [
     PeerInfo(id='B', shell_price=0.1, reputation=0.95, load=0.3),
     PeerInfo(id='C', shell_price=0.0, reputation=0.8, load=0.7)
   ]
   ```

2. **Request allocation from Peer B**:
   ```
   A → B (gRPC): Allocate(ResourceSpec(
       resource_type=SHELL,
       quantity=1,  # 1 shell session
       duration_s=7200,
       mode=EXCLUSIVE
   ))
   ```

3. **Peer B's policy evaluation**:
   ```python
   def can_allocate(self, spec: ResourceSpec, requester: str) -> Optional[ResourceLease]:
       # 1. Shell service enabled?
       if not self.config.shell_enabled: return None

       # 2. Requester allowed?
       if requester not in self.allowed_users: return None

       # 3. Capacity available?
       if self.active_shells >= self.max_shells: return None

       # 4. Balance/price check?
       balance = ledger.get_balance(requester)
       if balance < -self.max_debt: return None

       # 5. Yes! Create lease
       lease_id = uuid4()
       port = self._allocate_port()
       creds = self._create_credentials(requester, lease_id)

       lease = ResourceLease(
           lease_id=lease_id,
           provider_id=self.peer_id,
           resource_type=ResourceType.SHELL,
           quantity=1,
           endpoint=f"ssh://{self.address}:{port}",
           credentials={'username': 'evollm', 'password': creds},
           expires_at=time.time() + spec.duration_s
       )

       # Track lease
       self.leases[lease_id] = lease
       self.active_shells += 1

       # Bookkeeping
       ledger.debit(requester, self.shell_price * 3600, "shell lease")

       return lease
   ```

4. **Peer A receives lease**:
   ```json
   {
     "lease_id": "abc123...",
     "endpoint": "ssh://10.0.0.2:2222",
     "credentials": {
       "username": "evollm",
       "password": "temporary-token-xyz"
     },
     "expires_at": 1710748800
   }
   ```

5. **Alice connects automatically**:
   ```bash
   # EvoOS client on Peer A:
   evosh exec --lease /tmp/lease.json
   # This runs: ssh -p 2222 evollm@10.0.0.2
   # Alice gets shell on Peer B
   ```

6. **Control relationship**:
   - Alice has **exclusive access** to that shell for 2 hours
   - She can **take back control** anytime: `evosh release --lease abc123`
   - Peer B can **revoke** early if abused: `evosh revoke --lease abc123 --reason "spam"`
   - Both sides see lease in their `active_leases` list

---

## Example: Compute Offload (GPU Task Forwarding)

### Scenario
Peer A has inference task but limited GPU. Offload compute to Peer B's powerful GPU.

```python
# User code on Peer A:
from evollm import EvoLLM, ResourceSpec, ResourceType

model = EvoLLM.from_pretrained("llama-2-70b", peer_config={
    'mode': 'hybrid',
    'compute_offload': True  # Enable remote compute
})

# Normally: model(input_ids) runs locally
# With compute_offload: some layers execute on remote peers

output = model(input_ids)  # System automatically:
# 1. Identifies layers that would benefit from remote GPU
# 2. Requests compute leases from peers
# 3. Sends layer weights + activations to remote
# 4. Remote peer executes, returns results
# 5. Local continues forward pass
```

**Protocol**:

1. **Compute Offer Advertisement**:
   ```
   Peer B advertises:
   ResourceOffer(
       provider_id='B',
       resource_type=ResourceType.GPU,
       available=16.0,  # GB of VRAM
       price_per_unit=0.05,  # per GB-hour
       capabilities={
           'model': 'NVIDIA A100',
           'compute_cap': '8.0',
           'tensor_cores': True,
           'memory_bandwidth_gb_s': 2000
       },
       load_score=0.2,
       quality=0.98
   )
   ```

2. **Allocation Request**:
   ```
   A → B: Allocate(ResourceSpec(
       resource_type=GPU,
       quantity=2.5,  # 2.5GB for this layer
       constraints={'min_compute_capability': '7.0'},
       duration_s=300  # 5 minutes
   ))
   ```

3. **Lease**:
   ```json
   {
     "lease_id": "compute-xyz",
     "endpoint": "grpc://10.0.0.2:50062/compute",
     "credentials": {"auth_token": "Bearer ..."},
     "resource_type": "gpu",
     "quantity": 2.5
   }
   ```

4. **Usage**:
   ```python
   # EvoLLM layer executor sees lease endpoint
   # Connects via gRPC, sends:
   ExecuteRequest(
       operation="matmul",
       inputs=[tensor1, tensor2],
       dtype="float16",
       stream_output=True
   )
   # Receives ExecuteResponse with result tensor
   ```

5. **Accounting**:
   ```
   Time used: 5 minutes
   Quantity: 2.5 GB
   Price: $0.05/GB-hour
   Cost: 2.5 * 0.05 * (5/60) = $0.0104 credits
   → ledger.debit(A, 0.0104)
   → ledger.credit(B, 0.0104)
   ```

---

## Unified Coordination Layer

### Registry Schema Extension

**Current** (Phase 2):
```python
# layer_name → [(peer_id, checksum)]
registry.layer_to_peers = {
    "layer.20": [("A", "checksum_abc"), ("D", "checksum_def")],
}
```

**Extended** (Phase 3):
```python
# resource_type + filters → [offers]
registry.resource_index = {
    ('model_layer', None): [
        (peer_id='A', layer='layer.20', checksum='abc', price=0.1),
        (peer_id='D', layer='layer.20', checksum='def', price=0.0)
    ],
    ('gpu', {}): [
        (peer_id='B', vram_gb=24.0, price=0.05, capability='8.0', load=0.3),
        (peer_id='D', vram_gb=80.0, price=0.10, capability='8.0', load=0.1)
    ],
    ('shell', {}): [
        (peer_id='C', shell_price=0.0, max_sessions=10, active=3)
    ],
    ('cpu', {}): [
        (peer_id='A', cores=16, price=0.01, load=0.5),
        (peer_id='B', cores=32, price=0.02, load=0.2)
    ]
}
```

**Query API**:
```python
# Find GPUs with at least 4GB, capability >= 7.0
offers = registry.query(
    resource_type=ResourceType.GPU,
    min_quantity=4.0,
    min_capability='7.0',
    max_price=0.10,
    max_load=0.8
)

# Find shell sessions
offers = registry.query(
    resource_type=ResourceType.SHELL,
    max concurrent=5
)
```

---

## Permission & Access Control

### Multi-Level Permissions

```
┌─────────────────────────────────────────────────────┐
│               ACCESS CONTROL MATRIX                 │
├─────────────────────────────────────────────────────┤
│ Resource Type │ Read     │ Compute │ Admin │ Share │
├───────────────┼──────────┼─────────┼──────────────┤
│ Model Layer   │ ✅ View  │ ✅ Use  │ ❌     │ ❌  │
│ GPU           │ ❌       │ ✅ Run  │ ✅ Config│ ❌  │
│ Shell         │ ❌       │ ✅ Exec │ ✅ Kill  │ ❌  │
│ Disk          │ ✅ Read  │ ✅ R/W  │ ✅ Format│ ❌  │
│ Network       │ ❌       │ ✅ Send │ ✅ Filter│ ❌  │
│ License       │ ✅ View  │ ❌      │ ✅ Grant│ ✅  │
└───────────────┴──────────┴─────────┴──────────────┴────┘
```

**Permissions cascade**:
- **User** → has **Role** → grants **Permissions** on **Resource Types**
- Example roles:
  ```yaml
  - name: researcher
    permissions:
      - resource: model_layer
        actions: [read, compute]
      - resource: gpu
        actions: [compute]
      - resource: shell
        actions: []

  - name: cluster_admin
    permissions:
      - resource: "*"
        actions: ["*"]
  ```

**Policy enforcement points**:
1. **Registry query**: Can I see this resource offer?
2. **Allocation request**: Can I allocate this resource?
3. **Data transfer**: Can I read/write this data?
4. **Control operation**: Can I release/revoke this lease?

---

## Forwarding Control: "Take Back Control"

### Bidirectional Resource Streams

The architectural pattern enables **reverse control**:

```
Scenario: You allocate a shell on Peer B, but need to revoke it.

1. You (Peer A) have lease:
   lease_id = abc123
   endpoint = ssh://peer-b:2222

2. You decide to terminate early:
   evosh release --lease abc123

   A → B (gRPC): Release(lease_id="abc123")
   B validates:
     - Is lease_id valid? Yes
     - Is requester the lease holder? Yes (check lease.credentials)
     - Is lease still active? Yes
   B terminates shell:
     - Kill user's shell process
     - Free PTY resources
     - Remove from active_leases
   B responds: ReleaseResponse(success=True)
   A deletes lease from local state

3. Revocation by provider ( Peer B):
   B determines lease holder is abusive
   B sends: Revoke(lease_id, reason="policy violation")
   B terminates shell immediately
   B notifies lease holder via callback or push
```

### Control Channels

Each resource lease defines:
- **Primary channel**: Data path (GPU compute, shell I/O)
- **Control channel**: Management API (gRPC methods on the provider)
  - `Release()`, `Renew()`, `Transfer()`, `Suspend()`, `Resume()`

**Example: GPU Compute Lease with Control**:

```python
# User gets lease
lease = provider.allocate(ResourceSpec(
    resource_type=GPU,
    quantity=4.0,
    duration_s=3600
))

# Primary: send compute jobs
compute_channel = grpc.secure_channel(lease.endpoint, creds)
stub = ComputeServiceStub(compute_channel)
result = stub.Execute(ExecuteRequest(op="matmul", ...))

# Control: extend lease if needed
control_channel = grpc.secure_channel(lease.endpoint + "/control", creds)
control_stub = ControlServiceStub(control_channel)
control_stub.Renew(RenewRequest(lease_id=lease.lease_id, duration_s=1800))

# Control: transfer to another peer (handoff)
control_stub.Transfer(TransferRequest(
    lease_id=lease.lease_id,
    new_peer="10.0.0.3:50062"
))
```

**Transfer semantics**:
- Current peer migrates resource state to new peer
- Lease endpoint updates to new peer
- Seamless failover or scheduled migration

---

## Integrated Example: Distributed Inference Pipeline

### Full Workflow with Mixed Resource Types

```
Alice wants to run Llama-2-70B on 4GB GPU using peer cluster.

1. Model loading phase:
   ├─ EvoLLM queries registry: Who has model layers?
   ├─ 4 peers each advertise 15-20 layers (total 70)
   ├─ System ranks peers by reputation/latency/price
   ├─ Allocates leases for missing layers from optimal peers
   └─ Result: All 80 layers available across cluster

2. Inference phase (per token):
   ├─ Layer 0-2: Local GPU (reserved)
   │   └─ Execute locally
   ├─ Layer 3-10: Fetched from Peer A's RAM cache
   │   └─ Network fetch (5ms) → local GPU compute
   ├─ Layer 11-15: Offloaded to Peer B's GPU
   │   └─ Send layer weights + activations to Peer B
   │   └─ Peer B executes, returns results (10ms)
   │   └─ Local receives, continues
   ├─ Layer 16-25: Peer C's GPU (cheaper, slower)
   └─ Layer 26-79: Mix of local RAM and peer fetches

   Meanwhile:
   - Async prefetch: next layers requested in background
   - Load balancer: monitors peer latencies, adjusts ranking
   - Ledger: debits for GPU compute time, credits for serving

3. Resource accounting:
   ├─ Borrowed: 60 layers from peers (storage)
   ├─ Remote compute: 15 layers on Peer B's GPU (compute)
   ├─ Cost: 60 layer-hours + 15 GPU-minutes
   └─ Balance: Alice's account debited, peers credited

4. Control operations:
   ├─ Halfway through, Alice pauses: `evosh pause --job job123`
   │   → Sends pause to all remote compute peers
   │   → They checkpoint state, release GPU leases (but keep layers)
   ├─ Alice resumes: `evosh resume --job job123`
   │   → Reallocates GPUs (may be different peers)
   │   → Transfers checkpoint state
   └─ Alice cancels: `evosh cancel --job job123`
       → Releases all leases immediately

Result: Alice ran 70B model on 4GB GPU using cluster resources.
Cost: 1.2 credits.
Time: 1.5× slower than local 8xH100, but 10× cheaper.
```

---

## Implementation Path: From Layers to General Resources

### Phase 3 Roadmap (Extending P2P)

**Phase 3a: General Resource Abstraction** (Weeks 1-3)
- [ ] Rename `ResourceProvider.get_layer()` → `allocate_resource()`
- [ ] Keep `get_layer()` as specialized convenience wrapper
- [ ] Define `ResourceSpec`, `ResourceOffer`, `ResourceLease` dataclasses
- [ ] Implement `GeneralResourceProvider` base
- [ ] Extend `LocalResourceManager` to support:
  - `allocate(RAM)` → Local RAM lease (virtual allocation)
  - `allocate(CPU)` → CPU affinity + cgroup limits
  - `allocate(GPU)` → CUDA stream + memory pool
- [ ] Registry extends to store resource offers per peer

**Phase 3b: Compute Offloading** (Weeks 4-6)
- [ ] Define `ComputeService` gRPC API
  - `Execute(request)` → returns result
  - `StreamExecute(request)` → streaming results
  - `AllocCompute(spec)` → returns lease with endpoint
  - `ReleaseCompute(lease_id)`
- [ ] Implement `RemoteComputeResourceProvider`
  - Connects to peer's compute service
  - Serializes execution requests (PyTorch ops, TensorFlow graphs)
  - Deserializes results
  - Handles failures (retry, failover)
- [ ] Add to `EvoLLMModel`:
  - Decision: which layers to compute remotely?
  - Factors: layer size, peer GPU availability, network latency
  - Prefetch remote compute tasks (like prefetch layers)
- [ ] Tests: offload matmul to remote peer, verify correctness

**Phase 3c: Interactive Sessions** (Weeks 7-8)
- [ ] Define `ShellService` gRPC API
  - `CreateShell(spec)` → lease with SSH endpoint
  - `ExecuteCommand(lease_id, command)` → output
  - `InteractiveSession(lease_id)` → bidirectional stream
  - `Terminate(lease_id)`
- [ ] Implement `RemoteShellResourceProvider`
  - Provides PTY allocation on server
  - Proxy stdin/stdout/stderr over gRPC
  - Shell environment: restricted, resource-limited
- [ ] Command forwarding: `evosh exec "ls -la"` through peer
- [ ] Port forwarding: `evosh forward --local 8080 --peer remote:80`

**Phase 3d: Storage Leases** (Weeks 9-10)
- [ ] Define `StorageService` gRPC API
  - `MountBucket(lease_id)` → returns mount credentials (S3, NFS)
  - `Upload/download(path)` ✓ streaming
  - `List/Delete` operations
- [ ] Implement `RemoteStorageResourceProvider`
  - Peer provides temporary storage bucket
  - Quota enforced per lease
  - Auto-delete on lease expiry
- [ ] Use case: Upload large dataset to peer with fast disk for training

**Phase 3e: Resource Marketplaces** (Weeks 11-12)
- [ ] Dynamic pricing: prices adjust to supply/demand
- [ ] Spot market: auction excess capacity
- [ ] Reserved capacity: advance booking
- [ ] Resource bundling: "2 GPUs + 32GB RAM" package
- [ ] Quality tiers: bronze/silver/gold SLAs

---

## Security Model for Generalized Resources

### Permission Granularity

**Current (layers)**: Binary - can serve layer or not.
**Extended**: Fine-grained per-resource permissions.

```python
class AccessPolicy:
    def can_allocate(self, requester: str, spec: ResourceSpec) -> AllocationDecision:
        # 1. Is resource type allowed for this user?
        if not self.has_permission(requester, spec.resource_type, 'allocate'):
            return AllocationDecision(allow=False, reason='permission_denied')

        # 2. Is quantity within quota?
        used = self.get_current_usage(requester, spec.resource_type)
        quota = self.get_quota(requester, spec.resource_type)
        if used + spec.quantity > quota:
            return AllocationDecision(allow=False, reason='quota_exceeded')

        # 3. Can afford price?
        estimated_cost = self.estimate_cost(spec)
        balance = ledger.get_balance(requester)
        if balance < estimated_cost:
            return AllocationDecision(allow=False, reason='insufficient_credits')

        # 4. Resource available?
        available = self.get_available_quantity(spec.resource_type)
        if available < spec.quantity:
            return AllocationDecision(allow=False, reason='capacity_full')

        return AllocationDecision(allow=True)
```

### Certificate-Based Identity

Each peer/user has a certificate that encodes:
```
Subject: CN=alice@example.com
 issuer ✓
Subject Alternative Name:
  URI:urn:evollm:peer_id=peer-a-12345
  URI:urn:evollm:user_id=alice
  URI:urn:evollm:role=researcher
  IP:10.0.0.5
Valid from: 2026-03-17 to 2027-03-17
```

**mTLS handshake**:
1. Client presents cert
2. Server verifies CA signature
3. Server extracts `user_id` and `peer_id` from SAN
4. Server checks ACL: Can `user_id` allocate `GPU`?
5. If yes → allocation proceeds

---

## API Surface: Unified Client

```bash
# evosh - unified shell for EvoOS resource management

# Query available resources
evosh resources list --type gpu --min-quantity 4 --max-price 0.10

# Allocate resource
evosh resources allocate \
  --type gpu \
  --quantity 4.0 \
  --duration 2h \
  --output lease.json

# Use leased resource (implicit through evollm)
export EVOLLM_GPU_LEASES=lease.json
evollm infer --model llama-2-70b --input "Hello"

# Interactive shell
evosh shell allocate --peer peer-b --duration 1h
# Connects, saves lease to ~/.evollm/leases/abc123.json
# User gets shell prompt on remote peer

# Forward port
evosh forward --local 8080 --peer peer-b:80
# Local port 8080 → Peer B port 80

# Execute one-off command on remote
evosh exec --peer peer-b -- "python train.py --epochs 10"

# Release resources
evosh resources release --lease lease.json
evosh shell release --all  # Release all shell leases

# Monitor usage
evosh resources watch  # Live updating table of leases, costs

# Bill/accounting
evosh ledger balance --peer alice
evosh ledger transactions --last 24h
```

**Under the hood**:
- `evosh` is CLI wrapper around `EvoLLM` Python API
- Resources stored in `~/.evollm/leases/` as JSON
- Auto-renew if lease expiring and still in use

---

## Comparison: Layers vs Generalized Resources

| Aspect | Model Layers (Phase 2) | Generalized Resources (Phase 3) |
|--------|----------------------|-------------------------------|
| Allocation | Implicit (on fetch) | Explicit (allocate lease) |
| Endpoint | None (layer loaded to local memory) | Network endpoint (gRPC, SSH, mount) |
| Duration | Persistent (until evicted) | Time-bound lease |
| Control | None (fire-and-forget fetch) | Full control (release, renew, transfer) |
| Billing | Per-GB transferred | Per-time-unit (GB-hour, core-second) |
| Discovery | Registry: layer → peers | Registry: (type, specs) → offers |
| QoS | Best-effort (checkpoint on fail) | Guaranteed (lease enforces reservation) |
| Use Case | Read-only weights | Arbitrary computation/access |

**Progression**:
- Phase 2: **What's available?** (discover layers)
- Phase 3: **What can I reserve?** (lease resources)
- Phase 4: **Run my workload** (execute on leased resources)

---

## Open Questions

1. **Lease duration granularity**:
   - Minimum: 1 second? 1 minute?
   - Too short → overhead dominates
   - Too long → inflexible, resource fragmentation
   - Recommendation: Default 1 hour, min 1 minute

2. **How to handle stateful resources (shells, databases)**:
   - Eviction would lose state
   - Need checkpoint/restore (CRIU for processes, snapshot for VMs)
   - Recommendation: Stateful resources require explicit migration, not auto-evict

3. **Network topology optimization**:
   - If A needs GPU, and B has GPU but C is in between:
     - Direct A→B (may be slow)
     - Relay through C (adds hop but might be faster net)
   - Need path selection (like internet routing)
   - Recommendation: Let requester choose based on latency measurement

4. **Resource fragmentation**:
   - Peer has 2×16GB GPUs, requester needs 24GB
   - Can't satisfy even though total capacity 32GB
   - Solutions: allow peer to split GPU (time-sharing), or aggregate multiple peers
   - Recommendation: Peer can choose to split or not; requesters can ask for multiple smaller units

5. **Cross-resource dependencies**:
   - Training job needs: 4 GPUs + 128GB RAM + 1TB disk
   - Must allocate all from same peer or coordinate across peers
   - Recommendation: Allow compound lease requests (allocator tries to satisfy all constraints)

---

## Conclusion

Extending the architectural pattern to **generalized resource forwarding** is **natural and necessary** for EvoOS vision:

1. **ResourceProvider abstraction** already exists - just add `allocate()` method
2. **Registry** already tracks who has what - extend to all resource types
3. **Ledger** already does accounting - use same credit system
4. **Security** model (mTLS, ACLs) applies universally
5. **Autonomy** preserved: each peer decides locally whether to serve

**Key new concepts**:
- **Lease**: Time-bound allocation with endpoint + credentials
- **Control channel**: Management API for lifecycle operations
- **Offer**: Dynamic resource advertisement (capacity, price, QoS)
- **Resource spec**: Declarative requirements (quantity, constraints, duration)

**Result**: EvoOS becomes a **universal resource marketplace** where any compute/storage/interactive resource can be shared, leased, and controlled across the network - all with the same peer-to-peer coordination protocol that powers layer sharing.

This is essentially **"Linux for AI"** where `fork()` becomes `allocate(resource_spec)` and `exec()` becomes `lease.execute(command)` - but distributed across the cluster with economic incentives.

---

## Next Steps

If pursuing this vision:

1. **Document** this extension as Phase 3 in `TODO.md`
2. **Create** `evollm/resource_provider_generalized.py` (new module)
3. **Prototype** compute offload (simplest after layers)
4. **Benchmark** GPU-to-GPU RPC latency vs local execution
5. **Design** resource negotiation protocol (beyond simple query)
6. **Implement** resource marketplace (supply/demand pricing)

**Estimated timeline**: Phase 3 could be 6 months of engineering if pursued full-time.

---

*This document extends the peer coordination analytics to show how the same architectural pattern scales from model layer sharing to general distributed resource management - the full EvoOS vision.*
