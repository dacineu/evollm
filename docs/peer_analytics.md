# EvoLLM Peer Coordination Analytics

**Created**: 2026-03-17
**Purpose**: Analysis of peer-to-peer coordination architecture for EvoOS vision
**Related Plans**: `.claude/plans/hashed-brewing-candle.md`
**Project TODO**: `TODO.md`

---

## Executive Summary

This document analyzes the design and implementation approach for **modular peer-to-peer coordination** in EvoLLM. The goal is to extend the single-node hierarchical caching system to support distributed collaboration across multiple nodes, enabling effective sharing of model layers over the network.

**Key Insight**: Treat every layer source (GPU cache, CPU cache, Disk, Remote Peer) as a pluggable `ResourceProvider`. This abstraction allows seamless composition of local and remote resources without modifying core caching logic.

---

## Architectural Analysis

### 1. Modularity Assessment

#### Current Modularity Score: **8/10**

The existing codebase demonstrates strong modularity:

**Strengths:**
- Clear separation: `TensorCacheManager` coordinates hierarchy, `LayerCache` implements caching, `HierarchicalTensorLoader` handles async
- Well-defined interfaces: `get_layer(layer_name, layer_idx, load_fn, move_fn)` is the key contract
- Dependency inversion: High-level components depend on abstractions, not concretions
- Single responsibility: Each class has one reason to change

**Gaps:**
- `TensorCacheManager` hardcoded to local resources only
- No abstraction for "remote" resource providers
- Tight coupling: `HierarchicalTensorLoader` expects `TensorCacheManager` specifically

**Opportunity**: Introduce `ResourceProvider` ABC to generalize the `get_layer()` contract. Everything becomes a provider.

---

### 2. Communication Patterns

#### Pattern: **Registry-Mediated Peer Discovery with Direct P2P Transfer**

```
┌─────────┐         ┌────────────┐         ┌──────────┐
│ Peer A  │────────►│  Registry  │◄────────│ Peer B  │
│ (client)│  query  │   (DHT/    │advertise│ (server)│
│         │◄────────│   HTTP)    │────────►│         │
└─────────┘  peer   └────────────┘  layers  └──────────┘
    │               │
    │ fetch layer   │
    ├──────────────►│
    │ gRPC stream   │ layer data
    │◄──────────────┤
    │
    │ cache locally
    ▼
```

**Why this pattern?**

1. **Registry as directory only** - No central coordination, just "who has what"
2. **Direct P2P transfer** - No relay through central server (bandwidth efficient)
3. **Autonomous peer selection** - Each requester ranks peers independently (decentralized decision)
4. **Registry can be DHT** - Future-proof for large clusters

**Alternatives considered:**
- ❌ Central broker (bottleneck, SPOF, scales poorly)
- ❌ Broadcast queries (inefficient, doesn't scale)
- ❌ Structured overlay (too complex for initial version)

✅ **Chosen**: Simple registry (static → HTTP → DHT evolution)

---

### 3. Coordination Without Central Authority

#### Challenge: How do peers coordinate resource allocation without a scheduler?

**Solution: Multi-level priority enforcement + economic incentives**

Each peer runs an **autonomous ResourceManager** that enforces:

```
Priority Stack:
1. SELF-INFERENCE (hard bound)
   └─ Never evict layers needed for own next forward pass
   └─ Reserve prefetch layers even if others offer high price

2. LOCAL CACHE HEALTH (backpressure)
   └─ If cache > max_share_percent → reject all borrow requests
   └─ If cache pressure high → be conservative

3. ECONOMIC OPTIMIZATION (profit motive)
   └─ Accept if price × size > marginal_cost
   └─ Dynamic pricing: base × (1 + load) × reputation_bonus

4. REPUTATION & TRUST (long-term)
   └─ Serve high-reputation peers even at slight discount
   └─ Reject low-reputation abusive peers
```

**Result**: Emergent cooperation. Peers that share generously earn credits and reputation → more future requests → more income. Hoarders waste capacity and get isolated.

**Game theory**: This is a repeated prisoner's dilemma with reputation. Tit-for-tat strategies emerge naturally.

---

### 4. Cache Consistency Model

#### Problem: Distributed caches with version mismatch

**Strict consistency** (all nodes see same version) requires consensus → expensive.

**Chosen: Eventual consistency with verification**

```
Mechanism:
1. Each layer has checksum (SHA256 of tensors)
2. Peers advertise: (layer_name, checksum, peer_id)
3. Registry stores: layer_name → [(peer_id, checksum), ...]
4. On fetch:
   - Requester knows expected checksum (from model manifest)
   - Compare received checksum to expected
   - If mismatch → reject, evict stale cache, try another peer
5. On model upgrade:
   - New checksums computed
   - Peer broadcasts invalidation(old_checksum) + announcement(new_checksum)
   - Registry updates mappings
   - Peers with stale checksums will detect on next fetch attempt

Properties:
- No locking or consensus protocols
- Stale data may exist temporarily but never used
- Detection latency ≤ one fetch attempt
- Anti-entropy: periodic registry cleanup removes dead peers
```

**Trade-off**: Simplicity over strict consistency. Acceptable because:
- Model weights are immutable once loaded (read-only)
- Cache misses just load from disk or another peer
- No risk of using wrong version (checksum prevents it)

---

### 5. Financial Flow Analysis

#### Credit System Design

```
Transaction Graph:

  Peer A (borrower)            Peer B (lender)
       │                            │
       │  request layer.X           │
       ├───────────────────────────>│
       │                            │ check policy
       │                            │ ├─ cache_hit? YES
       │                            │ ├─ sharing_enabled? YES
       │                            │ ├─ requester_ok? YES
       │                            │ └─ price_ok? YES
       │                            │      ↓
       │                            │ ┌─────────────┐
       │                            │ │  STREAM     │
       │                            │ │  layer.X    │
       │                            │ └─────────────┘
       │                            │      ↓
       │                            │ ledger.credit(B, price×size)
       │                            │ ledger.debit(A, price×size)
       │                            │ reputation.record(B, success)
       │                            │
       │◄───────────────────────────┤
       │   receive layer.X         │
       │   verify checksum         │
       │   add to local cache      │
       │                            │
       └────────────────────────────┘
```

**Ledger semantics:**

- **Double-entry**: Every debit has matching credit (conservation of credits)
- **No central mint**: Credits are merely accounting entries (IOUs)
- **Initial distribution**: New peers get `initial_credits` (free trial)
- **Overdraft**: Optional, with configurable `max_debt_gb`
- **Settlement**: Out-of-band (real world) or future blockchain integration

**Economic equilibrium:**

Let supply = total cache capacity across all peers
Let demand = number of requests × size per request

If demand > supply:
- Prices rise (peers set higher prices)
- Requesters prioritize high-value requests
- Either demand reduces or more peers join (profit incentive)

If supply > demand:
- Prices fall (competition)
- Peers lower prices to attract requests
- Some peers may turn off sharing (not worth overhead)

**Free riding problem**: If all peers set price=0 (communal sharing), system works if:
- Peers are symmetric (similar capacities)
- All contribute ≈ what they consume
- No malicious free-riders

But if asymmetric:
- Large-capacity peers subsidize small ones
- May need tiered pricing or reputation-based allocation

**Recommendation**: Default to free sharing (price=0), but allow pricing to emerge organically in larger clusters.

---

### 6. Security Model

#### Threat Model

| Threat | Likelihood | Impact | Mitigation |
|--------|-----------|--------|------------|
| Data leakage (model weights exposed) | Low | High | mTLS encryption in transit; don't serve non-cache layers |
| Unauthorized borrowing (theft) | Medium | Medium | Whitelist enforcement; ledger debits require sufficient balance |
| Man-in-the-middle (tampering) | Low | High | Checksum verification on every fetch |
| Denial of service (flood requests) | Medium | Medium | Rate limiting per peer; circuit breaker |
| Reputation manipulation | Low | Medium | EMA smoothing; Sybil resistance via identity |
| Payment fraud (double-spend) | Low | Medium | Ledger atomic transactions; signed entries |
| Malicious server (garbage data) | Low | High | Checksum verification; peer blacklisting |
| Identity spoofing | Low | High | mTLS with certificate IDs; CA verification |

#### Security Components

**1. Authentication: mTLS**
```python
# Each peer has:
#   - private key (peer.key)
#   - certificate signed by CA (peer.crt)
#   - certificate includes peer_id in SAN

# Handshake:
#   Client: "I am Peer A, here's my cert"
#   Server: Verify cert signature, extract peer_id
#   If peer_id in whitelist → allow
#   Else → reject
```

**2. Authorization: Sharing Policy**
```python
class SharingPolicy:
    def allow_fetch(self, layer_name, requester_id) -> bool:
        # 1. Is layer in shareable set? (not embed/norm/head)
        if layer_name in EXCLUDE_PATTERNS:
            return False

        # 2. Is requester whitelisted?
        if requester_id not in self.allowed_peers:
            return False

        # 3. Is cache usage below threshold?
        if self.cache.usage_gb > self.max_share_gb:
            return False

        # 4. Does requester have sufficient credit?
        balance = ledger.get_balance(requester_id)
        if balance < -self.max_debt_gb:
            return False

        return True
```

**3. Integrity: Checksums**
```python
def verify_layer_integrity(state_dict: Dict, expected_checksum: str) -> bool:
    computed = compute_checksum(state_dict)
    return computed == expected_checksum
```

**4. Confidentiality: Transport Encryption**
- gRPC with TLS (AES-256-GCM)
- Optional: additional layer encryption (XOR with shared secret)

---

### 7. Performance Projections

#### Baseline (Phase 1 - Single Node)

```
Configuration: 70B Llama-2, 32GB CPU cache, 2 GPU layers
Results:
  - Throughput: 2-5 tokens/sec
  - Cache hit rate: 60-80%
  - Disk reads/token: ~30 (80 layers - cached)
  - Latency per token: 200-500ms
```

#### With 4-Node P2P Cluster (Phase 2)

```
Cluster: 4 peers × 32GB cache = 128GB effective cache
Each peer caches ~20 layers locally (32GB / 1.8GB/layer ≈ 18 layers)
Collective cache: ~80 layers (enough for 70B model)

Expected improvements:
  - Cache hit rate: 95%+ (most layers cached somewhere in cluster)
  - Disk reads/token: ~4 (only cold layers not cached by anyone)
  - Throughput: 8-15 tokens/sec (4-3× speedup)
  - Latency: 50-150ms (peer fetch ~10ms vs disk ~200ms)

Network overhead:
  - 1-2MB/layer transfer (compressed)
  - 1 fetches/token on average (mostly hits)
  - Bandwidth: 10-50 MB/s sustained per peer (negligible on 1Gbps+ LAN)
```

**Caveats:**
- Assumes low network latency (<10ms intra-rack)
- Assumes peers have overlapping but not identical cached layers
- Assumes sharing policies allow borrowing
- First token slower (cold cache), steady-state faster

---

### 8. Risk Analysis

#### Technical Risks

| Risk | Probability | Severity | Mitigation |
|------|-------------|----------|------------|
| Network latency dominates | High | High | Prefetch multiple peers in parallel; cache borrowed layers locally; penalize slow peers in ranking |
| Cache inconsistency (stale data) | Medium | High | Checksum verification on every fetch; version invalidation; reject mismatches |
| gRPC serialization bottleneck | Medium | Medium | Stream large layers; use safetensors format; cache serialized bytes |
| Thread safety issues | Medium | High | Extensive unit tests; use thread-safe data structures; review lock hierarchy |
| Memory leaks (connections, caches) | Medium | Medium | Connection pooling with TTL; periodic cleanup; memory profiling |
| Deadlocks in HybridResourceManager | Low | High | Careful lock ordering; use async/await; timeouts |
| Protocol incompatibility across versions | Medium | Medium | Version field in protocol; backward compatibility testing |

#### Operational Risks

| Risk | Probability | Severity | Mitigation |
|------|-------------|----------|------------|
| Peer failure mid-generation | High | Medium | Circuit breaker; timeout; fallback to disk; retry with alternate |
| Registry downtime (if centralized) | Medium | High | Peer local cache of registry data; fallback to static bootstrap; DHT future |
| Ledger corruption/loss | Low | Medium | Transaction logging; periodic backups; redundant storage |
| Security breach (unauthorized access) | Low | Critical | mTLS mandatory in production; audit logging; intrusion detection |
| Economic game exploits (sybil, collusion) | Medium | Medium | Reputation system; rate limiting; analysis of transaction patterns |
| Configuration complexity overwhelms users | High | Medium | Sensible defaults; auto-config; validation; clear error messages |
| Legal/compliance (sharing model weights) | Low | High | License compatibility check; terms of service; opt-in sharing |

---

### 9. Alternatives Considered

#### Alternative 1: Centralized Scheduler
```
Central brain coordinates all peers
- Pros: Optimal load balancing, can enforce fairness, global view
- Cons: SPOF, scaling challenges, complex, doesn't match EvoOS P2P vision
→ Rejected
```

#### Alternative 2: P2P DHT Only (No Registry)
```
Fully decentralized Kademlia DHT for layer→peer mapping
- Pros: No central point, scales well, resilient
- Cons: Complex to implement, harder to debug, bootstrapping complex
→ Deferred to Phase 3 (DHT tier)
```

#### Alternative 3: Broadcast Queries
```
"Who has layer.X?" multicast to all peers
- Pros: Simple, no registry needed
- Cons: Doesn't scale (>100 peers), network storm
→ Rejected
```

#### Alternative 4: Ownership-based (Static Partitioning)
```
Pre-assign each layer to specific peer (like sharding)
- Pros: Predictable, no coordination needed
- Cons: Inflexible, load imbalance, single point of failure per layer
→ Rejected (but useful for specialized deployments)

**Chosen approach**: Dynamic discovery with voluntary sharing
```

---

### 10. Implementation Trade-offs

#### gRPC vs HTTP/JSON

**gRPC chosen because:**
- ✅ Binary protocol → efficient for large layers (1-2GB)
- ✅ Streaming → don't buffer entire layer in memory
- ✅ Strong typing → Protocol Buffers schema
- ✅ Bidirectional streaming → future: real-time progress updates
- ✅ Built-in compression (optional)
- ✅ Multi-language support

**Disadvantages:**
- ❌ Requires protoc compiler (build complexity)
- ❌ Less human-readable than JSON (debugging harder)
- ❌ HTTP tooling (curl, Postman) doesn't work directly

**Mitigation**: Add `--debug-json` mode that uses HTTP for development? Or use grpc-gateway for REST/JSON proxy.

---

#### Checksum on Every Fetch vs Occasional

**Option A: Verify every fetch** (chosen)
- ✅ Guarantees integrity
- ✅ Early detection of staleness
- ❌ CPU overhead (hash 2GB layer ≈ 1-2s on CPU? expensive)

**Option B: Verify only on suspicious patterns**
- ✅ Saves checksum computation
- ❌ May use stale data longer

**Mitigation**: Cache checksum value with advertised layer. Only recompute if:
- Checksum not cached
- Layer size differs
- Transferred size ≠ expected

Actually: Compute checksum once when advertising, reuse. Requester checksum already known from model manifest. So verification is just hash comparison, not hash computation. ✓ No issue.

---

#### Cache Borrowed Layers Locally?

**Question**: When Peer A fetches layer from Peer B, should A cache it locally for future reuse?

**Pros:**
- ✅ Reduces subsequent network fetches (within generation reuse)
- ✅ Aggregates remote layers into local cache (natural distribution)
- ✅ Reduces load on Peer B

**Cons:**
- ❌ Increases Peer A's cache pressure (may evict other useful layers)
- ❌ Cache ownership semantics unclear (who evicts?)

**Decision**: YES, cache borrowed layers. Treat them as any other cache entry. Eviction policy same.credit cost already paid, so benefit from reuse.

---

#### Peer Selection: Requester vs Server Choice

**Who decides which peer serves a request?**

**Option A: Requester selects** (chosen)
- Requester queries registry for all peers with layer
- Requester ranks and picks best
- Request goes directly to chosen peer
- ✅ Load balancing distributed across requesters
- ✅ No central coordination
- ❌ Peer may reject (too busy, insufficient credits) → requester retries

**Option B: Server decides (peer volunteering)**
- Requester broadcasts "need layer.X"
- Peers with layer.X volunteer (push model)
- Requester picks first available
- ✅ Load spreads naturally (available peers respond)
- ❌ More network chatter (broadcasts)

**Hybrid**: Requester queries, but peers indicate load, and requester respects `Retry-After` headers if overloaded.

✅ **Chosen**: Requester-driven with server feedback (load, commerce)

---

## Data Flow: Detailed Request Trace

### Scenario: Peer A needs layer.20, doesn't have it locally

```
Timeline (ms):
────────────────────────────────────────────────────────────────────────────

  0ms   1. Forward pass reaches layer.20
         EvoLLMModel._load_layer_with_cache("layer.20")
         HierarchicalTensorLoader.load_layer("layer.20", idx=20, ...)
         HybridResourceManager.get_layer("layer.20", 20, load_fn, move_fn)

  1ms   2. Try LocalResourceManager (cache miss)
         LayerCache.get("layer.20") → None
         Return: LayerNotFoundError

  2ms   3. Try PeerClientResourceManager
         PeerLayerFetcher.fetch("layer.20", 20)
         PeerRegistry.find_peers_for_layer("layer.20")
         → Query local registry cache (updated via last advertisement)

  3ms   4. Registry returns:
         [
           PeerInfo(id="B", addr="10.0.0.2:50051", lat=6ms, rep=0.88, price=0.0, load=0.4),
           PeerInfo(id="D", addr="10.0.0.4:50051", lat=3ms, rep=0.98, price=0.0, load=0.1)
         ]

  4ms   5. Rank peers:
         score(B) = 2.0×0.88 + 1.0×(1-0.006) + 0.5×0.5 - 0.1×0 - 0.3×0.4 = 3.0
         score(D) = 2.0×0.98 + 1.0×(1-0.003) + 0.5×0.9 - 0.1×0 - 0.3×0.1 = 3.77
         Ranked: [D, B]

  5ms   6. Try Peer D (highest score)
         PeerLayerFetcher._fetch_from_peer(D, "layer.20")
         gRPC stub = connection_pool.get("10.0.0.4:50051")
         if not connected: create channel with mTLS, connect

  10ms  7. gRPC call: FetchLayer(layer.20, requester_id="A", timeout=5000)
         Stream: FetchLayerRequest → FetchLayerResponse chunks

  15ms  8. Receive stream (1.8GB over 5ms = 360 GB/s effective, but likely cached on D's side)
         Assemble tensors into state_dict
         Compute checksum of received data

  16ms  9. Verify checksum:
         received_checksum = compute(state_dict)
         expected = model_manifest["layer.20"]
         if received != expected:
             PeerLayerFetcher._update_peer_reputation(D, success=False)
             raise LayerChecksumMismatchError
         ✅ match

  17ms 10. Update Peer D stats:
          PeerInfo.latency_ms = 15ms (EMA update)
          PeerInfo.load_score += 0.1 (temporary boost for this request)
          PeerLayerFetcher.stats['fetch_success'] += 1

  18ms 11. Ledger transaction (async):
          ledger.debit("A", amount=1.8, desc="layer.20 from D")
          ledger.credit("D", amount=1.8, desc="layer.20 to A")
          → Ledger balances: A: +15.2 -1.8 = +13.4, D: +45.3 +1.8 = +47.1

  19ms 12. Reputation update:
          ReputationManager.record("D", success=True, latency=15ms, bytes=1.8GB)
          ReputationManager.record("A", success=True, latency=15ms, bytes=1.8GB)

  20ms 13. Return to HybridResourceManager:
          source = "peer:D"
          state_dict = { ... }

  21ms 14. HybridResourceManager returns to tensor loader:
          stats['peer_hits'] += 1

  22ms 15. HierarchicalTensorLoader returns to EvoLLMModel:
          state_dict, "peer:D"

  23ms 16. EvoLLMModel.move_layer_to_device(state_dict)
          Move 1.8GB from CPU to GPU: ~10-50ms (PCIe bandwidth)

  33ms 17. Run transformer layer.20 computation
          ~50-200ms depending on hardware

  83ms 18. Eviction decision (if not multi-layer caching):
          Should we keep layer.20 in GPU?
          if gpu_layers=2 and layer_idx=20 → NO
          layer.to("meta")  # Evict

  84ms 19. Continue to next layer...

Total overhead vs local disk load:
- Local disk: ~200ms (read) + 50ms (to GPU) = 250ms
- Peer fetch: ~15ms (network) + 50ms (to GPU) = 65ms
- Speedup: 3.8× faster for this layer!
```

**Key observation**: Network can be faster than SSD (10-50ms vs 200ms) if on same LAN. So peer fetch is **strictly better** than disk for cached layers.

---

## Configuration Taxonomy

### Mode Selection Guide

| Use Case | Local Cache | Peer Mode | Rationale |
|----------|-------------|-----------|-----------|
| Single node, no network | Yes | Disabled | Default, no dependencies |
| Single node with spare RAM, want to share | Yes | Server | Contribute to cluster, earn credits |
| Single node, need more cache | Limited | Client | Borrow from peers to augment cache |
| Multi-node cluster, all powerful | Yes | Hybrid | Both borrow and lend |
| Mobile/edge (limited) | Small | Client only | Borrow from cloud peers |
| Cloud server (high capacity) | Large | Server only | Lend to edge devices |

### Configuration Presets

```yaml
# conservative: Memory-first, minimal sharing
cpu_cache_gb: 48
gpu_layers: 1
peer:
  enabled: false

# balanced: Good performance, moderate sharing
cpu_cache_gb: 24
gpu_layers: 2
peer:
  enabled: true
  mode: hybrid
  max_share_cache_percent: 0.3
  price_per_gb: 0.0
  fetch_timeout_ms: 3000

# performance: Push limits, aggressive sharing
cpu_cache_gb: 48
gpu_layers: 4
peer:
  enabled: true
  mode: client
  max_peers_to_query: 5
  prefetch_from_peers: true

# memory_saver: Low RAM, rely on peers
cpu_cache_gb: 8
gpu_layers: 0
peer:
  enabled: true
  mode: client
  bootstrap_peers: ["10.0.0.1:50051"]
  fallback_to_local: false  # all from peers or fail

# mobile: Battery/thermal constraints
cpu_cache_gb: 4
gpu_layers: 0
peer:
  enabled: true
  mode: client
  max_peers_to_query: 2
  fetch_timeout_ms: 10000  # tolerate slower networks
```

---

## Monitoring & Metrics

### Key Performance Indicators (KPIs)

**Throughput**
- `tokens_per_second` overall
- `tokens_per_second` per layer (should be flat after first pass)

**Cache Effectiveness**
- `overall_hit_rate` = (hits) / (hits + misses)
- `local_cache.hit_rate` vs `peer_hit_rate`
- `disk_loads_per_token` (target < 5)

**Network Utilization**
- `peer_fetch_latency_ms` p50, p95, p99
- `peer_fetch_throughput_mb_s`
- `peer_connection_errors` rate

**Economic**
- `ledger_balance` per peer
- `credits_earned_per_hour`
- `credits_spent_per_hour`
- `reputation_score` trends

**Quality**
- `checksum_mismatch_rate` (should be 0)
- `peer_failure_rate` (fraction of fetches that fail)
- `fallback_to_disk_rate`

### Metrics Export (Prometheus)

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
peer_fetch_success = Counter('evollm_peer_fetch_success', 'Successful peer fetches')
peer_fetch_failure = Counter('evollm_peer_fetch_failure', 'Failed peer fetches')
peer_bytes_transferred = Counter('evollm_peer_bytes_total', 'Bytes transferred from peers')

# Histograms
peer_fetch_latency = Histogram('evollm_peer_fetch_latency_seconds', 'Peer fetch latency')
peer_ranking_score = Histogram('evollm_peer_ranking_score', 'Scores assigned to ranked peers')

# Gauges
cache_hit_rate = Gauge('evollm_cache_hit_rate', 'Overall cache hit rate')
ledger_balance = Gauge('evollm_ledger_balance', 'Credit balance per peer', ['peer_id'])
peer_count = Gauge('evollm_peer_count', 'Number of peers in registry')
```

### Logging Strategy

```python
import structlog

logger = structlog.get_logger()

# During fetch:
logger.info("peer.fetch.start",
    layer=layer_name,
    peer_id=selected_peer.id,
    latency_est_ms=selected_peer.latency_ms,
    price=selected_peer.price_per_gb)

# On success:
logger.info("peer.fetch.success",
    layer=layer_name,
    peer_id=selected_peer.id,
    latency_ms=elapsed,
    bytes_transferred=size,
    checksum=checksum,
    ledger_debit=debit_amount)

# On failure:
logger.warning("peer.fetch.failed",
    layer=layer_name,
    peer_id=selected_peer.id,
    error=str(error),
    retry_count=retries)

# Periodic stats:
logger.info("peer.stats",
    hit_rate=cache_hit_rate,
    peer_hits=peer_hits,
    disk_loads=disk_loads,
    ledger_balance=balance)
```

---

## Open Design Questions

### 1. Should borrowed layers be cached for longer than one generation?

**Arguments for**:
- Reuse within same generation (prefill → decode)
- Amortize network cost across multiple token predictions
- Reduce pressure on remote peer

**Arguments against**:
- Cache pollution (evicts other useful layers)
- Layer may become stale (model upgrade)
- Borrowing cost may exceed benefit if not reused

**Hypothesis**: Cache borrowed layers with **shorter TTL** than locally loaded layers. E.g., local layers: LRU with unlimited retention; borrowed layers: LRU with max 1 hour or 10 uses.

**Implementation**: `BorrowedLayerCacheEntry` with `borrowed_at` timestamp and `max_age_seconds`. Evict if expired even if recently used.

---

### 2. Should peers be able to specify "quality of service" tiers?

**Idea**: High-reputation peers or high-balance peers get priority service (faster fetch, guaranteed bandwidth).

**Mechanism**:
- `SharingPolicy` has `tier_priority` mapping:
  ```
  tier_priority = {
      'platinum': 1.0,   # balance > 1000, rep > 0.95
      'gold': 0.8,       # balance > 100, rep > 0.8
      'silver': 0.6,     # default
      'free': 0.4,       # free tier, lower priority
  }
  ```
- Server's `FetchLayer` handler checks peer's tier, applies rate limits accordingly
- High-tier peers get:
  - Higher `max_concurrent_requests`
  - Longer `timeout_ms`
  - Access to more expensive (high-demand) layers

**Trade-off**: Adds complexity. May not be needed initially.

**Recommendation**: Defer to v2. Keep it simple: FIFO or shortest-job-first based on known layer sizes.

---

### 3. Should there be a "warm-up" phase where peers exchange layers preemptively?

**Problem**: Cold start: When cluster first starts, no one has any layers cached. All borrow from disk → no benefit.

**Solution**: Warm-up phase based on model layer popularity.
```
1. Each peer loads model manifest (layer sizes, expected access frequency)
2. Estimate hot layers (e.g., first 20 transformer layers used in every generation)
3. Each peer downloads hot layers in advance (from disk or peers)
4. After warm-up, all peers have hot layers cached locally
5. Cold layers borrowed on demand
```

**Alternative**: Don't warm up. First few generations slow. Acceptable if warm-up cost is distributed and happens organically as peers generate.

**Recommendation**: Simple warm-up: On server start, automatically load first `N` layers (configurable, e.g., N=10). No peer coordination needed.

---

### 4. How to handle multi-tenancy (multiple users/models on same cluster)?

**Scenario**: Cloud provider wants to run multiple customers' models on same cluster, isolated.

**Challenges**:
- Sharing should be within same tenant, not cross-tenant
- Cache isolation: Tenant A shouldn't evict Tenant B's layers
- Billing: Tenant A pays for their own resource usage

**Possible solutions**:

1. **Namespace per model**:
   ```
   cache keys: "model_id/layer_name"
   peer ads: {"model_id/llama2-70b": ["layer.0", ...]}
   ```
   Different models don't see each other's layers.

2. **Multi-tenant registry**:
   Registry partitions by tenant_id. Only peers with same tenant_id see each other.
   Harder: requires authentication to query registry by tenant.

3. **Separate clusters per tenant**:
   Simplest: just run separate EvoLLM instances. No cross-tenant sharing.

**Recommendation**: Namespace approach (1). Lightweight, no extra infrastructure. Model ID encoded in layer name prefix.

---

## Implementation Roadmap (Detailed)

### Sprint 1 (Weeks 1-2): Foundation

**Goal**: ResourceProvider abstraction working, LocalResourceManager passes all tests.

**Tasks**:
1. Create `evollm/resource_provider.py`
   - Define `ResourceProvider` ABC with docstrings
   - Define exceptions: `LayerNotFoundError`, `ResourceUnavailableError`
   - Add type hints throughout

2. Implement `LocalResourceManager`
   - Wrap `TensorCacheManager.get_layer()` exactly
   - Translate `cache_layer()` to `cpu_cache.put()` if cache exists
   - `get_stats()` returns underlying manager's stats
   - `shutdown()` no-op

3. Write unit tests
   ```python
   def test_local_resource_manager_wraps_cache():
       mock_cache = Mock(spec=TensorCacheManager)
       mock_cache.get.return_value = {"tensor": torch.rand(2, 2)}
       manager = LocalResourceManager(mock_cache, load_fn, move_fn)
       state, source = manager.get_layer("layer.0", 0, None, None)
       assert state == {"tensor": ...}
       assert source in ['gpu', 'cpu', 'disk']
   ```

4. Refactor `create_cache()` to return `ResourceProvider`
   - Rename to `create_resource_manager()` or keep both
   - Update callers: `evollm_base.py`, tests

5. Update `HierarchicalTensorLoader`
   - Change `__init__(self, resource_manager: ResourceProvider, ...)`
   - Replace `self.cache_manager.get_layer()` with `self.resource_manager.get_layer()`
   - Update tests

**Acceptance criteria**:
- [x] All existing unit tests pass ( Phase 1 tests need to be written first)
- [ ] ResourceProvider interface documented
- [ ] LocalResourceManager tests pass with mocks
- [ ] No regression in single-node performance

---

### Sprint 2 (Weeks 3-4): Peer Client

**Goal**: Can query registry and fetch layer from mock peer.

**Tasks**:

1. Define gRPC protocol
   - Create `evollm/peer/protocol.proto`
   - Define messages: `FetchLayerRequest`, `FetchLayerResponse`, `HasLayerRequest`, `HasLayerResponse`, `AdvertiseRequest`, `AdvertiseResponse`
   - Define service: `PeerService`
   - Generate stubs: `python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. evollm/peer/protocol.proto`
   - Add to git: generated files

2. Implement `PeerRegistry` (static mode)
   - `__init__(mode='static', bootstrap_peers=None)`
   - `add_peer(PeerInfo)`, `remove_peer(peer_id)`
   - `find_peers_for_layer(layer_name) → List[PeerInfo]`
   - Load bootstrap peers on init
   - Periodic refresh (pull from static list? Not needed for static)
   - Unit tests: add/remove, find, multiple peers

3. Implement `PeerLayerFetcher`
   - `__init__(registry, config)`
   - `fetch(layer_name, layer_idx)` with try-except and retry
   - `_rank_peers(peers)` scoring
   - `_fetch_from_peer(peer, layer_name)` gRPC call
   - `_update_peer_reputation(peer_id, success)` after fetch
   - Connection pooling dict: `self.connections[peer_id] = grpc.Channel`
   - Stats tracking: `self.stats`
   - Unit tests: mock gRPC server, test success/failure/retry/ranking

4. Implement `PeerClientResourceManager`
   - `get_layer()` delegates to `fetcher.fetch()`
   - On `PeerUnavailableError`: if `fallback_to_local`, call `local_fallback_fn(layer_name)`
   - `cache_layer()` returns False (no caching)
   - `get_stats()` combines own stats + fetcher stats
   - `shutdown()` calls `fetcher.shutdown()`

5. Mock gRPC server for testing
   - `MockPeerServer` with layers dict
   - gRPC servicer implementation using `unittest.mock` or `grpc_testing`
   - Configurable latency, failures

6. Integrate in `HybridResourceManager`
   - Test policy: `local_first` tries local, then `PeerClientResourceManager`
   - Verify fallback to disk works if all peers fail

**Acceptance criteria**:
- [ ] Can fetch layer from mock peer with correct checksum
- [ ] Ranking correctly orders peers by score
- [ ] Retry works after transient failure
- [ ] Circuit breaker blacklists peer after 3 failures
- [ ] Fallback to local disk on no peers
- [ ] Integration test: `HybridResourceManager([Local, PeerClient])` fetches from peer on local miss

---

### Sprint 3 (Weeks 5-6): Peer Server

**Goal**: Can serve layers to requesting peers.

**Tasks**:

1. Define `PeerServerConfig` in `config.py`
   - Port, advertise_interval, max_connections
   - Sharing policy: `sharing_enabled`, `max_share_cache_percent`, `share_exclude_layers`
   - Security: `require_tls`, `allowed_peer_cidrs`

2. Implement `PeerServer`
   - `__init__(config, local_cache, ledger=None)`
   - `async start()`: start gRPC server (`grpc.aio.server()`), start advertise task
   - `async stop()`: cancel tasks, stop server
   - `_make_servicer()`: create `LayerServiceServicer` instance
   - `_periodic_advertise()`: every N seconds, call `self.advertise_to_registry()`
   - `_advertise_to_registry()`: send Advertise RPC to registry (if enabled)
   - `_generate_peer_id()`: uuid4

3. Implement gRPC servicer methods
   - `HasLayer`: check `layer_name in self.local_cache.cache`, return bool
   - `FetchLayer`: stream chunks
     ```python
     async def FetchLayer(self, request, context):
         # Auth: extract peer_id from TLS cert
         peer_id = extract_peer_id(context)
         if not self.policy.allow_fetch(request.layer_name, peer_id):
             yield FetchLayerResponse(error="forbidden")
             return

         if request.layer_name not in self.local_cache.cache:
             yield FetchLayerResponse(error="not_found")
             return

         entry = self.local_cache.cache[request.layer_name]
         checksum = compute_checksum(entry.state_dict)

         # Stream in chunks
         for chunk in serialize_in_chunks(entry.state_dict, chunk_size_mb=10):
             yield FetchLayerResponse(data=chunk, checksum=checksum, size=len(chunk))

         # Ledger: credit self
         size_gb = entry.size_bytes / 1e9
         price = self.policy.get_price(request.requester_id)
         self.ledger.credit(self.peer_id, price * size_gb, f"saved:{request.layer_name}")
     ```
   - `AdvertiseLayers`: registry callback, update registry's view
   - `Heartbeat`: update peer liveness

4. Implement serialization
   - `serialize_state_dict(state_dict) → bytes` using `torch.save` with ` BytesIO`
   - `deserialize_state_dict(bytes) → dict` using `torch.load`
   - Consider `safetensors` for security (no pickle)
   - Chunking: split into N MB chunks for streaming

5. Security middleware
   - gRPC interceptor to extract client cert, validate
   - IP whitelist check: `context.peer()` → IP, check against `allowed_peer_cidrs`
   - Rate limiting per peer: track requests/sec, reject if exceeds threshold

6. Tests
   - Unit: Server init, servicer methods with mock cache
   - Integration: Client fetches from server, verifies data
   - Test sharing policy blocks excluded layers
   - Test rate limiting rejects flooded requests
   - Test mTLS handshake (if enabled)

**Acceptance criteria**:
- [ ] PeerServer starts and listens on port
- [ ] HasLayer returns correct bool for cached/uncached layers
- [ ] FetchLayer streams layer data correctly
- [ ] Sharing policy enforced (exclude layers, max %)
- [ ] Ledger credits updated on successful fetch
- [ ] Mock client can fetch and verify checksum
- [ ] Server handles concurrent requests (ThreadPoolExecutor)

---

### Sprint 4 (Week 7): Hybrid & Ledger

**Goal**: Combine multiple providers; implement credit accounting.

**Tasks**:

1. Implement `HybridResourceManager` policies
   - `local_first`: sequential try, first success wins
   - `parallel`: ThreadPoolExecutor, wait for first success, cancel others
   - `cheapest`: query all providers for price, pick cheapest that can serve
   - Stats: track `local_hits`, `peer_hits`, `fallbacks`
   - `cache_layer()`: best-effort on all providers
   - Unit tests for each policy with mocks

2. Implement `Ledger`
   - Choose implementation: start with `LocalInMemoryLedger`
   - `credit(peer_id, amount, desc)`, `debit(peer_id, amount, desc) → bool`
   - `get_balance(peer_id) → float`
   - `enforce_rate_limit(peer_id, max_debt) → bool`
   - Transaction log: `[(timestamp, peer_id, amount, desc)]`
   - Thread safety: `threading.Lock` or `asyncio.Lock`
   - Tests: credit/debit, overdraft prevention, transaction ordering

3. Implement `ReputationManager`
   - `PeerReputation` dataclass: success_rate (EMA), avg_latency_ms (EMA), total_volume_gb, last_seen
   - `record_transaction(peer_id, success, latency_ms, bytes)`: update EMA
   - `get_reputation_score(peer_id) → float`: weighted combination
   - Persistence: Optional save/load to JSON (for single-node)

4. Wire ledger into `PeerLayerFetcher` ranking
   - Include balance in score: penalize debtors
   - After successful fetch: `ledger.debit(requester_id, price * size)`, `ledger.credit(server_id, price * size)`
   - In `_rank_peers()`: factor in `price_per_gb` from `PeerInfo`

5. Wire reputation into `PeerRegistry`
   - `PeerInfo` has `reputation` field
   - `PeerRegistry` owns `ReputationManager`
   - When peer advertises, update `PeerInfo.reputation` from manager
   - `PeerLayerFetcher._rank_peers()` uses `PeerInfo.reputation`

6. End-to-end test with 2 peers
   - Peer A: `HybridResourceManager([Local, PeerClient])`, `PeerServer` running
   - Peer B: Same config, bootstrap_peers=[A]
   - Generate inference workload
   - Verify: Ledger balances reflect transactions
   - Verify: Reputation scores updated
   - Verify: Cache hit rates improved

**Acceptance criteria**:
- [ ] Ledger tracks debits/credits correctly
- [ ] Reputation scores converge with experiences
- [ ] Peer ranking incorporates price, reputation
- [ ] End-to-end: A borrows from B, B earns credits, both reputations improve
- [ ] HybridResourceManager local_first policy works

---

### Sprint 5 (Week 8): Consistency & Security

**Goal**: Model version coherence; secure communication.

**Tasks**:

1. Implement checksumming
   - `compute_checksum(state_dict) → str` using hashlib.sha256
   - Cache checksum in `PeerInfo` (per layer)
   - When advertising, include checksums
   - Registry stores: `layer_to_peers[layer_name] = [(peer_id, checksum), ...]`
   - `PeerLayerFetcher` compares received checksum to expected (from local model manifest)
   - On mismatch: reject, mark peer stale, evict local cached copy

2. Implement cache invalidation
   - `PeerServer` detects model version change
   - Broadcast to registry: `invalidate_layer(layer_name, old_checksum)`
   - Registry removes `(peer_id, old_checksum)` from `layer_to_peers`
   - Peers with stale checksum will not find that peer in future queries

3. Implement mTLS
   - Generate CA cert (self-signed for dev)
   - Generate peer certs signed by CA, with `peer_id` in Subject Alternative Name
   - `grpc.aio.server(credentials=grpc.ssl_channel_credentials(...))`
   - Client: `grpc.ssl_channel_credentials(root_certs=ca, private_key=peer_key, certificate_chain=peer_cert)`
   - Interceptor to extract `peer_id` from cert and pass to handlers via `context`

4. Implement whitelist enforcement
   - In servicer: check `peer_id` against `allowed_peer_ids` set
   - Also check IP against `allowed_peer_cidrs` (ipaddress module)
   - Reject with `UNAUTHENTICATED` gRPC status

5. Security tests
   - Client without cert rejected
   - Client with wrong cert rejected
   - Client on unauthorized IP rejected
   - Valid client succeeds
   - Checksum mismatch detected and rejected

6. Integration tests
   - Simulate model upgrade on Peer D
   - Verify Peer A detects stale layer and fetches fresh
   - Test invalidation propagation

**Acceptance criteria**:
- [ ] All peer communication encrypted with TLS
- [ ] Only whitelisted peers can connect
- [ ] Checksum verified on every fetch
- [ ] Stale layers detected and evicted
- [ ] Model upgrade triggers invalidation
- [ ] Security tests pass

---

### Sprint 6 (Week 9): Polish & Release

**Goal**: Production-ready release v0.2.0.

**Tasks**:

1. Comprehensive testing
   - [ ] All unit tests (target 80% coverage)
   - [ ] Integration tests: 2-node, 4-node clusters
   - [ ] Stress test: 100+ peers (simulated)
   - [ ] Long-running stability test (24h)
   - [ ] Fault injection: kill peers, network partitions, disk failures

2. Performance tuning
   - [ ] Profile serialization overhead → optimize
   - [ ] Tune ThreadPoolExecutor sizes
   - [ ] Benchmark different chunk sizes for streaming
   - [ ] Optimize checksum computation (maybe cache in state_dict metadata)
   - [ ] Reduce memory copies during deserialization

3. Documentation
   - [ ] Complete PEER_GUIDE.md with deployment examples
   - [ ] Update README with peer setup instructions
   - [ ] API reference (auto-generated Sphinx)
   - [ ] Troubleshooting guide (common errors, network config, firewall)
   - [ ] Performance tuning guide (how to choose prefetch_depth, cache sizes)

4. Examples
   - [ ] `examples/peer_server.py`: Start server with config
   - [ ] `examples/peer_client.py`: Connect to cluster and run inference
   - [ ] `examples/cluster_demo.py`: Automated 2-node demo using subprocess
   - [ ] Update existing examples to include peer options

5. Release preparation
   - [ ] Update version to 0.2.0 in `__init__.py`
   - [ ] Update CHANGELOG.md
   - [ ] Build wheels: `python -m build`
   - [ ] Test install from wheel in clean venv
   - [ ] Upload to PyPI: `twine upload dist/*`
   - [ ] Create GitHub release with changelog
   - [ ] Docker images: build and push to ghcr.io

6. Post-release
   - [ ] Monitor issues
   - [ ] Gather feedback from early adopters
   - [ ] Plan v0.2.1 bugfix release
   - [ ] Start planning v0.3 (DHT discovery)

**Acceptance criteria**:
- [ ] All tests pass, coverage ≥80%
- [ ] Performance meets projections (within 20%)
- [ ] Documentation complete and reviewed
- [ ] Package publishes successfully
- [ ] Example scripts work end-to-end on test cluster

---

## Conclusion

This analytics document provides a comprehensive analysis of the peer-to-peer coordination architecture for EvoLLM, covering:

- ✅ **Modularity assessment** and abstraction strategy
- ✅ **Communication patterns** (registry-mediated P2P)
- ✅ **Coordination mechanisms** (autonomous resource managers with priority stacks)
- ✅ **Cache consistency** (checksums, eventual consistency)
- ✅ **Financial flows** (ledger, reputation, economic equilibrium)
- ✅ **Security model** (mTLS, checksums, whitelisting)
- ✅ **Performance projections** (3-4× speedup expected)
- ✅ **Risk analysis** with mitigations
- ✅ **Alternatives** and why they were rejected
- ✅ **Implementation roadmap** with 6 sprints, ~9 weeks

The design prioritizes:
1. **Autonomy**: No central coordinator, each peer makes local decisions
2. **Simplicity**: Build on existing abstractions, minimal new concepts
3. **Incentive alignment**: Credit system encourages sharing
4. **Graceful degradation**: Works as standalone if network fails
5. **Backward compatibility**: Phase 1 code unchanged

**Next steps**: Begin Sprint 1 (ResourceProvider abstraction) after Phase 1 unit tests are in place.

---

**Document Status**: DRAFT - Awaiting implementation feedback to refine assumptions.

**Feedback Loop**: As implementation progresses, this document should be updated with:
- Revised performance numbers from actual benchmarks
- New risks discovered during development
- API changes based on implementation details
- User feedback on configuration complexity
