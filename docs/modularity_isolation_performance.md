# EvoLLM Peer Modularity, Isolation, and Network Performance

**Visual Analysis**: How peers maintain autonomy while collaborating, and the performance implications of network communication.

---

## 1. Modularity & Isolation: The Peer as a Black Box

### Architectural Diagram

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                                   PEER A                                          ║
║                                                                                    ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐  ║
║  │                         SHARED NOTHING ARCHITECTURE                        │  ║
║  │                                                                             │  ║
║  │  ╔═══════════════════════════════════════════════════════════════════════╗ │  ║
║  │  ║                    PEER A'S PRIVATE RESOURCES                         ║ │  ║
║  │  ║                                                                       ║ │  ║
║  │  ║  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               ║ │  ║
║  │  ║  │   GPU VRAM   │  │   CPU RAM    │  │     SSD      │               ║ │  ║
║  │  ║  │  ┌─────────┐ │  │  ┌─────────┐ │  │  ┌─────────┐ │               ║ │  ║
║  │  ║  │  │ Layers: │ │  │  │ Layers: │ │  │  │ All     │ │               ║ │  ║
║  │  ║  │  │ 0-2     │◄┼──┼─►│ 3-19    │◄┼──┼─►│ Layers   │ │               ║ │  ║
║  │  ║  │  │ (fast)  │ │  │  │ (cache) │ │  │  │ 0-79     │ │               ║ │  ║
║  │  ║  │  └─────────┘ │  │  └─────────┘ │  │  └─────────┘ │               ║ │  ║
║  │  ║  │  4GB used   │  │  28GB used  │  │  140GB total │               ║ │  ║
║  │  ║  └──────────────┘  └──────────────┘  └──────────────┘               ║ │  ║
║  │  ║                                                                       ║ │  ║
║  │  ║  ┌─────────────────────────────────────────────────────┐              ║ │  ║
║  │  ║  │           PEER A'S CONTROL PLANE                    │              ║ │  ║
║  │  ║  │  ┌─────────────────────────────────────────────┐   │              ║ │  ║
║  │  ║  │  │ EvoLLM Model                                │   │              ║ │  ║
║  │  ║  │  │ • LayerCache (LRU)                         │   │              ║ │  ║
║  │  ║  │  │ • TensorLoader (prefetch)                  │   │              ║ │  ║
║  │  ║  │  │ • Forward pass logic                       │   │              ║ │  ║
║  │  ║  │  └─────────────────────────────────────────────┘   │              ║ │  ║
║  │  ║  │  ┌─────────────────────────────────────────────┐   │              ║ │  ║
║  │  ║  │  │ ResourceManager (Local)                    │   │              ║ │  ║
║  │  ║  │  │ • get_layer() → LocalResourceManager       │   │              ║ │  ║
║  │  ║  │  │ • Policy: local_first, cache_only          │   │              ║ │  ║
║  │  ║  │  └─────────────────────────────────────────────┘   │              ║ │  ║
║  │  ║  │  ┌─────────────────────────────────────────────┐   │              ║ │  ║
║  │  ║  │  │ Sharing Policy (Outgoing)                  │   │              ║ │  ║
║  │  ║  │  │ • max_share_percent = 50%                  │   │              ║ │  ║
║  │  ║  │  │ • price_per_gb = 0.1 FIT                   │   │              ║ │  ║
║  │  ║  │  │ • exclude = ['embed', 'norm']              │   │              ║ │  ║
║  │  ║  │  │ • whitelist = ['peer-B', 'peer-C']         │   │              ║ │  ║
║  │  ║  │  └─────────────────────────────────────────────┘   │              ║ │  ║
║  │  ║  │  ┌─────────────────────────────────────────────┐   │              ║ │  ║
║  │  ║  │  │ PeerServer (gRPC)                          │   │              ║ │  ║
║  │  ║  │  │ • Listen: 0.0.0.0:50051                   │   │              ║ │  ║
║  │  ║  │  │ • Authenticate: mTLS                       │   │              ║ │  ║
║  │  ║  │  │ • Authorize: SharingPolicy above           │   │              ║ │  ║
║  │  ║  │  │ • Advertise to Registry every 30s          │   │              ║ │  ║
║  │  ║  │  └─────────────────────────────────────────────┘   │              ║ │  ║
║  │  ║  │  ┌─────────────────────────────────────────────┐   │              ║ │  ║
║  │  ║  │  │ PeerClient (gRPC)                          │   │              ║ │  ║
║  │  ║  │  │ • Query Registry for needed layers         │   │              ║ │  ║
║  │  ║  │  │ • Rank peers by score                      │   │              ║ │  ║
║  │  ║  │  │ • Fetch from best peer                    │   │              ║ │  ║
║  │  ║  │  │ • Circuit breaker on failures             │   │              ║ │  ║
║  │  ║  │  └─────────────────────────────────────────────┘   │              ║ │  ║
║  │  ║  │                                                       │              ║ │  ║
║  │  ║  └───────────────────────────────────────────────────────┘              ║ │  ║
║  │  ║                                                                       ║ │  ║
║  │  ║  ┌─────────────────────────────────────────────────────────────────┐   ║ │  ║
║  │  ║  │                    RESOURCE OWNERSHIP                           │   ║ │  ║
║  │  ║  │  ┌──────────────────────────────────────────────────────────┐  │   ║ │  ║
║  │  ║  │  │ SOVEREIGNITY: Peer A fully controls its resources        │  │   ║ │  ║
║  │  ║  │  │                                                                 │  │   ║ │  ║
║  │  ║  │  │ • Can EVICT any layer at any time (eviction policy)     │  │   ║ │  ║
║  │  ║  │  │ • Can DENY any sharing request (policy)                 │  │   ║ │  ║
║  │  ║  │  │ • Can REVOKE leases early (abuse, overload)             │  │   ║ │  ║
║  │  ║  │  │ • Can SET prices dynamically                            │  │   ║ │  ║
║  │  ║  │  │ • Can BLACKLIST peers                                  │  │   ║ │  ║
║  │  ║  │  │ • Controls: who, what, when, how long                   │  │   ║ │  ║
║  │  ║  │  └──────────────────────────────────────────────────────────┘  │   ║ │  ║
║  │  ║  │                                                                   │  │   ║ │  ║
║  │  ║  │  ┌──────────────────────────────────────────────────────────┐  │   ║ │  ║
║  │  ║  │  │ ISOLATION: Private resources never exposed without policy │  │   ║ │  ║
║  │  ║  │  │                                                               │  │   ║ │  ║
║  │  ║  │  │ • Layers NOT in share_whitelist: never served             │  │   ║ │  ║
║  │  ║  │  │ • Private layers: embedding, norm, head                   │  │   ║ │  ║
║  │  ║  │  │ • Shares only cached copies (not disk-only)               │  │   ║ │  ║
║  │  ║  │  │ • Each peer's cache is physically separate memory        │  │   ║ │  ║
║  │  ║  │  │ • No shared memory across network                         │  │   ║ │  ║
║  │  ║  │  │ • mTLS encryption in transit                              │  │   ║ │  ║
║  │  ║  │  └──────────────────────────────────────────────────────────┘  │   ║ │  ║
║  │  ║  │                                                                   │  │   ║ │  ║
║  │  ║  │  ┌──────────────────────────────────────────────────────────┐  │   ║ │  ║
║  │  ║  │  │ ECONOMIC AUTONOMY: Independent pricing & accounting       │  │   ║ │  ║
║  │  ║  │  │                                                               │  │   ║ │  ║
║  │  ║  │  │ • Sets own price_per_gb (0.0 to ∞)                       │  │   ║ │  ║
║  │  ║  │  │ • Own Ledger account (credits/earnings)                  │  │   ║ │  ║
║  │  ║  │  │ • Dynamic pricing based on load/reputation               │  │   ║ │  ║
║  │  ║  │  │ • Decides when to accept/reject offers                   │  │   ║ │  ║
║  │  ║  │  └──────────────────────────────────────────────────────────┘  │   ║ │  ║
║  │  ║  │                                                                   │  │   ║ │  ║
║  │  ║  └───────────────────────────────────────────────────────────────┘  │   ║ │  ║
║  │  ║                                                                       ║ │  ║
║  │  ╚═══════════════════════════════════════════════════════════════════════╝ │  ║
║  │                                                                             │  ║
║  │  ┌─────────────────────────────────────────────────────────────────────┐   │  ║
║  │  │                      NETWORK INTERFACE                              │   │  ║
║  │  │  ┌─────────────────────────────────────────────────────────────┐   │   │  ║
║  │  │  │ gRPC Server (Port 50051)                                   │   │   │  ║
║  │  │  │ • FetchLayer() - Streaming                                 │   │   │  ║
║  │  │  │ • HasLayer() - Quick check                                 │   │   │  ║
║  │  │  │ • AdvertiseLayers() - Registry updates                    │   │   │  ║
║  │  │  │ • Heartbeat() - Health monitoring                         │   │   │  ║
║  │  │  │                                                              │   │   │  ║
║  │  │  │ Security:                                                   │   │   │  ║
║  │  │  │ • mTLS encryption                                          │   │   │  ║
║  │  │  │ • Client certificate authentication                       │   │   │  ║
║  │  │  │ • Request whitelist enforcement                           │   │   │  ║
║  │  │  │ • Rate limiting per peer_id                               │   │   │  ║
║  │  │  └─────────────────────────────────────────────────────────────┘   │   │  ║
║  │  │                                                                     │   │  ║
║  │  │  ┌─────────────────────────────────────────────────────────────┐   │   │  ║
║  │  │  │ PeerRegistry Client                                         │   │   │  ║
║  │  │  │ • Bootstrap: 10.0.0.1:50050 (central) or static list       │   │   │  ║
║  │  │  │ • Advertises: {"peer_id": "A", "layers": ["0-19"], ...}   │   │   │  ║
║  │  │  │ • Queries: "Who has layer.20?" → [PeerInfo(B), PeerInfo(D) │   │   │  ║
║  │  │  └─────────────────────────────────────────────────────────────┘   │   │  ║
║  │  │                                                                     │   │  ║
║  │  └─────────────────────────────────────────────────────────────────────┘   │  ║
║  │                                                                             │  ║
║  └─────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════╝

                               │
                               │ network (LAN/WAN)
                               │ latency: 0.1-50ms
                               │ bandwidth: 1Gbps-100Gbps
                               │ faults: drops, partitions
                               ▼

╔═══════════════════════════════════════════════════════════════════════════════════╗
║                                   PEER B                                          ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐  ║
║  │                    INDEPENDENT SOVEREIGN SYSTEM                            │  ║
║  │  • Own control plane (decides what to share)                              │  ║
║  │  • Own resources (GPU, RAM, SSD)                                          │  ║
║  │  • Own ledger account (credits)                                          │  ║
║  │  • Own sharing policy (may differ from Peer A)                           │  ║
║  │  • Own reputation score                                                  │  ║
║  │  • May be in different timezone, owned by different entity               │  ║
║  └─────────────────────────────────────────────────────────────────────────────┘  ║
║                                                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Modularity Dimensions

### 2.1 Horizontal Modularity (Between Peers)

Each peer is a **completely independent process** (even separate machine):

```
Peer A Process Space                Peer B Process Space
╔═══════════════════╗               ╔═══════════════════╗
║   Peer A          ║               ║   Peer B          ║
║ ┌───────────────┐ ║               ║ ┌───────────────┐ ║
║ │ EvoLLM Model  │ ║               ║ │ EvoLLM Model  │ ║
║ └───────────────┘ ║               ║ └───────────────┘ ║
║ ┌───────────────┐ ║               ║ ┌───────────────┐ ║
║ │ LayerCache    │ ║               ║ │ LayerCache    │ ║
║ │  • layer.0-19 │ ║               ║ │  • layer.20-39│ ║
║ └───────────────┘ ║               ║ └───────────────┘ ║
║ ┌───────────────┐ ║               ║ ┌───────────────┐ ║
║ │ PeerServer    │ ║               ║ │ PeerClient    │ ║
║ └───────────────┘ ║               ║ └───────────────┘ ║
║ ┌───────────────┐ ║               ║ ┌───────────────┐ ║
║ │ Ledger (local)│ ║               ║ │ Ledger (local)│ ║
║ │  balance: 15  │ ║               ║ │  balance: -3  │ ║
║ └───────────────┘ ║               ║ └───────────────┘ ║
╚═══════════════════╝               ╚═══════════════════╝

         │                                   │
         │ gRPC over TCP/IP                  │ gRPC over TCP/IP
         │ encryption: TLS                   │ encryption: TLS
         │ auth: mTLS certs                  │ auth: mTLS certs
         ▼                                   ▼
    ╔═══════════════╗                ╔═══════════════╗
    ║   Network     ║                ║   Network     ║
    ║  (switch/     ║                ║  (switch/     ║
    ║   router)     ║                ║   router)     ║
    ╚═══════════════╝                ╚═══════════════╝
```

**Key Isolation Properties:**
- **No shared memory**: Each peer's cache is physically separate
- **No shared filesystem**: Layers copied over network (or memory-mapped locally)
- **Independent failure**: Peer B crash doesn't affect Peer A (except missing layers)
- **Independent evolution**: Peer A can upgrade software without affecting Peer B
- **Independent configuration**: Each peer sets own cache sizes, policies, prices

---

### 2.2 Vertical Modularity (Within a Peer)

```
┌─────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER                         │
│  EvoLLMModel.forward()                                      │
│  - Calls resource_manager.get_layer()                       │
│  - Doesn't know if layer is local or remote                │
│  - Transparent integration                                 │
└─────────────────────────────────────────────────────────────┘
                            │ uses
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                 RESOURCE ABSTRACTION LAYER                  │
│  ResourceProvider (ABC)                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ LocalResourceManager                               │   │
│  │   → wraps TensorCacheManager (GPU/CPU/SSD)        │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ PeerClientResourceManager                         │   │
│  │   → queries registry, fetches from peers          │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ PeerServerResourceManager                         │   │
│  │   → exposes local cache as remote service         │   │
│  ├─────────────────────────────────────────────────────┤   │
│  │ HybridResourceManager                             │   │
│  │   → composes multiple providers with policy       │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │ delegates to
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              CACHE & NETWORK IMPLEMENTATIONS                │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │ LayerCache  │  │ PeerClient  │  │ PeerServer       │   │
│  │  (LRU)      │  │  (gRPC)     │  │  (gRPC)          │   │
│  └─────────────┘  └─────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                EXTERNAL DEPENDENCIES                        │
│  • AirLLM (load_layer)                                     │
│  • PyTorch (tensor operations)                            │
│  • grpcio (network RPC)                                   │
│  • psutil (hardware detection)                            │
└─────────────────────────────────────────────────────────────┘
```

**Vertical Modularity Benefits:**
- **Swap implementations**: Replace LayerCache with different policy without touching model
- **Testability**: Mock ResourceProvider for unit tests
- **Extensibility**: Add new provider types (e.g., Redis cache) without modifying core
- **Single Responsibility**: Each layer has clear purpose

---

## 3. Isolation Boundaries & Security

### 3.1 Trust Boundary Diagram

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                               TRUST DOMAIN 1                                 │
│                              (Your Network)                                   │
│                                                                               │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │   Peer A    │    │   Peer B    │    │   Peer C    │                      │
│  │             │    │             │    │             │                      │
│  │ • Full trust│◄──►│ • Full trust│◄──►│ • Full trust│                      │
│  │   in self   │    │   in peers  │    │   in peers  │                      │
│  │             │    │             │    │             │                      │
│  │ Ledger:     │    │ Ledger:     │    │ Ledger:     │                      │
│  │  balance=15 │    │  balance=-3 │    │  balance=8  │                      │
│  │ Rep: 0.92   │    │ Rep: 0.88   │    │ Rep: 0.95   │                      │
│  └─────────────┘    └─────────────┘    └─────────────┘                      │
│         │                  │                  │                             │
│         │ mTLS encrypted   │ mTLS encrypted   │ mTLS encrypted              │
│         └──────────────────┼──────────────────┘                             │
│                            │                                                │
│                            ▼                                                │
│                 ╔═══════════════════════╗                                   │
│                 ║   CERTIFICATE AUTHORITY║                                   │
│                 ║   (Self-signed dev)   ║                                   │
│                 ╚═══════════════════════╝                                   │
│                            │                                                │
│                            │ verifies                                      │
│                            ▼                                                │
│  ┌─────────────────────────────────────────────────────────────────┐         │
│  │  Only peers with certs signed by this CA can join               │         │
│  │  Certificate includes: peer_id, role, expiry                    │         │
│  └─────────────────────────────────────────────────────────────────┘         │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐         │
│  │                    Registry (Central or DHT)                   │         │
│  │                                                                 │         │
│  │  Metadata only - NO SENSITIVE DATA                             │         │
│  │  • peer_id → address mapping                                   │         │
│  │  • layer_name → [peer_id, checksum]                           │         │
│  │  • peer metrics (latency, load, reputation)                   │         │
│  │                                                                 │         │
│  │  NOT a trust anchor - just a directory                        │         │
│  └─────────────────────────────────────────────────────────────────┘         │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘

                                   │
                                   │ outside trust boundary
                                   ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                         UNTRUSTED (Internet)                                 │
│  • No direct access to peers                                                  │
│  • Must go through firewall/NAT                                               │
│  • Registry may be public (read-only)                                        │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Isolation Guarantees**:
1. **No shared memory**: Each peer's RAM is private
2. **No shared filesystem**: Data transferred over network
3. **No privilege escalation**: mTLS ensures身份; cannot impersonate
4. **No data leakage**: Only layers explicitly shared are accessible
5. **No cross-contamination**: Process A cannot access Process B's memory

---

## 4. Network Performance Characteristics

### 4.1 Latency Breakdown

```
Scenario: Peer A needs layer.20, cached on Peer B
Location: Same datacenter, 10GbE network

Timeline (milliseconds):
┌──────────────────────────────────────────────────────────────────────────────┐
│ Event                              │ Time (ms) │ Cumulative (ms)             │
├─────────────────────────────────────┼───────────┼────────────────────────────┤
│ 1. Local cache miss detected       │    0      │  0                          │
│ 2. Query registry (local cache)    │    1      │  1                          │
│ 3. Rank peers (CPU compute)        │    0.5    │  1.5                        │
│ 4. Get gRPC channel (pooled)       │    0      │  1.5  (already connected)  │
│ 5. Send FetchLayer request         │    0.5    │  2.0                        │
│ 6. Network propagation (1ms RTT)   │    1      │  3.0                        │
│ 7. Peer B receives & authorizes    │    0.5    │  3.5                        │
│ 8. Read from Peer B's RAM cache    │    0.1    │  3.6  (memory speed)       │
│ 9. Stream chunks (1.8GB @ 10Gbps)  │  1440     │ 1443.6  (bulk transfer)    │
│10. Network propagation back        │    1      │ 1444.6                     │
│11. Receive & reassemble            │    0.5    │ 1445.1                     │
│12. Verify checksum (SHA256)        │   500     │ 1945.1  (CPU bound!)       │
│13. Add to local cache              │    0.1    │ 1945.2                     │
│14. Move to GPU (PCIe 3.0 x16)      │    50     │ 1995.2                     │
│15. Run layer computation           │   150     │ 2145.2                     │
└─────────────────────────────────────┴───────────┴────────────────────────────┘

Total time: ~2145ms

Breakdown by category:
  • Network: 2.5ms (0.12%) - negligible for small messages
  • Data transfer: 1440ms (67%) - 1.8GB over 10Gbps = 1440s
  • Checksum verification: 500ms (23%) - hashing 1.8GB on CPU
  • GPU compute: 150ms (7%)
  • Other: 50ms (2.3%)

Comparison: Local SSD load (no peer)
  • SSD read 1.8GB @ 500MB/s = 3600ms
  • Move to GPU: 50ms
  • Compute: 150ms
  Total: ~3800ms

Speedup: 2145ms vs 3800ms = 1.77× faster

BUT: If we cache checksums and trust peer's advertised hash:
  • Skip step 12 (500ms saved)
  New total: 1645ms → 2.31× speedup

Key insight: Network transfer can be FASTER than SSD if:
  • Peers have fast RAM (DDR4/DDR5 ~50GB/s vs SSD 0.5-3GB/s)
  • Network is high-speed (10Gbps+ = 1.25GB/s)
  • Layer size < 2GB
  • Transfer time < SSD read time

Crossover point:
  SSD_read_time ≈ network_transfer_time
  2 seconds (typical 1GB on NVMe) = L / bandwidth
  L = 2s × 1.25GB/s = 2.5GB

So for layers > 2.5GB, network could be slower!
But typical layer size: 1-2GB → network wins if LAN is fast.
```

### 4.2 Network Topology Impact

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NETWORK TOPOLOGY EFFECTS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Topology 1: Same Rack (1ms RTT, 40Gbps)                                  │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
│  Latency: ~1ms                                                            │
│  Bandwidth: 40 Gbps (5 GB/s effective ~4 GB/s)                           │
│  1.8GB transfer: 450ms                                                    │
│  Total fetch (including hash): 450 + 500 + overhead ≈ 1000ms              │
│  vs SSD (2000ms) → 2× speedup                                            │
│                                                                             │
│  Topology 2: Same Datacenter (5ms RTT, 10Gbps)                            │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
│  Latency: 5ms RTT                                                         │
│  Bandwidth: 10 Gbps (1.25 GB/s)                                          │
│  1.8GB transfer: 1440ms                                                  │
│  Total fetch: ~2000ms                                                     │
│  vs SSD (2000ms) → ≈ same speed (break-even)                            │
│                                                                             │
│  Topology 3: Cross-region (50ms RTT, 1Gbps)                               │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
│  Latency: 50ms RTT                                                        │
│  Bandwidth: 1 Gbps (125 MB/s)                                            │
│  1.8GB transfer: 14400ms (14.4 seconds!)                                 │
│  Total fetch: ~14500ms                                                    │
│  vs SSD (2000ms) → 7× slower!                                             │
│  Conclusion: Cross-region P2P is BAD for large layers                    │
│                                                                             │
│  Topology 4: Peer on same host (localhost)                                │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│
│  Latency: ~0.1ms (kernel shared memory)                                  │
│  Bandwidth: ∞ (memory copy ~10-20GB/s)                                   │
│  1.8GB transfer via memcpy: 90ms                                         │
│  Total fetch: ~590ms (with hash)                                         │
│  vs SSD: 7× faster!                                                      │
│  Use case: Multi-process on same machine (NUMA optimization)             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Performance implications**:
- **LAN clustering** (same rack/datacenter): Peer fetch competitive with or better than SSD
- **WAN** (cross-region): Too slow for large layers (use only for hot layers <100MB)
- **Same host**: Very fast, useful for multi-process coordination

---

### 4.3 Bandwidth Utilization & Chunking

```
Layer Transfer with Streaming:

Peer B sends 1.8GB layer in chunks:
┌────────────────────────────────────────────────────────────────┐
│ Chunk 1: 10 MB  ──┐                                              │
│ Chunk 2: 10 MB  ──┼──> TCP packets over network                 │
│ Chunk 3: 10 MB  ──┘                                              │
│ ...                                                        │
│ Chunk 180: 10 MB                                                │
├────────────────────────────────────────────────────────────────┤
│ Stream progress: 0% → 10% → 20% → ... → 100%                   │
│                                                                 │
│ Parallel streams: Can fetch from multiple peers concurrently:  │
│   Fetch layer.20 from Peer B (1.8GB)                           │
│   Prefetch layer.21 from Peer C (1.8GB)                        │
│   Load layer.22 from local disk (in parallel!)                │
│                                                                 │
│ Bandwidth sharing:                                              │
│   If network link = 10 Gbps (1.25 GB/s)                        │
│   Single fetch: 1.8GB @ 1.25 GB/s = 1.44s                     │
│   3 parallel: each gets ~0.42 GB/s → 4.3s each                 │
│   Total wall time for all 3: ~4.3s (vs 1.44×3=4.32s sequential)│
│   Benefit: pipeline, not parallelize                          │
└────────────────────────────────────────────────────────────────┘

Optimal strategy:
  1. Start fetching next layer BEFORE finishing current
  2. Overlap network I/O with GPU compute
  3. If prefetch_depth=3, have 3 fetches in flight
  4. Network utilization: ~constant (no idle time)
```

---

## 5. Performance Tuning Parameters

### 5.1 Client-Side Tuning (Requester)

```yaml
# EvoLLMConfig.peer settings affecting performance:

prefetch_from_peers: true        # Enable async peer prefetch
max_peers_to_query: 3            # Parallel fetch from N peers
fetch_timeout_ms: 5000           # How long to wait per peer
fetch_retries: 2                 # Retries on failure
prefetch_depth: 3                # Lookahead layers to prefetch
prefetch_batches: 2              # Thread pool size for prefetch

# Ranking weights (customize for your cluster):
peer_ranking_weights:
  reputation: 2.0    # Prioritize reliable peers
  latency: 1.0       # Prefer low-latency
  price: 0.5         # Cost sensitivity
  load: 0.5          # Avoid overloaded
  local_affinity: 0.3 # Prefer peers with other cached layers

# Cache policy for borrowed layers:
borrowed_layer_ttl: 3600        # Keep borrowed layers for 1 hour max
borrowed_layer_max_uses: 10     # Evict after 10 accesses (prevents hoarding)
cache_borrowed_layers: true     # Yes, cache them locally for reuse
```

### 5.2 Server-Side Tuning (Provider)

```yaml
# PeerServerConfig affecting performance & isolation:

sharing_enabled: true
max_share_cache_percent: 0.5    # Reserve 50% for local use
max_concurrent_serves: 10       # Max simultaneous fetches
serve_rate_limit_per_peer: 1    # Max GB/s per requesting peer
serve_timeout_s: 300            # Max time per fetch

# Pricing strategy:
pricing:
  mode: 'static'                # or 'dynamic'
  price_per_gb: 0.1             # 0.1 FIT/GB
  dynamic:
    base_price: 0.05
    load_multiplier: true       # Price × (1 + load_score)
    reputation_discount: true   # High rep peers get discount

# Resource QoS:
qos:
  bandwidth_per_peer_mb_s: 500  # Throttle per-request bandwidth
  cpu_limit_percent: 20         # Max CPU for serving (leave for local)
  io_priority: 'low'            # Nice level for serving threads
```

---

## 6. Isolation Guarantees & Failure Modes

### 6.1 Isolation Guarantees

```
Guarantee                          │ Mechanism
───────────────────────────────────┼─────────────────────────────────────────────
No memory sharing                 │ Each peer separate process/machine
No filesystem sharing             │ Data copied over network
Independent failure domains       │ Peer crash → other peers continue
Config isolation                  │ Each peer's config independent
Data isolation                   │ Only shared data is explicitly served
Performance isolation           │ Rate limiting per requester
Economic isolation             │ Independent ledger accounts
Failure recovery isolation     │ Peer B failure doesn't affect Peer A's cache
```

### 6.2 Failure Modes & Recovery

```
Failure Scenario 1: Peer B crashes mid-fetch
┌────────────────────────────────────────────────────────────────┐
│ Peer A → Peer B: FetchLayer("layer.20", ...)                  │
│ │                                                              │
│ │ Peer B process dies ⚡                                        │
│ │                                                              │
│ ← Connection reset (RST)                                       │
│                                                                 │
│ Peer A detects: ConnectionError                              │
│ → Marks Peer B as unhealthy (circuit breaker)               │
│ → Removes Peer B from eligible pool for 5 minutes           │
│ → Tries next peer in ranking (Peer D)                       │
│ → Fetch continues from Peer D                               │
│                                                                 │
│ Impact: 1-2 second delay, automatic recovery                │
└────────────────────────────────────────────────────────────────┘

Failure Scenario 2: Peer B slow (disk thrashing)
┌────────────────────────────────────────────────────────────────┐
│ Peer A fetches from Peer B                                    │
│ Layer transfer rate: 10 MB/s (slow)                          │
│ 1.8GB takes 180 seconds (!)                                 │
│                                                                 │
│ Peer A monitoring:                                            │
│ • Tracks per-peer latency in PeerInfo                        │
│ • After fetch: Peer B latency updated to 180s                │
│ • Ranking algorithm: latency score ↓ → Peer B ranked lower  │
│ Future requests: Prefer Peer D (fast) over Peer B (slow)    │
│                                                                 │
│ Impact: Degraded performance, self-correcting over time     │
└────────────────────────────────────────────────────────────────┘

Failure Scenario 3: Registry down (centralized mode)
┌────────────────────────────────────────────────────────────────┐
│ Peer A needs layer.20, queries registry                      │
│ → Registry HTTP 503                                           │
│                                                                 │
│ Peer A behavior:                                              │
│ • Uses last known registry cache (stale but functional)     │
│ • Tries to fetch from peers in cache                         │
│ • If none have it → fallback to local disk                  │
│ • Retries registry every 30s                                 │
│                                                                 │
│ Impact: Reduced peer discovery, fallback to disk ok         │
└────────────────────────────────────────────────────────────────┘

Failure Scenario 4: Peer B sends corrupt data (malicious)
┌────────────────────────────────────────────────────────────────┐
│ Peer A receives layer.20 from Peer B                         │
│ Checksum verification:                                        │
│   expected = model_manifest["layer.20"] // from local copy  │
│   received = sha256(received_data)                          │
│   if received != expected:                                   │
│       → Log security alert                                  │
│       → Mark Peer B as compromised (reputation → 0)         │
│       → Blacklist Peer B (circuit breaker permanent)       │
│       → Evict any cached layers from Peer B                 │
│       → Notify registry of bad checksum                     │
│                                                                 │
│ Impact: Security incident contained, no corruption           │
└────────────────────────────────────────────────────────────────┘

Failure Scenario 5: Network partition
┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   Peer A     │────────►│   Peer B     │────────►│   Peer C     │
│              │         │              │         │              │
│ Partition!   │  ✗✗✗   │              │  ✓✓✓   │              │
└──────────────┘         └──────────────┘         └──────────────┘
                                                     │
                                                     ▼
                                          Network link ok

Peer A isolated from B and C:
  • Peer A: Can still fetch from any remaining peers (D, E, ...)
  • Peer A: Will eventually mark B, C as unhealthy (no heartbeat)
  • Peer B & C: Can still serve each other and fetch from others
  • Cluster continues with subset

Impact: Partial outage, graceful degradation
Recovery: When partition heals, peers re-sync registry, resumes
```

---

## 7. Performance Comparison Table

### 7.1 Latency (per layer fetch)

| Scenario | SSD Read | Peer Fetch (same rack) | Peer Fetch (same DC) | Peer Fetch (WAN) |
|----------|----------|-----------------------|---------------------|------------------|
| 100 MB   | 200 ms   | 80 ms  ✅             | 150 ms ✅           | 5000 ms ❌       |
| 500 MB   | 1000 ms  | 300 ms ✅             | 900 ms ✅           | 25000 ms ❌      |
| 1800 MB  | 3600 ms  | 1000 ms ✅            | 2000 ms ≈           | 90000 ms ❌      |
| 5000 MB  | 10000 ms | 3000 ms ✅            | 6000 ms ❌          | 250000 ms ❌     |

**Legend**:
- ✅ Peer fetch faster than SSD
- ≈ Similar performance
- ❌ Peer fetch slower than SSD

**Crossover point**: ~2.5GB on 10Gbps network (1.25 GB/s)

### 7.2 Throughput (tokens/sec) - 70B Model

```
Configuration:
  • 80 layers total
  • Hot layers (0-20): cached by all peers (redundancy)
  • Cold layers (21-79): cached by some peers

Scenario A: Single node, 32GB cache (20 layers cached)
  • Cache hit rate: 25% (20/80)
  • Disk reads/token: 60
  • Disk read time/token: 60 × 200ms = 12 seconds ❌
  • Compute time/token: 0.1 seconds
  • Total: ~12 seconds/token → 0.08 tokens/sec

Scenario B: Single node with 32GB cache AND P2P (4 peers)
  • Each peer caches 20 layers, collectively 80 layers
  • Cache hit rate: 95% (assume layering)
  • Network fetches/token: 4
  • Network fetch time: 4 × 200ms = 0.8 seconds
  • Compute: 0.1s
  • Total: ~0.9s/token → 1.1 tokens/sec ✅ 14× speedup

Scenario C: 4-node cluster with compute offload (Phase 3)
  • Some layers computed on remote GPUs (faster)
  • Reduced network (only send activations, not weights)
  • Throughput: 3-5 tokens/sec (GPU parallelization)
```

**Key insight**: P2P mainly improves **cold cache scenarios**. If you already have all layers cached locally, no benefit. But for limited RAM, P2P dramatically expands effective cache.

---

## 8. Modularity & Isolation in Code

### 8.1 Module Dependencies

```
evollm/
│
├── __init__.py
│   └── exports: EvoLLMModel, EvoLLMConfig, AutoModel
│
├── config.py
│   └── EvoLLMConfig, PeerBackendConfig
│       └── no dependencies on network modules
│
├── resource_provider.py          (Phase 2)
│   ├── ResourceProvider (ABC)
│   ├── LocalResourceManager
│   ├── PeerClientResourceManager
│   ├── PeerProviderResourceManager
│   └── HybridResourceManager
│       └── depends on: peer.client, peer.server (interfaces only)
│
├── cache_policy.py               (unchanged Phase 1)
│   ├── LayerCache, TensorCacheManager
│   └── Used by LocalResourceManager (wrapped)
│
├── tensor_loader.py              (modified)
│   └── HierarchicalTensorLoader(resource_manager: ResourceProvider)
│       └── no direct dependency on peer modules
│
├── evollm_base.py                (modified)
│   └── EvoLLMModel
│       └── creates ResourceManager via factory
│
├── peer/
│   ├── __init__.py
│   ├── client.py                 (Phase 2)
│   │   ├── PeerRegistry
│   │   ├── PeerLayerFetcher
│   │   └── Exceptions
│   │       └── depends on: grpc, protocol_pb2
│   │
│   ├── server.py                 (Phase 2)
│   │   ├── PeerServer
│   │   ├── PeerServerConfig
│   │   └── Servicer (implements protocol)
│   │       └── depends on: cache_policy (for local cache), ledger
│   │
│   ├── ledger.py                 (Phase 2)
│   │   ├── Ledger (in-memory, HTTP, SQLite)
│   │   ├── ReputationManager
│   │   └── Account
│   │       └── independent (could be extracted to separate package)
│   │
│   ├── security.py               (Phase 2)
│   │   ├── mTLS certificate utilities
│   │   ├── Whitelist enforcement
│   │   └── Authentication middleware
│   │
│   └── protocol.proto            (Phase 2)
│       └── gRPC service definition
│           └── generates: protocol_pb2.py, protocol_pb2_grpc.py
│
└── docs/
    ├── peer_analytics.md
    ├── resource_forwarding_architecture.md
    └── modularity_isolation_performance.md  ← this file
```

**Dependency Direction** (acyclic):
```
config ← resource_provider ← peer.client
    ↑                        ↑
    └── evollm_base ← tensor_loader ───────────┘
                          ↑
                    cache_policy (Phase 1, unchanged)
```

**Clean separation**: No circular dependencies. Peer modules only depend on interfaces (ResourceProvider ABC), not on concrete implementations in other peers.

---

## 9. Isolation Verification

### Can Peer A...

| Action | Allowed? | Mechanism |
|--------|----------|-----------|
| Read Peer B's RAM directly | ❌ NO | No shared memory; must go through gRPC API |
| Write to Peer B's filesystem | ❌ NO | No NFS; only streaming API |
| Execute code on Peer B without permission | ❌ NO | mTLS + authorization required |
| Evict Peer B's cached layers | ❌ NO | Each peer controls own cache |
| Impersonate Peer B | ❌ NO | mTLS certificates signed by CA |
| Access Peer B's ledger account | ❌ NO | Separate ledger instances/partitioning |
| Bypass Peer B's sharing policy | ❌ NO | Policy enforced at server per-request |
| Ping Peer B's gRPC server | ✅ YES | But may be rate-limited |
| Query Peer B's advertised layers | ✅ YES | Via registry (public metadata) |
| Fetch layer from Peer B (if allowed) | ✅ YES | Through authorized gRPC call |
| Revoke own lease early | ✅ YES | Control channel: Release(lease_id) |
| Renew lease before expiry | ✅ YES | Renew(lease_id, duration) |
| Transfer lease to another peer | ⚠️ OPTIONAL | Requires protocol extension |
| Monitor own usage/stats | ✅ YES | get_stats() on resource_manager |

**Bottom line**: Strong isolation - peers can only interact through well-defined APIs with authentication/authorization. No "break glass" access.

---

## 10. Performance Optimization Opportunities

### 10.1 Reduce Network Overhead

```
Current: Send entire 1.8GB layer every fetch
Optimization 1: Differential transfer
  • Peer A: has layer.20 v1 (checksum ABC)
  • Peer B: has layer.20 v2 (checksum DEF)
  • Instead of sending full layer, send delta
  • Requires version control system (git-style) for layer weights
  • Complexity: high, benefit: moderate (if versions seldom differ)

Optimization 2: Compression
  • Use lz4 or zstd on layer tensors before sending
  • Compression ratio: 2-4× for fp16 weights (redundant data)
  • CPU overhead: ~10% of decompression time
  • Net improvement: 2-4× faster transfer for large layers
  • Implementation: Add optional compression to gRPC

Optimization 3: Caching serialized blobs
  • Peer B: after first fetch, serialized layer cached in RAM
  • Subsequent fetches: stream from cached bytes (no re-serialization)
  • Saves CPU on serving peer
```

### 10.2 Reduce Checksum Cost

```
Current: SHA256 1.8GB = ~500ms on single core
Optimization 1: Fast hash (BLAKE3, xxHash)
  • BLAKE3: 5-10× faster than SHA256, still cryptographically strong
  • 1.8GB in ~50-100ms ✅
  • Change: compute_checksum() uses blake3

Optimization 2: Checksum caching
  • When Peer B advertises layer, compute checksum once
  • Store with layer in cache: entry.checksum = "blake3:..."
  • Fetching: Peer A already knows expected checksum (from registry)
  • Verification: still hash received data to ensure integrity
  • But: can verify as chunks arrive (pipeline) instead of all-at-once

Optimization 3: Hardware acceleration
  • Intel SHA extensions: SHA256 ~2× faster
  • NVIDIA GPU: can hash on GPU (100GB/s+) if available
  • Not worth complexity unless profiling shows hash is bottleneck
```

### 10.3 Overlap Compute & Communication

```
Current sequential:   [fetch layer] → [compute layer] → [fetch next] → ...
Optimal pipelined:    [fetch N+1] [compute N] [fetch N+2] [compute N+1] ...

Implementation:
  • Prefetch_depth=3 means always fetching 3 layers ahead
  • While computing layer i, concurrently fetching layer i+1, i+2, i+3
  • Hides network latency behind compute

Example timeline with prefetch_depth=3:
  t=0ms:  Start compute layer.17 (local)
  t=10ms: Start fetch layer.18 (from Peer B) ─┐
  t=20ms: Start fetch layer.19 (from Peer C) ─┼─ 3 concurrent fetches
  t=30ms: Start fetch layer.20 (from Peer D) ─┘
  t=150ms: Compute layer.17 done
  t=151ms: Start compute layer.18 (just arrived ✅)
  t=250ms: Compute layer.18 done, layer.19 ready ✅
  t=350ms: Compute layer.19 done, layer.20 ready ✅
  t=450ms: Compute layer.20 done, layer.21 fetching...

Without prefetch:
  [fetch 18: 200ms][compute 18: 100ms][fetch 19: 200ms]...
  Total for 3 layers: 3×300 = 900ms

With prefetch:
  Fetch all overlapped: max(fetch_time) ≈ 200ms
  Compute: 3×100ms = 300ms
  Total: 500ms (1.8× faster!)

Key: Network bandwidth shared but compute continues.
Assumes: prefetch_depth * network_latency < compute_time_per_layer
```

---

## 11. Summary & Key Takeaways

### Modularity

1. **Horizontal**: Each peer is a standalone process/machine
   - No shared memory
   - Independent lifecycle
   - Private resources

2. **Vertical**: Clear abstraction layers
   - `ResourceProvider` interface
   - Pluggable implementations
   - Minimal coupling

3. **Deployment**: Can mix and match
   - Some peers: Server only (lend)
   - Some peers: Client only (borrow)
   - Some peers: Hybrid (both)
   - All interoperate via standard protocol

### Isolation

1. **Security boundaries**: mTLS, whitelists, per-request authorization
2. **Resource boundaries**: Each peer's cache is private
3. **Failure isolation**: Peer crash doesn't cascade
4. **Config isolation**: Each peer sets own policies
5. **Economic isolation**: Separate ledger accounts

### Performance

1. **Network matters**: LAN (10ms, 10Gbps) is acceptable; WAN is not for large layers
2. **Crossover point**: ~2.5GB layer size where network = SSD on 10Gbps
3. **Prefetch is critical**: Overlap network I/O with compute to hide latency
4. **Checksum cost**: Significant (~500ms/GB) → optimize with fast hash
5. **Compression helps**: 2-4× bandwidth reduction, modest CPU cost

### Architecture Verdict

This design achieves:
- ✅ **Modularity**: Clean abstractions, no circular dependencies
- ✅ **Isolation**: Strong security boundaries, independent failure domains
- ✅ **Scalability**: Add more peers → more aggregate cache (linear)
- ✅ **Performance**: LAN peer fetch can match or beat SSD for <2.5GB layers
- ✅ **Autonomy**: Each peer makes own decisions, no central coordinator
- ✅ **Economics**: Credits incentivize sharing while preventing tragedy of commons

**Trade-offs**:
- ⚠️ Network dependence: If all peers down, fallback to slow SSD
- ⚠️ Configuration complexity: Many knobs to tune
- ⚠️ Security overhead: mTLS, certificates, ACLs
- ⚠️ Consistency model: Eventual (not strict), relies on checksums

**Overall**: The architecture successfully balances modularity, isolation, and performance for a distributed LLM inference system. The peer abstraction is clean, the isolation is strong, and the performance is acceptable for LAN-scale deployments.

---

## Next: Implementation Readiness

The modularity and isolation analysis confirms the design is **implementation-ready**:

1. **Phase 1**: Complete the single-node module tests (foundation)
2. **Phase 2**: Implement ResourceProvider abstraction (enables P2P)
3. **Phase 2**: Build gRPC client/server with security
4. **Phase 2**: Integrate and test 2-node cluster
5. **Phase 3**: Extend to general resources (compute, shell)

All while maintaining:
- Modular boundaries (no circular deps)
- Isolation (mTLS, policies)
- Performance (prefetch, caching, compression)

Ready to code? 🚀
