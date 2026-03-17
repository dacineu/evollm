# Generalized Resource Sharing: ANY System Object

**Question**: Can EvoLLM/EvoOS architecture share ANY system object?
- Keyboard? ✅ (with caveats)
- Video console/display? ✅ (with caveats)
- Running application/process? ✅
- Files/directories? ✅
- GPU? ✅
- Network ports? ✅
- Audio devices? ✅
- Sensors (camera, microphone)? ✅

**Answer**: **Yes**, but resources fall into **categories** with different sharing models.

---

## 1. Resource Taxonomy & Sharing Models

### Classification by Access Pattern

```
Resource
├── DATA RESOURCES (Passive, storage-like)
│   ├── File/Directory (read/write)
│   ├── Block device (disk partition)
│   ├── Memory region (shared memory segment)
│   └── Database (SQL/NoSQL)
│       Sharing model: Stream/transfer copies, or remote mount
│
├── COMPUTE RESOURCES (Active, execution)
│   ├── Process (running application)
│   ├── Container (Docker/Pod)
│   ├── Function (serverless)
│   ├── GPU compute context
│   └── FPGA bitstream
│       Sharing model: Execute remotely, results returned
│
├── INTERACTIVE RESOURCES (Bidirectional streams)
│   ├── Keyboard/input events
│   ├── Display/framebuffer
│   ├── Audio input/output
│   ├── Pointer/mouse
│   ├── Touchscreen
│   └── Haptic feedback
│       Sharing model: Stream I/O events bidirectionally
│
├── HARDWARE DEVICES (Exclusive or time-shared)
│   ├── GPU (VRAM + compute units)
│   ├── CPU cores (with cache isolation)
│   ├── TPU/NPU/AI accelerator
│   ├── Network interface (NIC, with queue)
│   ├── Disk I/O queue
│   ├── Camera (video capture)
│   ├── Microphone (audio capture)
│   └── Serial port (UART, GPIO)
│       Sharing model: Time-division multiplexing, scheduling
│
└── VIRTUAL RESOURCES (Namespaces, abstractions)
    ├── PID namespace (share process tree)
    ├── Network namespace (share network stack)
    ├── User namespace (share UID/GID mapping)
    ├── IPC namespace (share message queues)
    └── Mount namespace (share filesystem view)
        Sharing model: Namespace join/leaving
```

---

## 2. Sharing Models by Category

### Model A: COPY & STREAM (Data Resources)

**Resources**: Files, directories, database query results, memory blobs

**Mechanism**:
```
Peer A (owner)                          Peer B (consumer)
    │                                      │
    │ 1. Request: "Give me file /path/X"  │
    ├─────────────────────────────────────>│
    │                                      │
    │ 2. Stream file contents             │
    │    (chunk by chunk)                 │
    │    ┌─────────────────────────────┐  │
    │    │ chunk0, chunk1, chunk2...  │  │
    │    └─────────────────────────────┘  │
    │◄────────────────────────────────────┤
    │                                      │
    │ 3. B receives, writes to local file │
    │    (or processes in memory)         │
    │                                      │
    ledger: debit(B, size * price)        │
```

**Characteristics**:
- ✅ Simple, stateless
- ✅ No consistency issues (copy is independent)
- ❌ Duplicate storage (wastes space)
- ✅ No interference (writer doesn't affect original)
- ✅ Can cache indefinitely on consumer side

**Example**: Sharing a dataset file
```python
lease = provider.allocate(ResourceSpec(
    resource_type=ResourceType.FILE,
    path="/datasets/imagenet/train",
    mode=READ_ONLY,
    duration_s=3600
))
# lease.endpoint = "grpc://peer-a:50061/files/download?path=/datasets/imagenet/train"
# Stream to local:
for chunk in download(lease.endpoint):
    local_file.write(chunk)
```

**EvoLLM use case**: Sharing model checkpoint files (already implemented)

---

### Model B: REMOTE EXECUTION (Compute Resources)

**Resources**: Processes, containers, GPU kernels, functions

**Mechanism**:
```
Peer A (owner)                          Peer B (consumer)
    │                                      │
    │ 1. Request: "Run Python script X"  │
    ├─────────────────────────────────────>│
    │                                      │
    │ 2. Provider creates subprocess      │
    │    - Sets up environment            │
    │    - Allocates resources (CPU, RAM) │
    │    - Returns execution context      │
    │                                      │
    │ 3. Stream I/O bidirectionally       │
    │    stdin  ──────────────┐            │
    │    stdout ◄─────────────┤            │
    │    stderr ◄─────────────┤            │
    │                                      │
    │ 4. When done: return exit code      │
    │    + resource usage metrics         │
    │◄────────────────────────────────────┤
    │                                      │
    ledger: debit(B, CPU_time * price)    │
```

**Characteristics**:
- ✅ No local resource usage (everything remote)
- ✅ Access to software only installed on provider
- ❌ Network latency for I/O
- ✅ Strong isolation (subprocess can be sandboxed)
- ⚠️ Hard to checkpoint/migrate (but possible with CRIU)

**Example**: Remote Python execution (like xterm)
```python
lease = provider.allocate(ResourceSpec(
    resource_type=ResourceType.PROCESS,
    command=["python", "train.py"],
    env={"CUDA_VISIBLE_DEVICES": "0"},
    stdin=True,
    stdout=True,
    duration_s=None  # until completion
))
# lease.endpoint = "grpc://peer-a:50062/exec"
# Interactive:
session = grpc.channel(lease.endpoint)
session.stdin.write(b"print('Hello')\n")
output = session.stdout.read(1024)
```

**EvoLLM use case**: Your xterm example! Allocate shell session on remote peer.

---

### Model C: PROXY / FORWARDING (Interactive Resources)

**Resources**: Keyboard, display, audio, mouse, haptic

**Mechanism**:
```
User on Peer B                        Display on Peer A
    │                                      │
    │ 1. Request: "Forward display"       │
    ├─────────────────────────────────────>│
    │                                      │
    │ 2. Peer A: Capture framebuffer      │
    │    (e.g., /dev/fb0, X11, Wayland)  │
    │                                      │
    │ 3. Stream frames to Peer B          │
    │    (compressed, e.g., H.264)        │
    │    ┌─────────────────────────────┐  │
    │    │ frame0, frame1, frame2...  │  │
    │    └─────────────────────────────┘  │
    │◄────────────────────────────────────┤
    │                                      │
    │ 4. Peer B: Display to local screen  │
    │                                      │
    │ 5. User input on Peer B:             │
    │    keypress ───────────────┐         │
    │    mouse move ─────────────┤         │
    │                           ▼         │
    │ 6. Stream events back to Peer A      │
    │◄────────────────────────────────────┤
    │                                      │
    │ 7. Peer A: Inject into input stack   │
    │    (e.g., uinput device, XTest)     │
    │                                      │
```

**Characteristics**:
- ✅ Bidirectional streaming required
- ✅ High bandwidth for display (compressed video)
- ✅ Low latency critical for interactivity
- ❌ Requires persistent connection
- ✅ Can be disconnected/reconnected (stateful)
- ⚠️ Security: Full control over peer's I/O!

**Example**: Remote desktop / VNC-style
```bash
# User on Peer B:
evosh display allocate --peer peer-a --resolution 1920x1080
# Opens window showing Peer A's desktop
# User can move mouse, type, see output
```

**For keyboard specifically**:
- Treat as input device forwarding
- Peer A creates virtual keyboard device (uinput on Linux)
- Peer B sends key events → injected into A's input stack
- Use case: Kiosk mode, shared workstation

**EvoLLM use case**: Not directly relevant for inference, but for **EvoOS** (distributed OS) this is essential for remote administration, shared workstations, cluster management.

---

### Model D: SHARED ACCESS (Concurrent Resources)

**Resources**: Files opened read-write, databases, message queues, shared memory

**Mechanism**:
```
Peer A (writer)                        Peer B (reader)
    │                                      │
    │ 1. Open file in SHARED mode         │
    │    (with advisory locking)          │
    ├─────────────────────────────────────>│
    │ lease.endpoint = "grpc://.../file" │
    │                                      │
    │ 2. Both peers get lease             │
    │    to SAME resource                │
    │                                      │
    │ 3. Peer A writes:                   │
    │    lease.write(data) ──────────────>│
    │                                      │
    │ 4. Peer B reads (blocking/poll):    │
    │    data = lease.read() ◄────────────┤
    │                                      │
    │ 5. Changes visible to all holders  │
    │    (coherence protocol optional)   │
```

**Characteristics**:
- ✅ True multi-producer/multi-consumer
- ⚠️ Needs consistency protocol (locking, CRDT, or single-writer)
- ✅ Low latency (direct access via proxy)
- ❌ Requires provider to stay online
- ⚠️ Conflict resolution needed

**Example**: Shared dataset for distributed training
```python
# Multiple peers share same dataset file
lease = provider.allocate(ResourceSpec(
    resource_type=ResourceType.FILE,
    path="/shared/train_data.bin",
    mode=READ_WRITE,
    lock_mode='advisory'  # or 'mandatory'
))

# All peers read/write synchronously
# Changes by one peer visible to others
```

---

### Model E: TIME-SHARING (Hardware Devices)

**Resources**: GPU, CPU cores, NIC, disk I/O queue, camera, microphone

**Mechanism**:
```
 scheduling: Round-robin, priority, or fair-share

Peer A (GPU user)                     Peer B (GPU user)
    │                                      │
    │ 1. Request: "Allocate GPU for 5min"│
    ├─────────────────────────────────────>│
    │                                      │ scheduler
    │                                      │ - Queue if busy
    │                                      │ - Preempt if higher priority
    │                                      │
    │ 2. Granted: GPU context 0x1234      │
    │    lease.endpoint = "cuda://..."   │
    │                                      │
    │ 3. Use GPU via remote API:          │
    │    - Submit kernels                 │
    │    - Allocate memory               │
    │    - Copy tensors                  │
    │                                      │
    │ 4. Time-slice: GPU switches between │
    │    Peer A and Peer B (10ms slices) │
    │    or: Peer A preempted by Peer B  │
    │                                      │
```

**Characteristics**:
- ✅ Fair sharing of scarce hardware
- ⚠️ Performance interference (noisy neighbor)
- ✅ Isolation: CUDA contexts separate
- ⚠️ Context switch overhead (like OS multitasking)
- ✅ Can preempt (kill long-running kernels)

**Example**: GPU time-sharing
```python
lease = provider.allocate(ResourceSpec(
    resource_type=ResourceType.GPU,
    quantity=1,  # 1 GPU
    constraints={'min_compute_capability': '7.5'},
    duration_s=3600,
    preemptible=True  # Can be suspended if higher priority arrives
))
# lease.endpoint = "cuda://peer-a:7000"
# Use via CUDA driver API or RPC:
remote_gpu = connect(lease.endpoint)
tensor = remote_gpu.malloc(1e9)  # 1GB allocation
result = remote_gpu.launch_kernel(kernel, ...)
```

**EvoLLM use case**: GPU compute offload for layer execution (Phase 3)

---

## 3. Extended ResourceType Enum

### Current (Phase 2) - Model Layers Only

```python
class ResourceType(Enum):
    MODEL_LAYER = "model_layer"
```

### Extended (Phase 3+) - Full System

```python
class ResourceCategory(Enum):
    STORAGE = "storage"
    COMPUTE = "compute"
    INTERACTIVE = "interactive"
    DEVICE = "device"
    VIRTUAL = "virtual"

class ResourceType(Enum):
    # Storage / Data
    FILE = ("file", ResourceCategory.STORAGE)
    DIRECTORY = ("directory", ResourceCategory.STORAGE)
    BLOB = ("blob", ResourceCategory.STORAGE)  # arbitrary bytes
    DATABASE = ("database", ResourceCategory.STORAGE)
    CHECKPOINT = ("checkpoint", ResourceCategory.STORAGE)

    # Compute
    PROCESS = ("process", ResourceCategory.COMPUTE)
    CONTAINER = ("container", ResourceCategory.COMPUTE)
    FUNCTION = ("function", ResourceCategory.COMPUTE)  # serverless
    GPU_KERNEL = ("gpu_kernel", ResourceCategory.COMPUTE)
    JOB = ("job", ResourceCategory.COMPUTE)  # batch job

    # Interactive I/O
    KEYBOARD = ("keyboard", ResourceCategory.INTERACTIVE)
    DISPLAY = ("display", ResourceCategory.INTERACTIVE)
    FRAMEBUFFER = ("framebuffer", ResourceCategory.INTERACTIVE)
    AUDIO_INPUT = ("audio_input", ResourceCategory.INTERACTIVE)
    AUDIO_OUTPUT = ("audio_output", ResourceCategory.INTERACTIVE)
    POINTER = ("pointer", ResourceCategory.INTERACTIVE)
    CONSOLE = ("console", ResourceCategory.INTERACTIVE)  # tty

    # Hardware devices (time-shared)
    GPU = ("gpu", ResourceCategory.DEVICE)
    CPU_CORE = ("cpu_core", ResourceCategory.DEVICE)
    NPU = ("npu", ResourceCategory.DEVICE)
    TPU = ("tpu", ResourceCategory.DEVICE)
    NIC = ("nic", ResourceCategory.DEVICE)
    CAMERA = ("camera", ResourceCategory.DEVICE)
    MICROPHONE = ("microphone", ResourceCategory.DEVICE)
    DISK_IO = ("disk_io", ResourceCategory.DEVICE)
    SERIAL_PORT = ("serial_port", ResourceCategory.DEVICE)

    # Virtual/Namespaces
    PID_NAMESPACE = ("pid_namespace", ResourceCategory.VIRTUAL)
    NET_NAMESPACE = ("net_namespace", ResourceCategory.VIRTUAL)
    USER_NAMESPACE = ("user_namespace", ResourceCategory.VIRTUAL)
    IPC_NAMESPACE = ("ipc_namespace", ResourceCategory.VIRTUAL)
    MOUNT_NAMESPACE = ("mount_namespace", ResourceCategory.VIRTUAL)
    NETWORK_NAMESPACE = ("network_namespace", ResourceCategory.VIRTUAL)

    # Special
    MODEL_LAYER = ("model_layer", ResourceCategory.SPECIAL)  # Phase 2
    LICENSE = ("license", ResourceCategory.SPECIAL)  # software license

    def __init__(self, value, category):
        self.value = value
        self.category = category
```

---

## 4. Unified Allocation API

### Generalized ResourceSpec

```python
@dataclass
class ResourceSpec:
    """Request to allocate a resource"""
    resource_type: ResourceType

    # Quantity (interpretation varies by type)
    quantity: float = 1.0  # GB, cores, 1 for boolean resources

    # Constraints (filter offers)
    constraints: Dict[str, Any] = field(default_factory=dict)
    # Examples:
    #   For GPU: {'min_compute_capability': '7.0', 'vendor': 'nvidia'}
    #   For FILE: {'path_prefix': '/datasets', 'min_size_gb': 10}
    #   For PROCESS: {'command': ['python'], 'env': {'CUDA': '1'}}

    # Duration (None = until explicitly released)
    duration_s: Optional[float] = None

    # Access mode
    mode: AllocationMode = AllocationMode.EXCLUSIVE
    #   EXCLUSIVE: only this requester
    #   SHARED: multiple requesters (read-only or with locking)
    #   BURSTABLE: exclusive but can be preempted

    # I/O direction (for interactive resources)
    io_方向: List[str] = field(default_factory=lambda: ['read', 'write'])
    #   For DISPLAY: ['view'] (consumer only) or ['view', 'inject'] (control)
    #   For KEYBOARD: ['inject'] (send keystrokes) or ['capture'] (sniff)

    # Quality of Service
    qos: Optional[Dict[str, Any]] = None
    #   {'latency_ms': 10, 'bandwidth_mb_s': 100, 'priority': 5}

    # Reservation strategy
    strategy: str = 'best_effort'  # or 'reserved', 'spot', 'guaranteed'
```

### Generalized ResourceOffer

```python
@dataclass
class ResourceOffer:
    """What a provider can offer"""
    provider_id: str
    resource_type: ResourceType
    available: float  # quantity available (GB, cores, or 1 for boolean)
    price_per_unit: float  # credits per unit-hour
    capabilities: Dict[str, Any]  # type-specific attributes
    load_score: float  # 0-1
    quality: float  # 0-1 (reliability, performance)
    reserved: Optional[str] = None  # reservation_id if pre-allocated
```

### Generalized ResourceLease

```python
@dataclass
class ResourceLease:
    """Granted allocation with access credentials"""
    lease_id: str  # UUID
    provider_id: str
    resource_type: ResourceType
    quantity: float
    endpoint: str  # Connection info - varies by type:
                   #   FILE: "grpc://peer:50061/files/abc123"
                   #   PROCESS: "grpc://peer:50062/exec/xyz789"
                   #   DISPLAY: "vnc://peer:5901?password=..."
                   #   GPU: "cuda://peer:7000?context=123"
                   #   SHELL: "ssh://user@peer:2222?key=..."
    credentials: Dict[str, str]  # Auth tokens, passwords, certs
    access_control: Dict[str, Any]  # Permissions: {'read': true, 'write': false}
    expires_at: float  # Unix timestamp
    renewable: bool = True
    metadata: Dict[str, Any]  # Usage limits, QoS guarantees, restrictions
```

---

## 5. Specific Resource Implementations

### 5.1 File/Directory Sharing

```python
class FileResourceProvider(ResourceProvider):
    """Share files/directories from local filesystem"""

    def __init__(self, base_path: str, max_share_size_gb: float):
        self.base_path = Path(base_path)
        self.max_share_size = max_share_size_gb * 1e9
        self.active_transfers: Dict[str, Transfer] = {}

    def advertise_capabilities(self) -> List[ResourceOffer]:
        # Scan base_path, compute total size
        # Return offers for directories with sizes
        offers = []
        for path in self.base_path.glob('*'):
            if path.is_dir():
                size_gb = self._dir_size(path) / 1e9
                offers.append(ResourceOffer(
                    provider_id=self.peer_id,
                    resource_type=ResourceType.DIRECTORY,
                    available=size_gb,
                    price_per_gb=0.1,
                    capabilities={'path': str(path), 'item_count': len(list(path.glob('**/*')))},
                    load_score=self._current_load(),
                    quality=0.99
                ))
        return offers

    def allocate(self, spec: ResourceSpec) -> ResourceLease:
        # Validate path within allowed base
        requested_path = Path(spec.constraints['path'])
        if not requested_path.is_relative_to(self.base_path):
            raise AccessDeniedError("Path outside shared directory")

        # Check capacity
        if requested_path.stat().st_size > self.max_share_size:
            raise ResourceUnavailableError("Too large")

        # Create transfer endpoint
        transfer_id = uuid4()
        endpoint = f"grpc://{self.address}:50061/files/{transfer_id}"

        # Start serving (background thread reads file on demand)
        self.active_transfers[transfer_id] = FileTransfer(
            path=requested_path,
            offset=0,
            lease_id=transfer_id
        )

        return ResourceLease(
            lease_id=str(transfer_id),
            provider_id=self.peer_id,
            resource_type=spec.resource_type,
            quantity=requested_path.stat().st_size / 1e9,
            endpoint=endpoint,
            credentials={'auth_token': self._generate_token(transfer_id)},
            access_control={'read': True, 'write': False},
            expires_at=time.time() + (spec.duration_s or 3600)
        )

    def get_endpoint_stream(self, lease_id: str):
        """gRPC handler: stream file chunks"""
        transfer = self.active_transfers[lease_id]
        with open(transfer.path, 'rb') as f:
            while chunk := f.read(10 * 1024 * 1024):  # 10MB chunks
                yield FileChunk(data=chunk, offset=f.tell())
```

**Client usage**:
```python
lease = provider.allocate(ResourceSpec(
    resource_type=ResourceType.FILE,
    constraints={'path': '/shared/imagenet/train.zip'},
    mode=READ_ONLY,
    duration_s=7200
))

# Download
with open('local_copy.zip', 'wb') as f:
    for chunk in grpc_stream(lease.endpoint):
        f.write(chunk.data)
```

---

### 5.2 Process / Shell Sharing (Your XTerm Example)

```python
class ProcessResourceProvider(ResourceProvider):
    """Execute and control remote processes"""

    def __init__(self, max_processes: int, sandbox: bool = True):
        self.max_processes = max_processes
        self.active_processes: Dict[str, subprocess.Popen] = {}
        self.sandbox = sandbox  # Use containers/chroot if available

    def allocate(self, spec: ResourceSpec) -> ResourceLease:
        # Check capacity
        if len(self.active_processes) >= self.max_processes:
            raise ResourceUnavailableError("Max processes reached")

        # Create process
        proc = subprocess.Popen(
            spec.constraints.get('command', ['/bin/bash']),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=spec.constraints.get('env', {}),
            cwd=spec.constraints.get('cwd')
        )

        lease_id = uuid4()
        endpoint = f"grpc://{self.address}:50062/process/{lease_id}"

        self.active_processes[lease_id] = proc

        # Background monitor for exit
        threading.Thread(target=self._monitor, args=(lease_id,), daemon=True).start()

        return ResourceLease(
            lease_id=str(lease_id),
            provider_id=self.peer_id,
            resource_type=spec.resource_type,
            quantity=1,
            endpoint=endpoint,
            credentials={'auth_token': self._generate_token(lease_id)},
            access_control={'stdin': True, 'stdout': True, 'stderr': True, 'kill': True},
            expires_at=time.time() + (spec.duration_s or 86400)
        )

    def _monitor(self, lease_id: str):
        """Watch process, clean up on exit"""
        proc = self.active_processes[lease_id]
        proc.wait()
        del self.active_processes[lease_id]
        # Optionally: record exit code, resource usage to ledger

    def execute(self, lease_id: str, stdin_data: bytes) -> bytes:
        """gRPC handler: interact with process"""
        proc = self.active_processes[lease_id]
        proc.stdin.write(stdin_data)
        proc.stdin.flush()
        return proc.stdout.read(4096)
```

**Client usage** (xterm-style):
```python
# Allocate interactive shell
lease = provider.allocate(ResourceSpec(
    resource_type=ResourceType.PROCESS,
    constraints={'command': ['/bin/bash', '-i'], 'cwd': '/home/user'},
    mode=EXCLUSIVE,
    duration_s=3600,
    io_direction=['read', 'write']
))

# Connect via gRPC (or wrap in SSH for compatibility)
channel = grpc.secure_channel(lease.endpoint, creds)
stub = ProcessServiceStub(channel)

# Interactive session
stdin = sys.stdin
stdout = sys.stdout
while True:
    # Read from local keyboard
    user_input = stdin.read(1)
    # Send to remote
    stub.execute(lease.lease_id, user_input.encode())
    # Get response
    output = stub.read_output(lease.lease_id)
    # Display on local screen
    stdout.write(output.decode())
```

**Alternative: SSH passthrough**
```bash
# Simpler: use standard SSH, but with EvoOS allocation
evosh process allocate --peer peer-b --command "bash" --interactive
# Returns SSH connection string: ssh user@peer-b -p 2222
# User runs: ssh user@peer-b -p 2222
# Behind scenes: EvoOS allocated PTY on peer-b, gives SSH tunnel
```

---

### 5.3 Display / Framebuffer Sharing

```python
class DisplayResourceProvider(ResourceProvider):
    """Share screen / framebuffer"""

    def __init__(self, display_device: str = ':0'):
        self.display = display_device
        self.capture_method = self._detect_capture_method()
        # Methods: X11 (xwd), Wayland (screencopy), DRM (kms)

    def allocate(self, spec: ResourceSpec) -> ResourceLease:
        # Check permissions (who can access display?)
        if not self._can_access_display(spec.requester):
            raise AccessDeniedError()

        # Create VNC server or streaming endpoint
        lease_id = uuid4()
        port = self._allocate_port()

        # Start screen capture thread
        self._start_streaming(lease_id, port)

        endpoint = f"vnc://{self.address}:{port}"
        if spec.constraints.get('quality') == 'high':
            endpoint += '?quality=95'

        return ResourceLease(
            lease_id=str(lease_id),
            provider_id=self.peer_id,
            resource_type=ResourceType.DISPLAY,
            quantity=1,
            endpoint=endpoint,
            credentials={'vnc_password': self._gen_vnc_pass()},
            access_control={'view': True, 'inject': spec.io_direction.includes('inject')},
            expires_at=time.time() + (spec.duration_s or 3600)
        )

    def _start_streaming(self, lease_id: str, port: int):
        """Capture framebuffer, stream via VNC"""
        import pygame  # or use PyAV for H.264
        # Or use noVNC (VNC over WebSocket)
        # ...

    def inject_input(self, lease_id: str, event: InputEvent):
        """Inject mouse/keyboard events into display"""
        # Use XTestFakeKeyEvent, uinput, or similar
        pass
```

**Client usage**:
```bash
# User gets VNC client window showing Peer A's desktop
evosh display allocate --peer peer-a --interactive
# VNC viewer opens, user sees and controls Peer A's desktop
```

**Use case**: Remote administration, shared workstations, cluster control panels.

---

### 5.4 Keyboard/Input Device Sharing

```python
class KeyboardResourceProvider(ResourceProvider):
    """Inject keystrokes into peer's input stack"""

    def __init__(self):
        self.uinput_device = None  # Linux: /dev/uinput

    def allocate(self, spec: ResourceSpec) -> ResourceLease:
        # Create virtual input device
        device = self._create_uinput_device()
        lease_id = uuid4()

        return ResourceLease(
            lease_id=str(lease_id),
            provider_id=self.peer_id,
            resource_type=ResourceType.KEYBOARD,
            quantity=1,
            endpoint=f"grpc://{self.address}:50063/input/{lease_id}",
            credentials={'auth_token': self._gen_token(lease_id)},
            access_control={'inject': True, 'capture': False},
            expires_at=time.time() + (spec.duration_s or 3600)
        )

    def inject_key(self, lease_id: str, keycode: int, pressed: bool):
        """gRPC handler: simulate key press/release"""
        self.uinput_device.emit(keycode, pressed)

    def capture_keys(self, lease_id: str):
        """Optionally: sniffer mode (keylogger) - DANGEROUS"""
        # Requires root/admin privileges
        # Ethically: only with explicit consent
        pass
```

**Client usage** (legitimate):
```python
# Automated testing: remote control of console app
lease = provider.allocate(ResourceSpec(
    resource_type=ResourceType.KEYBOARD,
    io_direction=['inject']
))
# Send keystrokes:
grpc_stub.inject_key(lease.lease_id, keycode=ord('h'), pressed=True)
grpc_stub.inject_key(lease.lease_id, keycode=ord('h'), pressed=False)
# User on Peer A sees 'h' typed
```

**Security warning**: Keyboard injection is **privileged operation**. Must:
- Require mTLS + explicit user consent
- Log all injections (audit trail)
- Only allow on dedicated service accounts
- Consider ethical use: kiosk mode, testing, accessibility

---

### 5.5 GPU Compute Sharing (Phase 3 of EvoLLM)

```python
class GPUResourceProvider(ResourceProvider):
    """Time-share GPU compute (CUDA, ROCm)"""

    def __init__(self, gpu_id: int, memory_gb: float):
        self.gpu_id = gpu_id
        self.total_memory = memory_gb * 1e9
        self.contexts: Dict[str, CUDAContext] = {}  # lease_id → context
        self.scheduler = RoundRobinScheduler()

    def allocate(self, spec: ResourceSpec) -> ResourceLease:
        # Check available memory
        used = sum(ctx.memory_used for ctx in self.contexts.values())
        available_gb = (self.total_memory - used) / 1e9
        if available_gb < spec.quantity:
            raise ResourceUnavailableError("Insufficient VRAM")

        # Create CUDA context (MPS or MIG if available for isolation)
        context = self._create_context(
            memory_gb=spec.quantity,
            compute_capability=spec.constraints.get('min_compute_capability')
        )

        lease_id = uuid4()
        endpoint = f"cuda://{self.address}:7000/context/{lease_id}"

        self.contexts[lease_id] = context

        return ResourceLease(
            lease_id=str(lease_id),
            provider_id=self.peer_id,
            resource_type=ResourceType.GPU,
            quantity=spec.quantity,
            endpoint=endpoint,
            credentials={'cuda_context_id': context.id},
            access_control={'kernel_execute': True, 'memory_alloc': True},
            expires_at=time.time() + (spec.duration_s or 3600),
            renewable=True
        )

    def execute_kernel(self, lease_id: str, kernel_code: bytes, *args):
        """gRPC handler: launch CUDA kernel"""
        context = self.contexts[lease_id]
        # Load PTX, launch kernel with args
        # Return event for async completion
        pass
```

**Client usage** (EvoLLM compute offload):
```python
# Peer B offloading layer computation to Peer A's GPU
lease = peer_a.allocate(ResourceSpec(
    resource_type=ResourceType.GPU,
    quantity=4.0,  # 4GB VRAM for this layer
    duration_s=300
))

# EvoLLM layer executor:
gpu = RemoteGPU(lease.endpoint, lease.credentials)
activations = gpu.matmul(weights, previous_activations)
```

---

## 6. Unified ResourceProvider Interface (Extended)

```python
class ResourceProvider(ABC):
    """Unified interface for ANY system resource"""

    @abstractmethod
    def advertise_capabilities(self) -> List[ResourceOffer]:
        """Broadcast what resources I can provide"""
        pass

    @abstractmethod
    def allocate(self, spec: ResourceSpec) -> ResourceLease:
        """Allocate resource according to spec. Returns lease with access info."""
        pass

    @abstractmethod
    def release(self, lease_id: str):
        """Free allocated resource (early termination)"""
        pass

    @abstractmethod
    def get_available(self, resource_type: ResourceType) -> List[ResourceOffer]:
        """List currently available offers of this type"""
        pass

    @abstractmethod
    def get_utilization(self) -> Dict[ResourceType, float]:
        """Current utilization 0-1 per resource type"""
        pass

    # Optional hooks:
    @abstractmethod
    def can_preempt(self, lease_id: str) -> bool:
        """Can this lease be preempted (for higher priority allocation)?"""
        pass

    @abstractmethod
    def transfer(self, lease_id: str, new_provider: str) -> bool:
        """Migrate resource to another peer (live migration)"""
        pass

    @abstractmethod
    def snapshot(self, lease_id: str) -> bytes:
        """Checkpoint resource state (for migration or persistence)"""
        pass

    @abstractmethod
    def restore(self, snapshot: bytes, new_lease_id: str) -> ResourceLease:
        """Restore from checkpoint"""
        pass
```

---

## 7. Updated EvoOS Architecture Diagram

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                              EVOOS (ANY RESOURCE)                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────┐
│                         UNIFIED RESOURCE LAYER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ResourceSpec  ──►  Registry (query offers)  ──►  ResourceProvider        │
│       │                                                │                   │
│       │ allocate(spec)                                 │ advertise()       │
│       ▼                                                │                   │
│  ResourceLease ◄───────────────────────────────────────┘                   │
│       │                                                                     │
│       │ use endpoint (various protocols)                                   │
│       ▼                                                                     │
│  ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐           │
│  │  File   │ Process │ Display │ GPU     │ Keyboard│...      │           │
│  │ Transfer│ Exec    │ Stream  │ Compute │ Inject  │         │           │
│  └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘           │
│                                                                             │
│  Control: release(), transfer(), snapshot(), restore()                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ implements
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RESOURCE PROVIDER TYPES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FileProvider          ProcessProvider          DisplayProvider            │
│  • read()              • spawn()                • capture_framebuffer()    │
│  • write()             • stdin.write()          • inject_mouse()           │
│  • list()              • stdout.read()          • inject_keyboard()        │
│                                                                             │
│  GPUProvider           KeyboardProvider        ContainerProvider          │
│  • alloc_memory()      • inject_key()          • create_container()       │
│  • launch_kernel()     • capture_keys()        • exec()                  │
│  • copy_to_gpu()                                                           │
│                                                                             │
│  All inherit: get_available(), get_utilization()                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ uses
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SHARED INFRASTRUCTURE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Registry              Ledger/Accounting          Security (mTLS)          │
│  • Who has what        • Credits/debits          • Authentication         │
│  • Offer indexing      • Pricing                 • Authorization         │
│  • Health tracking    • Reputation              • Audit log             │
│                                                                             │
│  Transport: gRPC (with streaming for large data)                          │
│  Protocols: varies by resource (file download, exec stream, VNC, CUDA)    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Adapting Existing Plan

### Changes to Original Plan (Phase 2/3)

**Phase 2** (as planned) remains:
- ResourceProvider abstraction ✓
- PeerClientResourceManager (layers only)
- PeerServer (serving layers)
- Ledger & reputation

**Phase 3** (Generalized Resources):
```
Sprint 3a: Generalize ResourceProvider
  - Rename get_layer() → allocate(spec)
  - Keep get_layer() as convenience wrapper for backward compatibility
  - Define full ResourceType enum
  - Implement FileResourceProvider (simplest)
  - Implement ProcessResourceProvider (xterm)
  - Extend Registry to index all resource types
  - Tests: allocate file, process

Sprint 3b: GPU Compute Offload
  - Implement GPUResourceProvider (CUDA remote)
  - Define ComputeService gRPC (kernel launch, memory ops)
  - Integrate with EvoLLM: layer execution on remote GPU
  - Tests: matmul on remote GPU

Sprint 3c: Interactive Resources
  - DisplayResourceProvider (framebuffer capture & injection)
  - KeyboardResourceProvider (input injection)
  - VNC server integration or custom protocol
  - Tests: remote desktop allocation

Sprint 3d: Advanced Features
  - Resource migration (snapshot/restore)
  - Checkpoint/restart long-running processes (CRIU)
  - Live GPU migration (NVIDIA MIG, MPS)
  - Hierarchical allocation (compound specs)
```

### Updated TODO.md Tasks

Add to Phase 3 section:

- [ ] **Generalized ResourceProvider**
  - [ ] Rename `get_layer()` to `allocate()` in ABC
  - [ ] Add `ResourceSpec`, `ResourceOffer`, `ResourceLease` dataclasses
  - [ ] Implement `FileResourceProvider`
  - [ ] Implement `ProcessResourceProvider` (xterm)
  - [ ] Implement `DisplayResourceProvider` (remote desktop)
  - [ ] Implement `GPUResourceProvider` (compute offload)
  - [ ] Tests for each provider type

- [ ] **Unified Registry**
  - [ ] Index by `(resource_type, constraints)` not just `layer_name`
  - [ ] Query API: `registry.query(type=GPU, min_vram=4, max_price=0.5)`
  - [ ] Advertisements include resource_type field

- [ ] **evosh CLI** (replaces evollm CLI for resource management)
  - [ ] `evosh resources list` - show available
  - [ ] `evosh resources allocate --type gpu --quantity 4`
  - [ ] `evosh process execute --peer peer-b "python train.py"`
  - [ ] `evosh display allocate --peer peer-a` (opens VNC)
  - [ ] `evosh shell allocate --peer peer-b` (xterm)
  - [ ] `evosh forward --local 8080 --peer peer-b:80`

---

## 9. Security & Ethics for Dangerous Resources

### High-Risk Resources Requiring Extra Safeguards

| Resource | Risk | Safeguards Required |
|----------|------|-------------------|
| KEYBOARD | Keystroke logging, input injection | • Explicit user consent UI<br>• Session recording (audit)<br>• Time-limited (max 1h)<br>• Require 2FA per allocation |
| DISPLAY  | Screen observation, control | • Visible indicator (recording dot)<br>• Require physical presence approval<br>• Encrypted stream only<br>• Don't allow clipboard access |
| CAMERA   | Privacy violation | • Hardware LED always on when shared<br>• User must manually enable<br>• Stream watermark "shared via EvoOS"<br>• Expire after 10min |
| MICROPHONE| Eavesdropping | • Audio watermark<br>• Visible audio level meter<br>• Mute button always visible<br>• Auto-stop after 5min |
| PROCESS  | Code execution, data access | • Sandbox (container, firejail)<br>• Limit syscalls (seccomp)<br>• Filesystem namespace<br>• No network egress |
| GPU      | Crypto-mining abuse | • Limit compute duration<br>• Monitor temperature/power<br>• Rate-limit kernel launches |
| ADMIN/ROOT| Full system compromise | • Never share root<br>• Use separate service accounts<br>• Capability bounding sets |

**Ethical use policy**:
- Resources that can observe human activity (camera, mic, keyboard) require **explicit, informed consent** each session
- Allocations must be **time-bounded** (auto-revoke)
- **Audit logging**: who allocated what, when, with whom
- **Revocation**: User can instantly revoke any lease at any time

---

## 10. Performance & Feasibility Analysis

### Can We Really Share These Resources Efficiently?

| Resource | Network Overhead | Latency Sensitivity | Feasibility | Use Case Viability |
|----------|------------------|---------------------|--------------|-------------------|
| File/Dir | High (size) | Low | ✅ Easy | ✅ Backups, datasets |
| Process | Medium (I/O streaming) | Medium | ✅ Easy | ✅ Remote execution, xterm |
| Display | Very high (video) | High (60fps) | ⚠️ Hard | ✅ Remote desktop (with compression) |
| Keyboard | Very low | Very high | ✅ Trivial | ✅ Kiosk, testing |
| GPU compute | Medium (kernel + data) | Medium | ✅ Hard but done (NVIDIA GRID) | ✅ Compute offload |
| Camera | High (video) | High | ⚠️ Legal issues | ⚠️ Only with consent |
| Audio | Medium (streaming) | High | ✅ | ✅ Remote conferencing |

**Feasibility结论**:
- ✅ **Trivial**: Files, processes, keyboard, GPU compute
- ⚠️ **Challenging but possible**: Display, audio (need good compression)
- ❌ **Not recommended**: Camera/mic (privacy), privileged devices (root)

---

## 11. Example: Complete EvoOS Workflow

```bash
# User wants to run training job requiring:
# - Datasets (files)
# - GPU compute
# - Monitor progress (display)

# 1. Allocate dataset files from Peer A (has fast storage)
$ evosh file allocate --peer peer-a --path "/datasets/imagenet" --duration 24h
Lease: file-abc123 (endpoint: grpc://peer-a:50061/files/abc)
Mounting to /mnt/remote-imagenet...

# 2. Allocate GPU from Peer B (has A100)
$ evosh gpu allocate --peer peer-b --quantity 8.0 --duration 4h
Lease: gpu-def456 (endpoint: cuda://peer-b:7000/ctx456)
CUDA_VISIBLE_DEVICES=$(evosh gpu device --lease gpu-def456) python train.py

# 3. Forward display to monitor progress (Peer B's GPU outputs to its screen)
$ evosh display allocate --peer peer-b --view-only
Opening VNC viewer to peer-b:5901...
# User sees training metrics plots updating in real-time

# 4. Allocate 4 CPU cores from Peer C (for data loading)
$ evosh cpu allocate --peer peer-c --cores 4
Lease: cpu-ghi789

# 5. Run distributed training job
$ torchrun \
  --nproc_per_node=1 \
  --nnodes=3 \
  --node_rank=0 \
  --master_addr=peer-a \
  train.py \
  --data /mnt/remote-imagenet \
  --gpu-count 8 \
  --cpu-count 4 &
# Job uses 3 peers' resources transparently

# 6. Monitor usage/cost
$ evosh ledger balance
Alice: +12.3 credits
# (Dataset: +3, GPU: +8, CPU: +1.3)

# 7. Cleanup
$ evosh release --all
# All leases terminated, resources returned
```

---

## 12. Summary

**Yes, EvoOS can share ANY system object**, but categorize by:

1. **Data Resources** (copy/stream): Files, datasets, blobs
2. **Compute Resources** (remote exec): Processes, containers, functions
3. **Interactive Resources** (bidirectional streams): Display, keyboard, audio
4. **Hardware Devices** (time-share): GPU, CPU cores, NIC, camera
5. **Virtual Resources** (namespace join): PID, network, mount namespaces

**Key insight**: The `ResourceProvider` abstraction generalizes beautifully because:
- `allocate(spec)` → `lease` with `endpoint` is universal
- Endpoint protocol varies by type (gRPC-RPC for process, VNC for display, CUDA for GPU, plain stream for files)
- Lifecycle is similar: allocate → use → release (with optional migration)
- Security model (mTLS, ACLs) applies universally

**Implementation order**:
1. Phase 2: Model layers (already planned)
2. Phase 3a: Files + Processes (xterm/SSH) - **your use case ✓**
3. Phase 3b: GPU compute offload
4. Phase 3c: Display/keyboard (remote desktop)
5. Phase 3d: Advanced: containers, namespaces, live migration

**Security imperative**: Resources like keyboard/display/camera require **explicit consent, audit logging, and time-limits** to prevent abuse.

---

**Conclusion**: The architecture scales from "share model layers" to "share entire computers" naturally. Your vision of sharing a keyboard or xterm session is not only possible—it's a **core use case** for Phase 3 (distributed OS). The same ResourceProvider abstraction works for all resource types; only the endpoint protocol and lifecycle semantics differ.

**Updated plan**: See amended TODO.md with Phase 3 tasks for generalized resources.
