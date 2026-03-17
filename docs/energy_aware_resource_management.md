# Energy-Aware Resource Management for EvoLLM

**Status**: Proposed Design (Phase 2½)
**Date**: 2026-03-17
**Author**: EvoLLM Team
**Related**: [Main PLAN.md](../PLAN.md), [Resource Economy](../PLAN.md#resource-economy-incentives--benefits)

---

## Executive Summary

Extend EvoLLM/EvoOS peer network with **dynamic resource state management** to optimize energy consumption and costs. This feature enables peers to:

1. **Manage power states** (sleep, hibernate, functional) based on utilization
2. **Track energy consumption** per inference task with approximate billing
3. **Crowd-sourced state changes** - peers can request others to wake resources for shared benefit
4. **Dynamic pricing** based on time-of-day, carbon intensity, and market conditions

### Expected Impact

- **Cost savings**: 30-70% reduction in energy bills for idle resources
- **Community efficiency**: Lower total AI infrastructure energy through pooling underutilized resources
- **Carbon reduction**: Enable green computing by preferring renewable-rich grid periods
- **Economic incentive**: Earn credits by providing resources during demand
- **Hardware longevity**: Reduce thermal cycles, extend SSD/GPU lifespan

---

## Problem Statement

Current EvoLLM focuses on memory-efficient inference but lacks:

1. **Energy awareness**: Resources run at full power even when idle
2. **Dynamic power management**: No way to temporarily reduce consumption
3. **Cost tracking**: No accounting for energy used per inference task
4. **Crowd coordination**: Peers can't request others to wake resources for shared benefit

### Example Scenario

- **Peer A**: Powerful GPU but idle (wasting energy)
- **Peer B**: Needs compute for inference but only has CPU
- **Current system**: B can't leverage A's idle GPU
- **With energy features**: B requests A to wake GPU; A's EnergyManager decides based on:
  - Projected energy cost vs expected earnings from serving
  - Current energy prices (if in volatile market)
  - Owner policy (min acceptable payment)
  - Timeline: wake GPU, execute B's inference, return to sleep

---

## Architecture: Resource State Lattice

### Resource States

```
┌────────────────────────────────────────────────┐
│ FUNCTIONAL (Active)                            │
│ • Operating at target performance              │
│ • Full power consumption (baseline × 1.0)      │
│ • Ready for immediate work                     │
│ • Latency: 0ms (instant access)               │
│ • Energy cost: P_max (Watts)                   │
└────────────────────────────────────────────────┘

                    ▲  wake_request()
                    │  execute()
                    │  finish()
                    ↓  idle_timeout() or explicit

┌────────────────────────────────────────────────┐
│ HIBERNATE (Low Power)                          │
│ • Partially powered, minimal state retained   │
│ • Reduced power (P_sleep × 0.3-0.5)            │
│ • Wake time: 500ms - 2s                        │
│ • Used for: short breaks (< 5 min)            │
│ • Energy cost: P_hibernate                     │
└────────────────────────────────────────────────┘

                    ▲  wake_request()
                    ↓  idle_timeout() or explicit

┌────────────────────────────────────────────────┐
│ SLEEP (Deep Sleep)                             │
│ • Minimal power, near-off state                │
│ • Minimal power (P_sleep × 0.05-0.1)           │
│ • Wake time: 2-10s                             │
│ • Used for: extended idle periods (> 5 min)   │
│ • Energy cost: P_sleep                         │
└────────────────────────────────────────────────┘

                    ▲  explicit or external signal
                    ↓  idle_timeout()

┌────────────────────────────────────────────────┐
│ OFFLINE / SHUTDOWN                             │
│ • No power consumption                         │
│ • Wake time: 30s - 5min (boot)                │
│ • Manual only (or scheduled)                  │
│ • Energy cost: 0                               │
└────────────────────────────────────────────────┘
```

### State Transition Triggers

**Internal (autonomous)**:
- `idle_timeout`: FUNCTIONAL → HIBERNATE after N seconds
- `idle_timeout`: HIBERNATE → SLEEP after M seconds
- `energy_price_spike`: FUNCTIONAL → HIBERNATE if price > threshold

**External (requests)**:
- `crowd_request`: SLEEP/HIBERNATE → FUNCTIONAL (if profitable)
- `owner_override`: ANY → explicit state
- `scheduled_event`: SLEEP → FUNCTIONAL (expected workload)

---

## Energy Accounting System

### Core Data Structures

```python
from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
import time

class ResourceState(Enum):
    FUNCTIONAL = "functional"
    HIBERNATE = "hibernate"
    SLEEP = "sleep"
    OFFLINE = "offline"

@dataclass
class ResourceEnergyProfile:
    """Energy characteristics of a resource"""
    resource_id: str  # "gpu_0", "cpu_16", "nic_0"

    # Power consumption (Watts)
    p_max_w: float      # Full load power
    p_idle_w: float     # Idle but ready power
    p_hibernate_w: float  # Hibernate power
    p_sleep_w: float    # Deep sleep power

    # Wake energy cost (Joules to transition)
    e_wake_from_hibernate_j: float
    e_wake_from_sleep_j: float

    # Time costs (seconds)
    wake_latency_s: Dict[ResourceState, float]  # state → latency

    # Carbon intensity (optional)
    carbon_intensity_kg_co2_per_kwh: float = 0.0

@dataclass
class ResourceEnergyLease:
    """Energy cost associated with a resource lease"""
    lease_id: str
    resource_id: str
    state: ResourceState

    # Energy consumed during lease (cumulative)
    energy_consumed_j: float = 0.0
    energy_consumed_wh: float = 0.0

    # Cost (monetary)
    energy_cost_credits: float = 0.0

    # Timestamps
    start_time: float = 0.0
    last_update: float = 0.0

    def accumulate(self, duration_s: float, power_w: float, price_per_kwh: float):
        """Accumulate energy and cost for duration at given power"""
        energy_j = power_w * duration_s
        energy_wh = energy_j / 3600
        cost = (energy_wh / 1000) * price_per_kwh  # kWh × $/kWh

        self.energy_consumed_j += energy_j
        self.energy_consumed_wh += energy_wh
        self.energy_cost_credits += cost
        self.last_update = time.time()
```

### Energy Price Model

```python
class EnergyPriceModel:
    """
    Models energy cost for resources based on:
    - Time-of-day rates (if known)
    - Real-time spot prices (if connected to market)
    - Carbon cost (optional carbon tax)
    """

    def __init__(self):
        self.price_schedule = {}  # hour → $/kWh
        self.current_spot_price = 0.15  # $/kWh default
        self.carbon_tax_per_kg = 0.0

    def get_current_price_per_kwh(self, resource_id: str) -> float:
        """Get current effective price for resource"""
        hour = time.localtime().tm_hour
        base_rate = self.price_schedule.get(hour, self.current_spot_price)

        # Add carbon cost
        profile = get_resource_profile(resource_id)
        carbon_cost = profile.carbon_intensity_kg_co2_per_kwh * self.carbon_tax_per_kg
        return base_rate + carbon_cost

    def get_power_for_state(self, profile: ResourceEnergyProfile, state: ResourceState) -> float:
        """Get power consumption for a given state"""
        power_map = {
            ResourceState.FUNCTIONAL: profile.p_idle_w,  # or p_max_w if fully loaded
            ResourceState.HIBERNATE: profile.p_hibernate_w,
            ResourceState.SLEEP: profile.p_sleep_w,
            ResourceState.OFFLINE: 0.0,
        }
        return power_map.get(state, profile.p_idle_w)

    def estimate_lease_cost(self, resource_id: str, state: ResourceState,
                           duration_s: float) -> float:
        """Estimate energy cost for a lease given duration and state"""
        profile = get_resource_profile(resource_id)
        power_w = self.get_power_for_state(profile, state)
        price_kwh = self.get_current_price_per_kwh(resource_id)

        energy_wh = power_w * duration_s / 3600
        energy_kwh = energy_wh / 1000
        return energy_kwh * price_kwh
```

---

## Integration with Existing Components

### Extended ResourceProvider Interface

```python
class EnergyAwareResourceProvider(GeneralResourceProvider):
    """
    Extends ResourceProvider with energy-aware capabilities.
    """

    def __init__(self, energy_model: EnergyPriceModel, policy: EnergyPolicy):
        self.energy_model = energy_model
        self.policy = policy
        self.resource_profiles = {}  # resource_id → ResourceEnergyProfile
        self.active_leases = {}  # lease_id → ResourceEnergyLease
        self.current_states = {}  # resource_id → ResourceState
        self.state_timers = {}  # resource_id → last_transition_time

    async def allocate_with_energy(self, spec: ResourceSpec,
                                   max_price_per_hour: float) -> ResourceLease:
        """
        Allocate resource considering energy cost constraints.

        The provider evaluates:
        1. Can we satisfy resource requirements?
        2. What state will resource be in?
        3. What is projected energy cost?
        4. Does it fit within requester's budget?
        5. Should we wake a sleeping resource (include wake cost)?
        """
        resource_id = self._find_best_resource(spec.resource_type, spec.quantity)
        current_state = self.current_states[resource_id]

        # Compute total cost including wake energy if needed
        estimated_duration = spec.duration_s or 3600  # default 1 hour
        wake_cost = 0.0
        if current_state == ResourceState.SLEEP:
            profile = self.resource_profiles[resource_id]
            wake_cost_j = profile.e_wake_from_sleep_j
            wake_cost = (wake_cost_j / 3600 / 1000) * \
                        self.energy_model.get_current_price_per_kwh(resource_id)
        elif current_state == ResourceState.HIBERNATE:
            profile = self.resource_profiles[resource_id]
            wake_cost_j = profile.e_wake_from_hibernate_j
            wake_cost = (wake_cost_j / 3600 / 1000) * \
                        self.energy_model.get_current_price_per_kwh(resource_id)

        operational_cost = self.energy_model.estimate_lease_cost(
            resource_id, ResourceState.FUNCTIONAL, estimated_duration
        )
        total_cost = operational_cost + wake_cost

        # Check if requester can afford
        cost_per_hour = (total_cost / estimated_duration) * 3600
        if max_price_per_hour < cost_per_hour:
            raise ResourceUnavailable(f"Insufficient budget: need ${cost_per_hour:.3f}/hr, have ${max_price_per_hour:.3f}/hr")

        # Decision: wake if profitable
        expected_revenue = self._compute_expected_revenue(spec)
        if expected_revenue < total_cost:
            raise ResourceUnavailable("Not profitable to wake resource (cost > revenue)")

        # Proceed with allocation
        if current_state != ResourceState.FUNCTIONAL:
            await self._transition_to_functional(resource_id)

        lease = await super().allocate(spec)
        lease.metadata['energy_cost_estimate'] = total_cost
        lease.metadata['start_state'] = current_state.value
        lease.metadata['wake_cost_credits'] = wake_cost

        # Track energy
        self.active_leases[lease.lease_id] = ResourceEnergyLease(
            lease_id=lease.lease_id,
            resource_id=resource_id,
            state=ResourceState.FUNCTIONAL,
            start_time=time.time()
        )

        return lease

    def crowd_request_wake(self, resource_type: ResourceType,
                          quantity: float, duration_s: float,
                          bid_price_credits: float) -> Tuple[bool, Optional[str], Optional[ResourceLease]]:
        """
        External peers can request that this peer wake resources.

        Parameters
        ----------
        resource_type : ResourceType
            Type of resource needed (GPU, CPU, etc.)
        quantity : float
            How much needed (GB, cores, etc.)
        duration_s : float
            How long needed
        bid_price_credits : float
            Maximum price requester will pay per hour

        Returns
        -------
        (accepted, reason, lease) : Tuple[bool, str, Optional[ResourceLease]]
            accepted: True if peer agrees to wake resources
            reason: If rejected, explanation string
            lease: If accepted, the allocated ResourceLease
        """
        # Find suitable resource
        resource_id = self._select_resource_for_type(resource_type, quantity)
        if not resource_id:
            return False, f"No {resource_type} resource available with quantity {quantity}", None

        # Evaluate profitability
        estimated_cost = self.energy_model.estimate_lease_cost(
            resource_id=resource_id,
            state=ResourceState.FUNCTIONAL,
            duration_s=duration_s
        )
        # Add amortized wake cost
        current_state = self.current_states[resource_id]
        if current_state == ResourceState.SLEEP:
            profile = self.resource_profiles[resource_id]
            wake_cost = profile.e_wake_from_sleep_j / 3600 * duration_s
        elif current_state == ResourceState.HIBERNATE:
            profile = self.resource_profiles[resource_id]
            wake_cost = profile.e_wake_from_hibernate_j / 3600 * duration_s
        else:
            wake_cost = 0.0

        total_cost = estimated_cost + wake_cost

        # Convert bid (per hour) to total for duration
        bid_total = (bid_price_credits / 3600) * duration_s

        # Compare to bid
        if bid_total < total_cost:
            return False, f"Bid ${bid_total:.3f} below cost ${total_cost:.3f}", None

        # Check owner policy
        allowed, reason = self.policy.can_wake_for_bid(resource_type, bid_price_credits / duration_s * 3600)
        if not allowed:
            return False, f"Policy violation: {reason}", None

        # Wake resource if in sleep/hibernate
        if current_state != ResourceState.FUNCTIONAL:
            try:
                await self._transition_to_functional(resource_id)
            except Exception as e:
                return False, f"Failed to wake resource: {str(e)}", None

        # Allocate lease
        spec = ResourceSpec(
            resource_type=resource_type,
            quantity=quantity,
            duration_s=duration_s
        )
        try:
            lease = await self.allocate_with_energy(spec, bid_price_credits)
            return True, None, lease
        except Exception as e:
            return False, f"Allocation failed: {str(e)}", None
```

### PeerRegistry Extension

```python
class EnergyAwareRegistry(PeerRegistry):
    """
    Extends registry to track peer energy states and prices.
    """

    def __init__(self):
        super().__init__()
        self.peer_energy_states = {}  # peer_id → {resource_type: state, timestamp, load}
        self.peer_energy_prices = {}  # peer_id → {resource_type: $/kWh}
        self.peer_resource_profiles = {}  # peer_id → ResourceEnergyProfile

    def advertise_energy_state(self, peer_id: str, resource_type: ResourceType,
                              state: ResourceState, current_load: float,
                              price_per_kwh: Optional[float] = None):
        """
        Peers periodically advertise:
        - Current power state of each resource type
        - Current energy price they're charging
        - Current load/utilization
        """
        if peer_id not in self.peer_energy_states:
            self.peer_energy_states[peer_id] = {}

        self.peer_energy_states[peer_id][resource_type] = {
            'state': state,
            'timestamp': time.time(),
            'load': current_load
        }

        if price_per_kwh is not None:
            if peer_id not in self.peer_energy_prices:
                self.peer_energy_prices[peer_id] = {}
            self.peer_energy_prices[peer_id][resource_type] = price_per_kwh

    def query_available_with_energy(self, resource_type: ResourceType,
                                   min_quantity: float,
                                   max_price_per_hour: float,
                                   preferred_states: List[ResourceState] = None
                                  ) -> List[Tuple[str, ResourceOffer, ResourceState, float]]:
        """
        Find peers that can provide resource, considering state.

        Prefers FUNCTIONAL resources, but may include HIBERNATE/SLEEP
        if requester is willing to pay wake-up cost premium.

        Returns
        -------
        List of tuples: (peer_id, offer, state, adjusted_price_per_hour)
        """
        if preferred_states is None:
            preferred_states = [ResourceState.FUNCTIONAL,
                               ResourceState.HIBERNATE,
                               ResourceState.SLEEP]

        candidates = []
        for peer_id, state_info in self.peer_energy_states.items():
            if resource_type not in state_info:
                continue

            state = state_info[resource_type]['state']
            if state not in preferred_states:
                continue

            # Check if resource has capacity
            offer = self._get_peer_offer(peer_id, resource_type)
            if not offer or offer.available < min_quantity:
                continue

            # Base price from offer
            base_price = offer.price_per_unit

            # Adjust price based on state (add wake premium)
            adjusted_price = base_price
            if state == ResourceState.HIBERNATE:
                # Estimate wake premium (amortized over typical 1hr lease)
                profile = self.peer_resource_profiles.get(peer_id, {}).get(resource_type)
                if profile:
                    wake_cost_j = profile.e_wake_from_hibernate_j
                    price_per_kwh = self.peer_energy_prices.get(peer_id, {}).get(resource_type, 0.15)
                    wake_premium = (wake_cost_j / 3600 / 1000) * price_per_kwh
                    adjusted_price += wake_premium
            elif state == ResourceState.SLEEP:
                profile = self.peer_resource_profiles.get(peer_id, {}).get(resource_type)
                if profile:
                    wake_cost_j = profile.e_wake_from_sleep_j
                    price_per_kwh = self.peer_energy_prices.get(peer_id, {}).get(resource_type, 0.15)
                    wake_premium = (wake_cost_j / 3600 / 1000) * price_per_kwh
                    adjusted_price += wake_premium

            # Check if within budget
            if adjusted_price <= max_price_per_hour:
                candidates.append((peer_id, offer, state, adjusted_price))

        # Sort by (state priority, then price)
        def sort_key(item):
            peer_id, offer, state, adjusted_price = item
            state_rank = preferred_states.index(state)
            return (state_rank, adjusted_price)

        candidates.sort(key=sort_key)
        return candidates
```

---

## Owner Policy Configuration

### Configuration File Format

Location: `~/.evollm/energy_policy.yaml`

```yaml
energy_policy:
  # When to automatically hibernate/sleep
  auto_hibernate:
    enabled: true
    idle_timeout_s: 300      # 5 minutes idle → HIBERNATE
    idle_timeout_sleep: 1800 # 30 minutes hibernating → SLEEP

  # When to wake on crowd request
  wake_acceptance:
    min_bid_price_per_kwh: 0.05  # Minimum $/kWh to wake from sleep
    max_daily_wakes: 20          # Throttle to conserve lifespan
    allowed_hours: "06:00-23:00"  # Only wake during these hours
    max_energy_consumption_kwh_per_day: 10.0  # Hard cap

  # Resource-specific policies
  gpu:
    sleep_enabled: true
    hibernate_enabled: true
    min_time_between_wakes_s: 60  # Cool-down to prevent thrashing
    degradation_aware: true  # Factor in wear-and-tear

  cpu:
    sleep_enabled: false  # CPUs wake too fast, not worth it
    hibernate_enabled: false

  network:
    # Bandwidth throttling instead of sleep
    idle_throttle_mbps: 10

  # Carbon awareness
  carbon_aware:
    enabled: false
    max_carbon_intensity_kg_co2_per_kwh: 0.5
    shift_to_low_carbon: true  # Prefer to run when grid is green

  # Scheduled operations (cron-style)
  schedule:
    - resource: gpu
      state: FUNCTIONAL
      cron: "*/5 * * * *"  # Every 5 minutes for cron jobs
    - resource: gpu
      state: SLEEP
      cron: "0 0 * * *"    # Sleep at midnight if idle
```

### EnergyPolicy Class

```python
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import re
from croniter import croniter

@dataclass
class ResourcePolicy:
    """Per-resource policy overrides"""
    sleep_enabled: bool = True
    hibernate_enabled: bool = True
    min_time_between_wakes_s: int = 60
    degradation_aware: bool = False
    max_wakes_per_day: Optional[int] = None

@dataclass
class EnergyPolicy:
    """Global energy management policy"""
    # Auto-transition timeouts
    auto_hibernate_idle_timeout_s: int = 300
    auto_sleep_idle_timeout_s: int = 1800

    # Wake acceptance criteria
    min_bid_price_per_kwh: float = 0.03
    max_daily_wakes: int = 50
    allowed_wake_hours: Tuple[int, int] = (0, 24)  # 24h format
    max_daily_energy_kwh: float = 100.0

    # Resource-specific overrides
    resource_overrides: Dict[str, ResourcePolicy] = field(default_factory=dict)

    # Runtime state (not in config)
    _wakes_today: int = 0
    _energy_consumed_today_kwh: float = 0.0
    _last_reset_day: int = field(default_factory=lambda: time.localtime().tm_yday)
    _last_wake_times: Dict[str, float] = field(default_factory=dict)  # resource_type → timestamp

    def can_wake_for_bid(self, resource_type: str, bid_price_per_kwh: float) -> Tuple[bool, str]:
        """Check if policy allows waking for this bid"""
        # Check daily limit
        self._check_daily_reset()
        if self._wakes_today >= self.max_daily_wakes:
            return False, f"Daily wake limit exceeded ({self._wakes_today}/{self.max_daily_wakes})"

        # Check minimum bid
        if bid_price_per_kwh < self.min_bid_price_per_kwh:
            return False, f"Bid ${bid_price_per_kwh:.3f}/kWh below minimum ${self.min_bid_price_per_kwh:.3f}/kWh"

        # Check allowed hours
        hour = time.localtime().tm_hour
        if not (self.allowed_wake_hours[0] <= hour < self.allowed_wake_hours[1]):
            return False, f"Outside allowed hours {self.allowed_wake_hours[0]}:00-{self.allowed_wake_hours[1]}:00"

        # Check resource-specific cooldown
        resource_policy = self.resource_overrides.get(resource_type, ResourcePolicy())
        last_wake = self._last_wake_times.get(resource_type, 0)
        cooldown = resource_policy.min_time_between_wakes_s
        if time.time() - last_wake < cooldown:
            return False, f"Resource {resource_type} in cooldown ({(time.time()-last_wake):.0f}s < {cooldown}s)"

        # Check max wakes for this resource
        if resource_policy.max_wakes_per_day:
            # Would need per-resource counter (stretch)
            pass

        return True, "OK"

    def record_wake(self, resource_type: str):
        """Record that a wake event occurred"""
        self._check_daily_reset()
        self._wakes_today += 1
        self._last_wake_times[resource_type] = time.time()

    def record_energy_consumption(self, energy_kwh: float):
        """Record energy consumed"""
        self._check_daily_reset()
        self._energy_consumed_today_kwh += energy_kwh

    def _check_daily_reset(self):
        """Reset daily counters if day changed"""
        current_day = time.localtime().tm_yday
        if current_day != self._last_reset_day:
            self._wakes_today = 0
            self._energy_consumed_today_kwh = 0.0
            self._last_reset_day = current_day
            self._last_wake_times.clear()

    def check_daily_limits(self) -> Dict[str, any]:
        """Check if any daily limits are approaching"""
        self._check_daily_reset()
        return {
            'wakes_used': self._wakes_today,
            'wakes_limit': self.max_daily_wakes,
            'wakes_remaining': max(0, self.max_daily_wakes - self._wakes_today),
            'energy_used_kwh': self._energy_consumed_today_kwh,
            'energy_limit_kwh': self.max_daily_energy_kwh,
            'energy_remaining_kwh': max(0, self.max_daily_energy_kwh - self._energy_consumed_today_kwh),
        }
```

---

## Implementation Roadmap

### Phase 1: Energy Profiling & Accounting (Week 1-2)

**Goal**: Add energy metering baseline

#### Tasks

1. **Create `evollm/energy.py`**:
   - `ResourceEnergyProfile` dataclass
   - Default profiles for common hardware (NVIDIA GPUs, CPUs)
   - `EnergyPriceModel` with time-of-day pricing
   - `ResourceEnergyLease` tracking

2. **Extend `ResourceLease`** (if exists) to include:
   ```python
   @dataclass
   class ResourceLease:
       ...
       energy_estimate_wh: Optional[float] = None  # Pre-allocation estimate
       energy_actual_wh: float = 0.0  # Accumulated during lease
       energy_cost_credits: float = 0.0
   ```

3. **Instrument `ResourceManager.allocate()`**:
   - Before allocation: estimate energy cost
   - During lease: periodically sample power and accumulate
   - On release: finalize accounting, credit provider ledger

4. **Add CLI commands**:
   ```bash
   evosh energy profile  # Show current energy profile
   evosh energy estimate --resource gpu --duration 2h  # Estimate cost
   evosh ledger energy --last 24h  # Show energy consumption breakdown
   ```

5. **Unit tests**:
   - Energy calculation accuracy (power × time = energy)
   - Cost calculation with different prices
   - Wake cost amortization

---

### Phase 2: State Management (Week 3-4)

**Goal**: Implement resource state transitions

#### Tasks

1. **Create `evollm/resource_state.py`**:
   ```python
   class ResourceState(Enum):
       FUNCTIONAL = "functional"
       HIBERNATE = "hibernate"
       SLEEP = "sleep"
       OFFLINE = "offline"

   class ResourceStateManager:
       def __init__(self, profile: ResourceEnergyProfile, policy: EnergyPolicy):
           self.state = ResourceState.FUNCTIONAL
           self.state_since = time.time()
           self.policy = policy
           self.profile = profile
           self.power_meter = PowerMeter(profile)  # Could be real sensor or estimator
           self._lock = asyncio.Lock()

       async def update(self):
           """Called periodically to check for auto-transitions"""
           async with self._lock:
               idle_time = time.time() - self.state_since

               if self.state == ResourceState.FUNCTIONAL:
                   if idle_time > self.policy.auto_hibernate_idle_timeout:
                       await self.transition_to(ResourceState.HIBERNATE)
               elif self.state == ResourceState.HIBERNATE:
                   if idle_time > self.policy.auto_sleep_idle_timeout:
                       await self.transition_to(ResourceState.SLEEP)

       async def transition_to(self, new_state: ResourceState):
           """Execute state transition with proper power management"""
           old_state = self.state
           profile = self.profile

           # Compute wake/sleep latency and energy
           if new_state == ResourceState.FUNCTIONAL:
               if old_state == ResourceState.SLEEP:
                   await self._wake_from_sleep()
               elif old_state == ResourceState.HIBERNATE:
                   await self._wake_from_hibernate()
           elif new_state == ResourceState.HIBERNATE:
               await self._enter_hibernate()
           elif new_state == ResourceState.SLEEP:
               await self._enter_sleep()
           elif new_state == ResourceState.OFFLINE:
               await self._shut_down()

           self.state = new_state
           self.state_since = time.time()
           self._notify_listeners(old_state, new_state)

       async def _wake_from_sleep(self):
           """Wake from deep sleep"""
           await asyncio.sleep(self.profile.wake_latency_s[ResourceState.SLEEP])
           # Could issue system calls: nvidia-smi --gpu-reset, etc.

       async def _wake_from_hibernate(self):
           """Wake from hibernate"""
           await asyncio.sleep(self.profile.wake_latency_s[ResourceState.HIBERNATE])

       async def _enter_hibernate(self):
           """Enter hibernate state"""
           # Reduce clock frequencies, power gate unused units
           pass

       async def _enter_sleep(self):
           """Enter sleep state"""
           # Deeper power reduction
           pass

       def _notify_listeners(self, old_state: ResourceState, new_state: ResourceState):
           """Notify observers of state change"""
           pass
   ```

2. **Integrate with `LocalResourceManager`**:
   - Wrap each resource (GPU, CPU cores, disk) with ResourceStateManager
   - On `allocate()`: ensure resource is FUNCTIONAL first (trigger wake if needed)
   - On `release()`: start idle timer, potentially transition to hibernate/sleep
   - Add `get_resource_state()` to ResourceProvider interface

3. **Power measurement abstraction**:
   - GPU: Use NVML (NVIDIA) or ROCm-SMI (AMD) to read real-time power
   - CPU: Use Intel RAPL or `/sys/class/powercap/`
   - Fallback: estimate from profile + utilization
   - Config: `power_measurement: auto|nvml|rapl|manual`

4. **Background state monitor**:
   - Periodic task (every 10s) calls `update()` on all ResourceStateManagers
   - Integrated with event loop or separate thread

5. **Unit tests**:
   - State transition timing
   - Auto-transition after idle timeout
   - Wake latency (mocked)
   - Policy-enforced transitions

---

### Phase 3: Crowd Wake Requests (Week 5-6)

**Goal**: Peers can request others to wake resources

#### Tasks

1. **Add `WakeService` to gRPC protocol** (`peer/protocol.proto`):
   ```protobuf
   service WakeService {
       rpc RequestWake(WakeRequest) returns (WakeResponse);
   }

   message WakeRequest {
       string requesting_peer_id = 1;
       ResourceType resource_type = 2;
       float quantity = 3;
       int64 duration_s = 4;
       float max_price_per_hour = 5;
   }

   message WakeResponse {
       bool accepted = 1;
       string reason = 2;  // if rejected
       Lease lease = 3;    // if accepted, includes allocation
   }
   ```

2. **Regenerate Python stubs**:
   ```bash
   python -m grpc_tools.protoc -I. --python_out=evollm/peer --grpc_python_out=evollm/peer peer/protocol.proto
   ```

3. **Implement `PeerServer.handle_wake_request()`**:
   - Validate requester (mTLS, whitelist check)
   - Query ResourceStateManager: is resource SLEEP/HIBERNATE?
   - Evaluate: is bid price > estimated cost?
   - Check owner policy (max_daily_wakes, allowed hours)
   - If yes: transition to FUNCTIONAL, allocate lease, return
   - If no: return rejection with reason

4. **Implement `PeerClient.request_remote_wake()`**:
   - Try to find peers with needed resource in SLEEP/HIBERNATE
   - Send WakeRequest with bid (based on user's urgency/budget)
   - On success: receive Lease and use normally
   - On failure: fallback to next cheapest peer or local-only
   - Retry logic with exponential backoff

5. **Registry updates**:
   - Peers advertise current state: `{"gpu": "hibernate", "cpu": "functional"}`
   - Query API filters by state preference

6. **Integration tests**:
   - Mock gRPC server simulating peer in SLEEP state
   - Test wake request accepted with sufficient bid
   - Test wake request rejected with insufficient bid
   - Test policy enforcement (daily limit, hours)

---

### Phase 4: Owner Policy & Control (Week 7)

**Goal**: Peer owners control their energy policy

#### Tasks

1. **Create `evollm/energy_policy.py`** (see implementation above)

2. **Add configuration files**:
   - `~/.evollm/energy_policy.yaml` for defaults
   - Can override via CLI: `--energy-policy.min-bid-price 0.05`

3. **CLI for policy management** (`evollm/cli/energy.py`):
   ```bash
   evosh energy policy show
   evosh energy policy set --min-bid-price 0.06
   evosh energy policy set --max-daily-wakes 10
   evosh energy policy enable-auto-hibernate --idle-timeout 600
   evosh energy policy reset-counters
   ```

4. **Real-time monitoring**:
   ```bash
   evosh energy watch  # Live view of:
                      #   States (FUNCTIONAL: 2 GPU, 8 CPU; HIBERNATE: 1 GPU)
                      #   Energy consumption today: 3.2 kWh ($0.48)
                      #   Wakes today: 5/50
                      #   Current price: $0.15/kWh
   ```

5. **Unit tests**:
   - Policy validation (min bid, hours, limits)
   - Daily counter reset at midnight
   - Resource-specific overrides
   - Cron schedule parsing

---

### Phase 5: Advanced Features (Week 8-9)

**Goal**: Optimize with predictions and carbon awareness

#### Predictive Warming

- Observe access patterns: "Inference job runs at 2pm daily"
- Auto-wake 5 minutes before expected
- Simple: cron-like schedule from config
- Advanced: ML model (prophet, LSTM) for prediction

#### Carbon-Aware Scheduling

- Integrate with carbon intensity API (WattTime, ElectricityMaps)
- Prefer to do heavy compute when grid is green
- May voluntarily hibernate during dirty periods
- Config: `max_carbon_intensity_kg_co2_per_kwh`
- Display carbon impact in CLI: `evosh energy carbon-footprint`

#### Energy Market Integration (Advanced)

- Connect to real-time energy spot markets (some regions have 5min pricing)
- Dynamic pricing based on grid load
- Auto-hibernate during price spikes
- Aggressively wake during negative price periods (over-supply)

#### Degradation Modeling

- Factor wake cycles into hardware lifespan
- NVMe: limit spin-up cycles (typically 50k-100k)
- GPU: memory wears with thermal cycles
- Policy: `max_wakes_per_day` with smart grouping

---

### Phase 6: Testing & Validation (Week 10)

#### Unit Tests

1. **State transitions**:
   ```python
   async def test_state_transitions():
       profile = ResourceEnergyProfile(
           resource_id="gpu_0",
           p_max_w=300.0, p_idle_w=50.0, p_hibernate_w=15.0, p_sleep_w=3.0,
           e_wake_from_hibernate_j=1000.0, e_wake_from_sleep_j=5000.0,
           wake_latency_s={ResourceState.FUNCTIONAL: 0.0,
                          ResourceState.HIBERNATE: 1.0,
                          ResourceState.SLEEP: 5.0}
       )
       policy = EnergyPolicy(auto_hibernate_idle_timeout_s=1, auto_sleep_idle_timeout_s=2)
       manager = ResourceStateManager(profile, policy)

       assert manager.state == ResourceState.FUNCTIONAL
       await asyncio.sleep(1.1)
       await manager.update()
       assert manager.state == ResourceState.HIBERNATE
       await asyncio.sleep(2.1)
       await manager.update()
       assert manager.state == ResourceState.SLEEP
   ```

2. **Energy accounting**:
   ```python
   def test_energy_lease_accumulate():
       lease = ResourceEnergyLease(
           lease_id="test",
           resource_id="gpu_0",
           state=ResourceState.FUNCTIONAL
       )
       lease.accumulate(duration_s=3600, power_w=200, price_per_kwh=0.15)
       assert lease.energy_consumed_wh == pytest.approx(200.0)  # 200W * 1h = 200Wh
       expected_cost = (200.0 / 1000) * 0.15  # 0.03 credits
       assert lease.energy_cost_credits == pytest.approx(0.03)
   ```

3. **Policy rejection**:
   ```python
   def test_policy_min_bid():
       policy = EnergyPolicy(min_bid_price_per_kwh=0.10)
       can_wake, reason = policy.can_wake_for_bid('gpu', bid_price_per_kwh=0.05)
       assert not can_wake
       assert "below minimum" in reason
   ```

#### Integration Tests

1. **Multi-peer wake scenario**:
   ```bash
   # Terminal 1: Start Peer A with GPU
   evollm-serve --peer-id A --gpu 1 --energy-policy min-bid 0.05

   # Terminal 2: Start Peer B (CPU-only)
   evollm-serve --peer-id B

   # Peer B requests wake from A
   evosh resources allocate \
     --type gpu \
     --quantity 4.0 \
     --duration 300 \
     --bid-price 0.06 \
     --peer A

   # Verify:
   # - A's GPU transitions SLEEP→FUNCTIONAL
   # - B receives lease, uses GPU
   # - On lease end, A returns to HIBERNATE after timeout
   # - Ledger: B charged $X credits, A earned $X credits
   ```

2. **Policy enforcement**:
   ```bash
   # Peer A sets: max-daily-wakes 3
   evosh energy policy set --max-daily-wakes 3

   # Attempt 4th wake request (should fail)
   for i in {1..4}; do
     evosh resources allocate --type gpu --peer A --bid-price 0.10 || echo "Failed: $i"
   done
   # Output: 3 successes, 1 failure "Daily wake limit exceeded"
   ```

#### Performance Validation

1. **Wake latency**:
   - Measure time from `allocate()` to usable resource
   - Target: < 2s from HIBERNATE, < 10s from SLEEP

2. **Overhead measurement**:
   - Run inference with energy tracking ON vs OFF
   - Measure throughput difference (should be < 1%)

3. **Energy savings**:
   - Scenario: GPU idle 80% of day
   - Without energy management: 100% idle power (300W) for 24h = 7.2 kWh
   - With auto-sleep (16h sleep at 5W): 8h×300W + 16h×5W = 2.4 + 0.08 = 2.48 kWh
   - Expected savings: **65%**

---

## Critical Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `evollm/energy.py` | Energy profiling, accounting, pricing models |
| `evollm/resource_state.py` | State machine for resource states |
| `evollm/energy_policy.py` | Owner policy configuration and enforcement |
| `evollm/cli/energy.py` | CLI commands for energy management |
| `config/energy_policy.example.yaml` | Example policy configuration |
| `tests/unit/test_energy.py` | Unit tests for energy module |
| `tests/unit/test_resource_state.py` | State machine tests |
| `tests/unit/test_energy_policy.py` | Policy tests |
| `tests/integration/test_energy_integration.py` | Integration tests |
| `tests/integration/test_crowd_wake.py` | Crowd wake request tests |

### Modified Files

| File | Changes |
|------|---------|
| `evollm/resource_provider.py` (or create) | Add `get_resource_state()`, `request_wake()` methods; integrate `ResourceStateManager` |
| `evollm/evollm_base.py` | Instrument forward pass with energy accounting; pass energy lease metadata to remote peers |
| `evollm/cache_policy.py` | Consider energy when making eviction decisions; prefer to evict from sleeping peers' cached layers |
| `evollm/hardware_profiler.py` | Add `profile_energy()` method to detect power capabilities; query hardware sensors (NVML, RAPL) |
| `evollm/config.py` | Add energy policy fields to `EvoLLMConfig` (see below) |
| `evollm/peer_server.py` | Add `WakeService` gRPC endpoint; implement policy check on wake requests; log all wake requests |
| `evollm/peer_client.py` | Add `request_peer_wake()` method; include energy price in peer ranking |
| `evollm/ledger.py` | Add energy cost accounting separate from data transfer credits |
| `evollm/cli.py` (or create) | Add `evosh energy *` subcommands |

### Config Extensions (`evollm/config.py`)

```python
@dataclass
class EnergyConfig:
    """Energy-aware resource management configuration"""
    enabled: bool = False  # Default OFF for backward compatibility

    # Policy file
    policy_file: Optional[str] = None  # Path to energy_policy.yaml

    # Auto-state transitions
    auto_hibernate_idle_s: int = 300
    auto_sleep_idle_s: int = 1800

    # Power measurement
    measurement: Literal['auto', 'nvml', 'rapl', 'manual'] = 'auto'

    # Crowd wake requests
    enable_crowd_wake: bool = True
    bid_price_per_kwh: float = 0.10  # What I'm willing to pay for remote wake

    # Carbon awareness
    carbon_aware_enabled: bool = False
    max_carbon_intensity: float = 0.5  # kg CO2/kWh

    # Predictive warming (stretch)
    predictive_warming_enabled: bool = False
    predictive_warming_schedule: Optional[str] = None  # cron expression

@dataclass
class EvoLLMConfig:
    ...
    energy: EnergyConfig = field(default_factory=EnergyConfig)
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Hardware incompatibility** | Can't measure power accurately | Provide software wattmeters (estimate from utilization), allow manual override |
| **Wear from frequent wakes** | Reduced hardware lifespan | Limit max wakes/day, minimum time between wakes, degradation modeling |
| **Policy complexity** | Owners confused | Provide sensible defaults, simple presets (aggressive, conservative, balanced) |
| **Crowd abuse** | Peers constantly waking resources | Require proof-of-work or small deposit for wake requests, reputation system |
| **Grid instability** (real-time markets) | Wild price swings | Cap price volatility, use moving average |
| **Race conditions** (multiple requests) | Over-commitment | Lock resource during state transition, queue requests |
| **State inconsistency** (peer crashes in SLEEP) | Missed wake requests | Periodic heartbeat in registry, mark peers unavailable after timeout |
| **Battery drain** (mobile) | User's device dies | Disable auto-sleep on battery; owner overrides prevent wake on battery |

---

## Backward Compatibility

- **Energy-aware default OFF** in v1 (opt-in)
- All new config fields have defaults
- Existing APIs unchanged (additive only)
- PeerServer API versioned: v1 (no energy), v2 (with energy)
- Peers negotiate protocol version during handshake
- Graceful degradation: peers without energy support ignore state advertisements

---

## Success Criteria

### Phase 1-2 (MVP)

- [ ] Resource states (FUNCTIONAL, HIBERNATE, SLEEP) implemented
- [ ] Energy metering works on NVIDIA GPUs (NVML) and Intel CPUs (RAPL)
- [ ] Automatic hibernate after 5min idle, sleep after 30min
- [ ] Ledger tracks energy consumed and cost
- [ ] 30% reduction in idle power on typical workstation

### Phase 3-4 (Usable)

- [ ] Peers can request wake with bid prices
- [ ] Owner policy controls (min bid, daily limits, hours)
- [ ] CLI tools for monitoring and configuration
- [ ] State advertised in registry

### Phase 5-6 (Production)

- [ ] Carbon-aware scheduling
- [ ] Predictive warming
- [ ] Comprehensive tests pass
- [ ] Documentation complete

---

## Expected Benefits

1. **Cost savings**: 30-70% reduction in energy bills for idle resources
2. **Community efficiency**: Total AI infrastructure energy ↓ by pooling underutilized resources
3. **Demand response**: Can hibernate during grid peak load → grid stability
4. **Carbon reduction**: Enable green computing (run when renewables available)
5. **Economic incentive**: Earn credits by providing resources during demand
6. **Hardware longevity**: Reduce thermal cycles, extend SSD/GPU lifespan
7. **Market flexibility**: Dynamic pricing responds to supply/demand

---

## Open Questions

1. **Power measurement accuracy**: Should we require hardware sensors or allow manual profiles?
   - **Proposal**: Support both. NVML/RAPL preferred, manual override allowed.

2. **Warranty concerns**: Some manufacturers void warranty if power management disabled.
   - **Proposal**: Document risks, make all features configurable, disable by default.

3. **Crowd request spam**: DOS by constant wake requests.
   - **Proposal**: Require proof-of-work or small deposit for wake requests, reputation system.

4. **State consistency**: What if peer crashes in SLEEP and misses wake request?
   - **Proposal**: Periodic heartbeat in registry, mark peers unavailable after timeout.

5. **Multi-resource coordination**: Request GPU+CPU together; wake both or none.
   - **Proposal**: Bundle wake requests across resource types; atomic state transitions.

6. **Battery-powered devices**: How to prevent energy management from draining laptop battery?
   - **Proposal**: Auto-disable when on battery; owner policy can override.

7. **Virtualization**: How to manage power states in VMs/Docker containers?
   - **Proposal**: Only control guest-visible states; hypervisor may override.

---

## References

- **Main Project Plan**: [PLAN.md](../PLAN.md)
- **Resource Economy**: Section in PLAN.md
- **Generalized Resource Sharing**: [docs/generalized_resource_sharing.md](generalized_resource_sharing.md)
- **Modularity & Isolation**: [docs/modularity_isolation_performance.md](modularity_isolation_performance.md)

---

**Next Steps**:
1. Review this design with stakeholders
2. Prioritize phases based on user needs
3. Begin Phase 1 implementation after P2P core is stable
4. Create detailed technical design documents for each phase

**Maintainer**: @dacineu
**Last Updated**: 2026-03-17
