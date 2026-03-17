// Peer class
// Represents a node in the EvoOS network with resources and energy state

class Peer {
    constructor(id, x, y, config = {}) {
        this.id = id;
        // Ensure valid coordinates
        this.x = isFinite(x) ? x : 100;
        this.y = isFinite(y) ? y : 100;
        this.vx = (Math.random() - 0.5) * 20; // Random velocity
        this.vy = (Math.random() - 0.5) * 20;

        // Resources (will be initialized based on config)
        this.resources = {};
        this.initializeResources(config);

        // Energy and state
        this.state = 'functional'; // functional, hibernate, sleep, offline
        this.state_since = Date.now();
        this.idle_since = Date.now();

        // Policy
        this.policy = {
            min_bid_price_per_kwh: config.minBid || 0.03,
            max_daily_wakes: config.maxWakes || 50,
            auto_hibernate_idle_s: config.idleTimeout || 300,
            auto_sleep_idle_s: (config.idleTimeout || 300) * 2,
        };

        // Ledger
        this.ledger_balance = config.initialBalance || (Math.random() * 50);
        this.wakes_today = 0;
        this.energy_consumed_today_kwh = 0;
        this.last_reset_day = new Date().getUTCDay();

        // Active leases
        this.active_leases = [];

        // Visual
        this.selected = false;
        this.radius = this.calculateRadius();
        this.last_state_change = 0;
        this.state_transition_progress = 0;
        this.current_color = this.getStateColor(this.state);
    }

    initializeResources(config) {
        // Default resource configuration
        const gpu_gb = config.gpuGB || Math.floor(Math.random() * 32) + 8;
        const cpu_cores = config.cpuCores || Math.floor(Math.random() * 8) + 4;
        const ram_gb = config.ramGB || Math.floor(Math.random() * 64) + 16;

        this.resources = {
            [ResourceType.GPU_MEMORY_GB]: new Resource(ResourceType.GPU_MEMORY_GB, gpu_gb),
            [ResourceType.CPU_CORES]: new Resource(ResourceType.CPU_CORES, cpu_cores),
            [ResourceType.RAM_GB]: new Resource(ResourceType.RAM_GB, ram_gb),
        };

        // Set initial states (some peers start sleeping/hibernating)
        const states = ['functional', 'functional', 'functional', 'hibernate', 'sleep'];
        const initial_state = config.state || states[Math.floor(Math.random() * states.length)];
        Object.values(this.resources).forEach(r => r.setState(initial_state));
        this.state = initial_state;
    }

    calculateRadius() {
        // Size based on total resource capacity
        let totalCapacity = 0;
        Object.values(this.resources).forEach(r => {
            totalCapacity += r.total;
        });
        return 15 + Math.sqrt(totalCapacity) * 2;
    }

    update(dt, sim_time, energy_model) {
        // Update position (bounce off walls)
        const canvas = document.getElementById('sim-canvas');
        if (!canvas) return;

        this.x += this.vx * dt;
        this.y += this.vy * dt;

        if (this.x < this.radius || this.x > canvas.width - this.radius) {
            this.vx *= -1;
            this.x = Math.max(this.radius, Math.min(canvas.width - this.radius, this.x));
        }
        if (this.y < this.radius || this.y > canvas.height - this.radius) {
            this.vy *= -1;
            this.y = Math.max(this.radius, Math.min(canvas.height - this.radius, this.y));
        }

        // Update state based on activity
        this.updateEnergyState(dt, sim_time);

        // Update active leases
        this.active_leases = this.active_leases.filter(lease => {
            lease.remaining_s -= dt;
            if (lease.remaining_s <= 0) {
                this.releaseLease(lease);
                return false;
            }
            return true;
        });
    }

    updateEnergyState(dt, sim_time) {
        // Check if any allocations active
        const hasAllocations = this.active_leases.length > 0;

        if (hasAllocations) {
            this.idle_since = Date.now();
            this.setState('functional');
        } else {
            const idle_s = (Date.now() - this.idle_since) / 1000;

            if (idle_s > this.policy.auto_sleep_idle_s) {
                this.setState('sleep');
            } else if (idle_s > this.policy.auto_hibernate_idle_s) {
                this.setState('hibernate');
            } else {
                this.setState('functional');
            }
        }

        // Update daily counters
        this.checkDailyReset();
    }

    setState(newState) {
        if (this.state !== newState) {
            const old_state = this.state;
            this.state = newState;
            this.state_since = Date.now();
            this.last_state_change = Date.now();
            this.state_transition_progress = 0;

            // Update all resources to match peer state
            Object.values(this.resources).forEach(r => r.setState(newState));

            // Log event
            if (window.simulation && window.simulation.log) {
                window.simulation.log(`Peer ${this.id} state: ${old_state} → ${newState}`, 'event');
            }

            return true;
        }
        return false;
    }

    checkDailyReset() {
        const current_day = new Date().getUTCDay();
        if (current_day !== this.last_reset_day) {
            this.wakes_today = 0;
            this.energy_consumed_today_kwh = 0;
            this.last_reset_day = current_day;
        }
    }

    canFulfill(spec) {
        const resource = this.resources[spec.resource_type];
        if (!resource) return false;
        return resource.canAllocate(spec.quantity);
    }

    allocate(spec) {
        if (!this.canFulfill(spec)) return null;

        const resource = this.resources[spec.resource_type];
        if (!resource.allocate(spec.quantity)) return null;

        const lease = {
            id: this.generateLeaseId(),
            spec: spec,
            provider: this,
            requester: null, // Will be set when lease is assigned
            startTime: Date.now(),
            remaining_s: spec.duration_s,
            energy_wh: 0,
        };

        this.active_leases.push(lease);
        return lease;
    }

    releaseLease(lease) {
        const resource = this.resources[lease.spec.resource_type];
        if (resource) {
            resource.release(lease.spec.quantity);
        }
        // Remove from active leases
        const idx = this.active_leases.indexOf(lease);
        if (idx > -1) {
            this.active_leases.splice(idx, 1);
        }
    }

    evaluateWakeRequest(request) {
        // Check daily wake limit
        if (this.wakes_today >= this.policy.max_daily_wakes) {
            return { accepted: false, reason: 'Daily wake limit reached' };
        }

        // Check bid price
        const estimated_cost = this.estimateOperationalCost(request);
        const wake_cost = this.getWakeEnergyCost();
        const total_cost = estimated_cost + wake_cost;

        if (request.bid_price_total < total_cost) {
            return { accepted: false, reason: `Bid $${request.bid_price_total.toFixed(4)} below cost $${total_cost.toFixed(4)}` };
        }

        return { accepted: true };
    }

    estimateOperationalCost(request) {
        // Rough estimate: power * duration * price
        const resource = this.resources[request.resource_type];
        if (!resource) return Infinity;

        // Get power profile
        const profile = window.energy_model ? window.energy_model.getResourcePowerProfile(request.resource_type) : null;
        if (!profile) return Infinity;

        // Assume 50% utilization during lease
        const power_w = (profile.p_max_w + profile.p_idle_w) / 2 * request.quantity;
        const energy_wh = (power_w * request.duration) / 3600;
        const energy_kwh = energy_wh / 1000;
        const current_price = window.energy_model ? window.energy_model.getCurrentPrice(window.simulation.getSimHour()) : 0.15;

        return energy_kwh * current_price;
    }

    getWakeEnergyCost() {
        // Wake energy cost in credits
        const profile = window.energy_model ? window.energy_model.getResourcePowerProfile(this.getPrimaryResourceType()) : null;
        if (!profile) return 0;

        const wake_energy_j = this.state === 'sleep' ? profile.e_wake_from_sleep_j || 5000 : profile.e_wake_from_hibernate_j || 1000;
        const wake_energy_wh = wake_energy_j / 3600 / 1000; // Convert J to kWh
        const current_price = window.energy_model ? window.energy_model.getCurrentPrice(window.simulation.getSimHour()) : 0.15;

        return wake_energy_wh * current_price;
    }

    getPrimaryResourceType() {
        // Return the resource type with the most capacity
        let max_type = ResourceType.GPU_MEMORY_GB;
        let max_total = 0;
        Object.entries(this.resources).forEach(([type, res]) => {
            if (res.total > max_total) {
                max_total = res.total;
                max_type = type;
            }
        });
        return max_type;
    }

    getPrimaryState() {
        // Return the state of the primary resource
        const primary_type = this.getPrimaryResourceType();
        const primary_res = this.resources[primary_type];
        return primary_res ? primary_res.state : 'functional';
    }

    totalResources() {
        let total = 0;
        Object.values(this.resources).forEach(r => {
            total += r.total;
        });
        return total;
    }

    getResourceUtilization(resource_type) {
        const res = this.resources[resource_type];
        return res ? res.getUtilization() : 0;
    }

    generateLeaseId() {
        return `lease-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`;
    }

    getStateColor(state) {
        const colors = {
            functional: '#4caf50',
            hibernate: '#ff9800',
            sleep: '#2196f3',
            offline: '#9e9e9e',
        };
        return colors[state] || colors.offline;
    }

    updateColor(dt) {
        // Smooth color transition
        const target_color = this.getStateColor(this.state);
        this.current_color = this.lerpColor(this.current_color, target_color, dt * 2);
    }

    lerpColor(color1, color2, t) {
        // Simple color interpolation
        const c1 = this.hexToRgb(color1);
        const c2 = this.hexToRgb(color2);
        if (!c1 || !c2) return color2;

        const r = Math.round(c1.r + (c2.r - c1.r) * t);
        const g = Math.round(c1.g + (c2.g - c1.g) * t);
        const b = Math.round(c1.b + (c2.b - c1.b) * t);
        return `rgb(${r}, ${g}, ${b})`;
    }

    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }
}

window.Peer = Peer;
