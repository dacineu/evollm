// Energy Pricing Model
// Simulates dynamic energy costs based on time-of-day and carbon intensity

class EnergyPriceModel {
    constructor() {
        this.base_price = 0.15;  // $/kWh base rate
        this.peak_multiplier = 2.5;  // Peak hours are more expensive
        this.off_peak_multiplier = 0.6;  // Off-peak cheaper
        this.carbon_tax_per_kg = 0.0;  // $/kg CO2
        this.carbon_intensity = 0.3;  // kg CO2/kWh (average grid)
        this.dynamic_enabled = false;
        this.price_spike_active = false;
        this.price_spike_end = 0;
        this.price_spike_multiplier = 5.0;
        this.time_scale = 1.0;  // Simulation speed affects time progression

        // Time-of-day schedule: { hour: multiplier }
        this.time_of_day_schedule = {
            // Night (12am-6am): cheap
            0: 0.6, 1: 0.6, 2: 0.5, 3: 0.5, 4: 0.55, 5: 0.6,
            // Morning (6am-9am): moderate
            6: 0.8, 7: 1.0, 8: 1.1, 9: 1.0,
            // Midday (9am-5pm): base
            10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.1,
            // Evening peak (5pm-9pm): expensive
            17: 1.3, 18: 1.5, 19: 2.0, 20: 2.2, 21: 2.0, 22: 1.5,
            // Late evening (9pm-12am): decreasing
            23: 1.2,
        };
    }

    /**
     * Get current energy price based on simulation time
     * @param {number} sim_hour - Current hour in simulation (0-23, can be fractional)
     * @returns {number} Price in $/kWh
     */
    getCurrentPrice(sim_hour) {
        let price = this.base_price;

        // Apply time-of-day multiplier
        const hour = Math.floor(sim_hour) % 24;
        let tod_multiplier = this.time_of_day_schedule[hour] || 1.0;

        // Smooth interpolation between hours
        const fraction = sim_hour - hour;
        const next_hour = (hour + 1) % 24;
        const next_multiplier = this.time_of_day_schedule[next_hour] || 1.0;
        const smoothed_multiplier = tod_multiplier + (next_multiplier - tod_multiplier) * fraction;

        price *= smoothed_multiplier;

        // Add carbon tax
        const carbon_cost = this.carbon_intensity * this.carbon_tax_per_kg;
        price += carbon_cost;

        // Apply price spike if active
        if (this.price_spike_active && this.price_spike_end > Date.now()) {
            price *= this.price_spike_multiplier;
        }

        return price;
    }

    /**
     * Get current carbon intensity
     * @returns {number} kg CO2/kWh
     */
    getCarbonIntensity() {
        return this.carbon_intensity;
    }

    /**
     * Trigger a temporary price spike
     * @param {number} duration_ms - Duration in milliseconds
     * @param {number} multiplier - Price multiplier (default 5x)
     */
    triggerPriceSpike(duration_ms = 300000, multiplier = 5.0) {
        this.price_spike_active = true;
        this.price_spike_end = Date.now() + duration_ms;
        this.price_spike_multiplier = multiplier;
    }

    /**
     * Update model (call each tick)
     */
    update() {
        // Check if price spike should end
        if (this.price_spike_active && Date.now() > this.price_spike_end) {
            this.price_spike_active = false;
        }
    }

    /**
     * Estimate energy cost for a resource allocation
     * @param {string} resource_type - Type of resource (e.g., 'gpu_memory_gb')
     * @param {number} quantity - Amount allocated
     * @param {string} state - Resource state (functional, hibernate, sleep)
     * @param {number} duration_s - Duration in seconds
     * @param {number} current_price - Current $/kWh
     * @returns {number} Cost in credits
     */
    estimateAllocationCost(resource_type, quantity, state, duration_s, current_price) {
        // Get power consumption profile for this resource type
        const profile = this.getResourcePowerProfile(resource_type);

        // Power depends on state and utilization
        let power_w;
        if (state === 'functional') {
            // Assume 50% utilization for idle functional state
            power_w = profile.p_idle_w + (profile.p_max_w - profile.p_idle_w) * 0.5;
        } else if (state === 'hibernate') {
            power_w = profile.p_hibernate_w;
        } else if (state === 'sleep') {
            power_w = profile.p_sleep_w;
        } else {
            power_w = 0;
        }

        // Scale power by quantity (assume linear scaling for simplicity)
        power_w *= quantity / profile.base_quantity;

        // Energy consumed
        const energy_wh = (power_w * duration_s) / 3600;
        const energy_kwh = energy_wh / 1000;

        // Cost
        return energy_kwh * current_price;
    }

    /**
     * Get power profile for a resource type
     * @param {string} resource_type
     * @returns {Object} Power profile with watts for different states
     */
    getResourcePowerProfile(resource_type) {
        // Default profiles (would be configured per-hardware in real implementation)
        const profiles = {
            'gpu_memory_gb': {
                base_quantity: 1,  // GB
                p_max_w: 20.0,     // Full load: 20W per GB (HBM3 ~300W for 24GB)
                p_idle_w: 5.0,     // Idle but ready
                p_hibernate_w: 0.5, // Hibernate: clock gated
                p_sleep_w: 0.1,    // Deep sleep: near zero
            },
            'cpu_cores': {
                base_quantity: 1,  // core
                p_max_w: 15.0,     // Full load per core
                p_idle_w: 3.0,     // Idle with C-state
                p_hibernate_w: 0.5, // Deep C-state
                p_sleep_w: 0.05,   // Package C-state
            },
            'ram_gb': {
                base_quantity: 1,  // GB
                p_max_w: 3.0,      // Active memory
                p_idle_w: 1.5,     // Self-refresh
                p_hibernate_w: 0.3,
                p_sleep_w: 0.1,
            },
            'model_layer': {
                base_quantity: 1,  // layer
                p_max_w: 50.0,     // Loading and computing with layer
                p_idle_w: 10.0,    // Cached in memory
                p_hibernate_w: 1.0,
                p_sleep_w: 0.2,
            }
        };

        return profiles[resource_type] || {
            base_quantity: 1,
            p_max_w: 10.0,
            p_idle_w: 2.0,
            p_hibernate_w: 0.5,
            p_sleep_w: 0.1,
        };
    }
}

// Export for use in other modules
window.EnergyPriceModel = EnergyPriceModel;
