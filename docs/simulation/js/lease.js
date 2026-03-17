// Lease class
// Represents a resource allocation between peers

class Lease {
    constructor(id, spec, provider, requester, duration_s) {
        this.id = id;
        this.spec = spec;
        this.provider = provider;
        this.requester = requester;
        this.duration_s = duration_s;
        this.remaining_s = duration_s;
        this.startTime = Date.now();
        this.energy_wh = 0;
        this.cost_credits = 0;
        this.progress = 0;
        this.particles = []; // For visualization
    }

    update(dt, energy_model) {
        this.remaining_s -= dt;
        this.progress = 1 - (this.remaining_s / this.duration_s);

        // Accumulate energy
        const resource = this.provider.resources[this.spec.resource_type];
        if (resource) {
            const profile = energy_model.getResourcePowerProfile(this.spec.resource_type);
            // Calculate power based on state and utilization
            let base_power;
            if (this.provider.state === 'functional') {
                base_power = profile.p_idle_w + (profile.p_max_w - profile.p_idle_w) * 0.5;
            } else if (this.provider.state === 'hibernate') {
                base_power = profile.p_hibernate_w;
            } else {
                base_power = profile.p_sleep_w;
            }

            // Scale by quantity
            const power_w = base_power * (this.spec.quantity / profile.base_quantity);
            const energy_wh = (power_w * dt) / 3600;
            this.energy_wh += energy_wh;

            // Calculate cost
            const price_per_kwh = energy_model.getCurrentPrice(energy_model.getSimHour ? energy_model.getSimHour() : new Date().getHours());
            this.cost_credits += (energy_wh / 1000) * price_per_kwh;
        }

        // Update particles
        this.updateParticles(dt);
    }

    updateParticles(dt) {
        // Add new particles occasionally
        if (Math.random() < 0.1) {
            this.particles.push({
                t: 0, // 0 to 1 along path
                size: 2 + Math.random() * 3,
                speed: 0.3 + Math.random() * 0.5,
            });
        }

        // Update existing particles
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            p.t += p.speed * dt;
            if (p.t >= 1) {
                this.particles.splice(i, 1);
            }
        }
    }

    isExpired() {
        return this.remaining_s <= 0;
    }
}

window.Lease = Lease;
