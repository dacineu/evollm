// Simulation class
// Main simulation engine that coordinates all components

class Simulation {
    constructor(canvas) {
        this.canvas = canvas;
        this.renderer = new Renderer(canvas);
        this.energy_model = new EnergyPriceModel();
        this.peers = [];
        this.leases = [];
        this.wake_requests = [];
        this.events = [];

        this.running = false;
        this.paused = false;
        this.speed = 1.0;
        this.time = 0; // Simulation time in seconds
        this.sim_hour = 0; // Current hour (0-23)

        this.selected_peer = null;
        this.last_frame_time = 0;

        // Stats
        this.stats = {
            total_wakes: 0,
            energy_kwh: 0,
            ledger_balance: 0,
        };
    }

    start() {
        this.running = true;
        this.paused = false;
        this.last_frame_time = performance.now();
        this.loop();
    }

    pause() {
        this.paused = true;
    }

    resume() {
        this.paused = false;
        this.last_frame_time = performance.now();
    }

    reset() {
        this.peers = [];
        this.leases = [];
        this.wake_requests = [];
        this.events = [];
        this.time = 0;
        this.selected_peer = null;
        this.stats = { total_wakes: 0, energy_kwh: 0, ledger_balance: 0 };
        this.energy_model = new EnergyPriceModel();
    }

    addPeer(config = {}) {
        const id = `peer-${String.fromCharCode(65 + this.peers.length)}-${Math.floor(Math.random() * 1000)}`;

        // Calculate position based on current canvas size
        // If canvas is zero-sized, use fallback coordinates
        let x, y;
        if (this.canvas.width > 100 && this.canvas.height > 100) {
            x = config.x || Math.random() * (this.canvas.width - 100) + 50;
            y = config.y || Math.random() * (this.canvas.height - 100) + 50;
        } else {
            // Fallback: use window size or default
            x = config.x || Math.random() * (window.innerWidth - 200) + 100;
            y = config.y || Math.random() * (window.innerHeight - 200) + 100;
        }

        const peer = new Peer(id, x, y, config);
        this.peers.push(peer);
        this.log(`Added ${id} at (${Math.round(x)}, ${Math.round(y)}) - total: ${peer.totalResources()}`, 'success');
        return peer;
    }

    update(dt) {
        if (this.paused) return;

        // Scale dt by simulation speed
        dt *= this.speed;
        this.time += dt;

        // Update simulation hour (1 hour per 60 seconds of sim time at 1x speed)
        this.sim_hour = (this.time / 60) % 24;

        // Update energy model
        this.energy_model.update();

        // Update peers
        for (const peer of this.peers) {
            peer.update(dt, this.sim_hour, this.energy_model);
            peer.updateColor(dt);
        }

        // Update leases
        for (const lease of this.leases) {
            lease.update(dt, this.energy_model);
            if (lease.isExpired()) {
                this.terminateLease(lease);
            }
        }

        // Update wake requests
        for (const wr of this.wake_requests) {
            wr.update(dt);
            if (!wr.accepted && wr.progress >= 1) {
                // Request reached target, evaluate it
                const result = wr.provider.evaluateWakeRequest(wr);
                wr.accepted = result.accepted;
                wr.reason = result.reason;

                if (wr.accepted) {
                    // Create lease
                    const lease = wr.provider.allocate({
                        resource_type: wr.resource_type,
                        quantity: wr.quantity,
                        duration_s: wr.duration,
                    });
                    if (lease) {
                        lease.requester = wr.requester;
                        lease.provider = wr.provider;
                        wr.requester.active_leases.push(lease);
                        this.leases.push(lease);
                        this.log(`Wake accepted: ${wr.requester.id} → ${wr.provider.id} for ${wr.quantity} ${wr.resource_type}`, 'success');
                        this.stats.total_wakes++;
                    }
                } else {
                    this.log(`Wake rejected: ${wr.provider.id} - ${wr.reason}`, 'warning');
                }
            }
        }

        // Remove expired wake requests
        this.wake_requests = this.wake_requests.filter(wr => !wr.isExpired());

        // Update stats
        this.updateStats();
    }

    updateStats() {
        // Total energy consumed
        let energy = 0;
        for (const peer of this.peers) {
            for (const lease of peer.active_leases) {
                energy += lease.energy_wh;
            }
        }
        this.stats.energy_kwh = energy / 1000;

        // Total ledger balance (sum of all peers)
        let balance = 0;
        for (const peer of this.peers) {
            balance += peer.ledger_balance;
        }
        this.stats.ledger_balance = balance;
    }

    terminateLease(lease) {
        // Release from provider
        lease.provider.releaseLease(lease);
        // Remove from requester's active leases
        const idx = lease.requester.active_leases.indexOf(lease);
        if (idx > -1) {
            lease.requester.active_leases.splice(idx, 1);
        }
        // Remove from simulation
        const lease_idx = this.leases.indexOf(lease);
        if (lease_idx > -1) {
            this.leases.splice(lease_idx, 1);
        }
        this.log(`Lease ${lease.id} completed`, 'info');
    }

    getSimHour() {
        return this.sim_hour;
    }

    getCurrentEnergyPrice() {
        return this.energy_model.getCurrentPrice(this.sim_hour);
    }

    render() {
        this.renderer.clear();

        // Draw wake requests (behind peers)
        for (const wr of this.wake_requests) {
            this.renderer.drawWakeRequest(wr);
        }

        // Draw leases (between peers)
        for (const lease of this.leases) {
            this.renderer.drawLease(lease);
        }

        // Draw peers
        for (const peer of this.peers) {
            this.renderer.drawPeer(peer);
        }

        // Draw legend (handled by HTML overlay)
    }

    loop(timestamp = 0) {
        if (!this.running) return;

        const dt = (timestamp - this.last_frame_time) / 1000;
        this.last_frame_time = timestamp;

        // Cap dt to prevent huge jumps
        const capped_dt = Math.min(dt, 0.1);

        try {
            this.update(capped_dt);
            this.render();
        } catch (error) {
            console.error('Simulation loop error:', error);
            this.paused = true;
            this.log(`Render error: ${error.message}`, 'error');
        }

        requestAnimationFrame((t) => this.loop(t));
    }

    log(message, type = 'info') {
        const entry = {
            time: new Date().toLocaleTimeString(),
            message,
            type,
        };
        this.events.unshift(entry);
        if (this.events.length > 100) {
            this.events.pop();
        }

        // Emit event for UI to update
        const event = new CustomEvent('simulation-log', { detail: entry });
        window.dispatchEvent(event);
    }

    selectPeer(peer) {
        // Deselect previous
        if (this.selected_peer) {
            this.selected_peer.selected = false;
        }

        this.selected_peer = peer;
        if (peer) {
            peer.selected = true;
            // Emit event for UI to show details
            const event = new CustomEvent('peer-selected', { detail: peer });
            window.dispatchEvent(event);
        } else {
            const event = new CustomEvent('peer-selected', { detail: null });
            window.dispatchEvent(event);
        }
    }

    sendWakeRequest(requester, provider, resource_type, quantity, duration_s, bid_price_per_kwh) {
        if (requester === provider) {
            this.log('Cannot send wake request to self', 'warning');
            return false;
        }

        const wr = new WakeRequest(
            requester, provider, resource_type, quantity, duration_s, bid_price_per_kwh
        );
        this.wake_requests.push(wr);
        this.log(`Wake request sent: ${requester.id} → ${provider.id} (bid: $${bid_price_per_kwh}/kWh)`, 'info');
        return true;
    }

    getStats() {
        return {
            ...this.stats,
            peer_count: this.peers.length,
            lease_count: this.leases.length,
            wake_request_count: this.wake_requests.length,
            sim_hour: this.sim_hour.toFixed(2),
            energy_price: this.getCurrentEnergyPrice().toFixed(4),
            carbon_intensity: this.energy_model.getCarbonIntensity().toFixed(3),
        };
    }

    getScenario(scenario_name) {
        const scenarios = {
            'small': () => {
                this.reset();
                for (let i = 0; i < 5; i++) {
                    this.addPeer({
                        gpuGB: 16 + Math.floor(Math.random() * 16),
                        cpuCores: 4 + Math.floor(Math.random() * 4),
                        ramGB: 32 + Math.floor(Math.random() * 32),
                        idleTimeout: 300,
                    });
                }
            },
            'large': () => {
                this.reset();
                for (let i = 0; i < 20; i++) {
                    this.addPeer({
                        gpuGB: 8 + Math.floor(Math.random() * 24),
                        cpuCores: 2 + Math.floor(Math.random() * 8),
                        ramGB: 16 + Math.floor(Math.random() * 48),
                        idleTimeout: 200 + Math.random() * 400,
                    });
                }
            },
            'energy-crisis': () => {
                this.reset();
                this.energy_model.triggerPriceSpike(Infinity, 5.0);
                for (let i = 0; i < 10; i++) {
                    this.addPeer({
                        gpuGB: 16,
                        cpuCores: 8,
                        ramGB: 64,
                        idleTimeout: 600, // Longer idle before sleep
                        minBid: 0.50, // High min bid
                    });
                }
            },
            'datacenter': () => {
                this.reset();
                this.energy_model.dynamic_enabled = true;
                for (let i = 0; i < 15; i++) {
                    this.addPeer({
                        gpuGB: 32,
                        cpuCores: 16,
                        ramGB: 128,
                        idleTimeout: 120, // Aggressive sleep
                        state: 'functional',
                    });
                }
            }
        };

        const scenario = scenarios[scenario_name];
        if (scenario) {
            scenario();
            this.log(`Loaded scenario: ${scenario_name}`, 'info');
            return true;
        }
        return false;
    }
}

window.Simulation = Simulation;
