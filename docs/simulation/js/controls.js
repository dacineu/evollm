// Controls module
// Handles UI events and connects DOM elements to simulation

class Controls {
    constructor(simulation) {
        this.sim = simulation;

        // Cache DOM elements
        this.elements = {
            btnStart: document.getElementById('btn-start'),
            btnPause: document.getElementById('btn-pause'),
            btnReset: document.getElementById('btn-reset'),
            btnAddPeer: document.getElementById('btn-add-peer'),
            simSpeed: document.getElementById('sim-speed'),
            simSpeedValue: document.getElementById('sim-speed-value'),
            dynamicPricing: document.getElementById('dynamic-pricing'),
            btnPriceSpike: document.getElementById('btn-price-spike'),
            peerDetails: document.getElementById('peer-details'),
            peerId: document.getElementById('peer-id'),
            peerState: document.getElementById('peer-state'),
            peerBalance: document.getElementById('peer-balance'),
            peerWakes: document.getElementById('peer-wakes'),
            peerResources: document.getElementById('peer-resources'),
            peerLeases: document.getElementById('peer-leases'),
            policyMinBid: document.getElementById('policy-min-bid'),
            policyMaxWakes: document.getElementById('policy-max-wakes'),
            policyIdleTimeout: document.getElementById('policy-idle-timeout'),
            btnApplyPolicy: document.getElementById('btn-apply-policy'),
            btnRequestWake: document.getElementById('btn-request-wake'),
            wakeRequestForm: document.getElementById('wake-request-form'),
            wakeResourceType: document.getElementById('wake-resource-type'),
            wakeQuantity: document.getElementById('wake-quantity'),
            wakeQuantityValue: document.getElementById('wake-quantity-value'),
            wakeDuration: document.getElementById('wake-duration'),
            wakeDurationValue: document.getElementById('wake-duration-value'),
            wakeBid: document.getElementById('wake-bid'),
            btnSendWake: document.getElementById('btn-send-wake'),
            btnCancelWake: document.getElementById('btn-cancel-wake'),
            eventLog: document.getElementById('event-log'),
            statPeers: document.getElementById('stat-peers'),
            statLeases: document.getElementById('stat-leases'),
            statWakes: document.getElementById('stat-wakes'),
            statEnergy: document.getElementById('stat-energy'),
            energyPrice: document.getElementById('energy-price'),
            carbonIntensity: document.getElementById('carbon-intensity'),
            simTime: document.getElementById('sim-time'),
        };

        this.bindEvents();
        this.startStatsUpdate();
    }

    bindEvents() {
        // Sim control buttons
        this.elements.btnStart.addEventListener('click', () => {
            this.sim.start();
            this.elements.btnStart.disabled = true;
            this.elements.btnPause.disabled = false;
        });

        this.elements.btnPause.addEventListener('click', () => {
            this.sim.pause();
            this.elements.btnStart.disabled = false;
            this.elements.btnPause.disabled = true;
        });

        this.elements.btnReset.addEventListener('click', () => {
            this.sim.reset();
            this.elements.btnStart.disabled = false;
            this.elements.btnPause.disabled = true;
            this.clearPeerDetails();
        });

        this.elements.btnAddPeer.addEventListener('click', () => {
            this.sim.addPeer();
            this.updateStats();
        });

        // Speed slider
        this.elements.simSpeed.addEventListener('input', (e) => {
            this.sim.speed = parseFloat(e.target.value);
            this.elements.simSpeedValue.textContent = this.sim.speed.toFixed(1) + 'x';
        });

        // Dynamic pricing
        this.elements.dynamicPricing.addEventListener('change', (e) => {
            this.sim.energy_model.dynamic_enabled = e.target.checked;
            this.log(e.target.checked ? 'Dynamic pricing enabled' : 'Dynamic pricing disabled', 'info');
        });

        this.elements.btnPriceSpike.addEventListener('click', () => {
            this.sim.energy_model.triggerPriceSpike(300000, 5.0); // 5 minutes, 5x price
            this.log('Price spike triggered! (+5x for 5 min)', 'warning');
        });

        // Peer selection events
        window.addEventListener('peer-selected', (e) => {
            this.showPeerDetails(e.detail);
        });

        // Policy apply
        this.elements.btnApplyPolicy.addEventListener('click', () => {
            const peer = this.sim.selected_peer;
            if (!peer) return;

            peer.policy.min_bid_price_per_kwh = parseFloat(this.elements.policyMinBid.value);
            peer.policy.max_daily_wakes = parseInt(this.elements.policyMaxWakes.value);
            peer.policy.auto_hibernate_idle_s = parseInt(this.elements.policyIdleTimeout.value);
            peer.policy.auto_sleep_idle_s = parseInt(this.elements.policyIdleTimeout.value) * 2;

            this.log(`Policy updated for ${peer.id}`, 'success');
        });

        // Wake request UI
        this.elements.wakeQuantity.addEventListener('input', (e) => {
            this.elements.wakeQuantityValue.textContent = e.target.value;
        });

        this.elements.wakeDuration.addEventListener('input', (e) => {
            this.elements.wakeDurationValue.textContent = e.target.value;
        });

        this.elements.btnRequestWake.addEventListener('click', () => {
            this.elements.wakeRequestForm.style.display = 'block';
            this.elements.btnRequestWake.disabled = true;
        });

        this.elements.btnCancelWake.addEventListener('click', () => {
            this.elements.wakeRequestForm.style.display = 'none';
            this.elements.btnRequestWake.disabled = false;
        });

        this.elements.btnSendWake.addEventListener('click', () => {
            const requester = this.sim.selected_peer;
            // Choose a different peer as provider
            const available_peers = this.sim.peers.filter(p => p !== requester && p.state !== 'offline');
            if (available_peers.length === 0) {
                this.log('No peers available for wake request', 'warning');
                return;
            }
            const provider = available_peers[Math.floor(Math.random() * available_peers.length)];

            const resource_type = this.elements.wakeResourceType.value;
            const quantity = parseInt(this.elements.wakeQuantity.value);
            const duration = parseInt(this.elements.wakeDuration.value);
            const bid = parseFloat(this.elements.wakeBid.value);

            this.sim.sendWakeRequest(requester, provider, resource_type, quantity, duration, bid);

            // Hide form
            this.elements.wakeRequestForm.style.display = 'none';
            this.elements.btnRequestWake.disabled = false;
        });

        // Canvas click for peer selection
        this.sim.canvas.addEventListener('click', (e) => {
            const rect = this.sim.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;

            // Find clicked peer
            for (const peer of this.sim.peers) {
                const dx = peer.x - x;
                const dy = peer.y - y;
                if (Math.sqrt(dx * dx + dy * dy) < peer.radius) {
                    this.sim.selectPeer(peer);
                    return;
                }
            }

            // Clicked on empty space - deselect
            this.sim.selectPeer(null);
        });

        // Listen for simulation log events
        window.addEventListener('simulation-log', (e) => {
            this.addLogEntry(e.detail);
        });

        // Scenario buttons
        window.loadScenario = (scenario) => {
            this.sim.getScenario(scenario);
            this.sim.selectPeer(null);
            this.clearPeerDetails();
            this.updateStats();
            this.log(`Scenario "${scenario}" loaded`, 'info');
        };
    }

    showPeerDetails(peer) {
        if (!peer) {
            this.elements.peerDetails.style.display = 'none';
            return;
        }

        this.elements.peerDetails.style.display = 'block';
        this.elements.peerId.textContent = peer.id;
        this.elements.peerState.textContent = peer.state;
        this.elements.peerState.className = 'state-badge ' + peer.state;
        this.elements.peerBalance.textContent = `$${peer.ledger_balance.toFixed(2)}`;
        this.elements.peerWakes.textContent = `${peer.wakes_today}/${peer.policy.max_daily_wakes}`;

        // Resources
        let resourcesHtml = '';
        Object.entries(peer.resources).forEach(([type, res]) => {
            const util = res.getUtilization().toFixed(1);
            resourcesHtml += `
                <div>
                    <strong>${res.getDisplayName()}:</strong>
                    ${res.allocated}/${res.total} ${res.getUnit()} (${util}%)
                    <span class="state-indicator ${res.state}"></span>
                </div>
            `;
        });
        this.elements.peerResources.innerHTML = resourcesHtml;

        // Active leases
        let leasesHtml = '';
        if (peer.active_leases.length === 0) {
            leasesHtml = '<em>No active leases</em>';
        } else {
            peer.active_leases.forEach(lease => {
                const remaining = Math.ceil(lease.remaining_s);
                leasesHtml += `
                    <div>
                        <strong>${lease.spec.resource_type}</strong>: ${lease.spec.quantity} ${this.getResourceUnit(lease.spec.resource_type)}
                        <br><small>${remaining}s left • ${lease.energy_wh.toFixed(1)}Wh • $${lease.cost_credits.toFixed(4)}</small>
                    </div>
                `;
            });
        }
        this.elements.peerLeases.innerHTML = leasesHtml;

        // Policy
        this.elements.policyMinBid.value = peer.policy.min_bid_price_per_kwh;
        this.elements.policyMaxWakes.value = peer.policy.max_daily_wakes;
        this.elements.policyIdleTimeout.value = peer.policy.auto_hibernate_idle_s;
    }

    clearPeerDetails() {
        this.elements.peerDetails.style.display = 'none';
    }

    addLogEntry(entry) {
        const div = document.createElement('div');
        div.className = `log-entry ${entry.type}`;
        div.innerHTML = `<span class="log-time">${entry.time}</span>${entry.message}`;
        this.elements.eventLog.appendChild(div);

        // Keep only last 50 entries
        while (this.elements.eventLog.children.length > 50) {
            this.elements.eventLog.removeChild(this.elements.eventLog.firstChild);
        }
    }

    log(message, type = 'info') {
        // Simulation already logs, but we can add our own
        const entry = {
            time: new Date().toLocaleTimeString(),
            message,
            type,
        };
        this.addLogEntry(entry);
    }

    updateStats() {
        const stats = this.sim.getStats();
        this.elements.statPeers.textContent = stats.peer_count;
        this.elements.statLeases.textContent = stats.lease_count;
        this.elements.statWakes.textContent = stats.total_wakes;
        this.elements.statEnergy.textContent = stats.energy_kwh.toFixed(1) + ' kWh';
        this.elements.energyPrice.textContent = `$${parseFloat(stats.energy_price).toFixed(4)}/kWh`;
        this.elements.carbonIntensity.textContent = `${parseFloat(stats.carbon_intensity).toFixed(3)} kg/kWh`;

        // Format sim time as HH:MM
        const hour = Math.floor(stats.sim_hour);
        const minute = Math.floor((stats.sim_hour % 1) * 60);
        this.elements.simTime.textContent = `${String(hour).padStart(2, '0')}:${String(minute).padStart(2, '0')}`;
    }

    startStatsUpdate() {
        // Update stats every 500ms
        setInterval(() => {
            this.updateStats();
        }, 500);
    }

    getResourceUnit(resource_type) {
        const units = {
            'gpu_memory_gb': 'GB',
            'cpu_cores': 'cores',
            'ram_gb': 'GB',
            'model_layer': 'layers',
        };
        return units[resource_type] || '';
    }
}

window.Controls = Controls;
