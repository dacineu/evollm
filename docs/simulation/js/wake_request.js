// WakeRequest class
// Represents a request for a peer to wake a sleeping resource

class WakeRequest {
    constructor(requester, provider, resource_type, quantity, duration_s, bid_price_per_kwh) {
        this.requester = requester;
        this.provider = provider;
        this.resource_type = resource_type;
        this.quantity = quantity;
        this.duration = duration_s;
        this.bid_price_per_kwh = bid_price_per_kwh;
        this.bid_price_total = this.calculateTotalBid();
        this.age = 0;
        this.accepted = false;
        this.reason = null;
        this.distance = this.calculateDistance();
        this.progress = 0; // 0 to 1 as it "travels" to provider
        this.travel_speed = 0.5; // Units per second
    }

    calculateTotalBid() {
        // Estimate total cost for the requested duration
        const estimated_cost = this.provider.estimateOperationalCost({
            resource_type: this.resource_type,
            quantity: this.quantity,
            duration_s: this.duration,
        });
        const wake_cost = this.provider.getWakeEnergyCost();
        const total_cost = estimated_cost + wake_cost;
        return total_cost;
    }

    calculateDistance() {
        // Distance between requester and provider on canvas
        const dx = this.provider.x - this.requester.x;
        const dy = this.provider.y - this.requester.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    update(dt) {
        this.age += dt;

        // Move request toward provider
        if (!this.accepted) {
            this.progress += this.travel_speed * dt / this.distance;
            this.progress = Math.min(1, this.progress);
        }
    }

    isExpired() {
        // Expire after 10 seconds
        return this.age > 10;
    }

    getCurrentPosition() {
        // Interpolate between requester and provider
        const x = this.requester.x + (this.provider.x - this.requester.x) * this.progress;
        const y = this.requester.y + (this.provider.y - this.requester.y) * this.progress;
        return { x, y };
    }
}

window.WakeRequest = WakeRequest;
