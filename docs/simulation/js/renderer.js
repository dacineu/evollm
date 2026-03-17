// Renderer class
// Handles all canvas drawing operations

class Renderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.resize();

        // Visual settings
        this.peer_radius_multiplier = 2.5;
        this.lease_width_multiplier = 1.5;
        this.particle_size = 3;
    }

    resize() {
        // Resize canvas to container with fallbacks
        const container = this.canvas.parentElement;
        let width = container.clientWidth;
        let height = container.clientHeight;

        // If container has no size, fall back to window size minus known UI elements
        if (width === 0 || height === 0) {
            console.warn('Container has zero size, falling back to window size');
            width = window.innerWidth - 350;  // Approximate sidebar width (320px) + padding
            height = window.innerHeight - 120; // Approximate header + footer height
        }

        // Minimum reasonable size
        width = Math.max(width, 800);
        height = Math.max(height, 600);

        this.canvas.width = width;
        this.canvas.height = height;

        console.log(`Canvas resized to: ${width}x${height}`);
    }

    clear() {
        // Draw background gradient
        const gradient = this.ctx.createLinearGradient(0, 0, this.canvas.width, this.canvas.height);
        gradient.addColorStop(0, '#0a0e17');
        gradient.addColorStop(1, '#1a1f2e');
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw grid (optional)
        this.drawGrid();
    }

    drawGrid() {
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
        this.ctx.lineWidth = 1;
        const grid_size = 50;

        for (let x = 0; x < this.canvas.width; x += grid_size) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
            this.ctx.stroke();
        }
        for (let y = 0; y < this.canvas.height; y += grid_size) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
            this.ctx.stroke();
        }
    }

    drawPeer(peer) {
        const radius = peer.radius;

        // Glow effect
        const gradient = this.ctx.createRadialGradient(
            peer.x, peer.y, radius * 0.5,
            peer.x, peer.y, radius * 1.2
        );
        gradient.addColorStop(0, peer.current_color + '80');
        gradient.addColorStop(1, peer.current_color + '00');
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(peer.x, peer.y, radius * 1.2, 0, Math.PI * 2);
        this.ctx.fill();

        // Main circle
        this.ctx.beginPath();
        this.ctx.arc(peer.x, peer.y, radius, 0, Math.PI * 2);
        this.ctx.fillStyle = peer.current_color;
        this.ctx.fill();

        // State border (thicker if selected)
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = peer.selected ? 4 : 2;
        this.ctx.stroke();

        // Additional ring if has active leases
        if (peer.active_leases.length > 0) {
            this.ctx.beginPath();
            this.ctx.arc(peer.x, peer.y, radius + 6, 0, Math.PI * 2);
            this.ctx.strokeStyle = '#ffeb3b';
            this.ctx.lineWidth = 3;
            this.ctx.stroke();
        }

        // Draw ID label
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = 'bold 12px monospace';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(peer.id, peer.x, peer.y - radius - 10);

        // Draw state label
        this.ctx.font = '10px sans-serif';
        this.ctx.fillStyle = '#cccccc';
        this.ctx.fillText(peer.state.toUpperCase(), peer.x, peer.y + radius + 12);

        // Draw lease count
        if (peer.active_leases.length > 0) {
            this.ctx.fillStyle = '#ffeb3b';
            this.ctx.font = 'bold 10px sans-serif';
            this.ctx.fillText(`${peer.active_leases.length} lease${peer.active_leases.length > 1 ? 's' : ''}`, peer.x, peer.y + radius + 24);
        }
    }

    drawLease(lease) {
        const from = lease.requester;
        const to = lease.provider;

        if (!from || !to) return;

        // Draw curved path
        const midX = (from.x + to.x) / 2;
        const midY = (from.y + to.y) / 2 - 40;
        const progress = lease.progress;

        // Draw path line
        this.ctx.beginPath();
        this.ctx.moveTo(from.x, from.y);
        this.ctx.quadraticCurveTo(midX, midY, to.x, to.y);
        this.ctx.strokeStyle = this.getResourceColor(lease.spec.resource_type);
        this.ctx.lineWidth = 2 + lease.spec.quantity / 4;
        this.ctx.stroke();

        // Draw particles along path
        for (const p of lease.particles) {
            const x = from.x * (1 - p.t) + midX * 2 * p.t * (1 - p.t) + to.x * p.t * p.t;
            // Simplified quadratic bezier; for accuracy, use proper formula
            const t = p.t;
            const mt = 1 - t;
            const px = mt * mt * from.x + 2 * mt * t * midX + t * t * to.x;
            const py = mt * mt * from.y + 2 * mt * t * midY + t * t * to.y;

            this.ctx.beginPath();
            this.ctx.arc(px, py, p.size, 0, Math.PI * 2);
            this.ctx.fillStyle = '#ffffff';
            this.ctx.fill();
        }

        // Draw resource label near midpoint
        if (progress < 0.5) {
            const label_x = from.x + (midX - from.x) * (progress * 2);
            const label_y = from.y + (midY - from.y) * (progress * 2);
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '10px sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(
                `${lease.spec.quantity} ${this.getResourceUnit(lease.spec.resource_type)}`,
                label_x, label_y - 10
            );
        }
    }

    drawWakeRequest(wr) {
        const x = wr.requester.x + (wr.provider.x - wr.requester.x) * wr.progress;
        const y = wr.requester.y + (wr.provider.y - wr.requester.y) * wr.progress;

        const base_radius = 15;
        const expansion = wr.age * 30;

        // Pulsating ring
        this.ctx.beginPath();
        this.ctx.arc(wr.requester.x, wr.requester.y, expansion, 0, Math.PI * 2);
        this.ctx.strokeStyle = wr.accepted ? 'rgba(76, 175, 80, 0.6)' : 'rgba(244, 67, 54, 0.6)';
        this.ctx.lineWidth = 3;
        this.ctx.stroke();

        // Inner ring
        this.ctx.beginPath();
        this.ctx.arc(wr.requester.x, wr.requester.y, expansion - 5, 0, Math.PI * 2);
        this.ctx.stroke();

        // Draw marker at current position
        this.ctx.beginPath();
        this.ctx.arc(x, y, 6, 0, Math.PI * 2);
        this.ctx.fillStyle = wr.accepted ? '#4caf50' : '#f44336';
        this.ctx.fill();
        this.ctx.strokeStyle = '#ffffff';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();

        // Draw symbol
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = 'bold 12px sans-serif';
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(wr.accepted ? '✓' : '✗', x, y);

        // Draw label
        if (wr.age < 2) {
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '10px sans-serif';
            this.ctx.fillText(
                `${wr.quantity} ${this.getResourceUnit(wr.resource_type)}`,
                wr.requester.x, wr.requester.y - expansion - 10
            );
        }
    }

    drawLegend() {
        // Handled by HTML/CSS overlay
    }

    getResourceColor(resource_type) {
        const colors = {
            [ResourceType.GPU_MEMORY_GB]: '#e91e63',
            [ResourceType.CPU_CORES]: '#9c27b0',
            [ResourceType.RAM_GB]: '#3f51b5',
            [ResourceType.MODEL_LAYER]: '#00bcd4',
        };
        return colors[resource_type] || '#ffffff';
    }

    getResourceUnit(resource_type) {
        const units = {
            [ResourceType.GPU_MEMORY_GB]: 'GB',
            [ResourceType.CPU_CORES]: 'cores',
            [ResourceType.RAM_GB]: 'GB',
            [ResourceType.MODEL_LAYER]: 'layers',
        };
        return units[resource_type] || '';
    }
}

window.Renderer = Renderer;
