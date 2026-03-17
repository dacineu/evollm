// Resource Types and Resource class
// Defines the different types of resources that can be allocated

const ResourceType = {
    GPU_MEMORY_GB: 'gpu_memory_gb',
    CPU_CORES: 'cpu_cores',
    RAM_GB: 'ram_gb',
    MODEL_LAYER: 'model_layer',
    DISK_IOPS: 'disk_iops',
    NET_BANDWIDTH_MBPS: 'net_bandwidth_mbps',
};

// Resource state colors for visualization
const ResourceStateColors = {
    functional: '#4caf50',
    hibernate: '#ff9800',
    sleep: '#2196f3',
    offline: '#9e9e9e',
};

/**
 * Resource class represents a specific resource type within a peer
 */
class Resource {
    constructor(type, total, state = 'functional') {
        this.type = type;
        this.total = total;
        this.allocated = 0;
        this.state = state;
        this.state_since = Date.now();
    }

    /**
     * Get available amount
     */
    getAvailable() {
        return Math.max(0, this.total - this.allocated);
    }

    /**
     * Check if can allocate quantity
     */
    canAllocate(quantity) {
        return this.getAvailable() >= quantity;
    }

    /**
     * Allocate quantity
     */
    allocate(quantity) {
        if (!this.canAllocate(quantity)) {
            return false;
        }
        this.allocated += quantity;
        return true;
    }

    /**
     * Release allocation
     */
    release(quantity) {
        this.allocated = Math.max(0, this.allocated - quantity);
    }

    /**
     * Get utilization percentage
     */
    getUtilization() {
        return this.total > 0 ? (this.allocated / this.total) * 100 : 0;
    }

    /**
     * Set state with timestamp
     */
    setState(newState) {
        if (this.state !== newState) {
            this.state = newState;
            this.state_since = Date.now();
            return true; // State changed
        }
        return false;
    }

    /**
     * Get time in current state (seconds)
     */
    getTimeInState() {
        return (Date.now() - this.state_since) / 1000;
    }

    /**
     * Get display name for resource type
     */
    getDisplayName() {
        const names = {
            [ResourceType.GPU_MEMORY_GB]: 'GPU Memory',
            [ResourceType.CPU_CORES]: 'CPU Cores',
            [ResourceType.RAM_GB]: 'RAM',
            [ResourceType.MODEL_LAYER]: 'Model Layers',
            [ResourceType.DISK_IOPS]: 'Disk IOPS',
            [ResourceType.NET_BANDWIDTH_MBPS]: 'Network Bandwidth',
        };
        return names[this.type] || this.type;
    }

    /**
     * Get unit for resource type
     */
    getUnit() {
        const units = {
            [ResourceType.GPU_MEMORY_GB]: 'GB',
            [ResourceType.CPU_CORES]: 'cores',
            [ResourceType.RAM_GB]: 'GB',
            [ResourceType.MODEL_LAYER]: 'layers',
            [ResourceType.DISK_IOPS]: 'IOPS',
            [ResourceType.NET_BANDWIDTH_MBPS]: 'Mbps',
        };
        return units[this.type] || '';
    }
}

// Export for use in other modules
window.ResourceType = ResourceType;
window.ResourceStateColors = ResourceStateColors;
window.Resource = Resource;
