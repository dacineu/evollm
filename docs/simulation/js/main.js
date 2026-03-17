// Main entry point
// Initialize simulation and controls when page loads

document.addEventListener('DOMContentLoaded', () => {
    // Get canvas
    const canvas = document.getElementById('sim-canvas');
    if (!canvas) {
        console.error('Canvas element not found!');
        return;
    }

    // Create simulation
    const simulation = new Simulation(canvas);
    window.simulation = simulation; // Expose globally for debugging

    // Create controls
    const controls = new Controls(simulation);

    // Handle window resize
    window.addEventListener('resize', () => {
        simulation.renderer.resize();
    });

    // Initialize with a small scenario
    simulation.getScenario('small');

    // Debug: Check peers were created
    console.log(`Created ${simulation.peers.length} peers`);
    simulation.peers.forEach(p => {
        console.log(`Peer ${p.id}: total=${p.totalResources()}, state=${p.state}`);
    });

    // Ensure canvas is properly sized before starting
    setTimeout(() => {
        // Resize to get correct dimensions
        simulation.renderer.resize();
        console.log(`Canvas size: ${simulation.canvas.width}x${simulation.canvas.height}`);

        // Start simulation
        simulation.start();

        // Update button states
        const btnStart = document.getElementById('btn-start');
        const btnPause = document.getElementById('btn-pause');
        if (btnStart && btnPause) {
            btnStart.disabled = true;
            btnPause.disabled = false;
        }
        console.log('Simulation auto-started');
    }, 200); // Wait for layout to settle

    console.log('EvoOS Simulation initialized');
    console.log('Click "Start" to begin the simulation');
    console.log('Click on peers to view details and send wake requests');

    // Log instructions
    simulation.log('Welcome to EvoOS Simulation!', 'info');
    simulation.log('Click "Start" to begin. Click peers to select them.', 'info');
    simulation.log('Try "Small" preset scenario to see energy-aware peers in action.', 'info');
});

// Expose simulation globally for console access
window.getSimulation = () => window.simulation;
