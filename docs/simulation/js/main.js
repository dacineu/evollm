// Main entry point
// Initialize simulation and controls when page loads

window.addEventListener('load', () => {
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

    // Ensure canvas is sized correctly (needs CSS to be loaded)
    simulation.renderer.resize();

    // Initialize with a small scenario
    simulation.getScenario('small');

    // Debug: Check peers were created
    console.log(`Created ${simulation.peers.length} peers`);
    simulation.peers.forEach(p => {
        console.log(`Peer ${p.id}: total=${p.totalResources()}, state=${p.state}, resources=`, p.resources);
    });

    // Log canvas dimensions
    console.log(`Canvas size: ${simulation.canvas.width}x${simulation.canvas.height}`);

    // Start simulation automatically
    simulation.start();

    // Update button states
    const btnStart = document.getElementById('btn-start');
    const btnPause = document.getElementById('btn-pause');
    if (btnStart && btnPause) {
        btnStart.disabled = true;
        btnPause.disabled = false;
    }

    console.log('Simulation auto-started');
    simulation.log('Simulation running - you should see peers on the canvas', 'success');
    simulation.log('Click on any peer to view details and send wake requests', 'info');
});

// Expose simulation globally for console access
window.getSimulation = () => window.simulation;
