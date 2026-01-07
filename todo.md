# PDE Status Summary

## Basic PDEs

| PDE | Status | Notes |
|-----|--------|-------|
| heat | OK | Shows 4 Gaussian heat spots diffusing over time. Gradual, visible changes. Classic heat diffusion behavior. |
| wave | OK | Wave propagates from single point source in expanding circular ripples. Clear wave fronts with reflections. |
| advection | OK | Two Gaussian blobs transported by velocity field. Clear advection with slight diffusion. |
| schrodinger | OK | Beautiful quantum particle-in-a-box pattern. 4x4 grid of probability density cells with phase evolution. Standing wave modes visible. |
| inhomogeneous-heat | OK | Heat diffusion with spatially varying source. Initial Gaussian spots evolve into checkerboard pattern due to source term. |
| inhomogeneous-wave | OK | Wave propagation with spatially varying wave speed. Complex interference patterns emerge as waves reflect at different speeds. |
| plate | NEEDS WORK | Frames mostly blank/uniform. Only tiny corner artifacts visible at end. Initial condition or amplitude too weak. |

## Physics PDEs

| PDE | Status | Notes |
|-----|--------|-------|
| burgers | OK | Clear shock wave formation. Wave front moves across domain with visible steepening. Classic nonlinear wave behavior. |
| cahn-hilliard | OK | Excellent phase separation. Starts from noise, evolves to distinct red/blue domains. Classic spinodal decomposition. |
| complex-ginzburg-landau | OK | Beautiful spiral wave and turbulence patterns. Regular grid evolves into chaotic spiral structures. |
| duffing | OK | Spatiotemporal chaos from coupled oscillators. Noise evolves into coherent domain patterns. |
| gray-scott | NEEDS WORK | Simulation runs but ends up blank/uniform. Pattern dies out. Need different initial conditions or parameters. |
| korteweg-de-vries | NEEDS WORK | Soliton pulse visible initially but diffuses to uniform state. Need longer domain or different parameters. |
| kuramoto-sivashinsky | OK | Excellent spatiotemporal chaos. Random initial conditions develop into coherent chaotic cellular patterns. |
| lorenz | NEEDS WORK | Simulation too slow - didn't complete. Very small adaptive timesteps needed. Consider scipy solver. |
| nonlinear-beam | NEEDS WORK | Gaussian initial condition diffuses to uniform/blank. Dynamics too fast or damping too strong. |
| nonlinear-schrodinger | OK | Soliton dynamics visible. Pulse moves across domain maintaining shape. Good optical soliton behavior. |
| perona-malik | NEEDS WORK | Minimal visible change between frames. Edge-preserving diffusion not apparent. Needs stronger edges in initial condition. |
| stochastic-gray-scott | ERROR | Backend error - numba implementation missing for stochastic term. |
| superlattice | INCOMPLETE | Very slow simulation with scipy solver. Did not complete in time. |
| swift-hohenberg | OK | Excellent labyrinthine pattern formation. Random noise develops into structured stripes/spots. Classic pattern formation. |
| swift-hohenberg-advection | NEEDS WORK | Beautiful initial radial pattern but evolves to uniform/blank. Advection or instability issue. |
| van-der-pol | OK | Spatiotemporal chaos from coupled oscillators. Develops coherent oscillation domains over time. |

## Fluid PDEs

| PDE | Status | Notes |
|-----|--------|-------|
| navier-stokes | NEEDS WORK | Smooth gradient, no visible dynamics. Initial conditions need adjustment for interesting flow patterns. Runtime: 27s, 108 steps. |
| shallow-water | OK | Excellent wave propagation. Initial height perturbation spreads as circular wave ring with boundary reflections. Runtime: 27s, 238 steps. |
| thermal-convection | ACCEPTABLE | Shows thermal diffusion from noisy initial state. Subtle convection visible but Rayleigh-Benard cells not clear. May need higher Rayleigh number. Runtime: 27s, 359 steps. |
| vorticity | ACCEPTABLE | Counter-rotating vortex pair (red/blue) with slow diffusion. Very expensive: 333k steps (~12min). Vortices mostly stationary. Consider lower viscosity for more dynamics. |

## Biology PDEs

| PDE | Status | Notes |
|-----|--------|-------|
| bacteria-advection | NEEDS WORK | Develops oscillating stripes at right boundary - numerical instability. Boundary conditions need adjustment. |
| bistable-allen-cahn | OK | Three population patches with Allee effect dynamics. Spots maintain shape with sharpening boundaries over time. |
| brusselator | OK | Excellent Turing patterns! Noise develops into beautiful spots and labyrinthine structures. Classic oscillating reaction-diffusion. |
| cross-diffusion-schnakenberg | NEEDS WORK | Pattern dies out to uniform green. Cross-diffusion parameters may need tuning for dark soliton formation. |
| cyclic-competition | OK | Excellent rock-paper-scissors dynamics! Species domains expand and form spiral patterns. Classic cyclic competition. |
| fisher-kpp | OK | Population front propagation visible. Wave fronts spread from initial patches, saturates at carrying capacity. |
| fitzhugh-nagumo | OK | Beautiful excitable media patterns! Red square pulse patterns emerge from uniform state. Classic Turing spots. |
| gierer-meinhardt | OK | Excellent Turing spots! Develops from dark uniform state into well-organized spot pattern. Classic activator-inhibitor. |
| harsh-environment | OK | Population patches spread to carrying capacity (uniform yellow). Shows logistic growth with diffusion. |
| heterogeneous-gierer-meinhardt | OK | Excellent! Forms stripes then spots with spatial gradient influence. Yellow bands at edges due to heterogeneous parameters. |
| hyperbolic-brusselator | NEEDS WORK | Pattern dies to uniform green. Turing wave instability not visible. Parameters may need adjustment. |
| immunotherapy | OK | Interesting tumor-immune dynamics! Tumor shrinks (immune response) then rebounds as expanding ring. Biologically relevant. |
| keller-segel | OK | Chemotaxis aggregation - bacteria cluster toward chemoattractant. Nice blob merging and corner aggregation. |
| klausmeier | NEEDS WORK | Vegetation dies out to uniform dark (desert state). Need different parameters for vegetation stripe patterns. |
| klausmeier-topography | NEEDS WORK | Blows up to white/infinity. Numerical instability with topography term. Needs smaller dt or different parameters. |
| schnakenberg | OK | Beautiful Turing spots! Develops from uniform blue into well-organized hexagonal spot pattern. Classic reaction-diffusion. |
