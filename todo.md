# Basic PDEs Analysis Status

| PDE | Config | Status | Notes |
|-----|--------|--------|-------|
| heat | basic/heat.yaml | OK | Diffusion of heat spots visible |
| wave | basic/wave.yaml | GOOD | Clear wave propagation with expanding circular ripples |
| wave-standing | basic/wave_standing.yaml | GOOD | Beautiful standing wave grid pattern with clear oscillations |
| advection | basic/advection.yaml | GOOD | Blobs clearly moving/advecting across domain |
| plate | basic/plate.yaml | OK | Starts nearly uniform, develops interesting vibration patterns (Chladni figures) |
| schrodinger | basic/schrodinger.yaml | OK | Quantum grid pattern visible, stationary (expected behavior) |
| inhomogeneous-heat | basic/inhomogeneous_heat.yaml | GOOD | Interesting checkerboard pattern evolving from heat sources |
| inhomogeneous-wave | basic/inhomogeneous_wave.yaml | GOOD | Nice wave interference patterns with variable wave speed |
| damped-wave | basic/damped_wave.yaml | GOOD | Waves propagate then damp nicely, visible dynamics |
| inhomogeneous-diffusion-heat | basic/inhomogeneous_diffusion_heat.yaml | GOOD | Interesting spiral/radial diffusion patterns from varying diffusion |

# Fluid PDEs Analysis Status

| PDE | Config | Status | Notes |
|-----|--------|--------|-------|
| navier-stokes | fluids/navier_stokes.yaml | GOOD | Kelvin-Helmholtz instability - shear layer rollup into cat's eye vortices, beautiful spiral mixing (shear-layer IC, domain=5, Re~375) |
| potential-flow-dipoles | fluids/potential_flow_dipoles.yaml | GOOD | Clean dipole potential field (coolwarm) - blue sink (left), red source (right). Relaxes to steady state quickly. (domain=40, strength=2000, d=10, t_end=5, plot phi) |
| potential-flow-images | fluids/potential_flow_images.yaml | GOOD | Symmetric sink pair (method of images) - two sinks visible with coolwarm colormap showing potential wells. (domain=40, strength=50, wall_x=0.5, t_end=5, plot phi) |
| shallow-water | fluids/shallow_water.yaml | GOOD | Nice circular wave propagation from central perturbation |
| thermal-convection | fluids/thermal_convection.yaml | GOOD | Custom PDEBase class enables per-field BCs. Heat flux at bottom creates rising thermal plumes - beautiful Rayleigh-Benard convection cells developing over time. (domain=500, nu=0.2, kappa=0.5, T_b=0.08, noise IC, t_end=300) |
| vorticity | fluids/vorticity.yaml | OK | Vortex pair orbital dynamics - counter-rotating vortices orbit each other (domain_size=20, nu=0.01, offset y positions for rotation) |
| vorticity-bounded | fluids/vorticity_bounded.yaml | GOOD | Oscillatory vorticity decay - checkerboard pattern (k=8) gradually fades over time showing viscous diffusion (nu=0.15, t_end=80) |

# Biology PDEs Analysis Status

| PDE | Config | Status | Notes |
|-----|--------|--------|-------|
| bacteria-advection | biology/bacteria_advection.yaml | WEAK | Advection visible at edges but mostly uniform - needs stronger initial condition |
| bistable-allen-cahn | biology/bistable_allen_cahn.yaml | GOOD | Three localized spots sharpening over time - bistable/Allee effect visible |
| brusselator | biology/brusselator.yaml | GOOD | Beautiful Turing patterns - spots from noise |
| cross-diffusion-schnakenberg | biology/cross_diffusion_schnakenberg.yaml | GOOD | Beautiful labyrinthine Turing patterns - b=1 enables pattern formation (b=2.5 was stable/uniform) |
| cyclic-competition | biology/cyclic_competition.yaml | GOOD | Beautiful spiral wave patterns from noise - rock-paper-scissors dynamics |
| cyclic-competition-wave | biology/cyclic_competition_wave.yaml | GOOD | Edge invasion front with chaotic wave dynamics spreading |
| fisher-kpp | biology/fisher_kpp.yaml | OK | Population wave spreads correctly but ends uniform (expected saturation) |
| fitzhugh-nagumo | biology/fitzhugh_nagumo.yaml | GOOD | Excellent Turing spot patterns evolving from noise to organized squares |
| fitzhugh-nagumo-3 | biology/fitzhugh_nagumo_3.yaml | GOOD | Beautiful tile/grid pattern from seed - pattern-oscillation competition |
| gierer-meinhardt | biology/gierer_meinhardt.yaml | GOOD | Beautiful spot patterns forming quasi-hexagonal lattice |
| gierer-meinhardt-stripes | biology/gierer_meinhardt_stripes.yaml | GOOD | Labyrinthine stripe patterns (different from spot pattern) |
| harsh-environment | biology/harsh_environment.yaml | OK | Population spreads from seeds with harsh corner boundary - ends uniform |
| heterogeneous-gierer-meinhardt | biology/heterogeneous_gierer_meinhardt.yaml | GOOD | Beautiful spot/stripe patterns from spatial heterogeneity |
| hyperbolic-brusselator | biology/hyperbolic_brusselator.yaml | GOOD | Beautiful Turing wave patterns - labyrinthine spots/stripes. Fixed Dv=1 (was 8) to achieve Du>Dv required for Turing wave instability |
| immunotherapy | biology/immunotherapy.yaml | GOOD | Beautiful Turing labyrinthine patterns - tumor islands (teal) with cleared regions (purple). Used Visual PDE params: Du=10, Dv=0.1 (100:1 ratio), domain=50, s_u=0.02, noise=0.3. Shows spatial tumor heterogeneity from immune-mediated Turing instability |
| keller-segel | biology/keller_segel.yaml | GOOD | Chemotaxis aggregation - spots and labyrinthine patterns |
| klausmeier | biology/klausmeier.yaml | GOOD | Beautiful banded vegetation patterns (tiger bush stripes) |
| klausmeier-topography | biology/klausmeier_topography.yaml | OK | Vegetation modulated by hills topography - higher vegetation in valleys (water accumulation), lower at peaks. Model differs from standard Klausmeier - shows terrain-following rather than stripe patterns. (V=30, amplitude=0.8, a=1.2, m=0.45, 2x2 hills), still not super big differences |
| schnakenberg | biology/schnakenberg.yaml | GOOD | Beautiful hexagonal spot patterns from Turing instability |

# Physics PDEs Analysis Status

| PDE | Config | Status | Notes |
|-----|--------|--------|-------|
| bistable-advection | physics/bistable_advection.yaml | GOOD | Clear invasion front propagation - yellow (invaded) expands across domain. Flow-assisted invasion with a=0.3, V=0.3, D=0.1 |
| burgers | physics/burgers.yaml | GOOD | Visible wave steepening and shock propagation - asymmetric profile with steep front, diffuse back (amplitude=2.0, epsilon=0.5) |
| cahn-hilliard | physics/cahn_hilliard.yaml | GOOD | Beautiful phase separation from noise to meandering domains |
| complex-ginzburg-landau | physics/complex_ginzburg_landau.yaml | GOOD | Excellent spiral wave chaos and defect dynamics |
| duffing | physics/duffing.yaml | GOOD | Coupled oscillator domain formation from noise |
| gray-scott | physics/gray_scott.yaml | GOOD | Beautiful labyrinthine pattern growth from single seed - classic self-replication |
| inviscid-burgers | physics/inviscid_burgers_shock_interaction.yaml | GOOD | Sharp shock formation from Gaussian pulse - front steepens into discontinuity, stretched tail behind (amplitude=1.5, epsilon=0.01) |
| korteweg-de-vries | physics/korteweg_de_vries.yaml | BAD | Soliton disperses and ends as blank white - numerical instability |
| kuramoto-sivashinsky | physics/kuramoto_sivashinsky.yaml | OK | Chaotic evolution but low resolution (64x64), looks noisy |
| lorenz | physics/lorenz.yaml | GOOD | Coupled Lorenz oscillators - synchronized arrow dynamics evolving |
| nonlinear-beam | physics/nonlinear_beam.yaml | BAD | Biharmonic equation too stiff - blows up even with tiny dt. Needs implicit solver |
| nonlinear-schrodinger | physics/nonlinear_schrodinger.yaml | GOOD | Soliton propagation maintaining shape across domain |
| perona-malik | physics/perona_malik.yaml | GOOD | Edge-preserving diffusion - checkerboard smoothing with sharp boundaries |
| stochastic-gray-scott | physics/stochastic_gray_scott.yaml | GOOD | Stochastic effects visible - noisy irregular boundaries from multiplicative noise |
| superlattice | physics/superlattice.yaml | GOOD | Beautiful coupled spot patterns - red cores with white halos on blue |
| swift-hohenberg | physics/swift_hohenberg.yaml | GOOD | Stunning labyrinthine stripe patterns from noise |
| swift-hohenberg-advection | physics/swift_hohenberg_advection.yaml | GOOD | Beautiful labyrinthine stripes evolving under advection (r=0.3 supercritical, a=0 stripes, V=0.6 diagonal flow) |
| van-der-pol | physics/van_der_pol.yaml | GOOD | Beautiful domain coarsening from noise - chaotic phase patterns |
