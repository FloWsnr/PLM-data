# TODO

## Existing PDE / Config Inventory

Current baseline: 37 PDE presets and 86 configs.

The descriptions below are meant to capture the main visible behavior of each
existing config so future configs can be chosen to add genuinely different
dynamics.

### Basic

#### Advection (`configs/basic/advection`)

1. `2d_default`: single blob stirred by periodic cellular flow with weak diffusion.
2. `2d_pure_advection`: single blob translated diagonally with no diffusion.
3. `2d_rotation`: single blob rotating around the domain center with slight diffusion.
4. `3d_default`: single blob carried by a 3D periodic swirling flow with weak diffusion.

#### Elasticity (`configs/basic/elasticity`)

1. `2d_default`: cantilever beam ringing after a localized upward velocity kick near the free end.
2. `3d_default`: 3D cantilever bar ringing after a localized transverse velocity kick.

#### Heat (`configs/basic/heat`)

1. `2d_default`: single hot blob diffusing under no-flux boundaries.
2. `2d_manufactured_source`: zero initial field driven by a standing cosine heat source.
3. `2d_stochastic_media`: hot/cold blobs diffusing through noisy conductivity with additive state noise.
4. `2d_variable_diffusivity`: hot spot spreading through radially varying conductivity with cold boundaries.
5. `3d_default`: single hot blob diffusing in a cube with no-flux walls.
6. `3d_stochastic_media`: hot/cold blobs diffusing in noisy 3D media with additive state noise.

#### Helmholtz (`configs/basic/helmholtz`)

1. `2d_default`: localized Gaussian forcing producing a confined oscillatory response.
2. `3d_default`: volumetric sine forcing producing a box-scale standing-wave response.

#### Plate (`configs/basic/plate`)

1. `2d_default`: simply supported plate vibrating from a single sinusoidal deflection mode.

#### Poisson (`configs/basic/poisson`)

1. `2d_default`: static potential field from a 2D sinusoidal source pattern.
2. `3d_default`: static potential field from a 3D sinusoidal source pattern.

#### Schrodinger (`configs/basic/schrodinger`)

1. `2d_default`: Gaussian wave packet traveling toward and scattering off a localized potential bump.
2. `3d_default`: 3D Gaussian wave packet traveling toward and scattering off a localized potential bump.

#### Wave (`configs/basic/wave`)

1. `2d_default`: localized initial velocity pulse radiating through a heterogeneous wave-speed field.
2. `3d_default`: localized 3D velocity pulse radiating through a heterogeneous wave-speed field.

### Biology

#### Bistable Travelling Waves (`configs/biology/bistable_travelling_waves`)

1. `2d_default`: planar invasion front moving rightward from a step initial condition.
2. `3d_default`: 3D planar invasion front moving along `x` from a step initial condition.

#### Cyclic Competition (`configs/biology/cyclic_competition`)

1. `2d_default`: rock-paper-scissors competition from noise, forming moving patches and spiral-like domains.
2. `3d_default`: 3D rock-paper-scissors competition from noise, forming moving domains.

#### Fisher-KPP (`configs/biology/fisher_kpp`)

1. `2d_default`: logistic invasion front expanding from a left/right step profile.
2. `3d_default`: 3D logistic invasion front expanding along `x`.

#### FitzHugh-Nagumo (`configs/biology/fitzhugh_nagumo`)

1. `2d_default`: excitable medium seeded by noise, developing traveling and spiral-like waves.
2. `3d_default`: 3D excitable medium seeded by noise, developing wavefronts and filament-like structures.

#### Gierer-Meinhardt (`configs/biology/gierer_meinhardt`)

1. `2d_default`: noise-seeded activator-inhibitor Turing spots and labyrinths.
2. `3d_default`: 3D activator-inhibitor Turing patterning from noise.

#### Immunotherapy (`configs/biology/immunotherapy`)

1. `2d_default`: tumor-immune-cytokine fields evolving into patchy competition and suppression zones.
2. `3d_default`: 3D tumor-immune-cytokine competition with patchy heterogeneous regions.

#### Keller-Segel (`configs/biology/keller_segel`)

1. `2d_default`: chemotactic aggregation from near-uniform noise into dense clusters.
2. `3d_default`: 3D chemotactic aggregation into clumps and filaments.

#### Klausmeier Topography (`configs/biology/klausmeier_topography`)

1. `2d_default`: vegetation bands forming on a slope and migrating uphill.
2. `3d_default`: 3D sloped-terrain vegetation bands or sheets migrating uphill.

### Fluids

#### Burgers (`configs/fluids/burgers`)

1. `2d_default`: orthogonal sinusoidal velocity modes steepening, interacting, and diffusing on a periodic box.
2. `2d_stochastic`: the same Burgers shear-mode interaction with additive noise.
3. `3d_default`: three sinusoidal velocity modes steepening and diffusing in a 3D periodic box.

#### Compressible Navier-Stokes (`configs/fluids/compressible_navier_stokes`)

1. `2d_heated_blob`: localized heating drives expansion and pressure waves in a closed box.
2. `3d_heated_blob`: 3D localized heating drives expansion and pressure waves in a closed box.

#### Darcy (`configs/fluids/darcy`)

1. `2d_default`: pressure-driven porous flow advecting and diffusing a tracer blob through heterogeneous mobility.

#### MHD (`configs/fluids/mhd`)

1. `2d_decaying_alfvenic`: coupled velocity and magnetic wave patterns decaying under viscosity and resistivity.
2. `2d_orszag_tang`: Orszag-Tang vortex evolving into interacting current sheets and MHD turbulence.
3. `3d_cyclic_alfvenic`: 3D periodic Alfvénic wave cells cycling and decaying.
4. `3d_helical_cells`: helical flow cells coupled to magnetic structures in a 3D periodic box.

#### Navier-Stokes (`configs/fluids/navier_stokes`)

1. `2d_cavity_re400`: lid-driven cavity building the primary vortex and corner eddies.
2. `2d_periodic_shear_layer`: perturbed double shear layer rolling up into vortices.
3. `2d_periodic_shear_layer_high_re`: higher-Re shear layer with finer roll-up and stronger secondary instabilities.
4. `3d_default`: 3D lid-driven cavity starting from rest and building cavity circulation.

#### Shallow Water (`configs/fluids/shallow_water`)

1. `2d_default`: localized free-surface hump radiating gravity waves with Coriolis deflection.

#### Stokes (`configs/fluids/stokes`)

1. `2d_default`: steady lid-driven cavity recirculation.
2. `3d_default`: steady 3D lid-driven cavity recirculation.

#### Thermal Convection (`configs/fluids/thermal_convection`)

1. `2d_default`: noisy stratification between hot bottom and cold top evolving into convection rolls and plumes.
2. `3d_default`: 3D Rayleigh-Benard-style convection evolving into plumes and cells.

### Physics

#### Brusselator (`configs/physics/brusselator`)

1. `2d_default`: noise-seeded Brusselator Turing and labyrinth patterns.
2. `3d_default`: 3D Brusselator pattern formation from noise.

#### Cahn-Hilliard (`configs/physics/cahn_hilliard`)

1. `2d_default`: spinodal decomposition into phase-separated domains.
2. `3d_default`: 3D spinodal decomposition and coarsening.

#### Complex Ginzburg-Landau (`configs/physics/cgl`)

1. `2d_default`: Benjamin-Feir-unstable complex amplitude turbulence from a noisy background.
2. `3d_default`: 3D complex amplitude turbulence from a noisy background.

#### Gray-Scott (`configs/physics/gray_scott`)

1. `2d_annulus`: Gray-Scott spot and stripe patterning on an annular domain from two seeded patches.
2. `2d_default`: two seeded reagent patches generating drifting Gray-Scott spots and stripes.
3. `2d_stochastic`: Gray-Scott spots and stripes with stochastic roughening and variability.
4. `3d_default`: 3D Gray-Scott spot and blob structures from two seeded reagent patches.

#### Kuramoto-Sivashinsky (`configs/physics/kuramoto_sivashinsky`)

1. `2d_default`: noise-seeded wrinkling and spatiotemporal chaos on a periodic plane.
2. `3d_default`: 3D spatiotemporal chaos from small random perturbations.

#### Lorenz (`configs/physics/lorenz`)

1. `2d_default`: spatially extended Lorenz chaos seeded from random initial fields.
2. `3d_default`: 3D spatially extended Lorenz chaos seeded from random initial fields.

#### Maxwell (`configs/physics/maxwell`)

1. `2d_default`: time-harmonic electromagnetic radiation from a localized source with absorbing walls.
2. `3d_default`: 3D time-harmonic electromagnetic radiation from a localized source with absorbing walls.

#### Maxwell Pulse (`configs/physics/maxwell_pulse`)

1. `2d_default`: localized electromagnetic pulse launched in a waveguide-like box with absorbing ends.
2. `3d_default`: 3D localized electromagnetic pulse launched in a partially confined box.

#### Schnakenberg (`configs/physics/schnakenberg`)

1. `2d_default`: noise-seeded Schnakenberg Turing spots and labyrinths.
2. `3d_default`: 3D Schnakenberg pattern formation from noise.

#### Superlattice (`configs/physics/superlattice`)

1. `2d_default`: coupled four-field reaction-diffusion pattern forming mixed-scale superlattice structures.
2. `3d_default`: 3D coupled four-field superlattice-style pattern formation.

#### Swift-Hohenberg (`configs/physics/swift_hohenberg`)

1. `2d_default`: localized seed growing into an advected stripe or roll pattern.
2. `2d_directed`: stronger uniform advection producing more clearly directed stripe growth and drift.
3. `2d_rotational`: noise-seeded pattern formation under rotational advection, giving swirling rolls.
4. `3d_default`: localized seed growing into a 3D patterned structure under weak uniform advection.

#### Van der Pol (`configs/physics/van_der_pol`)

1. `2d_default`: random low-amplitude modes relaxing into oscillatory Van der Pol wave patterns.
2. `3d_default`: 3D Van der Pol oscillatory wave patterns from random modal perturbations.

#### Zakharov-Kuznetsov (`configs/physics/zakharov_kuznetsov`)

1. `2d_default`: localized soliton-like pulse propagating and reshaping on a periodic domain.
2. `3d_default`: 3D localized soliton-like pulse propagating and reshaping.
