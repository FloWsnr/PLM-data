# TODO

## Existing PDE / Config Inventory

Current baseline: 37 PDE presets and 86 configs.

The descriptions below are meant to capture the main visible behavior of each
existing config so future configs can be chosen to add genuinely different
dynamics.

### Basic

#### Advection (`configs/basic/advection`)

1. `2d_cellular_blob_advection`: one or two blobs stirred by periodic cellular flow with weak diffusion.
2. `2d_diagonal_blob_transport`: one or two blobs translated diagonally with no diffusion.
3. `2d_rotating_blob_advection`: one or two blobs rotating around the domain center with slight diffusion.
4. `3d_swirling_blob_advection`: one or two blobs carried by a 3D periodic swirling flow with weak diffusion.

#### Elasticity (`configs/basic/elasticity`)

1. `2d_cantilever_impulse_ringdown`: cantilever beam ringing after a localized upward velocity kick near the free end.
2. `3d_cantilever_impulse_ringdown`: 3D cantilever bar ringing after a localized transverse velocity kick.

#### Heat (`configs/basic/heat`)

1. `2d_localized_blob_diffusion`: one or two hot blobs diffusing under no-flux boundaries.
2. `2d_cosine_source_heating`: zero initial field driven by a standing cosine heat source.
3. `2d_noisy_media_diffusion`: hot/cold blobs diffusing through noisy conductivity with additive state noise.
4. `2d_radial_diffusivity_spread`: hot spot spreading through radially varying conductivity with cold boundaries.
5. `3d_localized_blob_diffusion`: one or two hot blobs diffusing in a cube with no-flux walls.
6. `3d_noisy_media_diffusion`: hot/cold blobs diffusing in noisy 3D media with additive state noise.

#### Helmholtz (`configs/basic/helmholtz`)

1. `2d_localized_oscillatory_response`: localized Gaussian forcing producing a confined oscillatory response.
2. `3d_standing_wave_response`: volumetric sine forcing producing a box-scale standing-wave response.

#### Plate (`configs/basic/plate`)

1. `2d_simply_supported_mode_vibration`: simply supported plate vibrating from a single sinusoidal deflection mode.

#### Poisson (`configs/basic/poisson`)

1. `2d_sinusoidal_source_response`: static potential field from a 2D sinusoidal source pattern.
2. `3d_sinusoidal_source_response`: static potential field from a 3D sinusoidal source pattern.

#### Schrodinger (`configs/basic/schrodinger`)

1. `2d_wavepacket_barrier_scattering`: Gaussian wave packet traveling toward and scattering off a localized potential bump.
2. `3d_wavepacket_barrier_scattering`: 3D Gaussian wave packet traveling toward and scattering off a localized potential bump.

#### Wave (`configs/basic/wave`)

1. `2d_localized_pulse_propagation`: localized initial velocity pulse radiating through a heterogeneous wave-speed field.
2. `3d_localized_pulse_propagation`: localized 3D velocity pulse radiating through a heterogeneous wave-speed field.

### Biology

#### Bistable Travelling Waves (`configs/biology/bistable_travelling_waves`)

1. `2d_planar_invasion_front`: planar invasion front moving rightward from a step initial condition.
2. `3d_planar_invasion_front`: 3D planar invasion front moving along `x` from a step initial condition.

#### Cyclic Competition (`configs/biology/cyclic_competition`)

1. `2d_spatial_rps_domains`: rock-paper-scissors competition from noise, forming moving patches and spiral-like domains.
2. `3d_spatial_rps_domains`: 3D rock-paper-scissors competition from noise, forming moving domains.

#### Fisher-KPP (`configs/biology/fisher_kpp`)

1. `2d_logistic_invasion_front`: logistic invasion front expanding from a left/right step profile.
2. `3d_logistic_invasion_front`: 3D logistic invasion front expanding along `x`.

#### FitzHugh-Nagumo (`configs/biology/fitzhugh_nagumo`)

1. `2d_excitable_spiral_waves`: excitable medium seeded by noise, developing traveling and spiral-like waves.
2. `3d_excitable_wavefronts`: 3D excitable medium seeded by noise, developing wavefronts and filament-like structures.

#### Gierer-Meinhardt (`configs/biology/gierer_meinhardt`)

1. `2d_turing_spots_and_stripes`: noise-seeded activator-inhibitor Turing spots and labyrinths.
2. `3d_turing_patterning`: 3D activator-inhibitor Turing patterning from noise.

#### Immunotherapy (`configs/biology/immunotherapy`)

1. `2d_tumor_immune_patch_dynamics`: tumor-immune-cytokine fields evolving into patchy competition and suppression zones.
2. `3d_tumor_immune_patch_dynamics`: 3D tumor-immune-cytokine competition with patchy heterogeneous regions.

#### Keller-Segel (`configs/biology/keller_segel`)

1. `2d_chemotactic_aggregation`: chemotactic aggregation from near-uniform noise into dense clusters.
2. `3d_chemotactic_aggregation`: 3D chemotactic aggregation into clumps and filaments.

#### Klausmeier Topography (`configs/biology/klausmeier_topography`)

1. `2d_uphill_vegetation_bands`: vegetation bands forming on a slope and migrating uphill.
2. `3d_uphill_vegetation_bands`: 3D sloped-terrain vegetation bands or sheets migrating uphill.

### Fluids

#### Burgers (`configs/fluids/burgers`)

1. `2d_shear_mode_interaction`: orthogonal sinusoidal velocity modes steepening, interacting, and diffusing on a periodic box.
2. `2d_noisy_shear_mode_interaction`: the same Burgers shear-mode interaction with additive noise.
3. `3d_shear_mode_interaction`: three sinusoidal velocity modes steepening and diffusing in a 3D periodic box.

#### Compressible Navier-Stokes (`configs/fluids/compressible_navier_stokes`)

1. `2d_localized_heating_expansion`: localized heating drives expansion and pressure waves in a closed box.
2. `3d_localized_heating_expansion`: 3D localized heating drives expansion and pressure waves in a closed box.

#### Darcy (`configs/fluids/darcy`)

1. `2d_pressure_driven_tracer_transport`: pressure-driven porous flow advecting and diffusing a tracer blob through heterogeneous mobility.

#### MHD (`configs/fluids/mhd`)

1. `2d_decaying_alfven_waves`: coupled velocity and magnetic wave patterns decaying under viscosity and resistivity.
2. `2d_orszag_tang_vortex`: Orszag-Tang vortex evolving into interacting current sheets and MHD turbulence.
3. `3d_cyclic_alfven_waves`: 3D periodic Alfvénic wave cells cycling and decaying.
4. `3d_helical_magnetic_cells`: helical flow cells coupled to magnetic structures in a 3D periodic box.

#### Navier-Stokes (`configs/fluids/navier_stokes`)

1. `2d_lid_driven_cavity_vortices`: lid-driven cavity building the primary vortex and corner eddies.
2. `2d_shear_layer_rollup`: perturbed double shear layer rolling up into vortices.
3. `2d_high_re_shear_layer_rollup`: higher-Re shear layer with finer roll-up and stronger secondary instabilities.
4. `3d_lid_driven_cavity_flow`: 3D lid-driven cavity starting from rest and building cavity circulation.

#### Shallow Water (`configs/fluids/shallow_water`)

1. `2d_rotating_gravity_wave_pulse`: localized free-surface hump radiating gravity waves with Coriolis deflection.

#### Stokes (`configs/fluids/stokes`)

1. `2d_lid_driven_cavity_flow`: steady lid-driven cavity recirculation.
2. `3d_lid_driven_cavity_flow`: steady 3D lid-driven cavity recirculation.

#### Thermal Convection (`configs/fluids/thermal_convection`)

1. `2d_rayleigh_benard_rolls`: noisy stratification between hot bottom and cold top evolving into convection rolls and plumes.
2. `3d_rayleigh_benard_plumes`: 3D Rayleigh-Benard-style convection evolving into plumes and cells.

### Physics

#### Brusselator (`configs/physics/brusselator`)

1. `2d_turing_labyrinths`: noise-seeded Brusselator Turing and labyrinth patterns.
2. `3d_turing_patterning`: 3D Brusselator pattern formation from noise.

#### Cahn-Hilliard (`configs/physics/cahn_hilliard`)

1. `2d_spinodal_decomposition`: spinodal decomposition into phase-separated domains.
2. `3d_spinodal_decomposition`: 3D spinodal decomposition and coarsening.

#### Complex Ginzburg-Landau (`configs/physics/cgl`)

1. `2d_benjamin_feir_turbulence`: Benjamin-Feir-unstable complex amplitude turbulence from a noisy background.
2. `3d_benjamin_feir_turbulence`: 3D complex amplitude turbulence from a noisy background.

#### Gray-Scott (`configs/physics/gray_scott`)

1. `2d_annular_spot_stripe_patterns`: Gray-Scott spot and stripe patterning on an annular domain from two seeded patches.
2. `2d_drifting_spot_stripe_patterns`: two seeded reagent patches generating drifting Gray-Scott spots and stripes.
3. `2d_noisy_spot_stripe_patterns`: Gray-Scott spots and stripes with stochastic roughening and variability.
4. `3d_spot_blob_patterns`: 3D Gray-Scott spot and blob structures from two seeded reagent patches.

#### Kuramoto-Sivashinsky (`configs/physics/kuramoto_sivashinsky`)

1. `2d_spatiotemporal_chaos`: noise-seeded wrinkling and spatiotemporal chaos on a periodic plane.
2. `3d_spatiotemporal_chaos`: 3D spatiotemporal chaos from small random perturbations.

#### Lorenz (`configs/physics/lorenz`)

1. `2d_spatial_lorenz_chaos`: spatially extended Lorenz chaos seeded from random initial fields.
2. `3d_spatial_lorenz_chaos`: 3D spatially extended Lorenz chaos seeded from random initial fields.

#### Maxwell (`configs/physics/maxwell`)

1. `2d_localized_em_radiation`: time-harmonic electromagnetic radiation from a localized source with absorbing walls.
2. `3d_localized_em_radiation`: 3D time-harmonic electromagnetic radiation from a localized source with absorbing walls.

#### Maxwell Pulse (`configs/physics/maxwell_pulse`)

1. `2d_guided_em_pulse`: localized electromagnetic pulse launched in a waveguide-like box with absorbing ends.
2. `3d_guided_em_pulse`: 3D localized electromagnetic pulse launched in a partially confined box.

#### Schnakenberg (`configs/physics/schnakenberg`)

1. `2d_turing_spots_and_labyrinths`: noise-seeded Schnakenberg Turing spots and labyrinths.
2. `3d_turing_patterning`: 3D Schnakenberg pattern formation from noise.

#### Superlattice (`configs/physics/superlattice`)

1. `2d_superlattice_pattern_formation`: coupled four-field reaction-diffusion pattern forming mixed-scale superlattice structures.
2. `3d_superlattice_pattern_formation`: 3D coupled four-field superlattice-style pattern formation.

#### Swift-Hohenberg (`configs/physics/swift_hohenberg`)

1. `2d_advected_roll_growth`: localized seed growing into an advected stripe or roll pattern.
2. `2d_directed_roll_growth`: stronger uniform advection producing more clearly directed stripe growth and drift.
3. `2d_rotating_roll_patterns`: noise-seeded pattern formation under rotational advection, giving swirling rolls.
4. `3d_advected_pattern_growth`: localized seed growing into a 3D patterned structure under weak uniform advection.

#### Van der Pol (`configs/physics/van_der_pol`)

1. `2d_oscillatory_wave_relaxation`: random low-amplitude modes relaxing into oscillatory Van der Pol wave patterns.
2. `3d_oscillatory_wave_relaxation`: 3D Van der Pol oscillatory wave patterns from random modal perturbations.

#### Zakharov-Kuznetsov (`configs/physics/zakharov_kuznetsov`)

1. `2d_soliton_pulse_propagation`: localized soliton-like pulse propagating and reshaping on a periodic domain.
2. `3d_soliton_pulse_propagation`: 3D localized soliton-like pulse propagating and reshaping.

## Geometry Expansion Roadmap

The annulus unlocked two useful effects at once: curved boundaries and an interior
obstacle. The next batch of domains should keep that spirit and target visibly
different dynamics rather than just add geometric variety.

### High-Priority Domains

1. `disk` (`outer`): smooth closed 2D baseline for radial invasion, wave focusing, spiral anchoring, and edge-pinned patterns.
2. `ellipse` (`outer`): almost as cheap as `disk`, but breaks rotational symmetry and splits otherwise degenerate modes.
3. `stadium` (`outer`): chaotic-billiard cavity for Helmholtz, wave, Schrodinger, Maxwell, and passive-mixing examples.
4. `l_shape` (`outer`): re-entrant corner for stress concentration, corner singularities, pinned fronts, and corner-driven diffusion.
5. `dumbbell` (`outer`): two chambers coupled by a narrow neck, giving bottleneck transport, delayed synchronization, and chamber competition.
6. `channel_obstacle` (`inlet`, `outlet`, `walls`, `obstacle`): canonical wake, diffraction, shielding, and bypass-flow geometry.
7. `eccentric_annulus` (`inner`, `outer`): keeps the obstacle mechanics of the annulus, but breaks symmetry and creates preferred drift paths.
8. `parallelogram` (`x-`, `x+`, `y-`, `y+`): lowest-risk non-orthogonal periodic cell for the periodic-only 2D families.
9. `trapezoid` (`outer`, or segmented `uphill`, `downhill`, `left`, `right`): good for slope-driven ecology and plume steering.
10. `cylinder` (`side`, `bottom`, `top`, or `inlet`, `outlet`, `side`): first useful smooth 3D domain for pulses, plumes, and cylindrical modes.
11. `spherical_shell` (`inner`, `outer`): 3D annulus analogue for shell waves, patterning, and convection.
12. `double_chamber_3d` (`outer`): 3D dumbbell / twin-lobe geometry for delayed exchange and filament transfer.
13. `skew_box` (`x-`, `x+`, `y-`, `y+`, `z-`, `z+`): 3D periodic analogue of `parallelogram` for the periodic-only 3D presets.

If the goal is maximum new-config coverage per unit of implementation work,
`disk`, `dumbbell`, `parallelogram`, and `channel_obstacle` should come first.
Together they cover most of the ideas below.

Periodic-only presets (`cahn_hilliard`, `cgl`, `kuramoto_sivashinsky`,
`lorenz`, `shallow_water`, and `zakharov_kuznetsov`) should mostly target
`parallelogram` first, then `skew_box`.

### Geometry-Driven Config Ideas By Preset

#### Basic

1. `advection` -> `2d_dumbbell_chamber_exchange`: a recirculating field moves a tracer blob between two lobes through a thin neck, highlighting residence times and incomplete mixing.
2. `elasticity` -> `2d_l_shape_corner_ringdown`: impulse-driven vibration of an L-bracket, with strong corner stresses and richer mode coupling than the cantilever.
3. `heat` -> `2d_dumbbell_bottleneck_diffusion`: one chamber starts hot and the neck controls the equilibration timescale between lobes.
4. `helmholtz` -> `2d_stadium_cavity_response`: localized forcing inside a stadium should show scar-like hot spots and geometry-dependent resonance structure.
5. `plate` -> `2d_disk_plate_mode_vibration`: circular plate modes give a very different vibration catalog from the current rectangular simply supported plate.
6. `poisson` -> `2d_l_shape_source_response`: static potential on an L-shape to capture corner singularities and distorted equipotential contours.
7. `schrodinger` -> `2d_stadium_wavepacket_billiard`: wave packet launched into a stadium cavity for quantum-billiard reflections and interference.
8. `wave` -> `2d_dumbbell_pulse_transfer`: a localized pulse starts in one chamber and partly tunnels through the neck, producing delayed secondary arrivals.

#### Biology

1. `bistable_travelling_waves` -> `2d_dumbbell_front_switching`: a front invades one chamber, stalls at the neck, then either crosses or pins depending on curvature and width.
2. `cyclic_competition` -> `2d_dumbbell_species_exchange`: spiral-like domains form independently in each lobe and intermittently invade through the bottleneck.
3. `fisher_kpp` -> `2d_disk_radial_invasion`: colony growth from a central seed in a disk gives a clean curved-front benchmark instead of a planar front.
4. `fitzhugh_nagumo` -> `2d_eccentric_annulus_reentry`: classic obstacle-anchored reentry, but with off-center geometry that biases drift and breakup.
5. `gierer_meinhardt` -> `2d_disk_edge_pinned_spots`: smooth closed boundary should favor ring-aligned spots and boundary-pinned pattern selection.
6. `immunotherapy` -> `2d_dumbbell_tumor_refuge_competition`: one lobe acts as an immune-privileged refuge while cells and cytokines leak through the neck.
7. `keller_segel` -> `2d_disk_boundary_aggregation`: chemotactic collapse in a smooth bounded domain should accentuate radial aggregation and edge crowding.
8. `klausmeier_topography` -> `2d_trapezoidal_hillslope_bands`: widening or narrowing slope geometry should bend vegetation bands and change migration speed across the domain.

#### Fluids

1. `burgers` -> `2d_parallelogram_shear_mode_interaction`: same steepening physics as the periodic box, but on a skew periodic cell that tilts shock alignment.
2. `compressible_navier_stokes` -> `2d_channel_obstacle_heating_expansion`: localized heating upstream of an obstacle creates expanding waves, recirculation pockets, and thermal shadowing.
3. `darcy` -> `2d_dumbbell_porous_exchange`: pressure-driven porous transport between two reservoirs connected by a thin throat.
4. `mhd` -> `2d_parallelogram_alfven_wave_cells`: skew periodic geometry tests how magnetic and velocity structures align with a non-orthogonal lattice.
5. `navier_stokes` -> `2d_cylinder_wake_rollup`: the obvious Gmsh-flow showcase, with vortex shedding and wake instability behind an obstacle.
6. `shallow_water` -> `2d_parallelogram_rotating_gravity_wave_pulse`: periodic gravity waves on a skew cell should make wave crests and Coriolis deflection visually less axis-aligned.
7. `stokes` -> `2d_creeping_flow_past_cylinder`: low-Re flow around an obstacle gives a clean steady benchmark before the full Navier-Stokes wake case.
8. `thermal_convection` -> `3d_spherical_shell_convection_cells`: geophysically flavored shell convection with plume columns, boundary layers, and curved-cell organization.

#### Physics

1. `brusselator` -> `2d_dumbbell_turing_bridge`: Turing structures emerge in each chamber and either lock across the neck or remain phase-misaligned.
2. `cahn_hilliard` -> `2d_parallelogram_spinodal_coarsening`: phase separation on a skew periodic cell tests whether coarsening statistics stay isotropic in the lattice basis.
3. `cgl` -> `2d_parallelogram_phase_turbulence`: Benjamin-Feir turbulence on a skew torus should reduce obvious grid alignment in the complex-wave field.
4. `gray_scott` -> `2d_dumbbell_spot_competition`: spot colonies nucleate in both lobes and compete for reagent flux through the connecting neck.
5. `kuramoto_sivashinsky` -> `2d_parallelogram_spatiotemporal_chaos`: the same chaos as the periodic plane, but with oblique lattice directions to break rectangular bias.
6. `lorenz` -> `2d_parallelogram_spatial_lorenz_chaos`: skew periodic geometry is a simple way to make the extended attractor less box-aligned.
7. `maxwell` -> `2d_stadium_cavity_radiation`: time-harmonic EM forcing in a stadium should show geometry-controlled standing-wave scars and hot spots.
8. `maxwell_pulse` -> `2d_dumbbell_em_pulse_transfer`: launch a pulse in one lobe and watch partial transmission, trapping, and reverberation through the neck.
9. `schnakenberg` -> `2d_disk_edge_pinned_labyrinths`: smooth closed boundaries should bias stripe bending and spot pinning near the outer wall.
10. `superlattice` -> `2d_disk_superlattice_targets`: circular geometry should favor concentric competition between the fast and slow pattern scales.
11. `swift_hohenberg` -> `2d_eccentric_annulus_roll_pinning`: roll patterns and defects should lock to the broken-symmetry inner obstacle in a way the centered annulus cannot.
12. `van_der_pol` -> `2d_disk_relaxation_targets`: relaxation oscillations in a disk may form cleaner target-like wave trains than the current box examples.
13. `zakharov_kuznetsov` -> `2d_parallelogram_oblique_soliton_drift`: a skew periodic cell is the cleanest way to encourage oblique propagation without adding non-periodic boundaries.
