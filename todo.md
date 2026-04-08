# TODO

## Existing PDE / Config Inventory

Current baseline: 38 PDE presets and 148 configs.

The descriptions below capture the main visible behavior of each committed
config so future additions can target genuinely different dynamics rather than
rename existing cases.

### Basic

#### Advection (`configs/basic/advection`)

1. `2d_absorbing_diagonal_stripes`: oblique stripe field swept diagonally into absorbing walls, ending in clipped boundary remnants instead of periodic wrap-around.
2. `2d_annulus_rotating_twin_blobs`: two blobs orbit around an annular ring and gradually smooth while staying confined between the inner and outer walls.
3. `2d_cellular_blob_advection`: one or two blobs stretched into curved filaments by periodic cellular flow with weak diffusion.
4. `2d_channel_obstacle_clockwise_bypass`: an off-center blob is swept beneath a non-centered circular obstacle, then curls downstream along the lower wake side.
5. `2d_channel_obstacle_counterclockwise_bypass`: a blob launched beside the cylinder rises over the top of the same channel-obstacle geometry, giving a complementary upper-bypass trajectory.
6. `2d_diagonal_blob_transport`: one or two blobs translated diagonally across a periodic box, with clear wrap-around at the edges.
7. `2d_disk_outward_ring_sweep`: concentric signed rings advected outward inside a disk and filtered by an absorbing outer boundary.
8. `2d_dumbbell_neck_sweep_advection`: a blob launched in the left lobe is swept through the neck into the right chamber before strong decay sets in.
9. `2d_multicell_blob_mixing`: four compact blobs are torn apart and redistributed by a fast multi-cell periodic mixer.
10. `2d_parallelogram_drifting_bands_advection`: oblique sine bands drift and bend on a skew periodic cell under mean flow plus cross-cell recirculation.
11. `2d_periodic_shear_quadrant_filamentation`: a quadrant pattern shears into smooth S-shaped filaments under periodic shear.
12. `2d_rotating_blob_advection`: one or two blobs rotate around the square center under weak diffusion and no-flux walls.
13. `3d_swirling_blob_advection`: one or two blobs carried by a 3D periodic swirling flow with weak diffusion.

#### Elasticity (`configs/basic/elasticity`)

1. `2d_annulus_breathing_mode_ringdown`: annular ring breathing radially around the inner hole, with stress wrapping around the cavity and the outer rim held fixed.
2. `2d_annulus_inner_twist_settling`: inner annulus rim prescribed in tangential twist against a fixed outer rim, producing a localized torsional settling wave around the hole.
3. `2d_annulus_torsional_ringdown`: annular torsional swirl ringing around a clamped inner hole, producing a clean low-order tangential shear mode.
4. `2d_cantilever_diagonal_shear_ringdown`: cantilever beam excited by mixed horizontal and vertical tip kicks, producing coupled diagonal shear and bending.
5. `2d_cantilever_impulse_ringdown`: slender cantilever ringing after a localized upward kick near the free tip, with persistent asymmetric bending.
6. `2d_channel_obstacle_frame_ringdown`: clamped perforated plate around a circular opening ringing in mirrored frame-like lobes after a quadrupolar hole-centered kick.
7. `2d_channel_obstacle_hole_lift_settling`: fixed outer frame with the circular hole lifted upward, creating an obstacle-centered settling transient with mirrored side lobes and concentrated frame stress.
8. `2d_disk_impulse_ringdown`: clamped elastic disk ringing after an off-center upward kick, producing curved cavity-like lobes and arc-shaped stress bands.
9. `2d_disk_quadrupole_boundary_settling`: disk outer rim prescribed in a static quadrupole shape, launching a smooth global four-lobe settling mode across the cavity.
10. `2d_disk_quadrupole_mode_ringdown`: clamped elastic disk seeded with a global quadrupole displacement mode that evolves through a symmetric standing-wave family.
11. `2d_dumbbell_lobe_transfer_ringdown`: one dumbbell lobe kicked first, with delayed displacement and stress transfer through the neck into the second chamber.
12. `2d_parallelogram_oblique_edge_pull_settling`: skew panel pulled obliquely along one side, giving slanted displacement packets and diagonal stress lobes that are not axis-locked.
13. `2d_parallelogram_oblique_shear_ringdown`: skew elastic panel with oblique shear packets and diagonal stress bands aligned to the parallelogram cell.
14. `2d_strip_sinusoidal_clamp_settling`: strip with a sinusoidally prescribed top clamp, producing a boundary-launched transverse settling pattern with repeated sign changes in the interior motion.
15. `2d_strip_transverse_mode_beating`: top/bottom-clamped strip seeded with two transverse modes, producing a slow standing-wave beating pattern.
16. `3d_cantilever_impulse_ringdown`: 3D cantilever bar ringing after a localized transverse velocity kick.

#### Heat (`configs/basic/heat`)

1. `2d_cosine_source_heating`: zero initial field driven by a standing cosine heat source.
2. `2d_dumbbell_bottleneck_diffusion`: one chamber starts hot and equilibrates with the other through a narrow neck.
3. `2d_localized_blob_diffusion`: one or two hot blobs diffusing under no-flux boundaries.
4. `2d_noisy_media_diffusion`: hot and cold blobs diffusing through noisy conductivity with additive state noise.
5. `2d_radial_diffusivity_spread`: hot spot spreading through radially varying conductivity with cold boundaries.
6. `3d_localized_blob_diffusion`: one or two hot blobs diffusing in a cube with no-flux walls.
7. `3d_noisy_media_diffusion`: hot and cold blobs diffusing in noisy 3D media with additive state noise.

#### Helmholtz (`configs/basic/helmholtz`)

1. `2d_disk_localized_oscillatory_response`: localized forcing in a circular cavity producing curved nodal arcs and resonance hot spots.
2. `2d_localized_oscillatory_response`: localized Gaussian forcing producing a confined oscillatory response.
3. `3d_standing_wave_response`: volumetric sine forcing producing a box-scale standing-wave response.

#### Plate (`configs/basic/plate`)

1. `2d_disk_mode_vibration`: circular simply supported plate vibrating from a low sinusoidal deflection mode.
2. `2d_simply_supported_mode_vibration`: simply supported plate vibrating from a single sinusoidal deflection mode.

#### Poisson (`configs/basic/poisson`)

1. `2d_disk_sinusoidal_source_response`: static potential field from a sinusoidal source pattern on a disk with a grounded outer rim.
2. `2d_sinusoidal_source_response`: static potential field from a 2D sinusoidal source pattern.
3. `3d_sinusoidal_source_response`: static potential field from a 3D sinusoidal source pattern.

#### Schrodinger (`configs/basic/schrodinger`)

1. `2d_disk_wavepacket_reflection`: Gaussian wave packet launched across a disk cavity, reflecting from the circular wall and a localized barrier.
2. `2d_wavepacket_barrier_scattering`: Gaussian wave packet traveling toward and scattering off a localized potential bump.
3. `3d_wavepacket_barrier_scattering`: 3D Gaussian wave packet traveling toward and scattering off a localized potential bump.

#### Wave (`configs/basic/wave`)

1. `2d_dumbbell_pulse_transfer`: localized pulse launched in one chamber with delayed transmission through the dumbbell neck.
2. `2d_localized_pulse_propagation`: localized initial velocity pulse radiating through a heterogeneous wave-speed field.
3. `3d_localized_pulse_propagation`: localized 3D velocity pulse radiating through a heterogeneous wave-speed field.

### Biology

#### Bistable Travelling Waves (`configs/biology/bistable_travelling_waves`)

1. `2d_dumbbell_invasion_front`: bistable invasion front filling one lobe and then crossing the dumbbell neck into the second chamber.
2. `2d_planar_invasion_front`: planar invasion front moving rightward from a step initial condition.
3. `3d_planar_invasion_front`: 3D planar invasion front moving along `x` from a step initial condition.

#### Cyclic Competition (`configs/biology/cyclic_competition`)

1. `2d_parallelogram_spatial_rps_domains`: rock-paper-scissors competition on a skew parallelogram cell, forming oblique moving patches and spiral-like domains.
2. `2d_spatial_rps_domains`: rock-paper-scissors competition from noise, forming moving patches and spiral-like domains.
3. `3d_spatial_rps_domains`: 3D rock-paper-scissors competition from noise, forming moving domains.

#### Fisher-KPP (`configs/biology/fisher_kpp`)

1. `2d_disk_radial_invasion`: logistic invasion expanding radially from a localized seed in a disk.
2. `2d_logistic_invasion_front`: logistic invasion front expanding from a left-right step profile.
3. `3d_logistic_invasion_front`: 3D logistic invasion front expanding along `x`.

#### FitzHugh-Nagumo (`configs/biology/fitzhugh_nagumo`)

1. `2d_excitable_spiral_waves`: excitable medium seeded by noise, developing traveling and spiral-like waves.
2. `2d_parallelogram_excitable_spiral_waves`: excitable spiral-wave dynamics on a skew parallelogram cell with reduced axis locking.
3. `3d_excitable_wavefronts`: 3D excitable medium seeded by noise, developing wavefronts and filament-like structures.

#### Gierer-Meinhardt (`configs/biology/gierer_meinhardt`)

1. `2d_parallelogram_turing_spots_and_stripes`: activator-inhibitor Turing spots and stripes on a skew parallelogram cell.
2. `2d_turing_spots_and_stripes`: noise-seeded activator-inhibitor Turing spots and labyrinths.
3. `3d_turing_patterning`: 3D activator-inhibitor Turing patterning from noise.

#### Immunotherapy (`configs/biology/immunotherapy`)

1. `2d_dumbbell_tumor_immune_patch_dynamics`: tumor-immune-cytokine patches evolving across two chambers connected by a narrow neck.
2. `2d_tumor_immune_patch_dynamics`: tumor-immune-cytokine fields evolving into patchy competition and suppression zones.
3. `3d_tumor_immune_patch_dynamics`: 3D tumor-immune-cytokine competition with patchy heterogeneous regions.

#### Keller-Segel (`configs/biology/keller_segel`)

1. `2d_chemotactic_aggregation`: chemotactic aggregation from near-uniform noise into dense clusters.
2. `2d_parallelogram_chemotactic_aggregation`: chemotactic aggregation on a skew parallelogram cell, reducing obvious rectangular bias in the cluster layout.
3. `3d_chemotactic_aggregation`: 3D chemotactic aggregation into clumps and filaments.

#### Klausmeier Topography (`configs/biology/klausmeier_topography`)

1. `2d_parallelogram_uphill_vegetation_bands`: uphill vegetation bands forming on a skew sloped cell with oblique band alignment.
2. `2d_uphill_vegetation_bands`: vegetation bands forming on a slope and migrating uphill.
3. `3d_uphill_vegetation_bands`: 3D sloped-terrain vegetation bands or sheets migrating uphill.

### Fluids

#### Burgers (`configs/fluids/burgers`)

1. `2d_noisy_shear_mode_interaction`: Burgers shear-mode interaction with additive noise.
2. `2d_parallelogram_shear_mode_interaction`: orthogonal shear-mode steepening on a skew parallelogram cell.
3. `2d_shear_mode_interaction`: orthogonal sinusoidal velocity modes steepening, interacting, and diffusing on a periodic box.
4. `3d_shear_mode_interaction`: three sinusoidal velocity modes steepening and diffusing in a 3D periodic box.

#### Compressible Euler (`configs/fluids/compressible_euler`)

1. `2d_reflective_strong_quadrant_shock_interaction`: strong reflective four-quadrant Riemann problem producing interacting shocks, contacts, and jets in a square box.
2. `2d_reflective_strong_quadrant_shock_interaction_lowres`: lower-resolution, shorter-horizon smoke-test version of the same strong reflective quadrant shock interaction.

#### Compressible Navier-Stokes (`configs/fluids/compressible_navier_stokes`)

1. `2d_channel_obstacle_heating_expansion`: localized heating in a channel with an obstacle drives expansion waves and a thermal wake around the blockage.
2. `2d_localized_heating_expansion`: localized heating drives expansion and pressure waves in a closed box.
3. `3d_localized_heating_expansion`: 3D localized heating drives expansion and pressure waves in a closed box.

#### Darcy (`configs/fluids/darcy`)

1. `2d_channel_obstacle_tracer_transport`: pressure-driven porous flow bends around a channel obstacle while advecting and diffusing a tracer blob.
2. `2d_pressure_driven_tracer_transport`: pressure-driven porous flow advecting and diffusing a tracer blob through heterogeneous mobility.

#### MHD (`configs/fluids/mhd`)

1. `2d_decaying_alfven_waves`: coupled velocity and magnetic wave patterns decaying under viscosity and resistivity.
2. `2d_orszag_tang_vortex`: Orszag-Tang vortex evolving into interacting current sheets and MHD turbulence.
3. `2d_parallelogram_decaying_alfven_waves`: decaying velocity-magnetic wave patterns on a skew parallelogram cell.
4. `3d_cyclic_alfven_waves`: 3D periodic Alfvenic wave cells cycling and decaying.
5. `3d_helical_magnetic_cells`: helical flow cells coupled to magnetic structures in a 3D periodic box.

#### Navier-Stokes (`configs/fluids/navier_stokes`)

1. `2d_channel_obstacle_vortex_shedding`: channel flow past a circular obstacle shedding an alternating wake.
2. `2d_high_re_shear_layer_rollup`: higher-Re shear layer with finer roll-up and stronger secondary instabilities.
3. `2d_lid_driven_cavity_vortices`: lid-driven cavity building the primary vortex and corner eddies.
4. `2d_shear_layer_rollup`: perturbed double shear layer rolling up into vortices.
5. `3d_lid_driven_cavity_flow`: 3D lid-driven cavity starting from rest and building cavity circulation.

#### Shallow Water (`configs/fluids/shallow_water`)

1. `2d_parallelogram_gravity_wave_pulse`: localized free-surface hump radiating gravity waves on a skew parallelogram cell.
2. `2d_rotating_gravity_wave_pulse`: localized free-surface hump radiating gravity waves with Coriolis deflection.

#### Stokes (`configs/fluids/stokes`)

1. `2d_channel_obstacle_flow`: steady creeping channel flow around a circular obstacle.
2. `2d_lid_driven_cavity_flow`: steady lid-driven cavity recirculation.
3. `3d_lid_driven_cavity_flow`: steady 3D lid-driven cavity recirculation.

#### Thermal Convection (`configs/fluids/thermal_convection`)

1. `2d_parallelogram_rayleigh_benard_rolls`: Rayleigh-Benard convection in a skew 2D cell, producing oblique rolls and plumes.
2. `2d_rayleigh_benard_rolls`: noisy stratification between hot bottom and cold top evolving into convection rolls and plumes.
3. `3d_rayleigh_benard_plumes`: 3D Rayleigh-Benard-style convection evolving into plumes and cells.

### Physics

#### Brusselator (`configs/physics/brusselator`)

1. `2d_parallelogram_turing_labyrinths`: Brusselator Turing labyrinths on a skew parallelogram cell.
2. `2d_turing_labyrinths`: noise-seeded Brusselator Turing and labyrinth patterns.
3. `3d_turing_patterning`: 3D Brusselator pattern formation from noise.

#### Cahn-Hilliard (`configs/physics/cahn_hilliard`)

1. `2d_parallelogram_spinodal_decomposition`: spinodal decomposition on a skew parallelogram cell.
2. `2d_spinodal_decomposition`: spinodal decomposition into phase-separated domains.
3. `3d_spinodal_decomposition`: 3D spinodal decomposition and coarsening.

#### Complex Ginzburg-Landau (`configs/physics/cgl`)

1. `2d_benjamin_feir_turbulence`: Benjamin-Feir-unstable complex amplitude turbulence from a noisy background.
2. `2d_parallelogram_benjamin_feir_turbulence`: Benjamin-Feir turbulence on a skew parallelogram cell.
3. `3d_benjamin_feir_turbulence`: 3D complex amplitude turbulence from a noisy background.

#### Gray-Scott (`configs/physics/gray_scott`)

1. `2d_annular_spot_stripe_patterns`: Gray-Scott spot and stripe patterning on an annular domain from two seeded patches.
2. `2d_drifting_spot_stripe_patterns`: two seeded reagent patches generating drifting Gray-Scott spots and stripes.
3. `2d_dumbbell_spot_stripe_patterns`: spot and stripe colonies seeded in both dumbbell lobes and coupled through the narrow neck.
4. `2d_noisy_spot_stripe_patterns`: Gray-Scott spots and stripes with stochastic roughening and variability.
5. `3d_spot_blob_patterns`: 3D Gray-Scott spot and blob structures from two seeded reagent patches.

#### Kuramoto-Sivashinsky (`configs/physics/kuramoto_sivashinsky`)

1. `2d_parallelogram_spatiotemporal_chaos`: wrinkling and spatiotemporal chaos on a skew parallelogram cell.
2. `2d_spatiotemporal_chaos`: noise-seeded wrinkling and spatiotemporal chaos on a periodic plane.
3. `3d_spatiotemporal_chaos`: 3D spatiotemporal chaos from small random perturbations.

#### Lorenz (`configs/physics/lorenz`)

1. `2d_parallelogram_spatial_lorenz_chaos`: spatially extended Lorenz chaos on a skew parallelogram cell.
2. `2d_spatial_lorenz_chaos`: spatially extended Lorenz chaos seeded from random initial fields.
3. `3d_spatial_lorenz_chaos`: 3D spatially extended Lorenz chaos seeded from random initial fields.

#### Maxwell (`configs/physics/maxwell`)

1. `2d_disk_localized_em_radiation`: time-harmonic electromagnetic radiation from a localized source in a circular cavity with absorbing outer wall.
2. `2d_localized_em_radiation`: time-harmonic electromagnetic radiation from a localized source with absorbing walls.
3. `3d_localized_em_radiation`: 3D time-harmonic electromagnetic radiation from a localized source with absorbing walls.

#### Maxwell Pulse (`configs/physics/maxwell_pulse`)

1. `2d_guided_em_pulse`: localized electromagnetic pulse launched in a waveguide-like box with absorbing ends.
2. `2d_parallelogram_guided_em_pulse`: localized electromagnetic pulse launched on a skew parallelogram cell.
3. `3d_guided_em_pulse`: 3D localized electromagnetic pulse launched in a partially confined box.

#### Schnakenberg (`configs/physics/schnakenberg`)

1. `2d_parallelogram_turing_spots_and_labyrinths`: Schnakenberg Turing spots and labyrinths on a skew parallelogram cell.
2. `2d_turing_spots_and_labyrinths`: noise-seeded Schnakenberg Turing spots and labyrinths.
3. `3d_turing_patterning`: 3D Schnakenberg pattern formation from noise.

#### Superlattice (`configs/physics/superlattice`)

1. `2d_parallelogram_superlattice_pattern_formation`: coupled four-field superlattice pattern formation on a skew parallelogram cell.
2. `2d_superlattice_pattern_formation`: coupled four-field reaction-diffusion pattern forming mixed-scale superlattice structures.
3. `3d_superlattice_pattern_formation`: 3D coupled four-field superlattice-style pattern formation.

#### Swift-Hohenberg (`configs/physics/swift_hohenberg`)

1. `2d_advected_roll_growth`: localized seed growing into an advected stripe or roll pattern.
2. `2d_directed_roll_growth`: stronger uniform advection producing more clearly directed stripe growth and drift.
3. `2d_disk_roll_patterns`: noise-seeded roll formation in a disk with rotational advection and circular boundary pinning.
4. `2d_rotating_roll_patterns`: noise-seeded pattern formation under rotational advection, giving swirling rolls.
5. `3d_advected_pattern_growth`: localized seed growing into a 3D patterned structure under weak uniform advection.

#### Van der Pol (`configs/physics/van_der_pol`)

1. `2d_oscillatory_wave_relaxation`: random low-amplitude modes relaxing into oscillatory Van der Pol wave patterns.
2. `2d_parallelogram_oscillatory_wave_relaxation`: oscillatory Van der Pol wave relaxation on a skew parallelogram cell.
3. `3d_oscillatory_wave_relaxation`: 3D Van der Pol oscillatory wave patterns from random modal perturbations.

#### Zakharov-Kuznetsov (`configs/physics/zakharov_kuznetsov`)

1. `2d_parallelogram_soliton_pulse_propagation`: localized soliton-like pulse propagating on a skew parallelogram cell.
2. `2d_soliton_pulse_propagation`: localized soliton-like pulse propagating and reshaping on a periodic domain.
3. `3d_soliton_pulse_propagation`: 3D localized soliton-like pulse propagating and reshaping.

## Geometry Expansion Roadmap

`annulus`, `disk`, `dumbbell`, `parallelogram`, and `channel_obstacle` now
have real config coverage. The next geometry pass should focus on shapes that
unlock genuinely new behavior rather than backfilling the first round of
special domains.

### High-Priority Domains

1. `ellipse` (`outer`): cheap symmetry breaking for radial waves, invasion fronts, and plate or Helmholtz modes.
2. `stadium` (`outer`): chaotic-billiard cavity for Helmholtz, wave, Schrodinger, and Maxwell examples.
3. `l_shape` (`outer`): re-entrant corner benchmark for elasticity, Poisson, heat, and wave singular behavior.
4. `eccentric_annulus` (`inner`, `outer`): obstacle-driven drift and pinning with broken radial symmetry.
5. `trapezoid` (`outer`, or segmented sides): slope-driven ecology and plume steering in a nonuniform-width domain.
6. `cylinder` (`side`, `bottom`, `top`, or `inlet`, `outlet`, `side`): first smooth 3D cavity or channel geometry for pulses, plumes, and wakes.
7. `spherical_shell` (`inner`, `outer`): 3D shell benchmark for convection, waves, and pattern formation.
8. `double_chamber_3d` (`outer`): 3D dumbbell analogue for delayed exchange and filament transfer.
9. `skew_box` (`x-`, `x+`, `y-`, `y+`, `z-`, `z+`): 3D periodic analogue of `parallelogram` for the periodic-only 3D families.

Most 2D periodic families now have a `parallelogram` config, so the next
periodic-geometry priority is `skew_box`.

### Remaining Geometry-Driven Config Ideas

#### Basic

1. `elasticity` -> `2d_l_shape_corner_ringdown`: impulse-driven vibration of an L-bracket, with stronger corner stresses and richer mode coupling than the cantilever and disk cases.
2. `helmholtz` -> `2d_stadium_cavity_response`: localized forcing inside a stadium should show scar-like hot spots and geometry-dependent resonance structure.
3. `poisson` -> `2d_l_shape_source_response`: static potential on an L-shape to capture corner singularities and distorted equipotential contours.
4. `schrodinger` -> `2d_stadium_wavepacket_billiard`: wave packet launched into a stadium cavity for billiard reflections and interference.

#### Biology

1. `cyclic_competition` -> `2d_dumbbell_species_exchange`: independent spiral domains in each lobe with intermittent invasion through the bottleneck.
2. `fitzhugh_nagumo` -> `2d_eccentric_annulus_reentry`: obstacle-anchored reentry with biased drift and breakup.
3. `keller_segel` -> `2d_disk_boundary_aggregation`: chemotactic collapse in a smooth bounded domain with stronger radial crowding.
4. `klausmeier_topography` -> `2d_trapezoidal_hillslope_bands`: widening or narrowing slope geometry bending vegetation bands across the domain.

#### Fluids

1. `darcy` -> `2d_dumbbell_porous_exchange`: pressure-driven porous transport between two reservoirs connected by a thin throat.
2. `navier_stokes` -> `2d_cylinder_wake_rollup`: the canonical smooth-obstacle wake case, complementing the existing `channel_obstacle` example.
3. `stokes` -> `2d_creeping_flow_past_cylinder`: steady low-Re cylinder flow as the clean companion to the channel-obstacle case.
4. `thermal_convection` -> `3d_spherical_shell_convection_cells`: shell convection with plume columns, boundary layers, and curved-cell organization.

#### Physics

1. `brusselator` -> `2d_dumbbell_turing_bridge`: Turing structures emerge in each chamber and either lock across the neck or remain phase-misaligned.
2. `maxwell` -> `2d_stadium_cavity_radiation`: time-harmonic EM forcing in a stadium should show geometry-controlled standing-wave scars and hot spots.
3. `maxwell_pulse` -> `2d_dumbbell_em_pulse_transfer`: launch a pulse in one lobe and watch partial transmission and trapping through the neck.
4. `schnakenberg` -> `2d_disk_edge_pinned_labyrinths`: smooth closed boundaries should bias stripe bending and spot pinning near the outer wall.
5. `superlattice` -> `2d_disk_superlattice_targets`: circular geometry should favor concentric competition between the fast and slow pattern scales.
6. `swift_hohenberg` -> `2d_eccentric_annulus_roll_pinning`: roll patterns and defects should lock to the broken-symmetry inner obstacle.
7. `van_der_pol` -> `2d_disk_relaxation_targets`: relaxation oscillations in a disk may form cleaner target-like wave trains than the rectangular and parallelogram cases.
