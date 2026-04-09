# TODO

## Existing PDE / Config Inventory

Current baseline: 38 PDE presets and 313 configs.

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
13. `2d_airfoil_channel_lower_surface_wake_flip`: compact packets skim the lower airfoil surface, then flip into a narrow downstream wake with strong seed-to-seed asymmetry.
14. `2d_airfoil_channel_upper_surface_skimming`: upstream blobs ride over the top surface of the airfoil and peel into a smooth upper wake ribbon.
15. `2d_annulus_inward_spiral_filaments`: signed filaments spiral around the annulus while an inward radial bias pulls structure back toward the inner wall.
16. `2d_disk_saddle_filament_collapse`: a saddle-like velocity field stretches a disk-confined blob pair into crossing filaments before they collapse toward the absorbing rim.
17. `2d_dumbbell_bridge_vortex_exchange`: opposite-signed structures circulate within both lobes and repeatedly exchange mass through the narrow bridge.
18. `2d_l_shape_corner_capture`: a drifting packet is pulled into the re-entrant corner of an L-shaped cavity, leaving a trapped corner remnant behind.
19. `2d_l_shape_reentrant_shear_fold`: layered bands sweep past the L-shaped notch and fold into sharp re-entrant corner filaments under skew shear.
20. `2d_multi_hole_plate_interstitial_orbit`: compact blobs thread between multiple holes and loop around the solid islands instead of crossing the plate directly.
21. `2d_multi_hole_plate_weaving_shadow`: a staggered plume weaves through the perforations, leaving alternating sheltered low-amplitude shadows behind the holes.
22. `2d_porous_channel_reverse_flush`: a reverse-biased plume is forced back through a porous obstacle field, producing short recirculating pockets between pillars.
23. `2d_porous_channel_sinuous_percolation`: two inlet blobs percolate sinuously through the porous matrix, repeatedly splitting and rejoining between obstacles.
24. `2d_serpentine_channel_lane_reversal`: broad bands traverse a serpentine duct with alternating cross-stream drift, giving lane switches at successive bends.
25. `2d_serpentine_channel_slug_release`: a striped slug fills the serpentine passage and rides the full bend sequence with strong phase differences across seeds.
26. `2d_side_cavity_channel_cavity_entrainment`: a passing channel plume is pulled down into a side cavity, then stretched back out along the cavity mouth.
27. `2d_side_cavity_channel_cavity_purge`: material initially stored in the side cavity is purged into the main channel, giving a cavity-emptying transient rather than inlet-fed transport.
28. `2d_venturi_channel_centerline_focusing`: upstream blobs are compressed through the venturi throat into a narrow high-speed centerline jet before diffusing downstream.
29. `2d_venturi_channel_wall_peeling`: wall-adjacent structure is sheared off the venturi boundary and peeled into elongated downstream ribbons.
30. `2d_y_bifurcation_branch_merge_collision`: packets launched from the outlet branches advect upstream and collide near the bifurcation junction.
31. `2d_y_bifurcation_inlet_split_plume`: an inlet plume splits unevenly into the two daughter branches, with the junction geometry setting the branch preference.
32. `3d_swirling_blob_advection`: one or two blobs carried by a 3D periodic swirling flow with weak diffusion.

#### Elasticity (`configs/basic/elasticity`)

All Gmsh-backed 2D elasticity configs now sample their geometry as well as
their excitation / material parameters, so seed changes alter both the shape
and the resulting transient.

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
17. `2d_airfoil_lift_skin_settling`: airfoil boundary lifted and sheared inside a sampled channel, creating a strong body-centered settling wake rather than an upstream launch.
18. `2d_airfoil_wake_scatter_ringdown`: an upstream kick scatters around a sampled airfoil, producing asymmetric channel-filling wake lobes and moving stress shadows.
19. `2d_dumbbell_countertwist_settling`: the dumbbell outer boundary is prescribed in a global counter-twist, driving opposite lobe rotation and neck-focused torsional stress.
20. `2d_l_shape_notch_shear_settling`: the reentrant notch is pulled diagonally against a fixed outer frame, turning the inside corner into a strong shear actuator.
21. `2d_l_shape_reentrant_corner_ringdown`: a localized kick near the reentrant corner launches corner-trapped oscillations that spill differently into the two L-arms.
22. `2d_multi_hole_plate_opposed_hole_pull_settling`: grouped plate holes are pulled in different directions, producing perforation-to-perforation stress competition instead of one global plate mode.
23. `2d_multi_hole_plate_perforation_ringdown`: off-center kicks among several sampled holes create hole-scattered ringdown patterns and punctured stress halos.
24. `2d_porous_channel_maze_ringdown`: a porous channel is kicked from one side, generating maze-like elastic transfer through the obstacle lattice rather than a single open-channel mode.
25. `2d_porous_channel_obstacle_lift_settling`: the obstacle array is lifted against fixed ends, producing a pore-by-pore settling field with strongly seed-dependent lattice asymmetry.
26. `2d_serpentine_end_pull_settling`: one serpentine endpoint is pulled obliquely against the other, giving bend-to-bend settling with geometry-driven sign flips.
27. `2d_serpentine_lane_launch_ringdown`: a compact launch in the entrance lane propagates through the sampled turns as a delayed waveguide-style ringdown.
28. `2d_side_cavity_release_ringdown`: motion seeded in the side cavity leaks back into the main channel, producing a delayed reservoir-release transient.
29. `2d_side_cavity_roof_pull_settling`: a strong, localized cavity-roof actuation drives a cavity-confined settling plume that now varies sharply across seeds.
30. `2d_venturi_axial_squeeze_settling`: opposing inlet/outlet pulls squeeze a sampled venturi, concentrating displacement and stress through the throat.
31. `2d_venturi_throat_focus_ringdown`: paired kicks above and below the constriction focus motion into the venturi throat before it re-expands downstream.
32. `2d_y_bifurcation_branch_transfer_ringdown`: one bifurcation branch is kicked harder than the other, generating a branch-to-junction transfer transient instead of symmetric splitting.
33. `2d_y_bifurcation_split_arm_settling`: the two outlet arms are prescribed apart, launching a clean branch-opening settling mode through the Y junction.

#### Heat (`configs/basic/heat`)

All 2D heat configs now output at least `128x128` and use sampled coefficients,
geometry, boundary values, or initial conditions so seed changes are visible.

1. `2d_annulus_inner_heating_layer`: hot inner rim and cooled outer rim drive a ring-shaped heating layer while azimuthal seed blobs relax around the annulus.
2. `2d_annulus_outer_quench_rings`: concentric signed rings relax between a cold inner rim and a hot outer rim, reversing the annular boundary forcing direction.
3. `2d_airfoil_channel_heat_shadow`: mixed signed transients and hot-to-cold channel forcing create an airfoil-centered thermal wake with wall-guided downstream cooling.
4. `2d_airfoil_channel_hot_skin_relaxation`: oblique signed bands diffuse around a hot airfoil, forming surface-hugging thermal layers instead of inlet-to-outlet forcing.
5. `2d_channel_obstacle_heat_shadow`: mixed signed transients and hot-to-cold channel forcing create an obstacle-centered thermal shadow with asymmetric wall-guided fronts.
6. `2d_channel_obstacle_hot_cylinder_cooling`: a heated cylinder diffuses into a cooled channel, generating obstacle-centered fronts without the previous inlet-driven wake asymmetry.
7. `2d_cosine_source_heating`: a sampled quadrant start is overwritten by standing source-sink forcing, settling toward a seed-dependent four-cell heating pattern.
8. `2d_cross_gradient_quadrant_relaxation`: hot-left and cool-right boundaries pull a sharp quadrant field into a directional cross-gradient relaxation front.
9. `2d_disk_boundary_quench`: concentric hot rings inside a disk collapse inward and smooth out against a cold outer rim.
10. `2d_dumbbell_bottleneck_diffusion`: signed lobe imbalance equalizes slowly through a sampled neck, giving a clear two-chamber transfer transient.
11. `2d_l_shape_corner_heating`: a hot reentrant notch diffuses into a grounded outer frame, emphasizing corner-driven filling of the L-shaped cavity.
12. `2d_l_shape_reentrant_quench`: a mixed-sign quadrant start is pulled between warm outer walls and a cold notch, turning the reentrant corner into a sharp thermal sink.
13. `2d_localized_blob_diffusion`: one or two elongated hot spots diffuse and merge under insulated walls, serving as the clean slow-diffusion baseline.
14. `2d_multi_hole_plate_hole_heating`: uniformly hot holes diffuse into a colder perforated plate, yielding symmetric halo growth around the sampled hole layout.
15. `2d_multi_hole_plate_thermal_dipoles`: hot, warm, and cold holes compete across the plate, producing interacting thermal dipoles instead of uniform hole heating.
16. `2d_noisy_media_diffusion`: hot and cold blobs dissolve inside noisy conductivity with additive state noise, keeping high seed-to-seed texture variation.
17. `2d_parallelogram_periodic_stripe_relaxation`: oblique periodic bands decay on a skew cell without rectangular axis locking or wall effects.
18. `2d_porous_channel_hot_obstacle_array`: many heated inclusions radiate into a near-neutral porous channel, creating multi-source halos through the pore lattice.
19. `2d_porous_channel_trapping_diffusion`: hot and cold transients mix through a sampled porous lattice while chilled obstacles pin localized trapping zones.
20. `2d_radial_diffusivity_spread`: a warm square with signed interior perturbations cools toward zero walls while radially varying diffusivity shapes a centered relaxation halo.
21. `2d_serpentine_channel_guided_diffusion`: end-forced heating relaxes through a sampled serpentine path, giving a slow bend-to-bend filling transient.
22. `2d_serpentine_channel_lane_band_relaxation`: signed lane-aligned bands smooth through the bends with no-flux walls, making the geometry alone control the equilibration.
23. `2d_side_cavity_channel_cavity_source_release`: a strong cavity heater fights signed channel seed blobs and asymmetric end temperatures, so the side pocket leaks a bright plume back into a counter-biased main channel.
24. `2d_side_cavity_channel_delayed_release`: inlet-heated transients pool in the cavity and then bleed back into the outlet corridor after a sampled delay.
25. `2d_venturi_channel_oblique_stripe_relaxation`: oblique hot/cold bands squeeze through the venturi throat and smooth without any boundary forcing.
26. `2d_venturi_channel_throat_focusing`: inlet-driven heat diffuses through a constricted throat, producing a channel-scale thermal pinch point.
27. `2d_y_bifurcation_branch_imbalance_relaxation`: upper and lower branches start at opposite temperatures and equalize through the junction under zero-flux boundaries.
28. `2d_y_bifurcation_split_diffusion`: a hot trunk start diffuses into two cooled branches, giving a clean branch-splitting relaxation front.
29. `3d_localized_blob_diffusion`: one or two hot blobs diffusing in a cube with no-flux walls.
30. `3d_noisy_media_diffusion`: hot and cold blobs diffusing in noisy 3D media with additive state noise.

#### Helmholtz (`configs/basic/helmholtz`)

1. `2d_disk_localized_oscillatory_response`: localized forcing in a circular cavity producing curved nodal arcs and resonance hot spots.
2. `2d_localized_oscillatory_response`: localized Gaussian forcing producing a confined oscillatory response.
3. `3d_standing_wave_response`: volumetric sine forcing producing a box-scale standing-wave response.

#### Plate (`configs/basic/plate`)

All plate configs now use sampled geometry, material, and excitation parameters,
every 2D case outputs at least `128x128`, and every current Gmsh-backed 2D
plate domain has at least two distinct configs.

1. `2d_airfoil_channel_surface_skimming_ringdown`: an upstream oblique ringdown skims the airfoil surface and relaxes into thin wall-following wake ribbons.
2. `2d_airfoil_channel_wake_load_settling`: a downstream-biased load plus oblique standing modes create asymmetric wake pockets behind the airfoil.
3. `2d_annulus_multilobe_load_settling`: slowly forced annular plate settling into low-amplitude multi-lobe sag patterns around the inner hole.
4. `2d_annulus_radial_ringdown`: concentric annular ringdown with sampled radial spacing and a slight azimuthal asymmetry.
5. `2d_channel_obstacle_obstacle_halo_settling`: obstacle-centered loading creates a perforation halo that settles into screened lobes around the circular cutout.
6. `2d_channel_obstacle_shadow_scatter`: an upstream kick scatters around a circular obstacle, leaving a seed-dependent wake lobe on one side of the perforation.
7. `2d_disk_mode_vibration`: a disk seeded by concentric deflection plus an off-center impulse, producing circular-to-spiral mode mixing.
8. `2d_dumbbell_lobe_transfer_ringdown`: a left-chamber kick drives delayed neck-coupled motion toward the bridge and softer right chamber.
9. `2d_l_shape_arm_exchange_settling`: asymmetric forcing trades amplitude between the two arms of an L-shaped plate while the reentrant corner redirects the response.
10. `2d_l_shape_reentrant_corner_ringdown`: a corner-focused ringdown highlights the L-shape singular corner and the reflected diagonals it launches.
11. `2d_multi_hole_plate_interstitial_ringdown`: compact motion threads through the ligaments between holes, making the solid web ring more than the plate exterior.
12. `2d_multi_hole_plate_perforation_load_settling`: an off-axis perforated-plate load settles into hole-shadowed lobes and screened pockets around the sampled holes.
13. `2d_parallelogram_oblique_ripple_beating`: skew plate with oblique multi-mode beating and slanted nodal bands that avoid axis locking.
14. `2d_porous_channel_percolating_ringdown`: a broad porous-channel ringdown leaks through different pore corridors each seed, producing percolation-like transport of plate motion.
15. `2d_porous_channel_trapped_pocket_settling`: localized loading leaves trapped high-amplitude pockets in sheltered porous cavities before they slowly relax.
16. `2d_quadrant_snapthrough_beating`: quadrant release pattern breaking into energetic lobe exchange and high-contrast mode conversion.
17. `2d_serpentine_channel_bend_wave_beating`: a short-lived bend-localized packet sweeps across alternating turns before damping out, giving a compact fast transient.
18. `2d_serpentine_channel_lane_switch_ringdown`: an inlet-biased impulse switches the dominant serpentine lane between bends under a sampled stiffness gradient.
19. `2d_side_cavity_channel_cavity_release_ringdown`: cavity-trapped motion releases into the main duct and then rebounds back into the side pocket.
20. `2d_side_cavity_channel_mouth_beating_settling`: mouth-focused forcing drives alternating cavity-duct exchange with a stronger settling lobe at the cavity lip.
21. `2d_simply_supported_mode_vibration`: rectangular simply supported plate with oblique multimode beating instead of one textbook standing mode.
22. `2d_stiffness_interface_scatter`: localized kick hitting a left-right stiffness jump and refracting into asymmetric lobes.
23. `2d_strip_release_fronts`: strip-like step release launching axial fronts and broad standing packets with slower relaxation.
24. `2d_venturi_channel_oblique_throat_settling`: oblique excitation focuses through the venturi throat and leaves a compressed downstream lobe after the constriction.
25. `2d_venturi_channel_throat_focusing_ringdown`: a throat-centered packet rings between converging walls before expanding into the outlet section.
26. `2d_y_bifurcation_branch_split_ringdown`: an inlet pulse splits unevenly between the two branches and reflects off sampled branch geometry.
27. `2d_y_bifurcation_junction_collision_beating`: counter-signed branch excitations collide at the junction and radiate back up the trunk in a short beating transient.

#### Poisson (`configs/basic/poisson`)

1. `2d_disk_sinusoidal_source_response`: static potential field from a sinusoidal source pattern on a disk with a grounded outer rim.
2. `2d_sinusoidal_source_response`: static potential field from a 2D sinusoidal source pattern.
3. `3d_sinusoidal_source_response`: static potential field from a 3D sinusoidal source pattern.

#### Schrodinger (`configs/basic/schrodinger`)

1. `2d_annulus_radial_spoke_interference`: a top-launched annular packet fires inward toward the hole and rebounds into spoke-like radial interference under a weak cross-ring lattice.
2. `2d_annulus_ring_orbit_scattering`: a tangential packet orbits around the annulus, repeatedly shearing against the inner and outer rims while the sampled ring potential modulates its angular speed.
3. `2d_channel_obstacle_upper_bypass_scattering`: an upstream packet clips above the cylinder and leaves a clean upper-side bypass wake inside the channel frame.
4. `2d_channel_obstacle_wake_corral_echoes`: a lower launch slips around the cylinder and becomes trapped intermittently in a sampled downstream wake well, producing delayed echo packets.
5. `2d_disk_quantum_corral_beating`: broad low-order disk modes beat against a sampled radial corral potential, producing slow rotating lobe patterns instead of one reflected packet.
6. `2d_disk_wavepacket_reflection`: an off-center packet crosses the disk, ricochets off the circular wall, and interferes with a sampled interior barrier.
7. `2d_dumbbell_neck_tunneling`: a left-lobe packet tunnels through the sampled neck against a weak neck barrier, with delayed transfer into the opposite chamber.
8. `2d_dumbbell_transverse_lobe_ricochet`: a diagonal launch in one lobe ricochets off the lobe wall before being pulled through the neck toward a sampled capture well in the opposite chamber.
9. `2d_l_shape_notch_trap_echoes`: a top-arm launch falls toward the re-entrant corner and lingers around a sampled notch well, producing repeated corner echoes.
10. `2d_l_shape_reentrant_corner_diffraction`: a lower-arm packet runs directly into the L-shaped corner and breaks into sharply bent diffraction fronts.
11. `2d_multi_hole_plate_diagonal_weave_scattering`: a diagonal packet threads a sampled perforated plate under a step potential, weaving between holes instead of taking a straight corridor.
12. `2d_multi_hole_plate_multislit_diffraction`: a left-to-right launch turns the sampled hole pattern into a multi-slit scatterer with clear downstream interference fans.
13. `2d_periodic_strip_lattice_interference`: a diagonal packet wraps in the periodic streamwise direction while a sampled strip lattice bends it into repeated banded interference.
14. `2d_side_cavity_channel_cavity_release`: a packet seeded inside the side pocket is released through the cavity mouth and spills back into the main duct in bursts.
15. `2d_side_cavity_channel_passby_capture`: a channel packet passes the cavity opening and sheds part of its mass into a sampled cavity well before re-emerging downstream.
16. `2d_venturi_channel_throat_focusing`: an inlet packet is squeezed through the sampled venturi throat and sharpened by a weak throat well before expanding into the outlet.
17. `2d_venturi_channel_wall_ricochet`: an upper-wall launch bounces through the converging venturi section and refracts across the throat under a sampled cross-stream bias.
18. `2d_wavepacket_barrier_scattering`: a fast rectangular wave packet crosses a sampled box and scatters off a moving localized barrier with seed-dependent impact offsets.
19. `2d_y_bifurcation_junction_rebound_echoes`: a junction-centered packet rebounds into the two branches and back into the trunk, with a sampled vertical bias making one arm dominate.
20. `2d_y_bifurcation_split_branch_scattering`: an inlet packet hits a sampled junction barrier and splits asymmetrically into the two daughter branches.
21. `3d_wavepacket_barrier_scattering`: 3D Gaussian wave packet traveling toward and scattering off a localized potential bump.

#### Wave (`configs/basic/wave`)

1. `2d_annulus_arc_interference`: opposing pulses launched on an annulus circulate around the hole and interfere under a clamped inner rim and reflective outer rim.
2. `2d_channel_obstacle_scattering`: an upstream pulse scatters around a circular obstacle in a channel-like cavity, leaving wall-guided wake structures downstream.
3. `2d_dirichlet_mode_beating`: multiple low-order standing modes ring inside a fully clamped square cavity, producing a distorted beating pattern rather than a single textbook mode.
4. `2d_disk_focusing_reflections`: an off-center pulse in a circular cavity reflects from the outer wall and refocuses into crescent-shaped interference bands.
5. `2d_dumbbell_pulse_transfer`: localized left-lobe excitation crosses the dumbbell neck with delayed energy transfer into the second chamber.
6. `2d_layered_refraction_fronts`: a pulse launched toward a sharp wave-speed interface splits into reflected and refracted fronts with visibly different propagation speeds.
7. `2d_localized_pulse_propagation`: an off-center pulse radiates through a heterogeneous radial lens field, breaking into irregular curved fronts.
8. `2d_parallelogram_wraparound_lensing`: a localized pulse wraps around a skew periodic cell and is distorted by an oblique lensing medium.
9. `2d_periodic_oblique_wave_train`: oblique stripe-like wave trains circulate on a fully periodic box without wall reflections.
10. `2d_periodic_strip_waveguide`: x-periodic, y-clamped strip dynamics behave like a guided wave corridor with lateral reflections and repeated focusing.
11. `3d_localized_pulse_propagation`: localized 3D velocity pulse radiating through a heterogeneous wave-speed field.
12. `2d_airfoil_channel_bow_front_scattering`: a broad front sweeps past a sampled airfoil and produces bow-wave shadowing plus downstream wake refraction.
13. `2d_airfoil_channel_surface_pulse_ringing`: oblique modal excitation rings around a sampled airfoil with partly absorbing outer boundaries, giving asymmetric upper/lower-surface responses.
14. `2d_annulus_absorbing_ring_modes`: radial ring modes launched on a sampled annulus decay against an absorbing outer rim while the inner hole re-radiates echoes.
15. `2d_channel_obstacle_bow_front_scattering`: a stronger planar front hits the sampled cylinder and sheds asymmetric reflected bands into the downstream cavity.
16. `2d_disk_robin_mode_beating`: sampled modal mixtures ring inside a partially absorbing disk, producing rotating lobe patterns instead of a single focal reflection.
17. `2d_dumbbell_counterlobe_collision`: opposite-sign pulses launched from both lobes collide in the sampled neck and reflect back with asymmetric chamber timing.
18. `2d_l_shape_notch_mode_beating`: oblique standing-wave mixtures ring inside an L-shaped cavity with notch absorption, emphasizing re-entrant-corner mode distortions.
19. `2d_l_shape_reentrant_echoes`: a localized launch drives echoes off the notch corner and outer walls, producing delayed corner focusing.
20. `2d_multi_hole_plate_corridor_scatter`: a left-to-right pulse threads the perforated corridors and sheds hole-shadow interference bands.
21. `2d_multi_hole_plate_interstitial_beating`: quadrant-seeded interstitial modes rattle between differently conditioned holes, producing patchwork plate ringing.
22. `2d_porous_channel_percolation_echoes`: inlet-side pulses percolate through the sampled obstacle lattice and split into maze-like echo paths.
23. `2d_porous_channel_reverse_maze_ringdown`: a reverse launch from the outlet side drives obstacle-to-obstacle ringdown through the porous maze.
24. `2d_serpentine_channel_bend_mode_ringing`: distributed modal excitation fills the sampled bends with alternating corner hot spots and delayed turn-to-turn echoes.
25. `2d_serpentine_channel_guided_lane_ringdown`: an inlet lane launch follows the sampled serpentine guide and reflects from successive bends with clear travel-time delays.
26. `2d_side_cavity_channel_cavity_slosh`: a cavity-centered ringdown sloshes between the pocket and main channel, generating high-energy cavity resonances.
27. `2d_side_cavity_channel_passby_echoes`: a channel pulse couples energy into the side pocket and releases delayed echoes back into the main stream.
28. `2d_venturi_channel_throat_pulse_focusing`: a localized launch focuses through the sampled throat and expands into downstream refraction fans.
29. `2d_venturi_channel_wall_mode_ringing`: distributed wall modes ring inside the venturi and highlight repeated throat-to-wall reflections.
30. `2d_y_bifurcation_branch_echo_collision`: opposite branch launches collide at the junction and re-split into the outlet arms.
31. `2d_y_bifurcation_split_pulse`: an inlet pulse divides across the sampled bifurcation and produces asymmetric arm responses from mixed branch boundary conditions.

### Biology

#### Bistable Travelling Waves (`configs/biology/bistable_travelling_waves`)

The 2D suite now spans 10 domain families with 2 configs each, covering
fast and slow fronts, threshold-driven retreat, periodic coarsening, and
boundary-fed invasion through sampled Gmsh geometries.

1. `2d_annulus_azimuthal_front_wraparound`: elongated supercritical arc on an annulus is swept azimuthally by sampled rotation while diffusion and bistability smooth it into a travelling ring segment.
2. `2d_annulus_inner_rim_pinned_front`: inner annulus rim is held at `u=1` and slowly twists against an absorbing outer wall, producing a boundary-pinned outward invasion layer.
3. `2d_channel_obstacle_boundary_forced_bypass`: inlet-fed population front is forced around a sampled cylinder, leaving a clear obstacle shadow and asymmetric bypass path.
4. `2d_channel_obstacle_wake_recolonization`: localized patches upstream and in the wake interact with weak recirculation, giving obstacle-mediated recolonization instead of a clean boundary-driven fill.
5. `2d_disk_radial_survival_threshold`: one or two blobs in a sampled disk either settle into a supercritical radial invasion or hover near the critical survival size.
6. `2d_disk_robin_ring_collapse`: concentric radial-cosine bands in a disk retreat and smooth under an absorbing Robin wall, emphasizing shrinking ring-like fronts.
7. `2d_dumbbell_dual_lobe_threshold_exchange`: unequal lobe seeds in a narrow dumbbell compete near threshold, with one chamber often dominating the neck transfer.
8. `2d_dumbbell_invasion_front`: broad left-lobe step front fills a sampled dumbbell and is nudged through the bridge by weak rightward drift.
9. `2d_l_shape_corner_refuge_invasion`: supercritical corner seed expands along both L-shaped arms while the re-entrant notch acts as a no-flux refuge.
10. `2d_l_shape_notch_quench_retreat`: quadrant-style initial data is driven upward and partially quenched by a lossy notch, producing retreat biased toward the re-entrant corner.
11. `2d_multi_hole_plate_hole_shadow_retreat`: sampled quadrants retreat across a perforated plate while mixed hole conditions carve out persistent obstacle-shadow patterns.
12. `2d_multi_hole_plate_perforation_invasion`: left-to-right bistable front threads through sampled holes, with the center hole held at zero to split and delay the advancing interface.
13. `2d_parallelogram_swept_quadrant_invasion`: skew-periodic quadrant data is advected obliquely across a sampled parallelogram, giving wraparound invasion rather than a static phase split.
14. `2d_parallelogram_wraparound_stripe_relaxation`: oblique periodic bands coarsen and compete on a sampled skew cell, giving slow wraparound stripe selection.
15. `2d_periodic_shear_band_competition`: near-threshold sine bands in a periodic rectangle shear and merge slowly, targeting long-lived coarsening instead of a single fast front.
16. `2d_planar_invasion_front`: sampled rectangle supports a fast classical planar invasion front with randomized width, extent, and launch position.
17. `2d_serpentine_channel_guided_front`: inlet Dirichlet forcing launches a front that follows sampled serpentine bends with clear lane-guided advance.
18. `2d_serpentine_channel_reverse_lane_quench`: downstream seed is pushed back through the sampled serpentine against wall losses, giving a reverse-lane retreat rather than inlet growth.
19. `2d_y_bifurcation_asymmetric_branch_competition`: unequal daughter-branch seeds compete while a weak upstream drift pulls the winner back toward the junction.
20. `2d_y_bifurcation_branch_split_front`: inlet-fed front fills a sampled Y-branch and splits into the daughter arms with branch-angle-dependent timing.
21. `3d_planar_invasion_front`: 3D planar invasion front moving along `x` from a step initial condition.

#### Cyclic Competition (`configs/biology/cyclic_competition`)

The 2D suite now spans 10 domain families with 2 configs each: one set focuses
on self-organized spiral or stripe breakup, while the companion configs use
mixed boundaries or sampled geometry to drive directional chases and pinning.

1. `2d_annulus_azimuthal_patch_chase`: ring-confined patches chase each other azimuthally around a sampled annulus, with the gap width and curvature changing per seed.
2. `2d_annulus_inner_rim_refuge_competition`: one species is pinned to the inner rim while competitors arrive from the outer wall and radial modulation, creating inward-versus-outward ring fronts.
3. `2d_boundary_forced_corner_chase`: rectangular side forcing launches three fronts from different edges, producing directional corner chases instead of isotropic spiral breakup.
4. `2d_channel_obstacle_bypass_fronts`: inlet, outlet, and obstacle-pinned species fronts split around a sampled cylinder and reconnect in its downstream shadow.
5. `2d_channel_obstacle_obstacle_shadow_cycle`: off-center patches and an obstacle refuge create alternating upper/lower bypass takeovers around the circular blocker.
6. `2d_disk_absorbing_rim_competition`: radial shell structure collapses against a weakly absorbing disk rim while interior blobs seed secondary chases.
7. `2d_disk_spiral_core_breakup`: three off-axis colonies inside a disk spawn curved spiral cores that collide with the boundary and fragment.
8. `2d_dumbbell_bottleneck_lane_swap`: left/right species layers exchange through the sampled neck, repeatedly swapping dominance across the bottleneck.
9. `2d_dumbbell_species_exchange`: independent lobe populations form in each chamber and intermittently invade through the narrow bridge.
10. `2d_l_shape_notch_quench_chase`: mixed outer/notch forcing on an L-shape drives fronts into the re-entrant corner where one species repeatedly quenches the others.
11. `2d_l_shape_reentrant_spiral_trap`: blobs seeded on the two arms spiral into the re-entrant corner, which acts as a pinning trap for one species.
12. `2d_multi_hole_plate_hole_refugia`: three named holes act as separate refugia, emitting different species that screen each other across the perforated plate.
13. `2d_multi_hole_plate_perforation_screening`: outer-boundary invasion competes with hole-seeded colonies, producing sheltered shadow corridors between holes.
14. `2d_parallelogram_spatial_rps_domains`: separated colonies on a skew periodic cell generate oblique spirals and patch trains without obvious axis locking.
15. `2d_parallelogram_wraparound_stripe_chase`: phase-shifted oblique stripes wrap around the skew cell and shear into travelling pursuit bands.
16. `2d_serpentine_channel_lane_invasion`: opposite-end species fronts negotiate sampled bends and exchange the lead from turn to turn.
17. `2d_serpentine_channel_reverse_chase`: alternating bend blobs plus wall forcing produce reverse-moving chases and bend-pinned takeovers.
18. `2d_spatial_rps_domains`: periodic noise on a sampled rectangle produces the broadest self-organized spiral mosaic in the suite.
19. `2d_y_bifurcation_branch_species_split`: species injected from trunk and daughter branches compete at the junction, then recolonize the winning arm.
20. `2d_y_bifurcation_junction_relay`: branch-seeded colonies relay dominance through the junction under weak outlet damping.
21. `3d_spatial_rps_domains`: 3D rock-paper-scissors competition from noise, forming moving domains.

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
