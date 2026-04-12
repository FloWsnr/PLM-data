# TODO

## Existing PDE / Config Inventory

Current baseline: 38 PDE presets and 548 configs.

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

The 2D Fisher-KPP suite now spans 10 domain families with 2 configs each,
covering fast and slow fronts, periodic band competition, boundary-fed
colonization, obstacle-shadowed spreading, and sampled geometry changes from
seed to seed.

1. `2d_annulus_azimuthal_front_wraparound`: elongated seed arcs wrap around a sampled annulus under rotational advection, giving travelling circumferential fronts instead of purely radial expansion.
2. `2d_annulus_inner_rim_source_band`: the inner annulus wall is held at carrying capacity while the outer wall absorbs, producing a boundary-fed outward invasion band around the hole.
3. `2d_channel_obstacle_boundary_forced_bypass`: inlet forcing drives logistic growth around a sampled cylinder, leaving a clear wake shadow and asymmetric bypass route.
4. `2d_channel_obstacle_wake_recolonization`: upstream and wake-side seeds recolonize around the obstacle under weak recirculation, creating delayed wake filling rather than one clean front.
5. `2d_disk_radial_invasion`: one or two central seeds expand through a sampled disk with no-flux walls, giving radially biased but seed-sensitive invasion shapes.
6. `2d_disk_robin_ring_detachment`: radial-cosine bands evolve inside a sampled disk with a lossy Robin rim, producing concentric ring relaxation and boundary-detached growth patterns.
7. `2d_dumbbell_bridge_invasion_front`: a left-lobe step front fills a large sampled dumbbell and pushes through the bridge into the right chamber with mild drift.
8. `2d_dumbbell_dual_lobe_exchange`: unequal dumbbell lobe seeds expand and exchange through the neck under mild wall leakage, focusing on inter-lobe transfer rather than immediate global saturation or extinction.
9. `2d_l_shape_corner_refuge_invasion`: compact corner seeds grow along both arms of a sampled L-shaped domain while the notch acts as a no-flux refuge.
10. `2d_l_shape_notch_robin_shadow`: quadrant-style initial data drifts upward in an L-shape and is selectively depleted by a lossy notch boundary, creating a re-entrant shadow.
11. `2d_logistic_invasion_front`: sampled rectangle supports a classical fast planar invasion front with randomized launch position, width, and carrying capacity.
12. `2d_multi_hole_plate_hole_shadow_channels`: mixed hole and outer-wall losses carve out perforation shadows as quadrant data relaxes across a sampled plate.
13. `2d_multi_hole_plate_perforation_invasion`: a left-to-right logistic front threads through sampled holes, with a central absorbing hole splitting and delaying the advance.
14. `2d_parallelogram_swept_quadrant_invasion`: skew-periodic quadrant data is advected obliquely across a sampled parallelogram, producing wraparound invasion rather than a static phase split.
15. `2d_parallelogram_wraparound_stripe_relaxation`: positive stripe bands coarsen and smooth on a sampled periodic skew cell, reducing obvious axis-aligned bias.
16. `2d_periodic_shear_band_competition`: near-uniform periodic bands shear, merge, and drift slowly in a rectangular torus, targeting long-lived wraparound competition.
17. `2d_serpentine_channel_guided_front`: inlet Dirichlet forcing launches a front that follows sampled serpentine bends with clear lane-guided advance.
18. `2d_serpentine_channel_reverse_lane_shadow`: outlet forcing drives a reverse-propagating front back through the sampled serpentine against wall losses, giving a clear downstream-to-upstream contrast with the inlet-guided case.
19. `2d_y_bifurcation_asymmetric_branch_competition`: unequal daughter-branch seeds compete while a weak upstream drift pulls the stronger branch back toward the junction.
20. `2d_y_bifurcation_branch_split_front`: an inlet-fed front fills a sampled Y-bifurcation and splits into the daughter arms with branch-angle-dependent timing.
21. `3d_logistic_invasion_front`: 3D logistic invasion front expanding along `x`.

#### FitzHugh-Nagumo (`configs/biology/fitzhugh_nagumo`)

The 2D suite now spans 10 domain families with 2 configs each, covering
noise-driven spiral turbulence, boundary-forced pacemakers, absorbing-wall
quenching, guided channel pulses, and geometry-mediated wave collisions.

1. `2d_annulus_azimuthal_reentry`: localized annular arc seeds wrap around the hole and generate azimuthally biased reentry with seed-dependent breakup along the ring.
2. `2d_annulus_inner_rim_pacemaker`: excited inner rim continuously emits waves into an absorbing annulus, creating a boundary-forced reentry / target-wave hybrid.
3. `2d_channel_obstacle_inlet_bypass_pulses`: inlet forcing launches excitable pulses that split around the sampled cylinder and reunite in the wake.
4. `2d_channel_obstacle_obstacle_shed_rotor`: the obstacle itself acts as a sampled pacemaker, shedding wavefronts into the channel with alternating wake asymmetry.
5. `2d_disk_absorbing_rim_rotor_collapse`: off-center disk rotor drifts into a lossy rim and collapses after a short-lived spiral transient.
6. `2d_disk_target_ring_relaxation`: radial-cosine rings relax into expanding and interfering target-like fronts inside a no-flux disk.
7. `2d_dumbbell_counter_lobe_collision`: strong pulses launched from both lobes collide in the neck and then damp asymmetrically back into the chambers.
8. `2d_dumbbell_lobe_relay_pulses`: a dominant left-lobe launch relays through the narrow bridge into a delayed right-lobe response.
9. `2d_excitable_spiral_waves`: fully periodic rectangle with randomized size and noise now targets fast spiral turbulence rather than a single nearly deterministic pattern.
10. `2d_l_shape_notch_emitter_rebound`: a notch-held excitable state repeatedly emits waves into both arms while absorbing outer walls suppress long-lived cavity filling.
11. `2d_l_shape_reentrant_spiral_trap`: reentrant-corner geometry traps a rotor seeded near the notch, giving corner-pinned spiral drift and intermittent quenching.
12. `2d_multi_hole_plate_hole_rim_emitters`: a center hole acts as a pacemaker while neighboring holes stay clamped, producing hole-shadowed emission and screening.
13. `2d_multi_hole_plate_perforation_screening`: a left-right excitable step threads through sampled perforations and is split / delayed by a clamped center hole.
14. `2d_parallelogram_excitable_spiral_waves`: skew-periodic dual-blob launch now favors wraparound spiral pairs on a sampled oblique cell.
15. `2d_parallelogram_skew_labyrinth_breakup`: oblique bands on a skew periodic cell bend, reconnect, and break into labyrinth fragments with reduced axis locking.
16. `2d_reflective_quadrant_wave_collision`: quadrant-seeded rectangle with reflective walls drives multi-front collisions and rebound rather than periodic wraparound.
17. `2d_serpentine_channel_guided_pulse_train`: inlet forcing sends pulse trains through sampled bends so fronts arrive at successive turns with visible travel-time staggering.
18. `2d_serpentine_channel_reverse_bend_recovery`: outlet-driven recovery waves run backward through the sampled serpentine and partially die on lossy walls.
19. `2d_y_bifurcation_branch_split_fronts`: inlet pacemaker splits excitable fronts into both daughter branches with branch-angle-dependent timing.
20. `2d_y_bifurcation_junction_collision`: both outlets are driven so fronts collide at the junction and then reflect upstream toward an absorbing inlet.
21. `3d_excitable_wavefronts`: 3D excitable medium seeded by noise, developing wavefronts and filament-like structures.

#### Gierer-Meinhardt (`configs/biology/gierer_meinhardt`)

1. `2d_annulus_azimuthal_spot_wraparound`: concentric annular ring seeds break into azimuthal spot necklaces that wrap around the hole with seed-dependent spacing.
2. `2d_annulus_inner_rim_pinned_bands`: asymmetric annular blob seeds interact with the inner rim to produce either smooth rim bands or spot eruptions on selected outer sectors, depending on the seed.
3. `2d_channel_obstacle_inlet_wake_spots`: inlet-biased activator growth interacts with the cylinder to leave obstacle-shadow spot lanes and wall-biased wake patches.
4. `2d_channel_obstacle_reverse_shadow_labyrinth`: a right-biased initial activator step relaxes back around the obstacle, producing reverse-side shadow bands instead of a simple inlet-driven wake.
5. `2d_disk_boundary_pinned_labyrinth`: wall leakage plus interior blob seeds create a rim-pinned labyrinth with irregular interior hot spots.
6. `2d_disk_radial_spot_necklace`: radial ring seeds in a disk collapse into a circular spot necklace around a central hot spot.
7. `2d_dumbbell_dual_lobe_phase_lag`: unequal lobe seeds evolve out of phase in the two chambers and exchange through the neck before settling into asymmetric chamber hot spots.
8. `2d_dumbbell_neck_spot_exchange`: left-right step bias drives spot transfer across the neck, emphasizing bridge-mediated exchange between the dumbbell lobes.
9. `2d_l_shape_corner_hotspot_trap`: quadrant-biased activation and a clamped notch trap hot spots near the re-entrant corner rather than filling both arms uniformly.
10. `2d_l_shape_reentrant_labyrinth`: separate left-arm, lower-arm, and corner seeds braid around the re-entrant notch into strongly seed-dependent labyrinth channels and spot clusters.
11. `2d_multi_hole_plate_hole_shadow_spots`: mixed hole boundary conditions generate screened hot spots and hole-shadow patches across the perforated plate.
12. `2d_multi_hole_plate_perforation_band_weave`: localized corridor seeds and grouped hole exchange produce perforation-guided band weaving for one seed and multi-spot screening corridors for another.
13. `2d_parallelogram_oblique_labyrinth_competition`: a skew x-periodic cell with transverse wall exchange produces oblique labyrinth bands that compete and coarsen without axis locking.
14. `2d_parallelogram_turing_spots_and_stripes`: noise-seeded Turing spots and stripes on a skew periodic parallelogram, reducing obvious rectangular bias.
15. `2d_rectangular_band_competition`: an x-periodic rectangle with Robin top/bottom walls supports long lane-like bands and slower coarsening than the fully periodic box.
16. `2d_serpentine_channel_guided_lane_breakup`: inlet-biased patterns follow the sampled serpentine bends and break into bend-to-bend guided hot spots.
17. `2d_serpentine_channel_reverse_bend_reentry`: reverse-biased structure re-enters the serpentine from the downstream side and lights up successive bends in a different order than the inlet-guided case.
18. `2d_turing_spots_and_stripes`: noise-seeded periodic-box Turing patterning producing classic spots with strong seed-to-seed layout changes.
19. `2d_y_bifurcation_branch_selection`: trunk and junction blob seeds choose different daughter branches under weak Robin branch selection, yielding seed-dependent hotspot placement near the split.
20. `2d_y_bifurcation_reverse_branch_competition`: reverse branch-biased activator fronts propagate back toward the trunk and compete at the bifurcation junction.
21. `3d_turing_patterning`: 3D activator-inhibitor Turing patterning from noise.

#### Immunotherapy (`configs/biology/immunotherapy`)

1. `2d_annulus_inner_rim_treatment_bands`: inner-rim treatment drives azimuthal tumour bands and patch collisions around an annular refuge.
2. `2d_annulus_outer_wall_relapse_rings`: tumour-heavy outer-wall relapse wraps into inward-moving annular shells and fragmented rings.
3. `2d_channel_obstacle_inlet_treatment_wake`: inlet-fed immune and IL-2 treatment sweeps past an obstacle while a tumour wake clings to the cylinder.
4. `2d_channel_obstacle_obstacle_shadow_relapse`: reverse-direction treatment and an obstacle tumour reservoir create asymmetric shadowed relapse behind the blockage.
5. `2d_disk_boundary_treatment_core_escape`: boundary treatment closes in from the circular rim while a central tumour core tries to escape or fragment.
6. `2d_disk_radial_shell_breakup`: a tumour shell seeded by radial structure breaks into rotating-looking fragments and patchy remnants on a disk.
7. `2d_dumbbell_asymmetric_bridge_clearance`: one lobe starts immune-rich and the other tumour-rich, producing a directional neck-crossing clearance wave.
8. `2d_dumbbell_tumor_immune_patch_dynamics`: dual-lobe tumour and immune patches exchange through the bottleneck and coarsen at different rates in each chamber.
9. `2d_l_shape_corner_clearance_wave`: strong notch treatment launches a fast clearance wave that bends around the re-entrant corner.
10. `2d_l_shape_notch_refuge_patterns`: weaker notch treatment leaves a tumour refuge in the notch while the outer arms organize into competing patches.
11. `2d_multi_hole_plate_hole_shadow_screening`: a treated center hole screens tumour growth into perforation shadows and corridor-separated pockets.
12. `2d_multi_hole_plate_interstitial_patch_weave`: asymmetric hole forcing drives tumour and immune patches to weave through the interstitial channels.
13. `2d_parallelogram_oblique_band_competition`: skew-periodic band initial conditions evolve into oblique tumour-treatment band competition on a sheared cell.
14. `2d_parallelogram_skew_corner_infusion`: mixed top-bottom treatment on a skew cell drives diagonal patch invasion and corner-focused clearance.
15. `2d_rectangular_wraparound_treatment_bands`: vertical treatment gradients with horizontal periodicity create wraparound tumour bands and directional suppression fronts.
16. `2d_serpentine_channel_guided_clearance_wave`: inlet treatment steers a clearance wave through serpentine bends with lane-dependent delay and breakup.
17. `2d_serpentine_channel_reverse_lane_patch_breakup`: outlet treatment and wall-supported tumour refugia produce reverse-lane relapse and patch breakup in the bends.
18. `2d_tumor_immune_patch_dynamics`: randomized rectangular tumour, immune, and cytokine patches coarsen into heterogeneous competition zones.
19. `2d_y_bifurcation_branch_selective_clearance`: inlet treatment chooses one daughter branch for faster tumour suppression while the other branch lags behind.
20. `2d_y_bifurcation_reverse_branch_relapse`: outlet-driven treatment fights a lower-branch tumour reservoir that repeatedly tries to repopulate the junction.
21. `3d_tumor_immune_patch_dynamics`: 3D tumour-immune-cytokine competition with patchy heterogeneous regions.

#### Keller-Segel (`configs/biology/keller_segel`)

These 2D configs now cover ten domain families with two dynamics per shape, all with randomized PDE parameters and randomized domain geometry where supported.

1. `2d_annulus_azimuthal_spot_wraparound`: azimuthal annulus spots wrap around the ring and separate into seed-dependent rotating trains.
2. `2d_annulus_inner_rim_pinned_bands`: inner-rim chemoattractant bias pins broken aggregation bands to the annulus core.
3. `2d_channel_obstacle_inlet_wake_spots`: inlet-biased density sheds obstacle-wake spots and wall-attached streaks behind the cylinder.
4. `2d_channel_obstacle_reverse_shadow_collapse`: reverse obstacle forcing creates a different wake-shadow collapse pattern with downstream recovery.
5. `2d_chemotactic_aggregation`: rectangle-based chemotactic aggregation from near-uniform noise into dense clusters.
6. `2d_disk_boundary_crowding`: smooth-wall crowding collapses disk-wide mass into seed-dependent spot pairs and short arcs.
7. `2d_disk_radial_spot_necklace`: radial rings destabilize into a boundary necklace of discrete spots with different final constellations per seed.
8. `2d_dumbbell_dual_lobe_phase_lag`: both lobes aggregate with a phase lag, producing asymmetric lobe-to-lobe hotspot timing.
9. `2d_dumbbell_neck_spot_exchange`: chemotactic hot spots form in one chamber and exchange through the narrow neck.
10. `2d_l_shape_corner_hotspot_trap`: aggregation is trapped near the re-entrant corner, emphasizing the notch singularity.
11. `2d_l_shape_reentrant_boundary_crowding`: boundary crowding along the L-shape walls creates notch-skewed clumps and bands.
12. `2d_multi_hole_plate_hole_shadow_spots`: holes cast chemotactic shadow zones that seed different spot constellations downstream.
13. `2d_multi_hole_plate_perforation_band_weave`: perforations split the density into banded structures that weave between holes.
14. `2d_parallelogram_chemotactic_aggregation`: chemotactic aggregation on a skew parallelogram cell, reducing obvious rectangular bias in the cluster layout.
15. `2d_parallelogram_oblique_band_competition`: oblique periodic bands compete and break on a skew cell with shifted orientations across seeds.
16. `2d_rectangular_band_competition`: rectangular periodic bands compete, merge, and realign under randomized chemotactic forcing.
17. `2d_serpentine_channel_guided_lane_breakup`: internally seeded hotspots follow the serpentine bends and break into bend-localized spot chains.
18. `2d_serpentine_channel_reverse_bend_reentry`: reverse-biased structure re-enters the serpentine from the downstream side and lights up successive bends in a different order.
19. `2d_y_bifurcation_branch_selection`: trunk and junction blob seeds choose different daughter branches under weak branch-selection forcing.
20. `2d_y_bifurcation_reverse_branch_competition`: reverse branch-biased fronts propagate back toward the trunk and compete at the bifurcation junction.
21. `3d_chemotactic_aggregation`: 3D chemotactic aggregation into clumps and filaments.

#### Klausmeier Topography (`configs/biology/klausmeier_topography`)

The 2D suite now spans 10 domain families with 2 configs each. All 2D configs
sample PDE parameters, terrain coefficients, initial conditions, and geometry
where the domain factory supports it, so seed changes alter both the pattern and
the shape being simulated.

1. `2d_annulus_inner_hill_orbiting_bands`: annular vegetation arcs orbit around an inner terrain hill while the inner and outer rims anchor water and biomass differently.
2. `2d_annulus_outer_ridge_pinching`: outer-ridge forcing pinches annular biomass rings inward, contrasting the inner-hill annulus case.
3. `2d_channel_obstacle_downhill_wake`: slope-driven vegetation bands split around a sampled channel obstacle and form a downstream wake.
4. `2d_channel_obstacle_reversed_slope_split`: reversed terrain waves drive obstacle-split bands in the opposite direction, producing a complementary wake geometry.
5. `2d_disk_annular_ridge_spot_necklace`: disk-confined annular ridges destabilize into spot necklaces with different radial phases per seed.
6. `2d_disk_radial_dome_retreat`: radial dome topography drives ring retreat and reorganization inside a sampled disk.
7. `2d_dumbbell_cross_neck_rain_shadow`: cross-slope forcing creates asymmetric rain-shadow patches across the dumbbell neck.
8. `2d_dumbbell_lobe_exchange_bands`: lobe-seeded vegetation bands exchange through the dumbbell neck instead of staying in separate chambers.
9. `2d_l_shape_corner_capture`: sloped L-shaped terrain captures band fronts near the re-entrant corner.
10. `2d_l_shape_notch_shadow_pulses`: notch-centered terrain pulses leave persistent seed-dependent biomass spots around the L-shaped cutout.
11. `2d_multi_hole_plate_hole_shadow_bands`: perforation shadows break vegetation bands around grouped sampled holes.
12. `2d_multi_hole_plate_perforation_spot_wake`: spot wakes weave between individually sampled perforations instead of forming one global band set.
13. `2d_parallelogram_cross_slope_labyrinths`: skew-periodic cross-slope terrain produces labyrinthine bands that avoid rectangular axis locking.
14. `2d_parallelogram_uphill_vegetation_bands`: uphill vegetation bands form on a skew sloped cell with oblique band alignment.
15. `2d_rectangle_diagonal_fast_bands`: faster diagonal transport produces dense periodic bands moving across the rectangular baseline domain.
16. `2d_uphill_vegetation_bands`: vegetation bands form on a slope and migrate uphill in the rectangular baseline case.
17. `2d_venturi_reverse_wall_pinning`: reverse terrain waves pin biomass near sampled venturi walls and throat shoulders.
18. `2d_venturi_throat_focus_bands`: downhill forcing focuses bands through the venturi throat before they broaden downstream.
19. `2d_y_bifurcation_branch_selection_bands`: Y-branch geometry selects one daughter branch for stronger band growth under sampled slope forcing.
20. `2d_y_bifurcation_reverse_branch_collision`: branch-seeded biomass grows back toward the bifurcation junction and remains visibly seed-dependent through the final frames.
21. `3d_uphill_vegetation_bands`: 3D sloped-terrain vegetation bands or sheets migrating uphill.

### Fluids

#### Burgers (`configs/fluids/burgers`)

The 2D Burgers suite now has 20 configs: two each for rectangle,
parallelogram, disk, annulus, dumbbell, L-shape, multi-hole plate,
channel-obstacle, venturi-channel, and Y-bifurcation domains. All 2D cases use
`144x144` grid output and randomized geometry or field parameters so seed
changes alter both the domain and the nonlinear steepening pattern where the
domain supports sampling.

1. `2d_annulus_counter_rotating_bands`: slip annulus with opposed oblique velocity bands wrapping around the hole and forming low-viscosity shock-like ring interactions.
2. `2d_annulus_inner_wall_shock_layer`: spinning inner rim against a fixed outer rim, with radial cosine layers steepening into a boundary-launched shear transient.
3. `2d_channel_obstacle_bypass_shock`: left-to-right inlet-driven channel flow where a sharp front splits around a sampled cylinder and forms asymmetric bypass wakes.
4. `2d_channel_obstacle_wake_collision`: reverse outlet forcing and downstream source bias pull wake blobs back into the cylinder shadow.
5. `2d_disk_radial_plume_collision`: clamped disk with random elongated plumes colliding against radial wave layers.
6. `2d_disk_swirling_blob_collapse`: slower diffusive disk swirl with natural outer boundary, producing a smooth solid-body-like collapse baseline.
7. `2d_dumbbell_counterlobe_collision`: opposing lobe velocities collide through the sampled neck under weak forcing near the bridge.
8. `2d_dumbbell_lobe_jet_transfer`: left-lobe packet train jets through the neck into the right lobe before the no-slip boundary damps it.
9. `2d_l_shape_corner_shock_capture`: localized wave packets hit the re-entrant corner from both arms and leave corner-focused steep fronts.
10. `2d_l_shape_notch_shear_rebound`: notch-driven boundary shear rebounds into an oblique step field while the outer frame remains open.
11. `2d_multi_hole_plate_hole_pinned_dipoles`: sampled holes pin small boundary velocities while many interior blob dipoles compete through the perforated web.
12. `2d_multi_hole_plate_perforation_wake`: mean flow and signed packets weave around three sampled holes with no-flux perforation boundaries.
13. `2d_noisy_shear_mode_interaction`: periodic rectangle with multi-mode shear steepening plus additive vector noise for persistent seed-dependent texture.
14. `2d_parallelogram_oblique_shock_lattice`: low-viscosity skew-periodic step front crossing oblique bands on a randomized parallelogram.
15. `2d_parallelogram_shear_mode_interaction`: randomized skew-cell version of the orthogonal shear interaction with extra oblique perturbation modes.
16. `2d_shear_mode_interaction`: periodic rectangle baseline with two interacting shear families and sampled box aspect / phase / perturbation modes.
17. `2d_venturi_channel_reverse_wall_peeling`: reverse outlet forcing peels signed wall-adjacent blobs back through the constriction.
18. `2d_venturi_channel_throat_focusing`: inlet-driven packets compress through a sampled throat into a narrow high-speed Burgers front.
19. `2d_y_bifurcation_branch_collision`: branch-outlet forcing drives counter-propagating waves into the Y junction for a slower collision transient.
20. `2d_y_bifurcation_inlet_split_shock`: inlet-launched wave packet splits into randomized upper and lower branches with no-slip walls.
21. `3d_shear_mode_interaction`: three sinusoidal velocity modes steepening and diffusing in a 3D periodic box.

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

#### Thermal Convection (`configs/fluids/thermal_convection`)

1. `2d_parallelogram_rayleigh_benard_rolls`: Rayleigh-Benard convection in a skew 2D cell, producing oblique rolls and plumes.
2. `2d_rayleigh_benard_rolls`: noisy stratification between hot bottom and cold top evolving into convection rolls and plumes.
3. `3d_rayleigh_benard_plumes`: 3D Rayleigh-Benard-style convection evolving into plumes and cells.

### Physics

#### Brusselator (`configs/physics/brusselator`)

Validation: the 22 2D configs below were run with `./run.sh -n 2 ... --n-runs 2` into `output/brusselator_validation`, producing 44 successful runs with passing output, solver, and stagnation diagnostics.

1. `2d_annulus_azimuthal_spot_wraparound`: annular Turing bands and spot trains wrapping around a randomized inner obstacle.
2. `2d_annulus_inner_outer_ring_competition`: strong inner/outer boundary forcing drives competing concentric rings in an annulus.
3. `2d_channel_obstacle_inlet_wake_spots`: inlet-biased channel-obstacle chemistry grows wake-aligned spots and shadowed bands.
4. `2d_channel_obstacle_obstacle_shadow_labyrinth`: obstacle and wall exchange seed a stationary wake/labyrinth contrast behind the obstacle.
5. `2d_disk_absorbing_rim_labyrinth_collapse`: disk patterns grow from seeded bumps while an absorbing rim biases collapse and edge pinning.
6. `2d_disk_radial_target_spots`: circular-domain radial IC structure selects target-like rings and spot defects.
7. `2d_dumbbell_lobe_phase_lag`: two dumbbell lobes pattern with randomized phase offsets before communicating through the neck.
8. `2d_dumbbell_neck_pinned_exchange`: neck-biased boundary exchange produces asymmetric lobe-to-lobe spot transfer.
9. `2d_l_shape_notch_quench_spots`: notch-side forcing creates corner-pinned spots and slower pattern recovery in the L-shaped cutout.
10. `2d_l_shape_reentrant_corner_labyrinth`: reentrant-corner geometry bends labyrinths and defects around the missing quadrant.
11. `2d_multi_hole_plate_hole_rim_emitters`: perforation rims seed spot emitters and screened wakes between randomized holes.
12. `2d_multi_hole_plate_perforation_screening`: many-hole screening breaks otherwise regular Turing bands into perforation-controlled patches.
13. `2d_parallelogram_oblique_band_breakup`: skew periodic geometry steers oblique bands that fragment into slower spot fields.
14. `2d_parallelogram_turing_labyrinths`: randomized parallelogram cell selecting skewed Brusselator labyrinths.
15. `2d_rectangular_edge_pinned_spots`: asymmetric rectangle boundary values pin edge spots and feed inward-growing features.
16. `2d_serpentine_channel_guided_lane_breakup`: serpentine walls guide spot lanes around bends before they break into local defects.
17. `2d_serpentine_channel_reverse_bend_recovery`: stronger bend forcing produces fast recovery bands after each channel reversal.
18. `2d_turing_labyrinths`: baseline randomized rectangle Turing patterning with broad seed and coefficient variation.
19. `2d_venturi_channel_throat_focusing`: venturi throat geometry focuses bands and spots through the constriction.
20. `2d_venturi_channel_wall_pinned_labyrinths`: wall exchange in the venturi pins bands to the upper/lower walls and throat shoulders.
21. `2d_y_bifurcation_branch_split_patterns`: Y-junction geometry splits seeded patterns into independently evolving upper/lower branches.
22. `2d_y_bifurcation_junction_collision`: branch and inlet forcing collide at the junction, producing spots and broken bands.
23. `3d_turing_patterning`: 3D Brusselator pattern formation from noise.

#### Cahn-Hilliard (`configs/physics/cahn_hilliard`)

Validation: the 22 2D configs below were run serially with `./run.sh -n 2 ... --n-runs 2` into `output/cahn_hilliard_validation`, producing 44 successful runs with passing solver/output health. The analysis contact sheets and metrics in `output/cahn_hilliard_validation/analysis` show finite concentration fields, seed-specific final states, nontrivial phase separation/coarsening in every run, and masked-grid mass drift below about 1.1%.

1. `2d_annulus_azimuthal_spinodal_wraparound`: randomized annulus geometry with azimuthal noise and periodic-like wraparound coarsening around the hole.
2. `2d_annulus_inner_rim_droplet_necklace`: inner-rim-biased droplets grow into a necklace of coarsening domains around the sampled annulus.
3. `2d_channel_obstacle_obstacle_shadow_coarsening`: sampled cylinder channel where obstacle-shadowed spinodal bands coarsen differently upstream and downstream.
4. `2d_channel_obstacle_wall_stripe_pinching`: wall-aligned stripes pinch into droplets and bands around the channel obstacle under zero-flux boundaries.
5. `2d_disk_offcenter_droplet_nucleation`: off-center Gaussian droplets nucleate curved phase regions inside a randomized disk.
6. `2d_disk_radial_ring_coarsening`: radial-cosine rings in a disk break and coarsen into circular target-like phase bands.
7. `2d_dumbbell_dual_lobe_exchange`: two randomized lobes phase-separate with delayed interaction through the dumbbell neck.
8. `2d_dumbbell_neck_pinch_coarsening`: neck-localized structure pinches and seeds different coarsening rates in the two lobes.
9. `2d_l_shape_notch_pinned_coarsening`: notch-biased droplets and homogeneous no-flux walls pin domains near the reentrant corner.
10. `2d_l_shape_reentrant_stripe_breakup`: L-shaped stripes bend around the missing quadrant and fragment into corner-controlled bands.
11. `2d_multi_hole_plate_hole_rim_droplets`: perforation-rim droplets grow into hole-screened coarsening halos across sampled plates.
12. `2d_multi_hole_plate_perforation_screening`: noisy spinodal texture is screened by randomized hole layouts, yielding patchy interstitial bands.
13. `2d_parallelogram_oblique_band_breakup`: skew-cell oblique bands break into slanted domains instead of axis-aligned square patterns.
14. `2d_parallelogram_spinodal_decomposition`: randomized skew parallelogram spinodal decomposition with periodic boundaries and sampled kinetics.
15. `2d_rectangle_x_periodic_interface_coarsening`: x-periodic rectangular interface coarsening that produces directional stripe competition.
16. `2d_serpentine_channel_guided_band_coarsening`: serpentine walls guide lane-following bands around bends before coarsening.
17. `2d_serpentine_channel_lane_pinch_droplets`: broader sampled droplets pinch into lane-confined phase islands through the serpentine duct.
18. `2d_spinodal_decomposition`: baseline randomized rectangle spinodal decomposition into labyrinthine phase-separated domains.
19. `2d_venturi_channel_throat_focused_spinodal`: venturi geometry focuses spinodal bands through the throat and re-expands them downstream.
20. `2d_venturi_channel_wall_band_competition`: upper/lower wall bands compete through a sampled constriction and leave throat-pinned interfaces.
21. `2d_y_bifurcation_branch_split_spinodal`: branch geometry splits spinodal bands into the two daughter arms with seed-dependent asymmetry.
22. `2d_y_bifurcation_junction_droplet_collision`: trunk and branch droplets collide at the Y junction before coarsening along each arm.
23. `3d_spinodal_decomposition`: 3D spinodal decomposition and coarsening.

#### Complex Ginzburg-Landau (`configs/physics/cgl`)

Validation: the 20 2D configs below were run with
`./run.sh -n 2 ... --n-runs 2` into `output/cgl_validation`, producing 40
successful runs with passing solver, finite-value, dynamic-range, and
stagnation diagnostics after retuning the initially flat serpentine and
Y-bifurcation collision cases.

1. `2d_annulus_inner_outer_phase_drive`: inner and outer annulus rims impose different complex phases, driving ring-front competition through a sampled annular gap.
2. `2d_annulus_ring_phase_slips`: Neumann annulus seeded with radial rings and oblique phase waves, producing ring-wrapped phase slips around the hole.
3. `2d_benjamin_feir_turbulence`: randomized periodic rectangle in a Benjamin-Feir-unstable regime, evolving noisy amplitude into sustained defect turbulence.
4. `2d_channel_obstacle_boundary_drive`: channel-obstacle geometry with inlet/outlet phase drive and natural walls, creating obstacle-scattered boundary layers.
5. `2d_channel_obstacle_shadow_turbulence`: obstacle-pinned channel turbulence where the circular cutout damps the field and casts a phase shadow.
6. `2d_disk_absorbing_blob_collapse`: absorbing circular rim collapses and reshapes sampled complex Gaussian blobs without becoming static.
7. `2d_disk_radial_defect_glass`: Neumann disk seeded with radial amplitude rings plus random phase defects, yielding circular defect-glass dynamics.
8. `2d_dumbbell_absorbing_neck_defects`: absorbing dumbbell boundary turns stripe and quadrant phase starts into neck-focused defects.
9. `2d_dumbbell_lobe_exchange`: Neumann dumbbell with asymmetric lobe blobs that exchange activity through the sampled neck.
10. `2d_l_shape_notch_quench`: L-shaped domain where the reentrant notch is absorbing and oblique phase sheets quench into corner-bent fronts.
11. `2d_l_shape_reentrant_defect_cascade`: Neumann L-shape seeded near the reentrant corner, creating geometry-bent defect cascades.
12. `2d_multi_hole_plate_pinned_defects`: sampled multi-hole plate with hole rims pinned to zero, forcing interstitial defects around perforations.
13. `2d_multi_hole_plate_screened_phase_maze`: outer boundary phase drive with natural hole rims, producing a screened maze of phase fronts through the plate.
14. `2d_parallelogram_benjamin_feir_turbulence`: skew periodic Benjamin-Feir turbulence with sampled cell vectors and mixed noisy/oblique initial phase.
15. `2d_parallelogram_oblique_phase_glass`: skew periodic cell with quadrant phase jumps, generating oblique glassy phase domains.
16. `2d_rectangle_absorbing_phase_sheets`: x-periodic, y-absorbing rectangle where oblique sheets shear into damped phase bands.
17. `2d_serpentine_channel_lane_phase_drive`: serpentine channel with opposing inlet/outlet phases that drive lane-guided fronts around bends.
18. `2d_serpentine_channel_neumann_pulse_train`: Neumann serpentine with stripe-plus-noise initial data retuned into sustained BF-style lane turbulence.
19. `2d_y_bifurcation_branch_collision`: sampled Y branch with opposing outlet phases, producing branch-to-junction phase collision without late-time flattening.
20. `2d_y_bifurcation_split_phase_drive`: inlet-driven Y branch with split outlet phases on a tightened mesh, emphasizing asymmetric branch filling.
21. `3d_benjamin_feir_turbulence`: 3D complex amplitude turbulence from a noisy background.

#### Gray-Scott (`configs/physics/gray_scott`)

The 2D Gray-Scott preset now has 20 randomized configs, with two configs each
for `rectangle`, `annulus`, `disk`, `dumbbell`, `parallelogram`,
`l_shape`, `multi_hole_plate`, `channel_obstacle`, `y_bifurcation`, and
`serpentine_channel`. Validation outputs are under
`output/gray_scott_validation`; the final two-seed sweep passed 40/40 runs,
with output resolutions at or above 128 on the shortest side.

1. `2d_annular_spot_stripe_patterns`: high-feed annulus patches form spot and stripe colonies around the hole.
2. `2d_annulus_inner_outer_wave_collision`: nonnegative radial waves collide through an annular gap without boundary overdrive.
3. `2d_channel_obstacle_obstacle_spot_shedding`: inlet flow and obstacle pinning shed spot packets into the wake.
4. `2d_channel_obstacle_wake_labyrinths`: advected labyrinth fronts bend around a sampled obstacle with natural obstacle walls.
5. `2d_disk_absorbing_rim_gliders`: disk gliders drift under weak rotation with a sampled circular boundary.
6. `2d_disk_pulsating_spot_rings`: circular spot rings pulse and reorganize from radial seeds.
7. `2d_drifting_spot_stripe_patterns`: periodic rectangle gliders drift under weak substrate advection.
8. `2d_dumbbell_neck_glider_exchange`: low-feed gliders exchange through a sampled dumbbell neck.
9. `2d_dumbbell_spot_stripe_patterns`: broad dumbbell spot and stripe fields evolve in both lobes with Neumann outer walls.
10. `2d_l_shape_notch_worm_cascade`: high-feed worm cascades interact with a driven reentrant notch.
11. `2d_l_shape_reentrant_hole_shadow`: L-shape fronts cast a corner shadow from reentrant geometry.
12. `2d_multi_hole_plate_perforation_holes`: perforation rims pin substrate/autocatalyst values and split spot paths.
13. `2d_multi_hole_plate_screened_gliders`: screened gliders weave through a randomized multi-hole plate.
14. `2d_noisy_spot_stripe_patterns`: stochastic rectangle roughening keeps spot and stripe morphology varied across seeds.
15. `2d_parallelogram_labyrinth_wraparound`: skew periodic boundaries wrap labyrinths through an oblique cell.
16. `2d_parallelogram_oblique_worms`: advected oblique blobs generate skew worm motion.
17. `2d_serpentine_channel_guided_worms`: broad activator bands travel around sampled serpentine bends.
18. `2d_serpentine_channel_reverse_wave_train`: inlet/outlet forcing sends wave trains through the serpentine in the opposite regime.
19. `2d_y_bifurcation_branch_wave_split`: inlet-driven fronts split into two Y branches with bounded wall pinning.
20. `2d_y_bifurcation_junction_glider_collision`: opposing branch activator bands collide near the Y junction.
21. `3d_spot_blob_patterns`: 3D Gray-Scott spot and blob structures from two seeded reagent patches.

#### Kuramoto-Sivashinsky (`configs/physics/kuramoto_sivashinsky`)

Validation note: all 20 2D configs were run with `./run.sh -n 2 ... --n-runs 2` under `output/ks_validation`, producing 40 successful runs with `128x128` outputs, passing health diagnostics, clean logs, and two configs each for rectangle, parallelogram, disk, annulus, dumbbell, L-shape, multi-hole plate, channel obstacle, venturi channel, and Y-bifurcation domains. Absorbing/quench variants were retuned after the first validation pass so they remain slower and more damped than the open/Neumann cases without becoming visually flat.

1. `2d_annulus_absorbing_inner_outer_competition`: inner/outer absorbing annulus competition with sampled ring spacing; one seed damps into smooth rings while another retains stronger radial modulation.
2. `2d_annulus_azimuthal_band_churn`: Neumann annulus seeded with azimuthal/radial bands, producing sustained ring-wrapped churn.
3. `2d_channel_obstacle_absorbing_cylinder_shadow`: channel-obstacle shadowing with absorbing walls and cylinder damping, keeping a lower-amplitude wake variant.
4. `2d_channel_obstacle_wake_wrinkle_shedding`: stronger obstacle-driven wake wrinkling and downstream shedding through a sampled channel.
5. `2d_disk_absorbing_blob_collapse`: disk blob collapse against an absorbing rim, retuned to preserve a moving remnant instead of flattening.
6. `2d_disk_neumann_radial_cell_churn`: reflective disk with radial cells and rotating/advection-biased churn.
7. `2d_dumbbell_absorbing_neck_quench`: absorbing dumbbell neck transfer and quench, useful as a slow low-amplitude geometry-constrained case.
8. `2d_dumbbell_lobe_exchange_chaos`: asymmetric lobe-to-lobe chaotic exchange through the dumbbell throat.
9. `2d_l_shape_absorbing_notch_fronts`: absorbing L-shape fronts bent by the reentrant notch, retuned for seed-dependent weak/strong outcomes.
10. `2d_l_shape_reentrant_corner_churn`: Neumann L-shape with corner-driven folding, wrinkles, and cell turnover.
11. `2d_multi_hole_plate_absorbing_hole_shadows`: absorbing perforated-plate shadows around sampled holes, now retaining sparse active patches.
12. `2d_multi_hole_plate_perforation_screened_chaos`: stronger screened chaos through sampled perforations and interstitial channels.
13. `2d_parallelogram_slow_large_cell_wrinkles`: slow broad oblique wrinkles on a skew sampled cell.
14. `2d_parallelogram_spatiotemporal_chaos`: skew-cell chaos retuned with sampled Gaussian blobs so the two seeds separate clearly.
15. `2d_rectangle_x_periodic_wall_shear_fronts`: x-periodic rectangle with wall shear fronts and oblique drift.
16. `2d_spatiotemporal_chaos`: periodic rectangle noise growth into classical cellular spatiotemporal chaos.
17. `2d_venturi_channel_throat_accelerated_wrinkles`: high-energy venturi throat focusing with accelerated wrinkle bands.
18. `2d_venturi_channel_wall_quench_bands`: slower venturi wall-quench bands with strong seed variability after retuning.
19. `2d_y_bifurcation_branch_collision_chaos`: branch collision and junction-focused activity in a Y-domain, retuned to avoid late-time collapse.
20. `2d_y_bifurcation_split_wrinkle_plumes`: stronger asymmetric plume splitting through the Y branches.
21. `3d_spatiotemporal_chaos`: 3D spatiotemporal chaos from small random perturbations.

#### Lorenz (`configs/physics/lorenz`)

The 2D Lorenz suite now has 20 randomized configs, with two configs each for
`rectangle`, `annulus`, `disk`, `dumbbell`, `parallelogram`, `l_shape`,
`multi_hole_plate`, `channel_obstacle`, `y_bifurcation`, and
`serpentine_channel`. Validation outputs are under
`output/lorenz_2d_validation`; the two-seed sweep passed 40/40 runs on two MPI
ranks, with output resolutions at or above 128 on the shortest side. The
analysis artifacts in `output/lorenz_2d_validation/_analysis` include metric
summaries and visual start/middle/end plus seed-comparison sheets.

1. `2d_annulus_inner_outer_lobe_collision`: sampled annulus with opposing inner/outer Lorenz lobe Dirichlet states, creating radially colliding synchronized rings.
2. `2d_annulus_ring_phase_slips`: weakly coupled annulus with radial-cosine and oblique sine starts, producing ring phase slips around the inner hole.
3. `2d_channel_obstacle_obstacle_pinned_wake`: sampled channel-obstacle domain with the cylinder pinned to a Lorenz lobe and interior blobs wrapping into obstacle shadows.
4. `2d_channel_obstacle_wall_driven_lobe_collision`: inlet/outlet lobe forcing drives opposing sheets around the sampled obstacle under wall damping.
5. `2d_disk_offcenter_patch_burst`: weakly coupled disk where off-center signed Gaussian patches break into fragmented local Lorenz bursts.
6. `2d_disk_rim_pinned_synchronization`: stronger disk coupling with a pinned rim, smoothing interior waves into large-scale synchronized lobes.
7. `2d_dumbbell_lobe_exchange_chaos`: unequal signed lobes exchange through a sampled dumbbell neck while weak coupling preserves chamber asymmetry.
8. `2d_dumbbell_neck_synchronization_pulse`: narrow-neck dumbbell with stronger coupling and Robin damping, emphasizing bottleneck-mediated synchronization.
9. `2d_l_shape_corner_burst`: L-shaped cavity with the reentrant notch pinned to a Lorenz lobe, launching corner-biased chaotic bursts.
10. `2d_l_shape_notch_quenched_sheets`: notch-quenching L-shape where oblique sheets and quadrant `z` data are pulled toward a lossy inside corner.
11. `2d_multi_hole_plate_hole_shadow_chaos`: perforated plate with sampled hole layout and hole boundary forcing, producing screened chaotic shadows around holes.
12. `2d_multi_hole_plate_interstitial_lobe_exchange`: stronger plate coupling and left/right lobe starts drive exchange through interstitial corridors between sampled holes.
13. `2d_parallelogram_oblique_band_chaos`: skew periodic cell with oblique sine bands and moderate coupling, giving wraparound slanted chaotic bands.
14. `2d_parallelogram_synchronized_longwave_rolls`: strong-coupling skew periodic cell where low-frequency rolls synchronize into smooth longwave oscillations.
15. `2d_rectangle_absorbing_lobe_fronts`: rectangular lobe fronts collide between opposing Dirichlet sides with Robin damping on the transverse walls.
16. `2d_rectangle_fragmented_weak_coupling`: periodic rectangle with weak diffusion and random initial fields, retaining fragmented local chaotic patches.
17. `2d_serpentine_channel_guided_phase_bands`: sampled serpentine channel with no-flux walls, where oblique phase bands follow the bend sequence.
18. `2d_serpentine_channel_reverse_quench_pulses`: inlet/outlet lobe reversal in a sampled serpentine path creates bend-guided quench pulses.
19. `2d_y_bifurcation_branch_collision_chaos`: sampled Y bifurcation with opposing outlet lobe states that collide back toward the junction.
20. `2d_y_bifurcation_branch_split_chaos`: inlet-forced Y bifurcation where localized chaotic patches split unevenly into the two branches.
21. `3d_spatial_lorenz_chaos`: 3D spatially extended Lorenz chaos seeded from random initial fields.

#### Maxwell Pulse (`configs/physics/maxwell_pulse`)

Validation note: all 20 2D configs were run with
`./run.sh -n 2 ... --n-runs 2` under `output/maxwell_pulse_validation`,
producing 40 successful runs with output grids at or above 128 on the shortest
side. The sweep covers two configs each for rectangle, parallelogram, disk,
annulus, dumbbell, L-shape, multi-hole plate, channel obstacle, venturi channel,
and Y-bifurcation domains. Metadata health checks, clean log scans, masked
finite/dynamic metrics, seed-pair difference metrics, and start/middle/end
visual comparison sheets all passed; no retuning was needed after the
validation pass.

1. `2d_annulus_inner_outer_echo`: annular pulse with PEC inner rim and absorbing outer rim, emphasizing radial echoes and inner-wall scattering.
2. `2d_annulus_orbiting_dual_pulse`: slower annulus case with opposing tangential wave packets and inner absorption against an outer PEC wall.
3. `2d_channel_obstacle_reverse_wall_echo`: downstream-launched channel-obstacle pulse with obstacle absorption and PEC inlet/walls for reverse echoes.
4. `2d_channel_obstacle_wake_diffraction`: inlet pulse diffracts around a sampled circular obstacle with absorbing channel ends.
5. `2d_disk_pec_whispering_gallery`: off-center disk pulse in a PEC circular cavity, producing rim-guided reflections and standing interference.
6. `2d_disk_radial_pulse_scatter`: faster absorbing-disk case with radial-cosine forcing and seeded off-center field texture.
7. `2d_dumbbell_lobe_transfer_pulse`: absorbing dumbbell pulse launched from one lobe through a sampled neck into the opposite lobe.
8. `2d_dumbbell_neck_trapped_echoes`: narrow-neck PEC dumbbell with neck-centered forcing, emphasizing trapped and delayed lobe echoes.
9. `2d_guided_em_pulse`: randomized rectangular waveguide with absorbing ends, PEC walls, and localized source/IC variation.
10. `2d_l_shape_corner_diffraction_pulse`: L-shaped domain with a corner-launched pulse diffracting around a PEC notch into absorbing outer walls.
11. `2d_l_shape_notch_reflection_cavity`: slower L-shape cavity with PEC outer boundary and absorbing notch sink for reentrant-corner reflections.
12. `2d_multi_hole_plate_hole_trapped_echoes`: perforated plate with PEC outer boundary and absorbing holes, giving hole-localized echo patterns.
13. `2d_multi_hole_plate_perforation_diffraction`: absorbing outer plate with PEC holes, producing diffraction wakes around sampled perforations.
14. `2d_parallelogram_guided_em_pulse`: skew waveguide pulse with randomized oblique geometry, absorbing sides, and seeded wave packets.
15. `2d_parallelogram_pec_oblique_mode`: skew PEC/absorbing hybrid with oblique sine modes and radial forcing.
16. `2d_rectangle_pec_cavity_sine_bounce`: slower rectangular PEC cavity with sampled sine forcing and quadrant starts for standing-wave bounce patterns.
17. `2d_venturi_channel_reverse_focus_echo`: downstream venturi pulse reflected through a sampled throat with absorbing walls/outlet.
18. `2d_venturi_channel_throat_lensing`: inlet-launched fast venturi pulse focused by the sampled constriction under PEC walls.
19. `2d_y_bifurcation_branch_collision_echo`: branch-launched Y-bifurcation pulse collision back toward the junction with PEC walls.
20. `2d_y_bifurcation_inlet_split_pulse`: inlet pulse splitting into sampled Y branches with absorbing outlets and PEC walls.
21. `3d_guided_em_pulse`: 3D localized electromagnetic pulse launched in a partially confined box.

#### Schnakenberg (`configs/physics/schnakenberg`)

Validation note: all 20 2D configs were run with `./run.sh -n 2 ... --n-runs 2` under `output/schnakenberg_validation`, producing 40 successful runs with `>=128x128` outputs, solver convergence, passing health diagnostics, clean logs, and two configs each for rectangle, parallelogram, disk, annulus, dumbbell, L-shape, multi-hole plate, channel obstacle, venturi channel, and Y-bifurcation domains. `2d_annulus_inner_outer_ring_competition` and `2d_channel_obstacle_obstacle_shadow_labyrinth` were retuned after the first analysis pass to improve motion and seed-to-seed separation.

1. `2d_annulus_azimuthal_spot_necklace`: annular Turing spots seeded with angular modulation, producing necklace-like spot chains and ring defects around the inner hole.
2. `2d_annulus_inner_outer_ring_competition`: inner/outer Dirichlet competition in an annulus with sampled asymmetric blob and wave seeds, now breaking smooth rings into azimuthal patches.
3. `2d_channel_obstacle_inlet_wake_spots`: inlet/outlet-driven channel-obstacle case where sampled upstream blobs turn into downstream wake spots around the cylinder.
4. `2d_channel_obstacle_obstacle_shadow_labyrinth`: obstacle-shadow labyrinth variant with stronger sampled geometry and boundary contrast, producing seed-dependent wall bands and obstacle wake structure.
5. `2d_disk_edge_pinned_labyrinths`: disk with edge-pinned activator/inhibitor contrast, bending labyrinths and spots against the circular outer wall.
6. `2d_disk_neumann_radial_targets`: reflective disk seeded with radial targets, giving cleaner target rings and spot breakup without imposed edge values.
7. `2d_dumbbell_absorbing_bridge_quench`: dumbbell with bridge quenching and absorbing lobe contrast, producing slower lobe-separated activity and neck filtering.
8. `2d_dumbbell_lobe_phase_lag`: two-lobe dumbbell with phase-shifted seeds, creating delayed pattern competition across the narrow bridge.
9. `2d_l_shape_notch_quench_spots`: L-shape notch quench variant where the reentrant notch damps one arm while sampled blobs survive elsewhere.
10. `2d_l_shape_reentrant_corner_labyrinth`: reentrant-corner L-shape labyrinth growth, with the inside corner bending stripes and seeding corner-localized defects.
11. `2d_multi_hole_plate_hole_rim_emitters`: perforated plate with hole-rim forcing, producing bright spots and halos around sampled holes.
12. `2d_multi_hole_plate_perforation_screening`: multi-hole screening case where holes block and fragment broader Turing patterns into interstitial patches.
13. `2d_parallelogram_oblique_wall_bands`: skew parallelogram with oblique wall-biased bands, separating it from the fully periodic rectangle case.
14. `2d_parallelogram_turing_spots_and_labyrinths`: periodic skew-cell Turing patterns with randomized kinetics, ICs, and geometry, emphasizing oblique spot/labyrinth drift.
15. `2d_rectangle_edge_pinned_spots`: rectangle with edge-pinned boundary contrast, producing wall-nucleated spots instead of fully periodic texture.
16. `2d_turing_spots_and_labyrinths`: periodic rectangle baseline with randomized kinetics and ICs, giving dense classical spot/labyrinth Turing fields.
17. `2d_venturi_channel_throat_focusing`: venturi channel where sampled throat geometry focuses pattern growth into the constriction and downstream expansion.
18. `2d_venturi_channel_wall_pinned_bands`: slower venturi wall-pinned band case with boundary-driven activation near the tapered walls.
19. `2d_y_bifurcation_branch_selection`: Y-bifurcation split case where sampled branch geometry and boundary values select different daughter-branch activity.
20. `2d_y_bifurcation_junction_collision`: Y-bifurcation collision case where branch and inlet patterns meet near the junction and form localized defects.
21. `3d_turing_patterning`: 3D Schnakenberg pattern formation from noise.

#### Superlattice (`configs/physics/superlattice`)

The 2D sweep now covers 10 domains with 2 configs each and uses sampled
geometry, coefficients, initial conditions, and boundary values so seed changes
alter both the morphology and the timing. All 2D configs output at least a
128-scale grid and were validated with `./run.sh -n 2 ... --n-runs 2`; the
40 seed runs passed metadata health checks, log scans, masked finite/dynamic
metrics, and visual start/middle/end GIF inspection.

1. `2d_rectangle_dynamic_noise_lattice`: fast noisy lattice formation on a rectangular box with strong seed-to-seed variation.
2. `2d_rectangle_edge_pinned_spot_invasion`: edge-pinned spot invasion on a rectangle with x-side Dirichlet bands and zero-flux y walls.
3. `2d_annulus_azimuthal_necklace`: reliable azimuthal necklace spots wrapping around a sampled annulus.
4. `2d_annulus_inner_outer_ring_competition`: slower inner/outer ring competition on an annulus with radial contrast.
5. `2d_disk_radial_target_necklace`: radial target-like waves and necklace spots on a disk.
6. `2d_disk_absorbing_rim_breakup`: absorbing-rim breakup on a disk, driving faster edge-driven fragmentation.
7. `2d_dumbbell_lobe_phase_lag`: phase-lagged lobe exchange across a dumbbell neck.
8. `2d_dumbbell_neck_pinned_exchange`: neck-pinned exchange in a dumbbell with zero-flux outer walls and delayed cross-lobe coupling.
9. `2d_l_shape_reentrant_labyrinth`: re-entrant corner labyrinth formation in an L-shaped domain.
10. `2d_l_shape_notch_quench_spots`: notch-quench spots in an L-shaped domain with a Dirichlet notch sink.
11. `2d_multi_hole_plate_perforation_screening`: zero-flux perforation screening across a sampled multi-hole plate.
12. `2d_multi_hole_plate_hole_rim_emitters`: hole-rim emitters on a perforated plate with alternating hot/cold Dirichlet rims.
13. `2d_parallelogram_oblique_stripe_interference`: oblique stripe interference on a skew periodic cell.
14. `2d_parallelogram_skew_wall_quench`: skew-wall quench on a parallelogram, with periodic x and Dirichlet y bands.
15. `2d_channel_obstacle_obstacle_shadow_lattice`: obstacle-shadow lattice behind a channel cylinder.
16. `2d_channel_obstacle_inlet_wake_spots`: inlet-biased wake spots around a channel obstacle with zero-flux walls.
17. `2d_venturi_channel_throat_focusing`: venturi throat focusing, where the constriction sharpens the fast/slow scales.
18. `2d_venturi_channel_wall_pinned_bands`: wall-pinned bands in a venturi channel with Dirichlet wall contrast.
19. `2d_y_bifurcation_branch_split_lattice`: branch-split lattice patterns through a Y junction with branch-value Dirichlet ends.
20. `2d_y_bifurcation_junction_collision`: junction-collision dynamics in a Y bifurcation with outlet-driven contrast.

`3d_superlattice_pattern_formation` remains the 3D extension, but the new 2D
set is the current validation target for dataset expansion.

#### Swift-Hohenberg (`configs/physics/swift_hohenberg`)

Validation note: all 21 2D configs were run with
`./run.sh -n 2 ... --n-runs 2` under `output/swift_hohenberg_validation`,
producing 42 successful runs with `>=128x128` output grids, solver
convergence, passing health diagnostics, clean logs, and two configs each for
rectangle, parallelogram, disk, annulus, dumbbell, L-shape, multi-hole plate,
channel obstacle, venturi channel, and Y-bifurcation domains. The first pass
identified five over-damped cases; `2d_channel_obstacle_reverse_shadow_defects`,
`2d_disk_roll_patterns`, `2d_dumbbell_neck_defect_filter`,
`2d_multi_hole_plate_screened_rolls`, and
`2d_parallelogram_cellular_defect_drift` were retuned and rerun to keep visible
final-frame dynamics and seed-to-seed separation.

1. `2d_advected_roll_growth`: periodic rectangle with sampled localized blobs, positive near-critical growth, and oblique advection producing drifting roll/ring growth.
2. `2d_annulus_azimuthal_roll_orbit`: simply supported annulus with sampled inner/outer radii and rotational velocity, producing orbiting roll islands around the hole.
3. `2d_annulus_inner_outer_defect_competition`: wider annulus with radial target seeding and weak strain, producing inner/outer ring competition and defect locking.
4. `2d_channel_obstacle_reverse_shadow_defects`: reverse-flow channel-obstacle case with oblique sine seeds, retuned to keep downstream/obstacle-shadow defects active.
5. `2d_channel_obstacle_wake_roll_shedding`: forward-flow obstacle wake where sampled upstream blobs grow into obstacle-pinned roll patches.
6. `2d_directed_roll_growth`: rectangular periodic domain with stronger diagonal advection and oblique modal seeds for fast directional roll drift.
7. `2d_disk_radial_target_pinning`: disk with radial-cosine target seeding and slow rotation, producing concentric targets that pin to the circular boundary.
8. `2d_disk_roll_patterns`: disk noise-seeded roll formation, retuned to maintain late-time circular roll texture rather than decaying to the wall.
9. `2d_dumbbell_lobe_roll_exchange`: dumbbell with opposing lobe blob populations, producing lobe-specific growth and transfer through the neck.
10. `2d_dumbbell_neck_defect_filter`: dumbbell step-seeded lobe imbalance, retuned so the narrow neck filters persistent roll defects instead of flattening.
11. `2d_l_shape_notch_defect_cascade`: L-shape with oblique modal seeds and rotational shear, producing notch-triggered defect cascades.
12. `2d_l_shape_reentrant_roll_quench`: L-shape quadrant seed where the reentrant corner bends and localizes roll growth.
13. `2d_multi_hole_plate_hole_pinned_hexagons`: perforated plate with sampled holes and blob seeds, producing hole-pinned hexagon/spot structure.
14. `2d_multi_hole_plate_screened_rolls`: perforated plate with noise seeds, retuned to preserve screened inter-hole roll remnants and seed variation.
15. `2d_parallelogram_cellular_defect_drift`: periodic skew cell with sampled cellular velocity, retuned for persistent drifting defects.
16. `2d_parallelogram_oblique_roll_shear`: periodic parallelogram with oblique shear and modal seeds, producing skew-aligned roll bands.
17. `2d_rotating_roll_patterns`: simply supported rectangle with rotational advection and noise seeding, producing swirl-biased roll growth.
18. `2d_venturi_channel_throat_roll_focusing`: venturi channel where sampled throat geometry focuses growing rolls through the constriction.
19. `2d_venturi_channel_wall_pinned_labyrinth`: venturi channel with slower near-threshold wall-pinned labyrinth growth.
20. `2d_y_bifurcation_branch_roll_selection`: Y-bifurcation with inlet and upper-branch seeds, producing branch-selective roll growth.
21. `2d_y_bifurcation_junction_defect_collision`: Y-bifurcation with opposite-signed branch seeds, producing junction collisions and branch defects.
22. `3d_advected_pattern_growth`: localized seed growing into a 3D patterned structure under weak uniform advection.

#### Van der Pol (`configs/physics/van_der_pol`)

1. `2d_annulus_counterphase_ringdown`: annulus with inner and outer rims pinned in counterphase, producing ring phase slips and radial relaxation waves.
2. `2d_annulus_inner_outer_phase_slips`: annulus with driven inner rim and weakly absorbing outer rim, emphasizing target-like waves that shear around the hole.
3. `2d_channel_obstacle_obstacle_pinned_targets`: channel-obstacle case with a softened obstacle pin and obstacle-centered radial seed, producing target waves around the circular inclusion without velocity saturation.
4. `2d_channel_obstacle_wake_locking`: inlet-driven obstacle channel where upstream blobs and obstacle damping create wake-locked relaxation waves.
5. `2d_disk_relaxation_targets`: disk radial-cosine seed with weak rim absorption, producing clean circular target waves.
6. `2d_disk_rim_pinned_bursts`: disk with randomized rim pinning and blob/noise seeds, producing sharper burst fronts and rim-mediated breakup.
7. `2d_dumbbell_lobe_phase_lag`: two-lobe dumbbell with opposite-signed lobe seeds, creating phase-lagged oscillations across the bridge.
8. `2d_dumbbell_neck_synchronization_pulse`: narrow-neck dumbbell with a neck-centered pulse, producing delayed synchronization and strong bridge-front dynamics.
9. `2d_l_shape_corner_wave_reflection`: L-shape with notch damping and blob/sine seeds, bending wave fronts around the reentrant corner.
10. `2d_l_shape_notch_pinned_relaxation`: L-shape with a pinned notch and quadrant seed, producing corner-localized relaxation fronts.
11. `2d_oscillatory_wave_relaxation`: randomized periodic rectangle baseline with smooth low-`mu` oscillatory wave growth from modal perturbations.
12. `2d_parallelogram_oblique_phase_sheets`: skew cell with x-periodicity and y-side damping, producing oblique relaxation sheets from quadrant ICs.
13. `2d_parallelogram_oscillatory_wave_relaxation`: fully periodic skew parallelogram with oblique modal seeds and moderate relaxation dynamics.
14. `2d_rectangle_absorbing_lobe_fronts`: rectangle with opposing Dirichlet side pins and Robin top/bottom damping, producing absorbing lobe-front dynamics.
15. `2d_side_cavity_channel_cavity_phase_trap`: side-cavity channel with cavity-biased blobs, testing whether oscillations trap and release from the pocket.
16. `2d_side_cavity_channel_main_lane_sweep`: side-cavity channel with opposing lane pins, producing main-channel sweep fronts with cavity leakage.
17. `2d_venturi_channel_throat_burst_train`: Venturi channel with inlet drive and throat-biased blobs, producing constriction-focused burst trains.
18. `2d_venturi_channel_wall_quench_pulses`: Venturi channel with wall Robin quenching and oblique waves, producing wall-damped throat pulses.
19. `2d_y_bifurcation_branch_selection`: Y-bifurcation with asymmetric outlet conditions, producing branch-selected relaxation waves.
20. `2d_y_bifurcation_junction_collision`: Y-bifurcation with opposite branch seeds and outlet pins, producing collision fronts at the junction.
21. `3d_oscillatory_wave_relaxation`: 3D Van der Pol oscillatory wave patterns from random modal perturbations.

#### Zakharov-Kuznetsov (`configs/physics/zakharov_kuznetsov`)

The 2D ZK set now emphasizes dispersive nonlinear pulse motion across ten
domain families, with two randomized configs per domain shape. Periodic
rectangle/parallelogram cells keep the conservative periodic weak form; Gmsh
domains use homogeneous wall conditions to expose wall reflection, trapping,
splitting, and obstacle scattering.

1. `2d_soliton_pulse_propagation`: fast localized periodic rectangle pulse with randomized trailing satellites.
2. `2d_rectangle_wall_reflected_bands`: slower x-periodic rectangle bands clamped on the y-walls for wall-reflected stripe motion.
3. `2d_parallelogram_soliton_pulse_propagation`: fast oblique soliton-like pulse on a skew fully periodic cell.
4. `2d_parallelogram_oblique_wave_packet`: skew-cell Gaussian wave packet with x-periodic and y-Neumann wall behavior.
5. `2d_disk_radial_pulse_reflection`: Neumann disk radial cosine pulse that breaks into reflected dispersive rings.
6. `2d_disk_absorbing_edge_pulses`: Dirichlet disk edge case with several randomized elongated blobs decaying at the rim.
7. `2d_annulus_azimuthal_pulse_wrap`: annulus twin pulses wrapping around an inner obstacle under Neumann walls.
8. `2d_annulus_inner_wall_echo`: annulus mixed inner-Dirichlet / outer-Neumann band dynamics.
9. `2d_dumbbell_lobe_exchange`: Neumann dumbbell pulse transfer from one lobe through the neck.
10. `2d_dumbbell_neck_collision`: Dirichlet dumbbell neck collision between opposite-signed packets.
11. `2d_l_shape_corner_scatter`: L-shape corner scattering with a clamped notch and reflective outer boundary.
12. `2d_l_shape_notch_wave_trap`: L-shape wave packet launched into the re-entrant notch with clamped outer walls.
13. `2d_channel_obstacle_wake_split`: Neumann channel-obstacle pulse splitting around the cylinder.
14. `2d_channel_obstacle_shadow_collision`: obstacle/wall-clamped channel bands that collide in the wake shadow.
15. `2d_multi_hole_plate_perforation_scatter`: Neumann perforated plate pulse scattering through randomized holes.
16. `2d_multi_hole_plate_hole_lane_pulses`: outer-clamped perforated plate wave packet guided between hole lanes.
17. `2d_venturi_channel_throat_focus`: Neumann venturi pulse focusing through the throat.
18. `2d_venturi_channel_wall_echo`: venturi wall-clamped bands producing throat echoes.
19. `2d_y_bifurcation_branch_split`: Neumann Y-bifurcation inlet pulse splitting into both branches.
20. `2d_y_bifurcation_junction_collision`: wall-clamped Y-bifurcation packets colliding near the junction.
21. `3d_soliton_pulse_propagation`: 3D localized soliton-like pulse propagating and reshaping.

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
2. `schrodinger` -> `2d_stadium_wavepacket_billiard`: wave packet launched into a stadium cavity for billiard reflections and interference.

#### Biology

1. `cyclic_competition` -> `2d_dumbbell_species_exchange`: independent spiral domains in each lobe with intermittent invasion through the bottleneck.
2. `fitzhugh_nagumo` -> `2d_eccentric_annulus_reentry`: obstacle-anchored reentry with biased drift and breakup.
3. `klausmeier_topography` -> `2d_trapezoidal_hillslope_bands`: widening or narrowing slope geometry bending vegetation bands across the domain.

#### Fluids

1. `darcy` -> `2d_dumbbell_porous_exchange`: pressure-driven porous transport between two reservoirs connected by a thin throat.
2. `navier_stokes` -> `2d_cylinder_wake_rollup`: the canonical smooth-obstacle wake case, complementing the existing `channel_obstacle` example.
3. `thermal_convection` -> `3d_spherical_shell_convection_cells`: shell convection with plume columns, boundary layers, and curved-cell organization.

#### Physics

1. `maxwell_pulse` -> `2d_dumbbell_em_pulse_transfer`: launch a pulse in one lobe and watch partial transmission and trapping through the neck.
2. `superlattice` -> `2d_disk_superlattice_targets`: circular geometry should favor concentric competition between the fast and slow pattern scales.
3. `swift_hohenberg` -> `2d_eccentric_annulus_roll_pinning`: roll patterns and defects should lock to the broken-symmetry inner obstacle.
5. `van_der_pol` -> `2d_disk_relaxation_targets`: relaxation oscillations in a disk may form cleaner target-like wave trains than the rectangular and parallelogram cases.
