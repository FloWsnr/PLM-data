# PDE Systems and Configurations TODO

This document tracks PDE systems and parameter combinations that are mentioned in descriptions but not yet implemented, as well as completely new PDEs to consider.

**Current inventory: 61 PDE presets, 363 configs across 4 categories (basic: 12, biology: 20, physics: 21, fluids: 8)**

---

## Part 1: Parameter Regimes & Configs Missing (Mentioned in Descriptions)

### Gray-Scott (Currently 13 configs)
The descriptions mention many more behavioral regimes:
- [ ] **U-skate world** - Complex glider-like patterns
- [ ] **Pulsating spots** - Oscillating localized structures (partially covered)
- [ ] **Finger patterns** - Elongated growth structures
- [ ] **Wave-dominated** regimes with different wavelengths
- [ ] **Mitosis** - Self-replicating spots

### Gierer-Meinhardt (Currently 8 configs)
- [ ] Higher saturation constant K > 0.003 (explore intermediate K values)
- [ ] Stripe-to-spot transition systematic exploration
- [ ] **GM with source term** - External morphogen production

### FitzHugh-Nagumo (Currently 7 configs + 6 FHN-3 configs)
- [ ] **Driven/forced variants** - Periodic stimulation
- [ ] **Turing-Hopf boundary** - Systematic exploration
- [ ] **Excitable vs oscillatory** transition configs

### Schnakenberg (Currently 5 configs + 5 cross-diffusion-schnakenberg configs)
- [ ] **Turing-Hopf coexistence** boundary exploration

### Cahn-Hilliard (Currently 7 configs)
- [ ] Faster dynamics with r increased by 1-2 orders of magnitude
- [ ] **Nucleation dynamics** - Crystal formation scenarios
- [ ] **Grain boundary motion** configurations

### Bistable Allen-Cahn (Currently 7 configs)
- [ ] Critical a=0.5 wave direction reversal systematic study
- [ ] **Allee effect** spatial structure exploration

### Keller-Segel (Currently 6 configs)
- [ ] Chemotaxis-driven blow-up vs saturation regimes
- [ ] Population heterogeneity configurations

### Complex Ginzburg-Landau (Currently 6 configs)
- [ ] **Benjamin-Feir instability** systematic exploration
- [ ] Stable vs unstable wave regime transitions
- [ ] More dispersive regimes

### Nonlinear Schrodinger (Currently 7 configs)
- [ ] **Dark solitons** (defocusing, kappa<0) - currently only bright solitons
- [ ] **Two-soliton collision** dynamics
- [ ] **Multi-soliton interactions** and soliton fusion

### Swift-Hohenberg (Currently 6 configs + 5 swift-hohenberg-advection configs)
- [ ] **Snaking diagrams** - Localised structure bifurcations
- [ ] **Multi-bump localised solutions**
- [ ] **Breathing/pulsating** localised patterns
- [ ] Hexagon/square/stripe selection via parameter variation

### Kuramoto-Sivashinsky (Currently 7 configs)
- [ ] **Period-doubling route to chaos** systematic exploration
- [ ] Domain size effects on attractor dimension

### Burgers/Inviscid Burgers (Currently 7 + 7 configs)
- [ ] **Transonic** (mixed subsonic-supersonic) configurations
- [ ] External forcing or periodic driving terms
- [ ] Gap between inviscid shocks and smooth viscous waves

### Duffing (Currently 7 configs)
- [ ] **Softening springs** (beta < 0) - only hardening/double-well explored
- [ ] **Forced-damped variants**
- [ ] Bifurcation cascades systematic study

### Lorenz (Currently 6 configs)
- [ ] **Rho parameter transitions** through chaos
- [ ] Different initial condition basins of attraction

### Shallow Water (Currently 8 configs)
- [ ] **Rossby number** systematic studies
- [ ] **Tsunami/bore propagation** with realistic topography
- [ ] **Wind stress forcing**
- [ ] Geostrophic balance (high Coriolis) exploration

### Navier-Stokes (Currently 6 + 5 navier-stokes-cylinder configs)
- [ ] **High-Re turbulent regime** configurations
- [ ] **Temperature-dependent viscosity** (thermal effects)
- [ ] Other obstacle geometries: airfoil, multiple cylinders
- [ ] **Poiseuille flow** pressure-driven channel (mentioned but limited)

### Thermal Convection (Currently 5 configs)
- [ ] **Ra above/below critical** (~1707) systematic exploration
- [ ] **Durham convection** variant with stochastic forcing

### Vorticity-Bounded (only 1 config!)
- [ ] Different boundary geometries
- [ ] Viscosity range exploration
- [ ] Rotation rate variations
- [ ] Wavenumber studies

---

## Part 2: Related Systems Mentioned But Not Implemented

### Integrable Systems / Soliton Equations
- [x] **Korteweg-de Vries (KdV)** - Classic 1D soliton equation ✅ (4 configs)
- [x] **Sine-Gordon equation** - Topological kink solitons ✅ (1 config)
- [x] **Zakharov-Kuznetsov** - 2D/3D generalization of KdV ✅ (8 configs)
- [ ] **Modified KdV** - Cubic nonlinearity variant

### Ecology / Epidemiology
- [x] **SIR spatial epidemic model** - Reaction-diffusion epidemic ✅ (2 configs)
- [x] **Lotka-Volterra predator-prey** (2-species with diffusion) ✅ (2 configs)
- [x] **Cyclic competition** - 3-species rock-paper-scissors ✅ (7 + 6 wave configs)
- [x] **Klausmeier vegetation** - Dryland vegetation patterns ✅ (6 + 7 topography configs)
- [x] **Bacteria advection** - Bacterial colony with chemotaxis ✅ (7 configs)
- [x] **Immunotherapy** - Tumor-immune dynamics ✅ (6 configs)
- [x] **Harsh environment** - Population survival under stress ✅ (7 configs)
- [ ] **4+ species competition** - Extended beyond 3-species cyclic

### Stochastic/Statistical PDEs
- [x] **Fokker-Planck equation** - Probability diffusion ✅ (1 config)
- [x] **Stochastic Gray-Scott** - Noise-driven pattern formation ✅ (14 configs)
- [ ] **Stochastic reaction-diffusion** for other systems (beyond Gray-Scott)
  - Stochastic Turing patterns in FHN, Schnakenberg, etc.

### Materials Science / Mechanics
- [ ] **Viscoelastic wave equation** - Memory effects in materials
  - Kelvin-Voigt or Maxwell models
  - Frequency-dependent attenuation

- [ ] **Porous medium equation** - Nonlinear diffusion
  - du/dt = Laplacian(u^m), m > 1
  - Gas flow through porous media
  - Finite propagation speed (unlike heat equation)
  - References: [Wikipedia](https://en.wikipedia.org/wiki/Porous_medium_equation)

### Quantum Systems
- [x] **Schrodinger equation** - Linear quantum mechanics ✅ (7 configs)
- [ ] **Gross-Pitaevskii equation** - Bose-Einstein condensates
  - Similar to NLS but with trap potential
  - Vortex lattice formation
  - References: [COMSOL Model](https://www.comsol.com/model/grosspitaevskii-equation-for-boseeinstein-condensation-52221)

- [ ] **Finite potential well** - Schrodinger with barriers
  - Quantum tunneling configurations
  - Multiple well systems

### Fluid Mechanics Extensions
- [x] **Darcy flow** - Porous media flow ✅ (2 configs)
- [x] **Potential flow dipoles** - Inviscid irrotational flow ✅ (5 configs)
- [ ] **Boussinesq equations** (water waves version)
  - 2D shallow water with dispersion
  - Soliton interactions
  - References: [Cambridge Paper](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/twodimensional-boussinesq-equation-for-water-waves-and-some-of-its-solutions/7B7EB11441FA5169186D3E2204891BD7)

- [ ] **Compressible Navier-Stokes**
  - Shock dynamics
  - Aerodynamics applications

- [ ] **Kelvin-Helmholtz instability** - Shear layer roll-up

- [ ] **Rayleigh-Taylor instability** - Density stratification

### Coupled / Extended Systems
- [x] **Superlattice** - Multi-pattern coupling ✅ (5 configs)
- [x] **Hyperbolic Brusselator** - Wave-reaction coupling ✅ (7 configs)
- [x] **Cross-diffusion Schnakenberg** - Cross-diffusion effects ✅ (5 configs)
- [ ] **Coupled CGL** - Cross-phase modulation
- [ ] **Coupled NLS** - Multi-component waves (fiber optics)
- [ ] **Coupled Brusselator** - Multiple oscillating systems

### Image Processing / Computer Vision
- [x] **Perona-Malik** - Anisotropic diffusion denoising ✅ (6 configs)
- [ ] **Total Variation (TV) denoising** - ROF model
- [ ] **Level-set methods** - Interface tracking
- [ ] **Active contours** - Segmentation PDEs

---

## Part 3: New PDEs from Literature & Benchmarks

### From "The Well" (NeurIPS 2024)
Large-scale physics simulation collection - check for gaps

### Classic PDEs Still Missing
- [ ] **Biharmonic equation** - Laplacian^2 u = f
  - Plate bending statics (have dynamic plate)
  - Stokes stream function

- [ ] **Eikonal equation** - |grad u| = f
  - Wave front propagation
  - Fast marching methods

- [ ] **Helmholtz equation** - Laplacian u + k^2 u = f
  - Acoustic/electromagnetic waves
  - Scattering problems

- [ ] **Poisson-Boltzmann** - Electrostatics in solution
  - Biomolecular applications

- [ ] **Phase-field crystal** - Atomic-scale pattern formation
  - Crystallization dynamics
  - Grain boundaries

- [ ] **Convective Cahn-Hilliard** - Phase separation with flow
  - Spinodal decomposition in moving fluid

---

## Part 4: Priority Implementation List

### High Priority (Fundamental/Commonly Requested)
✅ All high priority PDEs implemented:
- [x] **KdV equation** - Canonical soliton system
- [x] **Sine-Gordon** - Topological solitons
- [x] **SIR epidemic** - High practical relevance
- [x] **Lotka-Volterra predator-prey** - Fundamental ecology
- [x] **Fokker-Planck** - Stochastic processes
- [x] **Darcy flow** - Porous media (PDEBench)

### Medium Priority (Extends Coverage)
- [ ] **Porous medium equation** - Nonlinear diffusion
- [ ] **Gross-Pitaevskii** - Quantum systems
- [ ] **Helmholtz equation** - Wave scattering
- [ ] **Phase-field crystal** - Materials

### Lower Priority (Specialized)
- [ ] **Boussinesq water waves**
- [ ] **Viscoelastic wave**
- [ ] **Compressible NS**
- [ ] **Coupled CGL/NLS**
- [ ] **TV denoising / Level sets**

---

## Part 5: Config Expansion for Existing PDEs

### Presets with Minimal Coverage (< 4 configs)
| Preset | Current Configs | Suggested Additions |
|--------|-----------------|---------------------|
| vorticity-bounded | 1 | +5 (viscosity sweep, rotation, wavenumbers) |
| fokker-planck | 1 | +4 (different potentials, drift terms, diffusion rates) |
| sine-gordon | 1 | +4 (breather, kink-antikink collision, multi-kink) |
| advection | 2 | +3 (different velocity fields, higher dimensions) |
| darcy | 2 | +3 (permeability fields, source terms) |
| sir | 2 | +3 (different R0, spatial heterogeneity, waves) |
| lotka-volterra | 2 | +3 (coexistence, exclusion, spatial patterns) |

### Well-Covered but Could Expand
| Preset | Current Configs | Gap Areas |
|--------|-----------------|-----------|
| stochastic-gray-scott | 14 | Good coverage |
| gray-scott | 13 | U-skate, mitosis, more chaos regimes |
| shallow-water | 8 | Rossby, wind forcing, tsunami |
| gierer-meinhardt | 8 | Source term, stripe-spot transition |
| zakharov-kuznetsov | 8 | Good coverage |
| nonlinear-schrodinger | 7 | Dark solitons, multi-soliton |
| swift-hohenberg | 6 | Localised structures, snaking |
| complex-ginzburg-landau | 6 | Benjamin-Feir, more chaos |

---

## Full PDE Inventory (61 presets)

### Basic (12 presets)
advection (2), advection-rotational (4), blob-diffusion-heat (6), damped-wave (7), heat (7), inhomogeneous-diffusion-heat (6), inhomogeneous-heat (6), inhomogeneous-wave (6), plate (7), schrodinger (7), wave (6), wave-standing (7)

### Biology (20 presets)
bacteria-advection (7), bistable-allen-cahn (7), brusselator (7), cross-diffusion-schnakenberg (5), cyclic-competition (7), cyclic-competition-wave (6), fisher-kpp (7), fitzhugh-nagumo (7), fitzhugh-nagumo-3 (6), gierer-meinhardt (8), harsh-environment (7), heterogeneous-gierer-meinhardt (7), hyperbolic-brusselator (7), immunotherapy (6), keller-segel (6), klausmeier (6), klausmeier-topography (7), lotka-volterra (2), schnakenberg (5), sir (2)

### Physics (21 presets)
bistable-advection (7), burgers (7), cahn-hilliard (7), complex-ginzburg-landau (6), duffing (7), fokker-planck (1), gray-scott (13), inviscid-burgers (7), kdv (4), kuramoto-sivashinsky (7), lorenz (6), nonlinear-schrodinger (7), perona-malik (6), sine-gordon (1), stochastic-gray-scott (14), superlattice (5), swift-hohenberg (6), swift-hohenberg-advection (5), van-der-pol (7), zakharov-kuznetsov (8)

### Fluids (8 presets)
darcy (2), navier-stokes (6), navier-stokes-cylinder (5), potential-flow-dipoles (5), shallow-water (8), thermal-convection (5), vorticity (6), vorticity-bounded (1)

---

## References

### Benchmark Datasets
- [PDEBench](https://github.com/pdebench/PDEBench) - NeurIPS 2022 benchmark
- [APEBench](https://github.com/tum-pbs/apebench) - JAX-based autoregressive benchmark
- [The Well](https://arxiv.org/abs/2410.01108) - NeurIPS 2024 large-scale collection

### Interactive Tools
- [VisualPDE](https://visualpde.com/) - Browser-based PDE explorer
- [COMSOL Models](https://www.comsol.com/models) - Application library

### Key Papers
- Turing patterns: [Science 2010](https://www.science.org/doi/10.1126/science.1179047)
- Reaction-diffusion review: [ScienceDirect 2024](https://www.sciencedirect.com/science/article/pii/S1571064524001465)
- Epidemic PDEs: [PMC Review](https://pmc.ncbi.nlm.nih.gov/articles/PMC8671691/)

---

*Last updated: 2026-02-11*
