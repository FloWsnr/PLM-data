# PDE Systems and Configurations TODO

This document tracks PDE systems and parameter combinations that are mentioned in descriptions but not yet implemented, as well as completely new PDEs to consider.

---

## Part 1: Parameter Regimes & Configs Missing (Mentioned in Descriptions)

### Gray-Scott (Currently 12 configs, but 13+ regimes exist)
The descriptions mention many more behavioral regimes:
- [ ] **U-skate world** - Complex glider-like patterns
- [ ] **Pulsating spots** - Oscillating localized structures (partially covered)
- [ ] **Finger patterns** - Elongated growth structures
- [ ] **Wave-dominated** regimes with different wavelengths
- [ ] **Mitosis** - Self-replicating spots

### Gierer-Meinhardt
- [ ] Higher saturation constant K > 0.003 (explore intermediate K values)
- [ ] Stripe-to-spot transition systematic exploration
- [ ] **GM with source term** - External morphogen production

### FitzHugh-Nagumo
- [ ] **Driven/forced variants** - Periodic stimulation
- [ ] **Turing-Hopf boundary** - Systematic exploration
- [ ] **Excitable vs oscillatory** transition configs

### Schnakenberg
- [ ] **Turing-Hopf coexistence** boundary exploration
- [ ] Gradient-driven Schnakenberg (like heterogeneous GM)

### Cahn-Hilliard
- [ ] Faster dynamics with r increased by 1-2 orders of magnitude
- [ ] **Nucleation dynamics** - Crystal formation scenarios
- [ ] **Grain boundary motion** configurations

### Bistable Allen-Cahn
- [ ] Critical a=0.5 wave direction reversal systematic study
- [ ] **Allee effect** spatial structure exploration

### Keller-Segel
- [ ] Chemotaxis-driven blow-up vs saturation regimes
- [ ] Population heterogeneity configurations

### Complex Ginzburg-Landau
- [ ] **Benjamin-Feir instability** systematic exploration
- [ ] Stable vs unstable wave regime transitions
- [ ] More dispersive regimes

### Nonlinear Schrodinger
- [ ] **Dark solitons** (defocusing, kappa<0) - currently only bright solitons
- [ ] **Two-soliton collision** dynamics
- [ ] **Multi-soliton interactions** and soliton fusion

### Swift-Hohenberg
- [ ] **Snaking diagrams** - Localised structure bifurcations
- [ ] **Multi-bump localised solutions**
- [ ] **Breathing/pulsating** localised patterns
- [ ] Hexagon/square/stripe selection via parameter variation

### Kuramoto-Sivashinsky
- [ ] **Period-doubling route to chaos** systematic exploration
- [ ] Domain size effects on attractor dimension

### Burgers/Inviscid Burgers
- [ ] **Transonic** (mixed subsonic-supersonic) configurations
- [ ] External forcing or periodic driving terms
- [ ] Gap between inviscid shocks and smooth viscous waves

### Duffing
- [ ] **Softening springs** (beta < 0) - only hardening/double-well explored
- [ ] **Forced-damped variants**
- [ ] Bifurcation cascades systematic study

### Lorenz
- [ ] **Rho parameter transitions** through chaos
- [ ] Different initial condition basins of attraction

### Shallow Water
- [ ] **Rossby number** systematic studies
- [ ] **Tsunami/bore propagation** with realistic topography
- [ ] **Wind stress forcing**
- [ ] Geostrophic balance (high Coriolis) exploration

### Navier-Stokes
- [ ] **High-Re turbulent regime** configurations
- [ ] **Temperature-dependent viscosity** (thermal effects)
- [ ] Other obstacle geometries: airfoil, multiple cylinders
- [ ] **Poiseuille flow** pressure-driven channel (mentioned but limited)

### Thermal Convection
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
- [x] **Korteweg-de Vries (KdV)** - Classic 1D soliton equation ✅ IMPLEMENTED
- [x] **Sine-Gordon equation** - Topological kink solitons ✅ IMPLEMENTED
- [ ] **Modified KdV** - Cubic nonlinearity variant

### Ecology / Epidemiology
- [x] **SIR spatial epidemic model** - Reaction-diffusion epidemic ✅ IMPLEMENTED
- [x] **Lotka-Volterra predator-prey** (2-species with diffusion) ✅ IMPLEMENTED
- [ ] **4+ species competition** - Extended rock-paper-scissors
  - More complex cyclic competition dynamics

### Stochastic/Statistical PDEs
- [x] **Fokker-Planck equation** - Probability diffusion ✅ IMPLEMENTED
- [ ] **Stochastic reaction-diffusion** (beyond Gray-Scott)
  - Noise-driven pattern formation for other systems
  - Stochastic Turing patterns

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
- [ ] **Gross-Pitaevskii equation** - Bose-Einstein condensates
  - Similar to NLS but with trap potential
  - Vortex lattice formation
  - References: [COMSOL Model](https://www.comsol.com/model/grosspitaevskii-equation-for-boseeinstein-condensation-52221)

- [ ] **Finite potential well** - Schrodinger with barriers
  - Quantum tunneling configurations
  - Multiple well systems

### Fluid Mechanics Extensions
- [ ] **Boussinesq equations** (water waves version)
  - 2D shallow water with dispersion
  - Soliton interactions
  - References: [Cambridge Paper](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/twodimensional-boussinesq-equation-for-water-waves-and-some-of-its-solutions/7B7EB11441FA5169186D3E2204891BD7)

- [ ] **Compressible Navier-Stokes**
  - Shock dynamics
  - Aerodynamics applications

- [ ] **Kelvin-Helmholtz instability** - Shear layer roll-up

- [ ] **Rayleigh-Taylor instability** - Density stratification

### Coupled Systems
- [ ] **Coupled CGL** - Cross-phase modulation
- [ ] **Coupled NLS** - Multi-component waves (fiber optics)
- [ ] **Coupled Brusselator** - Multiple oscillating systems
- [ ] **Three-way pattern coupling** - Beyond superlattice

### Image Processing / Computer Vision
- [ ] **Total Variation (TV) denoising** - ROF model
- [ ] **Level-set methods** - Interface tracking
- [ ] **Active contours** - Segmentation PDEs

---

## Part 3: New PDEs from Literature & Benchmarks

### From PDEBench (NeurIPS 2022)
Standard benchmark PDEs for ML:
- [x] Diffusion-reaction (have as various reaction-diffusion)
- [x] Compressible Navier-Stokes (partially - have incompressible)
- [x] Shallow water (have)
- [ ] **Darcy flow** - Porous media flow
  - -div(K grad p) = f
  - Used in groundwater modeling

### From APEBench (2024)
Autoregressive emulator benchmarks:
- [x] Most basic PDEs covered

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

### Medium Priority (Extends Coverage)
6. **Porous medium equation** - Nonlinear diffusion
7. **Gross-Pitaevskii** - Quantum systems
8. **Darcy flow** - Porous media (PDEBench)
9. **Helmholtz equation** - Wave scattering
10. **Phase-field crystal** - Materials

### Lower Priority (Specialized)
11. **Boussinesq water waves**
12. **Viscoelastic wave**
13. **Compressible NS**
14. **Coupled CGL/NLS**
15. **TV denoising / Level sets**

---

## Part 5: Config Expansion for Existing PDEs

### Presets with Minimal Coverage (< 4 configs)
| Preset | Current Configs | Suggested Additions |
|--------|-----------------|---------------------|
| vorticity-bounded | 1 | +5 (viscosity sweep, rotation, wavenumbers) |

### Well-Covered but Could Expand
| Preset | Current Configs | Gap Areas |
|--------|-----------------|-----------|
| gray-scott | 12 | U-skate, mitosis, more chaos regimes |
| nonlinear-schrodinger | 6 | Dark solitons, multi-soliton |
| swift-hohenberg | 5 | Localised structures, snaking |
| complex-ginzburg-landau | 5 | Benjamin-Feir, more chaos |

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

*Last updated: 2026-01-21*
