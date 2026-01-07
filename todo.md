# PDE Implementation TODO

## Equation Discrepancies

### `burgers` - Backward Difference for Advection Term
**Priority**: Medium
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 5617

**Issue**: Visual PDE uses a backward difference for the advection term to ensure numerical stability with upwinding.

**Visual PDE BurgersEquation**:
```javascript
reactionStr_1: "- u * u_xb"  // u_xb = backward difference
diffusionStr_1_1: "epsilon"
```

**Our implementation** (`pde_sim/pdes/physics/burgers.py`):
```python
rhs={"u": f"-u * d_dx(u) + {epsilon} * laplace(u)"}  # centered difference
```

**Impact**: Centered differences can cause oscillations near sharp fronts in advection-dominated flows. With sufficient viscosity (epsilon > 0), this is less critical, but for small epsilon or inviscid limits, backward differencing provides better stability.

**Required investigation**:
1. Check if py-pde supports backward/forward differences (`d_dx_b`, `d_dx_f`)
2. If not, consider implementing via custom PDE class with explicit finite differences
3. May need to use `gradient()` with appropriate boundary handling

**Config fixed** (2026-01-07):
- Updated `configs/defaults/physics/burgers.yaml` to match Visual PDE parameters:
  - amplitude: 0.1 (was 1.0)
  - bc: neumann (was periodic)
  - dimension: 1
  - domain_size: 1000 (was 10)
  - dt: 0.04, solver: euler

---

### `inhomogeneous-wave` - Visual PDE uses simplified equation
**Priority**: Medium
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 6840

Our implementation uses the mathematically correct div(f·∇u) formulation, but Visual PDE uses a simpler f·∇²u approximation.

**Our implementation** (`pde_sim/pdes/basic/wave.py:233`):
```
dv/dt = f*laplace(u) + df_dx*d_dx(u) + df_dy*d_dy(u)  # = div(f·grad(u))
```

**Visual PDE preset**:
```
dv/dt = f(x,y) * laplace(u)  # simplified approximation
```

where f(x,y) = D*(1+E*sin(m*π*x/L_x))*(1+E*sin(n*π*y/L_y))

**Issue**: The markdown documentation (`inhomogeneous-wave-equation.md`) states the true equation is ∂²u/∂t² = ∇·(f∇u), which matches our implementation. However, Visual PDE's actual preset uses a simpler approximation.

**Decision needed**:
1. Keep our mathematically correct implementation (current)
2. Add a parameter to toggle between formulations
3. Switch to Visual PDE's simplified version for consistency

---

## Missing Preset Variants

### `inviscid-burgers` - Missing Flux Splitting Preset
**Priority**: Medium
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 545

Visual PDE implements inviscid Burgers using a clever flux-splitting approach with an algebraic auxiliary variable:

**Equations**:
```
∂u/∂t = - (1/(p*u^(p-2))) * v_xb   (where p=4)
v = u^p                            (algebraic constraint)
```

**Parameters**:
- `epsilon = 0` (truly inviscid)
- `p = 4` (flux power)
- `dt = 0.03`
- Uses RK4 timestepping

**Initial condition**:
```
u = 0.001 + 0.1*exp(-0.00005*(x-L_x/5)^2)
```
Note: Small offset (0.001) prevents division by zero in reaction term.

**Implementation notes**:
- Requires `MultiFieldPDEPreset` with algebraic species
- The `v_xb` backward difference is critical for shock capturing
- This approach avoids Gibbs phenomena near discontinuities

---

### `inviscid-burgers-shock-interaction` - Missing Multiple Shock Preset
**Priority**: Low
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 520

Variant of `inviscid-burgers` demonstrating shock interaction:

**Initial condition** (multiple Gaussians at different positions and heights):
```
u = 0.001 + 0.10*exp(-0.001*(x-L_x/8)^2)
         + 0.08*exp(-0.001*(x-L_x/4)^2)
         + 0.06*exp(-0.001*(x-3*L_x/8)^2)
         + 0.04*exp(-0.001*(x-L_x/2)^2)
```

**Behavior**: Taller shocks (faster) overtake shorter ones, demonstrating shock merging.

**Implementation**: Once `inviscid-burgers` is implemented, this is just a config variant with different initial conditions.

---

### `damped-wave` - Wave equation with explicit velocity damping
**Priority**: Low
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 6864

Visual PDE has a `dampedWaveEquation` preset with explicit damping term `-d*v` in the v equation:
```
du/dt = v + C*D*laplace(u)
dv/dt = D*laplace(u) - d*v
```

Our current wave equation has damping only via the C term in the u equation. This is a different kind of damping.

**Parameters**:
- `d`: 0 in [0, 0.1] - velocity damping coefficient
- Dirichlet boundary conditions
- Inhomogeneous BC: u|∂Ω = cos(m*π*x/100)*cos(m*π*y/100)

---

### `wave-standing` - Standing wave initial conditions config
**Priority**: Low
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 7112 (waveEquationICs)

A preset config with standing wave initial conditions:
```yaml
init:
  type: "expression"
  expression: "cos(n*pi*x/L_x)*cos(m*pi*y/L_y)"
parameters:
  D: 1.0
  C: 0.0  # No damping for pure standing waves
  n: 4  # in [1, 10]
  m: 4  # in [1, 10]
```

Demonstrates membrane vibration modes. Could be added as a config file rather than new code.

---

### `bistable-advection` - Bistable Allen-Cahn with advection
**Priority**: Low
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 5495

Visual PDE has a `BistableAdvection` preset that adds a flow term to the bistable Allen-Cahn equation:
```
du/dt = D*laplace(u) + u*(u-a)*(1-u) + V*(cos(theta)*u_x + sin(theta)*u_y)
```

**Parameters**:
- `a`: 0.48 in [0, 1] - Allee threshold (near critical)
- `theta`: 1.8 in [-6.4, 6.4] - flow direction (radians)
- `V`: 0.5 in [0, 5] - flow velocity
- `D`: 0.02 - diffusion coefficient (note: smaller than base preset)
- `spatialStep`: 0.3, `dt`: 0.005

**Use case**: Models populations in flowing water (e.g., rivers). Flow can assist or hinder invasion depending on direction relative to wave propagation.

**Implementation**: Could be a separate preset or add optional `V`, `theta` parameters to existing `bistable-allen-cahn`.

---

### `cyclic-competition-wave` - Equal diffusion wave variant
**Priority**: Low
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 1233

Visual PDE has a `cyclicCompetitionWave` preset demonstrating wave-induced spatiotemporal chaos without Turing instability:

**Parameters**:
- Equal diffusion: `D_u = D_v = D_w = 0.3`
- dt: 0.01

**Initial condition** (requires Heaviside support):
```
u = H(0.1-x/L_x)*(1+0.001*RANDN)
v = H(0.1-x/L_x)
w = H(0.1-x/L_x)
```

Species present only in left 10% of domain initially.

**Implementation options**:
1. Add `cyclic-competition-wave` preset with custom IC
2. Add Heaviside IC type to generic IC generator
3. Add `ic_type: "heaviside"` support in `cyclic-competition` preset

---

### `fitzhugh-nagumo-3` - Three-species FitzHugh-Nagumo variant
**Priority**: Low
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 6489

Visual PDE has a three-species variant with competing oscillations and pattern formation:

**Equations**:
```
∂u/∂t = ∇²u + u - u³ - v
∂v/∂t = Dv∇²v + εv(u - av*v - aw*w - az)
∂w/∂t = Dw∇²w + εw(u - w)
```

**Parameters**:
- `Dv = 40`, `Dw = 200`
- `a_v = 0.2` (range: [0, 0.5]): Controls pattern vs oscillation dominance
- `e_v = 0.2`, `e_w = 1`, `a_w = 0.5`, `a_z = -0.1`

**Initial condition**:
```
u = 5*exp(-0.1*((x-Lx/2)² + (y-Ly/2)²))
v = cos(m*x*π/280)*cos(m*y*π/280)
w = 0
```

**Behavior**: Pattern formed initially will eventually be destroyed by oscillations. Increasing `a_v ≥ 0.3` stabilizes patterns that overtake oscillations.

**Note**: The base `fitzhugh-nagumo` implementation is correct. This is an extension with a third species.

---

### `gierer-meinhardt-stripes` - Saturation for stable stripe patterns
**Priority**: Medium
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 6347

Visual PDE has a `GiererMeinhardtStripes` preset with a saturation term that enables stable stripe/labyrinthine patterns:

**Equations**:
```
∂u/∂t = ∇²u + a + u²/(v*(1+K*u²)) - b*u
∂v/∂t = D∇²v + u² - c*v
```

**Parameters**:
- `K`: 0.003 in [0, 0.003] - saturation constant
- `domain_size`: 200 (larger domain)
- Other params same as standard: a=0.5, b=1, c=6.1, D=100

**Behavior**:
- K=0: spot-forming (same as standard)
- K~0.003: labyrinthine/stripe patterns
- K too large: no Turing patterns

**Implementation**: Add K parameter to existing preset or create separate `gierer-meinhardt-stripes` variant.

---

### `gierer-meinhardt-stripe-ics` - Stripe initial conditions
**Priority**: Low
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 6321

Demonstrates stripe-to-spot instability:

**Initial conditions**:
```
u(0) = 3*(1+cos(n*π*x/L)), v(0) = 1
```

**Parameters**:
- `n`: 6 in [1, 20] - number of initial stripes
- Neumann boundary conditions

**Use case**: Shows transverse instability where stripes break into spots when perturbed.

**Implementation**: Add as IC type option or separate config preset.

---

### `immunotherapy` - Two Visual PDE Formulations Exist
**Priority**: Medium
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` lines 717-892, 6557-6599

Visual PDE has **two different** immunotherapy formulations with different equation structures and parameter values:

**1. ImmunotherapyModel** (original, uses E, T, S notation):
```javascript
reactionStr_1: "c*T-mu*E+p_u*E*S/(g_u+S)+s_u"  // Note: g_u+S denominator
reactionStr_2: "T*(1-T)-p_v*E*T/(g_v+T)"       // Note: p_v coefficient
kineticParams: "c=0.3;mu=0.167;p_u=0.69167;g_u=20;p_v=0.5555556;g_v=0.1;..."
diffusion: D_E=0.5, D_T=0.008, D_S=4
```

**2. ImmunotherapyCircleNeumann** (paper version, uses u, v, w notation):
```javascript
reactionStr_1: "alpha*v-mu_u*u+rho_u*u*w/(1+w)+sigma_u+K_u*t"  // Note: 1+w denominator
reactionStr_2: "v*(1-v)-u*v/(gamma_v+v)"                       // Note: no p_v
kineticParams: "rho_u=0.692;alpha=0.07;delta_u=100;delta_w=100;..."
```

**Our implementation** follows the ImmunotherapyCircleNeumann equation structure (1+w denominator, no p_v) but uses ImmunotherapyModel parameter values (Du=0.5, c=0.3, etc.). This hybrid is reasonable but differs from both presets.

**Implemented fixes** (2025-01-07):
- Added `Dv` parameter for tumor diffusion (default=0.008, matching ImmunotherapyModel)
- Updated initial conditions to use near-equilibrium uniform values with noise (u=0.299, v=0.505, w=0.022)

**Missing features for full compatibility**:

1. **p_v parameter**: ImmunotherapyModel uses `p_v=0.5555556` for tumor killing rate coefficient:
   ```
   dv/dt = ... - p_v*u*v/(g_v+v)  # We use implicit p_v=1
   ```

2. **g_u parameter**: ImmunotherapyModel uses saturation in effector proliferation:
   ```
   du/dt = ... + p_u*u*S/(g_u+S)  # with g_u=20
   # vs our: p_u*u*w/(1+w)
   ```

3. **Time-dependent treatment (K_u, K_w)**: ImmunotherapyCircleNeumann supports:
   ```
   du/dt = ... + sigma_u + K_u*t
   dw/dt = ... + sigma_w + K_w*t
   ```

**Recommendations**:
1. Add optional `p_v` parameter (default=1.0 for backward compatibility)
2. Consider creating `immunotherapy-model` variant with g_u saturation for full ImmunotherapyModel compatibility
3. Low priority: Add K_u, K_w for time-dependent treatment protocols

---

### `heterogeneous-gierer-meinhardt` - Boundary Condition Architecture Mismatch
**Priority**: High
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 6369

**Issue**: Visual PDE uses **per-species** boundary conditions, while our framework uses **per-axis** boundary conditions.

**Visual PDE GMHeterogeneous2D**:
- `boundaryConditions_1: "dirichlet"` → u has Dirichlet (u=0) on ALL boundaries
- v defaults to Neumann on ALL boundaries

**Our config** (`configs/defaults/biology/heterogeneous_gierer_meinhardt.yaml`):
```yaml
bc:
  x: dirichlet
  y: neumann
```
This applies Dirichlet on x-edges, Neumann on y-edges for BOTH species, which is semantically different.

**Impact**: Pattern formation behavior may differ because:
1. Visual PDE: u=0 enforced everywhere on boundary, v free everywhere
2. Ours: Both u,v get Dirichlet on x-edges, Neumann on y-edges

**Required changes**:
1. Extend BC system to support per-species boundary conditions
2. Update config schema to allow: `bc: {u: dirichlet, v: neumann}`
3. Update `heterogeneous-gierer-meinhardt` config to use per-species BCs
4. Also affects other multi-field PDEs that need asymmetric BCs

**Initial Conditions**:
Visual PDE uses `v=1` (uniform), u unset/0. Patterns emerge from BC perturbation.
Our default uses noisy IC (mean=1.0, std=0.1). Consider adding uniform IC option.

---

### `navier-stokes` - Per-Field Boundary Conditions
**Priority**: Low
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 2019

**Issue**: Visual PDE uses **per-species** boundary conditions with different BCs for each field, while our config uses simplified per-axis BCs.

**Visual PDE NavierStokes**:
```javascript
boundaryConditions_1: "dirichlet"  // u
boundaryConditions_2: "dirichlet"  // v
boundaryConditions_3: "neumann"    // p
boundaryConditions_4: "combo"      // S
comboStr_1: "Bottom: Dirichlet = 0; Left: Periodic; Right: Periodic; Top: Dirichlet = 0;"
comboStr_2: "Bottom: Dirichlet = 0; Left: Periodic; Right: Periodic; Top: Dirichlet = 0;"
comboStr_4: "Bottom: Neumann = 0; Left: Dirichlet = 1; Right: Dirichlet = 0; Top: Neumann = 0;"
```

Summary:
- u: periodic left/right, Dirichlet=0 top/bottom (no-slip walls)
- v: periodic left/right, Dirichlet=0 top/bottom (no-slip walls)
- p: Neumann on all boundaries
- S: Dirichlet 1 left, 0 right (inlet/outlet), Neumann top/bottom

**Our config** (`configs/defaults/fluids/navier_stokes.yaml`):
```yaml
bc:
  x: periodic
  y: neumann
```
This applies the same BC to all fields per axis.

**Impact**: Our simulation is still valid but represents a different physical setup (fully periodic in x, slip walls in y) vs the reference (channel flow with no-slip walls and inlet/outlet conditions for the passive scalar).

**Required changes** (same as heterogeneous-gierer-meinhardt):
1. Extend BC system to support per-species boundary conditions
2. Update config schema to allow per-field BCs

**Note**: The equations and parameters are correct - only the boundary conditions differ.

---

### `shallow-water` - Per-Field Boundary Conditions
**Priority**: Medium
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 2881

**Issue**: Visual PDE uses **per-species** boundary conditions with mixed BCs for velocity fields.

**Visual PDE ShallowWaterEqns**:
```javascript
boundaryConditions_1: "neumann"   // h
boundaryConditions_2: "dirichlet" // u (simplified; combo shows left/right dirichlet, top/bottom neumann)
boundaryConditions_3: "dirichlet" // v (simplified; combo shows top/bottom dirichlet, left/right neumann)
comboStr_2: "Left: Dirichlet = 0; Top: Neumann = 0; Right: Dirichlet = 0; Bottom: Neumann = 0;"
comboStr_3: "Left: Neumann = 0; Top: Dirichlet = 0; Right: Neumann = 0; Bottom: Dirichlet = 0;"
```

Summary:
- h: Neumann on all boundaries (free surface)
- u: Dirichlet=0 on left/right (no-flow through walls), Neumann on top/bottom
- v: Dirichlet=0 on top/bottom (no-flow through walls), Neumann on left/right

**Our config** (`configs/defaults/fluids/shallow_water.yaml`):
```yaml
bc:
  x: neumann
  y: neumann
```
This applies the same BC to all fields per axis.

**Impact**: The simulation runs correctly but represents slip walls instead of no-slip walls at boundaries. For drop perturbation scenarios this is less critical, but for channel flow or dam-break with walls it affects the physics.

**Required changes**: Same as heterogeneous-gierer-meinhardt - extend BC system to support per-species boundary conditions.

**Equations and parameters**: ✅ Correct and match Visual PDE reference exactly.

---

### `vorticity` - Per-Field Boundary Conditions for S Field
**Priority**: Low
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 2172

**Issue**: Visual PDE uses **per-species** boundary conditions for the passive scalar field S.

**Visual PDE NavierStokesVorticity**:
```javascript
comboStr_3: "Bottom: Periodic; Left: Neumann = 0; Right: Neumann = 0; Top: Periodic;"
```

Summary:
- omega: periodic (all boundaries)
- psi: periodic (all boundaries)
- S: Neumann=0 on left/right, periodic top/bottom

**Our config** (`configs/defaults/fluids/vorticity.yaml`):
```yaml
bc:
  x: periodic
  y: periodic
```
This applies periodic to all fields on all axes.

**Impact**: Minor - passive scalar boundary behavior differs but main vorticity dynamics are correct.

**Required changes**: Same as heterogeneous-gierer-meinhardt - extend BC system to support per-species boundary conditions.

**Equations and parameters**: ✅ Correct and match Visual PDE reference exactly.

---

### `vorticity-bounded` - Missing Bounded Domain Variant
**Priority**: Low
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 2111

Visual PDE has a `NavierStokesVorticityBounded` preset demonstrating decay of oscillatory vorticity in a bounded domain:

**Initial condition**:
```
omega = A*cos(k*pi*x/L_x)*cos(k*pi*y/L_x)
```
where A = 0.005*k^1.5, k = 51 in [1, 100]

**Boundary conditions**:
- omega: Dirichlet=0 on left/right (combo), Periodic top/bottom
- S: Neumann=0 on left/right, Periodic top/bottom
- psi: uses parent defaults

**Parameters**:
- D = 0.1 (different from base D=0.05)
- Other params same as base

**Implementation options**:
1. Add `vorticity-bounded` preset with custom oscillatory IC
2. Add IC type `oscillatory-vorticity` that takes k parameter
3. Requires per-species BC support for proper omega boundary conditions

---

## Missing Implementations

### `schrodinger` - Add Potential V(x,y) Support
**Priority**: Medium
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 7396

The current Schrodinger implementation only supports the basic case (V=0). Need to add support for a potential function V(x,y) as in the `stabilizedSchrodingerEquationPotential` preset.

**Equations with potential**:
```
∂u/∂t = -D∇²v + CD∇²u + V(x,y)*v
∂v/∂t = D∇²u + CD∇²v - V(x,y)*u
```

**Implementation notes**:
- Add optional `V` parameter for potential type (e.g., "none", "sinusoidal")
- Sinusoidal potential: `V(x,y) = sin(n*π*x/L)*sin(m*π*y/L)`
- The n,m parameters would control both eigenstate IC and potential shape
- Note: VisualPDE's `stabilizedSchrodingerEquationPotential` preset has WRONG signs (`-V*v` and `+V*u`). The correct signs are `+V*v` and `-V*u` (confirmed by the math, the markdown docs, and the 1D preset).

**Additional IC needed**:
- Localized wave packet: `(sin(π*x/L)*sin(π*y/L))^10` for use with potential
- This creates a concentrated initial condition that tunnels between potential wells

---

### `inhomog-diffusion-heat` - Spatially-Varying Diffusion Heat Equation
**Priority**: Medium
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 6807

The inhomogeneous-heat.md description documents this variant but it's not implemented in Python.

**Equation**:
```
∂T/∂t = ∇·(g(x,y)∇T)
```

**Spatially-varying diffusion coefficient**:
```
g(x,y) = D*(1 + E*cos(n*π*sqrt((x-Lx/2)² + (y-Ly/2)²) / sqrt(Lx*Ly)))
```

**Visual PDE preset parameters**:
- `boundaryConditions`: dirichlet
- `initCond`: 1 (uniform initial temperature)
- `D`: 1.0 (base diffusion)
- `E`: 0.99 in [0, 1] (modulation amplitude)
- `n`: 40 in [1, 50] (radial oscillations)
- `dt`: 0.004
- `domainScale`: 100
- `spatialStep`: 0.5

**Implementation notes**:
- Requires implementing spatially-varying diffusion in py-pde
- Creates radially-oscillating regions of high/low diffusion
- Heat partitions into bands bounded by diffusion maxima
- Could be a new preset `inhomog-diffusion-heat` or a parameter variant of existing `inhomogeneous-heat`

---

### `potential-flow-dipoles` - Pre-computed Dipole Forcing Limitation
**Priority**: Low
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 2478

**Visual PDE equation**:
```
∂phi/∂t = ∇²phi + 1000*(ind(s[x-d*dx,y]>0) - ind(s[x+d*dx,y]>0))/(2*d*dx)
```

Uses shifted indicator functions `s[x±d*dx, y]` to create dipole effect dynamically. This allows the separation parameter `d` to be changed interactively via slider.

**Our implementation**:
Since py-pde doesn't support shifted field access, we pre-compute the dipole forcing in the s field initial condition:
```
s = (bump(x-d*dx,y) - bump(x+d*dx,y)) / (2*d*dx)
∂phi/∂t = ∇²phi + strength * s
```

**Limitation**: The parameter `d` is baked into the initial condition. Changing `d` during runtime has no effect - a new simulation must be started with different `d` value.

**Visual PDE boundary conditions**:
- phi: neumann (default)
- s: dirichlet (s=0 at boundaries)

**Our BCs**: Uses per-axis BCs (neumann on all). The per-species BC requirement (heterogeneous-gierer-meinhardt issue above) also applies here.

**Status**: Functionally correct for single simulations with fixed d. The parabolic relaxation approach is preserved. Interactive d-slider behavior cannot be replicated.

---

### `potential-flow-images` - Analytical vs Relaxation Initial Condition
**Priority**: Low
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 2538

**Visual PDE approach**:
- `initCond_1: "0"` - phi starts at zero
- Fixed source forcing: `-10*Bump(3*L_x/4, L_y/2, L/100)` always in equation
- User draws on s field to place additional sources (including the image)
- Demonstrates interactive discovery of the correct image position

**Our approach**:
- Pre-compute analytical log potential with source AND image in initial phi
- s field contains Gaussian bumps at both source and image positions
- Equation: `laplace(phi) - strength * s`
- Demonstrates the complete solution, not interactive discovery

**Key differences**:
1. Visual PDE is interactive - user must find the correct image position
2. Our version shows the completed method of images solution
3. Our initial phi is near steady-state; Visual PDE relaxes from zero

**Boundary conditions**:
- Visual PDE: phi=neumann, s=dirichlet
- Ours: both neumann (per-axis BC limitation)

**Status**: Functionally correct for demonstrating method of images. Produces equivalent steady-state solutions. The per-species BC requirement (see heterogeneous-gierer-meinhardt) also applies here.


---

### `nonlinear-beam` - Fundamental Equation Structure Error
**Priority**: High
**Reference**: `reference/visual-pde/_nonlinear-physics/nonlinear-beams.md`, `presets.js` line 4073

**Issue 1: CRITICAL - Wrong equation structure**

The implementation incorrectly factors E outside the outer Laplacian.

**Our implementation** (`pde_sim/pdes/physics/nonlinear_beam.py:117`):
```python
du/dt = -(E_avg + E_var * tanh(laplace(u)/eps)) * laplace(laplace(u))
```
This is `-E(v) * laplace(v)` where `v = laplace(u)`.

**Correct equation** (Visual PDE markdown):
```
∂y/∂t = -∂²/∂x²(E(∂²y/∂x²) · ∂²y/∂x²)
```
This is `-laplace(E(v) * v)` where `v = laplace(u)`.

**Why this matters**: When E depends on v (which varies spatially), the product rule applies:
```
laplace(E*v) = E*laplace(v) + v*laplace(E) + 2*grad(E)·grad(v)
```
The extra terms are non-negligible and affect the physics significantly.

**Issue 2: Stiffness formula interpretation mismatch**

**Our doc/implementation**:
`E = E* + ΔE*(1+tanh(v/ε))/2`
- With E*=0.0001, ΔE=24: Range is 0.0001 to 24.0001

**Visual PDE preset** (`reactionStr_3`):
`w = E*((1+tanh(v/0.01))*Delta_E/2+1)*v`
implies `E_eff = E*(1 + (1+tanh(v/ε))*ΔE/2)`
- With E=0.0001, ΔE=24: Range is 0.0001 to 0.0025

The preset treats ΔE as a **multiplier** (stiffness varies ~25x), while our doc treats it as an **additive range** (stiffness varies ~240,000x). The markdown documentation formula matches ours, suggesting possible inconsistency in Visual PDE itself.

**Issue 3: 1D vs 2D dimensionality**
- Visual PDE: Strictly 1D (line plot visualization, dimension: "1")
- Our implementation: Attempts 2D via Laplacians

The correct 2D generalization of the beam equation is the **plate equation** (biharmonic), not simply replacing ∂²/∂x² with Laplacian.

**Issue 4: Time-dependent boundary conditions**

Visual PDE uses oscillating moment boundary condition:
```javascript
comboStr_1: "Left: Dirichlet = 0; Left: Neumann = 0.2*sin(2*t);"
comboStr_2: "Right: Dirichlet = 0; Right: Neumann = 0;"
```

Our framework doesn't support time-dependent BCs. The initial condition (`initCond_1: "0"`) relies entirely on boundary forcing to drive dynamics.

**Issue 5: Implementation approach - cross-diffusion vs direct**

Visual PDE avoids computing fourth-order derivatives by using cross-diffusion with algebraic species:
```
du/dt = -laplace(w)           # diffusionStr_1_3: "-1"
v = laplace(u)                # algebraic, diffusionStr_2_1: "1"
w = E(v)*v                    # algebraic, reactionStr_3
```

Our implementation tries to compute `laplace(laplace(u))` directly, which:
1. Is numerically less stable
2. Doesn't correctly implement the product E(v)*v inside the outer Laplacian

**Required changes**:
1. Rewrite as `MultiFieldPDEPreset` with auxiliary fields (v, w)
2. Use cross-diffusion structure to implement fourth-order equation
3. Decide on stiffness formula interpretation (preset vs markdown)
4. Consider restricting to 1D or implementing proper 2D plate equation
5. (Optional) Add support for time-dependent BCs

**Alternatively**: Mark this preset as "approximate" or "experimental" in docs if exact Visual PDE compatibility isn't required.

---

### `thermal-convection` - Per-Field Boundary Conditions Required
**Priority**: High
**Reference**: `reference/visual-pde/sim/scripts/RD/presets.js` line 1496

**Issue**: Visual PDE uses **per-species** boundary conditions with critical asymmetric BCs that drive the convection, while our config uses simplified per-axis BCs.

**Visual PDE thermalConvection**:
```javascript
comboStr_1: "Bottom: Dirichlet = 0; Left: Periodic; Right: Periodic; Top: Dirichlet = 0;"  // omega
comboStr_2: "Bottom: Dirichlet = 0; Left: Periodic; Right: Periodic; Top: Dirichlet = 0;"  // psi
comboStr_4: "Bottom: Neumann = T_b; Left: Periodic; Right: Periodic; Top: Dirichlet = 0;"  // b
```

Summary:
- omega: Dirichlet=0 top/bottom, Periodic left/right (vorticity vanishes at walls)
- psi: Dirichlet=0 top/bottom, Periodic left/right (no-slip walls)
- b: **Neumann=T_b bottom** (heat flux in), Dirichlet=0 top (cold), Periodic left/right

**Our config** (`configs/defaults/fluids/thermal_convection.yaml`):
```yaml
bc:
  x: periodic
  y: periodic
```
This applies periodic BCs to all fields - missing the critical bottom heat flux that drives convection.

**Impact**: The Neumann BC `∂b/∂y = T_b` at bottom boundary is essential for Rayleigh-Benard convection. Without it, there's no continuous heat source to drive instability. Our current config may still show convection from initial perturbations, but lacks sustained boundary forcing.

**Required changes** (same as heterogeneous-gierer-meinhardt):
1. Extend BC system to support per-species boundary conditions
2. Update config schema to allow:
   ```yaml
   bc:
     omega: {top: "dirichlet=0", bottom: "dirichlet=0", left: "periodic", right: "periodic"}
     psi: {top: "dirichlet=0", bottom: "dirichlet=0", left: "periodic", right: "periodic"}
     b: {top: "dirichlet=0", bottom: "neumann=T_b", left: "periodic", right: "periodic"}
   ```

**Current workaround**: Initial noise provides some perturbation, but convection will decay without sustained bottom heating.

---
