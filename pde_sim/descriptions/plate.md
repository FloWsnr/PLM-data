# Plate Equation (Linearly Elastic Models)

The plate equation (and related beam equation) describes the vibrations and deformations of thin elastic structures, involving the fourth-order biharmonic operator.

## Description

The plate equation is derived from Kirchhoff-Love plate theory, which extends Euler-Bernoulli beam theory to two-dimensional surfaces. These fourth-order PDEs govern the transverse deflections of thin elastic structures under loading.

### Physical Applications

- **Structural engineering**: Bridge decks, floor slabs, aircraft panels
- **Musical instruments**: Vibrating plates (cymbals, gongs, bells), piano soundboards
- **Microelectromechanical systems (MEMS)**: Micro-mirrors, pressure sensors
- **Aerospace**: Wing panels, fuselage sections under aerodynamic loads
- **Naval architecture**: Ship hull plating
- **Robotics**: Flexible manipulator arms
- **Energy**: Turbine blades, rotor dynamics

### Euler-Bernoulli Beam Theory

The 1D beam equation describes transverse vibrations of slender beams:
- Assumes cross-sections remain plane and perpendicular to the neutral axis
- Fourth-order spatial derivative captures bending stiffness
- Used for helicopter rotors, spacecraft structures, flexible robotic arms

### Kirchhoff-Love Plate Theory

The 2D plate equation extends beam theory to thin plates:
- Developed by Kirchhoff (1850) and Love (1888)
- Accounts for anticlastic curvature (saddle-like bending)
- Uses the biharmonic operator $\nabla^4 = \nabla^2(\nabla^2)$
- Assumes plate thickness is small compared to other dimensions

### Key Phenomena

- **Bending waves**: Fourth-order equations produce dispersive waves (speed depends on frequency)
- **Standing modes**: Characteristic patterns (Chladni figures on vibrating plates)
- **Buckling**: Plates under compression can suddenly buckle
- **Damping**: Energy loss through internal friction

## Equations

### Beam Equation (1D)

$$\frac{\partial^2 u}{\partial t^2} + C\frac{\partial u}{\partial t} = -D^2 \frac{\partial^4 u}{\partial x^4} - Q$$

### Plate Equation (2D)

$$\frac{\partial^2 u}{\partial t^2} + C\frac{\partial u}{\partial t} = -D^2 \nabla^4 u - Q$$

where:
- $u(x, y, t)$ is the transverse displacement (deflection)
- $D$ is related to bending rigidity (combines material stiffness and geometry)
- $C$ is the damping coefficient
- $Q$ is a constant load (gravity-like force)
- $\nabla^4 = \nabla^2(\nabla^2) = \frac{\partial^4}{\partial x^4} + 2\frac{\partial^4}{\partial x^2 \partial y^2} + \frac{\partial^4}{\partial y^4}$ is the biharmonic operator

### Boundary Conditions

**Fixed (clamped) edges:**
- $u = 0$ and $\nabla^2 u = 0$ on boundary

**Free edges (Neumann for all fields):**
- $\frac{\partial^2 u}{\partial n^2} = 0$ and $\frac{\partial^3 u}{\partial n^3} = 0$ on boundary

### Numerical Reformulation

Since VisualPDE handles only first-order time derivatives and second-order spatial derivatives, the plate equation is reformulated using three fields:

$$\frac{\partial u}{\partial t} = v + D D_c \nabla^2 u$$

$$\frac{\partial v}{\partial t} = -D \nabla^2 w - Cv - Q$$

$$w = D \nabla^2 u$$

where:
- $v$ is the velocity field
- $w$ is an auxiliary field representing $D \nabla^2 u$
- $D_c$ is a numerical stabilization parameter (set to 0 for pure plate equation)

Implementation:
- Reaction terms: `u: v`, `v: -Q - C*v`, `w: 0`
- Diffusion: `u-u: D_c*D`, `v-w: -D`, `w-u: D`
- The `w` field is algebraic (computed from $u$, not evolved in time)

## Default Config

### Plate Equation (plateEquation)
```yaml
solver: euler
dt: 0.0001
dx: 0.5
domain_size: 100

boundary_x: dirichlet
boundary_y: dirichlet

parameters:
  D: 10       # bending rigidity parameter
  Q: 0.003    # constant load (gravity)
  C: 0.1      # damping coefficient
  D_c: 0.1    # numerical stabilization

initial_condition: u = -4  # initial deformation

species:
  - name: u   # displacement
  - name: v   # velocity
  - name: w   # auxiliary (algebraic)
```

### Beam Equation (BeamEquation)
```yaml
solver: euler
dt: 0.0001
dimension: 1

boundary_x: dirichlet

parameters:
  D: 10
  Q: 0.0      # no gravity by default
  C: 0        # no damping
  D_c: 0.1

species:
  - name: u
  - name: v
  - name: w
```

## Parameter Variants

### BeamEquation
1D beam equation:
- One-dimensional (line plot)
- No gravity ($Q = 0$), no damping ($C = 0$)
- Fixed (Dirichlet) boundary conditions at both ends
- Click to push down on beam, creating traveling ripples
- Change to Neumann boundaries for "free end" conditions

### plateEquation
2D plate equation:
- Fixed boundaries ($u = 0$ at edges)
- Initial deformation $u = -4$ everywhere
- Instantaneous boundary condition creates inward-propagating compression waves
- Parameters: $D = 10$, $Q = 0.003$, $C = 0.1$
- Square domain enforced

### plateEquation3D
Same as plateEquation with 3D surface visualization:
- Rotatable 3D view of plate deformation
- No initial deformation ($u = 0$)
- No gravity ($Q = 0$)
- Interactive clicking to create local depressions

## References

- [Euler-Bernoulli beam theory - Wikipedia](https://en.wikipedia.org/wiki/Euler–Bernoulli_beam_theory)
- [Kirchhoff-Love plate theory - Wikipedia](https://en.wikipedia.org/wiki/Kirchhoff–Love_plate_theory)
- [Euler-Bernoulli Beams: Bending, Buckling, and Vibration - MIT OCW](https://ocw.mit.edu/courses/2-002-mechanics-and-materials-ii-spring-2004/bc25a56b5a91ad29ca5c7419616686f7_lec2.pdf)
- [Bernoulli-Euler Beams - enDAQ](https://endaq.com/pages/bernoulli-euler-beams)
