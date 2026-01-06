# Schrodinger Equation

The Schrodinger equation is the fundamental equation of quantum mechanics, describing how the quantum state of a physical system evolves over time.

## Description

The Schrodinger equation, discovered by Erwin Schrodinger in 1925-1926 (Nobel Prize 1933), is the quantum mechanical counterpart of Newton's second law. While Newton's law predicts the classical trajectory of a particle, the Schrodinger equation predicts the evolution of the wave function - a complex-valued function whose squared magnitude gives the probability density of finding a particle at a given location.

### Fundamental Concepts

- **Wave function** $\psi(x,y,t)$: Complex-valued function encoding the quantum state
- **Probability density** $|\psi|^2$: Probability of finding particle at position $(x,y)$
- **Energy eigenstates**: Special solutions that oscillate in time but maintain constant probability density
- **Quantization**: Only discrete energy values are allowed (energy levels)
- **Superposition**: General states are combinations of eigenstates

### Physical Applications

- **Atomic physics**: Electron orbitals in atoms
- **Quantum chemistry**: Molecular bonding and structure
- **Solid state physics**: Electrons in crystals, semiconductors
- **Quantum optics**: Photon behavior, lasers
- **Quantum computing**: Qubit dynamics
- **Nanotechnology**: Quantum dots, quantum wells
- **Optoelectronics**: Quantum well lasers, photodetectors

### Key Phenomena

**Particle in a box**: The simplest quantum system - a particle confined between infinite potential walls. Energy is quantized: $E_n = n^2\pi^2\hbar^2/(2mL^2)$. Wave functions are standing sine waves. This model explains quantum confinement in nanostructures.

**Quantum tunneling**: Unlike classical particles, quantum particles can penetrate potential barriers they don't have enough energy to overcome. The probability decreases exponentially with barrier thickness and height.

**Wave packets**: Localized "particles" constructed from superpositions of plane waves. In a potential well, wave packets bounce back and forth, exhibiting particle-like behavior.

## Equations

### Time-Dependent Schrodinger Equation

$$i\hbar\frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m}\nabla^2 \psi + V(x,y)\psi$$

In dimensionless form (setting constants to 1):

$$i\frac{\partial \psi}{\partial t} = -D\nabla^2 \psi + V(x,y)\psi$$

where:
- $\psi(x, y, t) = u + iv$ is the complex wave function
- $D$ is related to the particle's mass and Planck's constant
- $V(x,y)$ is the potential energy function
- $i = \sqrt{-1}$ is the imaginary unit

### Numerical Reformulation (Real System)

Since the solver works with real numbers, separate real ($u$) and imaginary ($v$) parts:

$$\frac{\partial u}{\partial t} = -D\nabla^2 v + CD\nabla^2 u + V(x,y)v$$

$$\frac{\partial v}{\partial t} = D\nabla^2 u + CD\nabla^2 v - V(x,y)u$$

where:
- $u = \text{Re}(\psi)$, $v = \text{Im}(\psi)$
- $C$ is an artificial diffusion parameter for numerical stabilization
- For pure Schrodinger equation, $C = 0$

Implementation:
- Reaction terms: `u: V*v` (or `0` if no potential), `v: -V*u` (or `0`)
- Cross-diffusion: `u-v: -D`, `v-u: D` (couples real and imaginary parts)
- Self-diffusion: `u-u: C*D`, `v-v: C*D` (stabilization)

### Probability Density

$$|\psi|^2 = u^2 + v^2$$

This is plotted by default and should remain approximately constant over time for eigenstates.

## Default Config

### Basic Schrodinger (stabilizedSchrodingerEquation)
```yaml
solver: midpoint
dt: 0.0001
dx: 0.5
domain_size: 100
timestepping_scheme: Mid

boundary_x: dirichlet
boundary_y: dirichlet

parameters:
  D: 1.0       # quantum mechanical parameter
  C: 0.004     # numerical stabilization
  n: 3         # range: [0, 10] - x wave number for initial eigenstate
  m: 3         # range: [0, 10] - y wave number for initial eigenstate

initial_condition:
  u: sin(n*pi*x/100)*sin(m*pi*y/100)
  v: 0

species:
  - name: u   # Re(psi)
  - name: v   # Im(psi)
```

## Parameter Variants

### stabilizedSchrodingerEquation
2D particle in a box (zero potential, infinite walls):
- Dirichlet boundaries ($\psi = 0$ at walls) create an infinite square well
- Initial condition: eigenstate $\psi = \sin(n\pi x/L)\sin(m\pi y/L)$
- Parameters `n`, `m` select which energy eigenstate to visualize
- Higher $n$, $m$ correspond to higher energy (faster oscillation)
- Probability density $|\psi|^2$ remains approximately stationary
- Views: $|\psi|^2$ (default), $\text{Re}(\psi)$, $\text{Im}(\psi)$
- Brush disabled (clicking would destroy eigenstate)

### stabilizedSchrodingerEquationPotential
2D Schrodinger with sinusoidal potential:
- Potential: $V(x,y) = \sin(n\pi x/L)\sin(m\pi y/L)$
- Initial condition: localized Gaussian-like packet $(\sin(\pi x/L)\sin(\pi y/L))^{10}$
- Parameters `n = 15`, `m = 15` control potential landscape
- Demonstrates quantum tunneling between potential wells
- Auto-scaling color range to track amplitude changes
- Reaction terms implement potential coupling

### stabilizedSchrodinger1D
1D Gaussian wave packet in quadratic potential (harmonic oscillator):
- Initial condition: Gaussian wave packet with momentum
- Potential: $V = p(x/L - 0.5)^2$ (parabolic, centered)
- Wave packet bounces like a classical particle for short times
- Parameters: `a = 4` (momentum), `s = 0.08` (width), `x0 = 0.5` (position)
- Overlay shows potential profile
- Total probability displayed (should be conserved)
- Uses algebraic species for potential $V$

### quantumTunneling
1D quantum tunneling through a barrier:
- Gaussian wave packet approaching a potential barrier
- Potential: smooth step function at domain center
- Part of wave packet transmits, part reflects
- Demonstrates classically forbidden penetration
- Parameters: `p = 15` (barrier height), `x0 = 0.2` (initial position)

## Numerical Notes

- The artificial diffusion parameter $C$ stabilizes the simulation but causes probability to slowly decay
- Smaller $C$ is more accurate but may develop oscillations
- Midpoint timestepping scheme improves conservation
- Probability is not exactly conserved by this explicit scheme
- Grid-level dispersion effects appear when wave packet interacts with barriers

## References

- [Schrodinger equation - Wikipedia](https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation)
- [Particle in a box - Wikipedia](https://en.wikipedia.org/wiki/Particle_in_a_box)
- [Particle in a Box - MIT OCW](https://ocw.mit.edu/courses/6-007-electromagnetic-energy-from-motors-to-lasers-spring-2011/187a737ba49baa8f397b80449a15e45a_MIT6_007S11_lec40.pdf)
- [The Schrodinger Equation and a Particle in a Box - Chemistry LibreTexts](https://chem.libretexts.org/Courses/Grinnell_College/CHM_364:_Physical_Chemistry_2_(Grinnell_College)/03:_The_Schrodinger_Equation_and_a_Particle_in_a_Box)
- [The Quantum Particle in a Box - OpenStax](https://openstax.org/books/university-physics-volume-3/pages/7-4-the-quantum-particle-in-a-box)
