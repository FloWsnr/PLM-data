# Fokker-Planck Equation

The Fokker-Planck equation describes the time evolution of a probability density function under drift and diffusion, providing the fundamental link between stochastic differential equations and deterministic probability evolution.

## Description

The Fokker-Planck equation (also known as the Kolmogorov forward equation) was developed independently by Adriaan Fokker (1914) and Max Planck (1917). It describes how the probability density of a stochastic process evolves over time, making it a cornerstone of statistical physics, chemical kinetics, and mathematical finance.

The equation arises naturally from considering the evolution of an ensemble of particles undergoing:
- **Drift**: Deterministic motion driven by forces (e.g., from a potential energy landscape)
- **Diffusion**: Random motion due to thermal fluctuations or other noise sources

Key physical phenomena:
- **Relaxation to equilibrium**: An initial probability distribution evolves toward a stationary (equilibrium) distribution
- **Probability conservation**: The total probability (integral of p over all space) remains constant
- **Detailed balance**: At equilibrium, probability flux vanishes everywhere
- **Fluctuation-dissipation relation**: The ratio D/gamma determines the equilibrium distribution width

For a harmonic potential (Ornstein-Uhlenbeck process), the equilibrium distribution is Gaussian with variance proportional to D/gamma. This represents a particle in a parabolic potential well with thermal fluctuations.

The Fokker-Planck equation is the dual (adjoint) of the backward Kolmogorov equation and is equivalent to the Langevin equation description via the Ito-Stratonovich calculus.

## Equations

**General Fokker-Planck equation**:

$$\frac{\partial p}{\partial t} = -\nabla \cdot (\boldsymbol{\mu} \, p) + \nabla \cdot (D \nabla p)$$

where:
- $p(\mathbf{x}, t)$ is the probability density function
- $\boldsymbol{\mu}(\mathbf{x})$ is the drift velocity field
- $D$ is the diffusion coefficient (assumed constant here)

**2D form with harmonic potential drift**:

For a harmonic potential $V(x,y) = \frac{\gamma}{2}[(x-x_0)^2 + (y-y_0)^2]$, the drift is $\boldsymbol{\mu} = -\nabla V = -\gamma(x-x_0, y-y_0)$:

$$\frac{\partial p}{\partial t} = \gamma \frac{\partial}{\partial x}[(x-x_0) p] + \gamma \frac{\partial}{\partial y}[(y-y_0) p] + D \nabla^2 p$$

Expanding the drift terms using the product rule:

$$\frac{\partial p}{\partial t} = 2\gamma p + \gamma(x-x_0)\frac{\partial p}{\partial x} + \gamma(y-y_0)\frac{\partial p}{\partial y} + D \nabla^2 p$$

**Equilibrium solution** (stationary state for harmonic potential):

$$p_{eq}(x,y) = \frac{\gamma}{2\pi D} \exp\left(-\frac{\gamma}{2D}[(x-x_0)^2 + (y-y_0)^2]\right)$$

This is a Gaussian centered at $(x_0, y_0)$ with variance $\sigma^2 = D/\gamma$.

## Connection to Stochastic Processes

The Fokker-Planck equation is the probability-space dual of the Langevin equation:

$$d\mathbf{X} = \boldsymbol{\mu}(\mathbf{X}) \, dt + \sqrt{2D} \, d\mathbf{W}$$

where $\mathbf{W}$ is a Wiener process (Brownian motion). A particle following this SDE has its probability density evolve according to the Fokker-Planck equation.

| Langevin (single particle) | Fokker-Planck (probability) |
|---------------------------|----------------------------|
| Position $\mathbf{X}(t)$ | Density $p(\mathbf{x}, t)$ |
| Drift force $\boldsymbol{\mu}$ | Probability current |
| Noise $\sqrt{2D} \, d\mathbf{W}$ | Diffusion term $D\nabla^2 p$ |
| Sample paths | Ensemble evolution |

## Default Config

```yaml
preset: fokker-planck

solver: euler
dt: 0.001
resolution: 128
domain_size: 10.0

bc:
  x-: neumann:0
  x+: neumann:0
  y-: neumann:0
  y+: neumann:0

init:
  type: default
  params:
    sigma: 0.15
    init_x_offset: 0.3
    init_y_offset: 0.3

parameters:
  D: 0.1       # Diffusion coefficient
  gamma: 0.5   # Drift strength (harmonic potential)
  x0: 0.0      # Potential center x (relative to domain center)
  y0: 0.0      # Potential center y (relative to domain center)
```

## Parameter Variants

### Relaxation to Equilibrium (Default)
Demonstrates a probability distribution relaxing to Gaussian equilibrium:
- Initial Gaussian offset from potential center
- `gamma = 0.5`, `D = 0.1` give moderate relaxation timescale
- Equilibrium variance: $\sigma_{eq}^2 = D/\gamma = 0.2$
- Characteristic relaxation time: $\tau = 1/\gamma = 2$

### Strong Drift
Fast relaxation with narrow equilibrium:
- `gamma = 2.0` (strong drift toward center)
- `D = 0.1` (same diffusion)
- Narrower equilibrium distribution
- Faster relaxation to equilibrium

### Weak Drift / High Diffusion
Slow relaxation with broad equilibrium:
- `gamma = 0.1` (weak drift)
- `D = 0.5` (high diffusion)
- Broad equilibrium distribution
- Slow convergence, strong spreading

### Off-Center Potential
Potential well not at domain center:
- `x0 = 0.3`, `y0 = -0.2` (offset potential center)
- Shows probability drifting toward off-center equilibrium

### Ring Collapse
Ring-shaped initial distribution collapsing inward:
- Initial condition: `gaussian-ring` with `radius = 0.4`
- `gamma = 1.0` (moderate drift)
- Ring contracts and fills to become centered Gaussian

### Double Peak Merger
Two-peak distribution merging:
- Initial condition: `double-gaussian` with `separation = 0.4`
- Both peaks drift toward center and merge
- Final state: single Gaussian at potential minimum

## Physical Interpretation

| Parameter | Physical Meaning |
|-----------|------------------|
| $p$ | Probability density (probability per unit area) |
| $D$ | Diffusion coefficient (thermal noise strength) |
| $\gamma$ | Drift rate / friction coefficient |
| $D/\gamma$ | Equilibrium distribution variance |
| $1/\gamma$ | Characteristic relaxation time |

### Probability Conservation

The Fokker-Planck equation conserves total probability:

$$\frac{d}{dt} \int p \, dA = 0$$

This is automatically satisfied for zero-flux (Neumann) boundary conditions, which represent a closed system with no probability leaking out.

### Fluctuation-Dissipation Relation

The ratio $D/\gamma$ connects noise strength to equilibrium fluctuations. In physical systems at temperature $T$:

$$D = \frac{k_B T}{\eta}$$

where $\eta$ is the friction coefficient, giving the Einstein relation.

## Applications

1. **Brownian motion**: Particle diffusion in viscous media
2. **Financial models**: Option pricing (Black-Scholes is a special case)
3. **Chemical kinetics**: Reaction coordinate dynamics
4. **Neuroscience**: Firing rate models, neural population dynamics
5. **Plasma physics**: Velocity distribution evolution
6. **Machine learning**: Diffusion models for generative AI

## References

- Fokker, A.D. (1914). "Die mittlere Energie rotierender elektrischer Dipole im Strahlungsfeld" - Ann. Physik 43:810
- Planck, M. (1917). "Uber einen Satz der statistischen Dynamik" - Sitzungsber. Preuss. Akad. Wiss.
- Risken, H. (1996). "The Fokker-Planck Equation: Methods of Solution and Applications" - Springer
- Gardiner, C.W. (2009). "Stochastic Methods: A Handbook for the Natural and Social Sciences" - Springer
- Uhlenbeck, G.E. & Ornstein, L.S. (1930). "On the theory of Brownian motion" - Phys. Rev. 36:823
