# Korteweg-de Vries (KdV) Equation

The foundational equation of soliton theory, describing weakly nonlinear dispersive waves in shallow water and many other physical systems.

## Description

The Korteweg-de Vries equation, derived by Diederik Korteweg and Gustav de Vries in 1895, is one of the most important equations in nonlinear wave theory. Originally developed to model shallow water waves in canals, it has since been found to describe a wide variety of physical phenomena including ion-acoustic waves in plasmas, internal waves in stratified fluids, and pressure waves in crystals.

The KdV equation is historically significant as the first equation shown to possess **soliton solutions** - localized wave packets that maintain their shape and speed during propagation and emerge unchanged from collisions with other solitons. This discovery by Zabusky and Kruskal in 1965 launched the modern field of soliton theory.

Key properties of the KdV equation:

- **Complete integrability**: Possesses an infinite number of conservation laws and can be solved exactly via the inverse scattering transform
- **Soliton solutions**: Supports stable, localized traveling wave solutions of the form sech^2
- **Amplitude-speed relationship**: Taller solitons travel faster (speed proportional to amplitude)
- **Elastic collisions**: Solitons pass through each other with only a phase shift
- **N-soliton solutions**: Exact solutions exist for any number of interacting solitons
- **Conservation**: Conserves mass, momentum, and energy (among infinitely many quantities)

The equation balances two competing effects:
1. **Nonlinear steepening** from the u*u_x term (larger amplitudes travel faster)
2. **Dispersion** from the u_xxx term (different wavelengths travel at different speeds)

This balance allows the soliton to maintain its shape indefinitely.

## Equations

The standard KdV equation:

$$\frac{\partial u}{\partial t} = -\frac{\partial^3 u}{\partial x^3} - 6u\frac{\partial u}{\partial x}$$

With optional biharmonic dissipation for numerical stability:

$$\frac{\partial u}{\partial t} = -\frac{\partial^3 u}{\partial x^3} - 6u\frac{\partial u}{\partial x} - b \nabla^4 u$$

Where:
- $u(x,t)$ is the wave amplitude
- $-u_{xxx}$ is the dispersive term (third-order derivative)
- $-6u \cdot u_x$ is the nonlinear advection term
- $b$ is a small dissipation coefficient (numerical stabilization)
- $\nabla^4 u$ is the biharmonic operator (fourth-order diffusion)

**Single soliton solution**:
$$u(x,t) = 2k^2 \, \text{sech}^2\left(k(x - 4k^2 t - x_0)\right)$$

where $k$ determines both the amplitude ($2k^2$) and speed ($4k^2$).

## Default Config

```yaml
preset: kdv
parameters:
  b: 0.0001

init:
  type: two-solitons
  params:
    k1: 0.6
    k2: 0.4

solver: rk4
backend: numba
adaptive: false
dt: 0.001
t_end: 60.0
resolution: 128

bc:
  x-: periodic
  x+: periodic
  y-: periodic
  y+: periodic

domain_size: 100
```

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| b | 0.0001 | 0 - 0.1 | Biharmonic dissipation coefficient |

### Initial Condition Parameters

#### Single soliton (type: soliton)
| Parameter | Default | Description |
|-----------|---------|-------------|
| k | 0.5 | Width parameter (amplitude = 2k^2, speed = 4k^2) |
| x0 | L/4 | Initial x-position of soliton center |

#### Two solitons (type: two-solitons)
| Parameter | Default | Description |
|-----------|---------|-------------|
| k1 | 0.6 | Width parameter for first (taller/faster) soliton |
| k2 | 0.4 | Width parameter for second (shorter/slower) soliton |
| x1 | L*0.15 | Initial position of first soliton |
| x2 | L*0.35 | Initial position of second soliton |

#### N-wave (type: n-wave)
| Parameter | Default | Description |
|-----------|---------|-------------|
| amplitude | 1.0 | Initial bump amplitude |
| width | 0.1 | Bump width (fraction of domain) |
| x0 | L*0.2 | Bump center position |

## Dynamics

### Soliton Behavior

1. **Single soliton**: Travels to the right at constant speed, wrapping around with periodic boundaries
2. **Two-soliton collision**: Faster (taller) soliton overtakes slower one; both emerge unchanged except for phase shift
3. **N-wave evolution**: Initial bump breaks into a train of solitons ordered by height

### Soliton Properties

| Amplitude | Width | Speed |
|-----------|-------|-------|
| $2k^2$ | $\propto 1/k$ | $4k^2$ |

- Taller solitons are narrower and faster
- Amplitude and speed are both proportional to $k^2$

### Historical Discovery

The soliton phenomenon was first observed by John Scott Russell in 1834 when he followed a "wave of translation" along a canal for over a mile. The KdV equation was derived 60 years later to explain this observation. The term "soliton" was coined by Zabusky and Kruskal in 1965 when they discovered the particle-like collision properties numerically.

### Comparison with Related Equations

| Equation | Dimensions | Integrability | Soliton Type |
|----------|------------|---------------|--------------|
| KdV | 1D | Complete | Line soliton |
| Zakharov-Kuznetsov | 2D | Near-integrable | Vortical |
| mKdV | 1D | Complete | Kink/antikink |
| Nonlinear Schrodinger | 1D | Complete | Envelope soliton |

### Numerical Considerations

The KdV equation with its third derivative requires:
- Sufficient resolution to resolve the soliton width
- Small time steps for stability with explicit schemes
- Periodic boundaries work well for soliton dynamics
- Small biharmonic dissipation (b > 0) helps reduce numerical artifacts

## References

- Korteweg, D.J. & de Vries, G. (1895). "On the change of form of long waves advancing in a rectangular canal" - Phil. Mag. 39:422
- Zabusky, N.J. & Kruskal, M.D. (1965). "Interaction of 'solitons' in a collisionless plasma and the recurrence of initial states" - Phys. Rev. Lett. 15:240
- Gardner, C.S., Greene, J.M., Kruskal, M.D. & Miura, R.M. (1967). "Method for solving the Korteweg-deVries equation" - Phys. Rev. Lett. 19:1095
- Drazin, P.G. & Johnson, R.S. (1989). "Solitons: An Introduction" - Cambridge University Press
- VisualPDE (2024). "KdV Equation" - visualpde.com
