# Duffing Oscillator (Diffusively Coupled)

A spatially extended forced nonlinear oscillator exhibiting chaos, jump phenomena, and hysteresis through the interplay of cubic stiffness and periodic forcing.

## Description

The Duffing equation, named after German engineer Georg Duffing (1918), describes a nonlinear oscillator with cubic stiffness - a spring that becomes harder or softer with displacement. It is one of the simplest systems that exhibits **chaotic behavior** under periodic forcing.

Physical systems modeled by the Duffing equation include:
- **Mechanical oscillators**: Beams, plates with geometric nonlinearity
- **Electrical circuits**: Inductor with saturating ferrite core
- **Magnetic pendulum**: Steel beam between two magnets (double-well potential)
- **MEMS devices**: Nonlinear microresonators

Key phenomena exhibited by the Duffing oscillator:
- **Hardening/softening frequency response**: The resonance peak bends depending on $\alpha$ sign
- **Jump phenomena and hysteresis**: Discontinuous amplitude changes when sweeping frequency
- **Period doubling to chaos**: Route to chaos through successive bifurcations
- **Strange attractors**: Complex fractal structure in phase space
- **Coexisting solutions**: Multiple stable states for the same parameters

The spatially extended version couples Duffing oscillators through diffusion, creating a system where:
- Local chaotic dynamics interact with spatial coupling
- Pattern formation emerges from the interplay of forcing and diffusion
- Spatiotemporal chaos can develop

## Equations

The diffusively coupled Duffing oscillator:

$$\frac{\partial X}{\partial t} = \varepsilon D \nabla^2 X + Y$$

$$\frac{\partial Y}{\partial t} = D \nabla^2 X - \delta Y - \alpha X - \beta X^3 + \gamma \cos(\omega t)$$

Where:
- $X$ is the oscillator displacement
- $Y = \dot{X}$ is the velocity
- $\alpha$ is the linear stiffness (can be negative for double-well potential)
- $\beta$ controls the cubic nonlinearity strength
- $\delta$ is the linear damping coefficient
- $\gamma$ is the forcing amplitude
- $\omega$ is the forcing frequency
- $D$ is the diffusion coupling strength
- $\varepsilon$ is artificial diffusion for numerical stability

**Spring classification**:
- $\beta > 0$, $\alpha > 0$: **Hardening spring** - stiffness increases with displacement
- $\beta > 0$, $\alpha < 0$: **Double-well potential** - bistable, snapping behavior
- $\beta < 0$, $\alpha > 0$: **Softening spring** - stiffness decreases with displacement

**Restoring force**: $F = -\alpha X - \beta X^3$

## Default Config

```yaml
solver: euler
dt: 0.0015
dx: 0.1
domain_size: 50

boundary_x: periodic
boundary_y: periodic

parameters:
  D: 0.5         # diffusion coupling
  alpha: -1.0    # [-5, -1] - linear stiffness (negative = double-well)
  beta: 0.25     # cubic stiffness
  delta: 0.1     # linear damping
  gamma: 2       # forcing amplitude
  omega: 2       # forcing frequency
  epsilon: 0.01  # artificial diffusion (numerical stability)
```

## Parameter Variants

### Duffing (Standard)
Double-well forced Duffing oscillator:
- `alpha = -1.0` (double-well potential)
- `beta = 0.25`, `delta = 0.1`
- `gamma = 2`, `omega = 2` (strong forcing)
- `D = 0.5`, `epsilon = 0.01`
- Initial condition: Random $X = \text{RANDN}$
- Embossed ice colormap for visualization

### Potential Types

| alpha | beta | Potential | Behavior |
|-------|------|-----------|----------|
| alpha > 0 | beta > 0 | Single well, hardening | Resonance bends right |
| alpha > 0 | beta < 0 | Single well, softening | Resonance bends left |
| alpha < 0 | beta > 0 | Double well | Bistability, chaos |

### Chaos and Bifurcations

The Duffing system can exhibit:
1. **Periodic motion**: Simple oscillation at forcing frequency
2. **Subharmonics**: Period-2, period-4, ... oscillations
3. **Quasiperiodic**: Two incommensurate frequencies
4. **Chaos**: Aperiodic, bounded, sensitive to initial conditions

The route to chaos typically involves:
- Period-doubling cascade (most common)
- Intermittency
- Crisis (sudden change in attractor size)

### Exploring Dynamics

- **Vary alpha** (in [-5, -1]): Changes potential well depth
- **Smaller damping**: Promotes more complex dynamics
- **Change forcing**: Different $\gamma$ and $\omega$ select different regimes
- **Click perturbations**: Test sensitivity to perturbations

### Poincare Section
For detailed analysis, sample $(X, Y)$ once per forcing period $T = 2\pi/\omega$. Regular orbits produce fixed points or simple curves; chaos produces fractal strange attractors.

## References

- Duffing, G. (1918). "Erzwungene Schwingungen bei veranderlicher Eigenfrequenz"
- Guckenheimer, J. & Holmes, P. (1983). "Nonlinear Oscillations, Dynamical Systems, and Bifurcations of Vector Fields" - Springer
- Moon, F.C. (2004). "Chaotic Vibrations: An Introduction for Applied Scientists and Engineers" - Wiley
- Strogatz, S.H. (2015). "Nonlinear Dynamics and Chaos" - Westview Press
