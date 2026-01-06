# FitzHugh-Nagumo Model

A simplified model of excitable systems that describes action potentials in neurons, exhibiting patterns, spiral waves, and chaotic dynamics.

## Description

The FitzHugh-Nagumo (FHN) model is a two-dimensional simplification of the four-dimensional Hodgkin-Huxley model of neuronal action potentials. Developed independently by Richard FitzHugh (1961) and Jin-ichi Nagumo (1962), it captures the essential dynamics of excitable systems while remaining mathematically tractable.

The model describes:
- **Voltage variable (u)**: Represents the membrane potential (fast variable)
- **Recovery variable (v)**: Represents the slow recovery processes (slow variable)

The FHN model exhibits several key behaviors:
- **Excitability**: Small perturbations decay, but perturbations above a threshold trigger large excursions (action potentials)
- **Oscillations**: For certain parameters, sustained periodic oscillations occur
- **Pattern formation**: In spatially extended systems, Turing-like patterns can emerge
- **Spiral waves**: Rotating spiral waves that model cardiac arrhythmias and other biological phenomena
- **Traveling waves**: Action potential propagation along neurons

Applications extend beyond neuroscience to cardiac physiology (modeling arrhythmias), cell division, population dynamics, and electronics. The model has been central to the development of nonlinear dynamics and continues to yield new discoveries.

## Equations

### Standard Form
$$\frac{\partial u}{\partial t} = \nabla^2 u + u - u^3 - v$$

$$\frac{\partial v}{\partial t} = D \nabla^2 v + \varepsilon_v (u - a_v v - a_z)$$

### Alternative Parametrization (Action Potential Form)
$$\frac{\partial v}{\partial t} = D \nabla^2 v + v(v-a)(1-v) - w$$

$$\frac{\partial w}{\partial t} = \varepsilon (\gamma v - w)$$

Where:
- $u$ (or $v$ in alternative form) represents membrane voltage
- $v$ (or $w$) represents the recovery variable
- $D > 1$ for pattern formation
- $\varepsilon_v$ controls the timescale separation

## Default Config

```yaml
solver: euler
dt: 0.001
dx: 1.4
domain_size: 280

boundary_x: periodic
boundary_y: periodic

parameters:
  e_v: 0.5    # recovery timescale
  a_v: 1      # recovery slope
  a_z: -0.1   # recovery offset
  D_v: 20     # inhibitor diffusion
```

## Parameter Variants

### FitzHugh-Nagumo (Standard)
Base configuration producing concentric ring patterns and excitable dynamics.
- `D_v = 20`: Diffusion coefficient for v
- `e_v = 0.5`, `a_v = 1`, `a_z = -0.1`

### FitzHugh-Nagumo-Hopf
Configuration supporting both pattern formation and oscillations (Turing-Hopf regime).
- `D_v = 26`: Modified diffusion
- `e_v = 0.2`, `a_v = 0.01`, `a_z = -0.1`
- `m = 4` (range: [3, 6]): Initial condition wavenumber
- Exhibits spatial, temporal, and spatiotemporal behaviors depending on initial conditions

### FitzHugh-Nagumo-3
Three-species variant with competing oscillations and pattern formation.
- Additional species $w$ with its own dynamics
- `D_v = 40`, `D_w = 200`
- `a_v = 0.2` (range: [0, 0.5]): Controls pattern vs oscillation dominance
- `e_v = 0.2`, `e_w = 1`, `a_w = 0.5`, `a_z = -0.1`

$$\frac{\partial w}{\partial t} = D_w \nabla^2 w + \varepsilon_w (u - w)$$

## References

- FitzHugh, R. (1961). Impulses and physiological states in theoretical models of nerve membrane. Biophysical Journal, 1(6), 445-466.
- Nagumo, J., Arimoto, S., & Yoshizawa, S. (1962). An active pulse transmission line simulating nerve axon. Proceedings of the IRE, 50(10), 2061-2070.
- Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. Journal of Physiology, 117(4), 500-544.
