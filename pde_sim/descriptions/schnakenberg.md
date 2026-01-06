# Schnakenberg Model

A two-component reaction-diffusion system that forms Turing patterns through activator-inhibitor dynamics, producing spots and stripes.

## Description

The Schnakenberg model is a classical reaction-diffusion system introduced by Juergen Schnakenberg in 1979 as a minimal, chemically sensible model exhibiting limit-cycle behaviour. The model has become a prototype for studying Turing instability in mathematical biology and pattern formation theory.

The system describes the interaction between an activator (u) and an inhibitor (v), where the activator undergoes autocatalytic production modulated by the inhibitor. When the diffusion coefficient of the inhibitor is sufficiently larger than that of the activator (D > 1), the system can exhibit diffusion-driven instability (Turing instability), leading to spontaneous pattern formation from an initially homogeneous state.

Key phenomena include:
- **Spot patterns**: Form when D is large (e.g., D = 100)
- **Stripe patterns**: Form when D is reduced (e.g., D = 30)
- **Hopf bifurcations**: The homogeneous equilibrium can undergo Hopf bifurcations for small values of 1 > b > a >= 0, leading to oscillations and complex spatiotemporal interactions between Turing and Hopf instabilities

The Schnakenberg model has applications in understanding biological morphogenesis, chemical pattern formation, and serves as a testbed for mathematical analysis of pattern-forming systems.

## Equations

$$\frac{\partial u}{\partial t} = \nabla^2 u + a - u + u^2 v$$

$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + b - u^2 v$$

Where:
- $u$ is the activator concentration
- $v$ is the inhibitor concentration
- $a, b > 0$ are production rate parameters
- $D_v > 1$ is required for pattern formation

## Default Config

```yaml
solver: euler
dt: 0.0005
dx: 0.5
domain_size: 100

boundary_x: periodic
boundary_y: periodic

parameters:
  D_v: 100  # range: [0, 100]
  a: 0.01
  b: 2
```

## Parameter Variants

### Schnakenberg (Standard)
The default configuration produces spot-like patterns with high diffusion ratio.
- `D_v = 100`: Large diffusion ratio favors spots
- `a = 0.01, b = 2`: Standard kinetic parameters

### SchnakenbergHopf
Configuration near the Hopf bifurcation boundary where Turing patterns, homogeneous oscillations, and complex spatiotemporal behaviors can coexist.
- `D_v = 8`: Reduced diffusion ratio
- `a = 0.05` (range: [0, 0.1])
- `b = 0.5` (range: [0, 1])
- Parameters chosen to be in the regime where 1 > b > a >= 0

## References

- Schnakenberg, J. (1979). Simple chemical reaction systems with limit cycle behaviour. Journal of Theoretical Biology, 81(3), 389-400.
- Turing, A. M. (1952). The chemical basis of morphogenesis. Philosophical Transactions of the Royal Society B, 237(641), 37-72.
