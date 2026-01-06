# FitzHugh-Nagumo Model

## Mathematical Formulation

The FitzHugh-Nagumo equations describe excitable and pattern-forming media:

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u + u - u^3 - v$$
$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + \varepsilon(u - a_v v - a_z)$$

where:
- $u$ is the fast activator (membrane potential analog)
- $v$ is the slow recovery variable
- $\varepsilon \ll 1$ controls timescale separation
- $a_v, a_z$ are parameters controlling the v-nullcline
- $D_u, D_v$ are diffusion coefficients (for Turing patterns, typically $D_v > D_u$)

This is the pattern-forming version from visualpde.com, with the full $u^3$ cubic nonlinearity.

## Physical Background

The FitzHugh-Nagumo model is a simplified version of the Hodgkin-Huxley equations for nerve impulse propagation. It captures the essential dynamics of excitable systems:

1. **Excitation**: Rapid transition from rest to excited state
2. **Recovery**: Slow return to rest state
3. **Refractory period**: Cannot be re-excited immediately

## Nullclines and Phase Portrait

The $u$-nullcline (cubic): $v = u - u^3$
The $v$-nullcline (linear): $v = (u - a_z)/a_v$

The intersection determines the steady state. System behavior depends on the number and stability of intersections.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Timescale ratio | $\varepsilon$ | Separation of fast/slow | 0.001 - 0.5 |
| Parameter a_v | $a_v$ | Coefficient of v in recovery | 0 - 2 |
| Parameter a_z | $a_z$ | Constant in recovery | -1 - 1 |
| Diffusion u | $D_u$ | Activator diffusion | 0.05 - 5 |
| Diffusion v | $D_v$ | Inhibitor diffusion | 0 - 100 |

## Turing Patterns

For Turing pattern formation, typically $D_v > D_u$ (diffusion-driven instability). The patterns emerge when the inhibitor diffuses faster than the activator.

## Spatiotemporal Patterns

1. **Traveling pulses**: Action potential propagation along nerves
2. **Spiral waves**: Rotating waves in 2D (cardiac tissue)
3. **Target patterns**: Concentric rings from pacemaker sources
4. **Scroll waves**: 3D spiral analogs
5. **Turbulence**: Breakup of spirals into chaotic activity

## Applications

1. **Cardiac electrophysiology**: Heart rhythm and arrhythmias
2. **Neuroscience**: Nerve impulse propagation
3. **Chemical systems**: BZ reaction dynamics
4. **Cell signaling**: Calcium waves
5. **Ecological systems**: Predator-prey waves

## Numerical Considerations

- Explicit schemes work well with proper CFL condition
- Spiral wave simulations require fine resolution
- Long-time integration for spiral dynamics

## References

- FitzHugh, R. (1961). *Impulses and Physiological States in Models of Nerve Membrane*
- Nagumo, J. et al. (1962). *An Active Pulse Transmission Line*
- Winfree, A.T. (1980). *The Geometry of Biological Time*
