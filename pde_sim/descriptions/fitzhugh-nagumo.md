# FitzHugh-Nagumo Model

## Mathematical Formulation

The FitzHugh-Nagumo equations describe excitable media:

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u + u - \frac{u^3}{3} - v$$
$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + \varepsilon(u + a - bv)$$

where:
- $u$ is the fast activator (membrane potential analog)
- $v$ is the slow recovery variable
- $\varepsilon \ll 1$ controls timescale separation
- $a, b$ are shape parameters
- $D_u$ is activator diffusion (often $D_v = 0$)

## Physical Background

The FitzHugh-Nagumo model is a simplified version of the Hodgkin-Huxley equations for nerve impulse propagation. It captures the essential dynamics of excitable systems:

1. **Excitation**: Rapid transition from rest to excited state
2. **Recovery**: Slow return to rest state
3. **Refractory period**: Cannot be re-excited immediately

## Nullclines and Phase Portrait

The $u$-nullcline (cubic): $v = u - u^3/3$
The $v$-nullcline (linear): $v = (u + a)/b$

The intersection determines the steady state. System behavior depends on the number and stability of intersections.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Timescale ratio | $\varepsilon$ | Separation of fast/slow | 0.001 - 1 |
| Parameter a | $a$ | Shifts v-nullcline | 0 - 2 |
| Parameter b | $b$ | Slope of v-nullcline | 0 - 2 |
| Diffusion u | $D_u$ | Activator diffusion | 0.1 - 1.0 |
| Diffusion v | $D_v$ | Recovery diffusion (often 0) | 0.0 - 0.5 |

## Excitable vs Oscillatory Regime

- **Excitable** ($a > 1 - b/3$): Single stable rest state, pulse propagation
- **Oscillatory** ($a < 1 - b/3$): Unstable rest state, sustained oscillations

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
