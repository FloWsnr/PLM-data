# Wave Equation

## Mathematical Formulation

The wave equation describes the propagation of waves:

$$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u$$

For numerical simulation, we convert this second-order equation to a first-order system:

$$\frac{\partial u}{\partial t} = v$$
$$\frac{\partial v}{\partial t} = c^2 \nabla^2 u$$

where:
- $u$ is the displacement (wave amplitude)
- $v = \partial u/\partial t$ is the velocity
- $c$ is the wave propagation speed

## Physical Background

The wave equation was first studied by d'Alembert in 1747 for vibrating strings. Key properties:

- **Hyperbolic PDE**: Information propagates at finite speed $c$
- **Energy conservation**: Total energy (kinetic + potential) is conserved
- **Superposition**: Solutions can be superposed due to linearity
- **Characteristic lines**: Solutions propagate along $x \pm ct = \text{const}$

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Wave speed | $c$ | Propagation velocity | 0.01 - 10 |

## Wave Phenomena

1. **Reflection**: Waves bounce off boundaries
2. **Interference**: Constructive and destructive patterns
3. **Standing waves**: Resonant modes in bounded domains
4. **Dispersion**: Not present in linear wave equation (all frequencies travel at same speed)

## Applications

1. **Acoustics**: Sound propagation in air and solids
2. **Electromagnetics**: Light and radio wave propagation
3. **Seismology**: Earthquake wave propagation
4. **Musical instruments**: String and membrane vibrations
5. **Water waves**: Shallow water approximation

## D'Alembert's Solution (1D)

The general solution in 1D is:
$$u(x,t) = f(x - ct) + g(x + ct)$$

representing right-traveling and left-traveling waves.

## Numerical Considerations

- Explicit schemes require CFL condition: $\Delta t \leq \frac{\Delta x}{c}$
- The system is not stiff; explicit methods work well
- Energy should be conserved (good check for numerical accuracy)

## References

- d'Alembert, J. (1747). *Recherches sur la courbe que forme une corde tendue*
- Strauss, W.A. (2007). *Partial Differential Equations: An Introduction*
