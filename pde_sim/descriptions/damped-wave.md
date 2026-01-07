# Damped Wave Equation

The damped wave equation extends the standard wave equation with explicit velocity damping, modeling energy dissipation proportional to the velocity of oscillation.

## Description

The damped wave equation models wave propagation in media with friction or viscous resistance. Unlike the standard wave equation where waves propagate indefinitely, the damping term causes wave amplitudes to decay exponentially over time. This is physically realistic for most real-world systems.

The equation has applications across many domains:

- **Mechanical vibrations**: Damped vibrations in springs, pendulums, and structures
- **Acoustic damping**: Sound absorption in materials and rooms
- **Electrical circuits**: RLC circuits with resistance
- **Seismic waves**: Energy loss as waves propagate through Earth's crust
- **Water waves**: Surface wave damping due to viscosity

### Key Physical Behaviors

**Exponential decay**: Wave amplitudes decay exponentially at a rate proportional to the damping coefficient $d$. Higher damping leads to faster energy loss.

**Overdamping vs underdamping**: For sufficiently high damping, oscillations cease entirely and the system returns to equilibrium monotonically (overdamped). Below this threshold, damped oscillations occur (underdamped).

**Quality factor**: The ratio of energy stored to energy lost per cycle defines how "resonant" the system is. Low damping gives high quality factor and sharp resonance peaks.

## Equations

The damped wave equation as a first-order system:

$$\frac{\partial u}{\partial t} = v + CD \nabla^2 u$$

$$\frac{\partial v}{\partial t} = D \nabla^2 u - d \cdot v$$

where:
- $u(x, y, t)$ is the displacement field
- $v(x, y, t)$ is the velocity field ($\partial u / \partial t$)
- $D$ is the wave speed squared ($c^2$)
- $C$ is a stabilization coefficient
- $d$ is the velocity damping coefficient

The $-d \cdot v$ term represents energy dissipation proportional to velocity, similar to viscous friction in mechanical systems.

## Default Config

```yaml
preset: damped-wave
solver: euler
dt: 0.002
domain_size: 100

bc:
  x: neumann
  y: neumann

parameters:
  D: 1.0       # wave speed squared
  C: 0.01      # stabilization coefficient
  d: 0.05      # velocity damping coefficient
```

## Parameter Variants

### Lightly Damped
- `d = 0.01`: Slow amplitude decay, many oscillations visible
- Best for observing wave interference patterns before decay

### Critically Damped
- `d ~ 0.1`: System returns to equilibrium as fast as possible without oscillating
- Optimal for systems where overshoot must be avoided

### Overdamped
- `d > 0.1`: No oscillations, slow return to equilibrium
- Motion is sluggish and monotonic

## Comparison with Standard Wave Equation

The standard wave equation (`wave` preset) has damping only via the $C$ term which affects the displacement equation. This is a different kind of damping:

- **Standard wave (C term)**: Artificial numerical damping that affects high-frequency modes
- **Damped wave (d term)**: Physical velocity damping that affects all modes proportionally

The damped wave equation more accurately models physical systems with friction.

## References

- [Damped wave equation - Wikipedia](https://en.wikipedia.org/wiki/Damped_wave)
- [Damped Harmonic Oscillator - HyperPhysics](http://hyperphysics.phy-astr.gsu.edu/hbase/oscda.html)
- VisualPDE Damped Wave Equation preset
