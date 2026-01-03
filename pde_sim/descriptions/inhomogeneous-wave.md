# Inhomogeneous Wave Equation (Damped Wave)

## Mathematical Formulation

The inhomogeneous wave equation with damping and source:

$$\frac{\partial^2 u}{\partial t^2} = c^2 \nabla^2 u - \gamma \frac{\partial u}{\partial t} + f$$

Converted to first-order system:

$$\frac{\partial u}{\partial t} = v$$
$$\frac{\partial v}{\partial t} = c^2 \nabla^2 u - \gamma v + f$$

where:
- $u$ is the displacement
- $v$ is the velocity
- $c$ is the wave speed
- $\gamma$ is the damping coefficient
- $f$ is an external forcing term

## Physical Background

This equation models realistic wave phenomena with energy dissipation and external driving:

- **Damping ($\gamma > 0$)**: Energy loss to friction or radiation
- **Source ($f \neq 0$)**: External driving force

Without forcing, the system exhibits:
- **Underdamped** ($\gamma < 2c/L$): Oscillatory decay
- **Critically damped** ($\gamma = 2c/L$): Fastest non-oscillatory decay
- **Overdamped** ($\gamma > 2c/L$): Slow exponential decay

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Wave speed | $c$ | Propagation velocity | 0.5 - 2.0 |
| Damping | $\gamma$ | Energy dissipation rate | 0 - 10 |
| Source | $f$ | External forcing | -10 to +10 |

## Energy Behavior

The total energy $E = \frac{1}{2}\int (v^2 + c^2|\nabla u|^2) dx$ satisfies:

$$\frac{dE}{dt} = -\gamma \int v^2 dx + \int f \cdot v dx$$

- Pure damping: Energy decays exponentially
- With forcing: Energy can reach steady state or grow

## Applications

1. **Vibrating structures**: Damped building oscillations
2. **Acoustics**: Sound absorption in materials
3. **Electrical circuits**: Damped LC oscillators
4. **Seismology**: Wave attenuation in Earth

## References

- Morse, P.M. & Ingard, K.U. (1968). *Theoretical Acoustics*
