# Standing Wave Equation

The standing wave equation is the pure (undamped) wave equation initialized with sinusoidal modes to produce standing wave patterns, similar to Chladni figures on vibrating membranes.

## Description

Standing waves occur when waves reflect from boundaries and interfere with themselves, creating stationary patterns with fixed nodal lines. This preset uses the wave equation without any damping (C=0), which preserves the wave energy and produces persistent oscillating patterns.

On a rectangular membrane with fixed or reflecting edges, standing wave patterns form characteristic nodal lines where the displacement remains zero. These patterns are indexed by mode numbers (n, m) representing the number of half-wavelengths in each direction.

### Applications

- **Chladni figures**: Sand patterns on vibrating plates reveal standing wave nodal lines
- **Musical instruments**: String and membrane vibration modes determine harmonic frequencies
- **Acoustics**: Room modes in enclosed spaces
- **Quantum mechanics**: Particle in a box wavefunctions have analogous standing wave solutions

## Equations

The undamped wave equation in two dimensions:

$$\frac{\partial^2 u}{\partial t^2} = D \nabla^2 u$$

Reformulated as a first-order system:

$$\frac{\partial u}{\partial t} = v$$

$$\frac{\partial v}{\partial t} = D \nabla^2 u$$

where:
- $u(x, y, t)$ is the displacement field
- $v$ is the velocity field $\partial u / \partial t$
- $D$ is the wave speed squared ($c^2$)

With sinusoidal initial conditions:
$$u(x,y,0) = A \sin(k_x \pi x / L_x) \sin(k_y \pi y / L_y)$$

the analytical solution is a standing wave:
$$u(x,y,t) = A \cos(\omega t) \sin(k_x \pi x / L_x) \sin(k_y \pi y / L_y)$$

where $\omega = \sqrt{D} \pi \sqrt{k_x^2/L_x^2 + k_y^2/L_y^2}$ is the angular frequency.

## Default Config

```yaml
preset: wave-standing

parameters:
  D: 1.0       # Wave speed squared

init:
  type: sine
  params:
    kx: 4      # Mode number in x direction
    ky: 4      # Mode number in y direction
    amplitude: 1.0

bc:
  x: neumann   # Reflecting boundaries
  y: neumann
```

## References

- [Standing wave - Wikipedia](https://en.wikipedia.org/wiki/Standing_wave)
- [Chladni figure - Wikipedia](https://en.wikipedia.org/wiki/Chladni_figure)
- [Vibrations of Rectangular Membranes - Mathematics LibreTexts](https://math.libretexts.org/Bookshelves/Differential_Equations/Introduction_to_Partial_Differential_Equations_(Herman)/06:_Problems_in_Higher_Dimensions/6.01:_Vibrations_of_Rectangular_Membranes)
