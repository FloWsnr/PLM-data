# Advection-Diffusion Equation (Uniform Flow)

The advection-diffusion equation combines diffusive spreading with transport by a uniform velocity field, describing how substances move through flowing media.

## Description

The convection-diffusion equation (also called the advection-diffusion equation or drift-diffusion equation) is a parabolic PDE that describes physical phenomena where particles, energy, or other quantities are transferred by two processes simultaneously: diffusion (random molecular motion) and advection (bulk transport by fluid flow).

This equation has crucial applications across many fields:

- **Environmental science**: Transport of pollutants in rivers, groundwater contamination, atmospheric dispersion
- **Oceanography**: Salt concentration in ocean currents, sediment transport
- **Chemical engineering**: Mass transfer in reactors, chromatography
- **Heat transfer**: Convective cooling in moving fluids
- **Atmospheric science**: Smoke dispersion, aerosol transport
- **Pharmacokinetics**: Drug distribution in blood flow
- **Semiconductor physics**: Drift-diffusion of charge carriers (electrons and holes)
- **Geology**: Solute transport in porous media

### Physical Interpretation

The equation describes a competition between two transport mechanisms:

1. **Diffusion**: Spreads the quantity from high to low concentration (smoothing effect)
2. **Advection**: Carries the quantity along with the fluid velocity (transport effect)

The **Peclet number** (Pe = vL/D) characterizes which mechanism dominates:
- Low Pe: Diffusion dominates, advection negligible
- High Pe: Advection dominates, concentration "rides" with the flow

## Equations

The advection-diffusion equation with uniform velocity field:

$$\frac{\partial u}{\partial t} = D \nabla^2 u + v_x \frac{\partial u}{\partial x} + v_y \frac{\partial u}{\partial y}$$

where:
- $u(x, y, t)$ is the concentration field
- $D$ is the diffusion coefficient
- $v_x$ is the x-component of velocity
- $v_y$ is the y-component of velocity

Note: This matches the visual-pde sign convention where the advection term is added (not subtracted) from the diffusion term.

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| D | 1.0 | [0, 10] | Diffusion coefficient |
| vx | 5.0 | [-10, 10] | x-component of velocity |
| vy | 0.0 | [-10, 10] | y-component of velocity |

## Default Config

```yaml
preset: advection
parameters:
  D: 0.5
  vx: 5.0
  vy: 0.0
bc:
  x-: periodic
  x+: periodic
  y-: periodic
  y+: periodic
```

### Typical Usage

- **Horizontal flow**: `vx > 0, vy = 0` - flow to the right
- **Vertical flow**: `vx = 0, vy > 0` - flow upward
- **Diagonal flow**: `vx > 0, vy > 0` - flow diagonally

With periodic boundary conditions, mass is conserved as the concentration wraps around.

## Numerical Notes

First-order spatial derivatives (advection terms) are numerically challenging:
- Sharp gradients can cause spurious oscillations
- High Peclet numbers require special upwind schemes
- Large velocity values may produce unstable or inaccurate solutions

## See Also

- `advection-rotational`: For rotational (vortex) velocity fields

## References

- [Convection-diffusion equation - Wikipedia](https://en.wikipedia.org/wiki/Convection–diffusion_equation)
- [Advection - Wikipedia](https://en.wikipedia.org/wiki/Advection)
- [Advection-Diffusion Equation - ScienceDirect](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/advection-diffusion-equation)
- [Advection-Diffusion Equation - Physics Across Oceanography](https://uw.pressbooks.pub/ocean285/chapter/advection-diffusion-equation/)
- [Advection-dominated equations - Finite Difference Computing](https://hplgit.github.io/fdm-book/doc/pub/book/sphinx/._book012.html)
