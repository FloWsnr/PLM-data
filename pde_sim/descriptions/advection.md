# Advection-Diffusion Equation (Convection-Diffusion)

The advection-diffusion equation combines diffusive spreading with transport by a velocity field, describing how substances move through flowing media.

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

### Velocity Field Types

Two velocity fields are supported (matching visual-pde):

1. **Rotational flow**: Counterclockwise rotation about domain center for positive V
2. **Unidirectional flow**: Directed transport controlled by angle θ and magnitude V

## Equations

The advection-diffusion equation as implemented (matching visual-pde convention):

$$\frac{\partial u}{\partial t} = D \nabla^2 u + \text{advection term}$$

where:
- $u(x, y, t)$ is the concentration field
- $D$ is the diffusion coefficient

**Rotational mode** (counterclockwise rotation for positive V):
$$\frac{\partial u}{\partial t} = D \nabla^2 u + V\left((y - L_y/2) \frac{\partial u}{\partial x} - (x - L_x/2) \frac{\partial u}{\partial y}\right)$$

**Directed mode** (transport opposite to angle θ):
$$\frac{\partial u}{\partial t} = D \nabla^2 u + V\left(\cos\theta \frac{\partial u}{\partial x} + \sin\theta \frac{\partial u}{\partial y}\right)$$

Note: This matches the visual-pde sign convention where the "advection term" is added (not subtracted) from the diffusion term. This differs from the standard textbook form $\partial_t u = D\nabla^2 u - \mathbf{v}\cdot\nabla u$.

## Default Config

### Rotational Advection
```yaml
solver: euler
dt: 0.002
dx: 1.25
domain_size: 320

boundary_x: dirichlet
boundary_y: dirichlet

parameters:
  V: 0.10    # range: [-5, 5], step: 0.01 - rotation speed

species:
  - name: u
    diffusion: 1.0
```

### Directed Advection
```yaml
solver: euler
dt: 0.002
dx: 1.25
domain_size: 320

boundary_x: periodic
boundary_y: periodic

parameters:
  V: 6.0       # range: [0, 10], step: 0.01 - flow speed
  theta: -2.0  # range: [-6.4, 6.4], step: 0.01 - flow direction (radians)

species:
  - name: u
    diffusion: 1.0
```

## Parameter Variants

### AdvectionEquationRotational
Rotational (vortex) velocity field:
- Velocity: $\mathbf{v} = V(y - L_y/2, -(x - L_x/2))$
- Dirichlet boundary conditions (concentration absorbed at boundaries)
- Parameter: `V = 0.10` in range `[-5, 5]`
- Positive V: counterclockwise rotation
- Negative V: clockwise rotation
- Mass is not conserved due to Dirichlet boundaries

### AdvectionEquationDirected
Uniform directional velocity field:
- Velocity: $\mathbf{v} = V(\cos\theta, \sin\theta)$
- Periodic boundary conditions (concentration wraps around)
- Parameters: `V = 6` in `[0, 10]`, `theta = -2` in `[-6.4, 6.4]`
- Changing $\theta$ rotates the flow direction
- Mass is conserved with periodic boundaries

## Numerical Notes

First-order spatial derivatives (advection terms) are numerically challenging:
- Sharp gradients can cause spurious oscillations
- High Peclet numbers require special upwind schemes
- The preset uses smoothed brush application to reduce oscillations
- Large V values may produce unstable or inaccurate solutions

## References

- [Convection-diffusion equation - Wikipedia](https://en.wikipedia.org/wiki/Convection–diffusion_equation)
- [Advection - Wikipedia](https://en.wikipedia.org/wiki/Advection)
- [Advection-Diffusion Equation - ScienceDirect](https://www.sciencedirect.com/topics/earth-and-planetary-sciences/advection-diffusion-equation)
- [Advection-Diffusion Equation - Physics Across Oceanography](https://uw.pressbooks.pub/ocean285/chapter/advection-diffusion-equation/)
- [Advection-dominated equations - Finite Difference Computing](https://hplgit.github.io/fdm-book/doc/pub/book/sphinx/._book012.html)
