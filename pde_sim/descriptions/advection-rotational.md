# Advection-Diffusion Equation (Rotational Flow)

The advection-diffusion equation with a rotational (vortex) velocity field, describing how substances are stirred around a central point while diffusing.

## Description

This variant of the advection-diffusion equation uses a position-dependent velocity field that creates rotational flow around the domain center. It models stirring, mixing, and vortex-like transport phenomena.

Applications include:

- **Fluid mixing**: Stirring in chemical reactors
- **Atmospheric vortices**: Simplified cyclone/anticyclone models
- **Ocean eddies**: Mesoscale circular currents
- **Laboratory flows**: Rotating fluid experiments

### Physical Interpretation

The velocity field at each point is tangent to circles centered on the domain:
- **Positive omega**: Counterclockwise rotation
- **Negative omega**: Clockwise rotation

Unlike uniform advection, the rotational flow:
- Has zero velocity at the domain center
- Has maximum velocity at the domain edges
- Creates spiral patterns as material diffuses while rotating

## Equations

The advection-diffusion equation with rotational velocity field:

$$\frac{\partial u}{\partial t} = D \nabla^2 u + \omega \left((y - L_y/2) \frac{\partial u}{\partial x} - (x - L_x/2) \frac{\partial u}{\partial y}\right)$$

where:
- $u(x, y, t)$ is the concentration field
- $D$ is the diffusion coefficient
- $\omega$ is the angular velocity
- $L_x, L_y$ are the domain dimensions

The velocity field is:
$$\mathbf{v} = \omega \begin{pmatrix} y - L_y/2 \\ -(x - L_x/2) \end{pmatrix}$$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| D | 1.0 | [0, 10] | Diffusion coefficient |
| omega | 0.1 | [-5, 5] | Angular velocity (positive = counterclockwise) |

## Default Config

```yaml
preset: advection-rotational
parameters:
  D: 1.0
  omega: 0.1
bc:
  x-: dirichlet:0
  x+: dirichlet:0
  y-: dirichlet:0
  y+: dirichlet:0
```

### Typical Usage

- **Slow stirring**: `omega = 0.1` - gentle rotation
- **Fast stirring**: `omega = 1.0` - rapid mixing
- **Clockwise**: `omega < 0` - reverses rotation direction

With Dirichlet boundary conditions, mass is absorbed at the edges as material rotates outward.

## Numerical Notes

- The position-dependent velocity field makes this more computationally intensive than uniform advection
- Large omega values may require smaller time steps for stability
- The velocity magnitude increases with distance from center, so edge effects are stronger

## See Also

- `advection`: For uniform (constant) velocity fields

## References

- [Convection-diffusion equation - Wikipedia](https://en.wikipedia.org/wiki/Convection–diffusion_equation)
- [Advection - Wikipedia](https://en.wikipedia.org/wiki/Advection)
- [VisualPDE - Advection Equation](https://visualpde.com/basic-pdes/advection-equation)
