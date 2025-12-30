# Method of Images Flow

## Mathematical Formulation

Vortex dynamics near a wall:

$$\frac{\partial \omega}{\partial t} = \nu \nabla^2 \omega$$

With initial condition incorporating **image vortices** to satisfy wall boundary conditions.

## Physical Background

The **method of images** is a technique from potential flow theory:

1. Real vortex near a solid wall
2. Fictitious "image" vortex placed symmetrically on other side of wall
3. Image vortex has opposite circulation
4. Combined flow satisfies no-penetration condition at wall

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Viscosity | $\nu$ | Kinematic viscosity | 0.001 - 1 |
| Wall distance | $d$ | Vortex-to-wall distance | 0.05 - 0.4 |

## Image Configuration

For a vortex of circulation $\Gamma$ at distance $d$ from wall:

| Position | Circulation | Effect |
|----------|-------------|--------|
| Real vortex at $y = d$ | $+\Gamma$ | Physical vortex |
| Image vortex at $y = -d$ | $-\Gamma$ | Boundary condition |

The velocity at the wall is tangent (no normal component).

## Vortex-Wall Interaction

The real vortex moves due to the image vortex:
- Translates parallel to wall
- Direction depends on circulation sign
- Speed: $v = \frac{\Gamma}{4\pi d}$

## Ground Effect

This models **ground effect** in aerodynamics:
- Aircraft near runway experience different lift
- Vortex interactions with ground alter wake
- Relevant for takeoff and landing

## Boundary Layer Separation

Vortices near walls are fundamental to:
- Boundary layer theory
- Flow separation
- Wake formation
- Drag generation

## Applications

1. **Aeronautics**: Ground effect, wingtip vortices
2. **Turbomachinery**: Blade-tip vortices
3. **Marine**: Ship propeller near hull
4. **Meteorology**: Atmospheric boundary layer
5. **Swimming/flying**: Locomotion near surfaces

## Viscous Effects

With viscosity:
- Vortices diffuse (spread out)
- Wall creates vorticity (no-slip condition)
- Vortex-wall interaction becomes complex
- Vortex can detach vorticity from wall

## Multiple Walls

For channel flow (two walls):
- Infinite series of images
- Each vortex has images in both walls
- Images of images create more images
- Series converges

## Potential Flow Limit

Inviscid, irrotational flow:
$$\nabla \times \mathbf{u} = 0, \quad \nabla \cdot \mathbf{u} = 0$$

Method of images gives exact solutions for complex geometries.

## References

- Saffman, P.G. (1992). *Vortex Dynamics*
- Lamb, H. (1932). *Hydrodynamics*
- Milne-Thomson, L.M. (1968). *Theoretical Hydrodynamics*
