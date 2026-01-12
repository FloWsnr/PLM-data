# Potential Flow Dipoles

Dipole (doublet) flows in 2D potential flow with time-dependent moving sources, creating dynamic trailing wake patterns.

## Description

In potential flow theory, singularity solutions represent idealized point forces, vortices, sources, and sinks. These building blocks can be superposed (added together) due to the linearity of the governing Laplace equation, creating complex flow patterns from simple components.

A dipole (or doublet) arises when two equal and opposite singularities are brought infinitesimally close together while maintaining a finite product of strength and separation. Common examples include:
- Source dipole: source and sink pair
- Force dipole: two opposing point forces
- Vortex dipole: counter-rotating vortex pair

This implementation features **moving source-sink pairs** that follow prescribed trajectories (circular orbits, oscillations, or figure-8 patterns). As the dipole moves, it leaves trailing potential patterns that create rich, time-evolving dynamics. This is physically analogous to a moving source/sink pair in a fluid, or equivalently, a moving electric dipole creating time-varying potential fields.

The motion types available are:
- **circular**: Dipole orbits around the domain center, creating spiral wake patterns
- **oscillating**: Dipole oscillates horizontally with source/sink stacked vertically, creating streaked patterns
- **figure8**: Dipole traces a Lissajous figure-8 path, creating complex crossing patterns
- **static**: Traditional fixed dipole that relaxes to steady state

Dipoles are fundamental in aerodynamics and hydrodynamics. A uniform flow plus a source dipole produces flow around a circular cylinder. The concept extends to electrostatics and magnetism (electric and magnetic dipoles), governed by analogous equations.

## Equations

The potential evolves via a parabolic equation with time-dependent Gaussian source forcing:

$$\frac{\partial \phi}{\partial t} = \nabla^2 \phi + \text{strength} \cdot \left(G_{\text{source}}(t) - G_{\text{sink}}(t)\right)$$

where the Gaussian sources have time-dependent centers:

$$G(x, y, t) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{(x - x_c(t))^2 + (y - y_c(t))^2}{2\sigma^2}\right)$$

For circular motion with orbit radius $R$ and angular velocity $\omega$:
$$x_c(t) = x_0 + R\cos(\omega t), \quad y_c(t) = y_0 + R\sin(\omega t)$$

Velocity from potential:
$$u = \frac{\partial \phi}{\partial x}, \quad v = \frac{\partial \phi}{\partial y}$$

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| strength | 500 | 50-2000 | Source/sink intensity |
| separation | 3.0 | 1-10 | Distance between source and sink |
| sigma | 1.0 | 0.5-3.0 | Gaussian width for smooth sources |
| omega | 1.0 | 0.1-5.0 | Angular velocity (rad/time) for motion |
| orbit_radius | 8.0 | 2-15 | Radius of orbit around domain center |

Motion type is specified via `init.params.motion`: `circular`, `oscillating`, `figure8`, or `static`.

## Default Config

```yaml
preset: potential-flow-dipoles
parameters:
  strength: 500.0
  separation: 3.0
  sigma: 1.0
  omega: 0.5
  orbit_radius: 8.0
init:
  type: default
  params:
    motion: circular
solver: euler
bc: neumann (all boundaries)
t_end: 12.56  # ~2*pi for one full orbit at omega=0.5
```

## Config Variants

### default (circular orbit)
Dipole orbits around domain center creating spiral wake patterns:
- omega: 0.5, orbit_radius: 8
- Creates smooth spiraling potential trails

### oscillating
Horizontal back-and-forth motion with vertically-stacked source/sink:
- omega: 0.8, orbit_radius: 10
- Creates horizontal streaked patterns

### figure8
Lissajous figure-8 trajectory:
- omega: 0.4, orbit_radius: 10
- Creates complex crossing wake patterns
- Uses seismic colormap for high contrast

### fast_rotation
Tight, fast circular orbit:
- omega: 1.5, orbit_radius: 6
- Creates tight spiral patterns
- Uses RdBu colormap

### wide_orbit
Large slow orbit with periodic boundaries:
- omega: 0.3, orbit_radius: 14, domain_size: 50
- Periodic BC allows wrap-around effects
- Shows extended trailing wakes
