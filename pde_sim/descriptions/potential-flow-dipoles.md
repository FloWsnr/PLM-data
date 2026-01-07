# Potential Flow Dipoles

Dipole (doublet) flows in 2D potential flow, formed by bringing equal and opposite singularities together.

## Description

In potential flow theory, singularity solutions represent idealized point forces, vortices, sources, and sinks. These building blocks can be superposed (added together) due to the linearity of the governing Laplace equation, creating complex flow patterns from simple components.

A dipole (or doublet) arises when two equal and opposite singularities are brought infinitesimally close together while maintaining a finite product of strength and separation. Common examples include:
- Source dipole: source and sink pair
- Force dipole: two opposing point forces
- Vortex dipole: counter-rotating vortex pair

For a source-sink pair separated by distance 2d with strength Q, the potential phi satisfies:
$$\nabla^2 \phi = Q\delta(x-d,y) - Q\delta(x+d,y)$$

Taking the limit as d approaches 0 while dQ remains constant yields a dipole - mathematically equivalent to taking a spatial derivative of a single source solution.

Dipoles are fundamental in aerodynamics and hydrodynamics. A uniform flow plus a source dipole produces flow around a circular cylinder. The concept extends to electrostatics and magnetism (electric and magnetic dipoles), governed by analogous equations.

The simulation allows exploration of how the flow field changes as the source-sink separation d varies. For finite d, one sees two distinct singularities; as d approaches zero, the flow transitions to the idealized dipole pattern.

## Equations

Potential equation (Laplace with singularities):
$$\nabla^2 \phi = \text{source forcing}$$

Visual PDE reference equation:
$$\frac{\partial \phi}{\partial t} = \nabla^2 \phi + \frac{1000}{2d \cdot dx}\left(\text{ind}(s[x-d\cdot dx,y]>0) - \text{ind}(s[x+d\cdot dx,y]>0)\right)$$

Our implementation pre-computes the shifted indicator difference in the s field initial condition, giving:
$$\frac{\partial \phi}{\partial t} = \nabla^2 \phi + \text{strength} \cdot s$$

where s encodes the dipole forcing:
$$s = \frac{\text{bump}(x-d\cdot dx, y) - \text{bump}(x+d\cdot dx, y)}{2d \cdot dx}$$

Velocity from potential:
$$u = \frac{\partial \phi}{\partial x}, \quad v = \frac{\partial \phi}{\partial y}$$

## Default Config

```yaml
solver: euler (parabolic relaxation)
dt: default
dx: 1
domain_size: default

boundary_phi: neumann
boundary_s: dirichlet = 0

parameters:
  d: 5  # separation parameter, range [1, 30]
```

## Parameter Variants

### potentialFlowDipoleSlider (base)
Interactive dipole formation exploration:
- Slider controls separation parameter d
- Source-sink pair centered at domain middle
- Shows transition from distinct singularities (large d) to idealized dipole (small d)
- Velocity arrows and potential contours displayed
- Views: potential phi, horizontal velocity u, vertical velocity v

### potentialFlowDipoleClick
Click-to-place version:
- Fixed source at domain center
- Clicking places a sink to create source-sink pair
- domain_size: 201
- No slider control for d
- Allows interactive exploration of source-sink flow patterns
