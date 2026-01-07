# Potential Flow - Method of Images

Using image singularities to enforce boundary conditions in potential flow, demonstrating the classical method of images technique.

## Description

The method of images is an elegant analytical technique for enforcing boundary conditions in potential flow (and electrostatics). When a singularity solution exists in an unbounded domain, placing an appropriate "image" singularity outside the physical domain can exactly satisfy certain boundary conditions on the boundary between them.

For potential flow in the half-space x > 0 with a no-flux condition u = 0 on x = 0, a source at position (x_S, y_S) requires an image source at (-x_S, y_S) to cancel the normal velocity at the wall. The physical intuition: we think of the two sources as "reflections" of each other in the boundary plane.

This works because:
1. The Laplace equation is linear, so solutions superpose
2. By symmetry, the combined flow from source and image has zero normal component at the midplane
3. The boundary becomes an effective rigid wall with no penetration

The method extends to:
- Sinks (use image sink)
- Vortices (use image vortex of opposite sign for no-slip, same sign for free-slip)
- Dipoles and more complex singularities
- Multiple boundaries (infinite image series)
- Circular and spherical boundaries (using Kelvin inversion)

The simulation shows a source in the right half-plane. By clicking to place an image source, one can experimentally discover the correct location and verify that it prevents flow through the virtual boundary at x = L_x/2.

## Equations

Laplace equation (potential flow):
$$\nabla^2 \phi = 0$$

With point source forcing:
$$\nabla^2 \phi = -\delta(x - x_S, y - y_S)$$

Velocity from potential:
$$u = \frac{\partial \phi}{\partial x}, \quad v = \frac{\partial \phi}{\partial y}$$

In the simulation, implemented as parabolic relaxation:
$$\frac{\partial \phi}{\partial t} = \nabla^2 \phi - \text{strength} \cdot s$$

Visual PDE reference equation (has fixed bump + drawable s):
$$\frac{\partial \phi}{\partial t} = \nabla^2 \phi - 10s - 10 \cdot \text{Bump}(3L_x/4, L_y/2, L/100)$$

Our implementation encodes all source forcing in the s field initial condition (Gaussian bumps at source and image positions), allowing the equation to be simpler while achieving the same steady-state solution.

No-flux boundary condition at wall x = L_x/2:
$$u|_{x=L_x/2} = \frac{\partial \phi}{\partial x}\bigg|_{x=L_x/2} = 0$$

## Default Config

```yaml
solver: euler (parabolic relaxation)
dt: default
dx: 1.5
domain_size: 320

boundary_phi: neumann
boundary_s: dirichlet = 0

parameters:
  # Source position fixed at (3*L_x/4, L_y/2)
  # User clicks to place image sources
```

## Parameter Variants

### potentialFlowHalfSpace (Visual PDE reference)
Interactive method of images demonstration:
- Half-space x > L_x/2 with no-flux boundary visualized as white line
- Fixed source in right half at (3*L_x/4, L_y/2)
- Clicking places additional sources
- Challenge: find correct image source location to prevent wall penetration
- Can toggle to true half-space domain to verify solution
- Views: potential phi, horizontal velocity u, vertical velocity v
- Contour lines show equipotential curves
- Initial phi = 0 (relaxes to solution)

### Our implementation (source-with-image)
Pre-computed method of images solution:
- Source at (3*L_x/4, L_y/2) with image at mirrored position about wall
- Initial phi uses analytical log potential formula (near steady-state)
- s field contains Gaussian bumps at source and image locations
- Demonstrates the completed solution rather than interactive discovery

## Notes

The simulation uses parabolic relaxation to solve the elliptic Laplace equation. Any time-dependence observed is actually the solver converging to the steady-state potential flow solution. This numerical approach demonstrates an interesting connection between parabolic (diffusion-type) and elliptic (steady-state) PDEs.
