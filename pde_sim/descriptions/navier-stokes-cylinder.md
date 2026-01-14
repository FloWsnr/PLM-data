# Navier-Stokes Flow Around Cylinder

The 2D Navier-Stokes equations with a cylindrical obstacle, demonstrating the von Kármán vortex street phenomenon.

## Description

Flow around a bluff body (like a cylinder) is one of the most studied problems in fluid dynamics. At low Reynolds numbers, the flow remains steady and symmetric. As the Reynolds number increases, the wake becomes unstable and develops a periodic pattern of alternating vortices known as the von Kármán vortex street, named after Theodore von Kármán who analyzed this phenomenon in 1911.

This preset implements the cylinder as an immersed boundary using a penalization method. The obstacle field S acts as an indicator function (S=1 inside the cylinder, S=0 outside), and additional damping terms -u*S and -v*S in the momentum equations force the velocity to zero inside the obstacle region.

The vortex shedding frequency is characterized by the Strouhal number St = f*D/U, where f is the shedding frequency, D is the cylinder diameter, and U is the free-stream velocity. For a wide range of Reynolds numbers (300 < Re < 100000), the Strouhal number remains approximately constant at St ≈ 0.2.

Applications include:
- Aerodynamic design (reducing drag and noise)
- Heat exchanger optimization
- Bridge and building design (preventing oscillations)
- Acoustic signal generation

## Equations

Momentum equations with obstacle damping:
$$\frac{\partial u}{\partial t} = -\left(u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y}\right) - \frac{\partial p}{\partial x} + \nu \nabla^2 u - u \cdot S$$

$$\frac{\partial v}{\partial t} = -\left(u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y}\right) - \frac{\partial p}{\partial y} + \nu \nabla^2 v - v \cdot S$$

Generalized pressure equation:
$$\frac{\partial p}{\partial t} = \nu \nabla^2 p - \frac{1}{M^2}\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right)$$

Obstacle field (static):
$$\frac{\partial S}{\partial t} = 0$$

where S is defined as:
$$S(x,y) = \begin{cases} 1 & \text{inside cylinder} \\ 0 & \text{outside cylinder} \end{cases}$$

## Default Config

```yaml
solver: euler
dt: 0.03
domain_size: 128

boundary_x_u: periodic (top/bottom), dirichlet = U (left/right)
boundary_x_v: periodic (top/bottom), dirichlet = 0 (left/right)
boundary_p: periodic (top/bottom), neumann (left/right)
boundary_S: dirichlet = 0 (all boundaries)

parameters:
  nu: 0.1             # kinematic viscosity
  M: 0.5              # Mach number parameter
  U: 0.7              # inflow velocity
  cylinder_radius: 0.05  # cylinder radius (fraction of domain)
```

## Parameter Variants

### High Reynolds Number
Lower viscosity for stronger vortex shedding:
- nu: 0.02
- Creates more pronounced vortex street
- May require smaller timestep for stability

### Multiple Cylinders
Two or more cylinders for complex vortex interactions:
- Use "multi-cylinder" initial condition type
- Specify positions as list of (x, y) tuples
- Demonstrates vortex interaction phenomena

## References

- von Kármán, T. (1911). "Über den Mechanismus des Widerstandes, den ein bewegter Körper in einer Flüssigkeit erfährt"
- Visual PDE NavierStokesFlowCylinder preset
- Williamson, C.H.K. (1996). "Vortex dynamics in the cylinder wake"
