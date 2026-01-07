# Vorticity in Bounded Domain

This preset demonstrates the decay of oscillatory vorticity in a bounded domain, using different boundary conditions for different fields.

## Description

The vorticity-bounded preset extends the standard vorticity formulation with:

1. **Bounded domain**: Non-periodic boundary conditions in x-direction
2. **Oscillatory initial conditions**: High-frequency cosine pattern in vorticity
3. **Per-field boundary conditions**: Different BCs for vorticity, stream function, and passive scalar

The simulation shows how vorticity diffuses and decays in a bounded channel, with the oscillatory pattern smoothing out over time.

### Key Physical Behaviors

**Vorticity decay**: High-frequency vorticity oscillations decay due to viscous diffusion. Higher wavenumbers decay faster.

**Boundary layer effects**: The Dirichlet boundary conditions on vorticity create boundary layers where vorticity is constrained to zero.

**Passive scalar mixing**: The passive scalar is advected and diffused, creating interesting mixing patterns as the flow evolves.

## Equations

Same as the standard vorticity formulation:

$$\frac{\partial \omega}{\partial t} = \nu \nabla^2 \omega - \frac{\partial \psi}{\partial y} \frac{\partial \omega}{\partial x} + \frac{\partial \psi}{\partial x} \frac{\partial \omega}{\partial y}$$

$$\varepsilon \frac{\partial \psi}{\partial t} = \nabla^2 \psi + \omega$$

$$\frac{\partial S}{\partial t} = D \nabla^2 S - \frac{\partial \psi}{\partial y} \frac{\partial S}{\partial x} + \frac{\partial \psi}{\partial x} \frac{\partial S}{\partial y}$$

### Boundary Conditions

| Field | Left/Right (x) | Top/Bottom (y) |
|-------|----------------|----------------|
| omega | Dirichlet = 0 | Periodic |
| psi | Periodic | Periodic |
| S | Neumann = 0 | Periodic |

### Initial Condition

Oscillatory vorticity:
$$\omega_0 = A \cos\left(\frac{k\pi x}{L_x}\right) \cos\left(\frac{k\pi y}{L_y}\right)$$

where $A = 0.005 \cdot k^{1.5}$ and $k = 51$ (default).

## Default Config

```yaml
preset: vorticity-bounded
solver: euler
dt: 0.001
domain_size: 100

bc:
  x: periodic
  y: periodic
  fields:
    omega:
      left: dirichlet:0
      right: dirichlet:0
    S:
      left: neumann
      right: neumann

parameters:
  nu: 0.1        # kinematic viscosity
  epsilon: 0.05  # Poisson relaxation
  D: 0.05        # scalar diffusion
  k: 51          # wavenumber
```

## Parameter Variants

### High Wavenumber (k = 51)
- Fine-scale oscillations
- Rapid initial decay
- Complex boundary layer structure

### Low Wavenumber (k = 10)
- Coarser oscillations
- Slower decay
- Simpler boundary layer

### Low Viscosity (nu = 0.02)
- Slower vorticity decay
- More persistent flow features
- Longer simulation time needed

## Comparison with Standard Vorticity

| Feature | vorticity | vorticity-bounded |
|---------|-----------|-------------------|
| nu (default) | 0.05 | 0.1 |
| BCs | Periodic everywhere | Per-field, mixed |
| IC | Vortex pair | Oscillatory cosine |
| Behavior | Vortex dynamics | Decay to equilibrium |

## References

- Chemin, J.-Y. (1998). Perfect Incompressible Fluids. Oxford University Press.
- VisualPDE NavierStokesVorticityBounded preset
