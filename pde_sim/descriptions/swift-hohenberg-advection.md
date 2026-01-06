# Swift-Hohenberg with Advection

Pattern formation under flow - localised structures translating or rotating while maintaining their form under the combined effects of diffusion, nonlinearity, and advection.

## Description

This extension of the Swift-Hohenberg equation incorporates advection (bulk transport by a velocity field), creating a system where patterns can both form and move. The interplay between pattern formation and transport leads to rich dynamics where structures can:

- **Translate**: Move in a fixed direction while maintaining shape
- **Rotate**: Spin around a center point
- **Deform**: Change shape due to boundary interactions or flow gradients
- **Destabilize**: Break apart if flow is too strong

The advected Swift-Hohenberg equation demonstrates concepts relevant to:
- **Convection patterns**: Thermal rolls moving with background flow
- **Chemical waves**: Reaction-diffusion patterns in flowing media
- **Ecological patterns**: Vegetation bands on hillslopes with water runoff
- **Ocean dynamics**: Phytoplankton patterns in currents

This model is particularly interesting because it shows how robust localised structures (spots, patches of stripes) can survive and propagate under forcing that would destroy less stable patterns.

## Equations

The Swift-Hohenberg equation with advection:

**Unidirectional advection** (constant velocity at angle $\theta$):

$$\frac{\partial u}{\partial t} = ru - (1 + \nabla^2)^2 u + au^2 + bu^3 + V\left(\cos\theta \frac{\partial u}{\partial x} + \sin\theta \frac{\partial u}{\partial y}\right)$$

**Rotational advection** (solid body rotation around domain center):

$$\frac{\partial u}{\partial t} = ru - (1 + \nabla^2)^2 u + au^2 + bu^3 + V\left((y-L_y/2)\frac{\partial u}{\partial x} - (x-L_x/2)\frac{\partial u}{\partial y}\right)$$

Via cross-diffusion with $v = \nabla^2 u$:

$$\frac{\partial u}{\partial t} = (r-1)u - 2v + au^2 + bu^3 + \text{(advection terms)}$$

Where:
- All Swift-Hohenberg parameters $(r, a, b, D)$ as in the standard equation
- $V$ is the advection velocity magnitude
- $\theta$ is the direction angle (for unidirectional)
- The advection terms $V \cdot \nabla u$ represent transport by the velocity field

## Default Config

```yaml
solver: euler
dt: 0.0008 (directed) / 0.001 (rotational)
dx: 1 (inferred)
domain_size: 150

boundary_x: periodic (directed) / dirichlet (rotational)
boundary_y: periodic (directed) / dirichlet (rotational)

parameters:
  r: -0.28       # [-2, 2] - bifurcation parameter (subcritical)
  a: 1.6         # [-2, 2] - quadratic coefficient
  b: -1          # [-2, 2] - cubic coefficient
  c: -1          # quintic coefficient
  D: 1           # diffusion scale
  V: 2.0         # [0, 10] - advection velocity (directed)
  theta: -2.0    # [-6.4, 6.4] - direction angle (radians)
  P: 3.0         # [1, 3] - pattern symmetry selector
```

## Parameter Variants

### swiftHohenbergLocalisedDirectedAdvection
Unidirectional translation:
- Periodic boundaries
- `V = 2.0`, `theta = -2.0` (adjustable)
- Localised pattern translates across domain
- Can change $P$ to select different symmetries

### swiftHohenbergLocalisedRotationalAdvection
Solid body rotation:
- Dirichlet boundaries (to handle rotating velocity field)
- `V = 0.1` (lower velocity for rotation)
- Localised pattern spins in place
- Warning: Large $V$ can cause boundary instabilities

### Pattern Symmetry via P

| P Value | Initial Pattern Symmetry |
|---------|-------------------------|
| P = 1 | D4 (square) symmetry |
| P = 2 | D6 (hexagonal) symmetry |
| P = 3 | D12 (dodecagonal) symmetry |

### Advection Effects

| Regime | Behavior |
|--------|----------|
| V = 0 | Standard Swift-Hohenberg (stationary patterns) |
| V small | Pattern translates/rotates with minor deformation |
| V moderate | Significant interaction with pattern dynamics |
| V large | Pattern may destabilize, wave selection, boundary effects |

### Stability Considerations

- **Periodic boundaries** (directed): Patterns wrap around smoothly
- **Dirichlet boundaries** (rotational): The rotating velocity field is not periodic, so patterns interacting with boundaries can misbehave
- **CFL condition**: Advection adds stability constraints; may need to reduce $V$ or $dt$

### Related Models

The Gray-Scott system with advection (`GrayScottGlidersAdvecting`) shows similar phenomena:
- Gliders "swimming upstream" against advective flow
- Flow can select between different pattern types
- Mass loss at boundaries orthogonal to flow

## References

- Cross, M.C. & Hohenberg, P.C. (1993). "Pattern formation outside of equilibrium" - Rev. Mod. Phys. 65:851
- Chomaz, J.M. (2005). "Global instabilities in spatially developing flows" - Annu. Rev. Fluid Mech. 37:357
- Knobloch, E. (2015). "Spatial localization in dissipative systems" - Annu. Rev. Condens. Matter Phys. 6:325
