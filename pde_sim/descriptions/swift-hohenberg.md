# Swift-Hohenberg Equation

A canonical model for pattern formation near criticality, exhibiting stripes, hexagons, and remarkably stable localised structures through subcritical bifurcations.

## Description

The Swift-Hohenberg equation was introduced in 1977 by Jack Swift and Pierre Hohenberg as a simplified model for Rayleigh-Benard convection near the onset of instability. It has since become the canonical "normal form" equation for studying pattern formation, particularly near a Turing-type bifurcation.

The equation is notable for several key features:
- **Pattern selection**: Depending on the signs of parameters r, a, and b, the system can produce stripes, hexagons, squares, or more complex patterns
- **Subcriticality**: When r < 0, a > 0, and b < 0, the system enters a subcritical regime where both patterned states and the homogeneous state u=0 are stable
- **Localised solutions**: In subcritical regimes, the system supports stable localised patterns (spots, patches of stripes, or quasipatterns) that decay to the background
- **Homoclinic snaking**: Localised solutions exhibit "snaking" in parameter space - infinitely many coexisting localised states of different widths

The Swift-Hohenberg equation serves as the amplitude equation for systems near a Hopf bifurcation and is generic for any pattern-forming system near threshold. Its simplicity makes it analytically tractable while still capturing rich phenomenology observed in fluid convection, chemical reactions, and biological pattern formation.

## Equations

The Swift-Hohenberg equation is a fourth-order PDE:

$$\frac{\partial u}{\partial t} = r u - (k_c^2 + \nabla^2)^2 u + a u^2 + b u^3 + c u^5$$

Expanding the operator, this becomes:

$$\frac{\partial u}{\partial t} = r u - k_c^4 u - 2 k_c^2 \nabla^2 u - \nabla^4 u + a u^2 + b u^3 + c u^5$$

Where:
- $u$ is the pattern amplitude
- $r$ is the control/bifurcation parameter (distance from onset)
- $k_c$ is the critical wavenumber (sets the pattern wavelength)
- $a$ controls quadratic nonlinearity (breaks up-down symmetry, enables hexagons)
- $b$ controls cubic nonlinearity (saturation)
- $c$ controls quintic nonlinearity (needed when b > 0 for stability)

**Stability requirement**: $c < 0$ (or $b < 0$ if $c = 0$) for bounded solutions.

In the simulation, the fourth-order equation is decomposed using an auxiliary variable $v = \nabla^2 u$ via cross-diffusion:

$$\frac{\partial u}{\partial t} = (r-1)u - 2v + a u^2 + b u^3 + c u^5$$

with the algebraic constraint relating $v$ and diffusive coupling.

## Default Config

```yaml
solver: euler
dt: 0.0005
dx: 1 (inferred from domainScale/resolution)
domain_size: 100 (default) or 150 (localised)

boundary_x: periodic
boundary_y: periodic

parameters:
  r: 0.1     # [-2, 2] - bifurcation parameter
  a: 1       # [-2, 2] - quadratic coefficient
  b: 1       # [-2, 2] - cubic coefficient
  c: -1      # quintic coefficient (must be negative)
  D: 1       # effective diffusion scale (k_c = 1)
```

## Parameter Variants

### swiftHohenberg (Standard)
Supercritical pattern formation:
- `r = 0.1` (above threshold)
- `a = 1`, `b = 1`, `c = -1`
- Produces domain-filling stripe or hexagonal patterns
- Initial condition: u = 0 with perturbations

### swiftHohenbergLocalised
Subcritical regime supporting localised structures:
- `r = -0.28` (below threshold - bistable regime)
- `a = 1.6`, `b = -1`, `c = -1`
- `domain_size = 150`
- Supports stable localised patches of pattern
- Different symmetries accessible via parameter P:
  - P = 1: D4 symmetric localised structure
  - P = 2: Hexagonal (D6) localised structure
  - P = 3: D12 symmetric localised structure

### Key Behavioral Regimes

| Regime | r | a | b | Behavior |
|--------|---|---|---|----------|
| Supercritical | r > 0 | any | b < 0 | Domain-filling patterns |
| Subcritical bistable | r < 0 | a > 0 | b < 0 | Localised solutions possible |
| Hexagon-forming | any | a != 0 | any | Up-down asymmetric patterns |
| Stripe-only | any | a = 0 | b < 0 | Symmetric roll patterns |

## References

- Swift, J. & Hohenberg, P.C. (1977). "Hydrodynamic fluctuations at the convective instability" - Phys. Rev. A 15:319
- Burke, J. & Knobloch, E. (2006). "Localized states in the generalized Swift-Hohenberg equation"
- Hill, D. et al. (2023). "Symmetric localised solutions of the Swift-Hohenberg equation" - IOP Nonlinearity
