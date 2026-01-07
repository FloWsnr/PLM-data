# Keller-Segel Model

A chemotaxis model describing cell movement in response to chemical signals, originally developed to explain slime mold aggregation.

## Description

The Keller-Segel model was introduced by Evelyn Keller and Lee Segel in 1970 to explain the aggregation behavior of the cellular slime mold Dictyostelium discoideum. When starved, these amoebae produce and respond to the chemoattractant cAMP, leading to collective migration and the formation of multicellular structures.

The model has become one of the most widely studied in mathematical biology, serving as a prototype for:
- **Chemotaxis**: Directed cell movement along chemical gradients
- **Self-organization**: How collective patterns emerge from individual cell behaviors
- **Blow-up phenomena**: Solutions can concentrate into singular clusters in finite time

The model couples two processes:
1. **Cell flux**: Combines random diffusion with directed movement toward higher chemical concentrations (chemotaxis)
2. **Chemical dynamics**: Production by cells and natural decay

Key mathematical features:
- **Critical mass phenomenon** (in 2D): Below a threshold cell density, solutions remain bounded; above it, blow-up can occur
- **Pattern formation**: Chemotaxis-driven instabilities lead to cell aggregation patterns
- **Hysteresis**: Patterns depend on initial conditions and parameter history

Applications include:
- Bacterial colony formation
- Tumor growth and angiogenesis
- Wound healing
- Embryonic development

## Equations

### General Form
$$\frac{\partial u}{\partial t} = \nabla^2 u - \nabla \cdot (\chi(u) \nabla v) + f_u(u)$$

$$\frac{\partial v}{\partial t} = D \nabla^2 v + f_v(u, v)$$

### Specific Implementation
$$\chi(u) = \frac{cu}{1 + u^2}$$

$$f_u(u) = u(1-u)$$

$$f_v(u,v) = u - av$$

Where:
- $u$ is the cell density
- $v$ is the chemoattractant concentration
- $\chi(u)$ is the chemotactic sensitivity (saturating)
- $c$ is the chemotaxis strength
- $D$ is the chemical diffusion coefficient
- $a$ is the chemical decay rate

### Instability Condition
Linear instability predicts pattern formation when:
$$2\sqrt{aD} < \frac{c}{2} - D - a$$

## Default Config

```yaml
solver: euler
dt: 0.005
dx: 0.2
domain_size: 100

boundary_x: periodic
boundary_y: periodic

parameters:
  c: 4     # range: [3, 4], step: 0.1
  D: 1     # range: [0, 1]
  a: 0.1   # range: [0, 0.2]

init:
  type: random-uniform
  params:
    low: 0.0
    high: 0.01
```

## Parameter Variants

### KellerSegel (Standard)
Base configuration with chemotaxis-driven pattern formation.
- `c = 4`: Strong chemotaxis
- `D = 1`: Unit chemical diffusion
- `a = 0.1`: Slow chemical decay
- Initial condition: Small random population (0 to 0.01)
- Chemoattractant: Starts at zero

The saturation in the chemotactic sensitivity $\chi(u) = cu/(1+u^2)$ prevents unbounded aggregation at high densities.

## Notes

- The standard Keller-Segel model (without saturation) can exhibit finite-time blow-up
- This implementation uses a saturating chemotactic sensitivity to ensure bounded solutions
- The initial condition uses a small random population that grows towards equilibrium ($u = 1$, $v = 1/a$)
- Pattern formation occurs as cells near the carrying capacity equilibrium
- Varying parameters (especially around $c = 3.3$ to $3.6$) can explore the instability boundary
- Patterns show hysteresis - they depend on initial conditions and parameter history

## References

- Keller, E. F., & Segel, L. A. (1970). Initiation of slime mold aggregation viewed as an instability. Journal of Theoretical Biology, 26(3), 399-415.
- Keller, E. F., & Segel, L. A. (1971). Model for chemotaxis. Journal of Theoretical Biology, 30(2), 225-234.
- Patlak, C. S. (1953). Random walk with persistence and external bias. Bulletin of Mathematical Biophysics, 15(3), 311-338.
