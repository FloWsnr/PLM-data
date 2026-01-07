# Superlattice Patterns

Complex spatial patterns arising from the coupling of two different pattern-forming systems, producing hierarchical structures with multiple characteristic length scales.

## Description

Superlattice patterns emerge when two or more pattern-forming mechanisms with different intrinsic wavelengths interact. Rather than simply superimposing, the coupled systems can produce entirely new patterns with hierarchical spatial structure - patterns within patterns.

This model couples two well-known reaction-diffusion systems:
- **Brusselator** ($u_1, v_1$): A classic model from theoretical chemistry
- **Lengyel-Epstein** ($u_2, v_2$): Models the CIMA (chlorite-iodide-malonic acid) reaction

When these systems are coupled through a nonlinear interaction term, three-wave resonance conditions can be satisfied, leading to:
- **Simple patterns**: When modes reinforce each other
- **Superlattice patterns**: When spatial resonance conditions are met
- **Superposition patterns**: When modes don't resonate but coexist
- **Spatiotemporal chaos**: When modes compete destructively

The key insight is that coupling two oscillatory pattern-forming systems gives access to a much richer space of structures than either system alone. This is relevant to:
- **Materials science**: Self-assembled nanostructures
- **Chemical systems**: Coupled reactor dynamics
- **Biological development**: Multiple morphogen systems

## Equations

The coupled Brusselator-Lengyel-Epstein system:

**Brusselator subsystem** ($u_1, v_1$):

$$\frac{\partial u_1}{\partial t} = D_{u_1} \nabla^2 u_1 + a - (b+1)u_1 + u_1^2 v_1 + \alpha u_1 u_2 (u_2 - u_1)$$

$$\frac{\partial v_1}{\partial t} = D_{v_1} \nabla^2 v_1 + b u_1 - u_1^2 v_1$$

**Lengyel-Epstein subsystem** ($u_2, v_2$):

$$\frac{\partial u_2}{\partial t} = D_{u_2} \nabla^2 u_2 + c - u_2 - \frac{4 u_2 v_2}{1 + u_2^2} + \alpha u_1 u_2 (u_1 - u_2)$$

$$\frac{\partial v_2}{\partial t} = D_{v_2} \nabla^2 v_2 + d \left( u_2 - \frac{u_2 v_2}{1 + u_2^2} \right)$$

Where:
- $(u_1, v_1)$ are the Brusselator activator-inhibitor pair
- $(u_2, v_2)$ are the Lengyel-Epstein activator-inhibitor pair
- $\alpha$ is the coupling strength between subsystems
- $(a, b)$ are Brusselator kinetic parameters
- $(c, d)$ are Lengyel-Epstein kinetic parameters
- $(D_{u_1}, D_{v_1}, D_{u_2}, D_{v_2})$ are the four diffusion coefficients

**Coupling mechanism**: The terms $\alpha u_1 u_2 (u_2 - u_1)$ and $\alpha u_1 u_2 (u_1 - u_2)$ represent symmetric cross-coupling that vanishes when both activators are equal.

## Default Config

```yaml
solver: euler
dt: 0.0003
dx: 1
domain_size: 200

boundary_x: periodic
boundary_y: periodic

parameters:
  a: 3           # Brusselator feed
  b: 9           # Brusselator removal
  c: 15          # Lengyel-Epstein feed
  d: 9           # Lengyel-Epstein rate
  alpha: 0.15    # coupling strength (key parameter)
  D_uone: 4.3    # diffusion for u_1
  D_utwo: 50     # diffusion for v_1
  D_uthree: 22   # diffusion for u_2
  D_ufour: 660   # diffusion for v_2
```

## Parameter Variants

### Superlattice (Dynamic)
Default showing time-varying superlattice patterns:
- `alpha = 0.15`
- `D_uone = 4.3`, `D_utwo = 50`, `D_uthree = 22`, `D_ufour = 660`
- Initial condition: $u_1 = 3 + 0.1 \cdot \text{noise}$, $v_1 = 3$, $u_2 = 3$, $v_2 = 10$

### Pattern Type Gallery

| Name | alpha | D_u1 | D_v1 | D_u2 | D_v2 | Description |
|------|-------|------|------|------|------|-------------|
| Holes and spots | 0.15 | 3.1 | 13.95 | 18.9 | 670 | Nested circular structures |
| Holes and stripes | 0.1 | 3.5 | 10 | 18.7 | 550 | Stripes with holes |
| Dynamic | 0.15 | 4.3 | 50 | 22 | 660 | Time-varying patterns |
| Irregular holes/spots | 0.7 | 3.1 | 13.95 | 18.9 | 670 | Disordered superlattice |

### Resonance Conditions

Superlattice formation requires:
1. **Wave number ratio**: The two Turing modes must have compatible wavelengths
2. **Same symmetry**: Both subsystems must produce patterns with the same symmetry class
3. **Moderate coupling**: Too weak = no interaction; too strong = destruction of one mode

When these conditions are met, the long-wave mode from one system modulates the short-wave mode from the other, creating multi-scale structure.

### Exploration Guide

- **Vary alpha**: Controls coupling strength
  - alpha = 0: Independent subsystems (superposition only)
  - alpha ~ 0.1-0.2: Superlattice formation
  - alpha > 0.5: Strong competition, irregular patterns
- **Vary diffusion ratios**: Changes wavelength ratio and resonance
- **Watch $u_1$**: The Brusselator activator shows the clearest patterns

## References

- De Wit, A. et al. (1999). "Simple and superlattice Turing patterns" - Physica D 134:385
- Yang, L. et al. (2004). "Turing pattern formation in a two-layer system" - Phys. Rev. E 69:026211
- Malchow, H. et al. (2008). "Spatiotemporal Patterns in Ecology and Epidemiology" - CRC Press
- Epstein, I.R. & Pojman, J.A. (1998). "An Introduction to Nonlinear Chemical Dynamics" - Oxford
