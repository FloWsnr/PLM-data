# Superlattice Pattern Formation

## Mathematical Formulation

Swift-Hohenberg equation with quadratic nonlinearity for superlattice patterns:

$$\frac{\partial u}{\partial t} = \epsilon u - (1 + \nabla^2)^2 u + g_2 u^2 - u^3$$

Expanding:
$$\frac{\partial u}{\partial t} = \epsilon u - u - 2\nabla^2 u - \nabla^4 u + g_2 u^2 - u^3$$

where:
- $u$ is the pattern amplitude
- $\epsilon$ is the control parameter
- $g_2$ is the quadratic nonlinearity coefficient

## Physical Background

Superlattice patterns are **complex spatial structures** with multiple characteristic length scales:
- Primary pattern wavelength
- Secondary modulation (superlattice)
- Quasi-crystalline arrangements possible

The quadratic nonlinearity ($g_2 u^2$) breaks up-down symmetry and enables hexagonal and more complex patterns.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Control parameter | $\epsilon$ | Distance from onset | -0.5 to 1 |
| Quadratic nonlinearity | $g_2$ | Symmetry breaking | 0 to 2 |

## Pattern Selection

**Without $g_2$ ($g_2 = 0$)**:
- Up-down symmetric
- Stripes preferred

**With $g_2$ ($g_2 \neq 0$)**:
- Hexagons become possible
- Superlattice patterns emerge
- Complex quasipatterns

## Superlattice Types

1. **Hexagonal superlattices**: Modulated hexagonal patterns
2. **12-fold quasipatterns**: Quasi-crystalline symmetry
3. **Square superlattices**: Nested square patterns
4. **Mixed patterns**: Coexistence of different symmetries

## Formation Mechanism

Superlattices form through:
1. Primary instability creates basic pattern
2. Secondary instability modulates primary
3. Nonlinear interactions stabilize complex structure
4. Multiple resonant triads couple different modes

## Applications

1. **Nonlinear optics**: Laser beam transverse patterns
2. **Faraday waves**: Vibrated fluid surfaces
3. **Magnetic films**: Domain patterns
4. **Block copolymers**: Self-assembled structures
5. **Photonic crystals**: Designed light-matter interaction

## Resonant Triads

Pattern selection involves wavevector resonances:
$$\mathbf{k}_1 + \mathbf{k}_2 + \mathbf{k}_3 = 0$$
$$|\mathbf{k}_1| = |\mathbf{k}_2| = |\mathbf{k}_3| = k_c$$

These constraints determine possible pattern symmetries.

## Numerical Considerations

- **4th-order spatial**: Implicit methods needed
- **Large domains**: Required for superlattice wavelength
- **Fine resolution**: Must resolve both scales
- **Long times**: Complex patterns develop slowly

## References

- Cross, M.C. & Hohenberg, P.C. (1993). *Pattern formation outside of equilibrium*
- Malomed, B.A. et al. (2000). *Patterns in nonlinear optics*
- Edwards, W.S. & Fauve, S. (1994). *Patterns and quasi-patterns in the Faraday experiment*
