# Turing Instability Demonstration

## Mathematical Formulation

A system designed to demonstrate Turing pattern conditions:

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u + u(a - u) - uv$$
$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + uv - dv$$

where:
- $u$ is the activator
- $v$ is the inhibitor
- $D_u$ is activator diffusion (small)
- $D_v$ is inhibitor diffusion (large)
- $a$ is activator growth rate
- $d$ is inhibitor decay rate

## Physical Background

This system is designed to clearly illustrate **Turing's mechanism**:

1. **Local activation**: $u(a-u)$ term provides growth
2. **Cross-inhibition**: $-uv$ limits activator
3. **Inhibitor production**: $+uv$ from activator presence
4. **Inhibitor decay**: $-dv$ natural turnover
5. **Differential diffusion**: $D_v \gg D_u$ is essential

## Turing's Original Insight (1952)

Alan Turing showed that diffusion, usually a stabilizing force, can **destabilize** uniform states:

> "A system of chemical substances, called morphogens, reacting together and diffusing through a tissue, is adequate to account for the main phenomena of morphogenesis."

## Conditions for Turing Instability

**Without diffusion** (ODE): Stable steady state
- $\text{tr}(J) < 0$: Eigenvalues have negative real parts
- $\det(J) > 0$: Both eigenvalues have same sign

**With diffusion** (PDE): Instability possible
- At some wavenumber $k$, eigenvalues become positive
- Requires $D_v/D_u$ sufficiently large
- Pattern wavelength $\lambda \sim 2\pi/k_c$

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Activator diffusion | $D_u$ | Must be small | 0.005 - 0.1 |
| Inhibitor diffusion | $D_v$ | Must be large | 0.1 - 5 |
| Activator growth | $a$ | Self-enhancement | 0.1 - 3 |
| Inhibitor decay | $d$ | Natural turnover | 0.1 - 3 |

## Critical Diffusion Ratio

For pattern formation:
$$\frac{D_v}{D_u} > \left(\sqrt{f_u} + \sqrt{-g_v}\right)^2 / \left(\sqrt{-f_v} + \sqrt{g_u}\right)^2$$

where $f_u, f_v, g_u, g_v$ are Jacobian elements of the kinetics.

Typically requires $D_v/D_u > 10-100$.

## Pattern Wavelength

The selected wavenumber:
$$k_c^2 = \sqrt{\frac{f_u g_v - f_v g_u}{D_u D_v}}$$

Pattern wavelength: $\lambda = 2\pi/k_c$

## Emerging Patterns

Near onset: Regular patterns
- **Stripes**: Rolls with one preferred direction
- **Hexagons**: Two-dimensional lattice
- **Squares**: At special parameter values

Far from onset: Irregular patterns
- **Labyrinths**: Curved stripes
- **Spots**: Isolated peaks
- **Mixed**: Coexisting patterns

## Applications

1. **Embryology**: Body plan formation
2. **Chemistry**: Belousov-Zhabotinsky patterns
3. **Ecology**: Vegetation patterns
4. **Materials**: Self-organized nanostructures

## References

- Turing, A.M. (1952). *The Chemical Basis of Morphogenesis*
- Murray, J.D. (2003). *Mathematical Biology II*
- Cross, M.C. & Hohenberg, P.C. (1993). *Pattern formation outside of equilibrium*
