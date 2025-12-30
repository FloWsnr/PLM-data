# Complex Ginzburg-Landau Equation

## Mathematical Formulation

The complex Ginzburg-Landau (CGL) equation:

$$\frac{\partial A}{\partial t} = A + (1 + ic_1)\nabla^2 A - (1 + ic_3)|A|^2 A$$

where:
- $A$ is a complex amplitude field
- $c_1$ is the linear dispersion coefficient
- $c_3$ is the nonlinear dispersion coefficient

## Physical Background

The CGL equation is a **universal amplitude equation** for systems near a Hopf bifurcation with spatial variation. It describes:

1. **Linear instability**: $+A$ term (growth)
2. **Spatial coupling**: $(1+ic_1)\nabla^2 A$ (diffusion + dispersion)
3. **Nonlinear saturation**: $-(1+ic_3)|A|^2 A$ (cubic damping)

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Linear dispersion | $c_1$ | Phase-amplitude coupling | -5 to 5 |
| Nonlinear dispersion | $c_3$ | Nonlinear frequency shift | -5 to 5 |

## Special Cases

**Real Ginzburg-Landau** ($c_1 = c_3 = 0$):
$$\frac{\partial A}{\partial t} = A + \nabla^2 A - |A|^2 A$$
Gradient flow, patterns relax to steady states.

**Conservative limit** ($c_1 = c_3$):
Related to nonlinear Schrödinger equation.

## Phase Diagram

The $(c_1, c_3)$ parameter space shows:

| Region | Behavior |
|--------|----------|
| $1 + c_1 c_3 > 0$ | Stable plane waves (Benjamin-Feir stable) |
| $1 + c_1 c_3 < 0$ | Unstable, turbulence (Benjamin-Feir unstable) |
| Large $|c_1|, |c_3|$ | Defect chaos, spiral turbulence |

## Plane Wave Solutions

Plane waves: $A = R_0 e^{i(qx - \omega t)}$

Exist for range of wavenumbers $q$.

**Benjamin-Feir instability**: Plane waves can become unstable to modulations.

## Spiral Waves

In 2D, CGL supports **spiral wave** solutions:
- Rotating spiral arms
- Spiral tip traces circular path
- Core singularity (phase defect)

## Defect Turbulence

At large dispersion, spirals become unstable:
- Spiral cores multiply
- Chaotic creation/annihilation of defects
- "Defect-mediated turbulence"

## Applications

1. **Nonlinear optics**: Laser pattern formation
2. **Fluid dynamics**: Taylor-Couette flow
3. **Chemical systems**: Oscillating reactions
4. **Superconductivity**: Order parameter dynamics
5. **Cardiac dynamics**: Spiral waves in tissue

## Variational vs Non-Variational

| Property | $c_1 = c_3 = 0$ | General |
|----------|-----------------|---------|
| Structure | Gradient flow | Non-gradient |
| Dynamics | Energy minimizing | Can increase "energy" |
| Attractors | Fixed points | Limit cycles, chaos |

## Numerical Considerations

- Complex arithmetic required
- Semi-implicit schemes efficient
- Spiral tip tracking for diagnostics
- Large domains for turbulence studies

## References

- Aranson, I.S. & Kramer, L. (2002). *The world of the complex Ginzburg-Landau equation*
- Cross, M.C. & Hohenberg, P.C. (1993). *Pattern formation outside of equilibrium*
- Chaté, H. & Manneville, P. (1996). *Phase diagram of the CGL equation*
