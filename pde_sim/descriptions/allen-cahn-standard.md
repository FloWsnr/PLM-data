# Standard Allen-Cahn Equation (Symmetric Double-Well)

## Mathematical Formulation

The standard Allen-Cahn equation with symmetric double-well potential:

$$\frac{\partial c}{\partial t} = \gamma \nabla^2 c - c^3 + c$$

where:
- $c$ is the order parameter (phase field)
- $\gamma$ is the interfacial width parameter
- The reaction term $-c^3 + c = c(1 - c^2)$ derives from a double-well potential

## Physical Background

The Allen-Cahn equation describes phase transitions:

1. **Double-well potential**: $F(c) = \frac{1}{4}(c^2 - 1)^2$
2. **Gradient energy**: Penalizes sharp interfaces
3. **Time evolution**: Relaxation toward equilibrium

The equation is the $L^2$ gradient flow of the Ginzburg-Landau free energy:
$$\mathcal{F}[c] = \int \left[ \frac{\gamma}{2}|\nabla c|^2 + \frac{1}{4}(c^2 - 1)^2 \right] dx$$

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Interface width | $\gamma$ | Controls diffusion/interface thickness | 0.01 - 10 |
| Mobility | $M$ | Rate of evolution (optional prefactor) | 0.1 - 10 |

## Equilibria and Stability

The homogeneous equilibria are solutions of $c(1 - c^2) = 0$:

| Equilibrium | $c$ | Stability |
|-------------|-----|-----------|
| Phase + | +1 | Stable |
| Phase - | -1 | Stable |
| Unstable | 0 | Unstable |

## Difference from Bistable Allen-Cahn

| Property | Standard (this) | Bistable |
|----------|-----------------|----------|
| Reaction | $c - c^3$ | $u(u-a)(1-u)$ |
| Equilibria | $\{-1, 0, +1\}$ | $\{0, a, 1\}$ |
| Symmetry | Symmetric about 0 | Asymmetric |
| Applications | Phase transitions | Population dynamics |

## Interface Solutions

The equation admits traveling front solutions connecting $c = \pm 1$:
$$c(x,t) = \tanh\left(\frac{x - vt}{\sqrt{2\gamma}}\right)$$

- Front velocity $v = 0$ for symmetric wells
- Interface width $\sim \sqrt{\gamma}$

## Applications

1. **Metallurgy**: Grain boundary motion
2. **Materials science**: Phase separation
3. **Crystal growth**: Solidification fronts
4. **Image processing**: Segmentation (threshold dynamics)
5. **Computational physics**: Phase field models

## Initialization

Common initial conditions:
- **Random**: Small amplitude noise to trigger spinodal decomposition
- **Step function**: Interface dynamics study
- **Tanh profile**: Pre-formed interface evolution

## Numerical Considerations

- Explicit methods work for small $\gamma$
- Semi-implicit recommended for stiffness
- No-flux boundaries preserve total mass
- Interface resolution requires $\Delta x < \sqrt{\gamma}$

## Connection to Cahn-Hilliard

| Property | Allen-Cahn | Cahn-Hilliard |
|----------|-----------|---------------|
| Dynamics | Non-conserved | Conserved |
| Gradient flow | $L^2$ | $H^{-1}$ |
| Order | 2nd | 4th |
| Application | Grain growth | Phase separation |

## References

- Allen, S.M. & Cahn, J.W. (1979). *A microscopic theory for antiphase boundary motion*, Acta Metallurgica 27
- Bates, P.W. & Fife, P.C. (1993). *Dynamics of nucleation for the Cahn-Hilliard equation*, SIAM J. Applied Math.
- Chen, X. (1992). *Generation and propagation of interfaces*, Trans. AMS
