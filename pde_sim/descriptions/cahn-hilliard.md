# Cahn-Hilliard Equation

## Mathematical Formulation

The Cahn-Hilliard equation describes phase separation:

$$\frac{\partial u}{\partial t} = M \nabla^2 (u^3 - u - \gamma \nabla^2 u)$$

Equivalently, introducing chemical potential $\mu$:
$$\frac{\partial u}{\partial t} = M \nabla^2 \mu$$
$$\mu = u^3 - u - \gamma \nabla^2 u = \frac{\delta F}{\delta u}$$

where:
- $u$ is the concentration difference (order parameter), $-1 \leq u \leq 1$
- $M$ is the mobility coefficient
- $\gamma$ controls interface width
- $F$ is the free energy functional

## Physical Background

The Cahn-Hilliard equation is a **conserved gradient flow**:

$$\frac{\partial u}{\partial t} = \nabla \cdot \left(M \nabla \frac{\delta F}{\delta u}\right)$$

with Ginzburg-Landau free energy:
$$F[u] = \int \left[\frac{(u^2-1)^2}{4} + \frac{\gamma}{2}|\nabla u|^2\right] dx$$

Key property: **Mass conservation** $\frac{d}{dt}\int u \, dx = 0$

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Mobility | $M$ | Diffusion rate | 0.1 - 3 |
| Interface parameter | $\gamma$ | Interface width | 0.001 - 0.1 |

## Phase Separation Dynamics

Starting from random perturbations:
1. **Spinodal decomposition**: Rapid separation into phases
2. **Coarsening**: Domains grow over time
3. **Ostwald ripening**: Large domains grow, small shrink
4. **Steady state**: Single interface (eventually)

## Coarsening Laws

Domain size $R(t)$ grows as:
- **Bulk diffusion**: $R \sim t^{1/3}$ (Lifshitz-Slyozov)
- **Surface diffusion**: $R \sim t^{1/4}$

## Comparison with Allen-Cahn

| Property | Cahn-Hilliard | Allen-Cahn |
|----------|---------------|------------|
| Order | 4th spatial | 2nd spatial |
| Conservation | Mass conserved | Not conserved |
| Interface motion | Surface diffusion | Mean curvature |
| Steady state | Coarsened domains | Single phase |

## Applications

1. **Metallurgy**: Alloy phase separation
2. **Polymer blends**: Demixing dynamics
3. **Lipid membranes**: Domain formation
4. **Image processing**: Inpainting, segmentation
5. **Crystal growth**: Solidification patterns

## Interface Width

The equilibrium interface width scales as:
$$\ell \sim \sqrt{\gamma}$$

The interface profile is approximately:
$$u(x) = \tanh\left(\frac{x}{\sqrt{2\gamma}}\right)$$

## Numerical Considerations

- **4th-order equation**: Requires implicit solvers or small time steps
- **Stiff**: Time step restrictions severe for explicit methods
- **Mass conservation**: Should be preserved numerically
- **Energy decay**: Free energy should decrease monotonically

## References

- Cahn, J.W. & Hilliard, J.E. (1958). *Free Energy of a Nonuniform System*
- Elliott, C.M. & Zheng, S. (1986). *On the Cahn-Hilliard equation*
- Bray, A.J. (1994). *Theory of phase-ordering kinetics*
