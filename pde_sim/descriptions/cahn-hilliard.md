# Cahn-Hilliard Equation

A fourth-order PDE describing phase separation and spinodal decomposition in binary mixtures, producing characteristic coarsening domain structures.

## Description

The Cahn-Hilliard equation, developed by John W. Cahn and John E. Hilliard in 1958, describes the process of phase separation (spinodal decomposition) in binary mixtures such as metal alloys or polymer blends. When a homogeneous mixture is rapidly quenched below a critical temperature, it becomes thermodynamically unstable and spontaneously separates into two distinct phases.

Key physical phenomena captured by this equation:
- **Spinodal decomposition**: Unlike nucleation (which requires overcoming an energy barrier), spinodal decomposition occurs spontaneously when the mixture is unstable to infinitesimal fluctuations
- **Coarsening dynamics**: After initial phase separation, domains grow over time through a process called Ostwald ripening, where larger domains grow at the expense of smaller ones
- **Conservation of mass**: The equation is conservative - the average concentration remains constant
- **Interfacial energy**: The higher-order derivative term penalizes sharp interfaces, leading to smooth domain boundaries

The characteristic "labyrinthine" or "bicontinuous" structures that emerge are ubiquitous in materials science, appearing in phase-separating polymers, lipid membranes, and even galaxy distributions. The coarsening follows a power law with time, making the Cahn-Hilliard equation a benchmark for studying domain growth kinetics.

The version implemented here includes an additional reaction term that can create steady-state patterns rather than full phase separation.

## Equations

The standard Cahn-Hilliard equation is:

$$\frac{\partial u}{\partial t} = \nabla^2 \left( F'(u) - g \nabla^2 u \right)$$

where $F(u)$ is a double-well free energy density (e.g., $F(u) = \frac{1}{4}(u^2-1)^2$).

The implemented version with reaction term (matching Visual PDE preset):

$$\frac{\partial u}{\partial t} = r \left[ D \nabla^2 \left( u^3 - u - a \nabla^2 u \right) + u - u^3 \right]$$

The parameter $r$ controls the overall timescale - increasing $r$ speeds up both diffusive separation and reaction dynamics uniformly.

Where:
- $u \in [-1, 1]$ represents the local concentration (relative to the critical mixture)
- $r$ controls the overall timescale (increase to speed up dynamics)
- $a$ controls the interfacial energy / interface width
- $D$ is the diffusion coefficient
- The term $(u^3 - u)$ comes from the derivative of the double-well potential
- The reaction term $u - u^3$ can stabilize intermediate states

## Default Config

```yaml
solver: euler
dt: 0.0005
resolution: 200
domain_size: 100

boundary_x: periodic
boundary_y: periodic

init:
  type: cahn-hilliard-default  # tanh(30*(RAND-0.5)) - values near ±1

parameters:
  r: 0.01    # [0, 1] - timescale parameter (increase to speed up)
  a: 1.0     # interfacial energy coefficient
  D: 1.0     # diffusion coefficient
```

## Parameter Variants

### CahnHilliard (Standard)
Phase separation from random initial conditions:
- `r = 0.01` (slow initial dynamics - increase to speed up)
- `a = 1`, `D = 1`
- Initial condition: `tanh(30*(RAND-0.5))` - random values near +/-1
- Fixed random seed for reproducibility
- Exhibits characteristic coarsening behavior

### CahnHilliardNonreciprocal
Non-reciprocal variant with two coupled fields:
- Additional diffusive coupling between species
- Parameters: `D_1 = 1`, `D_2 = 1` (coupling), `kappa = 0.1`
- Produces traveling and oscillating patterns
- Based on recent work by Brauns and Marchetti (2024)

### Physical Interpretation

| Parameter | Physical Meaning |
|-----------|------------------|
| u = +1 | Pure phase A |
| u = -1 | Pure phase B |
| u = 0 | Critical mixture (50-50) |
| r | Controls separation rate |
| g (or a) | Interface thickness / surface tension |

### Observing Coarsening

Starting from random initial conditions:
1. Initial rapid phase separation creates fine-scale domains
2. Domains coarsen over time (larger structures emerge)
3. Characteristic domain size grows as $L(t) \sim t^{1/3}$ (Lifshitz-Slyozov law)
4. Increase $r$ by 1-2 orders of magnitude to observe faster dynamics

## References

- Cahn, J.W. & Hilliard, J.E. (1958). "Free Energy of a Nonuniform System" - J. Chem. Phys. 28:258
- Bray, A.J. (2002). "Theory of phase-ordering kinetics" - Advances in Physics 51:481
- Brauns, F. & Marchetti, M.C. (2024). "Non-reciprocal pattern formation" - Phys. Rev. X 14:021014
- Trefethen, L.N. (2001). "Cahn-Hilliard Equation" - pdectb.pdf (Oxford)
