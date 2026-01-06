# Cross-Diffusion Schnakenberg Model

A Schnakenberg reaction-diffusion system extended with cross-diffusion terms, enabling dark solitons and extended Turing pattern spaces.

## Description

This model extends the classical Schnakenberg system by introducing cross-diffusion terms, where the diffusive flux of one species depends on the concentration gradient of the other species. Cross-diffusion significantly expands the range of behaviors beyond what is possible with standard (diagonal) diffusion.

In standard reaction-diffusion systems, Turing patterns require the inhibitor to diffuse much faster than the activator ($D_v \gg D_u$). Cross-diffusion relaxes this constraint and can enable pattern formation even when self-diffusion coefficients are equal.

Key phenomena include:
- **Dark solitons**: Localized inverted spots (regions of low concentration surrounded by high concentration)
- **Extended Turing space**: Pattern formation possible with equal self-diffusion rates
- **Spatiotemporal dynamics**: Oscillatory behaviors in Hopf regimes

The cross-diffusion terms represent:
- $D_{uv}$: Flux of u driven by gradients in v (e.g., chemoattraction/repulsion)
- $D_{vu}$: Flux of v driven by gradients in u

This type of model is relevant for systems where species interactions affect not just reactions but also movement, such as:
- Cell populations responding to chemical gradients
- Predator-prey systems with pursuit/evasion
- Charged particles in electric fields

## Equations

$$\frac{\partial u}{\partial t} = \nabla \cdot (D_{uu} \nabla u + D_{uv} \nabla v) + a - u + u^2 v$$

$$\frac{\partial v}{\partial t} = \nabla \cdot (D_{vu} \nabla u + D_{vv} \nabla v) + b - u^2 v$$

Where:
- $u$ is the activator concentration
- $v$ is the inhibitor concentration
- $D_{uu}, D_{vv}$ are self-diffusion coefficients
- $D_{uv}, D_{vu}$ are cross-diffusion coefficients
- $a, b > 0$ are kinetic parameters

The diffusion matrix is:
$$\mathbf{D} = \begin{pmatrix} D_{uu} & D_{uv} \\ D_{vu} & D_{vv} \end{pmatrix} = \begin{pmatrix} 1 & 3 \\ 0.2 & 1 \end{pmatrix}$$

## Default Config

```yaml
solver: euler
dt: 0.001
dx: 1.0
domain_size: 50

boundary_x: periodic
boundary_y: periodic

cross_diffusion: true

parameters:
  D_uu: 1
  D_uv: 3
  D_vu: 0.2
  D_vv: 1
  a: 0.01
  b: 2.5   # range: [0, 3]
```

## Parameter Variants

### crossDiffusionSchnakenberg
The default configuration creates localized dark solitons.
- Equal self-diffusion: `D_uu = D_vv = 1`
- Asymmetric cross-diffusion: `D_uv = 3`, `D_vu = 0.2`
- `b = 2.5` (default): Dark soliton behavior - localized inverted spots that do not propagate
- `b = 1`: Pattern formation closer to standard Schnakenberg
- `b = 0.1`: Spatiotemporal oscillatory behavior (Hopf regime)

## Notes

The key difference from standard Schnakenberg is:
1. Self-diffusion coefficients are equal ($D_{uu} = D_{vv} = 1$)
2. Cross-diffusion terms ($D_{uv} = 3$, $D_{vu} = 0.2$) enable pattern formation
3. The off-diagonal terms in the diffusion matrix create effective differential diffusion

## References

- Schnakenberg, J. (1979). Simple chemical reaction systems with limit cycle behaviour. Journal of Theoretical Biology, 81(3), 389-400.
- Vanag, V. K., & Epstein, I. R. (2009). Cross-diffusion and pattern formation in reaction-diffusion systems. Physical Chemistry Chemical Physics, 11(6), 897-912.
- Gambino, G., Lombardo, M. C., & Sammartino, M. (2012). Turing instability and traveling fronts for a nonlinear reaction-diffusion system with cross-diffusion. Mathematics and Computers in Simulation, 82(6), 1112-1132.
