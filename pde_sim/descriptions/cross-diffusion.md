# Cross-Diffusion Model

## Mathematical Formulation

The cross-diffusion system (Shigesada-Kawasaki-Teramoto type):

$$\frac{\partial u}{\partial t} = D_1 \nabla^2 u + \alpha \nabla^2(uv) + u(1 - u - v)$$
$$\frac{\partial v}{\partial t} = D_2 \nabla^2 v + \beta \nabla^2(uv) + v(1 - u - v)$$

where:
- $u, v$ are species densities
- $D_1, D_2$ are self-diffusion coefficients
- $\alpha, \beta$ are cross-diffusion coefficients
- The reaction term is logistic competition

## Physical Background

Cross-diffusion describes **density-dependent dispersal**: the movement of one species depends on the density of another.

**Standard diffusion**: Species move from high to low density of itself
**Cross-diffusion**: Species move from high to low density of the other

This can represent:
- **Avoidance**: Species moves away from competitor
- **Aggregation**: Species moves toward partner
- **Pressure effects**: Crowding affects movement

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Self-diffusion u | $D_1$ | Own-density diffusion | 0.01 - 0.5 |
| Self-diffusion v | $D_2$ | Own-density diffusion | 0.01 - 0.5 |
| Cross-diffusion u | $\alpha$ | u's response to v | 0 - 5 |
| Cross-diffusion v | $\beta$ | v's response to u | 0 - 5 |

## Pattern Formation Mechanism

Cross-diffusion can produce Turing-like patterns **without** the classical activator-inhibitor structure:

- Standard Turing: Requires $D_v \gg D_u$
- Cross-diffusion: Can work with equal self-diffusion

The cross-diffusion term effectively creates differential mobility based on local species composition.

## Full SKT Model

The original Shigesada-Kawasaki-Teramoto model:

$$\frac{\partial u}{\partial t} = \nabla \cdot [(d_1 + a_{11}u + a_{12}v)\nabla u] + f(u,v)$$
$$\frac{\partial v}{\partial t} = \nabla \cdot [(d_2 + a_{21}u + a_{22}v)\nabla v] + g(u,v)$$

where diffusivity depends on both species densities.

## Applications

1. **Spatial segregation**: Competing species avoiding each other
2. **Predator avoidance**: Prey moving away from predators
3. **Swarm behavior**: Collective movement patterns
4. **Population dynamics**: Spatial coexistence mechanisms
5. **Cell biology**: Cell sorting and tissue formation

## Mathematical Features

- Can have non-diagonal diffusion matrices
- May require regularization for well-posedness
- Entropy structure important for analysis
- Can exhibit finite-time blow-up

## References

- Shigesada, N., Kawasaki, K., & Teramoto, E. (1979). *Spatial segregation of interacting species*
- Lou, Y. & Ni, W.M. (1996). *Diffusion, self-diffusion and cross-diffusion*
- Jüngel, A. (2015). *The boundedness-by-entropy method for cross-diffusion systems*
