# Allen-Cahn Equation (Bistable Equation)

## Mathematical Formulation

The Allen-Cahn equation with threshold parameter:

$$\frac{\partial u}{\partial t} = D \nabla^2 u + u(u - a)(1 - u)$$

Alternatively written as:
$$\frac{\partial u}{\partial t} = D \nabla^2 u - F'(u)$$

where $F(u) = \frac{u^4}{4} - \frac{(1+a)u^3}{3} + \frac{au^2}{2}$ is the double-well potential.

## Physical Background

The Allen-Cahn equation models phase transitions with two stable states:
- $u = 0$: One phase (e.g., liquid)
- $u = 1$: Other phase (e.g., solid)
- $u = a$: Unstable intermediate state

The equation is a **gradient flow** that minimizes the free energy:
$$\mathcal{F}[u] = \int \left[\frac{D}{2}|\nabla u|^2 + F(u)\right] dx$$

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Diffusion | $D$ | Interface width control | 0.01 - 0.5 |
| Threshold | $a$ | Position of unstable state | 0 - 1 |

## Traveling Front Solutions

The equation admits traveling wave solutions connecting $u = 0$ and $u = 1$:

**Wave speed**: $c = \sqrt{D/2}(1 - 2a)$

- $a < 0.5$: Front moves toward $u = 0$ (phase 1 invades)
- $a = 0.5$: Stationary front (symmetric case)
- $a > 0.5$: Front moves toward $u = 1$ (phase 0 invades)

## Interface Dynamics

The Allen-Cahn equation describes interface motion by **mean curvature**:
$$v_n = D\kappa$$

where $v_n$ is the normal velocity and $\kappa$ is the curvature. This leads to:
- Shrinking of convex regions
- Growth of concave regions
- Eventually all interfaces disappear

## Applications

1. **Phase transitions**: Solidification, melting
2. **Materials science**: Grain boundary motion
3. **Image processing**: Edge detection, segmentation
4. **Population dynamics**: Allee effect (threshold for survival)
5. **Ecology**: Range boundaries

## Comparison with Cahn-Hilliard

| Property | Allen-Cahn | Cahn-Hilliard |
|----------|------------|---------------|
| Conservation | Not conserved | Mass conserved |
| Interface motion | Mean curvature | Surface diffusion |
| Order | 2nd order | 4th order |
| Long-time | Single phase | Coarsening |

## Allee Effect Interpretation

In ecology, the Allen-Cahn equation models populations with strong Allee effect:
- Below threshold $a$: Population declines to extinction
- Above threshold $a$: Population grows to carrying capacity

## References

- Allen, S.M. & Cahn, J.W. (1979). *A Microscopic Theory for Antiphase Boundary Motion*
- Fife, P.C. (1988). *Dynamics of Internal Layers and Diffusive Interfaces*
