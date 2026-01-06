# Bistable Allen-Cahn Equation

A bistable reaction-diffusion equation modeling invasion and persistence with Allee effects, where waves can advance or retreat depending on parameters.

## Description

The bistable Allen-Cahn equation (also called the Chaffee-Infante or Nagumo equation in some contexts) describes systems with two stable equilibria. Unlike the Fisher-KPP equation where zero is always unstable, here both $u = 0$ (extinction) and $u = 1$ (persistence) are stable, with an unstable intermediate state at $u = a$.

This bistability has profound biological implications:
- Small populations below the threshold $a$ go extinct even locally
- Populations above the threshold can persist and spread
- The direction of wave propagation depends on which stable state is "more stable"

The parameter $a \in (0,1)$ determines the wave direction:
- $a < 0.5$: Waves expand ($u = 1$ invades $u = 0$)
- $a > 0.5$: Waves contract ($u = 0$ invades $u = 1$)
- $a = 0.5$: Stationary waves (balanced bistability)

This models the **strong Allee effect** in ecology, where populations below a critical density face negative per-capita growth due to:
- Difficulty finding mates
- Insufficient cooperative defense against predators
- Loss of social benefits

Key phenomena:
- **Spatial Allee effects**: Small initial populations may fail to establish even when $u = 1$ is stable
- **Critical patch size**: Initial populations must exceed a minimum size to persist
- **Wave reversal**: Changing $a$ can reverse the direction of invasion
- **Eradication possibility**: Invasive species can potentially be driven to extinction

## Equations

$$\frac{\partial u}{\partial t} = D \nabla^2 u + u(u-a)(1-u)$$

Where:
- $u$ is the population density (scaled to [0,1])
- $D$ is the diffusion coefficient
- $a \in (0,1)$ is the Allee threshold parameter

The wave speed is:
$$c = \sqrt{\frac{D}{2}} (1 - 2a)$$

Or equivalently, $c \propto \int_0^1 u(u-a)(1-u)\, du = \frac{1-2a}{12}$

### With Advection
$$\frac{\partial u}{\partial t} = D \nabla^2 u + u(u-a)(1-u) + V(\cos\theta \, u_x + \sin\theta \, u_y)$$

Where $V$ is flow velocity and $\theta$ is flow direction.

## Default Config

```yaml
solver: euler
dt: 0.005
dx: 0.25
domain_size: 100

boundary_x: neumann
boundary_y: neumann

num_species: 1

parameters:
  a: 0.5   # range: [0, 1], step: 0.01, Allee threshold
  D: 1     # diffusion coefficient (implicit)
```

## Parameter Variants

### bistableTravellingWave
Base configuration for exploring wave direction dependence on $a$.
- `a = 0.5` (range: [0, 1]): Critical threshold
- Neumann boundary conditions
- Vector field visualization shows wave direction
- Test: `a = 0.4` (expansion), `a = 0.6` (contraction)

### bistableSurvival
Configuration for studying spatial Allee effects and critical patch sizes.
- `a = 0.3` (range: [0, 1]): Below midpoint, expansion should occur
- `D = 1` (range: [0, 2]): Diffusion strength
- `R = 6` (range: [0, 10]): Initial population patch radius
- Small R: Diffusion spreads population below threshold, leading to extinction
- Large R: Population establishes and invades
- Critical R depends on $a$ and $D$

### BistableAdvection
Configuration with advection (e.g., river flow).
- Adds flow term with velocity $V$ and direction $\theta$
- `a = 0.48`: Near critical threshold
- Flow can assist or hinder invasion depending on direction
- Models populations in flowing water

## Notes

- Unlike Fisher-KPP, this equation can model both invasion and retreat
- The critical patch size for establishment depends on $a$, $D$, and geometry
- Near $a = 0.5$, the system is very sensitive to initial conditions
- The equation is sometimes written with the reaction term as $-u(1-u)(u-a)$ with different sign convention
- Also appears in materials science (phase field models) and neuroscience (excitable media)

## References

- Allen, S. M., & Cahn, J. W. (1979). A microscopic theory for antiphase boundary motion and its application to antiphase domain coarsening. Acta Metallurgica, 27(6), 1085-1095.
- Barton, N. H., & Turelli, M. (2011). Spatial waves of advance with bistable dynamics: cytoplasmic and genetic analogues of Allee effects. American Naturalist, 178(3), E48-E75.
- Lewis, M. A., & Kareiva, P. (1993). Allee dynamics and the spread of invading organisms. Theoretical Population Biology, 43(2), 141-158.
