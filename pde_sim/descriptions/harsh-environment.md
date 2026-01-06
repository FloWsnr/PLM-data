# Harsh Environment Model

A logistic reaction-diffusion model demonstrating how boundary conditions determine population survival thresholds in hostile environments.

## Description

This model explores a fundamental question in spatial ecology: **How do hostile boundaries affect population persistence?** Using the classical logistic growth model with different boundary conditions, it demonstrates the interplay between diffusion, growth, and environmental constraints.

The key insight is that Dirichlet boundary conditions (hostile boundaries where population dies at the edge) create a critical relationship between diffusion rate and domain size. If diffusion is too fast relative to domain size, the population cannot persist because individuals diffuse out of the favorable region faster than they can reproduce.

This models real ecological scenarios:
- **Island populations**: Ocean surrounding a favorable habitat
- **Habitat patches**: Populations in fragmented landscapes
- **Reserves**: Protected areas surrounded by uninhabitable regions
- **Laboratory cultures**: Populations in bounded containers with hostile edges

The critical condition for persistence:
$$D < \frac{L^2}{2\pi^2}$$

where $D$ is the diffusion coefficient and $L$ is the domain size. For a domain of $L = 10$, the critical diffusion is approximately $D_c \approx 5.07$.

Key phenomena:
- Below $D_c$: Positive equilibrium exists and is globally stable
- Above $D_c$: Only extinction equilibrium is stable
- The carrying capacity $K$ affects equilibrium density but not the persistence threshold
- Near $D_c$: Very long transients and small equilibrium populations

## Equations

$$\frac{\partial u}{\partial t} = D \nabla^2 u + ru\left(1 - \frac{u}{K}\right)$$

With:
- **Neumann (no-flux) BC**: Population reflects at boundaries - always persists
- **Dirichlet (absorbing) BC**: Population dies at boundaries - may go extinct

Where:
- $u$ is the population density
- $D$ is the diffusion coefficient
- $r$ is the intrinsic growth rate
- $K$ is the carrying capacity
- $L$ is the domain length

Critical diffusion threshold: $D_c = \frac{L^2}{2\pi^2}$

## Default Config

```yaml
solver: euler
dt: 0.0001
dx: 0.05
domain_size: 10

boundary_x: neumann
boundary_y: neumann

num_species: 1

parameters:
  K: 1      # carrying capacity
  r: 1      # growth rate
  D: 0.01   # diffusion coefficient (adjustable)
```

## Parameter Variants

### harshEnvironment (Base Configuration)
Default setup with Neumann boundaries (population always persists).
- `D = 0.01`: Very small diffusion
- `K = 1`, `r = 1`: Standard logistic parameters
- Neumann BC: Population reflects at boundaries
- Initial condition: Sparse random nucleation sites

### Exploration Protocol
1. Start with Neumann BC - population spreads and fills domain
2. Change to Dirichlet BC ($u = 0$ at boundaries) - edges affected
3. Increase $D$ gradually:
   - $D = 4$: Population persists (below critical)
   - $D = 6$: Population goes extinct (above critical)
   - $D \approx 5$: Near threshold - slow dynamics

### High Carrying Capacity
- `K = 1000`: Larger equilibrium population
- Makes near-threshold behavior more visible
- Population amplitude varies more strongly with $D$
- Good for testing $D = 5$ vs $D = 5.2$

## Notes

- The threshold $D_c \approx 5.07$ is independent of $K$
- Near threshold, equilibration timescales become very long
- Clicking adds population, useful for testing stability of extinction state
- Square domain assumed; rectangular domains modify the threshold
- This is the simplest example of a "critical patch size" problem
- Related to KISS (Keep It Simple, Stupid) principle in ecological modeling

## Mathematical Details

For Dirichlet BC on a square domain $[0,L] \times [0,L]$:
- The lowest eigenvalue of the Laplacian is $\lambda_1 = 2\pi^2/L^2$
- Population persists iff $r > D\lambda_1$
- With $r = 1$: $D < L^2/(2\pi^2) \approx L^2/19.74$
- For $L = 10$: $D_c \approx 5.07$

## References

- Skellam, J. G. (1951). Random dispersal in theoretical populations. Biometrika, 38(1/2), 196-218.
- Kierstead, H., & Slobodkin, L. B. (1953). The size of water masses containing plankton blooms. Journal of Marine Research, 12(1), 141-147.
- Cantrell, R. S., & Cosner, C. (2003). Spatial Ecology via Reaction-Diffusion Equations. Wiley.
