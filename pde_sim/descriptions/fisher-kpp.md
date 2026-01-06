# Fisher-KPP Equation

The classical reaction-diffusion equation for population spread, describing traveling waves of invasion with logistic growth.

## Description

The Fisher-KPP equation (named after Ronald Fisher, Andrey Kolmogorov, Ivan Petrovsky, and Nikolai Piskunov) is one of the most important equations in mathematical biology. Fisher proposed it in 1937 to model the spatial spread of an advantageous gene through a population, while Kolmogorov, Petrovsky, and Piskunov independently analyzed its mathematical properties.

The equation combines two fundamental processes:
1. **Diffusion**: Random spatial movement of individuals
2. **Logistic growth**: Population growth that saturates at a carrying capacity

The key result is the existence of **traveling wave solutions** - moving fronts that describe the invasion of a species into vacant territory. The wave speed is determined by:
$$c = 2\sqrt{rD}$$
where $r$ is the growth rate and $D$ is the diffusion coefficient. This is the minimum speed selected by biologically relevant initial conditions.

Applications include:
- Species invasion and range expansion
- Epidemic spread (SIS model reduces to Fisher-KPP)
- Tumor growth
- Gene propagation
- Chemical reaction fronts

Key mathematical features:
- The wave speed depends only on $r$ and $D$, not on carrying capacity $K$
- Circular waves travel slightly slower than planar waves due to curvature effects
- The tail structure of initial conditions can affect transient wave speeds
- The equation admits smooth (not sharp) fronts

## Equations

$$\frac{\partial u}{\partial t} = D \nabla^2 u + ru\left(1 - \frac{u}{K}\right)$$

Where:
- $u$ is the population density
- $D$ is the diffusion coefficient
- $r$ is the intrinsic growth rate
- $K$ is the carrying capacity

### Epidemiological Form (SIS Model)
For the proportion of infected individuals $p$:
$$\frac{\partial p}{\partial t} = D \nabla^2 p + \beta p(1-p) - \delta p$$

Traveling waves exist when $R_0 = \beta/\delta > 1$.

## Default Config

```yaml
solver: euler
dt: 0.0005
dx: 1.0
domain_size: 100

boundary_x: neumann
boundary_y: neumann

num_species: 1

parameters:
  K: 1     # range: [0.5, 1.5], carrying capacity
  r: 1     # range: [0, 5], step: 0.1, growth rate
  D: 1     # diffusion coefficient (implicit in nabla^2)
```

## Parameter Variants

### travellingWave (2D)
Standard 2D configuration for visualizing planar and circular waves.
- Default brush: vertical line (for planar waves)
- Neumann boundary conditions
- Initial condition: zero (paint initial population)
- Wave speed scales as $c \propto \sqrt{rD}$

### travellingWave1D
One-dimensional version for cleaner visualization of wave profiles.
- Initial condition: Step function via tanh
- Useful for measuring wave speeds
- Demonstrates wave profile shape

### Experimental Explorations
- Simultaneously decrease $r$ and increase $D$: Same wave speed, different profile
- Carrying capacity $K$ affects equilibrium amplitude but not wave speed
- Circular brushes show curvature effects on wave propagation

## Notes

- The carrying capacity $K$ does not affect the wave speed, only the steady-state population level
- Wave speed formula $c = 2\sqrt{rD}$ is exact for semi-infinite domains with localized initial conditions
- Initial condition tails can lead to accelerating or variable wave speeds (see algebraic vs exponential tails)
- The Fisher-KPP equation cannot model population retreat - for that, see the bistable Allen-Cahn equation

## References

- Fisher, R. A. (1937). The wave of advance of advantageous genes. Annals of Eugenics, 7(4), 355-369.
- Kolmogorov, A., Petrovsky, I., & Piskunov, N. (1937). Study of the diffusion equation with growth of the quantity of matter and its application to a biological problem. Moscow University Mathematics Bulletin, 1, 1-25.
- Murray, J. D. (2002). Mathematical Biology I: An Introduction (3rd ed.). Springer.
