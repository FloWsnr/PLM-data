# Kardar-Parisi-Zhang (KPZ) Interface Growth

## Mathematical Formulation

The KPZ equation describes surface/interface growth:

$$\frac{\partial h}{\partial t} = \nu \nabla^2 h + \frac{\lambda}{2} |\nabla h|^2 + \eta(\mathbf{r}, t)$$

where:
- $h$ is the height of the interface above a reference plane
- $\nu$ is the diffusion coefficient (surface tension)
- $\lambda$ is the nonlinear growth coefficient
- $\eta$ is noise (omitted in deterministic version)

## Physical Background

The KPZ equation captures three key physical mechanisms:

1. **Diffusion ($\nu \nabla^2 h$)**: Surface tension smooths the interface
2. **Lateral growth ($\frac{\lambda}{2}|\nabla h|^2$)**: Growth normal to surface causes lateral spread on slopes
3. **Noise ($\eta$)**: Random deposition events (stochastic version)

The characteristic nonlinear term $|\nabla h|^2$ arises from geometric considerations: particles depositing on a tilted surface grow it faster in the normal direction.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Diffusion | $\nu$ | Surface tension/smoothing | 0.01 - 0.5 |
| Growth | $\lambda$ | Lateral growth rate | 0.1 - 2 |

## Universality Class

The KPZ equation defines a **universality class** for surface growth:
- Dynamic exponent $z = 3/2$ (in 1D)
- Roughness exponent $\alpha = 1/2$ (in 1D)
- Growth exponent $\beta = 1/3$ (in 1D)

Many discrete models belong to this class:
- Eden model (tumor/bacterial colony growth)
- Ballistic deposition
- Restricted solid-on-solid (RSOS) models
- Polynuclear growth models

## Special Cases

| Condition | Resulting Equation | Behavior |
|-----------|-------------------|----------|
| $\lambda = 0$ | Edwards-Wilkinson | Linear diffusion, Gaussian |
| $\nu = 0$ | Hamilton-Jacobi | Shock formation |
| Full KPZ | Nonlinear | Universal roughening |

## Applications

1. **Surface growth**: MBE (molecular beam epitaxy), sputtering
2. **Fire fronts**: Forest fire propagation
3. **Bacterial colonies**: Eden model growth
4. **Turbulence**: Burgers equation connection
5. **Directed polymers**: Statistical mechanics
6. **Traffic flow**: Related dynamics

## Initialization

Typical initial conditions:
- **Flat**: $h = 0$ with small random noise
- **Sinusoidal**: Test smoothing behavior
- **Step**: Study interface evolution

## Numerical Considerations

- Explicit methods work for moderate $\lambda$
- Large $\lambda$ can cause instabilities
- Periodic boundaries typical for statistical studies
- Long times needed for scaling regime

## Connection to Other Equations

| Related PDE | Transformation |
|-------------|---------------|
| Burgers | $u = \nabla h$ (velocity field) |
| Kuramoto-Sivashinsky | Additional stabilizing $\nabla^4$ term |
| Hamilton-Jacobi | Deterministic limit |

## References

- Kardar, M., Parisi, G., & Zhang, Y.C. (1986). *Dynamic Scaling of Growing Interfaces*, Physical Review Letters 56
- Barabasi, A.L. & Stanley, H.E. (1995). *Fractal Concepts in Surface Growth*, Cambridge University Press
- Halpin-Healy, T. & Zhang, Y.C. (1995). *Kinetic roughening phenomena, stochastic growth*, Physics Reports
