# Inhomogeneous Heat Equation

## Mathematical Formulation

The inhomogeneous heat equation extends the standard heat equation with a source term:

$$\frac{\partial T}{\partial t} = D \nabla^2 T + f$$

where:
- $T$ is the temperature field
- $D$ is the diffusion coefficient
- $f$ is a source/sink term (can be constant or spatially varying)

## Physical Background

This equation models heat conduction with internal heat generation or absorption:

- **Positive source ($f > 0$)**: Heat is continuously added (e.g., radioactive decay, chemical reactions)
- **Negative source ($f < 0$)**: Heat is continuously removed (e.g., cooling processes)

The steady-state solution satisfies Poisson's equation: $D \nabla^2 T = -f$

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Diffusion coefficient | $D$ | Rate of heat transport | 0.01 - 0.5 |
| Source term | $f$ | Heat generation rate | -10 to +10 |

## Applications

1. **Nuclear reactors**: Heat generation from fission
2. **Electronic cooling**: Joule heating in circuits
3. **Geothermal systems**: Radioactive heating in Earth's interior
4. **Chemical reactors**: Exothermic/endothermic reactions
5. **Biological tissues**: Metabolic heat generation

## Steady-State Behavior

With constant source, the system reaches equilibrium where diffusion balances generation:
- If $f > 0$: Temperature increases until boundary losses balance generation
- If $f < 0$: Temperature decreases to balance the sink

## References

- Carslaw, H.S. & Jaeger, J.C. (1959). *Conduction of Heat in Solids*
