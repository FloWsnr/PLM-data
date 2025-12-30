# Advected Turing Patterns

## Mathematical Formulation

Reaction-diffusion with advection (flow):

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u - v_f\frac{\partial u}{\partial x} + a - u + u^2 v$$
$$\frac{\partial v}{\partial t} = D_v \nabla^2 v - v_f\frac{\partial v}{\partial x} + b - u^2 v$$

where:
- $u, v$ are chemical concentrations
- $D_u, D_v$ are diffusion coefficients
- $v_f$ is the advection velocity (flow speed)
- $a, b$ are production rates

## Physical Background

This model explores how **background flow affects pattern formation**:

1. Without flow: Standard Turing patterns form
2. With flow: Patterns advect and may change character
3. Strong flow: Patterns can be suppressed or modified

Real systems often have flow (blood, rivers, air), so understanding flow-pattern interaction is important.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Activator diffusion | $D_u$ | Small (slow) | 0.01 - 1 |
| Inhibitor diffusion | $D_v$ | Large (fast) | 0.1 - 50 |
| Flow velocity | $v_f$ | Advection speed | 0 - 5 |
| Production a | $a$ | Activator source | 0 - 1 |
| Production b | $b$ | Inhibitor source | 0 - 2 |

## Flow Effects on Patterns

| Flow Strength | Effect |
|---------------|--------|
| None | Static Turing patterns |
| Weak | Slowly drifting patterns |
| Moderate | Pattern wavelength changes |
| Strong | Patterns elongate or suppress |
| Very strong | Uniform state (washout) |

## Pattern Types Under Flow

1. **Drifting spots**: Spots move with flow
2. **Elongated patterns**: Stripes align with flow
3. **Upstream patterns**: Patterns propagate against flow
4. **Stationary in moving frame**: Frame-dependent stationarity
5. **Flow-induced patterns**: Patterns exist only with flow

## Critical Velocity

There may be a **critical flow velocity** above which:
- Turing instability is suppressed
- Patterns cannot form
- System remains uniform

This depends on diffusion ratio and kinetics.

## Absolute vs Convective Instability

**Convective instability**: Perturbations grow but are advected away
**Absolute instability**: Perturbations grow in place

The transition affects whether patterns persist or wash out.

## Applications

1. **River ecosystems**: Vegetation patterns in flowing water
2. **Blood vessels**: Pattern formation in vasculature
3. **Chemical reactors**: Patterns in flow reactors
4. **Atmospheric patterns**: Cloud streets
5. **Developmental biology**: Morphogenesis in growing tissues

## Ecological Context

In rivers/streams:
- Algae/biofilm patterns affected by current
- Nutrient gradients modified by flow
- Upstream/downstream pattern differences

## Numerical Considerations

- Upwind schemes for advection stability
- Periodic BC natural for moving frame
- Careful CFL condition: $\Delta t \leq \Delta x/v_f$
- Large domains to see flow effects

## References

- Rovinsky, A.B. & Menzinger, M. (1992). *Chemical instability induced by a differential flow*
- Satnoianu, R.A. et al. (2000). *Turing instabilities in general systems*
- Klausmeier, C.A. (1999). *Regular and irregular patterns* (vegetation with flow)
