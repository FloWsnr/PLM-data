# Bacteria Advection Model

A simple advection-reaction model describing bacterial concentration decay as it flows downstream in a river reach.

## Description

This is a toy model capturing key aspects of bacterial transport in flowing water. Despite its simplicity, it demonstrates fundamental concepts in environmental transport modeling:

- **Advection**: Bacteria are carried downstream by water flow
- **Decay**: UV exposure and other factors cause bacterial die-off over time
- **Inlet sources**: Contamination enters at upstream boundaries

The model represents a single "reach" - a stretch of river without tributaries or other complications. It answers practical questions:
- How does contamination spread downstream?
- What is the steady-state concentration profile?
- How do flow speed and decay rate interact?

This type of model is used in:
- Water quality assessment
- Pollution tracking
- Recreational water safety
- Wastewater treatment planning

The model is inherently one-dimensional (along the river), though 2D extensions can account for lateral mixing in wide rivers.

## Equations

$$\frac{\partial C}{\partial t} = -u \frac{\partial C}{\partial x} - kC$$

Where:
- $C$ is the bacterial concentration
- $u$ is the flow speed (advection velocity)
- $k$ is the decay rate (die-off coefficient)
- $x$ is the distance along the river

This is a **first-order linear PDE** with:
- Advection term: $-u \frac{\partial C}{\partial x}$ (transport downstream)
- Reaction term: $-kC$ (exponential decay)

### Boundary Conditions
- **Inlet (left)**: Dirichlet condition $C = c_0$ (fixed upstream concentration)
- **Outlet (right)**: Ghost/outflow condition (bacteria leave freely)

### Steady-State Solution
At steady state ($\partial C/\partial t = 0$):
$$C(x) = c_0 \exp\left(-\frac{k x}{u}\right)$$

Concentration decays exponentially with distance, with decay length scale $L = u/k$.

## Default Config

```yaml
solver: euler
dt: 0.01
dx: 1.5
domain_size: 320

dimension: 1

boundary_x: combo  # Dirichlet left, Ghost right

num_species: 1

parameters:
  c0: 0.77   # range: [0, 1], inlet concentration
  k: 0.006   # range: [0, 0.1], decay rate
  u: 0.62    # range: [0.1, 4], flow speed
```

## Parameter Effects

| Parameter | Increase Effect |
|-----------|-----------------|
| $c_0$ | Higher overall concentrations |
| $u$ | Faster transport, longer decay length, more bacteria reach outlet |
| $k$ | Faster decay, shorter decay length, less bacteria reach outlet |

### Characteristic Length Scale
The decay length $L = u/k$ determines how far bacteria travel before significant die-off:
- $u = 0.62$, $k = 0.006$: $L \approx 103$ units
- $u = 2$, $k = 0.006$: $L \approx 333$ units
- $u = 0.62$, $k = 0.05$: $L \approx 12$ units

## Parameter Variants

### bacteriaInAReach (1D)
Base one-dimensional configuration.
- Line plot visualization
- Inlet/outlet boundary conditions
- Clicking adds bacteria at any point

### bacteriaInAReach2D
Two-dimensional extension with realistic river geometry.
- River shape from image mask
- Lateral mixing included
- Domain indicator function restricts flow to river channel
- Additional source term for point discharges

### bacteriaInAReachOscillatoryDecay
Extension with time-varying decay rate.
- $k(t) = k(1 + 0.9\sin(t/30))$
- Models diurnal UV variation
- Creates pulsating concentration patterns

## Notes

- This is a linear model - concentrations superpose
- No diffusion term - transport is purely advective
- Clicking instantly adds bacteria that then wash downstream
- The steady-state profile takes time to establish after parameter changes
- Real rivers have turbulent diffusion, stratification, and other complexities not captured here

## Physical Interpretation

The decay rate $k$ combines multiple processes:
- UV inactivation (dominant in sunlight)
- Temperature-dependent die-off
- Predation by other organisms
- Sedimentation

In reality, $k$ varies with:
- Time of day (UV exposure)
- Season (temperature, light)
- Water depth and clarity
- Bacterial species

## References

- Chapra, S. C. (1997). Surface Water-Quality Modeling. McGraw-Hill.
- Bowie, G. L., et al. (1985). Rates, constants, and kinetics formulations in surface water quality modeling. EPA/600/3-85/040.
- Fischer, H. B., et al. (1979). Mixing in Inland and Coastal Waters. Academic Press.
