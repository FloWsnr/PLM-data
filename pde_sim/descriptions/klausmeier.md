# Klausmeier Model

A vegetation-water reaction-advection-diffusion system modeling pattern formation in dryland ecosystems, producing characteristic banded vegetation patterns.

## Description

The Klausmeier model was developed by Christopher Klausmeier in 1999 to explain the striking banded vegetation patterns observed in semi-arid regions worldwide. These patterns, first documented through aerial photography in the 1940s, appear as alternating stripes of vegetation and bare soil on hillsides.

The model describes the coupled dynamics of:
- **Water (w)**: Rainfall input, evaporation loss, plant uptake, and downslope flow
- **Plant biomass (n)**: Growth dependent on water availability, mortality, and seed dispersal

Key mechanisms:
1. **Water-vegetation feedback**: Plants enhance local water infiltration, creating a positive feedback
2. **Downslope water flow**: Advection term transports water downhill
3. **Competition for water**: Plants deplete water, creating long-range inhibition

The resulting patterns are **traveling waves** that move uphill - vegetation bands migrate toward the water source over ecological timescales. This is distinct from Turing patterns, which are typically stationary.

Observed pattern types:
- **Stripes (tiger bush)**: Regular bands perpendicular to slope on moderate slopes
- **Spots**: Isolated vegetation patches in harsh conditions
- **Gaps/holes**: Isolated bare patches in favorable conditions
- **Labyrinths**: Intermediate irregular patterns

The model has been observed in:
- African Sahel and Horn of Africa
- Western Australia
- Chile
- Chihuahuan Desert (US/Mexico)
- Mediterranean regions

Understanding these patterns is crucial for dryland ecosystem management, as pattern degradation can signal approaching desertification.

## Equations

$$\frac{\partial w}{\partial t} = a - w - wn^2 + V\frac{\partial w}{\partial x} + D_w \nabla^2 w$$

$$\frac{\partial n}{\partial t} = wn^2 - mn + D_n \nabla^2 n$$

Where:
- $w$ is the water density
- $n$ is the plant biomass density
- $a$ is the rainfall rate
- $m$ is the plant mortality rate
- $V$ is the water advection velocity (downslope flow)
- $D_w, D_n$ are diffusion coefficients

Note: The preset uses species names `n` (plants) and `w` (water), with the water equation as species 2.

## Default Config

```yaml
solver: euler
dt: 0.001
dx: 1.0
domain_size: 100

boundary_x: periodic
boundary_y: periodic

num_species: 2

parameters:
  a: 2       # range: [0.01, 10], step: 0.01, rainfall rate
  m: 0.540   # range: [0.2, 1], step: 0.01, plant mortality
  V: 50      # water advection velocity
  D_w: 1     # water diffusion
  D_n: 1     # plant diffusion
```

## Parameter Variants

### KlausmeierModel (Standard)
Base configuration producing regular stripe patterns.
- `a = 2`: Moderate rainfall
- `m = 0.54`: Moderate mortality
- `V = 50`: Strong downslope water flow
- Initial conditions: Random plant biomass [0,1], constant water (w=1)
- Produces traveling wave patterns migrating uphill

### Harsh Environment Regime
- `a = 0.4`, `m = 0.4`: Low rainfall, high stress
- Produces irregular, sparse patterns
- Strong Allee-like effects - requires sufficient initial biomass to establish
- Risk of complete vegetation collapse

### Rainfall Gradients (KlausmeierRainfallGradient)
Extension with spatially varying rainfall.
- `a` replaced with linear function `a(y)`
- Models elevation-dependent rainfall
- Shows how patterns change along precipitation gradients

## Notes

- The plant-extinction state ($n = 0$, $w = a$) is always stable
- Patterns require sufficient initial biomass to overcome extinction stability
- Pattern wavelength depends primarily on $V$ and diffusion coefficients
- Reducing $V$ produces patterns more similar to classical Turing patterns
- The model can exhibit hysteresis between patterned and bare states

## References

- Klausmeier, C. A. (1999). Regular and irregular patterns in semiarid vegetation. Science, 284(5421), 1826-1828.
- Rietkerk, M., et al. (2002). Self-organization of vegetation in arid ecosystems. American Naturalist, 160(4), 524-530.
- Borgogno, F., et al. (2009). Mathematical models of vegetation pattern formation in ecohydrology. Reviews of Geophysics, 47(1), RG1005.
- Eigentler, L. (2020). Modelling dryland vegetation patterns. PhD Thesis, University of Edinburgh.
