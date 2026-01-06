# Klausmeier Model on Topography

An extension of the Klausmeier vegetation model incorporating realistic terrain, where water flows downhill and vegetation patterns follow landscape features.

## Description

This model extends the standard Klausmeier vegetation-water system to account for realistic topographic variation. Instead of assuming a uniform slope with constant water advection, water flow is governed by the actual terrain gradient, allowing exploration of how landscape features affect vegetation patterns.

The key modification is replacing the constant advection term with gradient-driven flow:
$$V \frac{\partial w}{\partial x} \rightarrow V \nabla \cdot (w \nabla T)$$

where $T(x,y)$ is the topographic height function.

This models real-world phenomena:
- **Valley accumulation**: Water collects in low-lying areas
- **Hilltop stress**: Vegetation stress on exposed ridges
- **Pattern disruption**: Irregular terrain breaks regular stripe patterns
- **Uphill migration**: Vegetation bands still migrate toward water sources
- **Preferential colonization**: Vegetation establishes first in water-rich valleys

The simulation uses topographic data (typically from elevation maps) to define $T(x,y)$, creating patterns that reflect actual landscape features rather than idealized uniform slopes.

Applications include:
- Understanding real dryland vegetation patterns
- Predicting restoration outcomes
- Assessing desertification risk on specific terrain
- Exploring climate change impacts on patterned ecosystems

## Equations

$$\frac{\partial w}{\partial t} = a - w - wn^2 + D_w \nabla^2 w + V \nabla \cdot (w \nabla T)$$

$$\frac{\partial n}{\partial t} = wn^2 - mn + D_n \nabla^2 n$$

Where:
- $w$ is the water density
- $n$ is the plant biomass density
- $T(x,y)$ is the topographic height function
- $a$ is the rainfall rate
- $m$ is the plant mortality rate
- $V$ controls the strength of gravity-driven water flow
- $D_w, D_n$ are diffusion coefficients

The advection term $V \nabla \cdot (w \nabla T)$ causes water to flow from high to low elevation, accumulating in valleys.

## Default Config

```yaml
solver: euler
dt: 0.001
dx: 0.5
domain_size: 100

boundary_x: periodic
boundary_y: periodic

num_species: 3  # n, w, T
cross_diffusion: true

parameters:
  a: 1.1     # range: [0.01, 10], step: 0.01, rainfall rate
  m: 0.630   # range: [0.2, 1], step: 0.01, plant mortality
  V: 100     # gravity-driven flow strength
  D_n: 1     # plant diffusion
  D_w: 2     # water diffusion
```

## Diffusion Structure

The system uses cross-diffusion to implement topography-driven water flow:

```
diffusionStr_2_2: "2"      # water self-diffusion
diffusionStr_2_3: "V*w"    # water flux driven by topography gradient
diffusionStr_3_3: "40*ind(t<0.1)"  # initial smoothing of topography
```

The topography $T$ is stored as a third "species" that is initially set from an image and then held constant (with a brief diffusion period to smooth the data).

## Parameter Variants

### KlausmeierOnTopography
Main configuration with topographic input.
- Topography loaded from image file
- Cross-diffusion implements $\nabla \cdot (w \nabla T)$
- 3D surface visualization available
- Initial condition: Topography from image + linear gradient

### Exploration Options
- Decrease $V$: Reduces terrain influence, patterns become more like flat-domain case
- Modify $a, m$: Change vegetation stress level and pattern type
- Use different topography images: Explore various landscape types

## Views

The simulation provides multiple visualization options:
1. **Foliage**: Plant biomass density $n$
2. **Water**: Water density $w$
3. **3D Surface**: Vegetation colored on 3D terrain

## Notes

- Topography is input via image file (grayscale represents elevation)
- Water accumulates in valleys, creating vegetation refugia
- Ridge lines tend to have sparse or absent vegetation
- Patterns interact with terrain features in complex ways
- The model demonstrates why real vegetation patterns are irregular despite simple underlying dynamics
- Can explore your local area by loading custom topography images

## Related Models

- **WaterOnTopography**: Simplified model of water flow alone (thin-film equation)
- **WaterOnTopographySpring**: Water with point sources (springs)
- **KlausmeierRainfallGradient**: Spatially varying rainfall without topography

## References

- Klausmeier, C. A. (1999). Regular and irregular patterns in semiarid vegetation. Science, 284(5421), 1826-1828.
- Saco, P. M., et al. (2007). Eco-geomorphology of banded vegetation patterns in arid and semi-arid regions. Hydrology and Earth System Sciences, 11(6), 1717-1730.
- Siteur, K., et al. (2018). How will increases in rainfall intensity affect semiarid ecosystems? Water Resources Research, 54(7), 4382-4395.
