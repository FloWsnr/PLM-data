# Shallow Water Equations

## Mathematical Formulation

Linearized shallow water equation (height component):

$$\frac{\partial h}{\partial t} = c^2 \nabla^2 h$$

Full shallow water equations:
$$\frac{\partial h}{\partial t} + H\left(\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right) = 0$$
$$\frac{\partial u}{\partial t} + g\frac{\partial h}{\partial x} = 0$$
$$\frac{\partial v}{\partial t} + g\frac{\partial h}{\partial y} = 0$$

where:
- $h$ is the surface elevation perturbation
- $H$ is the mean water depth
- $u, v$ are depth-averaged velocities
- $g$ is gravitational acceleration
- $c = \sqrt{gH}$ is the wave speed

## Physical Background

Shallow water equations describe **long wavelength** surface waves where:
- Wavelength $\lambda \gg H$ (water depth)
- Vertical motion negligible
- Hydrostatic pressure distribution

The linearized version reduces to a wave/diffusion-like equation.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Wave speed | $c$ | $\sqrt{gH}$ | 0.1 - 10 |

## Approximation Validity

Valid when:
- $\lambda/H \gg 1$ (long waves)
- $a/H \ll 1$ (small amplitude), for linear version
- No significant vertical structure

## Wave Properties

**Phase speed**: $c = \sqrt{gH}$ (non-dispersive)

**Group speed**: Same as phase speed (no dispersion)

All wavelengths travel at the same speed in linear shallow water.

## Phenomena Described

1. **Tsunami propagation**: Classic application
2. **Tidal waves**: Ocean basin resonance
3. **Storm surges**: Wind-driven elevation
4. **River flooding**: Dam break, flood waves
5. **Lake oscillations**: Seiches

## Tsunami Example

Pacific Ocean: $H \approx 4000$ m
$$c = \sqrt{9.8 \times 4000} \approx 200 \text{ m/s} \approx 700 \text{ km/h}$$

Wavelength: 100-500 km (much larger than depth)

## Conservation Properties

Full shallow water conserves:
- **Mass**: $\int h \, dA$
- **Momentum**: $\int \mathbf{u} \, dA$
- **Energy**: $\int (h^2 + Hu^2) dA$

## Nonlinear Effects

For finite amplitude waves:
- Wave steepening
- Bore/shock formation
- Breaking at shoreline

## Initial Conditions

**Water drop**: Gaussian perturbation
**Dam break**: Step function
**Oscillating basin**: Seiches

## Applications

1. **Oceanography**: Tide and tsunami modeling
2. **River hydraulics**: Flood prediction
3. **Coastal engineering**: Wave run-up
4. **Atmospheric dynamics**: Gravity waves
5. **Astrophysics**: Accretion disk dynamics

## References

- Pedlosky, J. (1987). *Geophysical Fluid Dynamics*
- Vreugdenhil, C.B. (1994). *Numerical Methods for Shallow-Water Flow*
- LeVeque, R.J. (2002). *Finite Volume Methods for Hyperbolic Problems*
