# Rayleigh-Bénard Thermal Convection

## Mathematical Formulation

Simplified thermal convection equations:

$$\frac{\partial T}{\partial t} = \kappa \nabla^2 T$$
$$\frac{\partial \omega}{\partial t} = \nu \nabla^2 \omega + \text{Ra} \frac{\partial T}{\partial x}$$

Full Boussinesq equations:
$$\frac{\partial T}{\partial t} + (\mathbf{u} \cdot \nabla)T = \kappa \nabla^2 T$$
$$\frac{\partial \omega}{\partial t} + (\mathbf{u} \cdot \nabla)\omega = \nu \nabla^2 \omega + g\alpha\frac{\partial T}{\partial x}$$

where:
- $T$ is temperature
- $\omega$ is vorticity
- $\kappa$ is thermal diffusivity
- $\nu$ is kinematic viscosity
- $\text{Ra}$ is the Rayleigh number
- $g\alpha$ represents buoyancy

## Physical Background

Rayleigh-Bénard convection occurs when fluid is heated from below:

1. Hot fluid is less dense (buoyant)
2. Cold fluid is denser (sinks)
3. At critical Rayleigh number, convection begins
4. Convection rolls or cells form

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Thermal diffusivity | $\kappa$ | Heat conduction rate | 0.001 - 0.1 |
| Kinematic viscosity | $\nu$ | Fluid viscosity | 0.001 - 0.1 |
| Rayleigh number | Ra | Buoyancy/dissipation ratio | 10 - 1000 |

## Rayleigh Number

$$\text{Ra} = \frac{g\alpha\Delta T H^3}{\nu\kappa}$$

where:
- $g$: gravitational acceleration
- $\alpha$: thermal expansion coefficient
- $\Delta T$: temperature difference
- $H$: layer height

## Critical Rayleigh Number

Convection onset at $\text{Ra}_c \approx 1708$ (rigid boundaries)

| Ra Range | Flow State |
|----------|------------|
| Ra < Ra_c | Conduction only |
| Ra_c < Ra < 10⁴ | Steady convection rolls |
| 10⁴ < Ra < 10⁶ | Time-dependent convection |
| Ra > 10⁶ | Turbulent convection |

## Prandtl Number

$$\text{Pr} = \frac{\nu}{\kappa}$$

- Pr << 1: Liquid metals (thermal effects dominate)
- Pr ~ 1: Gases
- Pr >> 1: Oils, Earth's mantle (viscous effects dominate)

## Pattern Formation

Near onset, patterns include:
- **Rolls**: Parallel convection cells
- **Hexagons**: Six-fold symmetric cells
- **Squares**: Four-fold cells (special cases)
- **Spirals**: Time-dependent patterns

## Applications

1. **Atmospheric dynamics**: Cumulus clouds
2. **Oceanography**: Thermohaline circulation
3. **Astrophysics**: Stellar convection
4. **Geophysics**: Mantle convection
5. **Engineering**: Heat exchangers, cooling

## Bénard Cells

Classic pattern: hexagonal cells with:
- Rising hot fluid in center
- Sinking cold fluid at edges
- Characteristic wavelength ~ 2H

## References

- Chandrasekhar, S. (1961). *Hydrodynamic and Hydromagnetic Stability*
- Getling, A.V. (1998). *Rayleigh-Bénard Convection*
- Bodenschatz, E. et al. (2000). *Recent developments in Rayleigh-Bénard convection*
