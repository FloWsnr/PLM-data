# Vortex Dipole Dynamics

## Mathematical Formulation

Vortex dipole evolution with viscous diffusion:

$$\frac{\partial \omega}{\partial t} = \nu \nabla^2 \omega$$

Initial condition: Two **opposite-sign vortices** in close proximity.

## Physical Background

A **vortex dipole** consists of two counter-rotating vortices:
- Positive vortex (counterclockwise)
- Negative vortex (clockwise)
- Separated by distance $d$

The mutual interaction causes **self-propulsion**: the dipole translates perpendicular to the line connecting the vortices.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Viscosity | $\nu$ | Kinematic viscosity | 0.001 - 0.1 |
| Separation | $d$ | Vortex spacing | 0.05 - 0.3 |

## Dipole Velocity

For point vortices of circulation $\pm\Gamma$ separated by $d$:

$$V_{\text{dipole}} = \frac{\Gamma}{2\pi d}$$

The dipole moves perpendicular to the separation axis.

## Gaussian Vortices

For finite-core Gaussian vortices:
$$\omega(r) = \frac{\Gamma}{2\pi\sigma^2}\exp\left(-\frac{r^2}{2\sigma^2}\right)$$

Dipole speed depends on core size $\sigma$ relative to separation $d$.

## Viscous Evolution

With viscosity:
1. **Core spreading**: Vortex cores grow as $\sigma \sim \sqrt{\nu t}$
2. **Weakening**: Peak vorticity decreases
3. **Slowing**: Dipole speed decreases
4. **Eventual decay**: Vortices merge and cancel

## Lamb-Chaplygin Dipole

An exact steady solution exists:
- Continuous vorticity distribution
- Translates at constant speed
- Self-similar in moving frame

## Dipole Interactions

Multiple dipoles can:
- Pass through each other
- Scatter at angles
- Exchange partners
- Form more complex structures

## Applications

1. **Geophysical flows**: Atmospheric/oceanic vortex pairs
2. **Swimming/flying**: Propulsive jet vortices
3. **Turbulence**: Coherent structures
4. **Mixing**: Dipoles enhance mixing
5. **Pollutant transport**: Vortex-driven dispersion

## Experimental Observations

Dipoles observed in:
- Laboratory rotating tanks
- Soap films
- Stratified fluids
- Electromagnetic fluid experiments

## Dipole Formation

Created by:
- Jet injection (starting jet)
- Paddle motion
- Density differences
- Boundary layer separation

## Conservation

Inviscid dipole conserves:
- Total impulse (dipole strength)
- Total circulation (zero for symmetric dipole)
- Kinetic energy

Viscous: Energy decays, impulse may be preserved approximately.

## References

- Saffman, P.G. (1992). *Vortex Dynamics*
- Couder, Y. & Basdevant, C. (1986). *Experimental and numerical study of vortex couples*
- van Heijst, G.J.F. & Flór, J.B. (1989). *Dipole formation and collisions in a stratified fluid*
