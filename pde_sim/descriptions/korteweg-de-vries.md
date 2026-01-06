# Korteweg-de Vries Equation

The prototypical integrable PDE describing soliton dynamics, where nonlinear and dispersive effects balance to produce stable, particle-like traveling waves.

## Description

The Korteweg-de Vries (KdV) equation, derived by Diederik Korteweg and Gustav de Vries in 1895, describes the evolution of long, weakly nonlinear water waves in shallow channels. It is historically significant as the equation where **solitons** were first mathematically identified and understood.

The remarkable history of solitons began in 1834 when Scottish engineer John Scott Russell observed a "wave of translation" on the Union Canal that traveled for miles without changing shape - contradicting the prevailing understanding of wave propagation. The KdV equation provided the mathematical explanation: a precise balance between nonlinear steepening (which would cause wave breaking) and dispersion (which spreads waves out) can produce stable, localized wave packets.

Key properties of the KdV equation:
- **Complete integrability**: Solved exactly via the Inverse Scattering Transform (Gardner, Greene, Kruskal, Miura, 1967)
- **Infinite conservation laws**: Energy, momentum, and infinitely many other quantities are conserved
- **Soliton solutions**: Localized waves that maintain shape and interact elastically
- **Elastic collisions**: When solitons collide, they pass through each other with only a phase shift - no radiation or energy loss

The KdV equation appears in many physical contexts: shallow water waves, internal waves in stratified fluids, ion-acoustic waves in plasmas, and nonlinear lattice dynamics. It spawned the modern field of integrable systems and has deep connections to algebraic geometry and representation theory.

## Equations

The standard KdV equation:

$$\frac{\partial \phi}{\partial t} = -\frac{\partial^3 \phi}{\partial x^3} - 6\phi \frac{\partial \phi}{\partial x}$$

Using an auxiliary variable $v = \phi_x$ to handle the third derivative via cross-diffusion:

$$\frac{\partial \phi}{\partial t} = -6 v \phi + \text{(third derivative from cross-diffusion)}$$

$$v = \frac{\partial \phi}{\partial x}$$

With cross-diffusion implementing:
- $\frac{\partial \phi}{\partial t} = -\nabla \cdot (v) - 6v\phi = -\frac{\partial v}{\partial x} - 6v\phi$
- The $v = \phi_x$ constraint is maintained through the cross-diffusion structure

Where:
- $\phi(x,t)$ is the wave amplitude (water surface elevation)
- The term $-\phi_{xxx}$ provides dispersion (different wavelengths travel at different speeds)
- The term $-6\phi \phi_x$ provides nonlinear steepening

**Single soliton solution**:
$$\phi(x,t) = \frac{c}{2} \operatorname{sech}^2\left(\frac{\sqrt{c}}{2}(x - ct - x_0)\right)$$

where $c$ is the speed (taller solitons move faster).

## Default Config

```yaml
solver: midpoint
dt: 0.00002
dx: 0.25
domain_size: 40
dimension: 1

boundary_x: periodic

parameters:
  # No adjustable kinetic parameters (inherent to KdV form)
```

## Parameter Variants

### KdV (Two-Soliton Interaction)
Demonstration of soliton collision dynamics:
- Two solitons of different amplitudes (and speeds)
- Initial condition: Sum of two sech^2 profiles
  - Soliton 1: amplitude 2, at x = L/4
  - Soliton 2: amplitude ~1.125 (0.75^2 scaling), at x = 0.4*L
- The faster (taller) soliton overtakes the slower one
- After "collision," both emerge unchanged except for phase shifts
- Midpoint timestepping scheme for better accuracy

### Soliton Properties

| Property | Dependence |
|----------|------------|
| Amplitude | $A$ |
| Speed | $c = 2A$ |
| Width | $\propto 1/\sqrt{A}$ |

Taller solitons are faster AND narrower.

### Conservation Laws
The KdV equation conserves:
1. Mass: $\int \phi \, dx$
2. Momentum: $\int \phi^2 \, dx$
3. Energy: $\int (\phi^3 - \frac{1}{2}\phi_x^2) \, dx$
4. ... infinitely many more

### Related: Zakharov-Kuznetsov (2D Solitons)
The 2D generalization admits "vortical solitons":
$$\frac{\partial u}{\partial t} = -\frac{\partial^3 u}{\partial x^3} - \frac{\partial^3 u}{\partial x \partial y^2} - u\frac{\partial u}{\partial x} - b\nabla^4 u$$

where $b$ is a small dissipative term to reduce radiation.

## References

- Korteweg, D.J. & de Vries, G. (1895). "On the change of form of long waves advancing in a rectangular canal" - Phil. Mag. 39:422
- Zabusky, N.J. & Kruskal, M.D. (1965). "Interaction of 'solitons' in a collisionless plasma" - Phys. Rev. Lett. 15:240
- Gardner, C.S. et al. (1967). "Method for solving the Korteweg-de Vries equation" - Phys. Rev. Lett. 19:1095
- Drazin, P.G. & Johnson, R.S. (1989). "Solitons: An Introduction" - Cambridge University Press
