# 2D Navier-Stokes (Vorticity-Stream Function)

## Mathematical Formulation

Vorticity transport with background flow:

$$\frac{\partial \omega}{\partial t} = \nu \nabla^2 \omega - u_x\frac{\partial \omega}{\partial x} - u_y\frac{\partial \omega}{\partial y}$$

Full vorticity equation:
$$\frac{\partial \omega}{\partial t} + (\mathbf{u} \cdot \nabla)\omega = \nu \nabla^2 \omega$$

where velocity is recovered from stream function:
$$\nabla^2 \psi = -\omega$$
$$u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$$

## Physical Background

The 2D Navier-Stokes equations describe **incompressible viscous flow**. In 2D:
- Vorticity is a scalar (perpendicular to plane)
- No vortex stretching (unlike 3D)
- Vorticity-stream function formulation eliminates pressure

The simplified version uses a **background velocity** rather than solving for $\psi$.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Viscosity | $\nu$ | Kinematic viscosity | 0.001 - 0.1 |
| Background velocity x | $u_x$ | Mean flow in x | -1 to 1 |
| Background velocity y | $u_y$ | Mean flow in y | -1 to 1 |

## Reynolds Number

$$\text{Re} = \frac{UL}{\nu}$$

| Re Range | Flow Regime |
|----------|-------------|
| Re < 1 | Creeping flow (Stokes) |
| 1 < Re < 100 | Laminar |
| 100 < Re < 1000 | Transition |
| Re > 1000 | Turbulent tendencies |

## 2D Turbulence

Unlike 3D, 2D turbulence has:
- **Inverse energy cascade**: Energy flows to large scales
- **Forward enstrophy cascade**: Enstrophy flows to small scales
- **Coherent vortices**: Long-lived structures

## Vortex Interactions

- **Co-rotating vortices**: Orbit each other
- **Counter-rotating (dipole)**: Translate together
- **Merger**: Close vortices can merge
- **Filamentation**: Stretched vorticity ribbons

## Initial Conditions

**Vortex pair**: Counter-rotating for dipole motion
**Single vortex**: Studies decay
**Random field**: Turbulence initialization

## Conservation Laws

Inviscid ($\nu = 0$):
- Energy: $E = \frac{1}{2}\int |\mathbf{u}|^2 dA$
- Enstrophy: $Z = \frac{1}{2}\int \omega^2 dA$
- Circulation: $\Gamma = \oint \mathbf{u} \cdot d\mathbf{l}$

Viscous: Energy and enstrophy decrease.

## Applications

1. **Geophysical flows**: Atmosphere, ocean
2. **Engineering**: 2D flow approximations
3. **Plasma physics**: Magnetized plasmas
4. **Soap films**: Nearly 2D flows
5. **Turbulence theory**: Fundamental studies

## Numerical Methods

- **Stream function solve**: Poisson equation at each step
- **Spectral methods**: Efficient for periodic domains
- **Finite difference**: Flexible boundaries
- **Vortex methods**: Lagrangian tracking

## References

- Batchelor, G.K. (1967). *An Introduction to Fluid Dynamics*
- Kraichnan, R.H. & Montgomery, D. (1980). *Two-dimensional turbulence*
- Tabeling, P. (2002). *Two-dimensional turbulence: a physicist approach*
