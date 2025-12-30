# 2D Vorticity Equation

## Mathematical Formulation

The vorticity diffusion equation (simplified Navier-Stokes):

$$\frac{\partial \omega}{\partial t} = \nu \nabla^2 \omega$$

Full vorticity equation with advection:
$$\frac{\partial \omega}{\partial t} + (\mathbf{u} \cdot \nabla)\omega = \nu \nabla^2 \omega$$

where:
- $\omega = \nabla \times \mathbf{u}$ is the vorticity (curl of velocity)
- $\nu$ is the kinematic viscosity
- $\mathbf{u} = (u, v)$ is the velocity field

## Physical Background

In 2D incompressible flow, vorticity is a **scalar**:
$$\omega = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}$$

The vorticity formulation eliminates pressure from the equations.

**Simplified version** (diffusion only): Models viscous decay of vortices without advection.

**Full version**: Includes vortex transport by the velocity field.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Kinematic viscosity | $\nu$ | Fluid viscosity | 0.001 - 1 |

## Velocity Recovery

For 2D incompressible flow, velocity comes from the **stream function** $\psi$:
$$u = \frac{\partial \psi}{\partial y}, \quad v = -\frac{\partial \psi}{\partial x}$$

where $\omega = -\nabla^2 \psi$ (Poisson equation).

## Vortex Dynamics

Without viscosity ($\nu = 0$):
- Vorticity conserved along particle paths
- Kelvin's circulation theorem
- Point vortices: Singular solutions

With viscosity ($\nu > 0$):
- Vorticity diffuses
- Vortices spread and weaken
- Enstrophy $\int \omega^2 dx$ decreases

## Reynolds Number

$$\text{Re} = \frac{UL}{\nu}$$

- Low Re: Viscosity dominates, smooth flow
- High Re: Advection dominates, turbulent tendencies

## Initial Conditions

**Vortex pair**: Counter-rotating vortices
**Single vortex**: Gaussian or Lamb-Oseen vortex
**Random field**: Turbulence studies

## Lamb-Oseen Vortex

Exact solution for decaying vortex:
$$\omega(r,t) = \frac{\Gamma}{4\pi\nu t}\exp\left(-\frac{r^2}{4\nu t}\right)$$

## Applications

1. **Aerodynamics**: Aircraft wake vortices
2. **Meteorology**: Hurricane dynamics
3. **Oceanography**: Ocean eddies
4. **Engineering**: Mixing processes
5. **Turbulence**: 2D turbulence studies

## 2D vs 3D Vorticity

| Property | 2D | 3D |
|----------|----|----|
| Vorticity | Scalar | Vector |
| Stretching | None | Present |
| Cascade | Inverse (to large scales) | Forward (to small scales) |
| Coherent structures | Long-lived | Transient |

## References

- Batchelor, G.K. (1967). *An Introduction to Fluid Dynamics*
- Majda, A.J. & Bertozzi, A.L. (2002). *Vorticity and Incompressible Flow*
- Saffman, P.G. (1992). *Vortex Dynamics*
