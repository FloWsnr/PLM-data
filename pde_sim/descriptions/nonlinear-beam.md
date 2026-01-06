# Nonlinear Beam Equation

A fourth-order equation describing the bending dynamics of elastic beams with state-dependent stiffness, where the bending resistance changes based on local curvature.

## Description

The beam equation, based on Euler-Bernoulli theory (developed in the 18th century by Leonhard Euler and Daniel Bernoulli), describes the relationship between a beam's deflection and the applied loads. It is fundamental to structural engineering and forms the basis for understanding bridges, buildings, aircraft wings, and microelectromechanical systems (MEMS).

In the **overdamped regime** (neglecting inertia), the standard beam equation becomes a fourth-order parabolic PDE. The simulation extends this to include **nonlinear, state-dependent stiffness** - the beam's resistance to bending depends on how much it is already bent.

This nonlinearity captures several physical phenomena:
- **Material nonlinearity**: Some materials become stiffer (or softer) under stress
- **Geometric nonlinearity**: Large deflections change the effective stiffness
- **Smart materials**: Shape-memory alloys, piezoelectrics with feedback
- **Biological structures**: Tendons, cartilage with strain-dependent properties

The state-dependent stiffness creates a feedback loop: regions of high curvature can become either harder or easier to bend further, leading to complex dynamics including localization and pattern formation.

## Equations

The overdamped nonlinear beam equation:

$$\frac{\partial y}{\partial t} = -\frac{\partial^2}{\partial x^2}\left( E\left(\frac{\partial^2 y}{\partial x^2}\right) \frac{\partial^2 y}{\partial x^2} \right)$$

With curvature-dependent stiffness:

$$E = E^\star + \Delta_E \frac{1 + \tanh\left(\frac{\partial^2 y}{\partial x^2}/\epsilon\right)}{2}$$

This is implemented via cross-diffusion with auxiliary variables:

$$\frac{\partial u}{\partial t} = -\frac{\partial w}{\partial x}$$

$$v = \frac{\partial u}{\partial x}$$

$$w = E(v) \cdot v$$

Where:
- $u(x,t) \equiv y$ is the beam deflection
- $v = u_{xx}$ is the curvature (second derivative)
- $w$ is the bending moment
- $E^\star$ is the baseline stiffness
- $\Delta_E$ is the stiffness change range
- $\epsilon$ controls the sharpness of the stiffness transition
- The $\tanh$ function smoothly switches between low and high stiffness

**Physical interpretation**:
- $\Delta_E = 0$: Constant stiffness (linear beam)
- $\Delta_E > 0$: Regions of positive curvature become stiffer
- The transition occurs around $v = 0$ (zero curvature)

## Default Config

```yaml
solver: euler
dt: 0.00004
dx: 0.04
domain_size: 1
dimension: 1

boundary_left: combo (Dirichlet = 0, Neumann = 0.2*sin(2*t))
boundary_right: combo (Dirichlet = 0, Neumann = 0)

parameters:
  E: 0.0001      # baseline stiffness scale
  Delta_E: 24    # [0, 24] - stiffness variation range
  epsilon: 0.01  # stiffness transition sharpness (implicit in tanh)
```

## Parameter Variants

### differentialStiffness (Standard)
Curvature-dependent stiffness demonstration:
- `E = 0.0001`, `Delta_E = 24` (adjustable 0-24)
- Left boundary: clamped with oscillating moment $0.2\sin(2t)$
- Right boundary: clamped (zero displacement and slope)
- 1D line plot visualization
- Brush disabled (watch dynamics unfold)

### BeamEquation (Classic Linear)
Standard Euler-Bernoulli beam:
- `D = 10` (stiffness)
- `Q = 0` (distributed load)
- `C = 0` (damping)
- Boundary: Dirichlet on both ends
- Shows standard beam vibration/relaxation

### Effect of Stiffness Variation

| Delta_E | Behavior |
|---------|----------|
| Delta_E = 0 | Linear beam, uniform stiffness |
| Delta_E small | Slight asymmetry in response |
| Delta_E ~ 12 | Noticeable localization effects |
| Delta_E = 24 | Strong differential stiffness, complex dynamics |

### Physical Applications

- **MEMS resonators**: Nonlinear stiffness affects frequency response
- **Composite materials**: Layered structures with varying properties
- **Biomechanics**: Tendons, ligaments with strain-dependent stiffness
- **Energy harvesting**: Nonlinear beams for vibration energy capture

### Mathematical Note

The fourth-order nature requires careful numerical treatment:
- Two auxiliary variables reduce the order
- Cross-diffusion implements the derivatives
- Small timestep needed for stability
- Boundary conditions on both $u$ and $u_x$

## References

- Euler, L. (1744). "De curvis elasticis" - Additamentum to Methodus inveniendi
- Timoshenko, S.P. (1921). "On the correction for shear of the differential equation for transverse vibrations of prismatic bars" - Phil. Mag. 41:744
- Nayfeh, A.H. & Pai, P.F. (2004). "Linear and Nonlinear Structural Mechanics" - Wiley
- Lacarbonara, W. (2013). "Nonlinear Structural Mechanics" - Springer
