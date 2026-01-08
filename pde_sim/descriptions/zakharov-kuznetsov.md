# Zakharov-Kuznetsov Equation

The 2D generalization of the Korteweg-de Vries (KdV) equation, supporting vortical solitons that propagate in two spatial dimensions.

## Description

The Zakharov-Kuznetsov (ZK) equation, introduced by V.E. Zakharov and E.A. Kuznetsov in 1974, extends the famous KdV equation to two spatial dimensions. While the 1D KdV equation describes solitons in shallow water waves, the ZK equation models ion-acoustic waves in magnetized plasmas and other 2D wave phenomena.

The key insight is that the 1D third derivative $\partial^3 u / \partial x^3$ generalizes to include a mixed derivative term $\partial^3 u / (\partial x \partial y^2)$, allowing the equation to support truly two-dimensional dynamics. This enables **vortical solitons** - radially symmetric, localized wave structures that propagate in a preferred direction (typically +x).

Key properties of the ZK equation:
- **2D generalization**: Extends KdV soliton physics to two dimensions
- **Vortical solitons**: Supports radially symmetric, localized wave packets
- **Directional propagation**: Solitons travel in the positive x direction
- **Near-integrability**: While not completely integrable like 1D KdV, exhibits soliton-like behavior
- **Dissipation term**: A small biharmonic dissipation $b \nabla^4 u$ is often added to reduce radiation

The equation arises naturally in plasma physics when considering weakly nonlinear waves propagating at a small angle to a magnetic field. It has also been studied as a mathematical model for understanding how integrability properties change in higher dimensions.

## Equations

The modified Zakharov-Kuznetsov equation:

$$\frac{\partial u}{\partial t} = -\frac{\partial^3 u}{\partial x^3} - \frac{\partial^3 u}{\partial x \partial y^2} - u \frac{\partial u}{\partial x} - b \nabla^4 u$$

This can be simplified using the identity:
$$-\frac{\partial^3 u}{\partial x^3} - \frac{\partial^3 u}{\partial x \partial y^2} = -\frac{\partial}{\partial x}\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = -\frac{\partial}{\partial x}(\nabla^2 u)$$

Giving the equivalent form used in the implementation:
$$\frac{\partial u}{\partial t} = -\frac{\partial}{\partial x}(\nabla^2 u) - u \frac{\partial u}{\partial x} - b \nabla^4 u$$

Where:
- $u(x,y,t)$ is the wave amplitude
- $\partial(\nabla^2 u)/\partial x$ combines the dispersive terms (more numerically stable)
- $u \partial u / \partial x$ provides nonlinear steepening
- $b \nabla^4 u$ is a small biharmonic dissipation to reduce radiation

**Vortical soliton initial condition**:
$$u(x,y,0) = \operatorname{sech}^2\left(a (x^2 + y^2)\right)$$

where $a$ controls the soliton width.

## Default Config

```yaml
preset: zakharov-kuznetsov
parameters:
  b: 0.008

init:
  type: offset-soliton
  params:
    a: 0.06

solver: midpoint
dt: 0.005
domain_size: 350
resolution: 350

bc:
  x-: periodic
  x+: periodic
  y-: periodic
  y+: periodic
```

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| b | 0.008 | 0 - 0.1 | Biharmonic dissipation coefficient |

### Initial Condition Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| a | 0.06 | Soliton width parameter (larger = narrower) |
| x_center | varies | x-position of soliton center |
| y_center | varies | y-position of soliton center |

## Dynamics

The vortical soliton dynamics include:

1. **Propagation**: The soliton moves in the positive x direction
2. **Shape preservation**: Like 1D solitons, the shape is maintained during propagation
3. **Weak radiation**: Some dispersive radiation may be emitted, reduced by the dissipation term
4. **Periodic boundaries**: Soliton wraps around and re-enters from the left

### Comparison with 1D KdV

| Property | 1D KdV | 2D Zakharov-Kuznetsov |
|----------|--------|----------------------|
| Spatial dimensions | 1D | 2D |
| Complete integrability | Yes | No |
| Soliton type | 1D pulse | Vortical (radial) |
| Conservation laws | Infinite | Fewer |
| Physical context | Shallow water | Magnetized plasma |

## References

- Zakharov, V.E. & Kuznetsov, E.A. (1974). "On three-dimensional solitons" - Sov. Phys. JETP 39:285
- Laedke, E.W. & Spatschek, K.H. (1982). "Nonlinear ion-acoustic waves in weak magnetic fields" - Phys. Fluids 25:985
- Iwasaki, H., Toh, S. & Kawahara, T. (1990). "Cylindrical quasi-solitons of the Zakharov-Kuznetsov equation" - Physica D 43:293
- VisualPDE (2024). "Solitons - Zakharov-Kuznetsov" - visualpde.com
