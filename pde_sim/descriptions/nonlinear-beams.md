# Nonlinear Beam/Plate Equation

## Mathematical Formulation

Fourth-order wave equation with geometric nonlinearity:

$$\frac{\partial^2 u}{\partial t^2} = -D \nabla^4 u + \alpha \nabla^2(|\nabla u|^2)$$

Converted to first-order system:
$$\frac{\partial u}{\partial t} = v$$
$$\frac{\partial v}{\partial t} = -D \nabla^4 u + \alpha \nabla^2(|\nabla u|^2)$$

where:
- $u$ is the displacement field
- $v$ is the velocity field
- $D$ is the bending stiffness
- $\alpha$ is the nonlinearity coefficient
- $\nabla^4 = \nabla^2(\nabla^2)$ is the biharmonic operator

## Physical Background

This equation models **thin elastic plates** undergoing large deflections:

1. **Linear bending**: $-D\nabla^4 u$ (Kirchhoff-Love plate theory)
2. **Geometric nonlinearity**: $\alpha\nabla^2(|\nabla u|^2)$ (stretching effects)

At large deflections, in-plane stretching couples to bending, creating nonlinear effects.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Bending stiffness | $D$ | Plate rigidity | 0.01 - 10 |
| Nonlinearity | $\alpha$ | Stretching coupling | 0 - 2 |

## Physical Interpretation

**Bending stiffness** $D = \frac{Eh^3}{12(1-\nu^2)}$
- $E$: Young's modulus
- $h$: Plate thickness
- $\nu$: Poisson's ratio

**Nonlinearity** from von Kármán plate theory:
- Accounts for mid-plane stretching
- Important when deflection $\sim$ thickness

## Linear Dispersion

For linear case ($\alpha = 0$):
$$\omega^2 = D k^4$$

- Higher modes are stiffer
- Different from wave equation ($\omega = ck$)
- Dispersive: wave packets spread

## Applications

1. **Structural engineering**: Vibrating floors, bridges
2. **MEMS/NEMS**: Micro-resonators
3. **Musical instruments**: Drums, cymbals
4. **Aerospace**: Panel flutter
5. **Civil engineering**: Glass panels, membranes

## Boundary Conditions

**Clamped edge**:
- $u = 0$ (no displacement)
- $\partial u/\partial n = 0$ (no rotation)

**Simply supported**:
- $u = 0$ (no displacement)
- $\nabla^2 u = 0$ (no moment)

**Free edge**:
- Moment and shear conditions

## Nonlinear Phenomena

With nonlinearity:
- Frequency-amplitude dependence
- Modal coupling
- Internal resonances
- Chaotic vibrations possible

## Comparison with Beam/String

| Property | String | Beam | Plate |
|----------|--------|------|-------|
| Restoring force | Tension | Bending | Bending |
| Spatial order | 2nd | 4th | 4th (2D) |
| Dispersion | None | $\omega \sim k^2$ | $\omega \sim k^2$ |
| Dimensions | 1D | 1D | 2D |

## Numerical Considerations

- **4th-order spatial**: Stiff, needs implicit methods
- **Two fields**: Position and velocity
- **Energy conservation**: Check in linear limit
- **Boundary conditions**: Require special treatment

## References

- Timoshenko, S. & Woinowsky-Krieger, S. (1959). *Theory of Plates and Shells*
- Nayfeh, A.H. & Mook, D.T. (1979). *Nonlinear Oscillations*
- von Kármán, T. (1910). *Festigkeitsprobleme im Maschinenbau*
