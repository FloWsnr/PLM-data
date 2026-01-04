# Plate Vibration Equation

## Mathematical Formulation

The plate equation describes thin elastic plate vibrations using the biharmonic operator:

$$\frac{\partial^2 u}{\partial t^2} = -D \nabla^4 u - C \frac{\partial u}{\partial t}$$

where $\nabla^4 = \nabla^2(\nabla^2)$ is the biharmonic operator (bilaplacian):

$$\nabla^4 u = \frac{\partial^4 u}{\partial x^4} + 2\frac{\partial^4 u}{\partial x^2 \partial y^2} + \frac{\partial^4 u}{\partial y^4}$$

This is converted to a first-order system:
- $\frac{\partial u}{\partial t} = v$
- $\frac{\partial v}{\partial t} = -D \nabla^4 u - C v$

where $u$ is displacement and $v$ is velocity.

## Physical Background

The plate equation arises from Kirchhoff-Love thin plate theory, describing:

- **Plate bending**: Elastic deformation of thin plates under load
- **Vibrations**: Oscillatory motion of plates when disturbed
- **Wave propagation**: Compression and bending waves in stiff materials

Key properties:
- Fourth-order spatial derivatives (stiffer than membrane/wave equation)
- Wave dispersion: different wavelengths travel at different speeds
- Damping term dissipates energy over time

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Bending stiffness | $D$ | Plate rigidity coefficient | 0.00001 - 0.001 |
| Damping | $C$ | Energy dissipation rate | 0.0 - 2.0 |

**Note**: The bending stiffness $D$ is related to material properties: $D = \frac{E h^3}{12(1-\nu^2)}$ where $E$ is Young's modulus, $h$ is plate thickness, and $\nu$ is Poisson's ratio.

## Comparison with Wave Equation

| Property | Wave Equation | Plate Equation |
|----------|---------------|----------------|
| Spatial order | 2nd ($\nabla^2$) | 4th ($\nabla^4$) |
| Wave speed | Constant | Frequency-dependent |
| Dispersion | No | Yes |
| Physics | Strings, membranes | Stiff plates |

## Interesting Initial Conditions

**Uniform displacement with Dirichlet boundaries**: Setting $u = -1$ everywhere with $u = 0$ at boundaries creates compression waves that propagate inward from all edges, meeting at the center. This demonstrates the wave-like nature of plate vibrations.

**Localized disturbance**: A Gaussian bump creates outward-propagating circular waves that reflect from boundaries.

## Applications

1. **Structural engineering**: Building floors, bridge decks
2. **Acoustics**: Vibrating panels, speaker cones
3. **Musical instruments**: Cymbals, bells, percussion
4. **Aerospace**: Aircraft skin panels
5. **MEMS devices**: Micro-scale vibrating plates

## Boundary Conditions

Common boundary conditions for plates:
- **Clamped (fixed)**: $u = 0$ and $\partial u/\partial n = 0$
- **Simply supported**: $u = 0$ (Dirichlet) - used in default config
- **Free edge**: Zero moment and shear force

## Numerical Considerations

- **Stiff equation**: Fourth-order derivatives require small time steps
- **Implicit solvers**: Recommended for efficiency
- **CFL condition**: $\Delta t \propto (\Delta x)^2$ (better than $(\Delta x)^4$ for diffusive form)
- **Damping helps**: Non-zero $C$ improves numerical stability

## References

- Timoshenko, S. & Woinowsky-Krieger, S. (1959). *Theory of Plates and Shells*
- Graff, K. F. (1991). *Wave Motion in Elastic Solids*
- VisualPDE: https://visualpde.com/basic-pdes/plate-equation
