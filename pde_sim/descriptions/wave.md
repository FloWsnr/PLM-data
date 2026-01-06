# Wave Equation

The wave equation is a fundamental hyperbolic partial differential equation that describes the propagation of waves through a medium, including sound waves, light waves, and vibrations in strings and membranes.

## Description

The wave equation governs how disturbances propagate through elastic media. Unlike the heat equation where disturbances diffuse and smooth out, wave equation solutions maintain their shape as they travel. This fundamental difference arises because the wave equation is second-order in time (hyperbolic) rather than first-order (parabolic).

The equation has broad applications across physics and engineering:

- **Acoustics**: Sound wave propagation through air, water, and solids
- **Vibrating strings**: Guitar strings, piano wires, and other musical instruments
- **Membranes**: Drumheads, speaker cones, and thin elastic sheets
- **Electromagnetism**: Light and radio wave propagation (Maxwell's equations reduce to wave equations)
- **Seismology**: Earthquake wave propagation through Earth's crust
- **Water waves**: Surface waves on liquids (in the linear approximation)

### Key Physical Behaviors

**D'Alembert's solution** (in 1D) shows that any disturbance decomposes into left-traveling and right-traveling waves: $u(x,t) = F(x + ct) + G(x - ct)$. This means an initial "bump" splits into two waves moving in opposite directions at speed $c$.

**Standing waves** occur when waves reflect from boundaries and interfere with themselves. On a rectangular membrane with fixed edges, standing wave patterns (Chladni figures) form with characteristic nodal lines where the membrane doesn't move. The modes are indexed by integers $(n, m)$ representing the number of half-wavelengths in each direction.

**Finite propagation speed** is a defining characteristic - unlike the heat equation, disturbances take finite time to reach distant points. This is consistent with causality in physical systems.

## Equations

The wave equation in two dimensions:

$$\frac{\partial^2 u}{\partial t^2} = D \nabla^2 u$$

where:
- $u(x, y, t)$ is the displacement field
- $D$ is the wave speed squared (often written as $c^2$)
- $\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$ is the Laplacian operator

Since VisualPDE only handles first-order time derivatives, the wave equation is reformulated as a system:

$$\frac{\partial u}{\partial t} = v + C D \nabla^2 u$$

$$\frac{\partial v}{\partial t} = D \nabla^2 u$$

where:
- $v$ represents the velocity field $\partial u / \partial t$
- $C$ is a damping/stabilization parameter (set to 0 for pure wave equation)
- The cross-diffusion term $D \nabla^2 u$ in the $v$ equation couples the fields

In the preset implementation:
- Reaction terms: `u: v`, `v: 0`
- Diffusion: `u-u: C*D`, `v-u: D` (cross-diffusion), `v-v: 0`

## Default Config

```yaml
solver: euler
dt: 0.002
dx: 0.5
domain_size: 100

boundary_x: neumann
boundary_y: neumann

parameters:
  D: 1.0       # range: [1, 100] - wave speed squared
  C: 0.01      # damping coefficient (0 for pure wave equation)

species:
  - name: u    # displacement
  - name: v    # velocity
```

## Parameter Variants

### waveEquation (2D)
The standard 2D wave equation with:
- Parameters: `D = 1` in range `[1, 100]`, `C = 0.01`
- No-flux (Neumann) boundary conditions
- Cross-diffusion formulation

### waveEquationICs
Preset with standing wave initial conditions:
- Initial condition: $u(x,y,0) = \cos(n\pi x/L_x)\cos(m\pi y/L_y)$
- Parameters: `n = 4` in `[1, 10]`, `m = 4` in `[1, 10]`, `C = 0`, `D = 1`
- Demonstrates standing wave modes of rectangular membrane

The analytical solution is:
$$u(x,y,t) = \cos\left(D\pi\sqrt{\frac{n^2}{L_x^2}+\frac{m^2}{L_y^2}}\,t\right)\cos\left(\frac{n \pi x}{L_x}\right)\cos\left(\frac{m \pi y}{L_y}\right)$$

### waveEquation1D
One-dimensional version demonstrating d'Alembert's solution:
- Initial condition: Gaussian pulse $u(x,0) = \exp(-35\pi(x/100-0.5)^2)$
- Shows splitting into left and right traveling waves
- Overlay shows analytical solution for comparison
- Parameters: `D = 1`

## References

- [Wave equation - Wikipedia](https://en.wikipedia.org/wiki/Wave_equation)
- [d'Alembert's formula - Wikipedia](https://en.wikipedia.org/wiki/D'Alembert's_formula)
- [D'Alembert Solution of the Wave Equation - Mathematics LibreTexts](https://math.libretexts.org/Bookshelves/Differential_Equations/Differential_Equations_for_Engineers_(Lebl)/4:_Fourier_series_and_PDEs/4.08:_DAlembert_solution_of_the_wave_equation)
- [Vibrations of Rectangular Membranes - Mathematics LibreTexts](https://math.libretexts.org/Bookshelves/Differential_Equations/Introduction_to_Partial_Differential_Equations_(Herman)/06:_Problems_in_Higher_Dimensions/6.01:_Vibrations_of_Rectangular_Membranes)
- [Standing wave - Wikipedia](https://en.wikipedia.org/wiki/Standing_wave)
