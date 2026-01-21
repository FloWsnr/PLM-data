# Sine-Gordon Equation

The Sine-Gordon equation is a fundamental nonlinear wave equation famous for supporting topological solitons called kinks and antikinks. These solitons represent stable, localized transitions between different ground states of the system.

## Description

The Sine-Gordon equation, first studied by Edmond Bour in 1862 and named as a pun on the Klein-Gordon equation, describes a rich variety of nonlinear wave phenomena. Unlike ordinary waves that disperse and dissipate, Sine-Gordon solitons maintain their shape and can interact elastically or form bound states called breathers.

The equation is **completely integrable** in one dimension, meaning it possesses an infinite number of conserved quantities and can be solved exactly using inverse scattering transform methods. This makes it one of the most important exactly solvable nonlinear equations in physics.

Physical systems modeled by the Sine-Gordon equation include:

- **Josephson junctions**: Superconducting weak links where kinks represent magnetic flux quanta (fluxons)
- **Crystal dislocations**: Edge dislocations in crystalline materials
- **DNA dynamics**: Torsional deformations of the double helix
- **Pendulum chains**: Coupled mechanical pendulums (the original physical model)
- **Magnetic domain walls**: Transitions between magnetic domains in ferromagnets
- **Elementary particle physics**: Relativistic field theory models

### Key Physical Behaviors

**Kink solitons** are topological transitions from one ground state (phi = 0) to another (phi = 2*pi). They cannot be continuously deformed to zero and thus are topologically protected. The kink profile is:

$$\phi_{\text{kink}}(x) = 4 \arctan\left(\exp\left(\frac{x - x_0}{w}\right)\right)$$

where $w$ is the characteristic width. Kinks can move at any velocity $v < c$ (the wave speed), with their width contracting due to Lorentz contraction:

$$w \to \frac{w}{\gamma} = w\sqrt{1 - v^2/c^2}$$

**Antikinks** are the reverse transition from phi = 2*pi to phi = 0, with opposite topological charge.

**Kink-antikink collisions** exhibit remarkable behavior depending on their initial velocities:
- At low velocities: They attract, oscillate, and can form a bound breather state
- At critical velocities: Complex resonance windows appear
- At high velocities: They pass through each other (nearly elastic scattering)

**Breathers** are localized, oscillating bound states that can be viewed as kink-antikink pairs that oscillate without annihilating. The breather frequency is lower than the linear wave frequency.

**Ring solitons** in 2D are circular wave fronts that can expand or contract, though they are less stable than 1D kinks due to curvature effects.

## Equations

The Sine-Gordon equation in two spatial dimensions:

$$\frac{\partial^2 \phi}{\partial t^2} = c^2 \nabla^2 \phi - \sin(\phi)$$

where:
- $\phi(x, y, t)$ is the field variable (often interpreted as a phase angle)
- $c$ is the characteristic wave speed
- $\nabla^2 = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2}$ is the Laplacian operator

The $-\sin(\phi)$ term creates a periodic potential with ground states at $\phi = 2\pi n$ for integer $n$.

Converted to a first-order system with optional stabilization:

$$\frac{\partial \phi}{\partial t} = \psi + \gamma \nabla^2 \phi$$

$$\frac{\partial \psi}{\partial t} = c^2 \nabla^2 \phi - \sin(\phi)$$

where:
- $\psi$ represents the velocity field $\partial \phi / \partial t$
- $\gamma$ is a small damping/stabilization parameter (set to 0 for the pure equation)

The kink soliton solution in 1D:

$$\phi(x,t) = 4 \arctan\left(\exp\left(\gamma_L \frac{x - vt - x_0}{w}\right)\right)$$

where $\gamma_L = 1/\sqrt{1 - v^2/c^2}$ is the Lorentz factor.

## Default Config

```yaml
preset: sine-gordon
parameters:
  c: 1.0       # wave speed
  gamma: 0.01  # damping coefficient

init:
  type: kink-antikink
  params:
    width: 2.0
    v_kink: 0.3
    v_antikink: -0.3

solver: euler
dt: 0.01
t_end: 100.0
resolution: 128
domain_size: 50

bc:
  x-: periodic
  x+: periodic
  y-: periodic
  y+: periodic
```

## Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| c | 1.0 | 0.1 - 10 | Wave propagation speed |
| gamma | 0.01 | 0 - 1 | Damping/stabilization coefficient |

### Initial Condition Types

| Type | Description |
|------|-------------|
| kink | Single kink soliton (0 to 2*pi transition) |
| antikink | Single antikink (2*pi to 0 transition) |
| kink-antikink | Kink-antikink pair for collision studies |
| breather | Oscillating bound state |
| ring | Circular ring soliton (2D) |
| random | Random perturbations |

### Initial Condition Parameters

| Parameter | Description |
|-----------|-------------|
| x0 | Position of soliton center |
| width | Characteristic width of soliton |
| velocity | Initial velocity (for kink/antikink) |
| v_kink, v_antikink | Velocities for kink-antikink pair |
| omega | Internal frequency for breather |
| radius | Initial radius for ring soliton |

## Dynamics

### Kink Dynamics

A single kink propagates at constant velocity, maintaining its shape. The kink connects the vacuum states:
- $\phi \to 0$ as $x \to -\infty$
- $\phi \to 2\pi$ as $x \to +\infty$

The kink carries topological charge $Q = (\phi(+\infty) - \phi(-\infty))/(2\pi) = +1$.

### Kink-Antikink Collisions

The collision dynamics depend sensitively on the initial velocity:
- **Resonance windows**: At certain velocities, the kink-antikink pair bounces multiple times before escaping
- **Capture**: At low velocities, they can form a bound breather state
- **Transmission**: At high velocities, they pass through each other

### Breather Solutions

Breathers are exact solutions that oscillate in time while remaining localized in space:

$$\phi_{\text{breather}}(x,t) = 4 \arctan\left(\frac{\sin(\omega t)}{\omega \cosh(x/w)}\right)$$

where $\omega < 1$ is the internal oscillation frequency.

### 2D Effects

In two dimensions, the Sine-Gordon equation loses complete integrability. However:
- Line solitons (1D kinks extended in y) remain stable
- Ring solitons can form but may be unstable to collapse or expansion
- More complex patterns can emerge from interactions

## References

- Perring, J.K. & Skyrme, T.H.R. (1962). "A Model Unified Field Equation" - Nuclear Physics 31:550
- Scott, A.C. (1969). "A nonlinear Klein-Gordon equation" - American Journal of Physics 37:52
- Ablowitz, M.J. & Clarkson, P.A. (1991). "Solitons, Nonlinear Evolution Equations and Inverse Scattering" - Cambridge University Press
- Dauxois, T. & Peyrard, M. (2006). "Physics of Solitons" - Cambridge University Press
- [Sine-Gordon equation - Wikipedia](https://en.wikipedia.org/wiki/Sine-Gordon_equation)
- [Sine-Gordon Kinks - VisualPDE](https://visualpde.com/nonlinear-physics/sine-gordon.html)
