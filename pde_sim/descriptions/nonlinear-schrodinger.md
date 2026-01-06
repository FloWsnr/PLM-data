# Nonlinear Schrodinger Equation

An integrable wave equation describing optical solitons, Bose-Einstein condensates, and water waves, where focusing or defocusing nonlinearity leads to bright or dark solitons.

## Description

The Nonlinear Schrodinger (NLS) equation is a universal model for weakly nonlinear dispersive wave packets. It arises as the leading-order envelope equation in numerous physical contexts where dispersion and weak nonlinearity interact, including:

- **Fiber optics**: Pulse propagation in optical fibers, enabling soliton-based communication
- **Water waves**: Deep water wave packet dynamics
- **Bose-Einstein condensates**: Mean-field description (Gross-Pitaevskii equation)
- **Plasma physics**: Langmuir wave dynamics
- **Nonlinear lattices**: Continuum limit of coupled oscillators

The NLS equation is completely integrable via the Inverse Scattering Transform, sharing this property with the KdV equation. This integrability leads to:
- Exact multi-soliton solutions
- Infinite conservation laws
- Elastic soliton collisions

**Focusing vs. Defocusing**:
The sign of the nonlinearity parameter $\kappa$ determines fundamentally different behavior:
- **Focusing** ($\kappa > 0$): Self-focusing nonlinearity, supports **bright solitons** - localized pulses that maintain shape
- **Defocusing** ($\kappa < 0$): Self-defocusing nonlinearity, supports **dark solitons** - localized dips on a continuous background

Bright solitons in focusing NLS enabled the concept of soliton-based optical fiber communication, where pulses propagate without spreading over transcontinental distances.

## Equations

The nonlinear Schrodinger equation (in standard physics form):

$$i \frac{\partial \psi}{\partial t} = -\nabla^2 \psi + \kappa \psi |\psi|^2$$

Or equivalently (multiplying by $-i$):

$$\frac{\partial \psi}{\partial t} = i \nabla^2 \psi - i\kappa \psi |\psi|^2$$

Separating into real and imaginary parts with $\psi = u + iv$:

$$\frac{\partial u}{\partial t} = -D \nabla^2 v - \kappa v (u^2 + v^2)$$

$$\frac{\partial v}{\partial t} = D \nabla^2 u + \kappa u (u^2 + v^2)$$

Via cross-diffusion formulation:
- Diffusion coupling: $(u,v) \leftrightarrow$ off-diagonal terms with coefficient $D$
- Reaction: $(-\kappa v)(u^2+v^2)$ and $(\kappa u)(u^2+v^2)$

Where:
- $\psi = u + iv$ is the complex wave envelope
- $D = 1$ is the dispersion coefficient
- $\kappa$ controls the nonlinearity sign and strength
- $|\psi|^2 = u^2 + v^2$ is the intensity/probability density

**Bright soliton solution** (focusing, $\kappa > 0$):
$$\psi(x,t) = \eta \operatorname{sech}[\eta(x-ct)] e^{i(cx/2 + \phi(t))}$$

**Dark soliton solution** (defocusing, $\kappa < 0$):
$$\psi(x,t) = \psi_0 \tanh[\psi_0(x-ct)/\sqrt{2}] e^{i\omega t}$$

## Default Config

```yaml
solver: midpoint
dt: 0.00002
dx: 0.1
domain_size: 30
dimension: 1

boundary_x: periodic

parameters:
  kappa: 1   # [-5, 5] - nonlinearity parameter
  D: 1       # dispersion coefficient
  c: 10      # soliton velocity (sets initial condition)
```

## Parameter Variants

### NonlinearSchrodingerSoliton
Soliton propagation demonstration:
- `kappa = 1` (focusing - bright soliton)
- `D = 1`, `c = 10` (soliton velocity)
- Initial condition: $u = \cos(cx)/\cosh(x-L/3)$, $v = \sin(cx)/\cosh(x-L/3)$
- This creates a moving sech-shaped envelope with carrier wave
- Midpoint timestepping for better accuracy
- Plotted: $|\psi|^2 = u^2 + v^2$ (intensity)

### Effect of Nonlinearity Sign

| kappa | Type | Soliton Type | Behavior |
|-------|------|--------------|----------|
| kappa > 0 | Focusing | Bright soliton | Stable localized pulses |
| kappa = 0 | Linear | None | Dispersive spreading |
| kappa < 0 | Defocusing | Dark soliton | Dips on background, eventual instability |

### Numerical Notes

The simulation does **not** preserve the conserved quantities exactly (energy, momentum, etc.), so:
- Long-time dynamics may deviate from exact solutions
- Soliton amplitude may drift over very long simulations
- Defocusing case ($\kappa < 0$) eventually shows instabilities

### Conservation Laws

The NLS equation conserves:
1. **Mass/Power**: $N = \int |\psi|^2 dx$
2. **Momentum**: $P = \frac{i}{2} \int (\psi^* \psi_x - \psi \psi_x^*) dx$
3. **Hamiltonian**: $H = \int (|\psi_x|^2 - \frac{\kappa}{2}|\psi|^4) dx$

## References

- Zakharov, V.E. & Shabat, A.B. (1972). "Exact theory of two-dimensional self-focusing" - Sov. Phys. JETP 34:62
- Hasegawa, A. & Tappert, F. (1973). "Transmission of stationary nonlinear optical pulses in dispersive dielectric fibers" - Appl. Phys. Lett. 23:142
- Ablowitz, M.J. & Segur, H. (1981). "Solitons and the Inverse Scattering Transform" - SIAM
- Sulem, C. & Sulem, P.L. (1999). "The Nonlinear Schrodinger Equation" - Springer
