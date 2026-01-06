# Complex Ginzburg-Landau Equation

One of the most universal equations in nonlinear physics, describing amplitude dynamics near oscillatory instabilities with applications from superconductivity to turbulence.

## Description

The Complex Ginzburg-Landau (CGL) equation is arguably the most-studied nonlinear model in physics. It describes the slow modulation of wave amplitudes in systems undergoing a Hopf bifurcation - the onset of oscillations from a steady state. First derived in the context of superconductivity by Ginzburg and Landau (1950), its complex version emerged from studies of pattern formation and has become the canonical model for dissipative wave systems.

The CGL equation arises as the generic amplitude equation for:
- **Superconductivity and superfluidity**: Order parameter dynamics
- **Chemical oscillations**: Belousov-Zhabotinsky reaction waves
- **Fluid instabilities**: Taylor-Couette flow, Rayleigh-Benard convection
- **Nonlinear optics**: Laser dynamics, mode-locked lasers
- **Biological oscillations**: Cardiac tissue, neural waves

Key phenomena exhibited by the CGL equation:
- **Spiral waves**: Self-organized rotating wave patterns
- **Defect turbulence**: Chaotic dynamics mediated by topological defects (spiral cores)
- **Plane waves and traveling waves**: Coherent propagating structures
- **Phase turbulence**: Disorder in the phase while amplitude remains nearly constant
- **Amplitude death**: Coupled oscillators can drive each other to extinction

The rich phenomenology depends critically on the "Benjamin-Feir" stability criterion for plane waves, which determines whether the system exhibits regular or turbulent behavior.

## Equations

The complex Ginzburg-Landau equation:

$$\frac{\partial \psi}{\partial t} = (D_r + i D_i) \nabla^2 \psi + (a_r + i a_i) \psi + (b_r + i b_i) \psi |\psi|^2$$

Separating into real and imaginary parts with $\psi = u + iv$:

$$\frac{\partial u}{\partial t} = D_r \nabla^2 u - D_i \nabla^2 v + a_r u - a_i v + (b_r u - b_i v)(u^2 + v^2)$$

$$\frac{\partial v}{\partial t} = D_i \nabla^2 u + D_r \nabla^2 v + a_r v + a_i u + (b_r v + b_i u)(u^2 + v^2)$$

Where:
- $\psi = u + iv$ is the complex amplitude (order parameter)
- $D_r$ is the real diffusion coefficient (must be $\geq 0$ for stability)
- $D_i$ is the imaginary diffusion (dispersion)
- $a_r$ controls linear growth/decay (typically $a_r > 0$ for instability)
- $a_i$ is the linear frequency
- $b_r$ controls nonlinear saturation (typically $b_r < 0$ for bounded solutions)
- $b_i$ is the nonlinear frequency shift

**Stability parameters**: The dynamics depend on $c_1 = D_i/D_r$ and $c_3 = b_i/|b_r|$ through the Benjamin-Feir criterion: $1 + c_1 c_3 > 0$ for stable plane waves.

## Default Config

```yaml
solver: euler
dt: 0.001
dx: 1 (inferred)
domain_size: 100 (inferred)

boundary_x: periodic
boundary_y: periodic

parameters:
  D_r: 0.1    # real diffusion
  D_i: 1      # imaginary diffusion (dispersion)
  a_r: 1      # linear growth rate
  a_i: 1      # linear frequency
  b_r: -1     # nonlinear saturation (must be negative)
  b_i: 0      # [-5, 5] - nonlinear frequency shift (main control parameter)
  n: 10       # initial condition mode number (x)
  m: 10       # initial condition mode number (y)
```

## Parameter Variants

### complexGinzburgLandau (Standard)
Rich dynamical exploration:
- `D_r = 0.1`, `D_i = 1`
- `a_r = 1`, `a_i = 1`, `b_r = -1`
- `b_i = 0` adjustable in [-5, 5]
- Initial condition: $\sin(n\pi x/100) \sin(m\pi y/100)$ for both $u$ and $v$
- Plotted variable: $|\psi|^2 = u^2 + v^2$ (amplitude squared)

### Dynamical Regimes (varying b_i)

| b_i Value | Typical Behavior |
|-----------|------------------|
| b_i = -5 | Strong phase turbulence, defect chaos |
| b_i = -1 | Moderate turbulence, spiral waves |
| b_i = 0 | Near-threshold, ordered spirals |
| b_i = 1 | Stable plane waves possible |
| b_i = 2 | Regular patterns, low turbulence |

### Related Systems

**Coupled CGL (Cross-Phase Modulation)**:
Two coupled complex fields $\psi_1$ and $\psi_2$ with cross-coupling:
$$\frac{\partial \psi_j}{\partial t} = (D_{jr} + i D_{ji})\nabla^2 \psi_j + (a_{jr} + i a_{ji})\psi_j + (b_{jr} + i b_{ji})\psi_j(|\psi_j|^2 + \alpha_j|\psi_k|^2)$$

Models optical interactions, amplitude death phenomena, and competition between wave patterns.

## References

- Ginzburg, V.L. & Landau, L.D. (1950). "On the theory of superconductivity" - Zh. Eksp. Teor. Fiz. 20:1064
- Aranson, I.S. & Kramer, L. (2002). "The world of the complex Ginzburg-Landau equation" - Rev. Mod. Phys. 74:99
- Cross, M.C. & Hohenberg, P.C. (1993). "Pattern formation outside of equilibrium" - Rev. Mod. Phys. 65:851
