# Van der Pol Oscillator (Diffusively Coupled)

A spatially extended version of the classic nonlinear oscillator, exhibiting self-sustained limit cycle oscillations coupled through diffusion to produce complex spatiotemporal patterns.

## Description

The Van der Pol oscillator, developed by Dutch physicist Balthasar van der Pol in the 1920s while studying vacuum tube circuits at Philips, is one of the most important models in nonlinear dynamics. It was the first system where **limit cycles** - isolated periodic orbits - were systematically studied.

The original Van der Pol oscillator exhibits:
- **Self-sustained oscillations**: Energy is pumped in at small amplitudes, dissipated at large amplitudes
- **Nonlinear damping**: The damping coefficient $(1-X^2)$ changes sign with amplitude
- **Limit cycle**: All trajectories converge to a single periodic orbit regardless of initial conditions
- **Relaxation oscillations**: For large $\mu$, oscillations consist of slow buildup and fast jumps

The spatially extended version couples oscillators at each point through diffusion, creating a reaction-diffusion system where:
- Each spatial point undergoes Van der Pol dynamics
- Neighboring points influence each other through diffusive coupling
- The competition between local oscillation and spatial coupling creates waves and patterns

Historical note: Van der Pol and van der Mark (1927) reported "irregular noise" at certain drive frequencies - later recognized as one of the first observations of deterministic chaos. The Van der Pol oscillator also served as an early model for the heartbeat.

## Equations

The diffusively coupled Van der Pol oscillator:

$$\frac{\partial X}{\partial t} = Y$$

$$\frac{\partial Y}{\partial t} = D(\nabla^2 X + \varepsilon \nabla^2 Y) + \mu(1-X^2)Y - X$$

Written in first-order system form:
- $X$ is the oscillator displacement
- $Y = \dot{X}$ is the velocity

Where:
- $\mu$ is the nonlinear damping parameter (controls oscillation character)
- $D$ is the diffusion coupling strength
- $\varepsilon$ is artificial diffusion for numerical stability
- The term $\mu(1-X^2)Y$:
  - When $|X| < 1$: Negative damping (energy injection, $\mu > 0$)
  - When $|X| > 1$: Positive damping (energy dissipation)

**Uncoupled limit cycle period**:
- For small $\mu$: $T \approx 2\pi$ (near-sinusoidal)
- For large $\mu$: $T \approx (3 - 2\ln 2)\mu$ (relaxation oscillations)

## Default Config

```yaml
solver: euler
dt: 0.0015
dx: 0.1
domain_size: 50

boundary_x: periodic
boundary_y: periodic

parameters:
  D: 1           # diffusion coupling strength
  mu: 8.53       # [1, 30] - nonlinear damping parameter
  epsilon: 0.005 # artificial diffusion (numerical stability)
```

## Parameter Variants

### VanDerPol (Standard)
Spatially coupled Van der Pol oscillators:
- `mu = 8.53` (intermediate relaxation regime)
- `D = 1` (moderate coupling)
- `epsilon = 0.005` (small numerical diffusion)
- Initial condition: Random $X = 0.05 \cdot \text{RANDN}$
- Visualization: $X$ field with 2D plot

### Effect of Parameters

| Parameter | Effect |
|-----------|--------|
| mu small (1-3) | Near-sinusoidal oscillations |
| mu large (>10) | Strong relaxation oscillations |
| D small | Localized, weakly coupled dynamics |
| D large | Synchronized large-scale oscillations |

### Behaviors to Observe

- **Random initial conditions**: System self-organizes into oscillating structures
- **Uniform initial condition**: Perturbations initiate spreading waves
- **Varying mu**: Different timescales and pattern wavelengths
- **Click interactions**: Perturbed regions create propagating wave fronts

### Relaxation Oscillation Character

For large $\mu$, the oscillation has a characteristic "slow-fast" structure:
1. **Slow phase**: $X$ gradually increases (buildup)
2. **Fast jump**: Rapid transition when $|X| \approx 1$ crossed
3. **Slow phase**: $X$ gradually decreases
4. **Fast jump**: Return transition

This gives the sawtooth-like waveform typical of relaxation oscillators.

### Warning
Rapid oscillations can produce rapid visual flashing. Be mindful when exploring large $\mu$ values.

## References

- Van der Pol, B. (1926). "On relaxation oscillations" - Phil. Mag. 2:978
- Van der Pol, B. & Van der Mark, J. (1927). "Frequency demultiplication" - Nature 120:363
- FitzHugh, R. (1961). "Impulses and physiological states in theoretical models of nerve membrane" - Biophys. J. 1:445
- Strogatz, S.H. (2015). "Nonlinear Dynamics and Chaos" - Westview Press
