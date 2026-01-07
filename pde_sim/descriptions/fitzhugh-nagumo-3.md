# Three-Species FitzHugh-Nagumo

An extension of the classic FitzHugh-Nagumo model with three species, demonstrating competition between local oscillations and spatial pattern formation.

## Description

This model adds a third species to the classic two-species FitzHugh-Nagumo equations. The third species (w) introduces an additional slow timescale that can either stabilize or destabilize spatial patterns depending on parameters.

The key parameter is $a_v$, which controls the self-inhibition of the first recovery variable. This parameter determines whether oscillations or patterns dominate:

- **Low $a_v$ (< 0.3)**: Initially formed patterns are eventually destroyed by oscillations
- **High $a_v$ (>= 0.3)**: Patterns stabilize and overtake oscillations

This creates rich dynamics where patterns and oscillations compete for dominance.

### Applications

- **Neural tissue**: Complex oscillation-pattern dynamics in neural networks
- **Cardiac dynamics**: Competing rhythms and patterns in heart tissue
- **Pattern selection**: How systems choose between different dynamical modes

### Key Physical Behaviors

**Pattern-oscillation competition**: The system can exhibit both local oscillations and spatial patterns. The relative strength of these behaviors depends on parameters.

**Transient patterns**: At low $a_v$, patterns form initially but are eventually destroyed as oscillations take over.

**Pattern stabilization**: At high $a_v$, patterns become stable and spread throughout the domain.

## Equations

$$\frac{\partial u}{\partial t} = \nabla^2 u + u - u^3 - v$$

$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + \varepsilon_v (u - a_v v - a_w w - a_z)$$

$$\frac{\partial w}{\partial t} = D_w \nabla^2 w + \varepsilon_w (u - w)$$

where:
- $u$ is the fast activator (voltage-like)
- $v$ is the first recovery variable (medium timescale)
- $w$ is the second recovery variable (slow, pattern-forming)
- $D_v$, $D_w$ are diffusion coefficients ($D_w > D_v$ for pattern formation)
- $\varepsilon_v$, $\varepsilon_w$ are timescale parameters
- $a_v$ controls pattern vs oscillation dominance
- $a_w$ couples v to w
- $a_z$ is an offset parameter

## Default Config

```yaml
preset: fitzhugh-nagumo-3
solver: euler
dt: 0.001
domain_size: 280

bc:
  x: periodic
  y: periodic

parameters:
  Dv: 40.0       # v diffusion
  Dw: 200.0      # w diffusion (larger for patterns)
  a_v: 0.2       # pattern vs oscillation control
  e_v: 0.2       # v timescale
  e_w: 1.0       # w timescale
  a_w: 0.5       # v-w coupling
  a_z: -0.1      # offset
```

## Parameter Variants

### Oscillation-Dominated (a_v = 0.1)
- Patterns form initially but decay
- System settles into uniform oscillations

### Pattern-Oscillation Competition (a_v = 0.2)
- Default behavior
- Transient patterns with eventual oscillation takeover

### Pattern-Dominated (a_v = 0.4)
- Patterns stabilize and spread
- Oscillations suppressed

## Initial Conditions

The default initial condition creates:
- **u**: Gaussian blob at domain center (amplitude 5.0)
- **v**: Cosine pattern across domain
- **w**: Zero everywhere

This seeds both pattern formation (from v) and localized excitation (from u).

## Comparison with Standard FitzHugh-Nagumo

| Feature | fitzhugh-nagumo | fitzhugh-nagumo-3 |
|---------|-----------------|-------------------|
| Species | 2 (u, v) | 3 (u, v, w) |
| Timescales | 2 | 3 |
| Pattern control | Via Dv | Via a_v |
| Dynamics | Spirals or patterns | Competition between patterns and oscillations |

## References

- FitzHugh, R. (1961). Impulses and physiological states in theoretical models of nerve membrane. Biophysical Journal, 1(6), 445-466.
- Nagumo, J., Arimoto, S., & Yoshizawa, S. (1962). An active pulse transmission line simulating nerve axon. Proceedings of the IRE, 50(10), 2061-2070.
- VisualPDE FitzHugh-Nagumo-3 preset
