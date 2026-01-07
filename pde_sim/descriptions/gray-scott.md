# Gray-Scott Model

A reaction-diffusion system exhibiting complex autocatalytic dynamics with an extraordinarily rich variety of pattern-forming behaviors, from spots and stripes to spatiotemporal chaos.

## Description

The Gray-Scott model, introduced by Peter Gray and Stephen Scott in 1984, describes two coupled chemical species undergoing autocatalytic reactions with diffusion. Originally derived to model cubic autocatalytic biochemical reactions, it has become one of the most studied systems in pattern formation due to its remarkably rich dynamical behavior.

The system exhibits an extraordinary range of patterns depending on the feed rate (a) and removal rate (b) parameters, including:
- Labyrinthine/stripe patterns
- Stationary and pulsating spots
- Self-replicating spots (mitosis-like behavior)
- Worm-like structures
- Holes in otherwise uniform states
- Spatiotemporal chaos
- Glider-like moving spots (similar to cellular automata)

The famous 1993 Pearson classification mapped out these behavioral regimes across the (a,b) parameter space, revealing the stunning complexity hidden in this deceptively simple system. Pattern selection depends critically on the diffusion ratio D between the two species - with D=2 providing the richest parameter space.

The model has connections to real chemical systems like the ferrocyanide-iodate-sulphite reaction and serves as a paradigm for understanding Turing-type pattern formation in biology, from animal coat markings to cellular organization.

## Equations

The Gray-Scott model consists of two coupled reaction-diffusion equations:

$$\frac{\partial u}{\partial t} = \nabla^2 u + u^2 v - (a + b) u$$

$$\frac{\partial v}{\partial t} = D \nabla^2 v - u^2 v + a(1 - v)$$

Where:
- $u$ is the activator concentration (autocatalytic species)
- $v$ is the substrate concentration (fuel species)
- The reaction $u^2 v$ represents the autocatalytic production of $u$ from $v$
- The term $-a u$ represents decay of $u$
- The term $-b u$ represents conversion/removal of $u$
- The term $a(1-v)$ represents feeding of substrate $v$ toward concentration 1

## Default Config

```yaml
solver: euler
dt: 0.2
domain_size: 1000
resolution: 333  # gives dx ≈ 3

boundary_x: periodic
boundary_y: periodic

parameters:
  a: 0.037  # [0, 0.1] - feed rate
  b: 0.06   # [0.04, 0.1] - kill/removal rate
  D: 2      # diffusion ratio (v diffuses twice as fast as u)
```

## Parameter Variants

### GrayScott (Standard)
The default configuration producing labyrinthine/stripe patterns:
- `a = 0.037`
- `b = 0.06`
- `D = 2` (diffusion ratio)
- Initial condition: u=0, v=1 (uniform substrate)

### GrayScottGliders
Parameters tuned for moving spot ("glider") dynamics:
- `a = 0.014`
- `b = 0.054`
- Exhibits spots that bob around and interact

### Notable Parameter Combinations
The system exhibits distinct behaviors for different (a,b) values:

| Pattern Type | a | b |
|-------------|---|---|
| Labyrinthine | 0.037 | 0.06 |
| Spots | 0.03 | 0.062 |
| Pulsating spots | 0.025 | 0.06 |
| Worms | 0.078 | 0.061 |
| Holes | 0.039 | 0.058 |
| Spatiotemporal chaos | 0.026 | 0.051 |
| Intermittent chaos/holes | 0.034 | 0.056 |
| Moving spots (gliders) | 0.014 | 0.054 |
| Small waves | 0.018 | 0.051 |
| Big waves | 0.014 | 0.045 |
| U-skate world | 0.062 | 0.061 |

## References

- Gray, P. & Scott, S.K. (1984). "Autocatalytic reactions in the isothermal, continuous stirred tank reactor"
- Pearson, J.E. (1993). "Complex Patterns in a Simple System" - Science 261(5118):189-192
- Munafo, R. "Xmorphia Gallery" - http://www.mrob.com/pub/comp/xmorphia/
