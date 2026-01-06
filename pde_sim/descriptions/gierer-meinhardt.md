# Gierer-Meinhardt Model

A classical activator-inhibitor reaction-diffusion system for morphogenesis and biological pattern formation, producing spots and stripes.

## Description

The Gierer-Meinhardt model was formulated by Alfred Gierer and Hans Meinhardt in 1972 as a molecularly plausible model for pattern formation in biological development. Based on Turing's 1952 theoretical framework, it provides a concrete mechanism for how patterns can spontaneously emerge from initially homogeneous tissue.

The model is built on two fundamental principles:
- **Short-range activation**: The activator (u) stimulates its own production through autocatalysis
- **Long-range inhibition**: The inhibitor (v) is produced by the activator and diffuses rapidly to suppress activation at a distance

This local self-enhancement combined with lateral inhibition creates the conditions for diffusion-driven instability. The inhibitor must diffuse faster than the activator (D > 1) for patterns to form.

Key phenomena include:
- **Spot patterns**: The default behavior, where isolated peaks of activator form
- **Stripe instability**: Stripes tend to break up into spots due to transverse instabilities
- **Labyrinthine patterns**: With saturation terms, stable stripe-like patterns can form
- **Spike dynamics**: In 1D, spikes can move, oscillate, split, or annihilate

The model has been successfully applied to:
- Hydra head regeneration and body patterning
- Drosophila segment polarity
- Leaf venation patterns
- Pigmentation patterns in animals
- Anthocyanin patterns in flowers (Mimulus/monkeyflowers)

## Equations

### Standard Form
$$\frac{\partial u}{\partial t} = \nabla^2 u + a + \frac{u^2}{v} - bu$$

$$\frac{\partial v}{\partial t} = D \nabla^2 v + u^2 - cv$$

### With Saturation (for stripes)
$$\frac{\partial u}{\partial t} = \nabla^2 u + a + \frac{u^2}{v(1+Ku^2)} - bu$$

$$\frac{\partial v}{\partial t} = D \nabla^2 v + u^2 - cv$$

Where:
- $u$ is the activator concentration
- $v$ is the inhibitor concentration
- $a$ is the base activator production rate
- $b$ is the activator decay rate
- $c$ is the inhibitor decay rate
- $D > 1$ is required for pattern formation
- $K$ is the saturation constant (optional)

## Default Config

```yaml
solver: euler
dt: 0.0003
dx: 0.5
domain_size: 100

boundary_x: periodic
boundary_y: periodic

parameters:
  a: 0.5
  b: 1
  c: 6.1
  D: 100
```

## Parameter Variants

### GiererMeinhardt (Standard)
Base configuration producing spot-like patterns.
- `D = 100`: High diffusion ratio
- `a = 0.5`, `b = 1`, `c = 6.1`
- Generically favors spot patterns

### GiererMeinhardtStripeICs
Configuration with stripe-promoting initial conditions.
- Same parameters as standard
- Initial condition: `u(0) = 3*(1 + cos(n*pi*x/L))`, `v(0) = 1`
- `n` (range: [1, 20]): Number of initial stripes
- Demonstrates stripe-to-spot instability when perturbed

### GiererMeinhardtStripes
Configuration with saturation for stable labyrinthine patterns.
- `K = 0.003` (range: [0, 0.003]): Saturation constant
- `domain_size = 200`: Larger domain
- Very small K: spot-forming behavior
- Intermediate K: labyrinthine/stripe patterns
- Very large K: no Turing patterns

### GMHeterogeneous2D
Spatially heterogeneous version with position-dependent parameters.
- `G(x) = Ax/L`, `H(x) = Bx/L`: Heterogeneous forcing
- `A` (range: [-1, 1]): Production gradient
- `B` (range: [0, 5]): Decay gradient
- Dirichlet BC for u, Neumann BC for v
- Demonstrates effects of spatial heterogeneity on pattern selection

## References

- Gierer, A., & Meinhardt, H. (1972). A theory of biological pattern formation. Kybernetik, 12(1), 30-39.
- Meinhardt, H., & Gierer, A. (2000). Pattern formation by local self-activation and lateral inhibition. BioEssays, 22(8), 753-760.
- Turing, A. M. (1952). The chemical basis of morphogenesis. Philosophical Transactions of the Royal Society B, 237(641), 37-72.
