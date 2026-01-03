# Schnakenberg Model (Turing Pattern System)

## Mathematical Formulation

The Schnakenberg model is a two-component reaction-diffusion system:

$$\frac{\partial u}{\partial t} = D_u \nabla^2 u + a - u + u^2 v$$
$$\frac{\partial v}{\partial t} = D_v \nabla^2 v + b - u^2 v$$

where:
- $u$ is the activator concentration
- $v$ is the inhibitor concentration
- $D_u, D_v$ are diffusion coefficients
- $a, b$ are feed rates

## Physical Background

The Schnakenberg model is a prototypical **activator-inhibitor system** exhibiting Turing patterns. The mechanism:

1. **Activator (u)**: Autocatalytic species that promotes its own production
2. **Inhibitor (v)**: Species consumed by the activator reaction
3. **Differential diffusion**: $D_v \gg D_u$ enables pattern formation

The homogeneous steady state $(u^*, v^*) = (a+b, b/(a+b)^2)$ becomes unstable to spatial perturbations when Turing conditions are satisfied.

## Turing Instability Conditions

For patterns to emerge:
1. The homogeneous state must be stable: $\text{tr}(J) < 0$ and $\det(J) > 0$
2. Diffusion must destabilize: $D_v/D_u > 1$ (typically 10-100)
3. Specific wavenumbers become unstable

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Feed rate u | $a$ | Activator source | 0 - 1 |
| Feed rate v | $b$ | Inhibitor source | 0 - 2 |
| Diffusion u | $D_u$ | Activator diffusion (slow) | 0.001 - 0.1 |
| Diffusion v | $D_v$ | Inhibitor diffusion (fast) | 0.1 - 10 |

## Pattern Types

Depending on parameters, the system produces:
- **Spots**: Localized activator peaks
- **Stripes**: Labyrinthine patterns
- **Inverse spots**: Activator holes in inhibitor background
- **Mixed patterns**: Coexistence of spots and stripes

## Applications

1. **Animal coat patterns**: Zebra stripes, leopard spots
2. **Embryonic development**: Digit formation, somite patterning
3. **Chemical systems**: Chlorite-iodide-malonic acid (CIMA) reaction
4. **Vegetation patterns**: Banded patterns in semi-arid regions

## References

- Schnakenberg, J. (1979). *Simple chemical reaction systems with limit cycle behaviour*
- Murray, J.D. (2003). *Mathematical Biology II: Spatial Models*
- Turing, A.M. (1952). *The Chemical Basis of Morphogenesis*
