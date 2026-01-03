# Korteweg-de Vries (KdV) Equation

## Mathematical Formulation

The KdV equation for shallow water waves and solitons:

$$\frac{\partial u}{\partial t} + \alpha u\frac{\partial u}{\partial x} + \beta\frac{\partial^3 u}{\partial x^3} = 0$$

Standard form (with $\alpha = 6$, $\beta = 1$):
$$\frac{\partial u}{\partial t} + 6u\frac{\partial u}{\partial x} + \frac{\partial^3 u}{\partial x^3} = 0$$

where:
- $u$ is the wave amplitude (surface elevation)
- $\alpha$ is the nonlinear coefficient
- $\beta$ is the dispersion coefficient

## Physical Background

The KdV equation balances:
1. **Nonlinearity**: $u\partial_x u$ causes wave steepening
2. **Dispersion**: $\partial_{xxx} u$ causes wave spreading

When balanced, these effects create **solitons**: localized waves that propagate without changing shape.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Linear speed | $c$ | Background wave speed | -2 to 2 |
| Nonlinearity | $\alpha$ | Steepening strength | 0 - 10 |
| Dispersion | $\beta$ | Dispersive effects | 0.01 - 2 |

## Soliton Solutions

Single soliton:
$$u(x,t) = \frac{A}{2}\,\text{sech}^2\left(\sqrt{\frac{A}{12}}(x - \frac{A}{3}t - x_0)\right)$$

Properties:
- Amplitude $A$ determines speed: faster solitons are taller
- Width decreases with amplitude: $w \sim 1/\sqrt{A}$
- Taller solitons are narrower and faster

## Soliton Interactions

Remarkable property: solitons **pass through each other** unchanged!
- Phase shift after collision
- No radiation or energy loss
- Elastic "particle-like" behavior

## Inverse Scattering Transform

The KdV equation is **exactly integrable**:
- Infinite conservation laws
- Solved by inverse scattering method (1967)
- Related to linear Schrödinger equation
- Soliton number = bound state count

## Dispersion Relation

Linear waves: $u \sim e^{i(kx - \omega t)}$
$$\omega = ck - \beta k^3$$

Phase velocity: $c_p = \omega/k = c - \beta k^2$
Group velocity: $c_g = d\omega/dk = c - 3\beta k^2$

Longer waves travel faster (anomalous dispersion).

## Applications

1. **Shallow water waves**: Original derivation
2. **Internal waves**: Ocean stratification
3. **Plasma physics**: Ion-acoustic waves
4. **Optical fibers**: Nonlinear pulse propagation
5. **Lattice dynamics**: Anharmonic chains

## Historical Significance

- Korteweg & de Vries (1895): Derived equation
- Zabusky & Kruskal (1965): Discovered solitons numerically
- Gardner, Greene, Kruskal, Miura (1967): Inverse scattering solution

The numerical discovery of solitons (Fermi-Pasta-Ulam-Tsingou problem) was a landmark in nonlinear science.

## Numerical Considerations

- **3rd-order spatial**: Implicit methods helpful
- **Conservation**: Energy and momentum should be preserved
- **Soliton resolution**: Need sufficient resolution for narrow peaks
- **Periodic BC**: Natural for numerical study

## References

- Korteweg, D.J. & de Vries, G. (1895). *On the change of form of long waves*
- Zabusky, N.J. & Kruskal, M.D. (1965). *Interaction of "Solitons"*
- Ablowitz, M.J. & Segur, H. (1981). *Solitons and the Inverse Scattering Transform*
