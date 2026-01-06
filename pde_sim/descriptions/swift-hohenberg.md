# Swift-Hohenberg Equation

## Mathematical Formulation

The Swift-Hohenberg equation for pattern formation:

$$\frac{\partial u}{\partial t} = ru - (k_0^2 + \nabla^2)^2 u + g_2 u^2 + g_3 u^3 + g_5 u^5$$

Expanding the squared operator:
$$\frac{\partial u}{\partial t} = ru - k_0^4 u - 2k_0^2 \nabla^2 u - \nabla^4 u + g_2 u^2 + g_3 u^3 + g_5 u^5$$

where:
- $u$ is the pattern amplitude
- $r$ is the control parameter (bifurcation parameter)
- $k_0$ is the critical wavenumber
- $g_2, g_3, g_5$ are nonlinear coefficients

For stability, we need $g_5 < 0$ (or $g_3 < 0$ if $g_5 = 0$).

## Physical Background

The Swift-Hohenberg equation is an **amplitude equation** derived from Rayleigh-Bénard convection near onset. It captures the universal features of pattern-forming instabilities.

Key features:
- **Band of unstable modes**: Wavenumbers near $k_0$ grow
- **Wavenumber selection**: Preferred pattern wavelength $\lambda = 2\pi/k_0$
- **Nonlinear saturation**: Cubic term limits growth

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Control parameter | $r$ | Distance from onset | -1 to 1 |
| Critical wavenumber | $k_0$ | Selected wavelength | 0.1 - 5 |
| Quadratic nonlinearity | $g_2$ | Breaks up/down symmetry | -5 to 5 |
| Cubic nonlinearity | $g_3$ | Saturation (< 0 if $g_5=0$) | -5 to 5 |
| Quintic nonlinearity | $g_5$ | Higher-order saturation (< 0 for stability) | -5 to 0 |

## Linear Stability

The growth rate of mode $k$ is:
$$\sigma(k) = r - (k_0^2 - k^2)^2$$

- Maximum at $k = k_0$: $\sigma_{\max} = r$
- Unstable when $r > 0$: Modes near $k_0$ grow
- Stable when $r < 0$: All modes decay

## Pattern Selection

Depending on nonlinearities:

**$g_2 = 0$** (symmetric):
- Stripes are preferred
- Up-down symmetry preserved

**$g_2 \neq 0$** (asymmetric):
- Hexagons can form
- Up-hexagons vs down-hexagons

## Pattern Types

| Pattern | Condition |
|---------|-----------|
| Stripes | Generic near onset |
| Hexagons | With quadratic nonlinearity |
| Squares | Special parameter values |
| Quasipatterns | Near certain boundaries |
| Localised patterns | Subcritical regime ($r<0$, $g_2>0$, $g_3<0$) |

## Subcriticality and Localised Solutions

When $r < 0$, $g_2 > 0$, and $g_3 < 0$ (with $g_5 < 0$ for stability), the system can be in a subcritical regime that supports both stable patterned states and the stable homogeneous state $u = 0$. This enables:

- **Multistability**: Coexistence of patterned and uniform states
- **Localised solutions**: Patterns that decay to $u = 0$ in most of the domain
- **Snaking bifurcations**: Complex bifurcation structure

The quintic term $g_5 u^5$ is essential for stabilizing subcritical patterns when $g_3$ alone is insufficient.

## Applications

1. **Convection patterns**: Bénard cells
2. **Crystal surfaces**: Growth patterns
3. **Optical systems**: Laser beam patterns
4. **Chemical reactions**: Spatial patterns
5. **Biological systems**: Morphogenesis models

## Variational Structure

For $g_2 = 0$, the equation is a gradient flow:
$$\frac{\partial u}{\partial t} = -\frac{\delta \mathcal{F}}{\delta u}$$

with Lyapunov functional:
$$\mathcal{F}[u] = \int \left[-\frac{r}{2}u^2 + \frac{1}{2}[(k_0^2 + \nabla^2)u]^2 - \frac{g_3}{4}u^4 - \frac{g_5}{6}u^6\right] dx$$

## Numerical Considerations

- **4th-order spatial**: Requires implicit methods or very small $\Delta t$
- **Stiff equation**: Explicit stability $\Delta t \sim (\Delta x)^4$
- **Large domains**: Needed for pattern selection to operate

## References

- Swift, J. & Hohenberg, P.C. (1977). *Hydrodynamic fluctuations at the convective instability*
- Cross, M.C. & Hohenberg, P.C. (1993). *Pattern formation outside of equilibrium*
- Hoyle, R. (2006). *Pattern Formation*
