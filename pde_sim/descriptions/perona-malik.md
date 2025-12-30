# Perona-Malik Anisotropic Diffusion

## Mathematical Formulation

The Perona-Malik equation for edge-preserving diffusion:

$$\frac{\partial u}{\partial t} = \nabla \cdot \left(g(|\nabla u|) \nabla u\right)$$

With specific diffusivity function:
$$g(s) = \frac{1}{1 + (s/K)^2}$$

Simplified form used in implementation:
$$\frac{\partial u}{\partial t} = \frac{D \nabla^2 u}{1 + |\nabla u|^2/K^2}$$

where:
- $u$ is the image intensity
- $D$ is the diffusion coefficient
- $K$ is the edge threshold parameter
- $|\nabla u|$ is the gradient magnitude

## Physical Background

Standard diffusion (heat equation) smooths all features uniformly. Perona-Malik diffusion is **adaptive**:

- **Smooth regions** ($|\nabla u| \ll K$): Strong diffusion (smoothing)
- **Edges** ($|\nabla u| \gg K$): Weak diffusion (preserving)

This enables noise reduction while maintaining edges.

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Diffusion coefficient | $D$ | Base diffusion rate | 0.01 - 10 |
| Edge threshold | $K$ | Gradient threshold | 0.1 - 10 |

## Diffusivity Functions

Several choices for $g(s)$:

| Function | Formula | Properties |
|----------|---------|------------|
| Perona-Malik 1 | $e^{-(s/K)^2}$ | Gaussian decay |
| Perona-Malik 2 | $1/(1+(s/K)^2)$ | Lorentzian decay |
| Total Variation | $1/s$ | Scale invariant |
| Tukey | Bounded | Robust statistics |

## Forward vs Backward Diffusion

For Perona-Malik 2:
- $|\nabla u| < K$: Forward diffusion (smoothing)
- $|\nabla u| > K$: **Backward diffusion** (sharpening)

Backward diffusion can cause:
- Edge enhancement
- Staircasing artifacts
- Mathematical ill-posedness

## Well-Posedness Issues

The Perona-Malik equation is **ill-posed** in the classical sense:
- Solutions may not depend continuously on data
- Can develop singularities
- Numerical discretization acts as regularization

## Image Processing Pipeline

Typical workflow:
1. Initialize with noisy image
2. Evolve Perona-Malik equation
3. Edges preserved, noise reduced
4. Stop before over-smoothing

## Applications

1. **Medical imaging**: CT, MRI denoising
2. **Computer vision**: Edge detection preprocessing
3. **Photography**: Noise reduction
4. **Microscopy**: Image enhancement
5. **Remote sensing**: Satellite image processing

## Comparison with Linear Diffusion

| Property | Linear (Heat) | Perona-Malik |
|----------|---------------|--------------|
| Edges | Blurred | Preserved |
| Noise | Reduced | Reduced |
| Selectivity | None | Gradient-based |
| Well-posed | Yes | No |

## Regularized Versions

To address ill-posedness:
- **Catté et al.**: Regularize gradient $|\nabla G_\sigma * u|$
- **Total Variation**: Different functional
- **Bilateral filtering**: Discrete approach

## References

- Perona, P. & Malik, J. (1990). *Scale-space and edge detection using anisotropic diffusion*
- Catté, F. et al. (1992). *Image selective smoothing and edge detection*
- Weickert, J. (1998). *Anisotropic Diffusion in Image Processing*
