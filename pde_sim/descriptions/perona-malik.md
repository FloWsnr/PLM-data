# Perona-Malik Equation

A nonlinear anisotropic diffusion equation for image denoising that smooths homogeneous regions while preserving and enhancing edges.

## Description

The Perona-Malik equation, introduced by Pietro Perona and Jitendra Malik in 1990, revolutionized image processing by providing a principled approach to denoising that respects image structure. Unlike linear diffusion (which blurs everything, including edges), anisotropic diffusion adapts to image content: smoothing within uniform regions while preserving sharp boundaries.

The key insight is making the diffusion coefficient depend on the local image gradient:
- **Small gradients** (smooth regions): Strong diffusion, aggressive smoothing of noise
- **Large gradients** (edges): Weak or no diffusion, edges remain sharp

This creates a scale-space where small-scale noise is removed while large-scale structure (edges, contours) is preserved. The process can even **enhance** edges by sharpening gradients over time.

Applications include:
- **Image denoising**: Removing sensor noise while preserving detail
- **Medical imaging**: Enhancing CT/MRI scans
- **Computer vision preprocessing**: Improving edge detection
- **Segmentation**: Clarifying region boundaries

The equation has sparked extensive mathematical analysis regarding well-posedness (the original formulation is ill-posed) and regularizations that maintain the edge-preserving properties while ensuring mathematical stability.

## Equations

The Perona-Malik equation:

$$\frac{\partial u}{\partial t} = \nabla \cdot \left( g(|\nabla u|) \nabla u \right)$$

where $g(s)$ is an edge-stopping function that decreases with gradient magnitude.

The implemented version uses an exponential edge-stopping function:

$$\frac{\partial u}{\partial t} = \nabla \cdot \left( e^{-D |\nabla u|^2} \nabla u \right)$$

**Implementation** (full divergence form):

The divergence form expands as:
$$\nabla \cdot (g \nabla u) = g \Delta u + \nabla g \cdot \nabla u$$

With $g = e^{-D|\nabla u|^2}$, computing $\nabla g$ requires second derivatives:
$$\nabla g = -2Dg \cdot (u_x u_{xx} + u_y u_{xy}, u_x u_{xy} + u_y u_{yy})$$

The full equation becomes:
$$\frac{\partial u}{\partial t} = g \left[ \Delta u - 2D(u_x^2 u_{xx} + 2u_x u_y u_{xy} + u_y^2 u_{yy}) \right]$$

This includes the cross-term $\nabla g \cdot \nabla u$ that produces edge sharpening.

Where:
- $u(x,y,t)$ is the image intensity (grayscale value)
- $g(|\nabla u|) = e^{-D|\nabla u|^2}$ is the diffusivity function
- $D$ controls edge sensitivity (larger D = more edge enhancement)
- $|\nabla u|^2$ measures local gradient magnitude (edge strength)

**Common edge-stopping functions**:
1. Exponential: $g(s) = e^{-(s/K)^2}$
2. Tukey: $g(s) = 1/(1 + (s/K)^2)$
3. Huber: $g(s) = 1$ if $s < K$, else $K/s$

## Default Config

```yaml
solver: euler
dt: 0.0002
dx: 0.2  # resolution 500 on domain 100
domain_size: 100

boundary_x: no-flux (neumann)
boundary_y: no-flux (neumann)

parameters:
  D: 5.0     # edge sensitivity parameter
  sigma: 1.0 # [0, 2] - initial noise level
```

## Parameter Variants

### PeronaMalik (Standard)
Image denoising demonstration:
- `D = 5` (edge sensitivity)
- `sigma = 1` (noise level in initial condition)
- Initial condition: Text image with added noise
- Greyscale colormap
- Starts paused - press play to denoise

### Image Sources

Two built-in images:
- `I_T`: Text quote from Bernt Oksendal (default)
- `I_S`: Aperiodic tiling pattern
- Change initial condition to use $I_S$ instead of $I_T$ for alternative

### Edge Sensitivity Parameter D

| D Value | Behavior |
|---------|----------|
| D small (~1) | Mild smoothing, weak edge preservation |
| D medium (~5) | Good balance of denoising and edge preservation |
| D large (>10) | Strong edge enhancement, risk of instability |

### Workflow

1. **Initial state**: Noisy image (text or pattern with added random noise)
2. **Play simulation**: Diffusion smooths noise within uniform regions
3. **Observe**: Edges remain sharp, may even sharpen slightly
4. **Adjust D**: Higher values preserve more edge detail
5. **Stop**: When image is sufficiently denoised

### Adding More Noise

Use the brush tool with value `0.3*(RAND-0.5)` to add noise to any region, then continue diffusion to denoise.

### Mathematical Notes

The original Perona-Malik formulation is mathematically **ill-posed** - small perturbations can grow unboundedly. However:
- In practice, discretization provides regularization
- Finite difference schemes are stable for reasonable timesteps
- The visual results remain useful for image processing
- Regularized versions (e.g., adding higher-order terms) can ensure well-posedness

## References

- Perona, P. & Malik, J. (1990). "Scale-space and edge detection using anisotropic diffusion" - IEEE Trans. PAMI 12:629
- Weickert, J. (1998). "Anisotropic Diffusion in Image Processing" - Teubner
- Catte, F. et al. (1992). "Image selective smoothing and edge detection by nonlinear diffusion" - SIAM J. Num. Anal. 29:182
