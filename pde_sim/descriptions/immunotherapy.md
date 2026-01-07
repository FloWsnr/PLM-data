# Immunotherapy Model

A three-species reaction-diffusion model of tumor-immune interactions demonstrating how spatial pattern formation affects cancer treatment efficacy.

## Description

This model describes the spatiotemporal dynamics of cancer immunotherapy, capturing the interplay between:
- **Effector cells (E/u)**: Immune cells (e.g., T-cells) that attack the tumor
- **Tumor cells (T/v)**: Cancer cells undergoing logistic growth
- **Cytokine (S/w)**: Signaling molecules (IL-2) that stimulate immune response

The model demonstrates a crucial insight: **spatial pattern formation (Turing instability) can confer treatment resistance**. When the system undergoes a Turing instability, the resulting heterogeneous patterns of tumor cells can be more resistant to immunotherapy than spatially uniform tumors.

Key phenomena:
- **Turing patterns in tumors**: Spots, stripes, and holes in tumor density
- **Spatial resilience**: Patterned tumors persist under treatment that would eliminate uniform tumors
- **Hysteresis**: Treatment can push the system to a new patterned equilibrium rather than elimination
- **Boundary-driven treatment**: Immunotherapy delivered through domain boundaries

The model incorporates:
1. Tumor stimulation of effector cell recruitment
2. Effector cell-mediated tumor killing
3. Cytokine production by effector-tumor interactions
4. Cytokine stimulation of effector cell proliferation
5. Time-dependent treatment protocols ($K_u t$, $K_w t$)

This work demonstrates that understanding spatial dynamics is essential for predicting treatment outcomes - homogeneous (well-mixed) models may significantly overestimate treatment efficacy.

## Equations

$$\frac{\partial u}{\partial t} = \delta_u \nabla^2 u + \alpha v - \mu_u u + \frac{\rho_u u w}{1 + w} + \sigma_u + K_u t$$

$$\frac{\partial v}{\partial t} = \delta_v \nabla^2 v + v(1-v) - \frac{u v}{\gamma_v + v}$$

$$\frac{\partial w}{\partial t} = \delta_w \nabla^2 w + \frac{\rho_w u v}{\gamma_w + v} - \mu_w w + \sigma_w + K_w t$$

Where:
- $u$ = effector cell density (E)
- $v$ = tumor cell density (T)
- $w$ = cytokine concentration (S/IL-2)
- $\alpha$ = tumor-induced effector recruitment
- $\mu_u, \mu_w$ = decay rates
- $\rho_u, \rho_w$ = proliferation/production rates
- $\gamma_v, \gamma_w$ = Michaelis-Menten constants
- $\delta_u, \delta_v, \delta_w$ = diffusion coefficients
- $\sigma_u, \sigma_w$ = constant treatment rates
- $K_u, K_w$ = time-ramped treatment rates

## Default Config

```yaml
solver: euler
dt: 0.0005
dx: 0.25
domain_size: 100

boundary_x: periodic
boundary_y: periodic

num_species: 3

parameters:
  # Kinetic parameters
  c: 0.3           # alpha (tumor-induced recruitment)
  mu: 0.167        # mu_u (effector decay)
  p_u: 0.69167     # rho_u (effector proliferation)
  g_u: 20          # gamma_u (effector saturation)
  p_v: 0.5555556   # killing rate coefficient
  g_v: 0.1         # gamma_v (tumor saturation)
  p_w: 27.778      # rho_w (cytokine production)
  g_w: 0.001       # gamma_w (cytokine saturation)
  nu: 55.55556     # mu_w (cytokine decay)
  s_w: 10          # sigma_w (constant cytokine source)
  s_u: 0.0         # sigma_u (constant effector source)

  # Diffusion coefficients (effector >> tumor, creates Turing instability)
  Du: 0.5          # effector diffusion (delta_u)
  Dv: 0.008        # tumor diffusion (delta_v, slow)
  Dw: 4            # cytokine diffusion (delta_w, fast)
```

## Parameter Variants

### ImmunotherapyModel (Base)
Default configuration with periodic treatment protocol.
- Neumann boundary conditions
- Effector diffusion 60x faster than tumor
- Cytokine diffuses fastest

### ImmunotherapyCircleNeumann
Circular domain with ramped effector treatment ($K_u > 0$).
- Demonstrates Turing pattern emergence under treatment
- Pattern formation leads to tumor persistence
- Setting $\eta = 0$ (no spatial noise) or $\delta_u = 1$ prevents patterns and allows tumor elimination

### ImmunotherapyCircleHysteresis
Time-varying treatment protocol with ramp-up and ramp-down.
- Cytokine source increases until $t = 200$, then decreases
- Demonstrates treatment hysteresis
- Final state is patterned (not homogeneous) despite parameters returning to initial values

### ImmunotherapySquareDirichlet
Boundary-driven treatment via Dirichlet conditions on cytokine.
- $w = B_w t$ on boundaries
- More realistic model of treatment delivery
- Boundary-induced patterns differ from internal Turing patterns

## Notes

- Setting $\delta_u = 1$ or $\delta_u = 10$ typically prevents Turing instability
- Tumor elimination is faster when spatial patterns do not form
- The model exhibits strong dependence on initial conditions (spatial noise)
- Pattern type transitions: holes -> stripes -> spots as treatment increases

## References

- Kuznetsov, V. A., et al. (1994). Nonlinear dynamics of immunogenic tumors. Bulletin of Mathematical Biology, 56(2), 295-321.
- Kirschner, D., & Panetta, J. C. (1998). Modeling immunotherapy of the tumor-immune interaction. Journal of Mathematical Biology, 37(3), 235-252.
- Related preprint: arXiv:2503.20909 (spatial pattern formation impact on tumor resistance)
