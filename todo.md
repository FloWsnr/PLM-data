# TODO

## Vector Robin Boundary Conditions

Current status:
- Vector `neumann` BCs are supported in shared config/core infrastructure.
- Vector `robin` BCs are intentionally blocked in config validation.

Why this is blocked:
- The current Robin schema is scalar-shaped: it only provides a single `alpha`.
- For scalar PDEs, Robin is unambiguous: `alpha * u = g` on the boundary.
- For vector PDEs, "Robin" can mean several different things:
  - isotropic penalty: `alpha * u`
  - componentwise coefficients
  - matrix/tensor coupling: `K u`
  - normal-only or tangential-only terms
  - PDE-specific laws such as slip/friction or elastic foundation models
- If shared infrastructure accepted vector Robin today, it would silently choose one interpretation, most likely the isotropic `alpha * I` case. That is too implicit for a modular PDE library.

What would be needed to support it cleanly:
- Decide the shared meaning first.
- Smallest acceptable extension:
  - define an explicit "isotropic vector Robin" contract
  - keep `value` vector-valued
  - interpret `alpha` as a scalar multiplier of `inner(u, v)` on the boundary
- More general extension:
  - allow componentwise or tensor-valued Robin coefficients in the config schema
  - possibly distinguish normal/tangential Robin operators

Recommended next step:
- Only add shared vector Robin once there is a concrete PDE that needs it and the required semantics are clear.
- For now, keep the shared layer limited to vector Dirichlet and vector Neumann, which are much less ambiguous.
