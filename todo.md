# TODO

## Vector Robin Boundary Conditions (resolved)

Decision:
- Keep generic vector `robin` unsupported in the shared config/core layer.
- Continue using shared vector `dirichlet` and `neumann`, plus preset-specific vector boundary types where the operator is structurally different, such as Maxwell `absorbing`.
- If a future preset needs the isotropic form `alpha * inner(u, v) ds`, add it as an explicit BC type such as `isotropic_robin` rather than overloading generic `robin`.

Why:
- The current Robin schema is scalar-shaped: it only provides a single `alpha`.
- For scalar PDEs, Robin is unambiguous: `alpha * u = g` on the boundary.
- For vector PDEs, "Robin" can mean several different things:
  - isotropic penalty: `alpha * u`
  - componentwise coefficients
  - matrix/tensor coupling: `K u`
  - normal-only or tangential-only terms
  - PDE-specific laws such as slip/friction or elastic foundation models
- If shared infrastructure accepted vector Robin today, it would silently choose one interpretation, most likely the isotropic `alpha * I` case. That is too implicit for a modular PDE library.
