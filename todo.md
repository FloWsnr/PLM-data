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

## Periodic Maxwell Boundary Conditions (resolved)

Decision:
- Keep periodic domains unsupported for `maxwell` and `maxwell_pulse` for now.
- Treat this as an upstream/library limitation in the current `dolfinx_mpc` stack, not as a small missing feature in the preset code.
- Revisit only if a newer `dolfinx_mpc` release adds periodic support for `N1curl` / H(curl) spaces, or if we decide to implement custom constraint machinery for Maxwell.

Why:
- Both Maxwell presets already reject periodic domains explicitly.
- A minimal local smoke test using `dolfinx_mpc 0.10.3` with an `("N1curl", 1)` function space and a periodic rectangle failed before the solve stage with:
  `RuntimeError: Periodic conditions for vector valued spaces are not implemented`
- Our current periodic implementation works for scalar Lagrange spaces, mixed scalar spaces, and blocked vector Lagrange spaces, but Maxwell is built on `N1curl` / H(curl) elements.
- Periodic identification in H(curl) spaces is more involved because the DOFs are tied to oriented edges/faces and require correct tangential/orientation handling.
- The local DOLFINx reference demos include Maxwell/Nedelec examples with PEC and absorbing boundaries, but do not provide a periodic Maxwell pattern to follow.
- This means periodic Maxwell is feasible only as a dedicated spike involving upstream verification or lower-level MPC work, not as a normal preset extension.
