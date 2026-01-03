# Schrödinger Equation

## Mathematical Formulation

The free-particle Schrödinger equation:

$$i\hbar \frac{\partial \psi}{\partial t} = -\frac{\hbar^2}{2m} \nabla^2 \psi$$

In dimensionless form (with $\hbar = m = 1$):

$$\frac{\partial \psi}{\partial t} = i D \nabla^2 \psi$$

where:
- $\psi$ is the complex wave function
- $D = \hbar/(2m)$ is the diffusion-like coefficient
- $|\psi|^2$ gives the probability density

## Physical Background

The Schrödinger equation is the fundamental equation of quantum mechanics, describing the time evolution of quantum states. Key features:

- **Complex-valued**: Phase carries physical information
- **Probability conservation**: $\int |\psi|^2 dx = 1$ is conserved
- **Dispersive**: Different wavelengths travel at different speeds
- **Linear superposition**: Quantum interference effects

## Parameters

| Parameter | Symbol | Description | Typical Range |
|-----------|--------|-------------|---------------|
| Diffusion coefficient | $D$ | Related to $\hbar/(2m)$ | 0.1 - 0.5 |

## Wave Packet Dynamics

A Gaussian wave packet evolves as:
- **Spreading**: Width increases as $\sigma(t) = \sigma_0\sqrt{1 + (Dt/\sigma_0^2)^2}$
- **Group velocity**: Center moves at $v_g = \hbar k/m$
- **Phase velocity**: Different from group velocity (dispersion)

## Quantum Phenomena

1. **Wave-particle duality**: Particles exhibit wave behavior
2. **Interference**: Double-slit patterns
3. **Tunneling**: Penetration through barriers (not shown in free particle)
4. **Uncertainty principle**: Position-momentum tradeoff

## Applications

1. **Atomic physics**: Electron orbitals
2. **Semiconductor physics**: Electron transport
3. **Quantum computing**: Qubit evolution
4. **Chemistry**: Molecular bonding
5. **Optics**: Paraxial wave equation (analogous form)

## Initial Conditions

**Wave packet**: Gaussian envelope with plane wave
$$\psi(x,y,0) = A \exp\left(-\frac{(x-x_0)^2 + (y-y_0)^2}{2\sigma^2}\right) \exp(i(k_x x + k_y y))$$

## Numerical Considerations

- Complex arithmetic required
- Probability conservation is a good numerical check
- Split-operator methods are efficient for this equation
- Absorbing boundary conditions often needed for open systems

## References

- Schrödinger, E. (1926). *Quantisierung als Eigenwertproblem*
- Griffiths, D.J. (2018). *Introduction to Quantum Mechanics*
