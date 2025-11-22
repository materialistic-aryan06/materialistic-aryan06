# Finite Difference Method for Diffusion Problem

## Overview

This project implements a numerical solution to the concentration evolution of Carbon (C) diffusing in a Steel-Fe system using the **Finite Difference Method with Implicit (backward Euler) scheme**.

## Problem Description

A block of Steel is sandwiched between two blocks of Fe, with Carbon as the only diffusing species:

- **System size**: L = 1 m
- **Initial concentration**: 
  - C = 1 in steel (0.25 < x < 0.75)
  - C = 0 in Fe (0 < x < 0.25 and 0.75 < x < 1)
- **Boundary conditions**: Neumann BC (zero flux) at x = 0 and x = 1: ∂C/∂x|_{x=0,1} = 0

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy scipy matplotlib
```

## Usage

Run the main solver script:

```bash
python diffusion_solver.py
```

This will:
1. Solve the diffusion problem for constant diffusivity (D = 1 m²/s)
2. Solve the diffusion problem for variable diffusivity (D = 1.1 - x² m²/s)
3. Generate plots showing concentration evolution over time
4. Generate plots of boundary concentrations vs time
5. Create observations about the effects of diffusivity

## Output Files

The script generates the following files:

1. **concentration_evolution_constant_D.png** - Concentration profiles over time for constant diffusivity
2. **concentration_evolution_variable_D.png** - Concentration profiles over time for variable diffusivity
3. **boundary_concentration_x025.png** - Concentration at x = 0.25 vs time (both cases)
4. **boundary_concentration_x075.png** - Concentration at x = 0.75 vs time (both cases)
5. **diffusion_observations.txt** - Detailed observations about diffusivity effects

## Implementation Details

### Numerical Method

The solver uses the **Implicit (backward Euler) scheme** for time discretization:

For **constant diffusivity**:
```
C^(n+1) - C^n = (D*dt/dx²) * (C^(n+1)_{i-1} - 2*C^(n+1)_i + C^(n+1)_{i+1})
```

For **variable diffusivity** D(x):
```
∂C/∂t = ∂/∂x(D(x) ∂C/∂x)
```

The implicit scheme requires solving a system of linear equations at each time step using sparse matrix solvers from scipy.

### Boundary Conditions

Neumann (zero flux) boundary conditions are implemented at both ends:
- At x = 0: ∂C/∂x = 0
- At x = 1: ∂C/∂x = 0

These are enforced using ghost points in the discretization.

### Convergence

The simulation continues until:
- The maximum change in concentration between time steps is less than 10⁻⁶, or
- The maximum number of time steps (2000) is reached

## Key Results

### Part (a): Constant Diffusivity
- Diffusivity: D = 1 m²/s
- Reaches steady state at approximately t ≈ 0.27 s
- Steady state: uniform concentration C = 0.5 throughout domain

### Part (b): Variable Diffusivity
- Diffusivity: D(x) = 1.1 - x² m²/s
- Reaches steady state at approximately t ≈ 1.0 s (slower due to lower D in center)
- Steady state: same uniform concentration C = 0.5 throughout domain

### Part (c): Boundary Concentrations
The plots show how concentration at the steel boundaries (x = 0.25 and x = 0.75) evolves over time, comparing both diffusivity cases.

### Part (d): Observations

**What Changed:**
- Time to reach steady state (variable D takes longer)
- Transient behavior and diffusion rates at different positions
- Rate of concentration change at boundaries

**What Didn't Change:**
- Final steady-state concentration (uniform C = 0.5)
- Conservation of total mass
- Qualitative diffusion behavior (high to low concentration)

**Key Insight:** Diffusivity affects **how fast** the system reaches equilibrium, but **not the final equilibrium state** itself (with the same boundary conditions).

## Code Structure

### Classes

- **DiffusionSolver**: Main solver class implementing the finite difference method
  - `solve_constant_diffusivity()`: Solves with constant D
  - `solve_variable_diffusivity()`: Solves with variable D(x)

### Functions

- `plot_concentration_evolution()`: Plots concentration profiles over time
- `plot_boundary_concentrations()`: Plots boundary concentrations vs time
- `print_observations()`: Displays and saves observations
- `main()`: Orchestrates the complete solution

## Physical Interpretation

The problem models carbon diffusion in a composite material system, which is relevant in:
- Heat treatment of steels
- Carburization and decarburization processes
- Surface hardening applications
- Multi-layer material systems

The zero-flux boundary conditions represent an isolated system where no carbon can enter or leave, leading to eventual equilibration at the average concentration.

## References

- Finite Difference Method for PDEs
- Fick's Laws of Diffusion
- Implicit (backward Euler) time stepping schemes
- Neumann boundary conditions

## Author

materialistic-aryan06

## License

This project is part of a materials engineering coursework.
