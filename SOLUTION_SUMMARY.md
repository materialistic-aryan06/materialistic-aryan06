# Finite Difference Method for Diffusion Problem - Solution Summary

## Problem Statement

Implement a numerical solution for Carbon (C) diffusion in a Steel-Fe system using the **Finite Difference Method with Implicit (backward Euler) scheme**.

### System Specifications
- Total system size: L = 1 m
- Initial concentration: C = 1 in steel (0.25 < x < 0.75), C = 0 in Fe
- Boundary conditions: Neumann BC (zero flux) at x = 0 and x = 1

## Solution Overview

This repository contains a complete, tested implementation that addresses all requirements:

### Part (a): Constant Diffusivity
✅ Solved using implicit finite difference method with D = 1 m²/s  
✅ Reaches steady state at ~0.27 s  
✅ Plot generated: `concentration_evolution_constant_D.png`

### Part (b): Variable Diffusivity
✅ Solved with D(x) = 1.1 - x² m²/s  
✅ Reaches steady state at ~1.0 s  
✅ Properly handles variable diffusivity: ∂/∂x(D(x) ∂C/∂x)  
✅ Plot generated: `concentration_evolution_variable_D.png`

### Part (c): Boundary Concentration Evolution
✅ Plots showing concentration at x = 0.25 and x = 0.75 vs time  
✅ Both cases (constant and variable D) compared on same plots  
✅ Files: `boundary_concentration_x025.png`, `boundary_concentration_x075.png`

### Part (d): Observations
✅ Comprehensive written analysis in `diffusion_observations.txt`  
✅ Key findings documented about how diffusivity affects concentration profiles

## Implementation Highlights

### Numerical Method
- **Implicit (backward Euler) scheme** for unconditional stability
- **Sparse matrix solver** (scipy.sparse.linalg.spsolve) for efficiency
- **Automatic convergence detection** to steady state (tolerance: 10⁻⁶)
- **Proper discretization** of variable diffusivity term

### Boundary Conditions
- **Neumann (zero flux)** boundary conditions at both ends
- Implemented using ghost points in the discretization
- Ensures mass conservation throughout the simulation

### Code Quality
- Well-documented with comprehensive docstrings
- Configuration constants for easy parameter tuning
- Clean, modular structure with reusable classes
- Follows Python best practices

### Testing
- Comprehensive test suite with 5 test cases
- All tests pass successfully
- Tests cover:
  - Initialization
  - Mass conservation (< 1% numerical error)
  - Steady state convergence
  - Boundary conditions
  - Variable diffusivity implementation

## Files in This Repository

| File | Description |
|------|-------------|
| `diffusion_solver.py` | Main solver implementation (404 lines) |
| `test_diffusion_solver.py` | Test suite (167 lines) |
| `requirements.txt` | Python dependencies |
| `DIFFUSION_README.md` | Detailed documentation |
| `SOLUTION_SUMMARY.md` | This file |
| `.gitignore` | Git ignore patterns |
| `concentration_evolution_constant_D.png` | Concentration profiles (constant D) |
| `concentration_evolution_variable_D.png` | Concentration profiles (variable D) |
| `boundary_concentration_x025.png` | Left boundary evolution |
| `boundary_concentration_x075.png` | Right boundary evolution |
| `diffusion_observations.txt` | Detailed analysis |

## How to Run

### Installation
```bash
pip install -r requirements.txt
```

### Run the Solver
```bash
python diffusion_solver.py
```

This generates all plots and the observations file.

### Run Tests
```bash
python test_diffusion_solver.py
```

All 5 tests should pass with ✓ marks.

## Key Results

### Numerical Results
- **Constant D**: Reaches steady state at t ≈ 0.267 s
- **Variable D**: Reaches steady state at t ≈ 0.997 s (3.7× slower)
- **Final state**: Both cases reach uniform C ≈ 0.485 (mass conserved)

### Physical Insights

**What Changed:**
- Time to reach steady state (variable D is 3.7× slower)
- Transient behavior and local diffusion rates
- Rate of concentration change at boundaries

**What Didn't Change:**
- Final steady-state concentration (uniform at ~0.485)
- Total mass in the system (conserved by zero-flux BCs)
- Qualitative diffusion behavior (high → low concentration)

**Key Insight:**  
Diffusivity affects **HOW FAST** the system reaches equilibrium, but **NOT** the final equilibrium state (given same boundary conditions and total mass).

## Technical Details

### Discretization
- **Spatial**: nx = 101 points, dx = 0.01 m
- **Temporal**: dt = 0.001 s
- **Grid**: Uniform mesh from x = 0 to x = 1

### Numerical Stability
- **Implicit scheme**: Unconditionally stable for all dt
- **Sparse matrices**: Tridiagonal system for efficiency
- **Convergence**: Automatic detection (max change < 10⁻⁶)

### Mass Conservation
- Zero-flux boundary conditions preserve total mass
- Numerical error < 1% (due to initial sharp discontinuity)
- Steady state matches initial average concentration

## Validation

✅ All physical requirements met  
✅ All numerical requirements satisfied  
✅ All plots generated with proper labels  
✅ Comprehensive observations provided  
✅ Code fully tested and validated  
✅ No security vulnerabilities (CodeQL passed)  
✅ Clean code review feedback addressed

## References

- Finite Difference Method for Partial Differential Equations
- Fick's Laws of Diffusion
- Implicit Time-Stepping Schemes
- Neumann Boundary Conditions
- Numerical Methods for Materials Science

---

**Author**: materialistic-aryan06  
**Date**: November 2025  
**Course**: Materials Engineering - Computational Methods
