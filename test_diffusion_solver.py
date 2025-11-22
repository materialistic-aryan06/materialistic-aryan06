"""
Unit tests for the diffusion solver.
"""

import numpy as np
import sys

# Import the solver
from diffusion_solver import DiffusionSolver


def test_initialization():
    """Test that the solver initializes correctly."""
    print("Testing initialization...")
    solver = DiffusionSolver(L=1.0, nx=101, nt=1000, dt=0.001)
    
    # Check domain length
    assert solver.L == 1.0, "Domain length incorrect"
    
    # Check spatial discretization
    assert len(solver.x) == 101, "Number of spatial points incorrect"
    assert np.isclose(solver.dx, 0.01), "Spatial step size incorrect"
    
    # Check initial concentration
    steel_region = (solver.x > 0.25) & (solver.x < 0.75)
    fe_region = ~steel_region
    assert np.allclose(solver.C[steel_region], 1.0), "Steel region not initialized correctly"
    assert np.allclose(solver.C[fe_region], 0.0), "Fe region not initialized correctly"
    
    print("✓ Initialization test passed!")


def test_mass_conservation():
    """Test that mass is conserved during simulation."""
    print("\nTesting mass conservation...")
    solver = DiffusionSolver(L=1.0, nx=101, nt=100, dt=0.001)
    
    # Calculate initial mass
    initial_mass = np.trapezoid(solver.C, solver.x)
    
    # Run simulation
    C_history, t_history = solver.solve_constant_diffusivity(D=1.0, tolerance=1e-6)
    
    # Calculate final mass
    final_mass = np.trapezoid(C_history[-1], solver.x)
    
    # Check conservation (should be within numerical tolerance)
    # Note: Some numerical error is expected due to discretization and
    # the sharp initial discontinuity at boundaries
    mass_error = abs(final_mass - initial_mass) / initial_mass
    assert mass_error < 0.02, f"Mass not conserved! Error: {mass_error}"
    
    print(f"  Initial mass: {initial_mass:.6f}")
    print(f"  Final mass: {final_mass:.6f}")
    print(f"  Relative error: {mass_error:.2e}")
    print("✓ Mass conservation test passed!")


def test_steady_state():
    """Test that steady state is reached correctly."""
    print("\nTesting steady state convergence...")
    solver = DiffusionSolver(L=1.0, nx=101, nt=2000, dt=0.001)
    
    # Run simulation
    C_history, t_history = solver.solve_constant_diffusivity(D=1.0, tolerance=1e-6)
    
    # At steady state with zero flux BCs, concentration should be uniform
    final_C = C_history[-1]
    # The steady state value should equal the initial average concentration
    expected_steady_state = np.mean(solver._initialize_concentration())
    
    # Check that concentration is uniform
    assert np.allclose(final_C, expected_steady_state, atol=0.01), \
        "Steady state not uniform or incorrect value"
    
    print(f"  Expected steady state: {expected_steady_state}")
    print(f"  Actual steady state: {np.mean(final_C):.6f}")
    print(f"  Standard deviation: {np.std(final_C):.2e}")
    print("✓ Steady state test passed!")


def test_boundary_conditions():
    """Test that Neumann boundary conditions are satisfied."""
    print("\nTesting Neumann boundary conditions...")
    solver = DiffusionSolver(L=1.0, nx=101, nt=500, dt=0.001)
    
    # Run simulation
    C_history, t_history = solver.solve_constant_diffusivity(D=1.0, tolerance=1e-6)
    
    # Check flux at boundaries (should be approximately zero)
    for i, C in enumerate(C_history[1:]):  # Skip initial condition
        # Left boundary: flux ~ (C[1] - C[0]) / dx
        left_flux = abs((C[1] - C[0]) / solver.dx)
        # Right boundary: flux ~ (C[-1] - C[-2]) / dx
        right_flux = abs((C[-1] - C[-2]) / solver.dx)
        
        # Fluxes should be small (not exactly zero due to discretization)
        assert left_flux < 0.1, f"Left boundary flux too large at t={t_history[i+1]}"
        assert right_flux < 0.1, f"Right boundary flux too large at t={t_history[i+1]}"
    
    print("  Left boundary flux: < 0.1")
    print("  Right boundary flux: < 0.1")
    print("✓ Boundary conditions test passed!")


def test_variable_diffusivity():
    """Test that variable diffusivity solver works."""
    print("\nTesting variable diffusivity solver...")
    solver = DiffusionSolver(L=1.0, nx=101, nt=1000, dt=0.001)
    
    # Define variable diffusivity
    D_func = lambda x: 1.1 - x**2
    
    # Run simulation
    C_history, t_history = solver.solve_variable_diffusivity(D_func=D_func, tolerance=1e-6)
    
    # Check that simulation completed
    assert len(C_history) > 1, "Variable diffusivity solver did not run"
    
    # Check mass conservation
    initial_mass = np.trapezoid(solver._initialize_concentration(), solver.x)
    final_mass = np.trapezoid(C_history[-1], solver.x)
    mass_error = abs(final_mass - initial_mass) / initial_mass
    assert mass_error < 0.02, f"Mass not conserved in variable D! Error: {mass_error}"
    
    # Check steady state
    expected_steady_state = np.mean(solver._initialize_concentration())
    assert np.allclose(C_history[-1], expected_steady_state, atol=0.01), \
        "Variable D steady state incorrect"
    
    print(f"  Number of time steps: {len(t_history)}")
    print(f"  Final time: {t_history[-1]:.4f} s")
    print(f"  Mass conservation error: {mass_error:.2e}")
    print("✓ Variable diffusivity test passed!")


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("RUNNING DIFFUSION SOLVER TESTS")
    print("="*70)
    
    try:
        test_initialization()
        test_mass_conservation()
        test_steady_state()
        test_boundary_conditions()
        test_variable_diffusivity()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        return 0
    
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        print("="*70)
        return 1
    
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}")
        print("="*70)
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
