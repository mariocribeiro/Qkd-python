"""
MathSolver: interface genérica entre o KeyRate module e o FW2StepSolver.
Equivalente ao MathSolverFunc.m do openQKDsecurity.
"""

from openqkd.solvers.fw2step import fw2step_solver


def math_solver(qkd_input, keyrate: dict) -> dict:
    """
    Recebe o output do KeyRate module e chama o fw2step_solver.
    """
    return fw2step_solver(
        rho0           = keyrate["rho0"],
        kraus_ops      = keyrate["kraus_ops"],
        constraints_fn = keyrate["constraints_fn"],
        key_dim        = keyrate["key_dim"],
        leak_ec        = keyrate["leak_ec"],
        max_iter       = keyrate["max_iter"],
        tol            = keyrate["tol"],
        solver         = keyrate["solver"],
        verbose        = keyrate["verbose"],
        Gamma          = keyrate["Gamma"],
        gamma_stats    = keyrate["gamma_stats"],
        key_proj       = keyrate.get("key_proj", None),   # ← adicione

    )
