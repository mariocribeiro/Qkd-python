"""
BasicBB84KeyRateFunc — port fiel do MATLAB.
"""

import numpy as np
import cvxpy as cp
from openqkd.core.qkd_param import QKDParam


def binary_entropy(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def bb84_keyrate(qkd_input: QKDParam,
                 description: dict,
                 channel: dict) -> dict:
    Gamma       = description["observables_joint"]   # 16 × (4×4)
    gamma_stats = channel["gamma_stats"]             # 16 valores
    qber_Z      = channel["qber_Z"]
    kraus_ops   = description["kraus_ops"]
    key_dim     = description["key_dim"]
    pz          = description["pz"]
    solver      = qkd_input.options.get("solver",  "MOSEK")
    tol         = qkd_input.options.get("tol",     1e-8)
    max_iter    = qkd_input.options.get("maxIter", 100)
    verbose     = qkd_input.options.get("verbose", False)

    gamma_arr = np.array(gamma_stats, dtype=np.float64)

    def constraints_fn(rho_var):
        cons = [
            rho_var >> 0,
            cp.real(cp.trace(rho_var)) == 1,
        ]
        for G, g in zip(Gamma, gamma_arr):
            cons.append(cp.real(cp.trace(G @ rho_var)) == g)
        return cons

    rho0 = channel["rho_channel"]

    # leak_EC em bits/round (inclui fator de sifting)
    # sifting_prob = fração de rounds que geram chave = pz² + (1-pz)²
    sifting_prob = pz**2 + (1 - pz)**2
    leak_ec      = sifting_prob * binary_entropy(qber_Z)

    return {
        "kraus_ops":      kraus_ops,
        "key_proj":       description["key_proj"],
        "constraints_fn": constraints_fn,
        "key_dim":        key_dim,
        "rho0":           rho0,
        "leak_ec":        leak_ec,
        "Gamma":          Gamma,
        "gamma_stats":    gamma_stats,
        "solver":         solver,
        "tol":            tol,
        "max_iter":       max_iter,
        "verbose":        verbose,
    }

