"""
BB84LossKeyRateFunc — mesma lógica do caso sem perdas,
mas com 24 observáveis (6×6) e sifting corrigido pela perda η.

leak_EC = fEC × (p_ZZ_det + p_XX_det) × h(QBER_Z)
        → reduz a fEC × (pz²+(1-pz)²) × h(QBER_Z) quando η=1  ✓
"""

import numpy as np
import cvxpy as cp
from openqkd.core.qkd_param import QKDParam


def _h(p: float) -> float:
    if p <= 0 or p >= 1:
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def bb84_loss_keyrate(qkd_input: QKDParam,
                      description: dict,
                      channel: dict) -> dict:

    Gamma       = description["observables_joint"]   # 24 × (6×6)
    gamma_stats = channel["gamma_stats"]             # 24 floats
    qber_Z      = channel["qber_Z"]
    kraus_ops   = description["kraus_ops"]
    key_dim     = description["key_dim"]
    p_ZZ_det    = channel["p_ZZ_det"]
    p_XX_det    = channel["p_XX_det"]

    solver   = qkd_input.options.get("solver",   "MOSEK")
    tol      = qkd_input.options.get("tol",       1e-8)
    max_iter = qkd_input.options.get("maxIter",   100)
    verbose  = qkd_input.options.get("verbose",   False)

    gamma_arr = np.array(gamma_stats, dtype=np.float64)

    def constraints_fn(rho_var):
        cons = [rho_var >> 0, cp.real(cp.trace(rho_var)) == 1]
        for G, g in zip(Gamma, gamma_arr):
            cons.append(cp.real(cp.trace(G @ rho_var)) == g)
        return cons

    # Sifting com perda: fração real de rounds que geram bits de chave
    sifting_with_loss = p_ZZ_det + p_XX_det   # ≈ (pz²+(1-pz)²)·η
    fEC     = float(qkd_input.get_param("fEC", 1.0))
    leak_ec = fEC * sifting_with_loss * _h(qber_Z)

    return {
        "kraus_ops":      kraus_ops,
        "key_proj":       description["key_proj"],
        "constraints_fn": constraints_fn,
        "key_dim":        key_dim,
        "rho0":           np.array(channel["rho_channel"], dtype=np.complex128),
        "leak_ec":        leak_ec,
        "Gamma":          Gamma,
        "gamma_stats":    gamma_stats,
        "solver":         solver,
        "tol":            tol,
        "max_iter":       max_iter,
        "verbose":        verbose,
    }
