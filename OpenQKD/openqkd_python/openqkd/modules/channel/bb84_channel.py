"""
BasicBB84ChannelFunc — port fiel do MATLAB.
Calcula gamma_i = tr(Gamma_i @ rho) para os 16 observáveis conjuntos.
"""

import numpy as np
from openqkd.core.qkd_param import QKDParam


def bb84_channel(qkd_input: QKDParam, description: dict) -> dict:
    e    = qkd_input.get_param("errorRate", 0.0)
    pz   = description["pz"]
    Gamma = description["observables_joint"]   # 16 operadores

    # ── Estado de Bell phi+ depolarizado ─────────────────────────────────────
    phi_plus    = np.array([1., 0., 0., 1.]) / np.sqrt(2)
    rho_bell    = np.outer(phi_plus, phi_plus)
    p_depol     = 2 * e
    rho_channel = (1 - p_depol) * rho_bell + p_depol * np.eye(4) / 4

    # ── gamma_i = tr(Gamma_i @ rho) — 16 valores ─────────────────────────────
    gamma_stats = [
        float(np.real(np.trace(G @ rho_channel)))
        for G in Gamma
    ]

    # Verificação: soma total = 1
    assert abs(sum(gamma_stats) - 1.0) < 1e-9, \
        f"gamma_stats soma {sum(gamma_stats):.8f}, esperado 1.0"

    # ── QBER na base Z: P(erro | base ZZ) ────────────────────────────────────
    # Índices ZZ: A_Z0_B_Z0=0, A_Z0_B_Z1=1, A_Z1_B_Z0=4, A_Z1_B_Z1=5
    p_ZZ    = gamma_stats[0] + gamma_stats[1] + gamma_stats[4] + gamma_stats[5]
    p_err_Z = gamma_stats[1] + gamma_stats[4]   # discordâncias na base Z
    qber_Z  = p_err_Z / p_ZZ if p_ZZ > 0 else 0.0

    # QBER na base X: análogo
    p_XX    = gamma_stats[10] + gamma_stats[11] + gamma_stats[14] + gamma_stats[15]
    p_err_X = gamma_stats[11] + gamma_stats[14]
    qber_X  = p_err_X / p_XX if p_XX > 0 else 0.0

    return {
        "gamma_stats": gamma_stats,
        "rho_channel": rho_channel,
        "error_rate":  e,
        "qber_Z":      qber_Z,
        "qber_X":      qber_X,
    }
