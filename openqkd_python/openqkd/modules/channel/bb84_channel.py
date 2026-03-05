"""
BasicBB84ChannelFunc — port fiel do MATLAB.
Aceita errorRate (modo theory) ou depolarization (modo matlab).
"""

import numpy as np
from openqkd.core.qkd_param import QKDParam


def bb84_channel(qkd_input: QKDParam, description: dict) -> dict:
    pz    = description["pz"]
    Gamma = description["observables_joint"]

    # ── parâmetro do canal ────────────────────────────────────────────────
    # O MATLAB usa depolarization ∈ [0,1] diretamente na Choi matrix.
    # Python modo theory: errorRate = QBER → depolarization = 2×QBER
    # Python modo matlab: depolarization passado diretamente
    depol = float(qkd_input.get_param("depolarization", 0.0))

    # ── estado de Bell phi+ depolarizado ─────────────────────────────────
    # Equivalente ao MATLAB:
    #   rhoAB = MaxEntangled(2)
    #   rhoAB = PartialMap(rhoAB, depolarizationChoiMat(2, depol), 2, [2,2])
    phi_plus    = np.array([1., 0., 0., 1.]) / np.sqrt(2)
    rho_bell    = np.outer(phi_plus, phi_plus)
    rho_channel = (1 - depol) * rho_bell + depol * np.eye(4) / 4

    # ── expectation values: gamma_i = tr(Gamma_i @ rho) ──────────────────
    gamma_stats = [
        float(np.real(np.trace(G @ rho_channel)))
        for G in Gamma
    ]
    assert abs(sum(gamma_stats) - 1.0) < 1e-9, \
        f"gamma_stats soma {sum(gamma_stats):.8f}, esperado 1.0"

    # ── QBER na base Z ────────────────────────────────────────────────────
    p_ZZ   = gamma_stats[0] + gamma_stats[1] + gamma_stats[4] + gamma_stats[5]
    p_err_Z = gamma_stats[1] + gamma_stats[4]
    qber_Z  = p_err_Z / p_ZZ if p_ZZ > 0 else 0.0

    # ── QBER na base X ────────────────────────────────────────────────────
    p_XX    = gamma_stats[10] + gamma_stats[11] + gamma_stats[14] + gamma_stats[15]
    p_err_X = gamma_stats[11] + gamma_stats[14]
    qber_X  = p_err_X / p_XX if p_XX > 0 else 0.0

    return {
        "gamma_stats":   gamma_stats,
        "rho_channel":   rho_channel,
        "depolarization": depol,
        "error_rate":    qber_Z,     # alias para compatibilidade
        "qber_Z":        qber_Z,
        "qber_X":        qber_X,
    }
