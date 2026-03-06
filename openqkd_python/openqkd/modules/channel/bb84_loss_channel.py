"""
BB84LossChannelFunc — canal com transmitância η + despolarização.
ρ_loss (6×6): qubit depolarizado embutido em espaço 2×3,
              mais componente vácuo ponderada por (1-η).
"""

import numpy as np
from openqkd.core.qkd_param import QKDParam


def bb84_loss_channel(qkd_input: QKDParam, description: dict) -> dict:
    Gamma = description["observables_joint"]      # 24 × (6×6)
    depol = float(qkd_input.get_param("depolarization", 0.0))
    eta   = float(qkd_input.get_param("eta",            1.0))

    # ── 1. Estado de Bell depolarizado (4×4, espaço A2 ⊗ B2) ───────────────
    phi_plus  = np.array([1., 0., 0., 1.]) / np.sqrt(2)
    rho_qubit = (1 - depol) * np.outer(phi_plus, phi_plus) + depol / 4 * np.eye(4)

    # ── 2. Embedding: A(2D) ⊗ B_qubit(2D) → A(2D) ⊗ B_loss(3D) ───────────
    # rho_qubit[iA*2:iA*2+2, jA*2:jA*2+2] → rho_loss[iA*3:iA*3+2, jA*3:jA*3+2]
    rho_loss = np.zeros((6, 6), dtype=complex)
    for iA in range(2):
        for jA in range(2):
            rho_loss[iA*3:iA*3+2, jA*3:jA*3+2] += (
                eta * rho_qubit[iA*2:iA*2+2, jA*2:jA*2+2]
            )

    # ── 3. Componente vácuo: (1-η)/2 × I_2 ⊗ |∅⟩⟨∅| ──────────────────────
    # rho_A = I_2/2 (invariante em depol para estado max. emaranhado)
    vac_proj = np.zeros((3, 3), dtype=complex)
    vac_proj[2, 2] = 1.0
    rho_loss += (1 - eta) / 2.0 * np.kron(np.eye(2), vac_proj)

    assert abs(np.trace(rho_loss) - 1.0) < 1e-9,    "rho_loss: traço ≠ 1"
    assert np.allclose(rho_loss, rho_loss.conj().T, atol=1e-10), "rho_loss: não hermitiano"

    # ── 4. Estatísticas: γ_i = Tr(Γ_i ρ_loss), 24 valores ─────────────────
    gamma_stats = [float(np.real(np.trace(G @ rho_loss))) for G in Gamma]
    assert abs(sum(gamma_stats) - 1.0) < 1e-9, \
        f"gamma_stats soma {sum(gamma_stats):.8f} ≠ 1.0"

    # ── 5. QBERs e sifting (condicionado em mesma base + detecção) ──────────
    # Layout (Alice i=0..3, Bob j=0..5): índice = i*6 + j
    # Alice: 0=Z0, 1=Z1, 2=X0, 3=X1
    # Bob:   0=Z0, 1=Z1, 2=Zvac, 3=X0, 4=X1, 5=Xvac
    g = gamma_stats

    p_ZZ_det = g[0]  + g[1]  + g[6]  + g[7]   # AZ* ∩ BZ_det
    p_err_Z  = g[1]  + g[6]
    qber_Z   = p_err_Z / p_ZZ_det if p_ZZ_det > 0 else 0.0

    p_XX_det = g[15] + g[16] + g[21] + g[22]   # AX* ∩ BX_det
    p_err_X  = g[16] + g[21]
    qber_X   = p_err_X / p_XX_det if p_XX_det > 0 else 0.0

    return {
        "gamma_stats":    gamma_stats,
        "rho_channel":    rho_loss,
        "depolarization": depol,
        "eta":            eta,
        "error_rate":     qber_Z,
        "qber_Z":         qber_Z,
        "qber_X":         qber_X,
        "p_ZZ_det":       p_ZZ_det,
        "p_XX_det":       p_XX_det,
    }
