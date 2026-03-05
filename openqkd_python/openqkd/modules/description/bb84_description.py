"""
BasicBB84Alice2DDescriptionFunc: port fiel do MATLAB.
G é CPTNI: só captura rounds onde Alice e Bob escolhem a mesma base.
Ref: BasicBB84Alice2DDescriptionFunc.m — openQKDsecurity
"""

import numpy as np
from openqkd.core.qkd_param import QKDParam


def bb84_description(qkd_input: QKDParam) -> dict:
    pz   = qkd_input.get_param("pz", 0.5)
    dimA = qkd_input.dimA   # 2
    dimB = qkd_input.dimB   # 2

    # ── Vetores (equivalentes a zket do MATLAB) ───────────────────────────────
    e1   = np.array([[1.], [0.]])                   # zket(2,1) = |0>
    e2   = np.array([[0.], [1.]])                   # zket(2,2) = |1>
    ketP = np.array([[1.], [1.]]) / np.sqrt(2)      # |+>
    ketM = np.array([[1.], [-1.]]) / np.sqrt(2)     # |->

    # ── POVMs de Alice (= Bob por simetria) ───────────────────────────────────
    # POVMsA = {pz*|0><0|, pz*|1><1|, (1-pz)*|+><+|, (1-pz)*|-><-|}
    POVMsA = [
        pz       * (e1 @ e1.T),
        pz       * (e2 @ e2.T),
        (1 - pz) * (ketP @ ketP.T),
        (1 - pz) * (ketM @ ketM.T),
    ]
    POVMsB = POVMsA   # protocolo simétrico

    # ── Observáveis conjuntos: 4×4 = 16 operadores 4×4 ───────────────────────
    ann_A = ["Z0", "Z1", "X0", "X1"]
    ann_B = ["Z0", "Z1", "X0", "X1"]
    observables_joint = []
    labels = []
    for i, Ai in enumerate(POVMsA):
        for j, Bj in enumerate(POVMsB):
            observables_joint.append(np.kron(Ai, Bj))
            labels.append(f"A{ann_A[i]}_B{ann_B[j]}")

    # Verificação: sum(observables_joint) = I_4
    assert np.allclose(sum(observables_joint), np.eye(dimA * dimB), atol=1e-10), \
        "Observáveis não somam à identidade!"

    # ── Kraus operators do mapa G (CPTNI) — port direto do MATLAB ────────────
    # Mapeiam ρ_AB (4×4) → ρ_RBC (8×8)
    # R = key register (2D), B = Bob (2D), C = announcement (2D)
    #
    # krausOpZ = pz * kron(I_A, kron(I_B, e1))        [8×4]
    # krausOpX = (1-pz) * kron(H, kron(I_B, e2))      [8×4]
    # onde H = e1@ketP.T + e2@ketM.T  (mapa de Hadamard)
    #
    # sum_i K_i†K_i = (pz² + (1-pz)²)*I_4 ≤ I_4  ← CPTNI ✓
    H_map    = e1 @ ketP.T + e2 @ ketM.T   # 2×2
    krausOpZ = pz       * np.kron(np.eye(dimA), np.kron(np.eye(dimB), e1))
    krausOpX = (1 - pz) * np.kron(H_map,        np.kron(np.eye(dimB), e2))
    kraus_ops = [krausOpZ, krausOpX]

    # Verificação CPTNI: sum_i K_i†K_i = (pz²+(1-pz)²)*I_4
    kraus_sum = sum(K.T @ K for K in kraus_ops)
    expected  = (pz**2 + (1-pz)**2) * np.eye(dimA * dimB)
    assert np.allclose(kraus_sum, expected, atol=1e-10), \
        f"CPTNI check falhou! kraus_sum:\n{kraus_sum}"

    # ── Key projection (mapa Z) — port direto do MATLAB ──────────────────────
    # Projeta no registro R do espaço de saída RBC (8D)
    # proj0 = kron(|0><0|_R, I_{BC}) = kron(diag([1,0]), I_4)   [8×8]
    # proj1 = kron(|1><1|_R, I_{BC}) = kron(diag([0,1]), I_4)   [8×8]
    proj0    = np.kron(np.diag([1., 0.]), np.eye(dimB * 2))
    proj1    = np.kron(np.diag([0., 1.]), np.eye(dimB * 2))
    key_proj = [proj0, proj1]

    # Verificação: proj0 + proj1 = I_8
    assert np.allclose(proj0 + proj1, np.eye(dimB * 4), atol=1e-10), \
        "Key projectors não somam à identidade!"

    return {
        "kraus_ops":         kraus_ops,          # [8×4, 8×4]
        "key_proj":          key_proj,            # [8×8, 8×8]
        "key_dim":           2,
        "observables_joint": observables_joint,   # 16 × (4×4)
        "POVMsA":            POVMsA,
        "POVMsB":            POVMsB,
        "pz":                pz,
        "basis_info": {
            "announcements_A": ["Z", "Z", "X", "X"],
            "announcements_B": ["Z", "Z", "X", "X"],
            "labels":          labels,
            "n_observables":   16,
        },
    }
