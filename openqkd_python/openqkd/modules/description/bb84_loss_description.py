"""
BB84LossDescriptionFunc — port do BasicBB84Alice2DLossDescription.m
dimA=2 (Schmidt decomp), dimB=3 (qubit + vácuo).
ρ_AB vive em C² ⊗ C³ = 6×6.
Kraus output: R(2) ⊗ B(3) ⊗ C(2) = 12D.
"""

import numpy as np
from openqkd.core.qkd_param import QKDParam


def bb84_loss_description(qkd_input: QKDParam) -> dict:
    pz   = qkd_input.get_param("pz", 0.5)
    dimA = qkd_input.dimA   # 2
    dimB = qkd_input.dimB   # 3

    # ── Alice (2D) ─────────────────────────────────────────────────────────
    e1   = np.array([[1.], [0.]])
    e2   = np.array([[0.], [1.]])
    ketP = np.array([[1.], [1.]]) / np.sqrt(2)
    ketM = np.array([[1.], [-1.]]) / np.sqrt(2)

    POVMsA = [
        pz       * (e1   @ e1.T),
        pz       * (e2   @ e2.T),
        (1 - pz) * (ketP @ ketP.T),
        (1 - pz) * (ketM @ ketM.T),
    ]

    # ── Bob (3D): {|0⟩, |1⟩, |∅⟩} ────────────────────────────────────────
    b0   = np.array([[1.], [0.], [0.]])
    b1   = np.array([[0.], [1.], [0.]])
    bvac = np.array([[0.], [0.], [1.]])
    bP   = np.array([[1.], [1.], [0.]]) / np.sqrt(2)
    bM   = np.array([[1.], [-1.], [0.]]) / np.sqrt(2)

    # 6 outcomes: Z={0,1,∅}, X={+,-,∅}
    POVMsB = [
        pz       * (b0   @ b0.T),
        pz       * (b1   @ b1.T),
        pz       * (bvac @ bvac.T),
        (1 - pz) * (bP   @ bP.T),
        (1 - pz) * (bM   @ bM.T),
        (1 - pz) * (bvac @ bvac.T),
    ]

    # ── 24 observáveis conjuntos (4×6), tamanho 6×6 ────────────────────────
    ann_A = ["Z0", "Z1", "X0", "X1"]
    ann_B = ["Z0", "Z1", "Zvac", "X0", "X1", "Xvac"]
    observables_joint, labels = [], []
    for i, Ai in enumerate(POVMsA):
        for j, Bj in enumerate(POVMsB):
            observables_joint.append(np.kron(Ai, Bj))
            labels.append(f"A{ann_A[i]}_B{ann_B[j]}")

    assert np.allclose(sum(observables_joint), np.eye(dimA * dimB), atol=1e-10), \
        "Observáveis não somam à I_6!"

    # ── Kraus G (CPTNI): 6D → 12D ─────────────────────────────────────────
    # K_Z = pz × kron(I_2A, kron(I_3B, c1_C))    shape 12×6
    # K_X = (1-pz) × kron(H_A, kron(I_3B, c2_C)) shape 12×6
    H_map = e1 @ ketP.T + e2 @ ketM.T          # 2×2: X→Z basis change
    c1 = np.array([[1.], [0.]])                  # |Z⟩ announcement
    c2 = np.array([[0.], [1.]])                  # |X⟩ announcement

    krausOpZ = pz       * np.kron(np.eye(dimA), np.kron(np.eye(dimB), c1))
    krausOpX = (1 - pz) * np.kron(H_map,        np.kron(np.eye(dimB), c2))
    kraus_ops = [krausOpZ, krausOpX]

    kraus_sum = sum(K.conj().T @ K for K in kraus_ops)
    expected  = (pz**2 + (1 - pz)**2) * np.eye(dimA * dimB)
    assert np.allclose(kraus_sum, expected, atol=1e-10), "CPTNI check falhou!"

    # ── Key projectors em RBC (12D) ────────────────────────────────────────
    # R=2D (primeiro sub-sistema); BC = 3×2 = 6D
    proj0 = np.kron(np.diag([1., 0.]), np.eye(dimB * 2))   # |0><0|_R ⊗ I_6
    proj1 = np.kron(np.diag([0., 1.]), np.eye(dimB * 2))   # |1><1|_R ⊗ I_6
    key_proj = [proj0, proj1]
    assert np.allclose(proj0 + proj1, np.eye(dimA * dimB * 2), atol=1e-10)

    return {
        "kraus_ops":         kraus_ops,
        "key_proj":          key_proj,
        "key_dim":           2,
        "observables_joint": observables_joint,
        "POVMsA":            POVMsA,
        "POVMsB":            POVMsB,
        "pz":                pz,
        "basis_info": {
            "announcements_A": ["Z", "Z", "X", "X"],
            "announcements_B": ["Z", "Z", "Zvac", "X", "X", "Xvac"],
            "labels":          labels,
            "n_observables":   24,
        },
    }
