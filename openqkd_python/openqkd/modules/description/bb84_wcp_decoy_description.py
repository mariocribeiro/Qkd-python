"""
BB84WCPDecoyDescriptionFunc.
Identical to qubit BB84 description — squashing guarantees
the single-photon component lives in the qubit subspace.
"""

import numpy as np
from openqkd.core.qkd_param import QKDParam


def bb84_wcp_decoy_description(qkd_input: QKDParam) -> dict:
    pz = float(qkd_input.get_param("pz", 0.5))

    # ── Kraus operators: same as qubit BB84 after squashing ────────────────
    # Z basis
    K_Z0 = np.sqrt(pz) * np.array([[1,0],[0,0]])   # |0><0|
    K_Z1 = np.sqrt(pz) * np.array([[0,0],[0,1]])   # |1><1|
    # X basis
    K_X0 = np.sqrt(1-pz) * np.array([[1, 1],[1, 1]]) / 2   # |+><+|
    K_X1 = np.sqrt(1-pz) * np.array([[1,-1],[-1,1]]) / 2   # |-><-|

    kraus_ops = [K_Z0, K_Z1, K_X0, K_X1]

    # ── Observables: coincidences por base e resultado ────────────────────
    # Alice prepares |0>,|1>,|+>,|-> with prob pz,pz,(1-pz),(1-pz)
    # Bob measures in Z or X; observable = projector onto matching outcome
    # dim(rho_AB) = 4x4  (Alice 2D after Schmidt decomposition x Bob qubit)
    observables = {}
    observables["Gamma_ZZ"] = np.kron(np.diag([pz,0,0,0]),   np.eye(2))
    observables["Gamma_ZX"] = np.kron(np.diag([0,pz,0,0]),   np.eye(2))
    observables["Gamma_XZ"] = np.kron(np.diag([0,0,1-pz,0]), np.eye(2))
    observables["Gamma_XX"] = np.kron(np.diag([0,0,0,1-pz]), np.eye(2))

    return {
        "kraus_ops":   kraus_ops,
        "observables": observables,
        "dim_A":       2,
        "dim_B":       2,
    }
