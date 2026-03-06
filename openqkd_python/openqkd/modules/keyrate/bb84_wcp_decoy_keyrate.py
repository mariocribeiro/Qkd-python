"""
BB84WCPDecoyKeyRateFunc.

Assembles the objective and constraints for FW2StepSolver
using the squashed single-photon state and the decoy LP bounds.
Key rate formula (GLLP):
  r = Q1_L * r_qubit(rho_AB^(1)) - Q_mu * fEC * h(E_mu)
"""

import numpy as np
from openqkd.core.qkd_param import QKDParam


def h(p):
    if p <= 0 or p >= 1: return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def bb84_wcp_decoy_keyrate(qkd_input: QKDParam,
                           description: dict,
                           channel: dict) -> dict:
    fEC  = float(qkd_input.get_param("fEC", 1.16))

    Q1_L = channel["Q1_L"]
    Q_mu = channel["Q_mu"]
    E_mu = channel["E_mu"]
    e1_U = channel["e1_U"]
    rho0 = channel["rho0"]

    # EC cost on the full signal pulse
    leak_ec = Q_mu * fEC * h(E_mu)

    # Upper bound on qubit key rate for single-photon component
    # r_qubit <= 1 - h(e1_U)  (Shor-Preskill / devetak-winter)
    r_qubit_ub = 1.0 - h(e1_U)

    return {
        "rho0":        rho0,
        "kraus_ops":   channel["kraus_ops"],
        "observables": channel["observables"],
        "Q1_L":        Q1_L,
        "leak_ec":     leak_ec,
        "r_qubit_ub":  r_qubit_ub,
        # FW2StepSolver will minimize D(G(rho)||Z(G(rho))) over rho_AB^(1)
        # and return key_rate_qubit; final key rate computed in preset
        "key_rate_scaling": Q1_L,
        "key_rate_offset":  -leak_ec,
    }
