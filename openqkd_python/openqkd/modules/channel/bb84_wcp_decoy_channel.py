"""
BB84WCPDecoyChannelFunc.

1. Runs the independent-basis decoy LP to get Y1_L and e1_U.
2. Computes the effective single-photon gain Q1_L.
3. Builds the squashed single-photon state rho_AB^(1) compatible
   with the FW2StepSolver input format.
"""

import numpy as np
import cvxpy as cp
from scipy.special import factorial
from openqkd.core.qkd_param import QKDParam


def _gain(mu, eta, d):
    return 1.0 - (1.0 - d) * np.exp(-eta * mu)

def _qber(mu, eta, d, e_d):
    Q = _gain(mu, eta, d)
    if Q <= 0: return 0.5
    return (0.5 * d * np.exp(-mu) + e_d * (Q - d * np.exp(-mu))) / Q

def _poisson(mu, n):
    return float(np.exp(-mu) * mu**n / factorial(n))


def _decoy_lp_independent(intensities, Q_obs, E_obs, d, N_max=20,
                          solver="CLARABEL"):
    """
    Independent-basis LP (decoyAnalysisIndependentLP.m).
    Returns Y1_L (lower bound single-photon yield) and
            e1_U (upper bound single-photon phase error).
    """
    P = np.array([[_poisson(mu, n) for n in range(N_max + 1)]
                  for mu in intensities])

    Y  = cp.Variable(N_max + 1, nonneg=True)
    eY = cp.Variable(N_max + 1, nonneg=True)

    cons = [
        Y  <= 1, eY <= Y,
        Y[0]  == d,
        eY[0] == 0.5 * d,
    ]
    for k, mu in enumerate(intensities):
        cons += [
            P[k] @ Y  == Q_obs[k],
            P[k] @ eY == E_obs[k] * Q_obs[k],
        ]

    # Step 1: tightest lower bound on Y_1
    p1 = cp.Problem(cp.Minimize(Y[1]), cons)
    p1.solve(solver=solver)
    Y1_L = max(float(Y[1].value), 0.0) if p1.status in (
        "optimal", "optimal_inaccurate") else 0.0

    if Y1_L <= 0:
        return 0.0, 0.5

    # Step 2: tightest upper bound on e_1 = eY_1 / Y_1
    p2 = cp.Problem(cp.Maximize(eY[1]), cons)
    p2.solve(solver=solver)
    eY1_U = float(eY[1].value) if p2.status in (
        "optimal", "optimal_inaccurate") else Y1_L * 0.5
    e1_U = min(eY1_U / (Y1_L + 1e-15), 0.5)

    return Y1_L, e1_U


def bb84_wcp_decoy_channel(qkd_input: QKDParam, description: dict) -> dict:
    eta = float(qkd_input.get_param("eta",   1.0))
    d   = float(qkd_input.get_param("d",     1e-6))
    e_d = float(qkd_input.get_param("e_d",   0.03))
    pz  = float(qkd_input.get_param("pz",    0.5))
    intensities = list(qkd_input.get_param("intensities", [0.5, 0.1, 0.01]))

    # Observed gains and QBERs for each intensity
    Q_obs = [_gain(mu, eta, d)      for mu in intensities]
    E_obs = [_qber(mu, eta, d, e_d) for mu in intensities]

    # ── Decoy LP ──────────────────────────────────────────────────────────
    Y1_L, e1_U = _decoy_lp_independent(intensities, Q_obs, E_obs, d)

    mu   = intensities[0]   # signal intensity
    Q1_L = Y1_L * mu * np.exp(-mu)   # single-photon gain (lower bound)
    Q_mu = Q_obs[0]
    E_mu = E_obs[0]

    # ── Build squashed single-photon state rho_AB^(1) ─────────────────────
    # After squashing, rho_AB^(1) is a 4x4 qubit-qubit state constrained by:
    #   Tr[Gamma_ZZ  rho] = pz^2 * Y1_L           (correct Z-Z coincidences)
    #   Tr[Gamma_err rho] = pz^2 * Y1_L * e1_U    (Z-Z errors, i.e., phase)
    # We parametrize the closest separable state consistent with these stats.
    # The FW2StepSolver will then optimize over all rho consistent with them.

    rho0 = _build_rho0(Y1_L, e1_U, pz)

    return {
        "rho0":          rho0,
        "Y1_L":          Y1_L,
        "e1_U":          e1_U,
        "Q1_L":          Q1_L,
        "Q_mu":          Q_mu,
        "E_mu":          E_mu,
        "mu":            mu,
        "intensities":   intensities,
        "kraus_ops":     description["kraus_ops"],
        "observables":   description["observables"],
    }


def _build_rho0(Y1_L, e1_U, pz):
    """
    Initial guess for rho_AB^(1): tensor product state
    consistent with Y1_L and e1_U at the single-photon level.
    Bit error rate in Z basis is e_bit = e_d (optical),
    phase error rate is e1_U.
    """
    # Werner-like state in qubit-qubit space
    # rho = (1-q)|Phi+><Phi+| + q * I/4
    # tuned so that phase error matches e1_U
    q = 4.0 * e1_U / 3.0
    q = min(max(q, 0.0), 1.0)
    phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho_pp   = np.outer(phi_plus, phi_plus)
    rho0     = (1 - q) * rho_pp + q * np.eye(4) / 4.0
    return rho0
