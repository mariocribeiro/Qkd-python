"""
BasicBB84WCPDecoyPreset — correto com FW2StepSolver.

Pipeline:
  1. description  → qubit Kraus + observables (squashed)
  2. channel      → decoy LP (Y1_L, e1_U) + rho0
  3. keyrate      → constraints para o FW2StepSolver
  4. mathSolver   → FW2StepSolver (mesmo do BB84 qubit)

Key rate final:  r = Q1_L * r_qubit - Q_mu * fEC * h(E_mu)
"""

from openqkd.core.qkd_param                                    import QKDParam
from openqkd.modules.description.bb84_wcp_decoy_description    import bb84_wcp_decoy_description
from openqkd.modules.channel.bb84_wcp_decoy_channel            import bb84_wcp_decoy_channel
from openqkd.modules.keyrate.bb84_wcp_decoy_keyrate            import bb84_wcp_decoy_keyrate
from openqkd.solvers.fw2step                                   import fw2step_solver


def BasicBB84WCPDecoyPreset(
    eta:          float = 1.0,
    intensities:  list  = None,    # [mu, nu1, nu2, ...]
    d:            float = 1e-6,
    e_d:          float = 0.03,
    pz:           float = 0.5,
    fEC:          float = 1.16,
) -> QKDParam:

    if intensities is None:
        intensities = [0.5, 0.1, 0.01]
    if len(intensities) < 2:
        raise ValueError("Need at least 2 intensities (signal + 1 decoy).")
    if not all(intensities[i] > intensities[i+1]
               for i in range(len(intensities)-1)):
        raise ValueError("Intensities must be strictly decreasing.")

    qkd_input = QKDParam(dimA=2, dimB=2)
    qkd_input.set_param("eta",         eta)
    qkd_input.set_param("intensities", intensities)
    qkd_input.set_param("d",           d)
    qkd_input.set_param("e_d",         e_d)
    qkd_input.set_param("pz",          pz)
    qkd_input.set_param("fEC",         fEC)

    qkd_input.descriptionModule = bb84_wcp_decoy_description
    qkd_input.channelModule     = bb84_wcp_decoy_channel
    qkd_input.keyRateModule     = bb84_wcp_decoy_keyrate
    qkd_input.mathSolverModule  = fw2step_solver   # ← mesmo solver do BB84 qubit

    return qkd_input
