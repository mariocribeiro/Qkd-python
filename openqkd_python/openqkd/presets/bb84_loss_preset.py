"""
BasicBB84LossPreset — BB84 com perdas η (transmitância).
dimA=2 (Schmidt decomp), dimB=3 (qubit + vácuo).
Dois modos idênticos ao bb84_preset.py:
  - modo "theory": error_rate = QBER direto
  - modo "matlab": depolarization = 2×QBER
"""

from openqkd.core.qkd_param                        import QKDParam
from openqkd.modules.description.bb84_loss_description import bb84_loss_description
from openqkd.modules.channel.bb84_loss_channel         import bb84_loss_channel
from openqkd.modules.keyrate.bb84_loss_keyrate          import bb84_loss_keyrate
from openqkd.solvers.math_solver                        import math_solver


def BasicBB84LossPreset(
    eta:          float = 1.0,    # transmitância ∈ (0, 1]
    error_rate:   float = None,   # modo theory — QBER direto
    depolarization: float = None, # modo matlab — parâmetro do canal
    pz:           float = 0.5,
    fEC:          float = 1.0,
    optimize_pz:  bool  = False,
) -> QKDParam:

    if error_rate is None and depolarization is None:
        raise ValueError("Forneça error_rate (modo theory) ou depolarization (modo matlab).")
    if error_rate is not None and depolarization is not None:
        raise ValueError("Forneça apenas um: error_rate OU depolarization.")

    if error_rate is not None:
        qber, depol, mode = float(error_rate), 2.0 * float(error_rate), "theory"
    else:
        depol, qber, mode = float(depolarization), float(depolarization) / 2.0, "matlab"

    if not (0 < eta <= 1.0):
        raise ValueError(f"eta deve estar em (0, 1]. Recebido: {eta}")

    qkd_input = QKDParam(dimA=2, dimB=3)
    qkd_input.set_param("errorRate",      qber)
    qkd_input.set_param("depolarization", depol)
    qkd_input.set_param("eta",            eta)
    qkd_input.set_param("pz",             pz)
    qkd_input.set_param("fEC",            fEC)
    qkd_input.set_param("_mode",          mode)

    qkd_input.descriptionModule = bb84_loss_description
    qkd_input.channelModule     = bb84_loss_channel
    qkd_input.keyRateModule     = bb84_loss_keyrate
    qkd_input.mathSolverModule  = math_solver
    qkd_input.optimizerModule   = None   # otimizador de pz pode ser adicionado aqui

    qkd_input.options.update({
        "solver":  "MOSEK",
        "maxIter": 100,
        "tol":     1e-8,
        "verbose": False,
    })
    return qkd_input
