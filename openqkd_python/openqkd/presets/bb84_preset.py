"""
BasicBB84Alice2DPreset — dois modos de operação:

  modo "theory"  → parâmetro errorRate  (QBER direto, comparação analítica)
  modo "matlab"  → parâmetro depolarization (compatível com mainExample.m)
"""

from openqkd.core.qkd_param             import QKDParam
from openqkd.modules.description.bb84_description import bb84_description
from openqkd.modules.channel.bb84_channel          import bb84_channel
from openqkd.modules.keyrate.bb84_keyrate          import bb84_keyrate
from openqkd.solvers.math_solver                   import math_solver
from openqkd.modules.optimizer.bb84_optimizer      import bb84_optimizer


def BasicBB84Alice2DPreset(
    error_rate:     float = None,   # modo theory  — QBER direto
    depolarization: float = None,   # modo matlab  — parâmetro do canal
    pz:             float = 0.5,
    fEC:            float = 1.0,    # eficiência EC (1.0 = Shannon limit)
    optimize_pz:    bool  = False,
) -> QKDParam:

    if error_rate is None and depolarization is None:
        raise ValueError("Forneça error_rate (modo theory) ou depolarization (modo matlab).")
    if error_rate is not None and depolarization is not None:
        raise ValueError("Forneça apenas um: error_rate OU depolarization.")

    # ── conversão entre os dois parâmetros ───────────────────────────────
    if error_rate is not None:
        # modo theory: QBER = errorRate → depolarization = 2×QBER
        qber        = float(error_rate)
        depol       = 2.0 * qber
        mode        = "theory"
    else:
        # modo matlab: depolarization → QBER = depolarization / 2
        depol       = float(depolarization)
        qber        = depol / 2.0
        mode        = "matlab"

    qkd_input = QKDParam(dimA=2, dimB=2)
    qkd_input.set_param("errorRate",     qber)
    qkd_input.set_param("depolarization", depol)
    qkd_input.set_param("pz",            pz)
    qkd_input.set_param("fEC",           fEC)
    qkd_input.set_param("_mode",         mode)

    qkd_input.descriptionModule = bb84_description
    qkd_input.channelModule     = bb84_channel
    qkd_input.keyRateModule     = bb84_keyrate
    qkd_input.mathSolverModule  = math_solver
    qkd_input.optimizerModule   = bb84_optimizer if optimize_pz else None

    qkd_input.options.update({
        "solver":  "MOSEK",
        "maxIter": 100,
        "tol":     1e-8,
        "verbose": False,
    })
    return qkd_input
