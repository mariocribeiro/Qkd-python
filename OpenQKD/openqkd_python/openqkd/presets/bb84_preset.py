"""
BasicBB84Alice2DPreset — port fiel do MATLAB.
"""

from openqkd.core.qkd_param import QKDParam
from openqkd.modules.description.bb84_description import bb84_description
from openqkd.modules.channel.bb84_channel import bb84_channel
from openqkd.modules.keyrate.bb84_keyrate import bb84_keyrate
from openqkd.solvers.math_solver import math_solver


def BasicBB84Alice2DPreset(error_rate: float = 0.01,
                            pz: float = 0.5) -> QKDParam:
    qkd_input = QKDParam(dimA=2, dimB=2)

    qkd_input.set_param("errorRate", error_rate)
    qkd_input.set_param("pz",        pz)

    qkd_input.descriptionModule = bb84_description
    qkd_input.channelModule     = bb84_channel
    qkd_input.keyRateModule     = bb84_keyrate
    qkd_input.mathSolverModule  = math_solver

    qkd_input.options.update({
        "solver":  "MOSEK",
        "maxIter": 100,
        "tol":     1e-8,
        "verbose": False,
    })
    return qkd_input
