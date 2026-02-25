"""
MainIteration: pipeline principal do framework openQKD.
Equivalente ao MainIteration.m do openQKDsecurity.
"""

from openqkd.core.qkd_param import QKDParam


def MainIteration(qkd_input: QKDParam) -> dict:
    """
    Executa o pipeline completo:
    Preset → Description → Channel → KeyRate → MathSolver

    Equivalente a:
        results = MainIteration(qkdInput)  % MATLAB
    """
    desc    = qkd_input.descriptionModule(qkd_input)
    channel = qkd_input.channelModule(qkd_input, desc)
    kr      = qkd_input.keyRateModule(qkd_input, desc, channel)
    results = qkd_input.mathSolverModule(qkd_input, kr)

    # Adiciona metadados ao resultado
    results["error_rate"] = channel["error_rate"]
    results["qber_Z"]     = channel["qber_Z"]
    results["qber_X"]     = channel["qber_X"]
    results["leak_ec"]    = kr["leak_ec"]

    qkd_input.results = results
    return results
