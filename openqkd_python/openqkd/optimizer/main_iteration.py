"""
MainIteration: pipeline principal do framework openQKD.
Equivalente ao MainIteration.m do openQKDsecurity.
"""

from openqkd.core.qkd_param import QKDParam


def MainIteration(qkd_input: QKDParam) -> dict:

    # ── modo com otimização de parâmetros ─────────────────────────────────
    if qkd_input.optimizerModule is not None:
        results = qkd_input.optimizerModule(qkd_input)
        qkd_input.results = results
        return results

    # ── modo direto (sem otimização) ──────────────────────────────────────
    desc    = qkd_input.descriptionModule(qkd_input)
    channel = qkd_input.channelModule(qkd_input, desc)
    kr      = qkd_input.keyRateModule(qkd_input, desc, channel)
    results = qkd_input.mathSolverModule(qkd_input, kr)

    results["error_rate"] = channel["error_rate"]
    results["qber_Z"]     = channel["qber_Z"]
    results["qber_X"]     = channel["qber_X"]
    results["leak_ec"]    = kr["leak_ec"]

    qkd_input.results = results
    return results
