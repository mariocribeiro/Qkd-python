"""
BB84Optimizer: otimiza pz para maximizar a key rate.
Equivalente ao OptimizerFunc.m do openQKDsecurity (MATLAB).
"""

import numpy as np
import copy
from scipy.optimize import minimize_scalar
from openqkd.core.qkd_param import QKDParam


def bb84_optimizer(qkd_input: QKDParam) -> dict:
    best = {"key_rate": -np.inf, "pz": 0.5, "results": {}}

    def neg_key_rate(pz: float) -> float:
        pz = float(np.clip(pz, 0.05, 0.95))   # ← limites mais seguros que 1e-3
        qi = copy.deepcopy(qkd_input)
        qi.set_param("pz", pz)
        qi.options["verbose"] = False

        try:
            desc    = qi.descriptionModule(qi)
            channel = qi.channelModule(qi, desc)
            kr      = qi.keyRateModule(qi, desc, channel)
            results = qi.mathSolverModule(qi, kr)
            key_rate = results.get("key_rate", 0.0)
        except Exception:
            # pz degenerado → penalidade: retorna 0 sem quebrar o otimizador
            return 0.0

        if key_rate > best["key_rate"]:
            best.update({"key_rate": key_rate, "pz": pz, "results": results})
        return -key_rate

    result = minimize_scalar(
        neg_key_rate,
        bounds=(0.5, 0.95),    # ← quebra a simetria: pz ∈ [0.5, 1) é suficiente
        method="bounded",
        options={"xatol": 1e-4}
    )


    best["optimizer_result"] = result
    return best
