"""
QKDParam: estrutura central de parâmetros do protocolo.
Equivalente ao classdef QKDParam do openQKDsecurity (MATLAB).
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np


@dataclass
class QKDParam:
    # ── Dimensões do protocolo ────────────────────────────────────────────────
    dimA: int = 0           # dimensão do sistema de Alice
    dimB: int = 0           # dimensão do sistema de Bob
    dimAB: int = 0          # dimensão do sistema conjunto (calculado)

    # ── Módulos plugáveis (callbacks) ─────────────────────────────────────────
    # Espelham os function handles do MATLAB
    descriptionModule: Optional[Callable] = None
    channelModule:     Optional[Callable] = None
    keyRateModule:     Optional[Callable] = None
    mathSolverModule:  Optional[Callable] = None
    optimizerModule:   Optional[Callable] = None

    # ── Parâmetros do protocolo e do canal ────────────────────────────────────
    params: dict = field(default_factory=dict)
    # Ex: {"errorRate": 0.01, "transmittance": 0.9, "darkCount": 1e-6}

    # ── Opções numéricas ──────────────────────────────────────────────────────
    options: dict = field(default_factory=dict)
    # Ex: {"solver": "CLARABEL", "maxIter": 100, "tol": 1e-8, "verbose": False}

    # ── Resultados (preenchidos pelo pipeline) ────────────────────────────────
    results: dict = field(default_factory=dict)

    def __post_init__(self):
        # calcula dimAB automaticamente se dimA e dimB foram fornecidos
        if self.dimA > 0 and self.dimB > 0 and self.dimAB == 0:
            self.dimAB = self.dimA * self.dimB
        # opções padrão
        defaults = {
            "solver":  "CLARABEL",
            "maxIter": 100,
            "tol":     1e-8,
            "verbose": False,
        }
        for k, v in defaults.items():
            self.options.setdefault(k, v)

    def set_param(self, key: str, value) -> None:
        self.params[key] = value

    def get_param(self, key: str, default=None):
        return self.params.get(key, default)

    def summary(self) -> None:
        print(f"QKDParam | dimA={self.dimA}, dimB={self.dimB}, dimAB={self.dimAB}")
        print(f"  params  : {self.params}")
        print(f"  options : {self.options}")
        print(f"  modules : description={self.descriptionModule is not None}, "
              f"channel={self.channelModule is not None}, "
              f"keyrate={self.keyRateModule is not None}")
