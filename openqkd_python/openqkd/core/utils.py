"""
Primitivas de álgebra quântica para o framework openQKD.
Substitui as funções de QETLAB usadas no openQKDsecurity (MATLAB).
"""

import numpy as np
from scipy.linalg import logm, expm
from typing import List


# ── Verificações de sanidade ──────────────────────────────────────────────────

def is_hermitian(M: np.ndarray, tol: float = 1e-9) -> bool:
    return np.allclose(M, M.conj().T, atol=tol)

def is_positive_semidefinite(M: np.ndarray, tol: float = 1e-9) -> bool:
    if not is_hermitian(M, tol):
        return False
    eigvals = np.linalg.eigvalsh(M)
    return bool(np.all(eigvals >= -tol))

def is_valid_state(rho: np.ndarray, tol: float = 1e-9) -> bool:
    """Verifica se rho é um estado quântico válido (hermitiano, PSD, traço=1)."""
    return (is_positive_semidefinite(rho, tol) and
            abs(np.trace(rho) - 1.0) < tol)


# ── Operações básicas ─────────────────────────────────────────────────────────

def partial_trace(rho: np.ndarray, dims: list, axis: int = 0) -> np.ndarray:
    """
    Traço parcial de rho sobre o subsistema `axis`.
    dims : lista com as dimensões de cada subsistema, ex: [2, 2]
    axis : índice do subsistema a ser traçado (0 = primeiro, 1 = segundo)
    """
    n    = len(dims)
    rho_t = rho.reshape(dims * 2)  # shape: (d0,d1,...,d0,d1,...)
    # contrair índice `axis` com seu par `axis+n`
    result = np.trace(rho_t, axis1=axis, axis2=axis + n)
    remaining = [d for k, d in enumerate(dims) if k != axis]
    dim_out   = int(np.prod(remaining))
    return result.reshape(dim_out, dim_out)

def tensor(*operators: np.ndarray) -> np.ndarray:
    """Produto tensorial de N operadores: A ⊗ B ⊗ C ..."""
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result

def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A

def enforce_hermitian(M: np.ndarray) -> np.ndarray:
    """Força hermiticidade após drift numérico: (M + M†) / 2"""
    return (M + M.conj().T) / 2

def enforce_psd(M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Regulariza M para ser PSD via eigendecomposição."""
    M = enforce_hermitian(M)
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.maximum(eigvals, tol)
    return eigvecs @ np.diag(eigvals) @ eigvecs.conj().T


# ── Canais quânticos ──────────────────────────────────────────────────────────

def apply_channel(kraus_ops: List[np.ndarray], rho: np.ndarray) -> np.ndarray:
    """
    Aplica canal CPTP via operadores de Kraus:
    G(rho) = sum_k  K_k @ rho @ K_k†
    """
    return sum(K @ rho @ K.conj().T for K in kraus_ops)

def dephasing_z(rho: np.ndarray, key_dim: int) -> np.ndarray:
    """
    Dephasing completo no registro de chave:
    Z(rho) = sum_i  (|i><i| ⊗ I) rho (|i><i| ⊗ I)
    """
    total_dim = rho.shape[0]
    block_size = total_dim // key_dim
    result = np.zeros_like(rho)
    for i in range(key_dim):
        P_i = np.zeros((key_dim, key_dim))
        P_i[i, i] = 1.0
        P = np.kron(P_i, np.eye(block_size))
        result += P @ rho @ P
    return result


# ── Entropias ─────────────────────────────────────────────────────────────────

def matrix_log(M: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """log2(M) regularizado — retorna em bits (base 2)."""
    return logm(M + tol * np.eye(M.shape[0])) / np.log(2)


def von_neumann_entropy(rho: np.ndarray, tol: float = 1e-12) -> float:
    """S(rho) = -tr[rho log rho]"""
    rho = enforce_psd(rho, tol)
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > tol]
    return float(-np.sum(eigvals * np.log2(eigvals)))

def quantum_relative_entropy(rho: np.ndarray,
                              sigma: np.ndarray,
                              tol: float = 1e-12) -> float:
    """
    D(rho || sigma) = tr[rho (log rho - log sigma)]
    Retorna inf se supp(rho) não está contido em supp(sigma).
    """
    rho   = enforce_psd(rho, tol)
    sigma = enforce_psd(sigma, tol)
    log_rho   = matrix_log(rho, tol)
    log_sigma = matrix_log(sigma, tol)
    return float(np.real(np.trace(rho @ (log_rho - log_sigma))))

def conditional_entropy_key_rate(rho: np.ndarray,
                                  kraus_ops: List[np.ndarray],
                                  key_dim: int,
                                  tol: float = 1e-12) -> float:
    """
    Calcula H(K|E) via entropia relativa:
    f(rho) = D(G(rho) || Z(G(rho)))
    Este é o núcleo do cálculo da key rate no formalismo de Winick et al.
    """
    Grho  = apply_channel(kraus_ops, rho)
    ZGrho = dephasing_z(Grho, key_dim)
    return quantum_relative_entropy(Grho, ZGrho)
