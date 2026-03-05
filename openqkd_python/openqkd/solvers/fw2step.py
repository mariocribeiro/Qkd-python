"""
FW2StepSolver: Frank-Wolfe + linearização + dual certificate.
Equivalente ao FW2StepSolver.m do openQKDsecurity (MATLAB).
Baseado em: Winick et al., Quantum 2 (2018). arXiv:1710.05511
"""

import numpy as np
import cvxpy as cp
from typing import List, Callable, Optional

from openqkd.core.utils import (
    apply_channel, dephasing_z, quantum_relative_entropy,
    enforce_hermitian, enforce_psd, matrix_log
)


# ── Dephasing via key projectors ──────────────────────────────────────────────

def dephasing_from_proj(sigma: np.ndarray,
                         key_proj: List[np.ndarray]) -> np.ndarray:
    """
    Z(sigma) = sum_i P_i @ sigma @ P_i
    Usa os key projectors do Description module (espaço RBC).
    Substitui dephasing_z quando key_proj está disponível.
    """
    return sum(P @ sigma @ P for P in key_proj)


def apply_dephasing(sigma: np.ndarray,
                    key_dim: int,
                    key_proj: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """
    Dispatcher: usa key_proj se disponível, senão dephasing_z genérico.
    """
    if key_proj is not None:
        return dephasing_from_proj(sigma, key_proj)
    return dephasing_z(sigma, key_dim)


# ── Gradiente de f(rho) ───────────────────────────────────────────────────────

def compute_gradient(rho: np.ndarray,
                     kraus_ops: List[np.ndarray],
                     key_dim: int,
                     key_proj: Optional[List[np.ndarray]] = None,
                     tol: float = 1e-12) -> np.ndarray:
    """
    Gradiente de f(rho) = D(G(rho) || Z(G(rho))) via regra da cadeia:
        grad_rho f = G†( log G(rho) - log Z(G(rho)) )
    onde G† é o mapa adjunto (dual) do canal G.
    """
    Grho  = apply_channel(kraus_ops, rho)
    ZGrho = apply_dephasing(Grho, key_dim, key_proj)

    grad_G = matrix_log(Grho, tol) - matrix_log(ZGrho, tol)

    # pullback via mapa adjunto G†: sum_k K†_k @ grad_G @ K_k
    return enforce_hermitian(
        sum(K.conj().T @ grad_G @ K for K in kraus_ops)
    )


# ── Passo de Frank-Wolfe (SDP linear) ────────────────────────────────────────

def frank_wolfe_step(grad: np.ndarray,
                     constraints_fn,
                     dim: int,
                     solver: str = "MOSEK") -> np.ndarray:

    sigma     = cp.Variable((dim, dim), hermitian=True)
    objective = cp.Minimize(cp.real(cp.trace(grad @ sigma)))
    prob      = cp.Problem(objective, constraints_fn(sigma))

    # ── tenta solver primário; cai para CLARABEL se falhar ───────────────
    try:
        prob.solve(solver=solver)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"status={prob.status}")
    except Exception:
        prob.solve(solver="CLARABEL")
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"FW step SDP falhou com ambos os solvers: {prob.status}")

    return enforce_hermitian(np.array(sigma.value))


# ── Busca de linha (line search) ──────────────────────────────────────────────

from scipy.optimize import minimize_scalar

def line_search(rho: np.ndarray,
                direction: np.ndarray,
                kraus_ops: List[np.ndarray],
                key_dim: int,
                key_proj: Optional[List[np.ndarray]] = None,
                tol: float = 1e-9) -> float:
    """
    Busca de linha exata via método de Brent (scipy).
    Substitui o grid de 20 pontos — O(log(1/tol)) avaliações.
    """
    def objective(gamma: float) -> float:
        rho_t = enforce_psd(rho + gamma * direction)
        Grho  = apply_channel(kraus_ops, rho_t)
        ZGrho = apply_dephasing(Grho, key_dim, key_proj)
        return quantum_relative_entropy(Grho, ZGrho)

    result = minimize_scalar(
        objective,
        bounds=(0.0, 1.0),
        method='bounded',
        options={'xatol': tol}
    )
    return float(result.x)




# ── Loop principal de Frank-Wolfe ─────────────────────────────────────────────

def frank_wolfe_loop(rho0, kraus_ops, constraints_fn, key_dim,
                     key_proj=None, max_iter=100, tol=1e-8,
                     solver="MOSEK", verbose=False) -> tuple:

    rho     = enforce_psd(rho0.copy())
    history = {"f_vals": [], "fw_gaps": [], "gammas": []}
    f_prev  = np.inf
    plateau = 0

    for k in range(max_iter):
        grad  = compute_gradient(rho, kraus_ops, key_dim, key_proj)
        sigma = frank_wolfe_step(grad, constraints_fn, rho.shape[0], solver)

        Grho   = apply_channel(kraus_ops, rho)
        ZGrho  = apply_dephasing(Grho, key_dim, key_proj)
        f_rho  = quantum_relative_entropy(Grho, ZGrho)
        fw_gap = f_rho - float(np.real(np.trace(grad @ sigma)))

        if verbose:
            print(f"  iter {k+1:3d} | f(rho)={f_rho:.8f} | FW gap={fw_gap:.2e}")

        # ── critério 1: plateau em f(rho) — robusto a ruído do SDP ──────
        if abs(f_rho - f_prev) < tol:
            plateau += 1
        else:
            plateau = 0
        f_prev = f_rho

        # ── critério 2: gap loose — 1000x tol para absorver ruído SDP ───
        gap_loose = abs(fw_gap) < tol * 1000   # 1e-5 para tol=1e-8

        if plateau >= 2 or gap_loose:
            if verbose:
                print(f"  Convergiu em {k+1} iterações.")
            break
        # ─────────────────────────────────────────────────────────────────

        direction = sigma - rho
        gamma     = line_search(rho, direction, kraus_ops, key_dim, key_proj)
        rho       = enforce_psd(rho + gamma * direction)

        history["f_vals"].append(f_rho)
        history["fw_gaps"].append(fw_gap)
        history["gammas"].append(gamma)

    return rho, fw_gap, history


# ── Dual certificate simplificado (gap linear) ───────────────────────────────

def dual_certificate(rho_star: np.ndarray,
                     kraus_ops: List[np.ndarray],
                     constraints_fn: Callable,
                     key_dim: int,
                     key_proj: Optional[List[np.ndarray]] = None,
                     solver: str = "MOSEK") -> dict:
    """
    Lower bound via gap de linearização.
    Não fornece garantia formal — usar apenas para desenvolvimento.
    Substituído por dual_certificate_winick quando Gamma/gamma_stats
    estiverem disponíveis.
    """
    dim  = rho_star.shape[0]
    grad = compute_gradient(rho_star, kraus_ops, key_dim, key_proj)

    Grho        = apply_channel(kraus_ops, rho_star)
    ZGrho       = apply_dephasing(Grho, key_dim, key_proj)
    upper_bound = quantum_relative_entropy(Grho, ZGrho)

    sigma_var = cp.Variable((dim, dim), hermitian=True)
    dual_prob = cp.Problem(
        cp.Maximize(cp.real(cp.trace(grad @ sigma_var))),
        constraints_fn(sigma_var)
    )
    dual_prob.solve(solver=solver)

    if dual_prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Dual SDP falhou: status={dual_prob.status}")

    max_linear  = float(np.real(dual_prob.value))
    epsilon     = abs(upper_bound - max_linear)
    lower_bound = upper_bound - epsilon

    return {
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "epsilon":     epsilon,
        "gap":         upper_bound - lower_bound,
    }


# ── Dual certificate rigoroso — Winick et al. Teorema 1/2/3 ──────────────────

def dual_certificate_winick(rho_star: np.ndarray,
                             kraus_ops: List[np.ndarray],
                             key_dim: int,
                             Gamma: List[np.ndarray],
                             gamma_stats: List[float],
                             key_proj: Optional[List[np.ndarray]] = None,
                             epsilon: Optional[float] = None,
                             epsilon_prime: float = 1e-12,
                             solver: str = "MOSEK") -> dict:
    """
    Implementação rigorosa do Teorema 2 + 3 de Winick et al. (arXiv:1710.05511).
    Garante lower bound certificado mesmo para estados na fronteira do
    conjunto viável.

    Parâmetros
    ----------
    rho_star    : estado convergido pelo FW loop
    kraus_ops   : operadores de Kraus do canal G
    key_dim     : dimensão do registro de chave
    Gamma       : lista de operadores observáveis {Gamma_i} do protocolo
    gamma_stats : lista de estatísticas observadas {gamma_i}
    key_proj    : key projectors do Description module (opcional)
    epsilon     : parâmetro de regularização (None = automático)
    epsilon_prime: tolerância de precisão finita (Teorema 3)
    solver      : solver CVXPY
    """
    dim   = rho_star.shape[0]
    d_out = apply_channel(kraus_ops, rho_star).shape[0]
    n     = len(Gamma)

    # ── Sanitiza entradas para float64 ───────────────────────────────────────
    rho_star  = np.array(rho_star,  dtype=np.complex128)
    kraus_ops = [np.array(K, dtype=np.complex128) for K in kraus_ops]
    Gamma     = [np.array(G, dtype=np.complex128) for G in Gamma]
    gamma_real = np.array(gamma_stats,    dtype=np.float64)
    if key_proj is not None:
        key_proj = [np.array(P.real,      dtype=np.float64) for P in key_proj]

    # ── Escolha automática de epsilon (Teorema 2) ─────────────────────────────
    # fw2step.py — dentro de dual_certificate_winick
    if epsilon is None:
    # Winick eq. (20): epsilon tal que zeta ≈ tol numérico
    # zeta = 2*eps*(d-1)*log(d/(eps*(d-1))) ≈ 2*eps*(d-1)*log(1/eps)
    # Para zeta < 1e-6: eps ≈ tol / (2*(d-1)*log(1/tol))
        tol_target = 1e-6
        d = d_out
        if d > 1:
            import math
            eps_auto = tol_target / (2 * (d - 1) * math.log(1 / tol_target))
            epsilon  = max(eps_auto, 1e-10)  # floor numérico
        else:
            epsilon = 1e-8


    # ── Canal perturbado G_epsilon = (1-ε)G + ε*I/d' ─────────────────────────
    def apply_channel_eps(rho):
        Grho = apply_channel(kraus_ops, rho)
        return (1 - epsilon) * Grho + epsilon * np.eye(d_out) / d_out

    sqrt_1meps = np.sqrt(1 - epsilon)
    sqrt_eps   = np.sqrt(epsilon / d_out)
    kraus_eps  = (
        [sqrt_1meps * K for K in kraus_ops] +
        [sqrt_eps * (np.eye(d_out)[:, [j]] @ np.ones((1, dim)))
         for j in range(d_out)]
    )

    # ── Gradiente com canal perturbado ────────────────────────────────────────
    Grho_eps  = apply_channel_eps(rho_star)
    ZGrho_eps = apply_dephasing(Grho_eps, key_dim, key_proj)
    grad_G    = matrix_log(Grho_eps) - matrix_log(ZGrho_eps)
    grad      = enforce_hermitian(
        sum(K.conj().T @ grad_G @ K for K in kraus_eps)
    )
    grad = np.array(grad.real, dtype=np.float64)

    # ── f_epsilon(rho*) e tr(rho^T * grad) ───────────────────────────────────
    f_eps   = quantum_relative_entropy(Grho_eps, ZGrho_eps)
    tr_term = float(np.real(np.trace(rho_star.conj().T @ grad)))

    # ── SDP dual (Eq. 17-18 de Winick et al.) ────────────────────────────────
    y   = cp.Variable(n)
    # dentro do SDP dual — trocar sum direto por conjugação explícita
    lhs = sum(cp.real(y[i]) * cp.real(Gamma[i].T) +
          cp.imag(y[i]) * cp.imag(Gamma[i].T)
          for i in range(n))

    dual_constraints = [(grad - lhs) >> 0]

    if epsilon_prime > 0:
        z = cp.Variable(n, nonneg=True)
        dual_constraints += [-z <= y, y <= z]
        obj = cp.Maximize(gamma_real @ y - epsilon_prime * cp.sum(z))
    else:
        obj = cp.Maximize(gamma_real @ y)

    dual_prob = cp.Problem(obj, dual_constraints)
    dual_prob.solve(solver=solver)

    if dual_prob.status not in ["optimal", "optimal_inaccurate"]:
        raise RuntimeError(f"Dual SDP (Winick) falhou: {dual_prob.status}")

    # ── Beta, zeta e lower bound (Eq. 19) ────────────────────────────────────
    beta = f_eps - tr_term + float(dual_prob.value)
    zeta = (2 * epsilon * (d_out - 1) * np.log2(d_out / (epsilon * (d_out - 1)))
        if d_out > 1 else 0.0)
    lower_bound = beta - zeta

    # upper bound: valor primal sem perturbação
    Grho        = apply_channel(kraus_ops, rho_star)
    ZGrho       = apply_dephasing(Grho, key_dim, key_proj)
    upper_bound = quantum_relative_entropy(Grho, ZGrho)

    return {
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "epsilon":     epsilon,
        "beta":        beta,
        "zeta":        zeta,
        "gap":         upper_bound - lower_bound,
    }


# ── Pipeline completo: FW2StepSolver ─────────────────────────────────────────

def fw2step_solver(rho0: np.ndarray,
                   kraus_ops: List[np.ndarray],
                   constraints_fn: Callable,
                   key_dim: int,
                   leak_ec: float = 0.0,
                   max_iter: int = 100,
                   tol: float = 1e-8,
                   solver: str = "MOSEK",
                   verbose: bool = False,
                   Gamma: Optional[List[np.ndarray]] = None,
                   gamma_stats: Optional[List[float]] = None,
                   key_proj: Optional[List[np.ndarray]] = None) -> dict:
    """
    Solver completo: Frank-Wolfe → dual certificate → key rate.

    Parâmetros
    ----------
    rho0          : ponto inicial
    kraus_ops     : operadores de Kraus do canal G
    constraints_fn: função sigma_var → lista de constraints CVXPY
    key_dim       : dimensão do registro de chave
    leak_ec       : vazamento de correção de erros (bits/round)
    solver        : "MOSEK" (produção) ou "CLARABEL" (desenvolvimento)
    verbose       : imprime progresso do FW loop
    Gamma         : observáveis — ativa certificado rigoroso de Winick
    gamma_stats   : estatísticas observadas correspondentes a Gamma
    key_proj      : key projectors do Description module
    """
    if verbose:
        print("── Frank-Wolfe loop ──")

    rho_star, fw_gap, history = frank_wolfe_loop(
        rho0, kraus_ops, constraints_fn, key_dim,
        key_proj, max_iter, tol, solver, verbose
    )


    if verbose:
        print("── Dual certificate ──")

    if Gamma is not None and gamma_stats is not None:
        if verbose:
            print("  (modo rigoroso: Winick et al. Teorema 1/2/3)")
        cert = dual_certificate_winick(
            rho_star, kraus_ops, key_dim,
            Gamma, gamma_stats,
            key_proj=key_proj, solver=solver
        )
    else:
        if verbose:
            print("  (modo simplificado)")
        cert = dual_certificate(
            rho_star, kraus_ops, constraints_fn,
            key_dim, key_proj, solver
        )

    key_rate = max(0.0, cert["lower_bound"] - leak_ec)

    if verbose:
        print(f"\n  Upper bound : {cert['upper_bound']:.8f} bits/round")
        print(f"  Lower bound : {cert['lower_bound']:.8f} bits/round")
        print(f"  Epsilon     : {cert['epsilon']:.2e}")
        print(f"  Leak EC     : {leak_ec:.8f} bits/round")
        print(f"  Key rate    : {key_rate:.8f} bits/round")

    return {
        "key_rate":    key_rate,
        "upper_bound": cert["upper_bound"],
        "lower_bound": cert["lower_bound"],
        "epsilon":     cert["epsilon"],
        "fw_gap":      fw_gap,
        "rho_star":    rho_star,
        "history":     history,          # ← novo: 2.3
    }

