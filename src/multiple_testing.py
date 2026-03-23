from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t


def _two_sample_t_pvalue(y: np.ndarray, z: np.ndarray) -> float:
    treated = y[z == 1]
    control = y[z == 0]
    n1 = treated.shape[0]
    n0 = control.shape[0]
    s1 = float(np.var(treated, ddof=1))
    s0 = float(np.var(control, ddof=1))
    se = float(np.sqrt(s1 / n1 + s0 / n0))
    diff = float(np.mean(treated) - np.mean(control))
    if se == 0.0:
        return 1.0
    t_stat = diff / se
    df_num = (s1 / n1 + s0 / n0) ** 2
    df_den = ((s1 / n1) ** 2) / (n1 - 1) + ((s0 / n0) ** 2) / (n0 - 1)
    if df_den == 0.0:
        return 1.0
    df = df_num / df_den
    return float(2.0 * t.sf(np.abs(t_stat), df=df))


def simulate_null_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under the complete null for L simulations.
    Return columns: sim_id, hypothesis_id, p_value.
    """
    rng = np.random.default_rng(int(config["seed_null"]))
    n = int(config["N"])
    m = int(config["M"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])

    rows: list[dict[str, float | int]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            y = rng.normal(loc=0.0, scale=1.0, size=n)
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def simulate_mixed_pvalues(config: dict[str, Any]) -> pd.DataFrame:
    """
    Generate p-values under mixed true and false null hypotheses for L simulations.
    Return columns: sim_id, hypothesis_id, p_value, is_true_null.
    """
    rng = np.random.default_rng(int(config["seed_mixed"]))
    n = int(config["N"])
    m = int(config["M"])
    m0 = int(config["M0"])
    l = int(config["L"])
    p_treat = float(config["p_treat"])
    tau_alt = float(config["tau_alternative"])

    rows: list[dict[str, float | int | bool]] = []
    for sim_id in range(l):
        z = (rng.random(n) < p_treat).astype(int)
        for hypothesis_id in range(m):
            is_true_null = hypothesis_id >= (m - m0)
            effect = 0.0 if is_true_null else tau_alt
            y = rng.normal(loc=0.0, scale=1.0, size=n) + effect * z
            p_value = _two_sample_t_pvalue(y=y, z=z)
            rows.append(
                {
                    "sim_id": sim_id,
                    "hypothesis_id": hypothesis_id,
                    "p_value": p_value,
                    "is_true_null": is_true_null,
                }
            )
    return pd.DataFrame(rows)


def bonferroni_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Bonferroni correction.
    """
    p = np.asarray(p_values)
    m = p.shape[0]
    threshold = alpha / m
    return p <= threshold


def holm_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Holm step-down correction.
    """
    p = np.asarray(p_values)
    m = p.shape[0]
    order = np.argsort(p)
    p_sorted = p[order]

    reject_sorted = np.zeros(m, dtype=bool)
    for k in range(m):
        threshold = alpha / (m - k)
        if p_sorted[k] <= threshold:
            reject_sorted[k] = True
        else:
            break

    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject


def benjamini_hochberg_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Hochberg correction.
    """
    p = np.asarray(p_values)
    m = p.shape[0]
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * alpha
    passed = p_sorted <= thresholds

    reject_sorted = np.zeros(m, dtype=bool)
    if np.any(passed):
        k_max = int(np.max(np.where(passed)[0]))
        reject_sorted[: k_max + 1] = True

    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject


def benjamini_yekutieli_rejections(p_values: np.ndarray, alpha: float) -> np.ndarray:
    """
    Return boolean rejection decisions under Benjamini-Yekutieli correction.
    """
    p = np.asarray(p_values)
    m = p.shape[0]
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1)
    harmonic = float(np.sum(1.0 / np.arange(1, m + 1)))
    thresholds = (ranks / m) * (alpha / harmonic)
    passed = p_sorted <= thresholds

    reject_sorted = np.zeros(m, dtype=bool)
    if np.any(passed):
        k_max = int(np.max(np.where(passed)[0]))
        reject_sorted[: k_max + 1] = True

    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject


def compute_fwer(rejections_null: np.ndarray) -> float:
    """
    Return family-wise error rate from a [L, M] rejection matrix under the complete null.
    """
    rej = np.asarray(rejections_null, dtype=bool)
    any_rejection = np.any(rej, axis=1)
    return float(np.mean(any_rejection))


def compute_fdr(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return FDR for one simulation: false discoveries among all discoveries.
    Use 0.0 when there are no rejections.
    """
    rej = np.asarray(rejections, dtype=bool)
    true_null = np.asarray(is_true_null, dtype=bool)
    n_rej = int(np.sum(rej))
    if n_rej == 0:
        return 0.0
    false_discoveries = int(np.sum(rej & true_null))
    return float(false_discoveries / n_rej)


def compute_power(rejections: np.ndarray, is_true_null: np.ndarray) -> float:
    """
    Return power for one simulation: true rejections among false null hypotheses.
    """
    rej = np.asarray(rejections, dtype=bool)
    true_null = np.asarray(is_true_null, dtype=bool)
    false_null = ~true_null
    n_false_null = int(np.sum(false_null))
    if n_false_null == 0:
        return 0.0
    true_rejections = int(np.sum(rej & false_null))
    return float(true_rejections / n_false_null)


def summarize_multiple_testing(
    null_pvalues: pd.DataFrame,
    mixed_pvalues: pd.DataFrame,
    alpha: float,
) -> dict[str, float]:
    """
    Return summary metrics:
      fwer_uncorrected, fwer_bonferroni, fwer_holm,
      fdr_uncorrected, fdr_bh, fdr_by,
      power_uncorrected, power_bh, power_by.
    """
    null_sorted = null_pvalues.sort_values(["sim_id", "hypothesis_id"])
    mixed_sorted = mixed_pvalues.sort_values(["sim_id", "hypothesis_id"])

    null_grouped = list(null_sorted.groupby("sim_id", sort=True))
    mixed_grouped = list(mixed_sorted.groupby("sim_id", sort=True))

    null_rej_uncorrected: list[np.ndarray] = []
    null_rej_bonf: list[np.ndarray] = []
    null_rej_holm: list[np.ndarray] = []
    for _, sim_df in null_grouped:
        p = sim_df["p_value"].to_numpy(dtype=float)
        null_rej_uncorrected.append(p <= alpha)
        null_rej_bonf.append(bonferroni_rejections(p, alpha))
        null_rej_holm.append(holm_rejections(p, alpha))

    fwer_uncorrected = compute_fwer(np.vstack(null_rej_uncorrected))
    fwer_bonferroni = compute_fwer(np.vstack(null_rej_bonf))
    fwer_holm = compute_fwer(np.vstack(null_rej_holm))

    fdr_uncorrected_vals: list[float] = []
    fdr_bh_vals: list[float] = []
    fdr_by_vals: list[float] = []
    power_uncorrected_vals: list[float] = []
    power_bh_vals: list[float] = []
    power_by_vals: list[float] = []

    for _, sim_df in mixed_grouped:
        p = sim_df["p_value"].to_numpy(dtype=float)
        is_true_null = sim_df["is_true_null"].to_numpy(dtype=bool)

        rej_uncorrected = p <= alpha
        rej_bh = benjamini_hochberg_rejections(p, alpha)
        rej_by = benjamini_yekutieli_rejections(p, alpha)

        fdr_uncorrected_vals.append(compute_fdr(rej_uncorrected, is_true_null))
        fdr_bh_vals.append(compute_fdr(rej_bh, is_true_null))
        fdr_by_vals.append(compute_fdr(rej_by, is_true_null))

        power_uncorrected_vals.append(compute_power(rej_uncorrected, is_true_null))
        power_bh_vals.append(compute_power(rej_bh, is_true_null))
        power_by_vals.append(compute_power(rej_by, is_true_null))

    return {
        "fwer_uncorrected": float(np.mean(fwer_uncorrected)),
        "fwer_bonferroni": float(np.mean(fwer_bonferroni)),
        "fwer_holm": float(np.mean(fwer_holm)),
        "fdr_uncorrected": float(np.mean(fdr_uncorrected_vals)),
        "fdr_bh": float(np.mean(fdr_bh_vals)),
        "fdr_by": float(np.mean(fdr_by_vals)),
        "power_uncorrected": float(np.mean(power_uncorrected_vals)),
        "power_bh": float(np.mean(power_bh_vals)),
        "power_by": float(np.mean(power_by_vals)),
    }
