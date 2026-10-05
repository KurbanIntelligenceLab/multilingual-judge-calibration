"""Shared estimators and inference used by every analysis in this package.

Design notes that matter for correctness:

* The same artifact is judged across languages, so errors within a task may
  be dependent. The uncertainty bound and task bootstrap retain that grouping.
  The residual-permutation diagnostic has a stronger, separate assumption:
  exchangeability over the permuted language positions within each group.
* `double_center` is the estimator. It is defined relative to the evaluator
  pool: adding or removing an evaluator changes every cell.
* A shift shared by all evaluators within a language is excluded by double
  centering and is not removed by CBC. `language_effect` reports that shift
  relative to the grand mean; its correctness requires an external anchor.
"""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd

LANGS = ["English", "Arabic", "Turkish", "Chinese", "Hindi",
         "Japanese", "Spanish", "Swahili"]
ABBR = dict(zip(LANGS, ["En", "Ar", "Tr", "Zh", "Hi", "Ja", "Es", "Sw"]))
EVALUATORS = ["GPT-4o", "Gemini", "GPT-5.4", "Sonnet", "DeepSeek", "Qwen"]


# ----------------------------------------------------------------- loading
def load_panel(task_level_csv: str):
    """Return (T, tasks) with T of shape (n_tasks, n_languages, n_evaluators)."""
    tl = pd.read_csv(task_level_csv)
    tasks = sorted(tl.task.unique())
    piv = (tl.pivot_table(index="task", columns=["language", "backbone"],
                          values="score").reindex(tasks))
    T = np.stack([[piv[(l, b)].to_numpy(float) for b in EVALUATORS]
                  for l in LANGS], 0).transpose(2, 0, 1)
    if np.isnan(T).any():
        raise ValueError("panel is not complete; the estimator assumes a "
                         "complete balanced design")
    return T, tasks


# -------------------------------------------------------------- estimators
def double_center(M: np.ndarray) -> np.ndarray:
    """Interaction of a (language x evaluator) cell-mean matrix."""
    return (M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean())


def beta_hat(T: np.ndarray) -> np.ndarray:
    return double_center(T.mean(0))


def language_effect(T: np.ndarray) -> np.ndarray:
    """g(l): the shift shared by all evaluators. Survives the correction."""
    M = T.mean(0)
    return M.mean(1) - M.mean()


def calibrate(T: np.ndarray, beta: np.ndarray | None = None) -> np.ndarray:
    return T - (beta_hat(T) if beta is None else beta)[None]


# ------------------------------------------------------------------ metrics
def gate_spread(T: np.ndarray, gate: float) -> float:
    """Mean over evaluators of the across-language range in pass rate (pp)."""
    r = (T > gate).mean(0)
    return float((r.max(0) - r.min(0)).mean() * 100)


def pass_rate(T: np.ndarray, gate: float) -> np.ndarray:
    return (T > gate).mean(0) * 100


# ---------------------------------------------------------------- inference
def bootstrap(T: np.ndarray, statistic, n_boot: int = 2000, seed: int = 0):
    """Resample TASKS with replacement. Returns an (n_boot, ...) array."""
    rng = np.random.default_rng(seed)
    n = T.shape[0]
    return np.stack([statistic(T[rng.integers(0, n, n)]) for _ in range(n_boot)])


def ci(draws: np.ndarray, alpha: float = 0.05, axis: int = 0):
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=axis)
    return lo, hi


def permutation_test(T: np.ndarray, statistic, axis: str = "language",
                     n_perm: int = 1000, seed: int = 0):
    """Permute fitted additive-null residuals over language positions.

    axis='language' removes the language-by-evaluator interaction while keeping
    every main effect. The diagnostic assumes residual exchangeability for
    the independently permuted within-task/evaluator groups; it does not
    preserve arbitrary within-task dependence. Do NOT centre the residual
    on its task mean first: that
    zeroes every cell mean and makes the null degenerate, which silently turns
    the test into one that can never reject.
    """
    G = T.mean()
    te = T.mean((1, 2)) - G
    le = T.mean((0, 2)) - G
    be = T.mean((0, 1)) - G
    F = G + te[:, None, None] + le[None, :, None] + be[None, None, :]
    E = T - F
    obs = statistic(T)
    rng = np.random.default_rng(seed)
    n, k, m = T.shape
    null = []
    for _ in range(n_perm):
        Ep = np.empty_like(E)
        for t in range(n):
            for b in range(m):
                Ep[t, :, b] = E[t, rng.permutation(k), b]
        null.append(statistic(F + Ep))
    null = np.asarray(null)
    p = (np.sum(null >= obs) + 1) / (n_perm + 1)
    return obs, null, float(p)


# ------------------------------------------------------------ rank reversal
def reversal_table(T: np.ndarray):
    """Per evaluator pair: witness languages, gaps, and the IU p-value.

    The witness pair is selected as the most opposed of 28 candidates, so the
    resulting p-value is NOT valid on its own. Calibrate it with
    permutation_test using `n_reversals` as the statistic.
    """
    from scipy.stats import ttest_1samp
    n, k, m = T.shape
    out = []
    for i, j in itertools.combinations(range(m), 2):
        d = T[:, :, i] - T[:, :, j]
        dm = d.mean(0)
        best = min(((dm[a] * dm[b], a, b)
                    for a, b in itertools.combinations(range(k), 2)),
                   key=lambda z: z[0])
        prod, a, b = best
        if prod >= 0:
            out.append(dict(i=EVALUATORS[i], j=EVALUATORS[j], la=LANGS[a],
                            lb=LANGS[b], da=dm[a], db=dm[b], reversal=False, p=1.0))
            continue
        ps = []
        for idx in (a, b):
            sg = np.sign(dm[idx])
            t = ttest_1samp(d[:, idx] * sg, 0)
            ps.append(t.pvalue / 2 if t.statistic > 0 else 1 - t.pvalue / 2)
        out.append(dict(i=EVALUATORS[i], j=EVALUATORS[j], la=LANGS[a],
                        lb=LANGS[b], da=dm[a], db=dm[b], reversal=True,
                        p=float(max(ps))))
    return pd.DataFrame(out)


def n_reversals(T: np.ndarray) -> int:
    n, k, m = T.shape
    c = 0
    for i, j in itertools.combinations(range(m), 2):
        dm = (T[:, :, i] - T[:, :, j]).mean(0)
        if min(dm[a] * dm[b] for a, b in itertools.combinations(range(k), 2)) < 0:
            c += 1
    return c


def benjamini_hochberg(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        prev = min(prev, p[idx] * m / (rank + 1))
        adj[idx] = prev
    return adj
