"""
A.7 — Fix 3: Root-Verb Residualization + Surprisal-Confidence Correlation.

Both surprisal and confidence are OLS-residualized on the binary `is_root`
covariate before computing Pearson r and Spearman rho, to remove the
confound that root tokens differ systematically in both surprisal and
attention confidence.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def residualize(y, covariate):
    X = np.column_stack([np.ones(len(y)), np.array(covariate, dtype=float)])
    b = np.linalg.lstsq(X, np.array(y, dtype=float), rcond=None)[0]
    return np.array(y, dtype=float) - X @ b


def run_correlation(df: pd.DataFrame):
    """
    df must contain columns: 'surprisal', 'conf_max', 'is_root'
    """
    surp_res = residualize(df["surprisal"], df["is_root"].astype(float))
    conf_res = residualize(df["conf_max"], df["is_root"].astype(float))

    r, p = pearsonr(surp_res, conf_res)
    rho, p_rho = spearmanr(surp_res, conf_res)

    return {
        "pearson_r": r,
        "pearson_p": p,
        "spearman_rho": rho,
        "spearman_p": p_rho,
        "n": len(df),
    }
