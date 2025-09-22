import math
from typing import List
import numpy as np
from scipy import stats

def cpd_adaptive_cusum(series: np.ndarray, threshold: float = 3.0, drift: float = 0.0) -> List[int]:
    mean = 0.0
    var = 0.0
    n = 0
    s_pos = 0.0
    s_neg = 0.0
    change_points = []
    for i, x in enumerate(series):
        n += 1
        old_mean = mean
        mean += (x - mean) / n
        var += (x - old_mean) * (x - mean)
        std = math.sqrt(var / n) if n > 1 else 1e-6
        z = (x - mean) / (std + 1e-8)
        s_pos = max(0.0, s_pos + z - drift)
        s_neg = min(0.0, s_neg + z + drift)
        if s_pos > threshold:
            change_points.append(i)
            s_pos = 0.0
        elif abs(s_neg) > threshold:
            change_points.append(i)
            s_neg = 0.0
    return sorted(list(set(change_points)))

def cpd_rolling_stat(series: np.ndarray, window: int = 20, alpha: float = 0.01) -> List[int]:
    n = len(series)
    cps = []
    for t in range(2*window, n+1):
        w1 = series[t - 2*window: t - window]
        w2 = series[t - window: t]
        try:
            tstat, pval = stats.ttest_ind(w1, w2, equal_var=False)
        except Exception:
            pval = 1.0
        var_ratio = (np.var(w2) + 1e-8) / (np.var(w1) + 1e-8)
        if pval < alpha or var_ratio > 2.0 or var_ratio < 0.5:
            cps.append(t-1)
    return cps
