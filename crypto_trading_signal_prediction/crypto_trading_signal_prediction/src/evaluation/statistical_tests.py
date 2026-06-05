"""
Statistical validation suite for trading strategies.

Rigorous tests to validate backtest results and guard against overfitting:
- Probabilistic & Deflated Sharpe Ratio (Bailey & López de Prado, 2014)
- Probability of Backtest Overfitting via CSCV (Bailey et al., 2017)
- Stationary-bootstrap confidence intervals
- Diebold-Mariano test
- McNemar test
- Multiple-testing correction

IMPORTANT UNITS NOTE: the (P)SR/DSR formulas operate on the PER-PERIOD Sharpe
ratio (mean/std of the raw return series), NOT an annualized one. Passing an
annualized Sharpe gives wrong results because the non-normality denominator is
nonlinear in SR. Use `from_returns(...)` to avoid the trap.

Author: Research Team
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging
from scipy import stats
from scipy.stats import norm
from dataclasses import dataclass
from itertools import combinations

logger = logging.getLogger(__name__)

_EULER_GAMMA = 0.5772156649015329


@dataclass
class StatisticalTestResult:
    test_name: str
    statistic: float
    p_value: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    interpretation: str
    details: Dict


def probabilistic_sharpe_ratio(
    sr: float, sr_benchmark: float, n: int, skew: float, kurt: float,
) -> Tuple[float, float]:
    """
    PSR = P(true SR > benchmark), using the per-period Sharpe `sr`.

    Args:
        sr: per-period observed Sharpe.
        sr_benchmark: threshold Sharpe to beat.
        n: number of return observations.
        skew: skewness of returns (gamma_3).
        kurt: NON-excess kurtosis of returns (gamma_4; normal = 3).
    Returns:
        (psr_probability, z_score)
    """
    denom = np.sqrt(max(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2, 1e-12))
    z = (sr - sr_benchmark) * np.sqrt(max(n - 1, 1)) / denom
    return float(norm.cdf(z)), float(z)


class DeflatedSharpeRatio:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014)."""

    def calculate(
        self,
        sr_per_period: float,
        n_trials: int,
        n_observations: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,          # NON-excess (normal = 3)
        sr_variance: Optional[float] = None,
    ) -> StatisticalTestResult:
        """
        Args:
            sr_per_period: PER-PERIOD observed Sharpe (not annualized).
            n_trials: number of independent strategy configurations tried.
            n_observations: number of returns.
            skewness, kurtosis: of the returns (kurtosis non-excess).
            sr_variance: variance of the per-period Sharpe ACROSS trials. If
                None, falls back to the SR estimator variance as a rough proxy.
        """
        if sr_variance is None:
            # SR estimator variance (non-normal), used as a proxy for the
            # cross-trial SR variance. Prefer passing the empirical variance of
            # SR across your Optuna trials.
            sr_variance = (1.0 - skewness * sr_per_period
                           + ((kurtosis - 1.0) / 4.0) * sr_per_period ** 2) / max(n_observations - 1, 1)
            logger.info("sr_variance not given; using SR-estimator variance proxy %.3e", sr_variance)

        sr0 = self._expected_max_sharpe(n_trials, sr_variance)
        dsr_prob, z = probabilistic_sharpe_ratio(
            sr_per_period, sr0, n_observations, skewness, kurtosis)
        p_value = 1.0 - dsr_prob   # prob the result is explained by selection

        interp = ("Significant after deflation (DSR > 0.95)" if dsr_prob > 0.95
                  else "NOT significant after deflation — likely selection bias")
        return StatisticalTestResult(
            "Deflated Sharpe Ratio", dsr_prob, p_value, None, interp,
            {'sr_per_period': sr_per_period, 'expected_max_sharpe': sr0,
             'z': z, 'n_trials': n_trials, 'n_observations': n_observations,
             'sr_variance': sr_variance},
        )

    def from_returns(self, returns: pd.Series, n_trials: int,
                     sr_variance: Optional[float] = None) -> StatisticalTestResult:
        """Compute DSR directly from a per-period return series (handles units)."""
        r = returns.dropna()
        sr = r.mean() / (r.std() + 1e-12)
        kurt_nonexcess = r.kurtosis() + 3.0   # pandas returns EXCESS kurtosis
        return self.calculate(sr, n_trials, len(r), r.skew(), kurt_nonexcess, sr_variance)

    @staticmethod
    def _expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
        """E[max SR] under the null (Gumbel approximation)."""
        if n_trials <= 1:
            return 0.0  # no deflation for a single trial -> DSR reduces to PSR(0)
        sigma = np.sqrt(max(sr_variance, 0.0))
        a = norm.ppf(1.0 - 1.0 / n_trials)
        b = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        return sigma * ((1.0 - _EULER_GAMMA) * a + _EULER_GAMMA * b)


class ProbabilityBacktestOverfitting:
    """
    PBO via Combinatorially Symmetric Cross-Validation (Bailey et al., 2017).

    Requires a MATRIX of candidate strategies' per-period returns (shape
    T x N): each column is one configuration you tried (e.g. one Optuna trial).
    PBO is the probability that the in-sample-best strategy lands below the
    out-of-sample median.
    """

    def __init__(self, n_splits: int = 16, n_combinations: Optional[int] = None,
                 metric: str = 'sharpe', random_state: int = 42):
        if n_splits % 2 != 0:
            raise ValueError("n_splits must be even for CSCV")
        self.n_splits = n_splits
        self.n_combinations = n_combinations
        self.metric = metric
        self.random_state = random_state

    def calculate(self, strategy_returns: pd.DataFrame) -> StatisticalTestResult:
        R = strategy_returns.to_numpy()
        T, N = R.shape
        if N < 2:
            raise ValueError("PBO needs >=2 candidate strategies (columns). "
                             "Feed the per-trial returns from your search.")

        # Even row groups.
        bounds = np.linspace(0, T, self.n_splits + 1).astype(int)
        groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(self.n_splits)]

        combos = list(combinations(range(self.n_splits), self.n_splits // 2))
        if self.n_combinations and len(combos) > self.n_combinations:
            rng = np.random.default_rng(self.random_state)
            combos = [combos[i] for i in rng.choice(len(combos), self.n_combinations, replace=False)]

        logits, is_best_oos = [], []
        for is_groups in combos:
            is_rows = np.concatenate([groups[g] for g in is_groups])
            oos_rows = np.concatenate([groups[g] for g in range(self.n_splits) if g not in is_groups])

            is_perf = self._metric(R[is_rows])
            oos_perf = self._metric(R[oos_rows])

            n_star = int(np.argmax(is_perf))       # best in-sample strategy
            # relative OOS rank of the IS-best (fraction of strategies it beats)
            rank = (np.sum(oos_perf <= oos_perf[n_star])) / (N + 1)
            rank = min(max(rank, 1e-6), 1 - 1e-6)
            logits.append(np.log(rank / (1 - rank)))
            is_best_oos.append(oos_perf[n_star])

        logits = np.array(logits)
        pbo = float(np.mean(logits <= 0))          # IS-best below OOS median
        interp = (f"High overfitting risk: PBO={pbo:.1%}" if pbo > 0.5
                  else f"Low overfitting risk: PBO={pbo:.1%}")
        return StatisticalTestResult(
            "Probability of Backtest Overfitting (CSCV)", pbo, None, None, interp,
            {'n_combinations': len(combos), 'n_strategies': N,
             'median_logit': float(np.median(logits)),
             'mean_oos_of_is_best': float(np.mean(is_best_oos))},
        )

    def _metric(self, block: np.ndarray) -> np.ndarray:
        """Per-strategy metric over a block of rows (returns one value/column)."""
        if self.metric == 'sharpe':
            mu = block.mean(axis=0)
            sd = block.std(axis=0) + 1e-12
            return mu / sd
        if self.metric == 'mean':
            return block.mean(axis=0)
        if self.metric == 'total':
            return np.prod(1 + block, axis=0) - 1
        raise ValueError(f"Unknown metric: {self.metric}")


class BootstrapConfidenceIntervals:
    """Stationary-bootstrap confidence intervals for a performance metric."""

    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95,
                 block_size: Optional[int] = None, random_state: int = 42):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.block_size = block_size
        self.random_state = random_state

    def calculate(self, returns: pd.Series, metric_func: Callable[[pd.Series], float],
                  metric_name: str = "metric") -> StatisticalTestResult:
        rng = np.random.default_rng(self.random_state)
        observed = metric_func(returns)
        vals = returns.to_numpy()
        n = len(vals)
        block = self.block_size or max(int(np.ceil(n ** (1 / 3))), 1)

        boot = np.empty(self.n_bootstrap)
        for b in range(self.n_bootstrap):
            sample, i = [], 0
            while i < n:
                start = rng.integers(0, n)
                length = rng.geometric(1 / block)
                for k in range(length):
                    if i >= n:
                        break
                    sample.append(vals[(start + k) % n]); i += 1
            boot[b] = metric_func(pd.Series(sample[:n]))

        alpha = 1 - self.confidence_level
        lo, hi = np.percentile(boot, [alpha / 2 * 100, (1 - alpha / 2) * 100])
        interp = f"{metric_name}={observed:.4f}, {self.confidence_level:.0%} CI [{lo:.4f}, {hi:.4f}]"
        return StatisticalTestResult(
            f"Bootstrap CI - {metric_name}", observed, None, (lo, hi), interp,
            {'n_bootstrap': self.n_bootstrap, 'std_error': float(boot.std())},
        )


class DieboldMarianoTest:
    """Diebold-Mariano test for equal forecast accuracy."""

    def __init__(self, loss_function: str = 'mse'):
        self.loss_function = loss_function

    def test(self, actual: pd.Series, forecast1: pd.Series, forecast2: pd.Series,
             horizon: int = 1) -> StatisticalTestResult:
        d = (self._loss(actual, forecast1) - self._loss(actual, forecast2)).dropna().to_numpy()
        mean_diff = d.mean()
        var = self._newey_west_variance(d, horizon)
        dm = mean_diff / np.sqrt(var / len(d) + 1e-18)
        p = 2 * (1 - norm.cdf(abs(dm)))
        if p < 0.05:
            interp = ("Model 1 significantly more accurate" if mean_diff < 0
                      else "Model 2 significantly more accurate")
        else:
            interp = "No significant difference in accuracy"
        return StatisticalTestResult("Diebold-Mariano Test", float(dm), float(p), None, interp,
                                     {'mean_loss_diff': float(mean_diff), 'horizon': horizon})

    def _loss(self, actual, forecast):
        e = actual - forecast
        if self.loss_function == 'mse':
            return e ** 2
        if self.loss_function == 'mae':
            return e.abs()
        if self.loss_function == 'qlike':
            return (actual / forecast) - np.log(actual / forecast) - 1
        raise ValueError(f"Unknown loss function: {self.loss_function}")

    @staticmethod
    def _newey_west_variance(series: np.ndarray, horizon: int) -> float:
        # Operate on numpy arrays positionally — no pandas index alignment.
        n = len(series)
        x = series - series.mean()
        var = np.sum(x ** 2)
        for lag in range(1, horizon):
            w = 1 - lag / (horizon + 1)
            autocov = np.sum(x[:-lag] * x[lag:])    # positional, correct
            var += 2 * w * autocov
        return var / n


class McNemarTest:
    def test(self, y_true, y_pred1, y_pred2) -> StatisticalTestResult:
        c1 = (y_true == y_pred1).astype(int)
        c2 = (y_true == y_pred2).astype(int)
        n01 = int(((c1 == 0) & (c2 == 1)).sum())
        n10 = int(((c1 == 1) & (c2 == 0)).sum())
        if n01 + n10 == 0:
            stat, p = 0.0, 1.0
        else:
            stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            p = 1 - stats.chi2.cdf(stat, df=1)
        if p < 0.05:
            interp = "Model 1 significantly better" if n10 > n01 else "Model 2 significantly better"
        else:
            interp = "No significant difference between models"
        return StatisticalTestResult("McNemar Test", float(stat), float(p), None, interp,
                                     {'n01': n01, 'n10': n10,
                                      'accuracy1': float(c1.mean()), 'accuracy2': float(c2.mean())})


class MultipleTestingCorrection:
    def __init__(self, method: str = 'bonferroni'):
        self.method = method

    def correct(self, p_values: List[float], alpha: float = 0.05) -> Dict:
        p = np.asarray(p_values, dtype=float)
        n = len(p)
        order = np.argsort(p)

        if self.method == 'bonferroni':
            corrected = np.clip(p * n, 0, 1)
            reject = corrected < alpha
        elif self.method == 'holm':
            corrected = np.empty(n)
            running = 0.0
            for rank, idx in enumerate(order):
                running = max(running, min(1.0, p[idx] * (n - rank)))  # enforce monotone
                corrected[idx] = running
            reject = corrected < alpha
        elif self.method in ('bh', 'by'):
            c = np.sum(1 / np.arange(1, n + 1)) if self.method == 'by' else 1.0
            corrected = np.empty(n)
            prev = 1.0
            for rank in range(n - 1, -1, -1):
                idx = order[rank]
                val = min(1.0, p[idx] * n * c / (rank + 1))
                prev = min(prev, val)
                corrected[idx] = prev
            reject = corrected < alpha
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return {'method': self.method, 'original_p_values': p.tolist(),
                'corrected_p_values': corrected.tolist(), 'reject': reject.tolist(),
                'n_rejected': int(reject.sum()), 'n_total': n}


class ComprehensiveValidator:
    """Runs the applicable tests on a strategy (and its candidate matrix)."""

    def validate_strategy(
        self,
        returns: pd.Series,
        n_trials: int = 1,
        strategy_matrix: Optional[pd.DataFrame] = None,
        sr_variance: Optional[float] = None,
    ) -> Dict:
        out: Dict[str, StatisticalTestResult] = {}

        # Deflated Sharpe (per-period, units handled by from_returns).
        out['deflated_sharpe'] = DeflatedSharpeRatio().from_returns(
            returns, n_trials=max(n_trials, 1), sr_variance=sr_variance)

        # Bootstrap CIs (annualized for human readability is fine here).
        bs = BootstrapConfidenceIntervals(n_bootstrap=1000)
        out['sharpe_ci'] = bs.calculate(
            returns, lambda r: r.mean() / (r.std() + 1e-12) * np.sqrt(252), "Annualized Sharpe")
        out['return_ci'] = bs.calculate(
            returns, lambda r: (1 + r).prod() - 1, "Total Return")

        # PBO needs the candidate matrix — only run if provided.
        if strategy_matrix is not None and strategy_matrix.shape[1] >= 2:
            try:
                ns = min(16, (strategy_matrix.shape[0] // 50) * 2 or 2)
                out['pbo'] = ProbabilityBacktestOverfitting(n_splits=ns).calculate(strategy_matrix)
            except Exception as e:
                logger.warning("PBO skipped: %s", e)
        else:
            logger.info("PBO skipped: pass strategy_matrix (one column per trial) to compute it.")

        return {'test_results': out, 'summary': self._summary(out)}

    @staticmethod
    def _summary(results: Dict) -> str:
        lines = ["=" * 70, "STATISTICAL VALIDATION SUMMARY", "=" * 70, ""]
        for r in results.values():
            lines.append(f"{r.test_name}:")
            lines.append(f"  statistic: {r.statistic:.4f}")
            if r.p_value is not None:
                lines.append(f"  p-value:   {r.p_value:.4f}")
            if r.confidence_interval is not None:
                lines.append(f"  95% CI:    [{r.confidence_interval[0]:.4f}, {r.confidence_interval[1]:.4f}]")
            lines.append(f"  -> {r.interpretation}\n")
        return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    rets = pd.Series(np.random.randn(1000) * 0.02 + 0.001)
    dsr = DeflatedSharpeRatio().from_returns(rets, n_trials=100)
    print(f"DSR prob={dsr.statistic:.3f} -> {dsr.interpretation}")