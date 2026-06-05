"""
Volatility estimators with point-in-time correctness.

Implements EWMA, ATR, and Yang-Zhang volatility estimators.

DESIGN CONTRACT
---------------
All estimators return *per-bar* (non-annualized) volatility by default, on a
comparable fractional scale. This matters because the triple-barrier labeller
treats them as interchangeable (`TripleBarrierConfig.volatility_estimator`):
a barrier set at `k * volatility` must mean the same thing regardless of which
estimator is selected. Annualization, if desired, is opt-in and frequency-aware
via `annualize=True` + `periods_per_year`.

CAUSALITY
---------
Each estimate is lagged by `lag` bars (default 1) so the value at index t uses
information only through bar t-`lag`. The lag is positional; on a single,
gap-free series this equals a time lag. If your series can contain gaps, prefer
a time-based lag upstream (see alignment.MultiTimeframeAligner) or pre-fill.

Author: Research Team
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Convenience: bars per year for common crypto timeframes (24/7 markets).
PERIODS_PER_YEAR = {
    '1H': 24 * 365,
    '4H': 6 * 365,
    '1D': 365,
}


class VolatilityEstimator:
    """Base class for volatility estimators."""

    def __init__(self, name: str = "volatility", lag: int = 1):
        """
        Args:
            name: Name for the volatility series.
            lag: Bars to shift the result for causality (default 1).
        """
        self.name = name
        self.lag = lag

    def _finalize(
        self,
        vol: pd.Series,
        annualize: bool,
        periods_per_year: Optional[float],
    ) -> pd.Series:
        """Apply optional annualization and the causal lag. Shared by subclasses."""
        if annualize:
            if periods_per_year is None:
                raise ValueError(
                    "annualize=True requires periods_per_year (e.g. "
                    "PERIODS_PER_YEAR['1H'])."
                )
            vol = vol * np.sqrt(periods_per_year)
        if self.lag:
            vol = vol.shift(self.lag)
        return vol.rename(self.name)

    def estimate(self, df: pd.DataFrame) -> pd.Series:
        """Estimate volatility. Must be implemented by subclasses."""
        raise NotImplementedError


class EWMAVolatility(VolatilityEstimator):
    """
    Exponentially Weighted Moving Average volatility (RiskMetrics-style).

    Per-bar volatility = sqrt( EWMA(r_t^2) ), where r_t = log(C_t / C_{t-1}).
    A zero-mean assumption is used (standard for high-frequency returns), so the
    second moment of returns is taken as the variance.
    """

    def __init__(
        self,
        span: int = 12,
        min_periods: Optional[int] = None,
        name: str = "ewma_vol",
        lag: int = 1,
    ):
        super().__init__(name, lag)
        self.span = span
        self.min_periods = min_periods or max(span // 2, 1)

    def estimate(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        annualize: bool = False,
        periods_per_year: Optional[float] = None,
    ) -> pd.Series:
        """
        Returns per-bar EWMA volatility (annualized only if requested).
        """
        returns = np.log(df[price_col] / df[price_col].shift(1))

        var = returns.pow(2).ewm(
            span=self.span,
            min_periods=self.min_periods,
            adjust=False,
        ).mean()

        vol = np.sqrt(var)
        return self._finalize(vol, annualize, periods_per_year)


class ATRVolatility(VolatilityEstimator):
    """
    Average True Range volatility, expressed as a fraction of price (per-bar).

        TR_t  = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
        ATR_t = SMA or EWMA of TR over `period`
        out   = ATR_t / C_t   (a per-bar fractional range)
    """

    def __init__(
        self,
        period: int = 14,
        method: str = 'sma',  # 'sma' or 'ewma'
        name: str = "atr",
        lag: int = 1,
    ):
        super().__init__(name, lag)
        self.period = period
        if method not in ('sma', 'ewma'):
            raise ValueError(f"Unknown method: {method}")
        self.method = method

    def estimate(
        self,
        df: pd.DataFrame,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        annualize: bool = False,
        periods_per_year: Optional[float] = None,
    ) -> pd.Series:
        high = df[high_col]
        low = df[low_col]
        close = df[close_col]
        prev_close = close.shift(1)

        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        if self.method == 'sma':
            atr = tr.rolling(window=self.period, min_periods=1).mean()
        else:  # 'ewma'
            atr = tr.ewm(span=self.period, min_periods=1, adjust=False).mean()

        # Per-bar fractional volatility proxy.
        atr_pct = atr / close
        return self._finalize(atr_pct, annualize, periods_per_year)


class YangZhangVolatility(VolatilityEstimator):
    """
    Yang-Zhang volatility estimator (per-bar by default).

        sigma^2_YZ = sigma^2_overnight + k * sigma^2_open_close + (1-k) * sigma^2_RS

    where the overnight and open-close variances use the sample (n-1) estimator,
    the Rogers-Satchell term is a simple mean (n), and
        k = 0.34 / (1.34 + (n+1)/(n-1)).

    Reference:
        Yang, D., & Zhang, Q. (2000). Drift-independent volatility estimation
        based on high, low, open, and close prices.
    """

    def __init__(
        self,
        window: int = 20,
        min_periods: Optional[int] = None,
        name: str = "yang_zhang_vol",
        lag: int = 1,
    ):
        super().__init__(name, lag)
        self.window = window
        self.min_periods = min_periods or max(window // 2, 1)

    def estimate(
        self,
        df: pd.DataFrame,
        open_col: str = 'open',
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        annualize: bool = False,
        periods_per_year: Optional[float] = None,
    ) -> pd.Series:
        o = df[open_col]
        h = df[high_col]
        l = df[low_col]
        c = df[close_col]

        # Overnight (close-to-open) and open-to-close log returns.
        co = np.log(o / c.shift(1))
        oc = np.log(c / o)

        # Rogers-Satchell per-bar term (always >= 0 for valid OHLC).
        rs = np.log(h / c) * np.log(h / o) + np.log(l / c) * np.log(l / o)

        k = 0.34 / (1.34 + (self.window + 1) / (self.window - 1))

        var_co = co.rolling(window=self.window, min_periods=self.min_periods).var()
        var_oc = oc.rolling(window=self.window, min_periods=self.min_periods).var()
        var_rs = rs.rolling(window=self.window, min_periods=self.min_periods).mean()

        var_yz = var_co + k * var_oc + (1 - k) * var_rs

        # Clip tiny negatives from floating-point noise before sqrt.
        per_bar_vol = np.sqrt(var_yz.clip(lower=0))
        return self._finalize(per_bar_vol, annualize, periods_per_year)


def calculate_volatility_of_volatility(
    volatility: pd.Series,
    window: int = 20,
    lag: int = 0,
) -> pd.Series:
    """
    Volatility of volatility = rolling std of the volatility series' changes.

    IMPORTANT: pass the *raw* (un-lagged) volatility here and let the feature
    pipeline apply a single lag, OR pass an already-lagged series and keep
    lag=0 (the default). The previous version always shifted by 1, which
    double-lagged an already-shifted input.

    Args:
        volatility: A volatility series.
        window: Rolling window for the std.
        lag: Additional causal lag to apply (default 0; see note above).
    """
    vol_returns = volatility.pct_change().replace([np.inf, -np.inf], np.nan)
    vol_of_vol = vol_returns.rolling(window=window).std()
    if lag:
        vol_of_vol = vol_of_vol.shift(lag)
    return vol_of_vol


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dates = pd.date_range('2023-01-01', periods=1000, freq='1h')
    np.random.seed(42)

    price = 100 + np.random.randn(1000).cumsum()
    df = pd.DataFrame({
        'open': price + np.random.randn(1000) * 0.5,
        'high': price + np.random.rand(1000) * 2,
        'low': price - np.random.rand(1000) * 2,
        'close': price,
    }, index=dates)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    ewma_vol = EWMAVolatility(span=12).estimate(df)
    atr_vol = ATRVolatility(period=14).estimate(df)
    yz_vol = YangZhangVolatility(window=20).estimate(df)

    result = pd.DataFrame({
        'close': df['close'],
        'ewma_vol': ewma_vol,
        'atr_vol': atr_vol,
        'yang_zhang_vol': yz_vol,
    })

    print(result.tail(10))
    print("\nMedian volatility by estimator (should be the same order of magnitude):")
    print(result[['ewma_vol', 'atr_vol', 'yang_zhang_vol']].median())

    # Optional, explicit, frequency-aware annualization:
    yz_annual = YangZhangVolatility(window=20).estimate(
        df, annualize=True, periods_per_year=PERIODS_PER_YEAR['1H']
    )
    print(f"\nAnnualized YZ (opt-in) median: {yz_annual.median():.3f}")