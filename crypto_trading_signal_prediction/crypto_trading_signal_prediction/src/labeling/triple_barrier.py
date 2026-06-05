"""
Triple Barrier Method Implementation.

Implements López de Prado's triple barrier method with:
- Vectorized / numba-accelerated computation (numba optional)
- Multiple volatility estimators (consistent per-bar scale)
- Correct right-edge handling (events with truncated horizons are not labelled)
- Configurable vertical-barrier label semantics
- Optional intrabar (high/low) barrier touches

Author: Research Team
Date: 2024

Reference:
    Advances in Financial Machine Learning, Chapter 3 — Marcos López de Prado
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, Literal, Union
import logging

from src.features.volatility import EWMAVolatility, ATRVolatility, YangZhangVolatility

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# numba is optional: fall back to a no-op decorator (pure Python) if absent.
# ---------------------------------------------------------------------------
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def _decorator(func):
            return func
        return _decorator


@njit(cache=True)
def _compute_barrier_touches(
    prices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    upper_barriers: np.ndarray,
    lower_barriers: np.ndarray,
    holding_period: int,
    min_return_threshold: float,
    use_intrabar: bool,
    vertical_mode: int,   # 0 = sign-of-return, 1 = always zero
):
    """
    Core barrier-touch scan.

    Returns (labels, valid, event_end_indices, barrier_types, returns).

    `valid[i] == 0` marks an event whose horizon was truncated by the end of
    the data AND which did not touch a price barrier within the available bars;
    such labels are unreliable and must be dropped before training.
    """
    n = len(prices)
    labels = np.zeros(n, dtype=np.int8)
    valid = np.ones(n, dtype=np.int8)
    event_end_indices = np.arange(n, dtype=np.int64)
    barrier_types = np.zeros(n, dtype=np.int8)   # 1=upper, -1=lower, 0=vertical
    returns = np.zeros(n, dtype=np.float64)

    for i in range(n):
        # Barriers undefined (e.g. volatility warm-up) -> cannot label.
        if np.isnan(upper_barriers[i]) or np.isnan(lower_barriers[i]):
            valid[i] = 0
            event_end_indices[i] = i
            continue

        entry_price = prices[i]
        upper = upper_barriers[i]
        lower = lower_barriers[i]

        # Vertical barrier (time limit) and whether the full horizon exists.
        vertical_idx = i + holding_period
        truncated = vertical_idx > (n - 1)
        last_j = vertical_idx if not truncated else (n - 1)

        hit_upper = False
        hit_lower = False
        hit_idx = -1

        for j in range(i + 1, last_j + 1):
            if use_intrabar:
                up_touch = highs[j] >= upper
                low_touch = lows[j] <= lower
            else:
                up_touch = prices[j] >= upper
                low_touch = prices[j] <= lower

            # Same-bar ambiguity: assume the adverse (lower) barrier first.
            if low_touch and up_touch:
                hit_lower = True
                hit_idx = j
                break
            if up_touch:
                hit_upper = True
                hit_idx = j
                break
            if low_touch:
                hit_lower = True
                hit_idx = j
                break

        if hit_upper or hit_lower:
            # A price barrier was hit within available data -> always valid.
            end_idx = hit_idx
            exit_price = prices[end_idx]
            ret = (exit_price - entry_price) / entry_price
            event_end_indices[i] = end_idx
            returns[i] = ret
            if abs(ret) < min_return_threshold:
                labels[i] = 0
                barrier_types[i] = 0
            elif hit_upper:
                labels[i] = 1
                barrier_types[i] = 1
            else:
                labels[i] = -1
                barrier_types[i] = -1
            continue

        # No price barrier hit within the available window.
        if truncated:
            # Horizon ran past the end of the data -> unreliable, drop it.
            valid[i] = 0
            event_end_indices[i] = n - 1
            returns[i] = (prices[n - 1] - entry_price) / entry_price
            continue

        # Genuine vertical-barrier event (full horizon observed).
        end_idx = vertical_idx
        exit_price = prices[end_idx]
        ret = (exit_price - entry_price) / entry_price
        event_end_indices[i] = end_idx
        returns[i] = ret
        barrier_types[i] = 0
        if vertical_mode == 1:
            labels[i] = 0
        else:  # sign of return
            if ret > min_return_threshold:
                labels[i] = 1
            elif ret < -min_return_threshold:
                labels[i] = -1
            else:
                labels[i] = 0

    return labels, valid, event_end_indices, barrier_types, returns


class TripleBarrierLabeler:
    """
    Triple Barrier Method for event-based labeling.

    Leakage profile: the *features/volatility* used to size barriers are lagged
    (see volatility.py), while the barrier scan looks strictly forward to
    determine the outcome (labels are allowed to use the future). Events whose
    forward horizon extends past the end of the data are flagged invalid.
    """

    def __init__(
        self,
        upper_barrier_multiplier: float = 2.0,
        lower_barrier_multiplier: float = 2.0,
        holding_period: int = 24,
        volatility_estimator: Literal['ewma', 'atr', 'yang_zhang'] = 'atr',
        volatility_window: int = 14,
        min_return_threshold: float = 0.001,
        symmetric_barriers: bool = True,
        use_intrabar_touches: bool = False,
        vertical_label: Literal['sign', 'zero'] = 'sign',
    ):
        """
        Args:
            upper_barrier_multiplier: Upper barrier width (in volatility units).
            lower_barrier_multiplier: Lower barrier width (in volatility units).
            holding_period: Maximum holding period in bars (vertical barrier).
            volatility_estimator: 'ewma' | 'atr' | 'yang_zhang' (now on a
                consistent per-bar scale, so this is genuinely interchangeable).
            volatility_window: Window for the volatility estimator.
            min_return_threshold: |return| below this is labelled neutral (0).
            symmetric_barriers: Force lower multiplier == upper multiplier.
            use_intrabar_touches: If True, detect touches with bar high/low
                instead of close. More realistic (you can be stopped intrabar),
                but introduces same-bar ambiguity (resolved pessimistically:
                the lower barrier is assumed hit first).
            vertical_label: What to label a genuine time-limit (vertical) event.
                'sign' -> sign of the holding-period return (binary-ish problem;
                          set num_class=2 downstream). This matches the shipped
                          behaviour.
                'zero' -> a distinct neutral class (true 3-class problem,
                          matching num_class=3 in model_config).
        """
        self.upper_barrier_mult = upper_barrier_multiplier
        self.lower_barrier_mult = lower_barrier_multiplier
        self.holding_period = holding_period
        self.volatility_estimator_name = volatility_estimator
        self.volatility_window = volatility_window
        self.min_return_threshold = min_return_threshold
        self.symmetric_barriers = symmetric_barriers
        self.use_intrabar_touches = use_intrabar_touches
        self.vertical_label = vertical_label

        if symmetric_barriers:
            self.lower_barrier_mult = self.upper_barrier_mult

        self.vol_estimator = self._get_volatility_estimator()

        if not _HAVE_NUMBA:
            logger.warning("numba not available; using pure-Python barrier scan "
                           "(slower but identical results).")
        logger.info(
            "Initialized TripleBarrierLabeler: barriers=[%s, %s], holding=%d, "
            "vol=%s, intrabar=%s, vertical_label=%s",
            self.upper_barrier_mult, self.lower_barrier_mult, self.holding_period,
            self.volatility_estimator_name, self.use_intrabar_touches,
            self.vertical_label,
        )

    def _get_volatility_estimator(self):
        if self.volatility_estimator_name == 'ewma':
            return EWMAVolatility(span=self.volatility_window)
        elif self.volatility_estimator_name == 'atr':
            return ATRVolatility(period=self.volatility_window)
        elif self.volatility_estimator_name == 'yang_zhang':
            return YangZhangVolatility(window=self.volatility_window)
        raise ValueError(f"Unknown volatility estimator: {self.volatility_estimator_name}")

    def generate_labels(
        self,
        df: pd.DataFrame,
        price_col: str = 'close',
        return_events: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate triple-barrier labels.

        Returns a labels DataFrame (label is NaN for invalid/truncated events),
        optionally with a detailed events DataFrame.
        """
        logger.info(f"Generating triple barrier labels for {len(df)} samples")

        volatility = self.vol_estimator.estimate(df)
        barriers = self._calculate_barriers(df, price_col, volatility)
        labels, events = self._find_barrier_touches(df, barriers, price_col, volatility)

        result_df = pd.DataFrame({
            'label': labels,                       # float; NaN where invalid
            'volatility': volatility,
            'upper_barrier': barriers['upper'],
            'lower_barrier': barriers['lower'],
        }, index=df.index)

        valid = events['is_valid'].astype(bool)
        vc = labels[valid].value_counts()
        logger.info(
            "Valid labels=%d / %d | Up=%d, Down=%d, Neutral=%d",
            int(valid.sum()), len(df),
            int(vc.get(1.0, 0)), int(vc.get(-1.0, 0)), int(vc.get(0.0, 0)),
        )

        return (result_df, events) if return_events else result_df

    def _calculate_barriers(
        self,
        df: pd.DataFrame,
        price_col: str,
        volatility: pd.Series,
    ) -> Dict[str, pd.Series]:
        price = df[price_col]
        upper_barrier = price * (1 + self.upper_barrier_mult * volatility)
        lower_barrier = price * (1 - self.lower_barrier_mult * volatility)
        return {'upper': upper_barrier, 'lower': lower_barrier}

    def _find_barrier_touches(
        self,
        df: pd.DataFrame,
        barriers: Dict[str, pd.Series],
        price_col: str,
        volatility: pd.Series,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        prices = df[price_col].to_numpy(dtype=np.float64)
        highs = df['high'].to_numpy(dtype=np.float64) if 'high' in df else prices
        lows = df['low'].to_numpy(dtype=np.float64) if 'low' in df else prices
        upper = barriers['upper'].to_numpy(dtype=np.float64)
        lower = barriers['lower'].to_numpy(dtype=np.float64)

        labels, valid, end_idx, btype, rets = _compute_barrier_touches(
            prices, highs, lows, upper, lower,
            int(self.holding_period),
            float(self.min_return_threshold),
            bool(self.use_intrabar_touches),
            1 if self.vertical_label == 'zero' else 0,
        )

        n = len(prices)
        # Float label with NaN for invalid events (so they can be dropped).
        label_f = labels.astype(np.float64)
        label_f[valid == 0] = np.nan

        events = pd.DataFrame({
            'label': label_f,
            'is_valid': valid.astype(bool),
            'event_end_idx': end_idx,
            'barrier_type': btype,
            'return': rets,
            'volatility': volatility.to_numpy(),
            'price_at_entry': prices,
            'price_at_exit': prices[end_idx],
            'upper_barrier': upper,
            'lower_barrier': lower,
        }, index=df.index)
        events['event_end_time'] = df.index[end_idx]
        events['holding_period'] = end_idx - np.arange(n)

        return pd.Series(label_f, index=df.index, name='label'), events

    def get_label_distribution(self, labels: pd.Series) -> pd.DataFrame:
        """Distribution of valid labels (NaN/invalid events are excluded)."""
        valid = labels.dropna()
        value_counts = valid.value_counts().sort_index()
        distribution = pd.DataFrame({
            'count': value_counts,
            'percentage': value_counts / len(valid) * 100 if len(valid) else 0,
        })
        distribution.index = distribution.index.map({
            -1.0: 'Down (Short)', 0.0: 'Neutral', 1.0: 'Up (Long)',
        })
        return distribution

    def analyze_barrier_effectiveness(self, events: pd.DataFrame) -> Dict:
        """Barrier hit rates / holding periods / returns over VALID events."""
        ev = events[events['is_valid']]
        total = len(ev) if len(ev) else 1
        bt = ev['barrier_type'].value_counts()

        def _mean(mask_val, col):
            sub = ev[ev['barrier_type'] == mask_val][col]
            return float(sub.mean()) if len(sub) else float('nan')

        return {
            'n_valid_events': int(len(ev)),
            'upper_barrier_hit_rate': bt.get(1, 0) / total,
            'lower_barrier_hit_rate': bt.get(-1, 0) / total,
            'vertical_barrier_hit_rate': bt.get(0, 0) / total,
            'avg_holding_period_upper': _mean(1, 'holding_period'),
            'avg_holding_period_lower': _mean(-1, 'holding_period'),
            'avg_holding_period_vertical': _mean(0, 'holding_period'),
            'avg_return_upper': _mean(1, 'return'),
            'avg_return_lower': _mean(-1, 'return'),
            'avg_return_vertical': _mean(0, 'return'),
            'avg_volatility': float(ev['volatility'].mean()) if 'volatility' in ev else None,
        }


class DynamicBarrierLabeler(TripleBarrierLabeler):
    """Triple barrier with volatility-regime-dependent barrier width."""

    def __init__(
        self,
        base_upper_multiplier: float = 2.0,
        base_lower_multiplier: float = 2.0,
        volatility_regime_window: int = 100,
        **kwargs,
    ):
        super().__init__(
            upper_barrier_multiplier=base_upper_multiplier,
            lower_barrier_multiplier=base_lower_multiplier,
            **kwargs,
        )
        self.base_upper_mult = base_upper_multiplier
        self.base_lower_mult = self.lower_barrier_mult
        self.vol_regime_window = volatility_regime_window

    def _calculate_barriers(self, df, price_col, volatility):
        price = df[price_col]
        # Regime z-score uses only past volatility (already-lagged input), and
        # rolling stats are causal, so this introduces no lookahead.
        vol_mean = volatility.rolling(self.vol_regime_window).mean()
        vol_std = volatility.rolling(self.vol_regime_window).std()
        vol_z = (volatility - vol_mean) / (vol_std + 1e-8)
        adjustment = 1 / (1 + vol_z.clip(-2, 2) * 0.2)

        upper = price * (1 + self.base_upper_mult * adjustment * volatility)
        lower = price * (1 - self.base_lower_mult * adjustment * volatility)
        return {'upper': upper, 'lower': lower}


def create_meta_labels(
    primary_model_predictions: pd.Series,
    triple_barrier_labels: pd.Series,
    side: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Meta-labels for bet sizing: 1 if the primary model's directional call was
    correct, else 0. NaN (invalid) triple-barrier labels propagate to NaN so
    they can be dropped.
    """
    p = primary_model_predictions
    y = triple_barrier_labels
    correct = (
        ((p == 1) & (y == 1)) |
        ((p == -1) & (y == -1)) |
        ((p == 0) & (y == 0))
    )
    meta = correct.astype('float')
    meta[y.isna()] = np.nan
    return meta


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1h')
    price = 100 + np.random.randn(1000).cumsum()
    df = pd.DataFrame({
        'open': price + np.random.randn(1000) * 0.5,
        'high': price + np.abs(np.random.randn(1000)) * 2,
        'low': price - np.abs(np.random.randn(1000)) * 2,
        'close': price,
    }, index=dates)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    labeler = TripleBarrierLabeler(
        upper_barrier_multiplier=2.0, holding_period=24,
        volatility_estimator='atr', volatility_window=14,
        vertical_label='sign',
    )
    labels_df, events_df = labeler.generate_labels(df, return_events=True)
    print("\nLabel distribution:\n", labeler.get_label_distribution(labels_df['label']))
    print("\nBarrier effectiveness:")
    for k, v in labeler.analyze_barrier_effectiveness(events_df).items():
        print(f"  {k}: {v}")