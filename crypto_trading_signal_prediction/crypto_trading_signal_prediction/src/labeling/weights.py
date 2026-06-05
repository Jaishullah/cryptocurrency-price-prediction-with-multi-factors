"""
Sample weighting based on uniqueness and return magnitude.

Implements López de Prado's sample weighting (AFML Chapter 4):
- Per-bar concurrency via an O(n + T) sweep (no O(n^2) iterrows loops)
- Average uniqueness as the mean of 1/concurrency over an event's lifespan
- Return-based weighting
- A genuine sequential bootstrap (draw probabilities updated after each pick)

All computation is done in integer bar-position space, which is both correct
and fast, and exactly matches the (start_index, end_index) that the triple
barrier labeller already produces.

Author: Research Team
Date: 2024
Reference: Advances in Financial Machine Learning, Chapter 4
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Core position-space primitives (correct + O(n + T))
# --------------------------------------------------------------------------- #
def concurrency_from_positions(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    n_bars: int,
) -> np.ndarray:
    """
    Number of events active at each bar position.

    For event i spanning [start_pos[i], end_pos[i]] (inclusive), add +1 at the
    start and -1 just past the end, then cumulative-sum. O(n + n_bars).
    """
    delta = np.zeros(n_bars + 1, dtype=np.float64)
    np.add.at(delta, start_pos, 1.0)
    np.add.at(delta, end_pos + 1, -1.0)
    return np.cumsum(delta)[:n_bars]


def average_uniqueness_from_positions(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    concurrency: np.ndarray,
) -> np.ndarray:
    """
    Average uniqueness of each event = mean over its lifespan of 1/concurrency.

    Computed with a prefix-sum over (1/concurrency) so each event is O(1).
    This is the true AFML metric (the previous numba path used 1/overlap_count,
    a different and incorrect quantity).
    """
    inv = 1.0 / np.maximum(concurrency, 1.0)
    prefix = np.concatenate(([0.0], np.cumsum(inv)))
    span_sum = prefix[end_pos + 1] - prefix[start_pos]
    span_len = (end_pos - start_pos + 1).astype(np.float64)
    return span_sum / span_len


class SampleWeightCalculator:
    """
    Calculate sample weights from average uniqueness and return magnitude.

    Preferred entry point is `calculate_weights`, which accepts either an events
    DataFrame keyed by timestamps (with a `bar_index`) or integer positions.
    """

    def __init__(
        self,
        use_return_weights: bool = True,
        return_weight_exponent: float = 1.0,
        min_weight: float = 0.01,
        max_weight: float = 10.0,
        normalize: bool = True,
    ):
        self.use_return_weights = use_return_weights
        self.return_weight_exponent = return_weight_exponent
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.normalize = normalize

    # ------------------------------------------------------------------ #
    # Position mapping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_positions(
        events: pd.DataFrame,
        bar_index: Optional[pd.DatetimeIndex],
        timestamp_col: str,
        event_end_col: str,
        start_pos_col: Optional[str],
        end_pos_col: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Resolve integer start/end positions and the number of bars."""
        # Fast path: positions already provided (e.g. from the labeller).
        if start_pos_col and end_pos_col and start_pos_col in events and end_pos_col in events:
            start_pos = events[start_pos_col].to_numpy(dtype=np.int64)
            end_pos = events[end_pos_col].to_numpy(dtype=np.int64)
            n_bars = int(end_pos.max()) + 1
            return start_pos, end_pos, n_bars

        # Timestamp path: map start/end times onto a bar grid.
        starts = (events.index.to_series() if events.index.name == timestamp_col
                  else events[timestamp_col])
        starts = pd.DatetimeIndex(starts)
        ends = pd.DatetimeIndex(events[event_end_col])

        if bar_index is None:
            # Infer a regular grid spanning all events. Recommended: pass the
            # actual price `bar_index` for data with gaps.
            all_ts = starts.append(ends).unique()
            all_ts = pd.DatetimeIndex(np.sort(all_ts))
            freq = pd.infer_freq(all_ts)
            if freq is None:
                diffs = np.diff(all_ts.values)
                step = pd.Timedelta(diffs.min()) if len(diffs) else pd.Timedelta('1h')
                bar_index = pd.date_range(all_ts[0], all_ts[-1], freq=step)
            else:
                bar_index = pd.date_range(all_ts[0], all_ts[-1], freq=freq)
            logger.warning("bar_index not supplied; inferred a regular grid of "
                           "%d bars. Pass the price index for gapped data.",
                           len(bar_index))

        start_pos = bar_index.get_indexer(starts, method='ffill')
        end_pos = bar_index.get_indexer(ends, method='ffill')
        if (start_pos < 0).any() or (end_pos < 0).any():
            raise ValueError("Some event timestamps fall outside bar_index.")
        return start_pos.astype(np.int64), end_pos.astype(np.int64), len(bar_index)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def calculate_weights(
        self,
        events: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        event_end_col: str = 'event_end_time',
        return_col: Optional[str] = 'return',
        bar_index: Optional[pd.DatetimeIndex] = None,
        start_pos_col: Optional[str] = None,
        end_pos_col: Optional[str] = 'event_end_idx',
    ) -> pd.Series:
        """
        Compute sample weights, returned on the events' own index.

        Fixes the previous index-alignment bug: uniqueness and the return
        weights are now combined positionally (by row), never by mismatched
        DatetimeIndex vs RangeIndex (which produced all-NaN weights).
        """
        logger.info(f"Calculating sample weights for {len(events)} events")

        # If the caller passes labeller output, start positions are the row
        # positions of the event index within bar_index; derive them when only
        # an end-position column exists.
        if start_pos_col is None and end_pos_col in (events.columns if events is not None else []):
            # Derive start positions from timestamps against bar_index below;
            # but if end positions are integer offsets into the same frame,
            # the natural start position is simply each row's own bar position.
            pass

        start_pos, end_pos, n_bars = self._resolve_positions(
            events, bar_index, timestamp_col, event_end_col,
            start_pos_col, end_pos_col if (end_pos_col in events.columns) else None,
        )

        conc = concurrency_from_positions(start_pos, end_pos, n_bars)
        uniqueness = average_uniqueness_from_positions(start_pos, end_pos, conc)
        weights = uniqueness.astype(np.float64)

        if self.use_return_weights and return_col is not None:
            if return_col not in events.columns:
                logger.warning("Return column %s not found; using uniqueness only.",
                               return_col)
            else:
                rw = self._return_weights(events[return_col].to_numpy())
                weights = weights * rw  # positional, same length, no index games

        weights = np.clip(weights, self.min_weight, self.max_weight)
        if self.normalize and weights.sum() > 0:
            weights = weights * len(weights) / weights.sum()

        out = pd.Series(weights, index=events.index, name='sample_weight')
        logger.info("Sample weights: min=%.4f max=%.4f mean=%.4f",
                    out.min(), out.max(), out.mean())
        return out

    def _return_weights(self, returns: np.ndarray) -> np.ndarray:
        abs_r = np.abs(np.nan_to_num(returns, nan=0.0)) ** self.return_weight_exponent
        return abs_r / (abs_r.mean() + 1e-8)

    # Backwards-compatible helpers (now O(n + T) and index-safe) -------- #
    def calculate_concurrency(
        self,
        events: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        event_end_col: str = 'event_end_time',
        bar_index: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        start_pos, end_pos, n_bars = self._resolve_positions(
            events, bar_index, timestamp_col, event_end_col, None, None)
        conc = concurrency_from_positions(start_pos, end_pos, n_bars)
        if bar_index is None:
            return pd.Series(conc, name='concurrency')
        return pd.Series(conc, index=bar_index[:n_bars], name='concurrency')

    def calculate_average_uniqueness(
        self,
        events: pd.DataFrame,
        concurrency: Optional[pd.Series] = None,
        timestamp_col: str = 'timestamp',
        event_end_col: str = 'event_end_time',
        bar_index: Optional[pd.DatetimeIndex] = None,
    ) -> pd.Series:
        start_pos, end_pos, n_bars = self._resolve_positions(
            events, bar_index, timestamp_col, event_end_col, None, None)
        conc = (concurrency.to_numpy() if concurrency is not None
                else concurrency_from_positions(start_pos, end_pos, n_bars))
        au = average_uniqueness_from_positions(start_pos, end_pos, conc)
        return pd.Series(au, index=events.index, name='avg_uniqueness')


class SequentialBootstrap:
    """
    Sequential bootstrap (AFML 4.5): draws are sampled with probability
    proportional to average uniqueness recomputed *given the already-drawn
    set*, so samples overlapping prior picks become progressively less likely.
    This is the property a plain bootstrap (and the previous implementation,
    which never updated the probabilities) violates.

    Operates in bar-position space for correctness and speed.
    """

    def __init__(self, sample_size: Optional[int] = None, random_state: int = 42):
        self.sample_size = sample_size
        self.random_state = random_state

    def generate_samples(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        n_bars: int,
        n_samples: int = 1,
    ) -> List[np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        size = self.sample_size or len(start_pos)
        return [self._single(start_pos, end_pos, n_bars, size, rng)
                for _ in range(n_samples)]

    @staticmethod
    def _single(start_pos, end_pos, n_bars, size, rng) -> np.ndarray:
        n = len(start_pos)
        # concurrency contributed by already-selected events (with replacement).
        sel_conc = np.zeros(n_bars, dtype=np.float64)
        selected = []

        span_len = (end_pos - start_pos + 1).astype(np.float64)
        for _ in range(size):
            # avg uniqueness of each candidate given current selection:
            # mean over its span of 1 / (1 + sel_conc[t]).
            inv = 1.0 / (1.0 + sel_conc)
            prefix = np.concatenate(([0.0], np.cumsum(inv)))
            avg_u = (prefix[end_pos + 1] - prefix[start_pos]) / span_len

            total = avg_u.sum()
            if total <= 0:
                break
            probs = avg_u / total
            pick = rng.choice(n, p=probs)        # with replacement (AFML)
            selected.append(pick)
            sel_conc[start_pos[pick]:end_pos[pick] + 1] += 1.0

        return np.array(selected, dtype=np.int64)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    n_events = 1000
    dates = pd.date_range('2023-01-01', periods=n_events, freq='1h')
    holding = np.random.randint(1, 50, size=n_events)
    events = pd.DataFrame({
        'timestamp': dates,
        'event_end_time': dates + pd.to_timedelta(holding, unit='h'),
        'holding_period': holding,
        'return': np.random.randn(n_events) * 0.02,
    })

    calc = SampleWeightCalculator(use_return_weights=True, normalize=True)
    weights = calc.calculate_weights(events, end_pos_col=None)  # timestamp path
    print(f"weights: non-NaN={weights.notna().sum()}/{len(weights)} "
          f"sum={weights.sum():.1f} (~{n_events}) mean={weights.mean():.3f}")

    conc = calc.calculate_concurrency(events)
    print(f"concurrency: min={conc.min():.0f} max={conc.max():.0f} mean={conc.mean():.2f}")