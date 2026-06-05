"""
Multi-timeframe data alignment with strict, time-based causality.

A higher-timeframe bar covering the window [open, close) is only *knowable*
once it has closed. This module guarantees that a secondary-timeframe feature
is never visible to a primary-timeframe bar that opens before the secondary
bar could have closed.

Two correctness improvements over a naive ffill-then-shift(N) approach:

1. The lag is measured in TIME, not in row count, using `merge_asof`. This is
   robust to missing primary bars (exchange downtime, gaps). A positional
   `shift(24)` silently breaks the moment the 1H series has a gap.

2. The bar timestamp convention ('open' vs 'close') is explicit. It is the
   single assumption that decides whether the lag is causal, so it is a
   first-class parameter rather than a hidden assumption.

NOTE: the convention describes your *data*; it cannot be inferred from the data
alone. Set it to match your source (Binance klines use OPEN time).

Author: Research Team
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Literal
import logging

logger = logging.getLogger(__name__)

# Higher-timeframe columns that should never be lifted onto a lower timeframe.
# Calendar / cyclical / dummy columns are identical regardless of timeframe,
# and OHLCV is recomputed per timeframe elsewhere.
DEFAULT_ALIGN_EXCLUDE = {
    'open', 'high', 'low', 'close', 'volume',
    'coin', 'timeframe', 'data_quality_flag',
    'hour_of_day', 'day_of_week', 'week_of_month', 'month',
    'is_weekend',
}


class MultiTimeframeAligner:
    """
    Align data from multiple timeframes with strict, time-based causality.

    Key principle: a secondary bar timestamped for period P is only available
    to a primary bar whose timestamp is >= the *close* time of period P.
    """

    def __init__(
        self,
        bar_timestamp_convention: Literal['open', 'close'] = 'open',
        allow_exact_matches: bool = True,
    ):
        """
        Args:
            bar_timestamp_convention:
                'open'  -> a bar's timestamp marks the START of its window
                           (Binance klines). It closes one period later, so the
                           availability time is timestamp + period.
                'close' -> a bar's timestamp already marks the END of its
                           window, so it is available at its timestamp.
            allow_exact_matches:
                If True, a primary bar at time T may use a secondary bar that
                closed exactly at T (standard). Set False for an extra-cautious
                strictly-before-open rule.
        """
        if bar_timestamp_convention not in ('open', 'close'):
            raise ValueError(
                f"bar_timestamp_convention must be 'open' or 'close', "
                f"got {bar_timestamp_convention!r}"
            )
        self.bar_timestamp_convention = bar_timestamp_convention
        self.allow_exact_matches = allow_exact_matches

        # Timeframe size expressed in hours (used to build the availability lag).
        self.timeframe_hierarchy = {
            '1H': 1,
            '4H': 4,
            '1D': 24,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _period(self, timeframe: str) -> pd.Timedelta:
        """Return the bar duration for a timeframe as a Timedelta."""
        if timeframe not in self.timeframe_hierarchy:
            raise ValueError(f"Unknown timeframe: {timeframe}")
        return pd.Timedelta(hours=self.timeframe_hierarchy[timeframe])

    def _availability_index(
        self,
        index: pd.DatetimeIndex,
        secondary_tf: str,
    ) -> pd.DatetimeIndex:
        """
        Map secondary-bar timestamps to the time at which each bar becomes
        knowable, according to the configured convention.
        """
        if self.bar_timestamp_convention == 'open':
            # Bar opens at `index`, closes one period later -> knowable then.
            return index + self._period(secondary_tf)
        # 'close' convention: timestamp already marks the close.
        return index

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def align_timeframes(
        self,
        primary_df: pd.DataFrame,
        secondary_df: pd.DataFrame,
        primary_tf: str,
        secondary_tf: str,
        features: List[str],
        prefix: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Align secondary-timeframe features onto the primary timeframe causally.

        Args:
            primary_df: Primary timeframe data (e.g., 1H), DatetimeIndex.
            secondary_df: Secondary timeframe data (e.g., 1D), DatetimeIndex.
            primary_tf: Primary timeframe string (e.g., '1H').
            secondary_tf: Secondary timeframe string (e.g., '1D').
            features: Secondary feature columns to align.
            prefix: Column prefix for aligned features (default: '{secondary_tf}_').

        Returns:
            Copy of primary_df with the aligned, prefixed feature columns added.
        """
        if prefix is None:
            prefix = f"{secondary_tf}_"

        if not isinstance(primary_df.index, pd.DatetimeIndex):
            raise ValueError("primary_df must have a DatetimeIndex")
        if not isinstance(secondary_df.index, pd.DatetimeIndex):
            raise ValueError("secondary_df must have a DatetimeIndex")

        if self.timeframe_hierarchy[secondary_tf] <= self.timeframe_hierarchy[primary_tf]:
            raise ValueError(
                f"Cannot align {secondary_tf} to {primary_tf}: the secondary "
                "timeframe must be strictly higher than the primary."
            )

        valid_features = [f for f in features if f in secondary_df.columns]
        missing = set(features) - set(valid_features)
        for f in missing:
            logger.warning(f"Feature {f} not found in secondary_df; skipping")

        aligned_df = primary_df.copy()
        if not valid_features:
            return aligned_df

        # Build a secondary frame keyed by AVAILABILITY time (not bar time).
        sec = secondary_df[valid_features].copy()
        sec.index = self._availability_index(sec.index, secondary_tf)
        sec = sec.sort_index()
        # If two bars resolve to the same availability instant, keep the latest.
        sec = sec[~sec.index.duplicated(keep='last')]
        sec.columns = [f"{prefix}{c}" for c in valid_features]

        # As-of join in time: for each primary timestamp, take the most recent
        # secondary value whose availability time is <= that timestamp.
        primary_sorted_idx = primary_df.index.sort_values()
        left = pd.DataFrame({'_avail_key': primary_sorted_idx})
        right = sec.reset_index()
        right = right.rename(columns={right.columns[0]: '_avail_key'})

        merged = pd.merge_asof(
            left,
            right,
            on='_avail_key',
            direction='backward',
            allow_exact_matches=self.allow_exact_matches,
        )
        merged.index = primary_sorted_idx

        aligned_cols = [c for c in merged.columns if c != '_avail_key']
        for col in aligned_cols:
            # reindex back to the primary frame's original ordering
            aligned_df[col] = merged[col].reindex(aligned_df.index)

        logger.info(
            f"Aligned {len(valid_features)} features from {secondary_tf} to "
            f"{primary_tf} (convention={self.bar_timestamp_convention})"
        )
        return aligned_df

    def create_multi_timeframe_features(
        self,
        data_dict: Dict[str, pd.DataFrame],
        coin: str,
        primary_tf: str = '1H',
        feature_list: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Build a unified primary-timeframe frame with higher-timeframe features.

        Args:
            data_dict: Mapping 'COIN_TF' -> DataFrame.
            coin: Cryptocurrency symbol.
            primary_tf: Primary (output) timeframe.
            feature_list: Explicit allowlist of secondary features to lift.
                If None, falls back to all numeric columns minus
                DEFAULT_ALIGN_EXCLUDE (an explicit list is strongly preferred,
                so a warning is emitted).

        Returns:
            DataFrame indexed by the primary timeframe with aligned features.
        """
        primary_key = f"{coin}_{primary_tf}"
        if primary_key not in data_dict:
            raise ValueError(f"Primary data {primary_key} not found in data_dict")

        result_df = data_dict[primary_key].copy()

        higher_timeframes = [
            tf for tf in ['4H', '1D']
            if self.timeframe_hierarchy.get(tf, 0) > self.timeframe_hierarchy[primary_tf]
        ]

        for secondary_tf in higher_timeframes:
            secondary_key = f"{coin}_{secondary_tf}"
            if secondary_key not in data_dict:
                logger.warning(f"Secondary data {secondary_key} not found, skipping")
                continue

            secondary_df = data_dict[secondary_key]

            if feature_list is None:
                numeric = secondary_df.select_dtypes(include=[np.number]).columns
                features = [c for c in numeric if c not in DEFAULT_ALIGN_EXCLUDE]
                logger.warning(
                    "No explicit feature_list given; lifting %d numeric columns "
                    "from %s by default. An explicit allowlist is recommended to "
                    "avoid collinear duplicates.",
                    len(features), secondary_tf,
                )
            else:
                features = [f for f in feature_list if f in secondary_df.columns]

            result_df = self.align_timeframes(
                primary_df=result_df,
                secondary_df=secondary_df,
                primary_tf=primary_tf,
                secondary_tf=secondary_tf,
                features=features,
            )

        return result_df

    def validate_alignment(
        self,
        secondary_df: pd.DataFrame,
        aligned_df: pd.DataFrame,
        secondary_tf: str,
        feature: str,
        prefix: Optional[str] = None,
    ) -> Dict:
        """
        Independently verify that an aligned column is causal.

        This recomputes the expected aligned values with a separate, obviously
        correct reference implementation (numpy searchsorted over availability
        times) and compares them against the production `align_timeframes`
        output. It also performs an explicit look-ahead scan: for every
        non-null aligned value it confirms the source secondary bar had already
        closed at that primary timestamp.

        IMPORTANT: this checks internal causal consistency *given* the
        configured `bar_timestamp_convention`. It cannot verify that the
        convention itself matches your data source — that is a property of the
        feed, not of the numbers.

        Args:
            secondary_df: Original secondary-timeframe data (bar-time index).
            aligned_df: Output of `align_timeframes` (primary-time index).
            secondary_tf: Secondary timeframe string.
            feature: Original (unprefixed) secondary feature name.
            prefix: Prefix used during alignment (default '{secondary_tf}_').

        Returns:
            Report dict with `is_causal`, mismatch counts, and diagnostics.
        """
        if prefix is None:
            prefix = f"{secondary_tf}_"
        aligned_col = f"{prefix}{feature}"

        if feature not in secondary_df.columns:
            return {'is_causal': False, 'reason': f"{feature} missing in secondary_df"}
        if aligned_col not in aligned_df.columns:
            return {'is_causal': False, 'reason': f"{aligned_col} missing in aligned_df"}

        # Availability timestamps for each secondary bar.
        avail = self._availability_index(secondary_df.index, secondary_tf)
        order = np.argsort(avail.values)
        avail_sorted = avail.values[order]
        vals_sorted = secondary_df[feature].values[order]

        primary_idx = aligned_df.index.values
        side = 'right' if self.allow_exact_matches else 'left'
        # Position of the most recent secondary bar available at each primary t.
        pos = np.searchsorted(avail_sorted, primary_idx, side=side) - 1

        expected = np.full(len(primary_idx), np.nan, dtype=float)
        valid = pos >= 0
        expected[valid] = vals_sorted[pos[valid]].astype(float)

        actual = aligned_df[aligned_col].values.astype(float)

        # Compare expected vs actual (NaN-aware).
        both_nan = np.isnan(expected) & np.isnan(actual)
        close = np.isclose(expected, actual, equal_nan=False)
        match = both_nan | close
        n_mismatch = int((~match).sum())

        # Explicit look-ahead scan: the matched source bar must have closed
        # at-or-before (or strictly before) the primary timestamp. Evaluated
        # only on matched rows to avoid NaT dtype gymnastics.
        src_avail_valid = avail_sorted[pos[valid]]
        primary_valid = primary_idx[valid]
        if self.allow_exact_matches:
            n_lookahead = int((src_avail_valid > primary_valid).sum())
        else:
            n_lookahead = int((src_avail_valid >= primary_valid).sum())

        finite = np.isfinite(expected) & np.isfinite(actual)
        max_abs_diff = float(np.max(np.abs(expected[finite] - actual[finite]))) if finite.any() else 0.0

        is_causal = (n_mismatch == 0) and (n_lookahead == 0)
        if not is_causal:
            logger.error(
                "Alignment validation FAILED for %s: %d mismatches, %d look-ahead violations",
                aligned_col, n_mismatch, n_lookahead,
            )

        first_valid = aligned_df.index[np.argmax(valid)] if valid.any() else None
        return {
            'is_causal': is_causal,
            'n_mismatches': n_mismatch,
            'n_lookahead_violations': n_lookahead,
            'max_abs_diff': max_abs_diff,
            'n_aligned_values': int(valid.sum()),
            'first_aligned_timestamp': first_valid,
            'convention': self.bar_timestamp_convention,
        }


# Example usage / self-test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dates_1h = pd.date_range('2023-01-01', periods=240, freq='1h')
    dates_1d = pd.date_range('2023-01-01', periods=10, freq='1D')

    df_1h = pd.DataFrame({
        'close': np.random.randn(240).cumsum() + 100,
        'volume': np.random.randn(240) * 1000 + 10000,
    }, index=dates_1h)

    df_1d = pd.DataFrame({
        'close': np.random.randn(10).cumsum() + 100,
        'daily_feature': np.arange(10, dtype=float),  # 0..9, easy to eyeball
    }, index=dates_1d)

    aligner = MultiTimeframeAligner(bar_timestamp_convention='open')

    aligned = aligner.align_timeframes(
        primary_df=df_1h,
        secondary_df=df_1d,
        primary_tf='1H',
        secondary_tf='1D',
        features=['daily_feature'],
    )
    print(aligned[['close', '1D_daily_feature']].head(30))

    report = aligner.validate_alignment(
        secondary_df=df_1d,
        aligned_df=aligned,
        secondary_tf='1D',
        feature='daily_feature',
    )
    print("\nValidation report:", report)