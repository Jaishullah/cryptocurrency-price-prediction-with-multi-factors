"""
Purged K-Fold Cross-Validation with Embargo.

Implements López de Prado's purged cross-validation (AFML Ch. 7). The label of
observation i spans [t0_i, t1_i] where t0_i is the bar timestamp (X.index) and
t1_i is the event end (`event_end_time`). A training observation leaks into a
test fold if its label interval overlaps the test fold's label interval — on
EITHER side of the test block. The previous implementation only kept training
samples whose label ended before the test started, which silently discarded the
entire post-test training set on every interior fold.

Author: Research Team
Date: 2024
Reference: Advances in Financial Machine Learning, Chapter 7
"""

import pandas as pd
import numpy as np
from typing import Iterator, Tuple, Optional, List
import logging
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)


def _contiguous_runs(sorted_idx: np.ndarray) -> List[Tuple[int, int]]:
    """Split a sorted index array into (start, end) inclusive contiguous runs."""
    if len(sorted_idx) == 0:
        return []
    runs = []
    run_start = sorted_idx[0]
    prev = sorted_idx[0]
    for v in sorted_idx[1:]:
        if v == prev + 1:
            prev = v
            continue
        runs.append((run_start, prev))
        run_start = v
        prev = v
    runs.append((run_start, prev))
    return runs


def _purge_and_embargo(
    t0: np.ndarray,            # bar timestamps (label start) for every obs
    t1: np.ndarray,            # event end times (label end) for every obs
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    embargo_bars: int,
    n: int,
) -> np.ndarray:
    """
    Drop training observations whose label interval overlaps any contiguous
    test block, plus an embargo buffer of `embargo_bars` immediately after each
    test block. Works for both single-block (KFold) and multi-block
    (combinatorial) test sets.
    """
    drop = np.zeros(n, dtype=bool)
    test_sorted = np.sort(test_indices)

    for run_start, run_end in _contiguous_runs(test_sorted):
        # Label envelope of this test block.
        block_t0 = t0[run_start:run_end + 1].min()
        block_t1 = t1[run_start:run_end + 1].max()

        # Overlap purge: train obs whose [t0, t1] intersects [block_t0, block_t1].
        overlap = (t0 <= block_t1) & (t1 >= block_t0)
        drop |= overlap

        # Embargo: positional bars immediately after this test block.
        if embargo_bars > 0:
            emb_lo = run_end + 1
            emb_hi = min(run_end + 1 + embargo_bars, n)
            if emb_lo < emb_hi:
                drop[emb_lo:emb_hi] = True

    # Never drop the test observations themselves from consideration; we only
    # filter the provided train_indices.
    keep_mask = ~drop[train_indices]
    return train_indices[keep_mask]


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold CV with embargo.

    Requires X to have a DatetimeIndex. If X has an `event_end_time` column the
    full overlap purge is applied; otherwise it degrades to embargo-only (with a
    warning), which is NOT leakage-safe for overlapping labels.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo_pct: float = 0.01,
        event_end_col: str = 'event_end_time',
        random_state: Optional[int] = None,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.event_end_col = event_end_col
        self.random_state = random_state

    def _resolve_t0_t1(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        t0 = X.index.values
        if self.event_end_col in X.columns:
            t1 = pd.to_datetime(X[self.event_end_col]).values
        else:
            logger.warning("No '%s' column; overlap purge disabled (embargo only). "
                           "This is NOT leakage-safe for overlapping labels.",
                           self.event_end_col)
            t1 = t0
        return t0, t1

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")

        n = len(X)
        indices = np.arange(n)
        t0, t1 = self._resolve_t0_t1(X)
        embargo_bars = int(n * self.embargo_pct)
        test_size = n // self.n_splits

        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n)
            if test_end - test_start < 2:
                continue

            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate([indices[:test_start], indices[test_end:]])

            kept = _purge_and_embargo(
                t0, t1, train_indices, test_indices, embargo_bars, n)

            logger.info("Fold %d/%d: train=%d (purged %d), test=%d",
                        i + 1, self.n_splits, len(kept),
                        len(train_indices) - len(kept), len(test_indices))
            yield kept, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged K-Fold (AFML 7.4) — now with real purging.

    Splits the data into `n_splits` contiguous groups and, for each combination
    of `n_test_groups` test groups, purges every training observation whose
    label overlaps any test block and embargoes after each block. The previous
    version applied a single embargo around the max test index and never purged,
    which is incorrect when the test groups are non-contiguous.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_test_groups: int = 2,
        embargo_pct: float = 0.01,
        event_end_col: str = 'event_end_time',
    ):
        from itertools import combinations
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.embargo_pct = embargo_pct
        self.event_end_col = event_end_col
        self.test_combinations = list(combinations(range(n_splits), n_test_groups))

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex")

        n = len(X)
        indices = np.arange(n)
        t0 = X.index.values
        t1 = (pd.to_datetime(X[self.event_end_col]).values
              if self.event_end_col in X.columns else t0)
        embargo_bars = int(n * self.embargo_pct)
        group_size = n // self.n_splits

        group_indices = [
            indices[g * group_size: min((g + 1) * group_size, n)]
            for g in range(self.n_splits)
        ]

        for i, test_groups in enumerate(self.test_combinations):
            test_indices = np.concatenate([group_indices[g] for g in test_groups])
            train_groups = [g for g in range(self.n_splits) if g not in test_groups]
            train_indices = np.concatenate([group_indices[g] for g in train_groups])

            kept = _purge_and_embargo(
                t0, t1, train_indices, test_indices, embargo_bars, n)

            logger.info("Combo %d/%d: train=%d (purged %d), test=%d",
                        i + 1, len(self.test_combinations), len(kept),
                        len(train_indices) - len(kept), len(test_indices))
            yield kept, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return len(self.test_combinations)


def visualize_cv_splits(cv_splitter, X: pd.DataFrame, title: str = "CV Splits"):
    """Visualize train/test/purged regions across folds."""
    import matplotlib.pyplot as plt
    splits = list(cv_splitter.split(X))
    fig, ax = plt.subplots(figsize=(15, 6))
    for i, (train_idx, test_idx) in enumerate(splits):
        all_idx = set(range(len(X)))
        train_set, test_set = set(train_idx), set(test_idx)
        purged = np.array(sorted(all_idx - train_set - test_set))
        ax.scatter(X.index[train_idx], [i] * len(train_idx), c='steelblue',
                   marker='|', s=80, label='Train' if i == 0 else '')
        if len(purged):
            ax.scatter(X.index[purged], [i] * len(purged), c='gold',
                       marker='|', s=80, label='Purged/Embargo' if i == 0 else '')
        ax.scatter(X.index[test_idx], [i] * len(test_idx), c='crimson',
                   marker='|', s=80, label='Test' if i == 0 else '')
    ax.set_xlabel('Time'); ax.set_ylabel('Fold'); ax.set_title(title)
    ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1h')
    holding = np.random.randint(1, 50, size=1000)
    X = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'event_end_time': dates + pd.to_timedelta(holding, unit='h'),
    }, index=dates)

    cv = PurgedKFold(n_splits=5, embargo_pct=0.02)
    for i, (tr, te) in enumerate(cv.split(X)):
        print(f"Fold {i+1}: train={len(tr)} test={len(te)} "
              f"train_after_test={(tr > te.max()).sum()}")