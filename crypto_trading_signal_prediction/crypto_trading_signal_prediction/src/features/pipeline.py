"""
Complete feature engineering pipeline with leakage prevention.

Orchestrates all feature calculations point-in-time correctly AND, crucially,
separates model-safe (lagged) features from the raw, contemporaneous columns
(OHLCV, raw market-structure) that must never be fed to the model. Use
`pipeline.select_features(df)` (or `pipeline.feature_columns_`) to get the
model-safe matrix; the full frame still carries raw `close` etc. for labelling
and backtesting.

Author: Research Team
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
from datetime import datetime

from src.features.volatility import (
    EWMAVolatility, ATRVolatility, YangZhangVolatility,
)
from config.feature_config import FEATURE_CONFIG

logger = logging.getLogger(__name__)

# Columns that are raw / contemporaneous and must be excluded from the model
# feature matrix (they reflect time-t information and would leak).
RAW_LEAKY_COLUMNS = {
    'open', 'high', 'low', 'close', 'volume',
    'quote_asset_volume', 'number_of_trades', 'taker_buy_ratio', 'funding_rate',
    'bitcoin_dominance', 'total_crypto_market_cap', 'altcoin_market_cap',
    'stablecoin_market_cap', 'stablecoin_supply_ratio', 'ssr',
    'fear_greed_value', 'fear_greed_label',
}


# --------------------------------------------------------------------------- #
# Supporting feature calculators (defined first so the pipeline can use them)
# --------------------------------------------------------------------------- #
class ReturnsFeatures:
    def __init__(self, config):
        self.config = config

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        close = df['close']
        if self.config.use_log_returns:
            for p in self.config.return_periods:
                result[f'log_return_{p}'] = np.log(close / close.shift(p)).shift(1)
        else:
            for p in self.config.return_periods:
                result[f'return_{p}'] = (close / close.shift(p) - 1).shift(1)
        for p in self.config.momentum_periods:
            result[f'momentum_{p}'] = (close / close.shift(p) - 1).shift(1)
        mom_col = f'momentum_{self.config.momentum_periods[0]}'
        for w in self.config.zscore_windows:
            if mom_col in result.columns:
                mean = result[mom_col].rolling(w).mean()
                std = result[mom_col].rolling(w).std()
                result[f'momentum_zscore_{w}'] = ((result[mom_col] - mean) / (std + 1e-8)).shift(1)
        return result


class VolumeFeatures:
    def __init__(self, config):
        self.config = config

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        volume = df['volume']
        for w in self.config.volume_zscore_windows:
            mean = volume.rolling(w).mean(); std = volume.rolling(w).std()
            result[f'volume_zscore_{w}'] = ((volume - mean) / (std + 1e-8)).shift(1)
        for p in self.config.volume_momentum_periods:
            result[f'volume_momentum_{p}'] = (volume / volume.shift(p) - 1).shift(1)
        for p in self.config.vwap_periods:
            vwap = ((df['close'] * df['volume']).rolling(p).sum()
                    / df['volume'].rolling(p).sum())
            result[f'vwap_distance_{p}'] = ((df['close'] - vwap) / vwap).shift(1)
        if self.config.calculate_dollar_volume:
            dv = (df['close'] * df['volume'])
            result['dollar_volume'] = dv.shift(1)
            for w in self.config.volume_zscore_windows:
                mean = dv.rolling(w).mean(); std = dv.rolling(w).std()
                result[f'dollar_volume_zscore_{w}'] = ((dv - mean) / (std + 1e-8)).shift(1)
        return result


class TechnicalIndicators:
    def __init__(self, config):
        self.config = config

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for p in self.config.rsi_periods:
            result[f'rsi_{p}'] = self._rsi(df['close'], p).shift(1)
        macd, signal, hist = self._macd(df['close'], self.config.macd_fast,
                                         self.config.macd_slow, self.config.macd_signal)
        result['macd'] = macd.shift(1); result['macd_signal'] = signal.shift(1)
        result['macd_hist'] = hist.shift(1)
        up, mid, low = self._bbands(df['close'], self.config.bb_period, self.config.bb_std)
        result['bb_width'] = ((up - low) / mid).shift(1)
        result['bb_position'] = ((df['close'] - low) / (up - low + 1e-8)).shift(1)
        k, d = self._stoch(df['high'], df['low'], df['close'],
                           self.config.stoch_k_period, self.config.stoch_d_period)
        result['stoch_k'] = k.shift(1); result['stoch_d'] = d.shift(1)
        for p in self.config.ma_periods:
            sma = df['close'].rolling(p).mean()
            result[f'price_to_sma_{p}'] = (df['close'] / sma - 1).shift(1)
            ema = df['close'].ewm(span=p, adjust=False).mean()
            result[f'price_to_ema_{p}'] = (df['close'] / ema - 1).shift(1)
        return result

    @staticmethod
    def _rsi(close, period):
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        return 100 - (100 / (1 + gain / (loss + 1e-8)))

    @staticmethod
    def _macd(close, fast, slow, signal):
        ema_f = close.ewm(span=fast, adjust=False).mean()
        ema_s = close.ewm(span=slow, adjust=False).mean()
        macd = ema_f - ema_s
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig, macd - sig

    @staticmethod
    def _bbands(close, period, std_dev):
        mid = close.rolling(period).mean(); std = close.rolling(period).std()
        return mid + std_dev * std, mid, mid - std_dev * std

    @staticmethod
    def _stoch(high, low, close, k_period, d_period):
        ll = low.rolling(k_period).min(); hh = high.rolling(k_period).max()
        k = 100 * (close - ll) / (hh - ll + 1e-8)
        return k, k.rolling(d_period).mean()


class MarketStructureFeatures:
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if 'bitcoin_dominance' in df:
            result['btc_dominance'] = df['bitcoin_dominance'].shift(1)
            result['btc_dominance_change'] = df['bitcoin_dominance'].diff().shift(1)
        if 'total_crypto_market_cap' in df:
            result['total_mcap'] = df['total_crypto_market_cap'].shift(1)
            result['total_mcap_growth'] = df['total_crypto_market_cap'].pct_change().shift(1)
        if 'stablecoin_supply_ratio' in df:
            result['ssr_feat'] = df['stablecoin_supply_ratio'].shift(1)
            result['ssr_change'] = df['stablecoin_supply_ratio'].diff().shift(1)
        if 'fear_greed_value' in df:
            result['fear_greed'] = df['fear_greed_value'].shift(1)
            result['fear_greed_change'] = df['fear_greed_value'].diff().shift(1)
        return result


class TimeFeatures:
    """Calendar features. These are known at the bar open, so they are NOT
    lagged (no leakage) and are explicitly whitelisted as model-safe."""

    SAFE_COLUMNS = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
                    'month_sin', 'month_cos', 'is_weekend', 'week_of_month']

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            ts = df.index
        elif 'timestamp' in df.columns:
            ts = pd.DatetimeIndex(pd.to_datetime(df['timestamp']))
        else:
            return result
        result['hour_sin'] = np.sin(2 * np.pi * ts.hour / 24)
        result['hour_cos'] = np.cos(2 * np.pi * ts.hour / 24)
        result['day_of_week_sin'] = np.sin(2 * np.pi * ts.dayofweek / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * ts.dayofweek / 7)
        result['month_sin'] = np.sin(2 * np.pi * ts.month / 12)
        result['month_cos'] = np.cos(2 * np.pi * ts.month / 12)
        result['is_weekend'] = (ts.dayofweek >= 5).astype(int)
        result['week_of_month'] = (ts.day - 1) // 7 + 1
        return result


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class FeaturePipeline:
    def __init__(self, config=FEATURE_CONFIG, cache_dir: Optional[Path] = None,
                 enable_cache: bool = True):
        self.config = config
        self.cache_dir = Path(cache_dir) if cache_dir else Path('data/features')
        self.enable_cache = enable_cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_calculators()
        self.feature_metadata: Dict = {}
        self.feature_columns_: List[str] = []   # model-safe columns

    def _init_calculators(self):
        self.vol_ewma = [EWMAVolatility(span=s) for s in self.config.volatility.ewma_spans]
        self.vol_atr = [ATRVolatility(period=p) for p in self.config.volatility.atr_periods]
        self.vol_yz = YangZhangVolatility(window=self.config.volatility.yang_zhang_window)
        self.returns_calculator = ReturnsFeatures(self.config.returns)
        self.volume_calculator = VolumeFeatures(self.config.volume)
        self.technical_calculator = TechnicalIndicators(self.config.technical)
        self.market_structure_calculator = MarketStructureFeatures()
        self.time_features_calculator = TimeFeatures()

    def generate_features(self, df: pd.DataFrame, coin: str, timeframe: str,
                          include_groups: Optional[List[str]] = None) -> pd.DataFrame:
        logger.info("Generating features for %s %s", coin, timeframe)
        if self.enable_cache:
            cached = self._load_cache(coin, timeframe)
            if cached is not None:
                self.feature_columns_ = [c for c in cached.columns
                                         if c not in df.columns and c != 'coin_id'] + ['coin_id']
                return cached

        original_cols = set(df.columns)
        result = df.copy()
        groups = {
            'volatility': self._add_volatility,
            'returns': lambda d: self.returns_calculator.calculate(d),
            'volume': lambda d: self.volume_calculator.calculate(d),
            'technical': lambda d: self.technical_calculator.calculate(d),
            'market_structure': lambda d: self.market_structure_calculator.calculate(d),
            'time': lambda d: self.time_features_calculator.calculate(d),
        }
        for name in (include_groups or groups.keys()):
            if name in groups:
                logger.info("  + %s", name)
                result = groups[name](result)

        result['coin_id'] = pd.Categorical([coin] * len(result))

        # Model-safe feature set = everything we ADDED, minus any raw passthrough.
        added = [c for c in result.columns if c not in original_cols]
        self.feature_columns_ = [c for c in added if c not in RAW_LEAKY_COLUMNS]

        self._validate_no_leakage(result, original_cols)
        if self.enable_cache:
            self._save_cache(result, coin, timeframe)
        self._update_metadata(coin, timeframe, self.feature_columns_)
        logger.info("Generated %d model-safe features (frame has %d cols total)",
                    len(self.feature_columns_), result.shape[1])
        return result

    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the model-safe (lagged/calendar) feature columns."""
        if not self.feature_columns_:
            raise ValueError("Call generate_features first.")
        cols = [c for c in self.feature_columns_ if c in df.columns]
        return df[cols]

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for calc in self.vol_ewma:
            result[f'vol_ewma_{calc.span}'] = calc.estimate(df)
        for calc in self.vol_atr:
            result[f'vol_atr_{calc.period}'] = calc.estimate(df)
        result['vol_yang_zhang'] = self.vol_yz.estimate(df)
        # vol-of-vol: the vol columns are ALREADY lagged by the estimator, so we
        # take a rolling std WITHOUT an extra shift (the original double-shifted).
        vol_cols = [c for c in result.columns if c.startswith('vol_')]
        win = self.config.volatility.vol_of_vol_window
        for col in vol_cols:
            result[f'{col}_of_vol'] = result[col].rolling(win).std()
        return result

    def _validate_no_leakage(self, features_df: pd.DataFrame, original_cols: set):
        safe_at_start = set(TimeFeatures.SAFE_COLUMNS) | {'coin_id'}
        for col in self.feature_columns_:
            if col in safe_at_start:
                continue
            fv = features_df[col].first_valid_index()
            if fv is not None and fv == features_df.index[0]:
                logger.warning("Feature %s is valid at the first timestamp — "
                               "check for leakage (lagged features should start NaN).", col)
        if 'close' in features_df.columns:
            fut = features_df['close'].pct_change().shift(-1)
            for col in self.feature_columns_[:15]:
                if pd.api.types.is_float_dtype(features_df[col]):
                    c = features_df[col].corr(fut)
                    if pd.notna(c) and abs(c) > 0.95:
                        logger.warning("Feature %s corr %.3f with next-bar return — "
                                       "possible leakage!", col, c)

    def _load_cache(self, coin, timeframe):
        f = self.cache_dir / f"{coin}_{timeframe}_features.parquet"
        if f.exists():
            try:
                return pd.read_parquet(f)
            except Exception as e:
                logger.warning("Cache load failed: %s", e)
        return None

    def _save_cache(self, features, coin, timeframe):
        f = self.cache_dir / f"{coin}_{timeframe}_features.parquet"
        try:
            features.to_parquet(f)
        except Exception as e:
            logger.warning("Cache save failed: %s", e)

    def _update_metadata(self, coin, timeframe, feature_names):
        self.feature_metadata[f"{coin}_{timeframe}"] = {
            'coin': coin, 'timeframe': timeframe,
            'n_features': len(feature_names), 'feature_names': feature_names,
            'generated_at': datetime.now().isoformat(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=2000, freq='1h')
    price = 100 + np.random.randn(2000).cumsum()
    df = pd.DataFrame({
        'open': price + np.random.randn(2000) * 0.5,
        'high': price + np.abs(np.random.randn(2000)) * 2,
        'low': price - np.abs(np.random.randn(2000)) * 2,
        'close': price,
        'volume': np.abs(np.random.randn(2000)) * 1000 + 10000,
        'bitcoin_dominance': 50 + np.random.randn(2000).cumsum() * 0.1,
        'total_crypto_market_cap': 1e12 + np.random.randn(2000).cumsum() * 1e10,
        'fear_greed_value': 50 + np.random.randn(2000).cumsum() * 2,
        'stablecoin_supply_ratio': 0.1 + np.abs(np.random.randn(2000).cumsum()) * 0.001,
    }, index=dates)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)

    pipe = FeaturePipeline(enable_cache=False)
    feats = pipe.generate_features(df, 'BTC', '1H')
    Xm = pipe.select_features(feats)
    leaky = set(Xm.columns) & RAW_LEAKY_COLUMNS
    print(f"input cols={df.shape[1]}  total cols={feats.shape[1]}  model-safe={Xm.shape[1]}")
    print(f"raw leaky columns in model matrix: {leaky}  (must be empty)")