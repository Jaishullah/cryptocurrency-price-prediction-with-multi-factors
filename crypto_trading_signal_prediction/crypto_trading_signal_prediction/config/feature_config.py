"""
Feature engineering configuration.

Defines all parameters for feature calculation, ensuring
point-in-time correctness and preventing data leakage.

Author: Research Team
Date: 2024
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class VolatilityConfig:
    """Configuration for volatility estimators."""
    
    # EWMA parameters
    ewma_spans: List[int] = field(default_factory=lambda: [12, 24, 48])
    ewma_min_periods: int = 10
    
    # ATR parameters
    atr_periods: List[int] = field(default_factory=lambda: [14, 21, 28])
    
    # Yang-Zhang parameters
    yang_zhang_window: int = 20
    yang_zhang_min_periods: int = 15
    
    # Volatility of volatility
    vol_of_vol_window: int = 20


@dataclass
class ReturnsConfig:
    """Configuration for returns-based features."""
    
    # Return periods (in bars)
    return_periods: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 24])
    
    # Momentum periods
    momentum_periods: List[int] = field(default_factory=lambda: [6, 12, 24, 48])
    
    # Z-score windows
    zscore_windows: List[int] = field(default_factory=lambda: [20, 50, 100])
    
    # Use log returns
    use_log_returns: bool = True


@dataclass
class VolumeConfig:
    """Configuration for volume-based features."""
    
    # Volume z-score windows
    volume_zscore_windows: List[int] = field(default_factory=lambda: [20, 50])
    
    # Volume momentum periods
    volume_momentum_periods: List[int] = field(default_factory=lambda: [6, 12, 24])
    
    # VWAP periods
    vwap_periods: List[int] = field(default_factory=lambda: [24, 48])
    
    # Dollar volume calculation
    calculate_dollar_volume: bool = True


@dataclass
class TechnicalConfig:
    """Configuration for technical indicators."""
    
    # RSI parameters
    rsi_periods: List[int] = field(default_factory=lambda: [14, 21])
    
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    
    # Stochastic
    stoch_k_period: int = 14
    stoch_d_period: int = 3
    
    # Moving averages
    ma_periods: List[int] = field(default_factory=lambda: [7, 25, 99])


@dataclass
class CrossAssetConfig:
    """Configuration for cross-asset features."""
    
    # Correlation windows
    correlation_windows: List[int] = field(default_factory=lambda: [24, 48, 168])
    
    # Reference asset for correlation
    reference_asset: str = 'BTC'
    
    # Calculate market breadth
    calculate_breadth: bool = True
    
    # Breadth window
    breadth_window: int = 24


@dataclass
class FractionalDiffConfig:
    """Configuration for fractional differentiation."""
    
    # Range of d values to test
    d_min: float = 0.0
    d_max: float = 1.0
    d_step: float = 0.1
    
    # ADF test parameters
    adf_max_lag: Optional[int] = None
    adf_pvalue_threshold: float = 0.05
    
    # Minimum weight threshold for truncation
    min_weight: float = 1e-5


@dataclass
class FeatureConfig:
    """Master feature configuration."""
    
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    returns: ReturnsConfig = field(default_factory=ReturnsConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    technical: TechnicalConfig = field(default_factory=TechnicalConfig)
    cross_asset: CrossAssetConfig = field(default_factory=CrossAssetConfig)
    fractional_diff: FractionalDiffConfig = field(default_factory=FractionalDiffConfig)
    
    # Multi-timeframe settings
    include_4h_features: bool = True
    include_daily_features: bool = True
    
    # Feature naming convention
    feature_prefix: str = "feat_"
    
    # Caching
    cache_features: bool = True
    cache_dir: str = "data/features"
    
    # Parallel processing
    n_jobs: int = -1  # Use all available cores
    
    def get_all_return_periods(self) -> List[int]:
        """Get all unique return periods across configurations."""
        return sorted(set(
            self.returns.return_periods + 
            self.returns.momentum_periods
        ))
    
    def get_all_windows(self) -> List[int]:
        """Get all unique window sizes across configurations."""
        windows = set()
        windows.update(self.volatility.ewma_spans)
        windows.update(self.volatility.atr_periods)
        windows.update([self.volatility.yang_zhang_window])
        windows.update(self.returns.zscore_windows)
        windows.update(self.volume.volume_zscore_windows)
        windows.update(self.volume.vwap_periods)
        windows.update([self.technical.bb_period])
        windows.update(self.technical.ma_periods)
        windows.update(self.cross_asset.correlation_windows)
        return sorted(windows)


# Global instance
FEATURE_CONFIG = FeatureConfig()