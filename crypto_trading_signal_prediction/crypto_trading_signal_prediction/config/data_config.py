"""
Data configuration for cryptocurrency trading signal prediction.

This module defines all data-related constants, paths, and parameters
used throughout the system.

Author: Research Team
Date: 2024
"""

from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, field
import os

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
LABELS_DIR = DATA_DIR / "labels"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DIR, LABELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    
    # Cryptocurrencies to analyze
    coins: List[str] = field(default_factory=lambda: [
        'BTC', 'ETH', 'BNB', 'SOL', 'XRP',
        'ADA', 'DOGE', 'TRX', 'LINK', 'AVAX'
    ])
    
    # Timeframes
    timeframes: List[str] = field(default_factory=lambda: ['1H', '4H', '1D'])
    
    # Primary timeframe for training
    primary_timeframe: str = '1H'
    
    # Expected columns in raw data
    required_columns: List[str] = field(default_factory=lambda: [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_ratio',
        'funding_rate', 'fear_greed_value', 'fear_greed_label',
        'hour_of_day', 'day_of_week', 'week_of_month', 'month',
        'is_weekend', 'bitcoin_dominance', 'total_crypto_market_cap',
        'altcoin_market_cap', 'stablecoin_market_cap',
        'stablecoin_supply_ratio', 'ssr'
    ])
    
    # OHLCV columns
    ohlcv_columns: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close', 'volume'
    ])
    
    # Price columns that must be positive
    price_columns: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close'
    ])
    
    # Data validation parameters
    max_missing_pct: float = 0.05  # Maximum 5% missing values allowed
    min_data_points: int = 1000    # Minimum data points required
    outlier_std_threshold: float = 10.0  # Z-score threshold for outliers
    
    # Data types
    dtype_mapping: Dict[str, str] = field(default_factory=lambda: {
        'timestamp': 'datetime64[ns]',
        'open': 'float64',
        'high': 'float64',
        'low': 'float64',
        'close': 'float64',
        'volume': 'float64',
        'quote_asset_volume': 'float64',
        'number_of_trades': 'int64',
        'taker_buy_ratio': 'float64',
        'funding_rate': 'float64',
        'fear_greed_value': 'float64',
        'fear_greed_label': 'category',
        'hour_of_day': 'int8',
        'day_of_week': 'int8',
        'week_of_month': 'int8',
        'month': 'int8',
        'is_weekend': 'bool',
        'bitcoin_dominance': 'float64',
        'total_crypto_market_cap': 'float64',
        'altcoin_market_cap': 'float64',
        'stablecoin_market_cap': 'float64',
        'stablecoin_supply_ratio': 'float64',
        'ssr': 'float64'
    })
    
    # Train/test split
    test_start_date: str = '2024-01-01'  # Adjust based on your data
    validation_start_date: str = '2023-07-01'
    
    # Resampling rules for multi-timeframe alignment
    timeframe_to_freq: Dict[str, str] = field(default_factory=lambda: {
        '1H': '1H',
        '4H': '4H',
        '1D': '1D'
    })
    
    def get_file_path(self, coin: str, timeframe: str, data_type: str = 'raw') -> Path:
        """
        Get file path for a specific coin and timeframe.
        
        Args:
            coin: Cryptocurrency symbol (e.g., 'BTC')
            timeframe: Timeframe (e.g., '1H')
            data_type: Type of data ('raw', 'processed', 'features', 'labels')
            
        Returns:
            Path object pointing to the file
        """
        if data_type == 'raw':
            base_dir = RAW_DATA_DIR
        elif data_type == 'processed':
            base_dir = PROCESSED_DATA_DIR
        elif data_type == 'features':
            base_dir = FEATURES_DIR
        elif data_type == 'labels':
            base_dir = LABELS_DIR
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        
        return base_dir / f"{coin}_{timeframe}.parquet"
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert len(self.coins) > 0, "At least one coin must be specified"
        assert len(self.timeframes) > 0, "At least one timeframe must be specified"
        assert self.primary_timeframe in self.timeframes, \
            f"Primary timeframe {self.primary_timeframe} not in timeframes list"
        assert 0 <= self.max_missing_pct <= 1, "max_missing_pct must be between 0 and 1"
        assert self.min_data_points > 0, "min_data_points must be positive"


# Global instance
DATA_CONFIG = DataConfig()