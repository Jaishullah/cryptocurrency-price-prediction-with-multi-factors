"""
Data loader with comprehensive validation.

Loads raw cryptocurrency data with point-in-time correctness guarantees.

Author: Research Team
Date: 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
from datetime import datetime

from config.data_config import DATA_CONFIG

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and validate cryptocurrency market data.

    Ensures:
    - Correct data types
    - Temporal ordering
    - No future information leakage
    - Data quality standards
    """

    def __init__(self, config=DATA_CONFIG):
        """
        Initialize DataLoader.

        Args:
            config: Data configuration object
        """
        self.config = config
        self.loaded_data: Dict[str, pd.DataFrame] = {}

    def load_single_file(
        self,
        coin: str,
        timeframe: str,
        validate: bool = True
    ) -> pd.DataFrame:
        """
        Load a single cryptocurrency data file.

        Args:
            coin: Cryptocurrency symbol (e.g., 'BTC')
            timeframe: Timeframe (e.g., '1H')
            validate: Whether to validate data after loading

        Returns:
            DataFrame with loaded data (DatetimeIndex on 'timestamp')

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If validation fails
        """
        file_path = self.config.get_file_path(coin, timeframe, 'raw')

        if not file_path.exists():
            # NOTE: reference file_path.parent rather than a non-existent
            # config attribute, so the FileNotFoundError is actually raised
            # (the previous `self.config.RAW_DATA_DIR` raised AttributeError
            # while building this message).
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Please ensure data files are placed in {file_path.parent}"
            )

        logger.info(f"Loading {coin} {timeframe} from {file_path}")

        # Load parquet file
        df = pd.read_parquet(file_path)

        # Add metadata
        df['coin'] = coin
        df['timeframe'] = timeframe

        if validate:
            df = self._validate_dataframe(df, coin, timeframe)
        else:
            # Even without full validation, normalise the index so that
            # downstream date filtering behaves identically on both paths.
            df = self._normalize_index(df)

        # Store in cache
        key = f"{coin}_{timeframe}"
        self.loaded_data[key] = df

        logger.info(f"Loaded {len(df)} rows for {coin} {timeframe}")

        return df

    def load_multiple_coins(
        self,
        coins: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        validate: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple cryptocurrencies and timeframes.

        Args:
            coins: List of coin symbols (default: all coins in config)
            timeframes: List of timeframes (default: all timeframes in config)
            validate: Whether to validate data

        Returns:
            Dictionary mapping 'COIN_TIMEFRAME' to DataFrame
        """
        if coins is None:
            coins = self.config.coins
        if timeframes is None:
            timeframes = self.config.timeframes

        data_dict = {}

        for coin in coins:
            for timeframe in timeframes:
                try:
                    df = self.load_single_file(coin, timeframe, validate)
                    key = f"{coin}_{timeframe}"
                    data_dict[key] = df
                except Exception as e:
                    logger.error(f"Failed to load {coin} {timeframe}: {str(e)}")
                    continue

        return data_dict

    def _normalize_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure a sorted DatetimeIndex on 'timestamp' without running the full
        validation suite. Used on the validate=False path so that get_data /
        get_train_test_split filter consistently regardless of `validate`.
        """
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').set_index('timestamp')
        elif not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("No 'timestamp' column or DatetimeIndex found; "
                           "date filtering may not work as expected.")
        return df

    def _validate_dataframe(
        self,
        df: pd.DataFrame,
        coin: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Validate DataFrame structure and content.

        Args:
            df: DataFrame to validate
            coin: Cryptocurrency symbol
            timeframe: Timeframe

        Returns:
            Validated DataFrame (DatetimeIndex on 'timestamp')

        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        missing_cols = set(self.config.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns for {coin} {timeframe}: {missing_cols}"
            )

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Check for duplicates
        n_duplicates = df['timestamp'].duplicated().sum()
        if n_duplicates > 0:
            logger.warning(
                f"Found {n_duplicates} duplicate timestamps in {coin} {timeframe}. "
                "Keeping first occurrence."
            )
            df = df.drop_duplicates(subset='timestamp', keep='first')

        # Check temporal ordering (strictly increasing)
        if not df['timestamp'].is_monotonic_increasing:
            raise ValueError(
                f"Timestamps are not strictly increasing for {coin} {timeframe}"
            )

        # Validate OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )

        if invalid_ohlc.any():
            n_invalid = invalid_ohlc.sum()
            logger.warning(
                f"Found {n_invalid} rows with invalid OHLC relationships in "
                f"{coin} {timeframe}. These will be marked for review."
            )
            df.loc[invalid_ohlc, 'data_quality_flag'] = 'invalid_ohlc'

        # Check for negative or zero prices
        for col in self.config.price_columns:
            invalid_prices = df[col] <= 0
            if invalid_prices.any():
                raise ValueError(
                    f"Found {invalid_prices.sum()} non-positive values in "
                    f"{col} for {coin} {timeframe}"
                )

        # Check for missing values
        missing_pct = df[self.config.required_columns].isnull().sum() / len(df)
        high_missing = missing_pct[missing_pct > self.config.max_missing_pct]

        if not high_missing.empty:
            logger.warning(
                f"Columns with high missing percentage in {coin} {timeframe}:\n"
                f"{high_missing}"
            )

        # Check minimum data points
        if len(df) < self.config.min_data_points:
            raise ValueError(
                f"Insufficient data points for {coin} {timeframe}: "
                f"{len(df)} < {self.config.min_data_points}"
            )

        # Apply data types
        df = self._apply_dtypes(df)

        # Set timestamp as index
        df = df.set_index('timestamp')

        return df

    def _apply_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply correct data types to columns.

        Integer targets are cast via pandas' nullable 'Int64' when the column
        contains missing values, since a plain int64 cast on NaN raises.

        Args:
            df: DataFrame to process

        Returns:
            DataFrame with correct dtypes
        """
        for col, dtype in self.config.dtype_mapping.items():
            if col not in df.columns or col == 'timestamp':
                continue
            try:
                if dtype in ('int8', 'int16', 'int32', 'int64') and df[col].isnull().any():
                    # Preserve missing values with a nullable integer dtype.
                    df[col] = df[col].astype('Int64')
                else:
                    df[col] = df[col].astype(dtype)
            except Exception as e:
                logger.warning(f"Could not convert {col} to {dtype}: {str(e)}")

        return df

    def get_data(
        self,
        coin: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get data for a specific coin and timeframe, optionally filtered by date.

        Args:
            coin: Cryptocurrency symbol
            timeframe: Timeframe
            start_date: Start date (inclusive) as string 'YYYY-MM-DD'
            end_date: End date (inclusive) as string 'YYYY-MM-DD'

        Returns:
            Filtered DataFrame
        """
        key = f"{coin}_{timeframe}"

        # Load if not already loaded
        if key not in self.loaded_data:
            self.load_single_file(coin, timeframe)

        df = self.loaded_data[key].copy()

        # Apply date filters
        if start_date is not None:
            df = df[df.index >= pd.to_datetime(start_date)]

        if end_date is not None:
            df = df[df.index <= pd.to_datetime(end_date)]

        return df

    def get_train_test_split(
        self,
        coin: str,
        timeframe: str,
        test_start_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets based on date.

        Args:
            coin: Cryptocurrency symbol
            timeframe: Timeframe
            test_start_date: Start date of test set (default from config)

        Returns:
            Tuple of (train_df, test_df)
        """
        if test_start_date is None:
            test_start_date = self.config.test_start_date

        df = self.get_data(coin, timeframe)

        train_df = df[df.index < pd.to_datetime(test_start_date)]
        test_df = df[df.index >= pd.to_datetime(test_start_date)]

        logger.info(
            f"Split {coin} {timeframe}: "
            f"Train={len(train_df)} rows, Test={len(test_df)} rows"
        )

        return train_df, test_df

    def get_combined_data(
        self,
        coins: Optional[List[str]] = None,
        timeframe: str = '1H'
    ) -> pd.DataFrame:
        """
        Get combined data for multiple coins at a single timeframe.

        Useful for global model training. NOTE: the result is sorted by
        (timestamp, coin); any time-series operation applied afterwards MUST be
        grouped by 'coin' to avoid bleeding values across assets.

        Args:
            coins: List of coins (default: all coins)
            timeframe: Timeframe

        Returns:
            Combined DataFrame with 'coin' column
        """
        if coins is None:
            coins = self.config.coins

        dfs = []
        for coin in coins:
            df = self.get_data(coin, timeframe)
            df = df.reset_index()
            df['coin'] = coin
            dfs.append(df)

        combined = pd.concat(dfs, axis=0, ignore_index=True)
        combined = combined.sort_values(['timestamp', 'coin']).reset_index(drop=True)

        logger.info(f"Combined {len(coins)} coins: {len(combined)} total rows")

        return combined


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    loader = DataLoader()

    # Load single file
    btc_1h = loader.load_single_file('BTC', '1H')
    print(f"BTC 1H shape: {btc_1h.shape}")
    print(f"Columns: {btc_1h.columns.tolist()}")

    # Load multiple files
    data_dict = loader.load_multiple_coins(
        coins=['BTC', 'ETH'],
        timeframes=['1H', '4H']
    )

    for key, df in data_dict.items():
        print(f"{key}: {df.shape}")

    # Get train/test split
    train, test = loader.get_train_test_split('BTC', '1H')
    print(f"Train: {train.shape}, Test: {test.shape}")