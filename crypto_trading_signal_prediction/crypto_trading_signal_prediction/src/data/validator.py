"""
Comprehensive data validation and quality checks.

Implements rigorous validation to detect issues that could compromise
research validity.

Author: Research Team
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validate data quality and integrity.
    
    Checks:
    - Missing values
    - Outliers
    - Temporal consistency
    - Statistical properties
    - Data leakage indicators
    """
    
    def __init__(
        self,
        max_missing_pct: float = 0.05,
        outlier_std_threshold: float = 10.0
    ):
        """
        Initialize DataValidator.
        
        Args:
            max_missing_pct: Maximum allowed missing percentage
            outlier_std_threshold: Z-score threshold for outlier detection
        """
        self.max_missing_pct = max_missing_pct
        self.outlier_std_threshold = outlier_std_threshold
        self.validation_results: Dict = {}
    
    def validate(
        self,
        df: pd.DataFrame,
        name: str = "data"
    ) -> Tuple[bool, Dict]:
        """
        Run all validation checks.
        
        Args:
            df: DataFrame to validate
            name: Name for reporting
            
        Returns:
            Tuple of (is_valid, validation_report)
        """
        logger.info(f"Validating {name}...")
        
        results = {
            'name': name,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'checks': {}
        }
        
        # Run individual checks
        results['checks']['missing_values'] = self._check_missing_values(df)
        results['checks']['duplicates'] = self._check_duplicates(df)
        results['checks']['temporal_ordering'] = self._check_temporal_ordering(df)
        results['checks']['outliers'] = self._check_outliers(df)
        results['checks']['data_distribution'] = self._check_data_distribution(df)
        
        # Determine if valid
        is_valid = all(
            check.get('passed', True)
            for check in results['checks'].values()
        )
        
        results['is_valid'] = is_valid
        
        # Store results
        self.validation_results[name] = results
        
        # Log summary
        self._log_validation_summary(results)
        
        return is_valid, results
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values."""
        missing_counts = df.isnull().sum()
        missing_pcts = missing_counts / len(df)
        
        high_missing = missing_pcts[missing_pcts > self.max_missing_pct]
        
        return {
            'passed': len(high_missing) == 0,
            'total_missing': int(missing_counts.sum()),
            'columns_with_high_missing': high_missing.to_dict(),
            'missing_by_column': missing_pcts.to_dict()
        }
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate rows."""
        if isinstance(df.index, pd.DatetimeIndex):
            n_duplicates = df.index.duplicated().sum()
        else:
            n_duplicates = df.duplicated().sum()
        
        return {
            'passed': n_duplicates == 0,
            'n_duplicates': int(n_duplicates),
            'duplicate_pct': float(n_duplicates / len(df))
        }
    
    def _check_temporal_ordering(self, df: pd.DataFrame) -> Dict:
        """Check temporal ordering and consistency."""
        if not isinstance(df.index, pd.DatetimeIndex):
            return {
                'passed': True,
                'message': 'Not a time series (no DatetimeIndex)'
            }
        
        # Check monotonic increasing
        is_monotonic = df.index.is_monotonic_increasing
        
        # Check for gaps
        time_diffs = df.index.to_series().diff()
        expected_freq = time_diffs.mode()[0] if len(time_diffs) > 0 else None
        
        # Count gaps larger than expected
        if expected_freq is not None:
            gaps = time_diffs[time_diffs > expected_freq * 1.5]
            n_gaps = len(gaps)
        else:
            n_gaps = 0
        
        return {
            'passed': is_monotonic and n_gaps == 0,
            'is_monotonic': is_monotonic,
            'expected_frequency': str(expected_freq) if expected_freq else None,
            'n_gaps': int(n_gaps),
            'largest_gap': str(time_diffs.max()) if len(time_diffs) > 0 else None
        }
    
    def _check_outliers(self, df: pd.DataFrame) -> Dict:
        """Check for outliers in numerical columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        outlier_report = {}
        
        for col in numeric_cols:
            # Skip columns with low variance
            if df[col].std() == 0:
                continue
            
            # Z-score method
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            n_outliers = (z_scores > self.outlier_std_threshold).sum()
            
            if n_outliers > 0:
                outlier_report[col] = {
                    'n_outliers': int(n_outliers),
                    'pct_outliers': float(n_outliers / len(df)),
                    'max_z_score': float(z_scores.max())
                }
        
        return {
            'passed': len(outlier_report) == 0,
            'columns_with_outliers': outlier_report
        }
    
    def _check_data_distribution(self, df: pd.DataFrame) -> Dict:
        """Check data distribution properties."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        distribution_stats = {}
        
        for col in numeric_cols[:10]:  # Limit to first 10 for efficiency
            distribution_stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis())
            }
        
        return {
            'passed': True,
            'distribution_stats': distribution_stats
        }
    
    def _log_validation_summary(self, results: Dict) -> None:
        """Log validation summary."""
        status = "PASSED" if results['is_valid'] else "FAILED"
        logger.info(f"Validation {status} for {results['name']}")
        
        for check_name, check_result in results['checks'].items():
            check_status = "✓" if check_result.get('passed', True) else "✗"
            logger.info(f"  {check_status} {check_name}")
            
            if not check_result.get('passed', True):
                logger.warning(f"    Details: {check_result}")
    
    def check_leakage_indicators(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str
    ) -> Dict:
        """
        Check for potential data leakage indicators.
        
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names
            label_col: Label column name
            
        Returns:
            Dictionary with leakage check results
        """
        results = {}
        
        # Check 1: Perfect correlation (suspicious)
        correlations = df[feature_cols + [label_col]].corr()[label_col].drop(label_col)
        high_corr_features = correlations[np.abs(correlations) > 0.95]
        
        results['high_correlation_features'] = {
            'features': high_corr_features.to_dict(),
            'warning': len(high_corr_features) > 0
        }
        
        # Check 2: Features with future information (NaN in early periods)
        nan_at_start = {}
        for col in feature_cols:
            first_valid_idx = df[col].first_valid_index()
            if first_valid_idx is not None:
                n_leading_nans = df.index.get_loc(first_valid_idx)
                if n_leading_nans > 0:
                    nan_at_start[col] = n_leading_nans
        
        results['features_with_leading_nans'] = nan_at_start
        
        # Check 3: Constant features (no predictive power)
        constant_features = [
            col for col in feature_cols
            if df[col].nunique() <= 1
        ]
        
        results['constant_features'] = constant_features
        
        logger.info("Leakage indicator check complete")
        if high_corr_features.any():
            logger.warning(
                f"Found {len(high_corr_features)} features with suspiciously "
                "high correlation to label"
            )
        
        return results


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='1H')
    df = pd.DataFrame({
        'price': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randn(1000) * 1000 + 10000,
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000)
    }, index=dates)
    
    # Add some issues
    df.loc[df.index[50:60], 'price'] = np.nan  # Missing values
    df.loc[df.index[100], 'volume'] = 1000000  # Outlier
    
    # Validate
    validator = DataValidator()
    is_valid, results = validator.validate(df, name="sample_data")
    
    print(f"\nValidation {'passed' if is_valid else 'failed'}")
    print(f"Results: {results}")