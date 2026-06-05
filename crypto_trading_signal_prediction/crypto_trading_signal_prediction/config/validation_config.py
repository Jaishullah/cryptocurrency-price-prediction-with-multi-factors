"""
Cross-validation and backtesting configuration.

Defines parameters for purged K-fold CV, embargo periods, and walk-forward validation.

Author: Research Team
Date: 2024
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class PurgedKFoldConfig:
    """Configuration for Purged K-Fold cross-validation."""
    
    # Number of splits
    n_splits: int = 5
    
    # Embargo period (in bars)
    # Should be at least as long as the holding period
    embargo_bars: int = 24
    
    # Purging parameters
    purge_overlapping_samples: bool = True
    
    # Minimum train/test size
    min_train_size: Optional[int] = None
    min_test_size: Optional[int] = None
    
    # Random state
    random_state: int = 42


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    
    # Training window size (in bars)
    train_window_size: int = 2000
    
    # Test window size (in bars)
    test_window_size: int = 500
    
    # Step size (how much to move forward each iteration)
    step_size: int = 250
    
    # Embargo period
    embargo_bars: int = 24
    
    # Expanding or rolling window
    window_type: Literal['rolling', 'expanding'] = 'rolling'
    
    # Minimum required training samples
    min_train_samples: int = 1000


@dataclass
class HyperparameterTuningConfig:
    """Configuration for hyperparameter tuning with Optuna."""
    
    # Number of trials
    n_trials: int = 100
    
    # Timeout (in seconds)
    timeout: Optional[int] = None
    
    # Optimization direction
    direction: str = 'maximize'
    
    # Metric to optimize
    metric: str = 'f1_weighted'
    
    # Pruning
    enable_pruning: bool = True
    pruner: str = 'median'  # 'median', 'hyperband', 'percentile'
    
    # Sampler
    sampler: str = 'tpe'  # 'tpe', 'random', 'grid'
    
    # Nested CV for tuning
    use_nested_cv: bool = True
    inner_cv_splits: int = 3
    
    # Random state
    random_state: int = 42
    
    # Parallel execution
    n_jobs: int = 1  # Optuna parallel trials
    
    # Study storage (for distributed tuning)
    storage: Optional[str] = None
    study_name: str = 'crypto_signal_prediction'


@dataclass
class ValidationConfig:
    """Master validation configuration."""
    
    purged_kfold: PurgedKFoldConfig = field(default_factory=PurgedKFoldConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    hyperparameter_tuning: HyperparameterTuningConfig = field(
        default_factory=HyperparameterTuningConfig
    )
    
    # Primary validation strategy
    validation_strategy: Literal['purged_kfold', 'walk_forward'] = 'purged_kfold'
    
    # Leakage detection
    enable_leakage_detection: bool = True
    leakage_detection_strict: bool = True
    
    # Metrics to compute
    compute_classification_metrics: bool = True
    compute_trading_metrics: bool = True
    compute_calibration_metrics: bool = True
    
    # Visualization
    plot_cv_splits: bool = True
    plot_performance: bool = True
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.purged_kfold.n_splits > 1, "n_splits must be > 1"
        assert self.purged_kfold.embargo_bars >= 0, "embargo_bars must be non-negative"
        assert self.walk_forward.train_window_size > 0, "train_window_size must be positive"
        assert self.walk_forward.test_window_size > 0, "test_window_size must be positive"
        assert self.walk_forward.step_size > 0, "step_size must be positive"


# Global instance
VALIDATION_CONFIG = ValidationConfig()