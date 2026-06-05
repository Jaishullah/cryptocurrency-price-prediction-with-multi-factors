"""
Triple barrier labeling configuration.

Defines parameters for the triple barrier method and sample weighting.

Author: Research Team
Date: 2024
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal


@dataclass
class TripleBarrierConfig:
    """Configuration for triple barrier labeling."""
    
    # Barrier width multipliers (in units of volatility)
    upper_barrier_multiplier: float = 2.0
    lower_barrier_multiplier: float = 2.0
    
    # Holding period (in bars)
    holding_period: int = 24  # 24 hours for 1H data
    
    # Volatility estimator to use for barrier calculation
    # Options: 'ewma', 'atr', 'yang_zhang'
    volatility_estimator: Literal['ewma', 'atr', 'yang_zhang'] = 'atr'
    
    # Volatility window (if applicable)
    volatility_window: int = 14
    
    # Minimum price move to consider (prevents noise labels)
    min_return_threshold: float = 0.001  # 0.1%
    
    # Allow asymmetric barriers
    symmetric_barriers: bool = True
    
    # Minimum number of observations for labeling
    min_observations_for_label: int = 100
    
    # Label encoding
    label_mapping: dict = field(default_factory=lambda: {
        'upper': 1,   # Profit target hit
        'lower': -1,  # Stop loss hit
        'vertical': 0  # Time limit reached
    })


@dataclass
class SampleWeightConfig:
    """Configuration for sample weighting."""
    
    # Use concurrency-based weights
    use_concurrency_weights: bool = True
    
    # Use return-based weights
    use_return_weights: bool = True
    
    # Return weight exponent (higher = more emphasis on large moves)
    return_weight_exponent: float = 1.0
    
    # Minimum weight (prevents zero weights)
    min_weight: float = 0.01
    
    # Maximum weight (prevents extreme weights)
    max_weight: float = 10.0
    
    # Normalize weights to sum to number of samples
    normalize_weights: bool = True


@dataclass
class SequentialBootstrapConfig:
    """Configuration for sequential bootstrap."""
    
    # Number of bootstrap samples
    n_bootstrap_samples: int = 1000
    
    # Sample size (fraction of original data)
    sample_size_fraction: float = 1.0
    
    # Random seed
    random_state: int = 42
    
    # Use indicator matrix for efficiency
    use_indicator_matrix: bool = True


@dataclass
class LabelingConfig:
    """Master labeling configuration."""
    
    triple_barrier: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    sample_weight: SampleWeightConfig = field(default_factory=SampleWeightConfig)
    sequential_bootstrap: SequentialBootstrapConfig = field(
        default_factory=SequentialBootstrapConfig
    )
    
    # Class balance handling
    handle_class_imbalance: bool = True
    class_weight_strategy: Literal['balanced', 'custom', None] = 'balanced'
    
    # Minimum samples per class
    min_samples_per_class: int = 50
    
    # Save label metadata
    save_metadata: bool = True
    
    # Multiple barrier configurations for ablation
    ablation_barrier_configs: List[dict] = field(default_factory=lambda: [
        {'upper': 1.5, 'lower': 1.5, 'holding': 24},
        {'upper': 2.0, 'lower': 2.0, 'holding': 24},
        {'upper': 2.5, 'lower': 2.5, 'holding': 24},
        {'upper': 2.0, 'lower': 2.0, 'holding': 48},
    ])
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert self.triple_barrier.upper_barrier_multiplier > 0, \
            "Upper barrier multiplier must be positive"
        assert self.triple_barrier.lower_barrier_multiplier > 0, \
            "Lower barrier multiplier must be positive"
        assert self.triple_barrier.holding_period > 0, \
            "Holding period must be positive"
        assert 0 <= self.triple_barrier.min_return_threshold < 1, \
            "Min return threshold must be between 0 and 1"


# Global instance
LABELING_CONFIG = LabelingConfig()