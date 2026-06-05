"""
Model configuration for all ML models.

Defines hyperparameters, training settings, and model-specific options.

Author: Research Team
Date: 2024
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class LightGBMConfig:
    """Configuration for LightGBM model."""
    
    # Core parameters
    objective: str = 'multiclass'
    num_class: int = 3
    boosting_type: str = 'gbdt'
    
    # Learning parameters
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    
    # Tree parameters
    min_child_samples: int = 20
    min_child_weight: float = 1e-3
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    
    # Regularization
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    
    # Training
    n_estimators: int = 1000
    early_stopping_rounds: int = 50
    
    # Other
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = -1
    
    # Custom parameters
    is_unbalance: bool = True
    metric: str = 'multi_logloss'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for LightGBM."""
        return {
            'objective': self.objective,
            'num_class': self.num_class,
            'boosting_type': self.boosting_type,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'min_child_samples': self.min_child_samples,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'subsample_freq': self.subsample_freq,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose,
            'is_unbalance': self.is_unbalance,
            'metric': self.metric,
        }


@dataclass
class XGBoostConfig:
    """Configuration for XGBoost model."""
    
    objective: str = 'multi:softprob'
    num_class: int = 3
    
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 1
    
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1
    
    n_estimators: int = 1000
    early_stopping_rounds: int = 50
    
    random_state: int = 42
    n_jobs: int = -1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for XGBoost."""
        return {
            'objective': self.objective,
            'num_class': self.num_class,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
        }


@dataclass
class CatBoostConfig:
    """Configuration for CatBoost model."""
    
    loss_function: str = 'MultiClass'
    classes_count: int = 3
    
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 3.0
    
    iterations: int = 1000
    early_stopping_rounds: int = 50
    
    random_state: int = 42
    thread_count: int = -1
    verbose: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CatBoost."""
        return {
            'loss_function': self.loss_function,
            'classes_count': self.classes_count,
            'learning_rate': self.learning_rate,
            'depth': self.depth,
            'l2_leaf_reg': self.l2_leaf_reg,
            'iterations': self.iterations,
            'early_stopping_rounds': self.early_stopping_rounds,
            'random_state': self.random_state,
            'thread_count': self.thread_count,
            'verbose': self.verbose,
        }


@dataclass
class GRUConfig:
    """Configuration for GRU model."""
    
    # Architecture
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    
    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Optimizer
    optimizer: str = 'adam'
    weight_decay: float = 1e-5
    
    # Sequence length
    sequence_length: int = 24
    
    # Device
    device: str = 'cuda'
    
    # Random state
    random_state: int = 42


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""
    
    # Architecture
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    
    # Training
    batch_size: int = 256
    learning_rate: float = 0.001
    num_epochs: int = 100
    early_stopping_patience: int = 10
    
    # Optimizer
    optimizer: str = 'adam'
    weight_decay: float = 1e-5
    
    # Sequence length
    sequence_length: int = 24
    
    # Device
    device: str = 'cuda'
    
    # Random state
    random_state: int = 42


@dataclass
class ModelConfig:
    """Master model configuration."""
    
    lightgbm: LightGBMConfig = field(default_factory=LightGBMConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
    gru: GRUConfig = field(default_factory=GRUConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    
    # Primary model
    primary_model: str = 'lightgbm'
    
    # Feature selection
    max_features: Optional[int] = None
    feature_importance_threshold: Optional[float] = None
    
    # Class weights
    use_sample_weights: bool = True
    use_class_weights: bool = True
    
    # Calibration
    calibration_method: str = 'isotonic'  # 'platt', 'isotonic', None
    
    # Ensemble
    use_ensemble: bool = False
    ensemble_method: str = 'voting'  # 'voting', 'stacking'
    ensemble_models: List[str] = field(default_factory=lambda: [
        'lightgbm', 'xgboost', 'catboost'
    ])
    
    # Model saving
    save_models: bool = True
    model_save_dir: str = 'models/saved'
    
    # Logging
    log_training: bool = True
    
    def get_model_config(self, model_name: str) -> Any:
        """Get configuration for specific model."""
        configs = {
            'lightgbm': self.lightgbm,
            'xgboost': self.xgboost,
            'catboost': self.catboost,
            'gru': self.gru,
            'lstm': self.lstm,
        }
        if model_name not in configs:
            raise ValueError(f"Unknown model: {model_name}")
        return configs[model_name]


# Global instance
MODEL_CONFIG = ModelConfig()