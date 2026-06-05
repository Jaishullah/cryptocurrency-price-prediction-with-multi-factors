"""
Backtesting configuration.

Defines parameters for realistic backtesting including fees, slippage,
position sizing, and risk management.

Author: Research Team
Date: 2024
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict


@dataclass
class TradingCostsConfig:
    """Configuration for trading costs."""
    
    # Trading fees (as fraction, e.g., 0.001 = 0.1%)
    maker_fee: float = 0.0005  # 0.05% for limit orders
    taker_fee: float = 0.001   # 0.1% for market orders
    
    # Assume we use market orders (taker)
    default_fee: float = 0.001
    
    # Slippage (as fraction of price)
    slippage: float = 0.0005  # 0.05%
    
    # Minimum trade size (in quote currency, e.g., USDT)
    min_trade_size: float = 10.0
    
    # Spread estimation (bid-ask spread as fraction)
    spread: float = 0.0002  # 0.02%


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing."""
    
    # Position sizing method
    # Options: 'fixed_fraction', 'kelly', 'equal_weight', 'volatility_target'
    method: Literal['fixed_fraction', 'kelly', 'equal_weight', 'volatility_target'] = 'kelly'
    
    # Fixed fraction of capital per trade
    fixed_fraction: float = 0.1  # 10% of capital
    
    # Kelly criterion parameters
    kelly_fraction: float = 0.5  # Half Kelly for safety
    kelly_max_position: float = 0.25  # Maximum 25% of capital
    
    # Volatility target (annualized)
    volatility_target: float = 0.20  # 20% annualized volatility
    
    # Maximum leverage
    max_leverage: float = 1.0  # No leverage by default
    
    # Minimum position size (as fraction of capital)
    min_position_size: float = 0.01  # 1%


@dataclass
class RiskManagementConfig:
    """Configuration for risk management."""
    
    # Stop loss (as fraction, e.g., 0.02 = 2%)
    stop_loss_pct: Optional[float] = None  # None = use triple barrier
    
    # Take profit (as fraction)
    take_profit_pct: Optional[float] = None  # None = use triple barrier
    
    # Trailing stop (as fraction)
    trailing_stop_pct: Optional[float] = None
    
    # Maximum drawdown limit
    max_drawdown_limit: float = 0.25  # Stop trading if 25% drawdown
    
    # Maximum concurrent positions
    max_concurrent_positions: int = 1  # Only one position per coin
    
    # Maximum total portfolio positions
    max_portfolio_positions: int = 5
    
    # Risk per trade (as fraction of capital)
    risk_per_trade: float = 0.02  # 2% risk per trade


@dataclass
class BacktestConfig:
    """Master backtesting configuration."""
    
    trading_costs: TradingCostsConfig = field(default_factory=TradingCostsConfig)
    position_sizing: PositionSizingConfig = field(default_factory=PositionSizingConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    
    # Initial capital (in quote currency)
    initial_capital: float = 10000.0
    
    # Prediction threshold for trading
    # Trade only if prediction probability > threshold
    long_threshold: float = 0.55   # Threshold for going long
    short_threshold: float = 0.55  # Threshold for going short
    neutral_band: float = 0.05     # Don't trade if probabilities are close
    
    # Trading mode
    # Options: 'long_only', 'short_only', 'long_short'
    trading_mode: Literal['long_only', 'short_only', 'long_short'] = 'long_short'
    
    # Rebalancing frequency (in bars)
    rebalance_frequency: int = 1  # Check every bar
    
    # Latency simulation (in bars)
    execution_latency: int = 0  # Assume instant execution (conservative)
    
    # Compound returns
    compound_returns: bool = True
    
    # Benchmark
    benchmark_coin: str = 'BTC'  # Compare against BTC buy-and-hold
    
    # Reporting
    generate_trade_log: bool = True
    generate_equity_curve: bool = True
    generate_drawdown_curve: bool = True
    
    # Performance metrics to compute
    metrics_to_compute: list = field(default_factory=lambda: [
        'total_return',
        'annualized_return',
        'sharpe_ratio',
        'sortino_ratio',
        'max_drawdown',
        'calmar_ratio',
        'win_rate',
        'profit_factor',
        'avg_win',
        'avg_loss',
        'num_trades',
        'avg_holding_period',
    ])
    
    # Risk-free rate for Sharpe/Sortino (annualized)
    risk_free_rate: float = 0.02  # 2%
    
    # Target annualized return for MAR ratio
    target_return: float = 0.0
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        assert 0 <= self.trading_costs.maker_fee <= 1, "maker_fee must be between 0 and 1"
        assert 0 <= self.trading_costs.taker_fee <= 1, "taker_fee must be between 0 and 1"
        assert 0 <= self.trading_costs.slippage <= 1, "slippage must be between 0 and 1"
        assert self.initial_capital > 0, "initial_capital must be positive"
        assert 0 <= self.long_threshold <= 1, "long_threshold must be between 0 and 1"
        assert 0 <= self.short_threshold <= 1, "short_threshold must be between 0 and 1"


# Global instance
BACKTEST_CONFIG = BacktestConfig()