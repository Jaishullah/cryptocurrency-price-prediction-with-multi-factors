"""
Realistic backtesting engine for cryptocurrency trading strategies.

Execution model (anti-lookahead):
- A signal decided on bar t (from already-lagged predictions) is FILLED at the
  OPEN of bar t+1 (`fill_timing='next_open'`). Filling at the decision bar's
  close is available but flagged optimistic.
- Stop-loss / take-profit are intrabar triggers and fill at the barrier price on
  the bar they are touched. When both are touchable in the same bar, the STOP is
  assumed to hit first (no optimistic bias).
- A time-based exit mirrors the triple-barrier vertical barrier: positions are
  closed after `max_holding_period` bars so the traded horizon matches the
  labelled horizon.

Author: Research Team
Date: 2024
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Literal
import logging
from dataclasses import dataclass

from config.backtest_config import BACKTEST_CONFIG

logger = logging.getLogger(__name__)

# Bars per year and bar length (hours) per timeframe, for correct annualization.
BARS_PER_YEAR = {'1H': 24 * 365, '4H': 6 * 365, '1D': 365}
BAR_HOURS = {'1H': 1, '4H': 4, '1D': 24}


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: Literal['long', 'short']
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    fees: float
    slippage: float
    holding_period: int   # bars held
    coin: str
    exit_reason: str


@dataclass
class Position:
    coin: str
    side: Literal['long', 'short']
    entry_time: pd.Timestamp
    entry_idx: int
    entry_price: float
    size: float
    current_price: float
    unrealized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def update_unrealized_pnl(self, current_price: float):
        self.current_price = current_price
        if self.side == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.size / self.entry_price
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size / self.entry_price


class BacktestEngine:
    def __init__(
        self,
        config=BACKTEST_CONFIG,
        timeframe: str = '1H',
        fill_timing: Literal['next_open', 'close'] = 'next_open',
        max_holding_period: Optional[int] = None,
    ):
        self.config = config
        self.timeframe = timeframe
        self.fill_timing = fill_timing
        # Should match the labeller's holding_period; None disables the time exit.
        self.max_holding_period = max_holding_period
        self.bars_per_year = BARS_PER_YEAR.get(timeframe, 24 * 365)
        self.bar_hours = BAR_HOURS.get(timeframe, 1)

        self.initial_capital = config.initial_capital
        self._reset()
        if fill_timing == 'close':
            logger.warning("fill_timing='close' fills at the decision bar's close "
                           "(optimistic / potential lookahead). Prefer 'next_open'.")
        if max_holding_period is None:
            logger.warning("max_holding_period is None: no time-based exit, so the "
                           "traded horizon will not match the labelled horizon.")
        logger.info("BacktestEngine: $%.2f, tf=%s, fill=%s",
                    config.initial_capital, timeframe, fill_timing)

    # ------------------------------------------------------------------ #
    def run(self, data: pd.DataFrame, predictions: pd.DataFrame, coin: str = 'BTC') -> Dict:
        logger.info("Backtest %s: %d bars", coin, len(data))
        self._validate_inputs(data, predictions)
        self._reset()

        n = len(data)
        pending = None  # order decided on previous bar, filled at this bar's open

        for idx in range(n):
            ts = data.index[idx]
            bar = data.iloc[idx]

            # (1) Fill any pending order at THIS bar's open.
            if pending is not None:
                fill_px = bar['open'] if self.fill_timing == 'next_open' else pending['decide_close']
                if pending['action'] == 'enter' and coin not in self.positions:
                    self._open_position(ts, idx, fill_px, pending['side'], coin,
                                        pending['pred'], pending['vol'])
                elif pending['action'] == 'exit' and coin in self.positions:
                    self._close_position(ts, fill_px, coin, pending['reason'])
                pending = None

            pred_row = predictions.loc[ts] if ts in predictions.index else None

            # (2) Mark-to-market at the close.
            self._update_positions(ts, bar, coin)

            # (3) Intrabar stop/target (stop assumed first if both touched).
            exited = self._check_barrier_exits(ts, bar, coin)

            # (4) Decide a signal/time exit or a new entry -> fill next bar.
            if coin in self.positions and not exited:
                pos = self.positions[coin]
                bars_held = idx - pos.entry_idx
                sig = self._get_signal(pred_row) if pred_row is not None else 'neutral'
                if self.max_holding_period is not None and bars_held >= self.max_holding_period:
                    pending = {'action': 'exit', 'reason': 'time_limit', 'decide_close': bar['close']}
                elif (pos.side == 'long' and sig in ('short', 'neutral')) or \
                     (pos.side == 'short' and sig in ('long', 'neutral')):
                    pending = {'action': 'exit', 'reason': 'signal_reversal', 'decide_close': bar['close']}
            elif coin not in self.positions and pending is None and pred_row is not None:
                if len(self.positions) < self.config.risk_management.max_portfolio_positions:
                    sig = self._get_signal(pred_row)
                    allow = (sig == 'long' and self.config.trading_mode != 'short_only') or \
                            (sig == 'short' and self.config.trading_mode != 'long_only')
                    if sig in ('long', 'short') and allow:
                        pending = {'action': 'enter', 'side': sig, 'pred': pred_row,
                                   'vol': bar.get('volatility', None), 'decide_close': bar['close']}

            # (5) Record equity / drawdown at the close.
            equity = self._calculate_equity()
            self.equity_curve.append(equity)
            self.timestamps.append(ts)
            self.peak_equity = max(self.peak_equity, equity)
            dd = (self.peak_equity - equity) / self.peak_equity
            self.drawdown_curve.append(dd)
            self.max_drawdown = max(self.max_drawdown, dd)
            if dd > self.config.risk_management.max_drawdown_limit:
                logger.warning("Max drawdown limit hit at %s: %.2%%", ts, dd * 100)
                break

        self._close_all_positions(data.index[-1], data.iloc[-1]['close'], 'end_of_data')
        results = self._calculate_performance_metrics()
        logger.info("Done: %d trades, final $%.2f, return %.2f%%",
                    self.total_trades, self.capital, results.get('total_return', 0) * 100)
        return results

    # ------------------------------------------------------------------ #
    def _validate_inputs(self, data, predictions):
        missing = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns in data: {missing}")
        has_prob = {'prob_upper', 'prob_lower'}.issubset(predictions.columns)
        if not has_prob and 'label' not in predictions.columns:
            raise ValueError("predictions need either ('prob_upper','prob_lower') or 'label'")

    def _reset(self):
        self.capital = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.timestamps: List[pd.Timestamp] = []
        self.drawdown_curve: List[float] = []
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0
        self.total_trades = self.winning_trades = self.losing_trades = 0

    def _update_positions(self, ts, bar, coin):
        if coin in self.positions:
            self.positions[coin].update_unrealized_pnl(bar['close'])

    def _check_barrier_exits(self, ts, bar, coin) -> bool:
        """Intrabar SL/TP. Stop assumed to hit first if both are touchable."""
        if coin not in self.positions:
            return False
        pos = self.positions[coin]
        # Stop-loss first (conservative).
        if pos.stop_loss is not None:
            if (pos.side == 'long' and bar['low'] <= pos.stop_loss) or \
               (pos.side == 'short' and bar['high'] >= pos.stop_loss):
                self._close_position(ts, pos.stop_loss, coin, 'stop_loss')
                return True
        if pos.take_profit is not None:
            if (pos.side == 'long' and bar['high'] >= pos.take_profit) or \
               (pos.side == 'short' and bar['low'] <= pos.take_profit):
                self._close_position(ts, pos.take_profit, coin, 'take_profit')
                return True
        return False

    def _open_position(self, ts, idx, price, side, coin, pred, vol):
        # Slippage against us on entry.
        slip = self.config.trading_costs.slippage
        entry_price = price * (1 + slip) if side == 'long' else price * (1 - slip)

        size = self._calculate_position_size(entry_price, pred, vol)
        if size < self.config.trading_costs.min_trade_size:
            return
        fees = size * self.config.trading_costs.default_fee
        if size + fees > self.capital:
            size = (self.capital * 0.99) / (1 + self.config.trading_costs.default_fee)
            fees = size * self.config.trading_costs.default_fee
        if size < self.config.trading_costs.min_trade_size:
            return

        self.capital -= (size + fees)
        self.positions[coin] = Position(
            coin=coin, side=side, entry_time=ts, entry_idx=idx, entry_price=entry_price,
            size=size, current_price=entry_price,
            stop_loss=self._stop(entry_price, side), take_profit=self._take(entry_price, side),
        )

    def _close_position(self, ts, exit_price, coin, reason):
        if coin not in self.positions:
            return
        pos = self.positions[coin]
        slip = self.config.trading_costs.slippage
        # Slippage against us on exit.
        exit_price = exit_price * (1 - slip) if pos.side == 'long' else exit_price * (1 + slip)

        pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price if pos.side == 'long'
                   else (pos.entry_price - exit_price) / pos.entry_price)
        pnl = pos.size * pnl_pct
        exit_value = pos.size * (1 + pnl_pct)
        fees = exit_value * self.config.trading_costs.default_fee
        net_pnl = pnl - fees
        self.capital += (pos.size + net_pnl)

        holding_bars = int(round((ts - pos.entry_time).total_seconds() / 3600 / self.bar_hours))
        self.trades.append(Trade(
            entry_time=pos.entry_time, exit_time=ts, side=pos.side,
            entry_price=pos.entry_price, exit_price=exit_price, size=pos.size,
            pnl=net_pnl, pnl_pct=pnl_pct, fees=fees,
            slippage=pos.size * slip, holding_period=holding_bars, coin=coin, exit_reason=reason,
        ))
        self.total_trades += 1
        self.winning_trades += int(net_pnl > 0)
        self.losing_trades += int(net_pnl <= 0)
        del self.positions[coin]

    def _close_all_positions(self, ts, close_px, reason):
        for coin in list(self.positions.keys()):
            self._close_position(ts, close_px, coin, reason)

    def _get_signal(self, pred) -> str:
        if pred is None:
            return 'neutral'
        if 'prob_upper' in pred.index and 'prob_lower' in pred.index:
            pl, ps = pred['prob_upper'], pred['prob_lower']
            pn = pred.get('prob_vertical', 0)
            if pl > self.config.long_threshold and pl - max(ps, pn) > self.config.neutral_band:
                return 'long'
            if ps > self.config.short_threshold and ps - max(pl, pn) > self.config.neutral_band:
                return 'short'
            return 'neutral'
        if 'label' in pred.index:
            return {1: 'long', -1: 'short'}.get(pred['label'], 'neutral')
        return 'neutral'

    def _calculate_position_size(self, entry_price, pred, vol) -> float:
        ps = self.config.position_sizing
        method = ps.method
        if method == 'fixed_fraction':
            size = self.capital * ps.fixed_fraction
        elif method == 'kelly':
            p = max(pred.get('prob_upper', 0), pred.get('prob_lower', 0)) if pred is not None else 0.55
            b = 2.0
            f = max(0.0, (p * b - (1 - p)) / b) * ps.kelly_fraction
            size = self.capital * min(f, ps.kelly_max_position)
        elif method == 'volatility_target':
            # NOTE: `vol` must be the LAGGED volatility feature; using the current
            # bar's realized vol here would be lookahead.
            v = vol if (vol is not None and not pd.isna(vol)) else 0.02
            size = self.capital * (ps.volatility_target / (v + 1e-8))
        elif method == 'equal_weight':
            size = self.capital / self.config.risk_management.max_portfolio_positions
        else:
            raise ValueError(f"Unknown position sizing method: {method}")
        size = max(size, ps.min_position_size * self.capital)
        return min(size, self.capital * 0.95)

    def _stop(self, entry_price, side):
        p = self.config.risk_management.stop_loss_pct
        if p is None:
            return None
        return entry_price * (1 - p) if side == 'long' else entry_price * (1 + p)

    def _take(self, entry_price, side):
        p = self.config.risk_management.take_profit_pct
        if p is None:
            return None
        return entry_price * (1 + p) if side == 'long' else entry_price * (1 - p)

    def _calculate_equity(self) -> float:
        return self.capital + sum(p.size + p.unrealized_pnl for p in self.positions.values())

    def _calculate_performance_metrics(self) -> Dict:
        if not self.equity_curve:
            return {}
        r: Dict = {}
        final = self.equity_curve[-1]
        total_return = (final - self.initial_capital) / self.initial_capital
        r.update(initial_capital=self.initial_capital, final_equity=final,
                 total_return=total_return, total_pnl=final - self.initial_capital,
                 total_trades=self.total_trades, winning_trades=self.winning_trades,
                 losing_trades=self.losing_trades,
                 win_rate=self.winning_trades / max(self.total_trades, 1))

        if self.trades:
            td = pd.DataFrame([vars(t) for t in self.trades])
            wins, losses = td[td['pnl'] > 0], td[td['pnl'] <= 0]
            gp = wins['pnl'].sum() if len(wins) else 0.0
            gl = abs(losses['pnl'].sum()) if len(losses) else 0.0
            r.update(avg_pnl=td['pnl'].mean(), avg_pnl_pct=td['pnl_pct'].mean(),
                     total_fees=td['fees'].sum(), total_slippage=td['slippage'].sum(),
                     avg_win=wins['pnl'].mean() if len(wins) else 0.0,
                     avg_loss=losses['pnl'].mean() if len(losses) else 0.0,
                     profit_factor=gp / max(gl, 1e-8),
                     avg_holding_period=td['holding_period'].mean())

        r['max_drawdown'] = self.max_drawdown
        eq = pd.Series(self.equity_curve, index=self.timestamps)
        rets = eq.pct_change().dropna()
        if len(rets) > 1:
            ppy = self.bars_per_year
            r['annualized_return'] = (1 + total_return) ** (ppy / len(rets)) - 1
            ex = rets - self.config.risk_free_rate / ppy
            r['sharpe_ratio'] = ex.mean() / (ex.std() + 1e-8) * np.sqrt(ppy)
            dn = ex[ex < 0].std()
            r['sortino_ratio'] = ex.mean() / (dn + 1e-8) * np.sqrt(ppy)
            r['calmar_ratio'] = r['annualized_return'] / max(self.max_drawdown, 1e-8)
        return r

    def get_trades_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([vars(t) for t in self.trades]) if self.trades else pd.DataFrame()

    def get_equity_curve(self) -> pd.Series:
        return pd.Series(self.equity_curve, index=self.timestamps, name='equity')

    def get_drawdown_curve(self) -> pd.Series:
        return pd.Series(self.drawdown_curve, index=self.timestamps, name='drawdown')


class MultiAssetBacktester:
    """Backtest multiple coins on a shared capital pool."""

    def __init__(self, config=BACKTEST_CONFIG, timeframe='1H',
                 fill_timing='next_open', max_holding_period=None):
        self.engine = BacktestEngine(config, timeframe, fill_timing, max_holding_period)
        self.config = config

    def run(self, data_dict: Dict[str, pd.DataFrame],
            predictions_dict: Dict[str, pd.DataFrame]) -> Dict:
        e = self.engine
        e._reset()
        common = self._common_index(data_dict)
        coins = list(data_dict.keys())
        pending: Dict[str, dict] = {}

        for idx, ts in enumerate(common):
            # Fills at open.
            for coin in coins:
                if coin in pending and ts in data_dict[coin].index:
                    bar = data_dict[coin].loc[ts]
                    p = pending.pop(coin)
                    fill = bar['open'] if e.fill_timing == 'next_open' else p['decide_close']
                    if p['action'] == 'enter' and coin not in e.positions:
                        e._open_position(ts, idx, fill, p['side'], coin, p['pred'], p['vol'])
                    elif p['action'] == 'exit' and coin in e.positions:
                        e._close_position(ts, fill, coin, p['reason'])

            for coin in coins:
                if ts not in data_dict[coin].index:
                    continue
                bar = data_dict[coin].loc[ts]
                pred = predictions_dict[coin].loc[ts] if ts in predictions_dict[coin].index else None
                e._update_positions(ts, bar, coin)
                exited = e._check_barrier_exits(ts, bar, coin)
                if coin in e.positions and not exited:
                    pos = e.positions[coin]
                    sig = e._get_signal(pred)
                    if e.max_holding_period is not None and (idx - pos.entry_idx) >= e.max_holding_period:
                        pending[coin] = {'action': 'exit', 'reason': 'time_limit', 'decide_close': bar['close']}
                    elif (pos.side == 'long' and sig in ('short', 'neutral')) or \
                         (pos.side == 'short' and sig in ('long', 'neutral')):
                        pending[coin] = {'action': 'exit', 'reason': 'signal_reversal', 'decide_close': bar['close']}
                elif coin not in e.positions and coin not in pending and pred is not None:
                    if len(e.positions) < self.config.risk_management.max_portfolio_positions:
                        sig = e._get_signal(pred)
                        if sig in ('long', 'short'):
                            pending[coin] = {'action': 'enter', 'side': sig, 'pred': pred,
                                             'vol': bar.get('volatility', None), 'decide_close': bar['close']}

            equity = e._calculate_equity()
            e.equity_curve.append(equity)
            e.timestamps.append(ts)
            e.peak_equity = max(e.peak_equity, equity)
            dd = (e.peak_equity - equity) / e.peak_equity
            e.drawdown_curve.append(dd)
            e.max_drawdown = max(e.max_drawdown, dd)

        # Close leftovers at each coin's last close.
        for coin in list(e.positions.keys()):
            last_ts = data_dict[coin].index[-1]
            e._close_position(last_ts, data_dict[coin].iloc[-1]['close'], coin, 'end_of_data')

        results = e._calculate_performance_metrics()
        td = e.get_trades_dataframe()
        if len(td):
            results['trades_by_coin'] = td.groupby('coin').size().to_dict()
            results['pnl_by_coin'] = td.groupby('coin')['pnl'].sum().to_dict()
        return results

    @staticmethod
    def _common_index(data_dict):
        idx = None
        for df in data_dict.values():
            idx = df.index if idx is None else idx.intersection(df.index)
        return idx.sort_values()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)
    n = 5000
    dates = pd.date_range('2023-01-01', periods=n, freq='1h')
    price = 100 + np.random.randn(n).cumsum()
    data = pd.DataFrame({
        'open': price + np.random.randn(n) * 0.5,
        'high': price + np.abs(np.random.randn(n)) * 2,
        'low': price - np.abs(np.random.randn(n)) * 2,
        'close': price, 'volume': np.abs(np.random.randn(n)) * 1000 + 10000,
        'volatility': np.abs(np.random.randn(n) * 0.02 + 0.02),
    }, index=dates)
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    preds = pd.DataFrame({
        'prob_lower': np.random.uniform(0.2, 0.4, n),
        'prob_vertical': np.random.uniform(0.3, 0.5, n),
        'prob_upper': np.random.uniform(0.2, 0.4, n),
    }, index=dates)
    preds = preds.div(preds.sum(axis=1), axis=0)

    eng = BacktestEngine(timeframe='1H', fill_timing='next_open', max_holding_period=24)
    res = eng.run(data, preds, coin='BTC')
    print({k: round(v, 4) for k, v in res.items() if isinstance(v, float)})