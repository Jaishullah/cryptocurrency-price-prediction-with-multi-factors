# Research Methodology: Cryptocurrency Trading Signal Prediction

## 1. Executive Summary

This research implements a rigorous, publication-grade system for predicting cryptocurrency trading signals using machine learning. The methodology strictly follows quantitative finance best practices to ensure academic validity and prevent common pitfalls such as data leakage, overfitting, and selection bias.

## 2. Research Objectives

### Primary Objective
Develop a multi-asset classification system that predicts directional price movements across 10 major cryptocurrencies using the triple barrier method for label generation.

### Secondary Objectives
1. Compare global (multi-asset) vs. per-coin models
2. Evaluate multiple volatility estimators for barrier setting
3. Assess impact of cross-asset and market structure features
4. Validate results using rigorous statistical tests

## 3. Data Description

### 3.1 Assets
- Bitcoin (BTC)
- Ethereum (ETH)
- Binance Coin (BNB)
- Solana (SOL)
- Ripple (XRP)
- Cardano (ADA)
- Dogecoin (DOGE)
- Tron (TRX)
- Chainlink (LINK)
- Avalanche (AVAX)

### 3.2 Timeframes
- 1-hour (primary)
- 4-hour (secondary)
- Daily (secondary)

### 3.3 Features
- OHLCV data
- Fear & Greed Index
- Bitcoin Dominance
- Total Crypto Market Cap
- Stablecoin Supply Ratio

## 4. Labeling Methodology

### 4.1 Triple Barrier Method

We implement López de Prado's triple barrier method:

**Barriers:**
- Upper barrier: P(t) × (1 + τ × σ_t)
- Lower barrier: P(t) × (1 - τ × σ_t)
- Vertical barrier: t + h

**Labels:**
- +1: Upper barrier hit first (long signal)
- -1: Lower barrier hit first (short signal)
-  0: Vertical barrier reached first (neutral)

**Parameters:**
- τ = 2.0 (barrier width multiplier)
- h = 24 bars (holding period)
- σ_t estimated via ATR(14)

### 4.2 Sample Weighting

Weights account for:
1. Label concurrency (overlapping events)
2. Return magnitude
3. Average uniqueness

Formula: w_i = u_i × |r_i|

Where:
- u_i = average uniqueness of sample i
- r_i = return achieved during event lifespan

## 5. Feature Engineering

### 5.1 Point-in-Time Correctness

All features strictly use only past information:
- All features shifted by 1 bar minimum
- Multi-timeframe features use strict alignment
- No future-looking calculations

### 5.2 Feature Groups

**Volatility (12 features)**
- EWMA volatility (3 windows)
- ATR volatility (3 periods)
- Yang-Zhang volatility
- Volatility of volatility

**Returns (15 features)**
- Log returns (5 periods)
- Momentum (4 periods)
- Z-scored momentum (3 windows)

**Volume (10 features)**
- Volume z-score (2 windows)
- Volume momentum (3 periods)
- VWAP distance (2 periods)
- Dollar volume metrics

**Technical Indicators (20 features)**
- RSI (2 periods)
- MACD (3 components)
- Bollinger Bands (4 metrics)
- Stochastic (2 components)
- Moving averages (3 periods × 2 types)

**Market Structure (8 features)**
- Bitcoin dominance
- Total market cap
- Stablecoin supply ratio
- Fear & Greed index

**Time Features (8 features)**
- Hour (sin/cos encoding)
- Day of week (sin/cos encoding)
- Month (sin/cos encoding)
- Weekend indicator

**Total: ~73 base features per timeframe**

## 6. Validation Strategy

### 6.1 Purged K-Fold Cross-Validation

Standard K-fold is inappropriate due to label overlap. We implement:

**Purging:**
Remove training samples whose event end times overlap with test period.

**Embargo:**
Remove samples within embargo period after training set.

**Parameters:**
- n_splits = 5
- embargo_pct = 0.02 (2% of data)

### 6.2 Walk-Forward Validation

Alternative validation using:
- Training window: 2000 bars
- Test window: 500 bars
- Step size: 250 bars
- Window type: Rolling

## 7. Model Architecture

### 7.1 Primary Model: LightGBM

**Hyperparameters (optimized via Optuna):**
```python
{
    'objective': 'multiclass',
    'num_class': 3,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1
}