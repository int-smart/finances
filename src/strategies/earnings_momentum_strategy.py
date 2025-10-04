"""
Earnings Momentum Strategy Implementation

This module implements the earnings momentum strategy using the trading-strategies package.
It focuses on Standardized Unexpected Earnings (SUE) to identify stocks with positive
earnings momentum for portfolio construction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from trading_strategies.price_momentum_strategy import (
    calculate_standardized_unexpected_earnings,
    calculate_earnings_momentum_metrics,
    create_earnings_momentum_portfolio,
    analyze_earnings_momentum_signals,
    validate_earnings_data
)
from .base_strategy import BaseStrategy


class EarningsMomentumStrategy(BaseStrategy):
    """Earnings momentum strategy based on Standardized Unexpected Earnings (SUE)"""
    
    def __init__(self):
        super().__init__("earnings_momentum")
    
    def get_default_configurations(self) -> Dict:
        """Return default configurations for earnings momentum strategy variants"""
        return {
            'sue_long_only': {
                'selection_metric': 'sue',
                'top_percentile': 0.1,
                'portfolio_type': 'long_only',
                'weighting_scheme': 'equal',
                'min_quarters': 8
            },
            'eps_change_long_only': {
                'selection_metric': 'eps_change',
                'top_percentile': 0.1,
                'portfolio_type': 'long_only',
                'weighting_scheme': 'equal',
                'min_quarters': 8
            }
        }
    
    def run_strategy(self, ticker: str, strategy_type: str, parameters: Dict, 
                    period: str = "1y", earnings_data: pd.DataFrame = None) -> Dict:
        """
        Run the earnings momentum strategy for a given ticker or multiple tickers.
        
        Args:
            ticker: Stock ticker symbol or list of tickers
            strategy_type: Strategy variant (e.g., 'sue_long_only', 'sue_dollar_neutral')
            parameters: Strategy parameters
            period: Data period (not used for earnings data)
            earnings_data: Optional pre-loaded earnings data DataFrame
        
        Returns:
            Dictionary with strategy results
        """
        try:
            # Handle both single ticker and multiple tickers
            if isinstance(ticker, str):
                tickers = [ticker]
            else:
                tickers = ticker
            
            # Validate earnings data
            if earnings_data is None:
                return {
                    'success': False,
                    'error': 'Earnings data is required for earnings momentum strategy'
                }
            
            # Validate earnings data quality (relaxed for limited data)
            validation = validate_earnings_data(earnings_data)
            min_quarters = parameters.get('min_quarters', 4)
            
            # Check if we have enough data for analysis (flexible requirement)
            available_quarters = len(earnings_data)
            if available_quarters < 5:  # Minimum 5 quarters required
                return {
                    'success': False,
                    'error': f'Insufficient earnings data: need at least 5 quarters, got {available_quarters}'
                }
            elif available_quarters < min_quarters:
                print(f"⚠️ Using {available_quarters} quarters instead of target {min_quarters} quarters")
            
            # Calculate earnings momentum metrics
            earnings_metrics = calculate_earnings_momentum_metrics(earnings_data)
            
            # Filter to only requested tickers if provided
            if tickers:
                available_tickers = earnings_metrics.index.intersection(tickers)
                if len(available_tickers) == 0:
                    return {
                        'success': False,
                        'error': f'No earnings data available for requested tickers: {tickers}'
                    }
                earnings_metrics = earnings_metrics.loc[available_tickers]
            
            # Create portfolio based on strategy type
            portfolio_weights = create_earnings_momentum_portfolio(
                earnings_metrics=earnings_metrics,
                selection_metric=parameters.get('selection_metric', 'sue'),
                top_percentile=parameters.get('top_percentile', 0.1),
                bottom_percentile=parameters.get('bottom_percentile', 0.1),
                portfolio_type=parameters.get('portfolio_type', 'long_only'),
                weighting_scheme=parameters.get('weighting_scheme', 'equal')
            )
            
            # Analyze signals
            signal_analysis = analyze_earnings_momentum_signals(earnings_metrics)
            
            # Determine signals for each ticker
            ticker_results = {}
            for ticker in earnings_metrics.index:
                weight = portfolio_weights.get(ticker, 0.0)
                
                # Determine signal based on weight
                if weight > 0:
                    signal = 'buy'
                elif weight < 0:
                    signal = 'sell'
                else:
                    signal = 'hold'
                
                # Get SUE value for this ticker
                sue_value = earnings_metrics.loc[ticker, 'sue'] if 'sue' in earnings_metrics.columns else np.nan
                eps_change = earnings_metrics.loc[ticker, 'eps_change'] if 'eps_change' in earnings_metrics.columns else np.nan
                
                ticker_results[ticker] = {
                    'current_signal': signal,
                    'weight': weight,
                    'sue_value': sue_value,
                    'eps_change': eps_change,
                    'recent_eps': earnings_metrics.loc[ticker, 'recent_eps'] if 'recent_eps' in earnings_metrics.columns else np.nan,
                    'eps_4q_ago': earnings_metrics.loc[ticker, 'eps_4q_ago'] if 'eps_4q_ago' in earnings_metrics.columns else np.nan
                }
            
            # Calculate portfolio metrics
            total_weight = portfolio_weights.sum()
            num_long = (portfolio_weights > 0).sum()
            num_short = (portfolio_weights < 0).sum()
            num_hold = (portfolio_weights == 0).sum()
            
            return {
                'success': True,
                'ticker_results': ticker_results,
                'portfolio_weights': portfolio_weights.to_dict(),
                'signal_analysis': signal_analysis,
                'portfolio_metrics': {
                    'total_weight': total_weight,
                    'num_long': num_long,
                    'num_short': num_short,
                    'num_hold': num_hold,
                    'portfolio_type': parameters.get('portfolio_type', 'long_only')
                },
                'strategy_type': strategy_type,
                'parameters': parameters
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Earnings momentum strategy failed: {str(e)}'
            }
    
    def get_strategy_summary(self, ticker: str, strategy_type: str, parameters: Dict) -> str:
        """Generate a human-readable summary of the earnings momentum strategy configuration"""
        selection_metric = parameters.get('selection_metric', 'sue')
        portfolio_type = parameters.get('portfolio_type', 'long_only')
        top_percentile = parameters.get('top_percentile', 0.1)
        weighting_scheme = parameters.get('weighting_scheme', 'equal')
        
        if selection_metric == 'sue':
            metric_desc = "Standardized Unexpected Earnings (SUE)"
        elif selection_metric == 'eps_change':
            metric_desc = "EPS Change (4 quarters)"
        else:
            metric_desc = f"{selection_metric.upper()}"
        
        if portfolio_type == 'long_only':
            portfolio_desc = f"Long-only portfolio (top {top_percentile*100:.0f}%)"
        else:
            bottom_percentile = parameters.get('bottom_percentile', 0.1)
            portfolio_desc = f"Dollar-neutral portfolio (long top {top_percentile*100:.0f}%, short bottom {bottom_percentile*100:.0f}%)"
        
        weighting_desc = {
            'equal': 'Equal weighting',
            'sue_weighted': 'SUE-weighted',
            'eps_weighted': 'EPS change-weighted'
        }.get(weighting_scheme, weighting_scheme)
        
        return f"Earnings Momentum Strategy ({strategy_type}): {metric_desc} selection, {portfolio_desc}, {weighting_desc}"
    
    def validate_earnings_data_quality(self, earnings_data: pd.DataFrame) -> Dict:
        """
        Validate earnings data quality for the strategy.
        
        Args:
            earnings_data: DataFrame with quarterly earnings data
            
        Returns:
            Dictionary with validation results
        """
        return validate_earnings_data(earnings_data)
    
    def get_earnings_momentum_analysis(self, earnings_data: pd.DataFrame) -> Dict:
        """
        Get comprehensive earnings momentum analysis.
        
        Args:
            earnings_data: DataFrame with quarterly earnings data
            
        Returns:
            Dictionary with analysis results
        """
        try:
            earnings_metrics = calculate_earnings_momentum_metrics(earnings_data)
            return analyze_earnings_momentum_signals(earnings_metrics)
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    def create_custom_portfolio(self, earnings_data: pd.DataFrame, 
                              selection_metric: str = 'sue',
                              top_percentile: float = 0.1,
                              bottom_percentile: float = 0.1,
                              portfolio_type: str = 'long_only',
                              weighting_scheme: str = 'equal') -> Dict:
        """
        Create a custom earnings momentum portfolio with specified parameters.
        
        Args:
            earnings_data: DataFrame with quarterly earnings data
            selection_metric: Metric to use for selection
            top_percentile: Fraction of top performers to go long
            bottom_percentile: Fraction of bottom performers to short
            portfolio_type: 'long_only' or 'dollar_neutral'
            weighting_scheme: Weighting scheme
            
        Returns:
            Dictionary with portfolio results
        """
        try:
            earnings_metrics = calculate_earnings_momentum_metrics(earnings_data)
            portfolio_weights = create_earnings_momentum_portfolio(
                earnings_metrics=earnings_metrics,
                selection_metric=selection_metric,
                top_percentile=top_percentile,
                bottom_percentile=bottom_percentile,
                portfolio_type=portfolio_type,
                weighting_scheme=weighting_scheme
            )
            
            return {
                'success': True,
                'portfolio_weights': portfolio_weights.to_dict(),
                'earnings_metrics': earnings_metrics.to_dict(),
                'portfolio_summary': {
                    'total_stocks': len(earnings_metrics),
                    'selected_stocks': len(portfolio_weights[portfolio_weights != 0]),
                    'long_positions': (portfolio_weights > 0).sum(),
                    'short_positions': (portfolio_weights < 0).sum(),
                    'total_weight': portfolio_weights.sum()
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Portfolio creation failed: {str(e)}'
            }
