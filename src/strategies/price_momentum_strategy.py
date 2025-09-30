"""
Price Momentum Strategy Implementation

This module implements price momentum strategies using the trading-strategies package.
It provides a Flask-compatible interface for the web application.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from .base_strategy import BaseStrategy

# Import functions from the trading-strategies package
from trading_strategies.price_momentum_strategy import (
    calculate_momentum_metrics,
    create_long_only_portfolio,
    create_dollar_neutral_portfolio,
    validate_portfolio_constraints
)


def fetch_price_data_for_momentum(tickers: List[str], period: str = "2y") -> pd.DataFrame:
    """
    Fetch stock price data for momentum calculations.
    
    Args:
        tickers: List of stock ticker symbols
        period: Time period (need longer period for momentum calculations)
    
    Returns:
        DataFrame with close prices for each ticker
    """
    try:
        if len(tickers) == 1:
            # For single ticker, use Ticker class
            ticker_obj = yf.Ticker(tickers[0])
            data = ticker_obj.history(period=period)
            if not data.empty and 'Close' in data.columns:
                return pd.DataFrame({tickers[0]: data['Close']})
            else:
                return pd.DataFrame()
        else:
            # For multiple tickers
            data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
            if 'Close' in data.columns:
                return data['Close']
            else:
                return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching momentum data: {e}")
        return pd.DataFrame()


def run_price_momentum_strategy(tickers: Union[str, List[str]],
                               strategy_type: str,
                               parameters: Dict,
                               period: str = "2y",
                               prices_df: pd.DataFrame = None) -> Dict:
    """
    Run price momentum strategy for tickers (single or multiple).
    
    Args:
        tickers: Single ticker string or list of ticker symbols
        strategy_type: 'cumulative_return', 'mean_return', or 'risk_adjusted'
        parameters: Strategy parameters
        period: Data period (only used if prices_df is None)
        prices_df: Optional pre-loaded price data
    
    Returns:
        Dictionary with strategy results for all tickers
    """
    try:
        # Normalize tickers input
        if isinstance(tickers, str):
            ticker_list = [tickers]
        else:
            ticker_list = tickers
        
        # Use provided data or fetch if not provided
        if prices_df is None:
            prices_df = fetch_price_data_for_momentum(ticker_list, period)
            
            if prices_df.empty:
                return {'error': f'Could not fetch momentum data for tickers: {ticker_list}'}
        
        # Ensure all requested tickers are in the data
        missing_tickers = [t for t in ticker_list if t not in prices_df.columns]
        if missing_tickers:
            print(f"Warning: Missing tickers in price data: {missing_tickers}")
        
        # Calculate momentum metrics for ALL tickers in the dataset
        formation_period = parameters.get('formation_period', 12)
        skip_period = parameters.get('skip_period', 1)
        
        momentum_metrics = calculate_momentum_metrics(
            prices_df, 
            formation_period=formation_period, 
            skip_period=skip_period
        )
        
        if momentum_metrics.empty:
            return {'error': 'Could not calculate momentum metrics for any ticker'}
        
        # Determine the selection metric based on strategy type
        if strategy_type == 'cumulative_return':
            selection_metric = 'rcum'
        elif strategy_type == 'mean_return':
            selection_metric = 'rmean'
        elif strategy_type == 'risk_adjusted':
            selection_metric = 'rrisk_adj'
        else:
            return {'error': f'Unknown momentum strategy type: {strategy_type}'}
        
        # Calculate rankings for all tickers
        valid_metrics = momentum_metrics[selection_metric].dropna()
        if len(valid_metrics) == 0:
            return {'error': 'No valid momentum metrics available'}
        
        # Prepare results for each ticker
        ticker_results = {}
        
        for ticker in ticker_list:
            if ticker not in momentum_metrics.index:
                ticker_results[ticker] = {
                    'current_signal': 'hold',
                    'error': f'No momentum data for {ticker}',
                    'momentum_metrics': {},
                    'percentile_rank': None,
                    'chart_data': {}
                }
                continue
            
            ticker_metrics = momentum_metrics.loc[ticker]
            signal_value = ticker_metrics[selection_metric]
            
            # Calculate percentile ranking against all other tickers
            if pd.isna(signal_value):
                current_signal = 'hold'
                percentile_rank = None
            else:
                percentile_rank = (valid_metrics <= signal_value).mean()
                
                # Generate trading signal based on percentile
                if percentile_rank >= 0.7:  # Top 30%
                    current_signal = 'long'
                elif percentile_rank <= 0.3:  # Bottom 30%
                    current_signal = 'short'
                else:
                    current_signal = 'hold'
            
            # Prepare chart data for this ticker
            chart_data = prepare_momentum_chart_data(momentum_metrics, ticker, selection_metric)
            
            ticker_results[ticker] = {
                'current_signal': current_signal,
                'momentum_metrics': {
                    'rcum': float(ticker_metrics['rcum']) if pd.notna(ticker_metrics['rcum']) else None,
                    'rmean': float(ticker_metrics['rmean']) if pd.notna(ticker_metrics['rmean']) else None,
                    'volatility': float(ticker_metrics['volatility']) if pd.notna(ticker_metrics['volatility']) else None,
                    'rrisk_adj': float(ticker_metrics['rrisk_adj']) if pd.notna(ticker_metrics['rrisk_adj']) else None,
                    'signal_value': float(signal_value) if pd.notna(signal_value) else None
                },
                'percentile_rank': float(percentile_rank) if percentile_rank is not None else None,
                'chart_data': chart_data
            }
        
        # Create portfolio weights for demonstration (using all tickers)
        try:
            portfolio_weights = create_long_only_portfolio(
                momentum_metrics,
                selection_metric=selection_metric,
                top_percentile=parameters.get('top_percentile', 0.3),
                weighting_scheme=parameters.get('weighting_scheme', 'equal')
            )
            
            # Validate portfolio
            portfolio_validation = validate_portfolio_constraints(portfolio_weights, 'long_only')
            
        except Exception as e:
            print(f"Warning: Could not create portfolio: {e}")
            portfolio_validation = {}
        
        return {
            'success': True,
            'strategy_type': strategy_type,
            'parameters': parameters,
            'ticker_results': ticker_results,
            'total_tickers_analyzed': len(valid_metrics),
            'portfolio_validation': portfolio_validation
        }
        
    except Exception as e:
        return {'error': f'Error running momentum strategy: {str(e)}'}


def prepare_momentum_chart_data(momentum_metrics: pd.DataFrame, ticker: str, selection_metric: str) -> Dict:
    """
    Prepare data for momentum strategy visualization.
    
    Args:
        momentum_metrics: DataFrame with momentum metrics
        ticker: Target ticker
        selection_metric: Metric used for selection
    
    Returns:
        Dictionary with chart data
    """
    try:
        # Create a simple comparison chart
        valid_metrics = momentum_metrics[selection_metric].dropna().sort_values(ascending=False)
        
        chart_data = {
            'ticker_rank': int((valid_metrics >= valid_metrics.loc[ticker]).sum()) if ticker in valid_metrics.index else 0,
            'total_stocks': len(valid_metrics),
            'ticker_value': float(valid_metrics.loc[ticker]) if ticker in valid_metrics.index else 0,
            'top_performers': valid_metrics.head(5).to_dict(),
            'bottom_performers': valid_metrics.tail(5).to_dict(),
            'selection_metric': selection_metric
        }
        
        return chart_data
        
    except Exception as e:
        print(f"Error preparing momentum chart data: {e}")
        return {}


def get_momentum_strategy_summary(ticker: str, strategy_type: str, parameters: Dict) -> str:
    """
    Generate a human-readable summary of the momentum strategy configuration.
    
    Args:
        ticker: Stock ticker
        strategy_type: Strategy type
        parameters: Strategy parameters
    
    Returns:
        Strategy summary string
    """
    formation_period = parameters.get('formation_period', 12)
    skip_period = parameters.get('skip_period', 1)
    
    if strategy_type == 'cumulative_return':
        return f"Cumulative Return Momentum ({formation_period}M formation, {skip_period}M skip) for {ticker}"
    elif strategy_type == 'mean_return':
        return f"Mean Return Momentum ({formation_period}M formation, {skip_period}M skip) for {ticker}"
    elif strategy_type == 'risk_adjusted':
        return f"Risk-Adjusted Momentum ({formation_period}M formation, {skip_period}M skip) for {ticker}"
    else:
        return f"Unknown momentum strategy for {ticker}"


class PriceMomentumStrategy(BaseStrategy):
    """Price Momentum Strategy implementation following the BaseStrategy interface"""
    
    def __init__(self):
        super().__init__("price_momentum")
    
    def get_default_configurations(self) -> Dict:
        """Return default configurations for all momentum strategy variants"""
        return {
            'cumulative_return': {
                'formation_period': 12,
                'skip_period': 1,
                'top_percentile': 0.3,
                'weighting_scheme': 'equal'
            },
            'mean_return': {
                'formation_period': 12,
                'skip_period': 1,
                'top_percentile': 0.3,
                'weighting_scheme': 'equal'
            },
            'risk_adjusted': {
                'formation_period': 12,
                'skip_period': 1,
                'top_percentile': 0.3,
                'weighting_scheme': 'equal'
            }
        }
    
    def run_strategy(self, ticker: str, strategy_type: str, parameters: Dict, 
                    period: str = "2y", prices_df: pd.DataFrame = None) -> Dict:
        """Run momentum strategy using the standalone function"""
        return run_price_momentum_strategy(ticker, strategy_type, parameters, period, prices_df)
    
    def get_strategy_summary(self, ticker: str, strategy_type: str, parameters: Dict) -> str:
        """Generate summary using the standalone function"""
        return get_momentum_strategy_summary(ticker, strategy_type, parameters)


if __name__ == "__main__":
    """Test the price momentum strategy"""
    print("🚀 Testing Price Momentum Strategy")
    print("=" * 50)
    
    # Test parameters
    test_cases = [
        {
            'ticker': 'AAPL',
            'strategy_type': 'cumulative_return',
            'parameters': {'formation_period': 12, 'skip_period': 1},
            'period': '2y'
        },
        {
            'ticker': 'MSFT',
            'strategy_type': 'risk_adjusted',
            'parameters': {'formation_period': 6, 'skip_period': 1},
            'period': '2y'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Test Case {i}: {test_case['strategy_type'].upper()} Momentum Strategy")
        print(f"Ticker: {test_case['ticker']}")
        print(f"Parameters: {test_case['parameters']}")
        print(f"Period: {test_case['period']}")
        print("-" * 30)
        
        try:
            result = run_price_momentum_strategy(
                test_case['ticker'],
                test_case['strategy_type'],
                test_case['parameters'],
                test_case['period']
            )
            
            if result.get('success'):
                print("✅ Strategy executed successfully!")
                print(f"Summary: {get_momentum_strategy_summary(test_case['ticker'], test_case['strategy_type'], test_case['parameters'])}")
                print(f"Current Signal: {result['current_signal'].upper()}")
                
                metrics = result.get('momentum_metrics', {})
                print(f"Cumulative Return: {metrics.get('rcum', 'N/A')}")
                print(f"Risk-Adjusted Return: {metrics.get('rrisk_adj', 'N/A')}")
                print(f"Percentile Rank: {metrics.get('percentile_rank', 'N/A')}")
                
            else:
                print("❌ Strategy failed!")
                print(f"Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception occurred: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🏁 Testing completed!")
