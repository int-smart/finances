"""
Moving Average Strategy Implementation

This module implements moving average strategies using the trading-strategies package.
It provides a Flask-compatible interface for the web application.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from .base_strategy import BaseStrategy

# Import functions from the trading-strategies package
from trading_strategies.moving_average_strategy import (
    calculate_moving_average,
    generate_single_ma_signals,
    generate_two_ma_signals,
    generate_three_ma_signals,
    create_ma_strategy_portfolio,
    analyze_ma_strategy_performance
)


def fetch_stock_prices(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """
    Fetch stock price data for multiple tickers using yfinance.
    
    Args:
        tickers: List of stock ticker symbols
        period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    
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
            return data['Close']
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


def convert_stock_data_to_prices_df(ticker: str, stock_data: Dict) -> pd.DataFrame:
    """
    Convert stock data from scheduler format to prices DataFrame.
    
    Args:
        ticker: Stock ticker symbol
        stock_data: Stock data from scheduler
    
    Returns:
        DataFrame with price data for the ticker
    """
    try:
        if not stock_data or 'stocks' not in stock_data or ticker not in stock_data['stocks']:
            return pd.DataFrame()
        
        ticker_data = stock_data['stocks'][ticker]
        if 'history' not in ticker_data:
            return pd.DataFrame()
        
        history = ticker_data['history']
        
        # Convert to DataFrame if needed
        if isinstance(history, dict):
            df = pd.DataFrame(history)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif 'index' in df.columns:
                df['Date'] = pd.to_datetime(df['index'])
                df.set_index('Date', inplace=True)
                df.drop('index', axis=1, inplace=True)
        else:
            df = history.copy()
        
        # Ensure we have Close prices
        if 'Close' in df.columns:
            return pd.DataFrame({ticker: df['Close']})
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error converting stock data for {ticker}: {e}")
        return pd.DataFrame()


def prepare_ma_parameters(strategy_type: str, parameters: Dict) -> Dict:
    """
    Prepare parameters for the trading-strategies package functions.
    
    Args:
        strategy_type: 'single', 'double', or 'triple'
        parameters: Strategy parameters
    
    Returns:
        Dictionary with formatted parameters
    """
    ma_params = {'ma_type': 'ema'}
    
    if strategy_type == 'single':
        ma_params['ma_window'] = parameters.get('ema_window', 20)
    elif strategy_type == 'double':
        ma_params['short_window'] = parameters.get('short_window', 12)
        ma_params['long_window'] = parameters.get('long_window', 26)
    elif strategy_type == 'triple':
        ma_params['window1'] = parameters.get('window1', 5)
        ma_params['window2'] = parameters.get('window2', 12)
        ma_params['window3'] = parameters.get('window3', 26)
    
    # Add alpha if provided
    if 'alpha' in parameters and parameters['alpha']:
        ma_params['alpha'] = parameters['alpha']
    
    return ma_params


def execute_ma_strategy_with_data(ticker: str, strategy_type: str, parameters: Dict, prices_df: pd.DataFrame) -> Dict:
    """
    Execute moving average strategy with provided price data.
    
    Args:
        ticker: Stock ticker symbol
        strategy_type: 'single', 'double', or 'triple'
        parameters: Strategy parameters
        prices_df: DataFrame with price data
    
    Returns:
        Dictionary with strategy results
    """
    try:
        # Map strategy types to package function names
        strategy_mapping = {
            'single': 'single',
            'double': 'two', 
            'triple': 'three'
        }
        
        if strategy_type not in strategy_mapping:
            return {'error': f'Unknown strategy type: {strategy_type}'}
        
        # Prepare parameters for the package functions
        ma_params = prepare_ma_parameters(strategy_type, parameters)
        
        # Create strategy portfolio using the package
        strategy_results = create_ma_strategy_portfolio(
            prices_df,
            strategy_type=strategy_mapping[strategy_type],
            ma_params=ma_params,
            portfolio_type='long_only'
        )
        
        # Analyze performance using the package
        analysis = analyze_ma_strategy_performance(strategy_results)
        
        # Get signals for the specific ticker
        ticker_signals = strategy_results['signals'].get(ticker)
        if ticker_signals is None or ticker_signals.empty:
            return {'error': f'No signals generated for {ticker}'}
        
        # Ensure all numeric columns are properly typed
        for col in ticker_signals.columns:
            if col != 'signal' and ticker_signals[col].dtype == 'object':
                ticker_signals[col] = pd.to_numeric(ticker_signals[col], errors='coerce')
        
        # Get current signal
        current_signal = analysis['current_signals'].get(ticker, 'hold')
        
        # Prepare chart data
        chart_data = prepare_chart_data(ticker_signals, strategy_type)
        
        return {
            'success': True,
            'ticker': ticker,
            'strategy_type': strategy_type,
            'parameters': parameters,
            'current_signal': current_signal,
            'chart_data': chart_data,
            'signals_data': ticker_signals.tail(10).to_dict('records'),  # Last 10 signals
            'analysis': analysis
        }
        
    except Exception as e:
        return {'error': f'Error running strategy: {str(e)}'}


def run_moving_average_strategy(ticker: str,
                               strategy_type: str,
                               parameters: Dict,
                               period: str = "1y",
                               prices_df: pd.DataFrame = None) -> Dict:
    """
    Run moving average strategy for a given ticker.
    
    Args:
        ticker: Stock ticker symbol
        strategy_type: 'single', 'double', or 'triple'
        parameters: Strategy parameters
        period: Data period (only used if prices_df is None)
        prices_df: Optional pre-loaded price data. If None, data will be fetched.
    
    Returns:
        Dictionary with strategy results
    """
    try:
        # Use provided data or fetch if not provided
        if prices_df is None:
            prices_df = fetch_stock_prices([ticker], period)
            
            if prices_df.empty:
                return {'error': f'Could not fetch data for {ticker}'}
        
        # Execute strategy with the data
        return execute_ma_strategy_with_data(ticker, strategy_type, parameters, prices_df)
        
    except Exception as e:
        return {'error': f'Error running strategy: {str(e)}'}


def prepare_chart_data(signals: pd.DataFrame, strategy_type: str) -> Dict:
    """
    Prepare data for chart visualization.
    
    Args:
        signals: DataFrame with signals and prices
        strategy_type: Type of strategy
    
    Returns:
        Dictionary with chart data
    """
    # Convert index to string for JSON serialization
    dates = signals.index.strftime('%Y-%m-%d').tolist()
    
    chart_data = {
        'dates': dates,
        'prices': signals['price'].tolist()
    }
    
    # Add MA data based on strategy type
    if strategy_type == 'single':
        if 'ma' in signals.columns:
            chart_data['ema'] = signals['ma'].tolist()
    elif strategy_type == 'double':
        if 'ma_short' in signals.columns:
            chart_data['ema_short'] = signals['ma_short'].tolist()
        if 'ma_long' in signals.columns:
            chart_data['ema_long'] = signals['ma_long'].tolist()
    elif strategy_type == 'triple':
        if 'ma1' in signals.columns:
            chart_data['ema1'] = signals['ma1'].tolist()
        if 'ma2' in signals.columns:
            chart_data['ema2'] = signals['ma2'].tolist()
        if 'ma3' in signals.columns:
            chart_data['ema3'] = signals['ma3'].tolist()
    
    # Add buy/sell signal points
    buy_signals = signals[signals['signal'] == 'long']
    sell_signals = signals[signals['signal'] == 'short']
    
    if not buy_signals.empty:
        chart_data['buy_signals'] = {
            'dates': buy_signals.index.strftime('%Y-%m-%d').tolist(),
            'prices': buy_signals['price'].tolist()
        }
    
    if not sell_signals.empty:
        chart_data['sell_signals'] = {
            'dates': sell_signals.index.strftime('%Y-%m-%d').tolist(),
            'prices': sell_signals['price'].tolist()
        }
    
    return chart_data


def get_strategy_summary(ticker: str, strategy_type: str, parameters: Dict) -> str:
    """
    Generate a human-readable summary of the strategy configuration.
    
    Args:
        ticker: Stock ticker
        strategy_type: Strategy type
        parameters: Strategy parameters
    
    Returns:
        Strategy summary string
    """
    if strategy_type == 'single':
        return f"Single EMA({parameters.get('ema_window', 20)}) strategy for {ticker}"
    elif strategy_type == 'double':
        return f"Double EMA({parameters.get('short_window', 12)}, {parameters.get('long_window', 26)}) crossover strategy for {ticker}"
    elif strategy_type == 'triple':
        return f"Triple EMA({parameters.get('window1', 5)}, {parameters.get('window2', 12)}, {parameters.get('window3', 26)}) strategy for {ticker}"
    else:
        return f"Unknown strategy for {ticker}"


if __name__ == "__main__":
    """
    Main function to test the moving average strategy
    """
    print("🚀 Testing Moving Average Strategy")
    print("=" * 50)
    
    # Test parameters
    test_cases = [
        {
            'ticker': 'AAPL',
            'strategy_type': 'single',
            'parameters': {'ema_window': 20},
            'period': '6mo'
        },
        {
            'ticker': 'AAPL',
            'strategy_type': 'double',
            'parameters': {'short_window': 12, 'long_window': 26},
            'period': '6mo'
        },
        {
            'ticker': 'MSFT',
            'strategy_type': 'triple',
            'parameters': {'window1': 5, 'window2': 12, 'window3': 26},
            'period': '6mo'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📊 Test Case {i}: {test_case['strategy_type'].upper()} EMA Strategy")
        print(f"Ticker: {test_case['ticker']}")
        print(f"Parameters: {test_case['parameters']}")
        print(f"Period: {test_case['period']}")
        print("-" * 30)
        
        try:
            result = run_moving_average_strategy(
                test_case['ticker'],
                test_case['strategy_type'],
                test_case['parameters'],
                test_case['period']
            )
            
            if result.get('success'):
                print("✅ Strategy executed successfully!")
                print(f"Summary: {get_strategy_summary(test_case['ticker'], test_case['strategy_type'], test_case['parameters'])}")
                print(f"Current Signal: {result['current_signal'].upper()}")
                
                performance = result.get('performance', {})
                print(f"Total Return: {performance.get('total_return', 0)}%")
                print(f"Win Rate: {performance.get('win_rate', 0)}%")
                print(f"Total Signals: {performance.get('total_signals', 0)}")
                print(f"Long Signals: {performance.get('long_signals', 0)}")
                print(f"Short Signals: {performance.get('short_signals', 0)}")
                
            else:
                print("❌ Strategy failed!")
                print(f"Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception occurred: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("🏁 Testing completed!")


class MovingAverageStrategy(BaseStrategy):
    """Moving Average Strategy implementation following the BaseStrategy interface"""
    
    def __init__(self):
        super().__init__("moving_average")
    
    def get_default_configurations(self) -> Dict:
        """Return default configurations for all MA strategy variants"""
        return {
            'single': {'ema_window': 20},
            'double': {'short_window': 12, 'long_window': 26},
            'triple': {'window1': 5, 'window2': 12, 'window3': 26}
        }
    
    def run_strategy(self, ticker: str, strategy_type: str, parameters: Dict, 
                    period: str = "1y", prices_df: pd.DataFrame = None) -> Dict:
        """Run moving average strategy using the standalone function"""
        return run_moving_average_strategy(ticker, strategy_type, parameters, period, prices_df)
    
    def get_strategy_summary(self, ticker: str, strategy_type: str, parameters: Dict) -> str:
        """Generate summary using the standalone function"""
        return get_strategy_summary(ticker, strategy_type, parameters)

