"""
Strategy Manager

This module manages all trading strategies and their execution.
It handles both scheduled strategy runs and on-demand calculations.
"""

import pandas as pd
import numpy as np
import os
import pickle
from typing import Dict, List, Optional
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.strategies.moving_average_strategy import convert_stock_data_to_prices_df, MovingAverageStrategy


class StrategyManager:
    """Manages all trading strategies and their execution"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.strategies_dir = os.path.join(data_dir, "strategies")
        os.makedirs(self.strategies_dir, exist_ok=True)
        
        # File paths
        self.historical_file = os.path.join(self.data_dir, "strategies_data.pkl")
        self.latest_file = os.path.join(self.strategies_dir, "strategies_latest.pkl")
        
        # Initialize strategies
        self.ma_strategy = MovingAverageStrategy()
        
        # Get default strategy configurations
        self.default_strategies = {
            'moving_average': self.ma_strategy.get_default_configurations()
        }
    
    def run_all_strategies_for_all_tickers(self, tickers: List[str], stock_data: Dict = None) -> Dict:
        """
        Run all default strategies for all tickers using existing stock data or fetching new data.
        
        Args:
            tickers: List of ticker symbols
            stock_data: Pre-loaded stock data from scheduler (optional)
        
        Returns:
            Dictionary with all strategy results
        """
        print(f"Running strategies for {len(tickers)} tickers...")
        
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'strategies': {}
        }
        
        # Run moving average strategies
        ma_results = self._run_moving_average_strategies(tickers, stock_data)
        all_results['strategies']['moving_average'] = ma_results
        
        # Save results
        self._save_strategy_results(all_results)
        
        print(f"✅ All strategies completed for {len(tickers)} tickers")
        return all_results
    
    def _run_moving_average_strategies(self, tickers: List[str], stock_data: Dict = None) -> Dict:
        """Run all moving average strategy variants for all tickers"""
        print("📊 Running Moving Average strategies...")
        
        ma_results = {}
        
        for strategy_type in self.ma_strategy.get_strategy_types():
            print(f"  Running {strategy_type} EMA strategy...")
            ma_results[strategy_type] = {}
            
            for ticker in tickers:
                try:
                    # Get parameters for this strategy type
                    params = self.default_strategies['moving_average'][strategy_type].copy()
                    
                    # Use pre-loaded stock data if available, otherwise fetch
                    if stock_data and self._has_stock_data(stock_data, ticker):
                        # Convert stock data to prices DataFrame
                        prices_df = convert_stock_data_to_prices_df(ticker, stock_data)
                        if prices_df.empty:
                            result = {'error': 'No price data available or data conversion failed'}
                        else:
                            result = self.ma_strategy.run_strategy(ticker, strategy_type, params, '1y', prices_df)
                    else:
                        result = self.ma_strategy.run_strategy(ticker, strategy_type, params, '1y')
                    
                    if result.get('success'):
                        ma_results[strategy_type][ticker] = {
                            'success': True,
                            'current_signal': result['current_signal'],
                            'summary': self.ma_strategy.get_strategy_summary(ticker, strategy_type, params),
                            'parameters': params,
                            'chart_data': result.get('chart_data', {}),
                            'timestamp': datetime.now().isoformat()
                        }
                        print(f"    ✅ {ticker}: {result['current_signal']}")
                    else:
                        ma_results[strategy_type][ticker] = {
                            'success': False,
                            'error': result.get('error', 'Unknown error'),
                            'timestamp': datetime.now().isoformat()
                        }
                        print(f"    ❌ {ticker}: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    ma_results[strategy_type][ticker] = {
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    print(f"    ❌ {ticker}: {str(e)}")
        
        return ma_results
    
    def _has_stock_data(self, stock_data: Dict, ticker: str) -> bool:
        """Check if stock data contains the required ticker data"""
        return (stock_data and 
                'stocks' in stock_data and 
                ticker in stock_data['stocks'] and 
                'history' in stock_data['stocks'][ticker])
    
    def _save_strategy_results(self, results: Dict):
        """Save strategy results to disk with historical data preservation"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Load existing historical data using the existing method
        historical_data = self.load_historical_strategy_data()
        
        # Add today's results to historical data
        historical_data[current_date] = results
        
        # Save updated historical data
        with open(self.historical_file, 'wb') as f:
            pickle.dump(historical_data, f)
        
        # Also save latest results for quick API access
        with open(self.latest_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"💾 Strategy results saved to historical file with date {current_date}")
        print(f"📊 Total dates in history: {len(historical_data)}")
    
    def load_latest_strategy_results(self) -> Optional[Dict]:
        """Load the latest strategy results from historical data"""
        historical_data = self.load_historical_strategy_data()
        
        if not historical_data:
            return None
        
        # Filter to only date keys (YYYY-MM-DD format)
        date_keys = [key for key in historical_data.keys() if self._is_valid_date_key(key)]
        
        if not date_keys:
            return None
        
        # Get the most recent date
        latest_date = max(date_keys)
        return historical_data[latest_date]
    
    def load_historical_strategy_data(self) -> Dict:
        """Load all historical strategy data"""
        if os.path.exists(self.historical_file):
            try:
                with open(self.historical_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading historical strategy data: {e}")
        
        return {}
    
    def get_strategy_data_for_date(self, date: str) -> Optional[Dict]:
        """Get strategy data for a specific date"""
        historical_data = self.load_historical_strategy_data()
        return historical_data.get(date, None)
    
    def get_available_strategy_dates(self) -> List[str]:
        """Get list of available strategy data dates"""
        historical_data = self.load_historical_strategy_data()
        # Filter to only valid date keys
        date_keys = [key for key in historical_data.keys() if self._is_valid_date_key(key)]
        return sorted(date_keys, reverse=True)
    
    def _is_valid_date_key(self, key: str) -> bool:
        """Check if a key is a valid date in YYYY-MM-DD format"""
        import re
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        return bool(re.match(pattern, key))
    
    def calculate_strategy_on_demand(self, ticker: str, strategy_name: str, strategy_type: str, params: Dict, period: str = "1y") -> Dict:
        """
        Calculate a strategy on-demand with custom parameters.
        This is used by the calculator API endpoint.
        
        Args:
            ticker: Stock ticker
            strategy_name: Name of strategy (e.g., 'moving_average')
            strategy_type: Type/variant of strategy (e.g., 'double')
            params: Custom parameters
            period: Time period for data
        
        Returns:
            Strategy result dictionary
        """
        print(f"🔄 Calculating {strategy_name} ({strategy_type}) for {ticker} with custom parameters")
        
        if strategy_name == 'moving_average':
            result = self.ma_strategy.run_strategy(ticker, strategy_type, params, period)
            
            if result.get('success'):
                result['summary'] = self.ma_strategy.get_strategy_summary(ticker, strategy_type, params)
                result['calculated_on_demand'] = True
                result['calculation_timestamp'] = datetime.now().isoformat()
            
            return result
        else:
            return {'error': f'Strategy {strategy_name} not implemented'}
    
    def get_strategy_data_for_ticker(self, ticker: str, strategy_name: str = None, strategy_type: str = None) -> Dict:
        """
        Get saved strategy data for a specific ticker.
        
        Args:
            ticker: Stock ticker
            strategy_name: Optional strategy name filter
            strategy_type: Optional strategy type filter
        
        Returns:
            Strategy data for the ticker
        """
        results = self.load_latest_strategy_results()
        
        if not results:
            return {'error': 'No strategy data available'}
        
        ticker_data = {}
        
        if strategy_name:
            # Get specific strategy
            if (strategy_name in results['strategies'] and 
                strategy_type in results['strategies'][strategy_name] and 
                ticker in results['strategies'][strategy_name][strategy_type]):
                ticker_data = results['strategies'][strategy_name][strategy_type][ticker]
        else:
            # Get all strategies for ticker
            for strat_name, strat_data in results['strategies'].items():
                ticker_data[strat_name] = {}
                for strat_type, type_data in strat_data.items():
                    if ticker in type_data:
                        ticker_data[strat_name][strat_type] = type_data[ticker]
        
        return ticker_data if ticker_data else {'error': f'No strategy data found for {ticker}'}
    
    def get_all_current_signals(self) -> Dict:
        """Get current signals for all tickers from all strategies"""
        results = self.load_latest_strategy_results()
        
        if not results:
            return {}
        
        signals = {}
        
        for strategy_name, strategy_data in results['strategies'].items():
            signals[strategy_name] = {}
            for strategy_type, type_data in strategy_data.items():
                signals[strategy_name][strategy_type] = {}
                for ticker, ticker_data in type_data.items():
                    if ticker_data.get('success'):
                        signals[strategy_name][strategy_type][ticker] = ticker_data.get('current_signal', 'hold')
        
        return signals


if __name__ == "__main__":
    """Test the strategy manager"""
    print("🧪 Testing Strategy Manager")
    print("=" * 50)
    
    # Test with a few tickers
    test_tickers = ['AAPL', 'MSFT']
    
    manager = StrategyManager()
    
    # Run all strategies
    results = manager.run_all_strategies_for_all_tickers(test_tickers)
    
    print("\n📊 Results Summary:")
    for strategy_name, strategy_data in results['strategies'].items():
        print(f"\n{strategy_name.upper()} Strategy:")
        for strategy_type, type_data in strategy_data.items():
            print(f"  {strategy_type}:")
            for ticker, ticker_data in type_data.items():
                if ticker_data.get('success'):
                    signal = ticker_data.get('current_signal', 'N/A')
                    print(f"    {ticker}: {signal}")
                else:
                    print(f"    {ticker}: ERROR - {ticker_data.get('error', 'Unknown')}")
    
    # Test on-demand calculation
    print("\n🔄 Testing on-demand calculation:")
    custom_result = manager.calculate_strategy_on_demand(
        'AAPL', 'moving_average', 'double', 
        {'short_window': 10, 'long_window': 30}, '6mo'
    )
    
    if custom_result.get('success'):
        print(f"✅ Custom calculation successful: {custom_result.get('current_signal')}")
    else:
        print(f"❌ Custom calculation failed: {custom_result.get('error')}")
    
    print("\n🏁 Strategy Manager test completed!")
