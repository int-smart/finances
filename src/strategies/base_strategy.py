"""
Base Strategy Interface

This module defines the base interface for all trading strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def get_default_configurations(self) -> Dict:
        """Return default configurations for all strategy variants"""
        pass
    
    @abstractmethod
    def run_strategy(self, ticker: str, strategy_type: str, parameters: Dict, 
                    period: str = "1y", prices_df: pd.DataFrame = None) -> Dict:
        """
        Run the strategy for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            strategy_type: Strategy variant (e.g., 'single', 'double', 'triple')
            parameters: Strategy parameters
            period: Data period (only used if prices_df is None)
            prices_df: Optional pre-loaded price data
        
        Returns:
            Dictionary with strategy results
        """
        pass
    
    @abstractmethod
    def get_strategy_summary(self, ticker: str, strategy_type: str, parameters: Dict) -> str:
        """Generate a human-readable summary of the strategy configuration"""
        pass
    
    def get_strategy_types(self) -> List[str]:
        """Get list of available strategy types for this strategy"""
        return list(self.get_default_configurations().keys())