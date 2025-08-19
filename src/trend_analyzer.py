import pandas as pd
import numpy as np

class TrendAnalyzer:
    def __init__(self, stock_data):
        """
        Initialize TrendAnalyzer with stock data
        
        Args:
            stock_data: DataFrame with columns ['Open', 'High', 'Low', 'Close', 'Volume']
        """
        self.data = stock_data.copy()
        self.patterns = {}
        
    def analyze_patterns(self):
        """
        Analyze all candlestick patterns and return results
        
        Returns:
            dict: Dictionary with pattern names as keys and boolean series as values
        """
        self.patterns = {
            'hammer': self.detect_hammer(),
            'doji': self.detect_doji(),
            'engulfing_bullish': self.detect_bullish_engulfing(),
            'engulfing_bearish': self.detect_bearish_engulfing(),
            'spinning_top': self.detect_spinning_top(),
            'hanging_man': self.detect_hanging_man(),
            'shooting_star': self.detect_shooting_star(),
            'marubozu_bullish': self.detect_bullish_marubozu(),
            'marubozu_bearish': self.detect_bearish_marubozu()
        }
        
        return self.patterns
    
    def detect_hammer(self):
        """Detect Hammer pattern"""
        body = abs(self.data['Close'] - self.data['Open'])
        lower_shadow = self.data['Low'] - np.minimum(self.data['Open'], self.data['Close'])
        upper_shadow = self.data['High'] - np.maximum(self.data['Open'], self.data['Close'])
        
        return (
            (lower_shadow >= 2 * body) &
            (upper_shadow <= 0.1 * body) &
            (body > 0)
        )
    
    def detect_doji(self):
        """Detect Doji pattern"""
        body = abs(self.data['Close'] - self.data['Open'])
        total_range = self.data['High'] - self.data['Low']
        
        return body <= 0.1 * total_range
    
    def detect_bullish_engulfing(self):
        """Detect Bullish Engulfing pattern"""
        prev_open = self.data['Open'].shift(1)
        prev_close = self.data['Close'].shift(1)
        
        return (
            (prev_close < prev_open) &  # Previous candle is bearish
            (self.data['Close'] > self.data['Open']) &  # Current candle is bullish
            (self.data['Open'] < prev_close) &  # Current open < previous close
            (self.data['Close'] > prev_open)  # Current close > previous open
        )
    
    def detect_bearish_engulfing(self):
        """Detect Bearish Engulfing pattern"""
        prev_open = self.data['Open'].shift(1)
        prev_close = self.data['Close'].shift(1)
        
        return (
            (prev_close > prev_open) &  # Previous candle is bullish
            (self.data['Close'] < self.data['Open']) &  # Current candle is bearish
            (self.data['Open'] > prev_close) &  # Current open > previous close
            (self.data['Close'] < prev_open)  # Current close < previous open
        )
    
    def detect_spinning_top(self):
        """Detect Spinning Top pattern"""
        body = abs(self.data['Close'] - self.data['Open'])
        upper_shadow = self.data['High'] - np.maximum(self.data['Open'], self.data['Close'])
        lower_shadow = np.minimum(self.data['Open'], self.data['Close']) - self.data['Low']
        
        return (
            (upper_shadow >= body) &
            (lower_shadow >= body) &
            (body > 0)
        )
    
    def detect_hanging_man(self):
        """Detect Hanging Man pattern"""
        body = abs(self.data['Close'] - self.data['Open'])
        lower_shadow = np.minimum(self.data['Open'], self.data['Close']) - self.data['Low']
        upper_shadow = self.data['High'] - np.maximum(self.data['Open'], self.data['Close'])
        
        return (
            (lower_shadow >= 2 * body) &
            (upper_shadow <= 0.1 * body) &
            (self.data['Close'] < self.data['Open'])  # Bearish candle
        )
    
    def detect_shooting_star(self):
        """Detect Shooting Star pattern"""
        body = abs(self.data['Close'] - self.data['Open'])
        upper_shadow = self.data['High'] - np.maximum(self.data['Open'], self.data['Close'])
        lower_shadow = np.minimum(self.data['Open'], self.data['Close']) - self.data['Low']
        
        return (
            (upper_shadow >= 2 * body) &
            (lower_shadow <= 0.1 * body) &
            (body > 0)
        )
    
    def detect_bullish_marubozu(self):
        """Detect Bullish Marubozu pattern"""
        return (
            (self.data['Close'] > self.data['Open']) &
            (self.data['High'] == self.data['Close']) &
            (self.data['Low'] == self.data['Open'])
        )
    
    def detect_bearish_marubozu(self):
        """Detect Bearish Marubozu pattern"""
        return (
            (self.data['Close'] < self.data['Open']) &
            (self.data['High'] == self.data['Open']) &
            (self.data['Low'] == self.data['Close'])
        )
    
    def get_latest_patterns(self):
        """
        Get the latest patterns detected in the most recent candle
        
        Returns:
            dict: Dictionary with pattern names and boolean values for latest candle
        """
        if not self.patterns:
            self.analyze_patterns()
        
        latest_patterns = {}
        for pattern_name, pattern_series in self.patterns.items():
            if len(pattern_series) > 0:
                # Convert numpy boolean to native Python boolean
                latest_patterns[pattern_name] = bool(pattern_series.iloc[-1])
            else:
                latest_patterns[pattern_name] = False
        
        return latest_patterns
    
    def get_pattern_dates(self, pattern_name):
        """
        Get all dates where a specific pattern occurred
        
        Args:
            pattern_name: Name of the pattern to search for
            
        Returns:
            list: List of dates where the pattern occurred
        """
        if pattern_name not in self.patterns:
            return []
        
        pattern_series = self.patterns[pattern_name]
        return self.data.index[pattern_series].tolist()
    
    def count_recent_patterns(self, days=10):
        """
        Count pattern occurrences in recent days
        
        Args:
            days: Number of recent days to analyze
            
        Returns:
            dict: Dictionary with pattern names and counts
        """
        if not self.patterns:
            self.analyze_patterns()
        
        recent_patterns = {}
        for pattern_name, pattern_series in self.patterns.items():
            if len(pattern_series) >= days:
                # Convert numpy int64 to native Python int
                recent_count = int(pattern_series.iloc[-days:].sum())
                recent_patterns[pattern_name] = recent_count
            else:
                recent_patterns[pattern_name] = 0
        
        return recent_patterns