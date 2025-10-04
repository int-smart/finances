"""
Dataset utilities for S&P 500 stock data processing and neural network training.
"""

import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import yfinance as yf
import time


def load_stock_data(pickle_path, min_date="2015-01-01", features=None):
    """
    Load and process stock data from pickle file with date filtering.
    
    Args:
        pickle_path (str): Path to the pickle file
        min_date (str): Minimum date to include data from (YYYY-MM-DD)
        features (list): List of feature column names
    
    Returns:
        dict: Dictionary of ticker -> DataFrame with processed data
    """
    if features is None:
        features = ["Open", "High", "Low", "Close", "Volume"]
    
    print(f"Loading data from: {pickle_path}")
    print(f"Filtering to data from {min_date} onwards")
    
    # Load raw data
    with open(pickle_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    # Process and filter data
    data = {}
    for ticker, df in raw_data.items():
        if not isinstance(df, pd.DataFrame):
            continue
        
        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col) for col in df.columns]
        
        # Create column mapping
        cols_map = {}
        for col in df.columns:
            col_str = str(col).lower()
            if 'open' in col_str:
                cols_map['open'] = col
            elif 'high' in col_str:
                cols_map['high'] = col
            elif 'low' in col_str:
                cols_map['low'] = col
            elif 'close' in col_str:
                cols_map['close'] = col
            elif 'volume' in col_str:
                cols_map['volume'] = col
        
        # Check required columns
        required = ["open", "high", "low", "close", "volume"]
        if len(cols_map) < 5:
            continue
        
        # Select and rename columns
        df = df[[cols_map[c] for c in required]].copy()
        df.columns = features
        
        # Clean data
        df = df.dropna().sort_index()
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        df = df[df["Volume"] > 0]
        
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                continue
        
        # Filter by minimum date
        df = df[df.index >= min_date]
        
        if len(df) < 64 + 1:  # Minimum window size + prediction horizon
            continue
        
        data[ticker] = df
    
    print(f"Loaded {len(data)} tickers with valid data from {min_date} onwards")
    if len(data) > 0:
        sample_ticker = list(data.keys())[0]
        print(f"Sample data shape for {sample_ticker}: {data[sample_ticker].shape}")
        print(f"Date range: {data[sample_ticker].index[0]} to {data[sample_ticker].index[-1]}")
    
    return data


def fetch_fresh_data(tickers, start_date="1950-01-01", end_date=None):
    """
    Fetch fresh historical data for all tickers from yfinance.
    
    Args:
        tickers (list): List of ticker symbols
        start_date (str): Start date for data fetching
        end_date (str): End date for data fetching (default: today)
    
    Returns:
        tuple: (fresh_data_dict, failed_tickers_list)
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    fresh_data = {}
    failed_tickers = []
    
    print(f"Fetching fresh data for {len(tickers)} tickers from {start_date} to {end_date}")
    
    for i, ticker in enumerate(tickers):
        try:
            print(f"Fetching {ticker} ({i+1}/{len(tickers)})...", end=" ")
            
            # Download data with auto_adjust=True for split/dividend adjustments
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, auto_adjust=True)
            
            if df.empty or len(df) < 65:  # Minimum required data points
                print("Insufficient data")
                failed_tickers.append(ticker)
                continue
            
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                print("Missing required columns")
                failed_tickers.append(ticker)
                continue
            
            # Clean the data
            df = df[required_cols].copy()
            df = df.dropna()
            df = df[df['Volume'] > 0]  # Remove zero volume days
            
            if len(df) < 65:
                print("Insufficient data after cleaning")
                failed_tickers.append(ticker)
                continue
            
            fresh_data[ticker] = df
            print(f"✓ ({len(df)} records)")
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            failed_tickers.append(ticker)
    
    print(f"\nSuccessfully fetched data for {len(fresh_data)} tickers")
    if failed_tickers:
        print(f"Failed to fetch data for {len(failed_tickers)} tickers: {failed_tickers[:10]}{'...' if len(failed_tickers) > 10 else ''}")
    
    return fresh_data, failed_tickers


class MultiStockWindowDataset(Dataset):
    """
    Dataset for multi-stock time series with uniform ticker sampling.
    """
    
    def __init__(self, data_dict, window_size=64, pred_horizon=1, tickers=None, normalize=True, normalization_type='relative'):
        """
        Initialize the dataset.
        
        Args:
            data_dict (dict): Dictionary of ticker -> DataFrame
            window_size (int): Number of timesteps per sample
            pred_horizon (int): Number of steps ahead to predict
            tickers (list): List of tickers to use (default: all)
            normalize (bool): Whether to normalize data
            normalization_type (str): Type of normalization ('relative' or 'zscore')
        """
        self.data_dict = data_dict
        self.window_size = window_size
        self.pred_horizon = pred_horizon
        self.tickers = tickers if tickers is not None else list(data_dict.keys())
        self.normalize = normalize
        self.normalization_type = normalization_type
        
        # Estimate length as sum of possible start positions across tickers
        self.length = int(sum(max(0, len(self.data_dict[t]) - (self.window_size + self.pred_horizon)) for t in self.tickers))
        self.rng = np.random.default_rng(42)
    
    def __len__(self):
        # We use a large virtual length to allow random sampling
        return max(self.length, 10000)
    
    def __getitem__(self, idx):
        # Sample ticker uniformly
        ticker = random.choice(self.tickers)
        df = self.data_dict[ticker]
        max_start = len(df) - (self.window_size + self.pred_horizon)
        start = self.rng.integers(0, max_start)
        end = start + self.window_size
        
        window = df.iloc[start:end].values.astype(np.float32)
        
        # Get future targets: end+1 to end+pred_horizon
        target_start = end  # end+1 in 0-indexed
        target_end = end + self.pred_horizon
        
        if self.pred_horizon == 1:
            target_close = df.iloc[target_start]["Close"].astype(np.float32)
        else:
            # Get multiple future close prices
            target_close = df.iloc[target_start:target_end]["Close"].values.astype(np.float32)
        
        if self.normalize:
            if self.normalization_type == 'relative':
                # Relative normalization: normalize each feature by its own last value
                # This preserves the relative relationships within each feature
                # Open/High/Low/Close are normalized by last Close price
                # Volume is normalized by last Volume (different scale)
                last_values = window[-1, :]  # Last values for each feature
                # Avoid division by zero
                last_values = np.where(last_values == 0, 1.0, last_values)
                window = window / last_values
                
                # For target, use the Close price normalization
                close_col_idx = 3  # Close is at index 3
                last_close_price = last_values[close_col_idx]
                target_close = target_close / last_close_price
                
            elif self.normalization_type == 'zscore':
                # Z-score normalization using window mean and std
                mean = np.mean(window, axis=0)
                std = np.std(window, axis=0)
                # Avoid division by zero
                std = np.where(std == 0, 1.0, std)
                # Normalize the window
                window = (window - mean) / std
                
                # Normalize target_close using the same scaling parameters
                close_col_idx = 3  # Close is at index 3
                target_mean = mean[close_col_idx]
                target_std = std[close_col_idx]
                target_close = (target_close - target_mean) / target_std
        
        # Channels-first for CNN: (C, T)
        window = window.T  # (5, window_size)
        x = torch.from_numpy(window)
        y = torch.tensor(target_close, dtype=torch.float32)
        return x, y, ticker


def create_data_loaders(data, train_split=0.8, batch_size=128, window_size=64, pred_horizon=1, num_workers=2):
    """
    Create train and validation data loaders.
    
    Args:
        data (dict): Dictionary of ticker -> DataFrame
        train_split (float): Fraction of tickers for training
        batch_size (int): Batch size for data loaders
        window_size (int): Window size for samples
        pred_horizon (int): Prediction horizon
        num_workers (int): Number of worker processes
    
    Returns:
        tuple: (train_loader, val_loader, train_tickers, val_tickers)
    """
    # Split tickers for train/val
    tickers = list(data.keys())
    random.shuffle(tickers)
    split = int(train_split * len(tickers))
    train_tickers, val_tickers = tickers[:split], tickers[split:]
    
    # Create datasets
    train_ds = MultiStockWindowDataset(data, window_size=window_size, pred_horizon=pred_horizon, tickers=train_tickers)
    val_ds = MultiStockWindowDataset(data, window_size=window_size, pred_horizon=pred_horizon, tickers=val_tickers)
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, train_tickers, val_tickers


def update_pickle_with_fresh_data(original_pickle_path, start_date="2010-01-01"):
    """
    Update pickle file with fresh data from yfinance.
    
    Args:
        original_pickle_path (str): Path to original pickle file
        start_date (str): Start date for fresh data
    
    Returns:
        str: Path to the fresh pickle file
    """
    # Load original data to get tickers
    with open(original_pickle_path, 'rb') as f:
        raw_data = pickle.load(f)
    
    all_tickers = list(raw_data.keys())
    print(f"Found {len(all_tickers)} tickers in original pickle file")
    
    # Fetch fresh data
    fresh_data, failed_tickers = fetch_fresh_data(all_tickers, start_date=start_date)
    
    # Save fresh data
    fresh_pickle_path = original_pickle_path.replace('.pkl', '_fresh.pkl')
    with open(fresh_pickle_path, 'wb') as f:
        pickle.dump(fresh_data, f)
    
    print(f"\nFresh data saved to: {fresh_pickle_path}")
    print(f"File size: {os.path.getsize(fresh_pickle_path) / (1024*1024):.2f} MB")
    
    return fresh_pickle_path
