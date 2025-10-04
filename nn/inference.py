#!/usr/bin/env python3
"""
Inference script for stock prediction model.
Loads checkpoint, makes predictions on new data, and denormalizes output.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional

# Import model and utilities
from model import create_model, create_unified_model
from dataset import load_stock_data


class StockPredictor:
    """
    Stock price predictor that loads checkpoints and makes predictions.
    Handles all model types: regression, quantile, and probabilistic.
    """
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Initialize predictor with a trained checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pt)
            device: Device to run inference on ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.checkpoint_path = checkpoint_path
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        self._load_checkpoint()
        
    def _load_checkpoint(self):
        """Load model checkpoint and configuration."""
        print(f"Loading checkpoint from: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract configuration
        self.config = checkpoint['config']
        self.model_type = self.config.get('model_type', 'regression')
        self.window_size = self.config.get('window_size', 64)
        self.pred_horizon = self.config.get('pred_horizon', 7)
        self.quantile_levels = self.config.get('quantile_levels', [0.1, 0.5, 0.9])
        
        print("Model configuration:")
        print(f"  Model type: {self.model_type}")
        print(f"  Window size: {self.window_size}")
        print(f"  Prediction horizon: {self.pred_horizon}")
        print(f"  Quantile levels: {self.quantile_levels}")
        
        # Create model based on type
        if self.model_type != 'regression' or 'quantile' in self.model_type or 'prob' in self.model_type:
            # Unified model for multi-objective training
            self.model = create_unified_model(self.config, self.device)
        else:
            # Simple model for regression only
            model_config = {
                'seq_len': self.window_size,
                'pred_horizon': self.pred_horizon,
                'embed_dim': self.config.get('embed_dim', 32),
                'num_heads': self.config.get('num_heads', 4),
                'num_layers': self.config.get('num_layers', 2)
            }
            self.model = create_model(model_config, self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded successfully (epoch {checkpoint['epoch']})")
        print(f"  Training loss: {checkpoint['train_loss']:.6f}")
        print(f"  Validation loss: {checkpoint['val_loss']:.6f}")
        
    def prepare_data(self, data: pd.DataFrame, normalization_type: str = 'relative') -> Tuple[np.ndarray, Dict]:
        """
        Prepare data for inference (apply normalization).
        
        Args:
            data: DataFrame with OHLCV columns, shape (window_size, 5)
            normalization_type: 'relative' or 'zscore'
        
        Returns:
            Tuple of (normalized_data, normalization_params)
        """
        if len(data) != self.window_size:
            raise ValueError(f"Data must have exactly {self.window_size} timesteps, got {len(data)}")
        
        # Extract required columns in correct order
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        window = data[required_cols].values.astype(np.float32)
        
        # Normalize based on type
        normalization_params = {
            'type': normalization_type,
            'close_col_idx': 3  # Close is at index 3
        }
        
        if normalization_type == 'relative':
            # Relative normalization: normalize by last values
            last_values = window[-1, :]
            last_values = np.where(last_values == 0, 1.0, last_values)  # Avoid division by zero
            
            normalized_window = window / last_values
            
            # Store denormalization parameters for Close price
            normalization_params['last_close_price'] = last_values[normalization_params['close_col_idx']]
            
        elif normalization_type == 'zscore':
            # Z-score normalization using window statistics
            mean = np.mean(window, axis=0)
            std = np.std(window, axis=0)
            std = np.where(std == 0, 1.0, std)  # Avoid division by zero
            
            normalized_window = (window - mean) / std
            
            # Store denormalization parameters for Close price
            close_col_idx = normalization_params['close_col_idx']
            normalization_params['close_mean'] = mean[close_col_idx]
            normalization_params['close_std'] = std[close_col_idx]
        
        else:
            raise ValueError(f"Unknown normalization type: {normalization_type}")
        
        return normalized_window, normalization_params
    
    def denormalize_predictions(self, predictions: Dict[str, np.ndarray], 
                                normalization_params: Dict) -> Dict[str, np.ndarray]:
        """
        Denormalize predictions back to original scale.
        
        Args:
            predictions: Dictionary of prediction outputs
            normalization_params: Parameters used for normalization
        
        Returns:
            Dictionary of denormalized predictions
        """
        denormalized = {}
        norm_type = normalization_params['type']
        
        for key, value in predictions.items():
            if norm_type == 'relative':
                # Multiply by last close price
                last_close = normalization_params['last_close_price']
                denormalized[key] = value * last_close
                
            elif norm_type == 'zscore':
                # Reverse z-score normalization
                close_mean = normalization_params['close_mean']
                close_std = normalization_params['close_std']
                denormalized[key] = value * close_std + close_mean
        
        return denormalized
    
    @torch.no_grad()
    def predict(self, data: pd.DataFrame, normalization_type: str = 'relative') -> Dict[str, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            data: DataFrame with OHLCV columns and DatetimeIndex
            normalization_type: 'relative' or 'zscore'
        
        Returns:
            Dictionary with prediction results (denormalized):
                - 'regression': Point predictions [pred_horizon]
                - 'quantiles': Quantile predictions [pred_horizon, num_quantiles]
                - 'prob_mean': Probabilistic mean [pred_horizon]
                - 'prob_std': Probabilistic std deviation [pred_horizon]
                - 'prob_var': Probabilistic variance [pred_horizon]
        """
        # Prepare data
        normalized_window, norm_params = self.prepare_data(data, normalization_type)
        
        # Convert to tensor (channels-first: C, T)
        x = torch.from_numpy(normalized_window.T).unsqueeze(0).to(self.device)  # (1, C, T)
        
        # Make predictions
        self.model.eval()
        outputs = self.model(x)
        
        # Process outputs based on model type
        predictions_normalized = {}
        
        if isinstance(outputs, dict):
            # Unified model - multiple outputs
            for key, value in outputs.items():
                predictions_normalized[key] = value.cpu().numpy().squeeze()
            
            # Convert variance to std for easier interpretation
            if 'prob_var' in predictions_normalized:
                predictions_normalized['prob_std'] = np.sqrt(predictions_normalized['prob_var'])
        else:
            # Simple regression model
            predictions_normalized['regression'] = outputs.cpu().numpy().squeeze()
        
        # Denormalize predictions
        predictions = self.denormalize_predictions(predictions_normalized, norm_params)
        
        return predictions
    
    def predict_from_ticker_data(self, ticker: str, data_dict: Dict[str, pd.DataFrame], 
                                 normalization_type: str = 'relative',
                                 use_last_window: bool = True) -> Dict[str, np.ndarray]:
        """
        Make predictions for a specific ticker from a data dictionary.
        
        Args:
            ticker: Stock ticker symbol
            data_dict: Dictionary of ticker -> DataFrame
            normalization_type: 'relative' or 'zscore'
            use_last_window: If True, use last window_size rows; otherwise random sample
        
        Returns:
            Dictionary with prediction results
        """
        if ticker not in data_dict:
            raise ValueError(f"Ticker {ticker} not found in data")
        
        df = data_dict[ticker]
        
        if len(df) < self.window_size:
            raise ValueError(f"Not enough data for {ticker}: {len(df)} < {self.window_size}")
        
        # Get window
        if use_last_window:
            window_data = df.iloc[-self.window_size:]
        else:
            # Random window
            max_start = len(df) - self.window_size
            start_idx = np.random.randint(0, max_start)
            window_data = df.iloc[start_idx:start_idx + self.window_size]
        
        # Make predictions
        predictions = self.predict(window_data, normalization_type)
        
        return predictions
    
    def batch_predict(self, data_list: List[pd.DataFrame], 
                     normalization_type: str = 'relative',
                     batch_size: int = 32) -> List[Dict[str, np.ndarray]]:
        """
        Make predictions on a batch of data samples.
        
        Args:
            data_list: List of DataFrames, each with window_size rows
            normalization_type: 'relative' or 'zscore'
            batch_size: Batch size for inference
        
        Returns:
            List of prediction dictionaries
        """
        all_predictions = []
        
        # Process in batches
        for i in range(0, len(data_list), batch_size):
            batch_data = data_list[i:i + batch_size]
            
            # Prepare batch
            normalized_windows = []
            norm_params_list = []
            
            for data in batch_data:
                norm_window, norm_params = self.prepare_data(data, normalization_type)
                normalized_windows.append(norm_window.T)  # (C, T)
                norm_params_list.append(norm_params)
            
            # Stack into batch tensor
            x_batch = torch.from_numpy(np.stack(normalized_windows)).to(self.device)  # (B, C, T)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(x_batch)
            
            # Process each sample in batch
            for j in range(len(batch_data)):
                predictions_normalized = {}
                
                if isinstance(outputs, dict):
                    # Unified model
                    for key, value in outputs.items():
                        predictions_normalized[key] = value[j].cpu().numpy()
                    
                    # Convert variance to std
                    if 'prob_var' in predictions_normalized:
                        predictions_normalized['prob_std'] = np.sqrt(predictions_normalized['prob_var'])
                else:
                    # Simple regression model
                    predictions_normalized['regression'] = outputs[j].cpu().numpy()
                
                # Denormalize
                predictions = self.denormalize_predictions(predictions_normalized, norm_params_list[j])
                all_predictions.append(predictions)
        
        return all_predictions
    
    def print_predictions(self, predictions: Dict[str, np.ndarray], ticker: str = None):
        """
        Pretty print predictions.
        
        Args:
            predictions: Dictionary of prediction outputs
            ticker: Optional ticker symbol for display
        """
        if ticker:
            print("\n" + "="*60)
            print(f"Predictions for {ticker}")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("Predictions")
            print("="*60)
        
        # Regression predictions
        if 'regression' in predictions:
            print("\n📈 Regression Predictions (Point Estimates):")
            reg_preds = predictions['regression']
            if reg_preds.ndim == 0:
                print(f"  Day 1: ${reg_preds:.2f}")
            else:
                for i, price in enumerate(reg_preds):
                    print(f"  Day {i+1}: ${price:.2f}")
        
        # Quantile predictions
        if 'quantiles' in predictions:
            print("\n📊 Quantile Predictions (Uncertainty Intervals):")
            quantiles = predictions['quantiles']  # Shape: [pred_horizon, num_quantiles]
            
            if quantiles.ndim == 1:
                # Single timestep
                for i, q_level in enumerate(self.quantile_levels):
                    print(f"  {q_level:.1%} quantile: ${quantiles[i]:.2f}")
            else:
                # Multiple timesteps
                for day in range(quantiles.shape[0]):
                    print(f"  Day {day+1}:")
                    for i, q_level in enumerate(self.quantile_levels):
                        print(f"    {q_level:.1%} quantile: ${quantiles[day, i]:.2f}")
        
        # Probabilistic predictions
        if 'prob_mean' in predictions and 'prob_std' in predictions:
            print("\n🎲 Probabilistic Predictions (Gaussian Distribution):")
            mean = predictions['prob_mean']
            std = predictions['prob_std']
            
            if mean.ndim == 0:
                # Single timestep
                print(f"  Mean: ${mean:.2f}")
                print(f"  Std Dev: ${std:.2f}")
                print(f"  68% Confidence Interval: [${mean - std:.2f}, ${mean + std:.2f}]")
                print(f"  95% Confidence Interval: [${mean - 1.96*std:.2f}, ${mean + 1.96*std:.2f}]")
            else:
                # Multiple timesteps
                for day in range(len(mean)):
                    print(f"  Day {day+1}:")
                    print(f"    Mean: ${mean[day]:.2f}")
                    print(f"    Std Dev: ${std[day]:.2f}")
                    print(f"    68% CI: [${mean[day] - std[day]:.2f}, ${mean[day] + std[day]:.2f}]")
                    print(f"    95% CI: [${mean[day] - 1.96*std[day]:.2f}, ${mean[day] + 1.96*std[day]:.2f}]")
        
        print("\n" + "="*60 + "\n")


def main():
    """Example usage of the StockPredictor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Price Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--data', type=str, required=True, help='Path to pickle data file')
    parser.add_argument('--ticker', type=str, help='Specific ticker to predict (optional)')
    parser.add_argument('--normalization', type=str, default='relative', 
                       choices=['relative', 'zscore'], help='Normalization type')
    parser.add_argument('--min-date', type=str, default='2005-01-01', 
                       help='Minimum date for data filtering')
    parser.add_argument('--device', type=str, default=None, 
                       choices=['cuda', 'cpu'], help='Device to run on')
    parser.add_argument('--num-samples', type=int, default=5, 
                       help='Number of random samples to predict')
    
    args = parser.parse_args()
    
    # Initialize predictor
    print("Initializing predictor...")
    predictor = StockPredictor(args.checkpoint, device=args.device)
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    features = ["Open", "High", "Low", "Close", "Volume"]
    data = load_stock_data(args.data, min_date=args.min_date, features=features)
    
    if len(data) == 0:
        print("No data loaded. Exiting.")
        return
    
    # Make predictions
    if args.ticker:
        # Predict for specific ticker
        if args.ticker not in data:
            print(f"Ticker {args.ticker} not found in data")
            print(f"Available tickers: {list(data.keys())[:10]}...")
            return
        
        print(f"\nMaking prediction for {args.ticker}...")
        predictions = predictor.predict_from_ticker_data(
            args.ticker, data, normalization_type=args.normalization, use_last_window=True
        )
        predictor.print_predictions(predictions, ticker=args.ticker)
    
    else:
        # Predict for random samples
        available_tickers = [t for t in data.keys() if len(data[t]) >= predictor.window_size]
        
        if len(available_tickers) == 0:
            print("No tickers with sufficient data")
            return
        
        print(f"\nMaking predictions for {args.num_samples} random tickers...")
        sample_tickers = np.random.choice(available_tickers, 
                                         size=min(args.num_samples, len(available_tickers)), 
                                         replace=False)
        
        for ticker in sample_tickers:
            predictions = predictor.predict_from_ticker_data(
                ticker, data, normalization_type=args.normalization, use_last_window=True
            )
            predictor.print_predictions(predictions, ticker=ticker)


if __name__ == "__main__":
    main()

