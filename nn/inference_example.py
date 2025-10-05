#!/usr/bin/env python3
"""
Example script demonstrating how to use the StockPredictor for inference.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from inference import StockPredictor
from dataset import load_stock_data
from pathlib import Path

def plot_predictions(tickers, window_data_list, predictions_list, pred_horizon):
    """
    Plot historical prices and predicted prices for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        window_data_list: List of DataFrames containing historical data (window)
        predictions_list: List of prediction dictionaries
        pred_horizon: Number of days predicted into the future
    """
    n_tickers = len(tickers)
    fig, axes = plt.subplots(n_tickers, 1, figsize=(14, 4 * n_tickers))
    
    # Handle single ticker case
    if n_tickers == 1:
        axes = [axes]
    
    for idx, (ticker, window_data, predictions) in enumerate(zip(tickers, window_data_list, predictions_list)):
        ax = axes[idx]
        
        # Get historical data
        historical_dates = window_data.index
        historical_prices = window_data['Close'].values
        
        # Generate future dates for predictions
        last_date = historical_dates[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_horizon, freq='D')
        
        # Plot historical prices
        ax.plot(historical_dates, historical_prices, 'b-', linewidth=2, label='Historical Close Price', alpha=0.7)
        
        # Plot predictions
        if 'regression' in predictions:
            predicted_prices = predictions['regression']
            ax.plot(future_dates, predicted_prices, 'r-', linewidth=2, label='Predicted Price', marker='o')
            
            # Connect last historical point to first predicted point
            ax.plot([historical_dates[-1], future_dates[0]], 
                   [historical_prices[-1], predicted_prices[0]], 
                   'r--', linewidth=1, alpha=0.5)
        
        # Add confidence intervals if available
        if 'prob_mean' in predictions and 'prob_std' in predictions:
            prob_mean = predictions['prob_mean']
            prob_std = predictions['prob_std']
            
            # Plot mean
            ax.plot(future_dates, prob_mean, 'g--', linewidth=1.5, label='Probabilistic Mean', alpha=0.7)
            
            # Plot confidence interval (mean ± 2*std for ~95% confidence)
            upper_bound = prob_mean + 2 * prob_std
            lower_bound = prob_mean - 2 * prob_std
            ax.fill_between(future_dates, lower_bound, upper_bound, 
                           color='green', alpha=0.2, label='95% Confidence Interval')
        
        # Add quantiles if available
        if 'quantiles' in predictions:
            quantiles = predictions['quantiles']
            num_quantiles = quantiles.shape[1]
            
            if num_quantiles >= 3:
                # Plot median (middle quantile)
                median_idx = num_quantiles // 2
                ax.plot(future_dates, quantiles[:, median_idx], 'purple', linestyle='--', linewidth=1.5, 
                       label='Median Quantile', alpha=0.7)
                
                # Plot outer quantiles
                ax.plot(future_dates, quantiles[:, 0], 'orange', linestyle=':', linewidth=1, 
                       label='Lower Quantile', alpha=0.6)
                ax.plot(future_dates, quantiles[:, -1], 'orange', linestyle=':', linewidth=1, 
                       label='Upper Quantile', alpha=0.6)
            elif num_quantiles == 1:
                ax.plot(future_dates, quantiles[:, 0], 'purple', linestyle='--', linewidth=1.5, 
                       label='Quantile Prediction', alpha=0.7)
        
        # Formatting
        ax.set_title(f'{ticker} - Price Prediction', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save to file instead of showing
    output_file = 'stock_predictions.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Plots saved to: {output_file}")


def evaluate_and_plot_predictions(ticker, data, predictor, num_samples=20, stride=5, checkpoint_path=None, output_dir=None):
    """
    Evaluate model on historical data by making predictions and comparing with actual values.
    
    Args:
        ticker: Stock ticker symbol
        data: DataFrame with full historical data
        predictor: StockPredictor instance
        num_samples: Number of prediction windows to evaluate
        stride: Number of days to stride between windows
        checkpoint_path: Path to checkpoint file (for labeling)
        output_dir: Directory to save output plots
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING MODEL ON {ticker} HISTORICAL DATA")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")
    
    window_size = predictor.window_size
    pred_horizon = predictor.pred_horizon
    
    # Calculate how many windows we can create
    total_length = len(data)
    max_windows = (total_length - window_size - pred_horizon) // stride
    num_samples = min(num_samples, max_windows)
    
    print(f"Total data points: {total_length}")
    print(f"Window size: {window_size}")
    print(f"Prediction horizon: {pred_horizon}")
    print(f"Evaluating {num_samples} prediction windows")
    print(f"Stride: {stride} days\n")
    
    # Storage for predictions and actuals
    all_predictions = []
    all_actuals = []
    all_predictions_std = []  # Store standard deviations for probabilistic models
    all_predictions_quantiles = []  # Store quantiles for quantile regression
    all_predictions_full = []  # Store full prediction dicts
    prediction_dates = []
    errors = []
    
    # Generate predictions for different time windows
    for i in range(num_samples):
        start_idx = i * stride
        window_end_idx = start_idx + window_size
        pred_end_idx = window_end_idx + pred_horizon
        
        if pred_end_idx > total_length:
            break
        
        # Get window for prediction
        window_data = data.iloc[start_idx:window_end_idx]
        
        # Get actual future prices
        actual_future = data.iloc[window_end_idx:pred_end_idx]['Close'].values
        
        # Make prediction
        try:
            predictions = predictor.predict(window_data, normalization_type='relative')
            
            # Get predicted prices based on model type
            pred_prices = None
            pred_std = None
            pred_quantiles = None
            
            if 'regression' in predictions:
                pred_prices = predictions['regression']
            elif 'prob_mean' in predictions:
                pred_prices = predictions['prob_mean']
                if 'prob_std' in predictions:
                    pred_std = predictions['prob_std']
            elif 'quantiles' in predictions:
                # Use median (middle quantile) as prediction
                quantiles = predictions['quantiles']
                pred_quantiles = quantiles
                if quantiles.ndim == 2 and quantiles.shape[1] >= 2:
                    pred_prices = quantiles[:, quantiles.shape[1] // 2]
                elif quantiles.ndim == 1:
                    pred_prices = quantiles
            
            # Handle unified model that has multiple outputs
            if 'quantiles' in predictions and pred_quantiles is None:
                pred_quantiles = predictions['quantiles']
            
            if pred_prices is not None:
                all_predictions.append(pred_prices)
                all_actuals.append(actual_future)
                all_predictions_std.append(pred_std)
                all_predictions_quantiles.append(pred_quantiles)
                all_predictions_full.append(predictions)
                prediction_dates.append(data.index[window_end_idx])
                
                # Calculate error for this window
                mse = np.mean((pred_prices - actual_future) ** 2)
                mae = np.mean(np.abs(pred_prices - actual_future))
                mape = np.mean(np.abs((pred_prices - actual_future) / actual_future)) * 100
                errors.append({'mse': mse, 'mae': mae, 'mape': mape})
            else:
                print(f"Warning: No valid predictions for window {i}")
                
        except Exception as e:
            print(f"Warning: Failed to predict for window {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_predictions:
        print("No successful predictions made!")
        return
    
    # Calculate overall metrics
    all_pred_flat = np.concatenate(all_predictions)
    all_actual_flat = np.concatenate(all_actuals)
    
    overall_mse = np.mean((all_pred_flat - all_actual_flat) ** 2)
    overall_mae = np.mean(np.abs(all_pred_flat - all_actual_flat))
    overall_mape = np.mean(np.abs((all_pred_flat - all_actual_flat) / all_actual_flat)) * 100
    overall_rmse = np.sqrt(overall_mse)
    
    print(f"{'='*60}")
    print(f"OVERALL PERFORMANCE METRICS:")
    print(f"{'='*60}")
    print(f"Mean Squared Error (MSE):  ${overall_mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${overall_rmse:.2f}")
    print(f"Mean Absolute Error (MAE): ${overall_mae:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {overall_mape:.2f}%")
    print(f"{'='*60}\n")
    
    # Check if we have probabilistic predictions with std or quantiles
    has_std = any(std is not None for std in all_predictions_std)
    has_quantiles = any(q is not None for q in all_predictions_quantiles)
    
    # Create comprehensive visualization
    if has_std or has_quantiles:
        fig = plt.figure(figsize=(16, 16))
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Full price history with prediction points
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(data.index, data['Close'], 'b-', linewidth=1, label='Actual Price', alpha=0.6)
    ax1.scatter(prediction_dates, [data.loc[d, 'Close'] for d in prediction_dates], 
               color='red', s=30, alpha=0.7, label='Prediction Start Points', zorder=5)
    
    # Add title with checkpoint info if available
    title = f'{ticker} - Full Price History with Prediction Points'
    if checkpoint_path:
        # Shorten checkpoint path for display
        checkpoint_name = Path(checkpoint_path).name
        title += f'\nCheckpoint: {checkpoint_name}'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Sample predictions vs actuals (first 5 windows) with confidence intervals
    ax2 = fig.add_subplot(gs[1, :])
    num_plot = min(5, len(all_predictions))
    colors = plt.cm.tab10(np.linspace(0, 1, num_plot))
    
    for i in range(num_plot):
        pred_dates = pd.date_range(start=prediction_dates[i], periods=pred_horizon, freq='D')
        
        # Plot actual prices
        ax2.plot(pred_dates, all_actuals[i], 'o-', color=colors[i], 
                linewidth=2, markersize=6, label=f'Actual {prediction_dates[i].date()}', alpha=0.8)
        
        # Plot predicted prices
        ax2.plot(pred_dates, all_predictions[i], 's--', color=colors[i], 
                linewidth=2, markersize=5, label=f'Predicted {prediction_dates[i].date()}', alpha=0.7)
        
        # Add confidence interval if available (for probabilistic models)
        if all_predictions_std[i] is not None:
            pred_std = all_predictions_std[i]
            upper_bound = all_predictions[i] + 2 * pred_std  # 95% CI
            lower_bound = all_predictions[i] - 2 * pred_std
            ax2.fill_between(pred_dates, lower_bound, upper_bound, 
                           color=colors[i], alpha=0.15, label=f'95% CI {prediction_dates[i].date()}')
        
        # Add quantile envelope if available (for quantile regression)
        if all_predictions_quantiles[i] is not None:
            quantiles = all_predictions_quantiles[i]
            if quantiles.ndim == 2 and quantiles.shape[1] >= 3:
                # Assume quantiles are [0.1, 0.5, 0.9] or similar
                # Use lowest and highest quantiles as envelope
                lower_quantile = quantiles[:, 0]
                upper_quantile = quantiles[:, -1]
                ax2.fill_between(pred_dates, lower_quantile, upper_quantile, 
                               color=colors[i], alpha=0.15, label=f'Quantile Range {prediction_dates[i].date()}')
    
    model_type = predictor.model_type
    ax2.set_title(f'Sample Predictions vs Actual Prices (First {num_plot} Windows) - Model: {model_type}', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Price ($)', fontsize=12)
    ax2.legend(loc='best', fontsize=7, ncol=3)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Residual Plot: Prediction Error vs Actual Price
    from sklearn.metrics import r2_score
    
    ax3 = fig.add_subplot(gs[2, 0])
    errors = all_pred_flat - all_actual_flat
    percentage_errors = (errors / all_actual_flat) * 100
    
    # Plot residuals vs actual prices with color mapping
    scatter = ax3.scatter(all_actual_flat, percentage_errors, alpha=0.5, s=20, 
                         c=percentage_errors, cmap='RdYlGn_r', vmin=-10, vmax=10)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax3.axhline(y=percentage_errors.mean(), color='b', linestyle='--', linewidth=1.5, 
               label=f'Mean: {percentage_errors.mean():.2f}%', alpha=0.7)
    
    # Add R² score to show overall fit quality
    r2 = r2_score(all_actual_flat, all_pred_flat)
    ax3.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax3.transAxes, 
            fontsize=11, verticalalignment='top', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Error (%)', rotation=270, labelpad=20)
    
    ax3.set_title('Prediction Error vs Actual Price', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Actual Price ($)', fontsize=11)
    ax3.set_ylabel('Prediction Error (%)', fontsize=11)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. Error distribution
    ax4 = fig.add_subplot(gs[2, 1])
    errors_flat = all_pred_flat - all_actual_flat
    ax4.hist(errors_flat, bins=30, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax4.axvline(x=errors_flat.mean(), color='g', linestyle='--', linewidth=2, 
               label=f'Mean Error: ${errors_flat.mean():.2f}')
    ax4.set_title('Prediction Error Distribution', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Error (Predicted - Actual) $', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Uncertainty Visualization (for probabilistic/quantile models)
    if has_std or has_quantiles:
        ax5 = fig.add_subplot(gs[3, :])
        
        # Plot a detailed view of predictions with uncertainty
        # Use fewer samples for clarity
        num_detailed = min(3, len(all_predictions))
        colors_detail = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        offset = 0
        for i in range(num_detailed):
            pred_dates = pd.date_range(start=prediction_dates[i], periods=pred_horizon, freq='D')
            pred_dates_offset = [d + pd.Timedelta(days=offset) for d in pred_dates]
            
            # Plot mean/median prediction
            ax5.plot(pred_dates_offset, all_predictions[i], 
                    color=colors_detail[i], linewidth=2.5, marker='s', 
                    markersize=8, label=f'Predicted {prediction_dates[i].date()}', zorder=3)
            
            # Plot actual
            ax5.plot(pred_dates_offset, all_actuals[i], 
                    color=colors_detail[i], linewidth=2.5, marker='o', 
                    markersize=8, linestyle=':', label=f'Actual {prediction_dates[i].date()}', 
                    alpha=0.8, zorder=3)
            
            # Probabilistic model with std
            if all_predictions_std[i] is not None:
                pred_std = all_predictions_std[i]
                
                # 68% confidence interval (1 std)
                upper_1std = all_predictions[i] + pred_std
                lower_1std = all_predictions[i] - pred_std
                ax5.fill_between(pred_dates_offset, lower_1std, upper_1std, 
                               color=colors_detail[i], alpha=0.3, label=f'68% CI (±1σ)', zorder=1)
                
                # 95% confidence interval (2 std)
                upper_2std = all_predictions[i] + 2 * pred_std
                lower_2std = all_predictions[i] - 2 * pred_std
                ax5.fill_between(pred_dates_offset, lower_2std, upper_2std, 
                               color=colors_detail[i], alpha=0.15, label=f'95% CI (±2σ)', zorder=0)
            
            # Quantile model
            elif all_predictions_quantiles[i] is not None:
                quantiles = all_predictions_quantiles[i]
                if quantiles.ndim == 2 and quantiles.shape[1] >= 3:
                    # Get quantile levels from predictor
                    q_levels = predictor.quantile_levels
                    
                    # Plot all quantile lines with distinct styles
                    linestyles = ['-', '--', '-.', ':', '-']  # Cycle through different line styles
                    linewidths = [1.0, 2.0, 1.0]  # Thicker line for median
                    
                    for q_idx in range(quantiles.shape[1]):
                        # Get actual quantile level if available, otherwise use index
                        if q_idx < len(q_levels):
                            q_level_label = f'Q{q_levels[q_idx]:.2f}'
                        else:
                            q_level_label = f'Q{q_idx}'
                        
                        # Different styles for different quantiles
                        median_idx = quantiles.shape[1] // 2
                        if q_idx == median_idx:
                            # Median: solid line, thicker, more opaque
                            linestyle = '-'
                            linewidth = 2.5
                            alpha = 1.0
                        elif q_idx == 0:
                            # Lowest quantile: dotted line
                            linestyle = ':'
                            linewidth = 1.5
                            alpha = 0.7
                        elif q_idx == quantiles.shape[1] - 1:
                            # Highest quantile: dotted line
                            linestyle = ':'
                            linewidth = 1.5
                            alpha = 0.7
                        else:
                            # Middle quantiles: dashed lines
                            linestyle = '--'
                            linewidth = 1.2
                            alpha = 0.6
                        
                        # Only label key quantiles to avoid clutter
                        show_label = q_idx in [0, quantiles.shape[1]//2, quantiles.shape[1]-1]
                        label = f'{q_level_label} {prediction_dates[i].date()}' if show_label else None
                        
                        ax5.plot(pred_dates_offset, quantiles[:, q_idx], 
                               color=colors_detail[i], linestyle=linestyle, 
                               linewidth=linewidth, alpha=alpha, label=label, zorder=2)
                    
                    # Fill between outer quantiles
                    ax5.fill_between(pred_dates_offset, quantiles[:, 0], quantiles[:, -1], 
                                   color=colors_detail[i], alpha=0.2, 
                                   label=f'Quantile Range', zorder=0)
            
            offset += 2  # Small offset to prevent overlap
        
        title_text = 'Detailed Prediction Uncertainty'
        if has_std:
            title_text += ' (Probabilistic Model)'
        if has_quantiles:
            title_text += ' (Quantile Regression)'
        
        ax5.set_title(title_text, fontsize=14, fontweight='bold')
        ax5.set_xlabel('Date', fontsize=12)
        ax5.set_ylabel('Price ($)', fontsize=12)
        ax5.legend(loc='best', fontsize=8, ncol=3)
        ax5.grid(True, alpha=0.3)
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Calculate coverage statistics
        if has_std:
            coverage_68 = 0
            coverage_95 = 0
            total_points = 0
            
            for i, pred_std in enumerate(all_predictions_std):
                if pred_std is not None:
                    within_1std = np.abs(all_predictions[i] - all_actuals[i]) <= pred_std
                    within_2std = np.abs(all_predictions[i] - all_actuals[i]) <= 2 * pred_std
                    coverage_68 += np.sum(within_1std)
                    coverage_95 += np.sum(within_2std)
                    total_points += len(within_1std)
            
            if total_points > 0:
                coverage_68_pct = (coverage_68 / total_points) * 100
                coverage_95_pct = (coverage_95 / total_points) * 100
                
                textstr = f'CI Coverage:\n68%: {coverage_68_pct:.1f}%\n95%: {coverage_95_pct:.1f}%'
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax5.text(0.02, 0.98, textstr, transform=ax5.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
                
                print(f"Confidence Interval Coverage:")
                print(f"  68% CI (±1σ): {coverage_68_pct:.1f}% (expected: 68%)")
                print(f"  95% CI (±2σ): {coverage_95_pct:.1f}% (expected: 95%)")
                print()
        
        if has_quantiles:
            # Calculate quantile coverage
            coverage_within = 0
            total_points = 0
            
            for i, quantiles in enumerate(all_predictions_quantiles):
                if quantiles is not None and quantiles.ndim == 2 and quantiles.shape[1] >= 3:
                    within_range = (all_actuals[i] >= quantiles[:, 0]) & (all_actuals[i] <= quantiles[:, -1])
                    coverage_within += np.sum(within_range)
                    total_points += len(within_range)
            
            if total_points > 0:
                coverage_pct = (coverage_within / total_points) * 100
                
                # Get actual quantile levels from predictor
                if len(predictor.quantile_levels) > 0:
                    q_low = predictor.quantile_levels[0]
                    q_high = predictor.quantile_levels[-1]
                    expected_coverage = (q_high - q_low) * 100
                else:
                    # Fallback if quantile levels not available
                    q_low, q_high = 0.1, 0.9
                    expected_coverage = 80.0
                
                textstr = f'Quantile Coverage:\n{coverage_pct:.1f}%\n(expected: {expected_coverage:.0f}%)'
                props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
                ax5.text(0.98, 0.98, textstr, transform=ax5.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='right', bbox=props)
                
                print(f"Quantile Range Coverage:")
                print(f"  {q_low:.1f}-{q_high:.1f} quantile range: {coverage_pct:.1f}% (expected: {expected_coverage:.0f}%)")
                print()
    
    plt.tight_layout()
    
    # Save figure
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with checkpoint info
        if checkpoint_path:
            checkpoint_name = checkpoint_path.split("/")[1] + "_" + checkpoint_path.split("/")[-1]  # e.g., checkpoint_batch_10000
            output_file = output_dir / f'{ticker}_{checkpoint_name}_evaluation.png'
        else:
            output_file = output_dir / f'{ticker}_evaluation.png'
    else:
        output_file = f'{ticker}_evaluation.png'
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Evaluation plots saved to: {output_file}\n")


# Example 1: Simple prediction for a single ticker
def example_single_prediction():
    """Example: Make prediction for a single ticker."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Single Ticker Prediction")
    print("="*80)
    
    # Initialize predictor with checkpoint
    checkpoint_path = "checkpoints/best_model.pt"
    predictor = StockPredictor(checkpoint_path, device='cpu')
    
    # Load data
    data_path = "/home/abhishek/Desktop/Projects/Segment/S&P500/Sap_500_data_fresh.pkl"
    data = load_stock_data(data_path, min_date="2020-01-01", features=["Open", "High", "Low", "Close", "Volume"])
    
    # Pick a ticker
    ticker = "AAPL"
    
    if ticker in data:
        # Make prediction using the last window
        predictions = predictor.predict_from_ticker_data(
            ticker, data, 
            normalization_type='relative',
            use_last_window=True
        )
        
        # Display predictions
        predictor.print_predictions(predictions, ticker=ticker)
    else:
        print(f"Ticker {ticker} not found in data")


# Example 2: Batch predictions for multiple tickers
def example_batch_predictions():
    """Example: Make predictions for multiple tickers at once."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch Predictions")
    print("="*80)
    
    # Initialize predictor
    checkpoint_path = "checkpoints/best_model.pt"
    predictor = StockPredictor(checkpoint_path, device='cpu')
    
    # Load data
    data_path = "/home/abhishek/Desktop/Projects/Segment/S&P500/Sap_500_data_fresh.pkl"
    data = load_stock_data(data_path, min_date="2020-01-01")
    
    # Select multiple tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    # Prepare data for batch prediction
    data_list = []
    valid_tickers = []
    window_data_list = []
    
    for ticker in tickers:
        if ticker in data and len(data[ticker]) >= predictor.window_size:
            # Get last window for each ticker
            window_data = data[ticker].iloc[-predictor.window_size:]
            data_list.append(window_data)
            valid_tickers.append(ticker)
            window_data_list.append(window_data)
    
    if data_list:
        # Make batch predictions
        batch_predictions = predictor.batch_predict(
            data_list, 
            normalization_type='relative',
            batch_size=8
        )
        
        # Display results
        for ticker, predictions in zip(valid_tickers, batch_predictions):
            predictor.print_predictions(predictions, ticker=ticker)
        
        # Plot results
        plot_predictions(valid_tickers, window_data_list, batch_predictions, predictor.pred_horizon)
    else:
        print("No valid tickers found")


# Example 3: Custom data prediction
def example_custom_data():
    """Example: Make prediction on custom DataFrame."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Custom Data Prediction")
    print("="*80)
    
    # Initialize predictor
    checkpoint_path = "checkpoints/best_model.pt"
    predictor = StockPredictor(checkpoint_path, device='cpu')
    
    # Create custom data (or load from CSV, API, etc.)
    # This should have window_size rows and OHLCV columns
    dates = pd.date_range(start='2024-01-01', periods=predictor.window_size, freq='D')
    
    # Example: synthetic data (in practice, load real data)
    custom_data = pd.DataFrame({
        'Open': np.random.uniform(150, 160, predictor.window_size),
        'High': np.random.uniform(160, 170, predictor.window_size),
        'Low': np.random.uniform(140, 150, predictor.window_size),
        'Close': np.random.uniform(150, 160, predictor.window_size),
        'Volume': np.random.uniform(1e6, 5e6, predictor.window_size)
    }, index=dates)
    
    # Make prediction
    predictions = predictor.predict(custom_data, normalization_type='relative')
    
    # Display predictions
    predictor.print_predictions(predictions, ticker="CUSTOM")


# Example 4: Compare normalization methods
def example_compare_normalization():
    """Example: Compare different normalization methods."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Comparing Normalization Methods")
    print("="*80)
    
    # Initialize predictor
    checkpoint_path = "checkpoints/best_model.pt"
    predictor = StockPredictor(checkpoint_path, device='cpu')
    
    # Load data
    data_path = "/home/abhishek/Desktop/Projects/Segment/S&P500/Sap_500_data_fresh.pkl"
    data = load_stock_data(data_path, min_date="2020-01-01")
    
    ticker = "AAPL"
    
    if ticker in data:
        # Get the same window
        window_data = data[ticker].iloc[-predictor.window_size:]
        
        # Predict with relative normalization
        print("\n--- Using Relative Normalization ---")
        predictions_relative = predictor.predict(window_data, normalization_type='relative')
        predictor.print_predictions(predictions_relative, ticker=f"{ticker} (Relative)")
        
        # Predict with z-score normalization
        print("\n--- Using Z-Score Normalization ---")
        predictions_zscore = predictor.predict(window_data, normalization_type='zscore')
        predictor.print_predictions(predictions_zscore, ticker=f"{ticker} (Z-Score)")
    else:
        print(f"Ticker {ticker} not found")


# Example 5: Accessing raw prediction values
def example_raw_predictions():
    """Example: Access raw prediction values for further processing."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Accessing Raw Prediction Values")
    print("="*80)
    
    # Initialize predictor
    checkpoint_path = "checkpoints/best_model.pt"
    predictor = StockPredictor(checkpoint_path, device='cpu')
    
    # Load data
    data_path = "/home/abhishek/Desktop/Projects/Segment/S&P500/Sap_500_data_fresh.pkl"
    data = load_stock_data(data_path, min_date="2020-01-01")
    
    ticker = "AAPL"
    
    if ticker in data:
        predictions = predictor.predict_from_ticker_data(
            ticker, data, use_last_window=True
        )
        
        print(f"\nRaw prediction data for {ticker}:")
        print("-" * 60)
        
        # Access different prediction types
        if 'regression' in predictions:
            print("\nRegression predictions:")
            print(f"  Shape: {predictions['regression'].shape}")
            print(f"  Values: {predictions['regression']}")
        
        if 'quantiles' in predictions:
            print("\nQuantile predictions:")
            print(f"  Shape: {predictions['quantiles'].shape}")
            print(f"  Quantile levels: {predictor.quantile_levels}")
            print(f"  Values:\n{predictions['quantiles']}")
        
        if 'prob_mean' in predictions:
            print("\nProbabilistic predictions:")
            print(f"  Mean shape: {predictions['prob_mean'].shape}")
            print(f"  Mean values: {predictions['prob_mean']}")
            print(f"  Std values: {predictions['prob_std']}")
            print(f"  Variance values: {predictions['prob_var']}")
        
        # Example: Calculate custom metrics
        if 'regression' in predictions:
            reg_pred = predictions['regression']
            print("\nCustom metrics:")
            print(f"  Average predicted price: ${reg_pred.mean():.2f}")
            print(f"  Price range: ${reg_pred.min():.2f} - ${reg_pred.max():.2f}")
            print(f"  Predicted change: {((reg_pred[-1] / reg_pred[0]) - 1) * 100:.2f}%")
    else:
        print(f"Ticker {ticker} not found")


# Example 6: Ensemble predictions (if you have multiple checkpoints)
def example_ensemble_predictions():
    """Example: Combine predictions from multiple models."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Ensemble Predictions")
    print("="*80)
    
    # List of checkpoint paths
    checkpoint_paths = [
        "checkpoints/best_model.pt",
        # "checkpoints/checkpoint_epoch_1.pt",
        # Add more checkpoints if available
    ]
    
    # Load data
    data_path = "/home/abhishek/Desktop/Projects/Segment/S&P500/Sap_500_data_fresh.pkl"
    data = load_stock_data(data_path, min_date="2020-01-01")
    
    ticker = "AAPL"
    
    if ticker not in data:
        print(f"Ticker {ticker} not found")
        return
    
    window_data = data[ticker].iloc[-64:]  # Assuming window_size=64
    
    # Collect predictions from all models
    all_predictions = []
    
    for checkpoint_path in checkpoint_paths:
        try:
            predictor = StockPredictor(checkpoint_path, device='cpu')
            predictions = predictor.predict(window_data, normalization_type='relative')
            all_predictions.append(predictions)
            print(f"✓ Loaded and predicted from {checkpoint_path}")
        except (FileNotFoundError, KeyError, RuntimeError) as e:
            print(f"✗ Failed to load {checkpoint_path}: {e}")
    
    if not all_predictions:
        print("No predictions available")
        return
    
    # Ensemble: Average predictions
    print(f"\n--- Ensemble Results for {ticker} ---")
    
    if 'regression' in all_predictions[0]:
        ensemble_reg = np.mean([p['regression'] for p in all_predictions], axis=0)
        print("\nEnsemble Regression Predictions:")
        for i, price in enumerate(ensemble_reg):
            print(f"  Day {i+1}: ${price:.2f}")
    
    if 'prob_mean' in all_predictions[0]:
        ensemble_mean = np.mean([p['prob_mean'] for p in all_predictions], axis=0)
        ensemble_std = np.mean([p['prob_std'] for p in all_predictions], axis=0)
        print("\nEnsemble Probabilistic Predictions:")
        for i in range(len(ensemble_mean)):
            print(f"  Day {i+1}: ${ensemble_mean[i]:.2f} ± ${ensemble_std[i]:.2f}")


# Example 7: Evaluate model on full historical data
def example_evaluate_on_history():
    """Example: Evaluate model performance on historical data."""
    print("\n" + "="*80)
    print("EXAMPLE 7: Model Evaluation on Historical Data")
    print("="*80)
    
    # Initialize predictor
    checkpoint_path = "/home/abhishek/Desktop/Projects/finances/nn/wandb/run-20251004_182833-60s1dt6l/files/checkpoints/checkpoint_batch_10000.pt"
    predictor = StockPredictor(checkpoint_path, device='cpu')
    
    # Load data
    data_path = "/home/abhishek/Desktop/Projects/Segment/S&P500/Sap_500_data_fresh.pkl"
    data_dict = load_stock_data(data_path, min_date="2020-01-01")
    
    # Evaluate on AAPL
    ticker = "AAPL"
    
    if ticker in data_dict:
        ticker_data = data_dict[ticker]
        print(f"\nLoaded {len(ticker_data)} data points for {ticker}")
        print(f"Date range: {ticker_data.index[0]} to {ticker_data.index[-1]}")
        
        # Evaluate with multiple prediction windows
        evaluate_and_plot_predictions(
            ticker=ticker,
            data=ticker_data,
            predictor=predictor,
            num_samples=300,  # Number of different windows to test
            stride=10  # Days between prediction windows
        )
    else:
        print(f"Ticker {ticker} not found in data")


def find_all_checkpoints(wandb_dir="wandb", checkpoint_pattern="checkpoint_batch_*.pt"):
    """
    Find all checkpoint files in wandb directory.
    
    Args:
        wandb_dir: Path to wandb directory
        checkpoint_pattern: Pattern to match checkpoint files
    
    Returns:
        List of checkpoint file paths
    """
    wandb_path = Path(wandb_dir)
    
    if not wandb_path.exists():
        print(f"Warning: wandb directory not found: {wandb_path}")
        return []
    
    # Find all checkpoints matching pattern in run-* folders
    checkpoint_paths = []
    for run_dir in wandb_path.glob("run-2025100*"):
        checkpoint_dir = run_dir / "files" / "checkpoints"
        if checkpoint_dir.exists():
            for checkpoint_file in checkpoint_dir.glob(checkpoint_pattern):
                checkpoint_paths.append(checkpoint_file)
    
    # Sort by modification time (newest first)
    checkpoint_paths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    return checkpoint_paths


def example_evaluate_all_checkpoints():
    """Example: Evaluate all checkpoints in wandb directory."""
    print("\n" + "="*80)
    print("EXAMPLE: Evaluating All Checkpoints in wandb Directory")
    print("="*80)
    
    # Find all checkpoints
    checkpoint_paths = find_all_checkpoints(wandb_dir="wandb")
    
    if not checkpoint_paths:
        print("No checkpoints found in wandb directory!")
        return
    
    print(f"Found {len(checkpoint_paths)} checkpoints")
    for cp in checkpoint_paths:
        print(f"  - {cp}")
    print()
    
    # Load data once
    data_path = "/home/abhishek/Desktop/Projects/Segment/S&P500/Sap_500_data_fresh.pkl"
    data_dict = load_stock_data(data_path, min_date="2020-01-01")
    
    ticker = "MSFT"
    
    if ticker not in data_dict:
        print(f"Ticker {ticker} not found in data")
        return
    
    ticker_data = data_dict[ticker]
    print(f"Loaded {len(ticker_data)} data points for {ticker}")
    print(f"Date range: {ticker_data.index[0]} to {ticker_data.index[-1]}\n")
    
    # Create output directory
    output_dir = Path("checkpoint_evaluations")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}\n")
    
    # Evaluate each checkpoint
    results = []
    for i, checkpoint_path in enumerate(checkpoint_paths, 1):
        print(f"\n{'='*80}")
        print(f"Evaluating checkpoint {i}/{len(checkpoint_paths)}")
        print(f"{'='*80}")
        
        try:
            # Load predictor
            predictor = StockPredictor(str(checkpoint_path), device='cuda')
            
            # Evaluate
            evaluate_and_plot_predictions(
                ticker=ticker,
                data=ticker_data,
                predictor=predictor,
                num_samples=300,
                stride=30,
                checkpoint_path=str(checkpoint_path),
                output_dir=output_dir
            )
            
            results.append({
                'checkpoint': str(checkpoint_path),
                'status': 'success'
            })
            
        except Exception as e:
            print(f"❌ Failed to evaluate {checkpoint_path}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'checkpoint': str(checkpoint_path),
                'status': 'failed',
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    print(f"Total checkpoints: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {output_dir.absolute()}")
    
    if failed > 0:
        print("\nFailed checkpoints:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - {r['checkpoint']}")
                print(f"    Error: {r.get('error', 'Unknown')}")


if __name__ == "__main__":
    # Run examples (comment out ones you don't want to run)
    
    # try:
    #     example_single_prediction()
    # except (FileNotFoundError, KeyError, ValueError, RuntimeError) as e:
    #     print(f"Example 1 failed: {e}")
    
    # try:
    #     example_batch_predictions()
    # except Exception as e:
    #     print(f"Example 2 failed: {e}")
    
    # try:
    #     example_custom_data()
    # except Exception as e:
    #     print(f"Example 3 failed: {e}")
    
    # try:
    #     example_compare_normalization()
    # except Exception as e:
    #     print(f"Example 4 failed: {e}")
    
    # try:
    #     example_raw_predictions()
    # except Exception as e:
    #     print(f"Example 5 failed: {e}")
    
    # try:
    #     example_ensemble_predictions()
    # except Exception as e:
    #     print(f"Example 6 failed: {e}")
    
    # try:
    #     example_evaluate_on_history()
    # except Exception as e:
    #     print(f"Example 7 failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    
    try:
        example_evaluate_all_checkpoints()
    except Exception as e:
        print(f"Example evaluate all checkpoints failed: {e}")
        import traceback
        traceback.print_exc()

