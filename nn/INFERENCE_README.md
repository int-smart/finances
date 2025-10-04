# Stock Price Prediction - Inference Module

This module provides comprehensive inference capabilities for the stock prediction model, supporting multiple prediction types: regression, quantile, and probabilistic predictions.

## Features

- **Load trained checkpoints** with automatic configuration extraction
- **Multiple model types** support (regression, quantile, probabilistic)
- **Automatic denormalization** of predictions to original price scale
- **Batch predictions** for efficient processing
- **Pretty printing** of predictions with confidence intervals
- **Command-line interface** for quick predictions

## Quick Start

### 1. Command-Line Usage

The simplest way to make predictions:

```bash
# Predict for a specific ticker
python inference.py --checkpoint checkpoints/best_model.pt \
                    --data path/to/data.pkl \
                    --ticker AAPL

# Predict for multiple random tickers
python inference.py --checkpoint checkpoints/best_model.pt \
                    --data path/to/data.pkl \
                    --num-samples 5

# Use z-score normalization
python inference.py --checkpoint checkpoints/best_model.pt \
                    --data path/to/data.pkl \
                    --ticker AAPL \
                    --normalization zscore
```

### 2. Python API Usage

```python
from inference import StockPredictor
from dataset import load_stock_data

# Initialize predictor
predictor = StockPredictor("checkpoints/best_model.pt", device='cpu')

# Load data
data = load_stock_data("path/to/data.pkl", min_date="2020-01-01")

# Make prediction
predictions = predictor.predict_from_ticker_data(
    ticker="AAPL", 
    data_dict=data,
    normalization_type='relative'
)

# Display results
predictor.print_predictions(predictions, ticker="AAPL")
```

## Detailed Usage

### Loading a Checkpoint

The checkpoint contains all necessary configuration (window size, model architecture, etc.):

```python
from inference import StockPredictor

# Auto-detects GPU/CPU
predictor = StockPredictor("checkpoints/best_model.pt")

# Force CPU
predictor = StockPredictor("checkpoints/best_model.pt", device='cpu')

# Force GPU
predictor = StockPredictor("checkpoints/best_model.pt", device='cuda')
```

### Making Predictions

#### Option 1: Direct DataFrame Prediction

If you have a DataFrame with exactly `window_size` rows:

```python
import pandas as pd

# Prepare data: must have window_size rows with OHLCV columns
window_data = df.iloc[-64:]  # Last 64 days

# Make prediction
predictions = predictor.predict(window_data, normalization_type='relative')
```

#### Option 2: Predict from Data Dictionary

If you have a dictionary of ticker -> DataFrame:

```python
# Use last window
predictions = predictor.predict_from_ticker_data(
    ticker="AAPL",
    data_dict=data,
    use_last_window=True  # Use most recent data
)

# Use random window (useful for evaluation)
predictions = predictor.predict_from_ticker_data(
    ticker="AAPL",
    data_dict=data,
    use_last_window=False  # Random window
)
```

#### Option 3: Batch Predictions

Process multiple samples efficiently:

```python
# Prepare list of DataFrames
data_list = [
    data['AAPL'].iloc[-64:],
    data['MSFT'].iloc[-64:],
    data['GOOGL'].iloc[-64:]
]

# Batch predict
predictions_list = predictor.batch_predict(
    data_list,
    normalization_type='relative',
    batch_size=32
)

# predictions_list is a list of prediction dictionaries
```

### Understanding Predictions

The `predict()` method returns a dictionary with different prediction types depending on the model:

#### Regression Model Output
```python
predictions = {
    'regression': array([150.2, 151.5, 149.8, ...])  # Shape: [pred_horizon]
}
```

#### Quantile Model Output
```python
predictions = {
    'quantiles': array([
        [145.0, 150.0, 155.0],  # Day 1: [10%, 50%, 90%]
        [146.0, 151.0, 156.0],  # Day 2: [10%, 50%, 90%]
        ...
    ])  # Shape: [pred_horizon, num_quantiles]
}
```

#### Probabilistic Model Output
```python
predictions = {
    'prob_mean': array([150.2, 151.5, ...]),  # Shape: [pred_horizon]
    'prob_std': array([2.5, 2.8, ...]),       # Shape: [pred_horizon]
    'prob_var': array([6.25, 7.84, ...])      # Shape: [pred_horizon]
}
```

#### Combined Model Output
A unified model may produce all of the above simultaneously.

### Accessing Raw Values

All predictions are automatically denormalized to original price scale:

```python
predictions = predictor.predict_from_ticker_data("AAPL", data)

# Access regression predictions
if 'regression' in predictions:
    reg_prices = predictions['regression']
    print(f"Next day prediction: ${reg_prices[0]:.2f}")
    print(f"Average 7-day prediction: ${reg_prices.mean():.2f}")

# Access quantile predictions
if 'quantiles' in predictions:
    quantiles = predictions['quantiles']
    # Get 50th percentile (median) for day 1
    median_price = quantiles[0, 1]  # Assuming [0.1, 0.5, 0.9]
    
# Access probabilistic predictions
if 'prob_mean' in predictions:
    mean = predictions['prob_mean'][0]  # Day 1 mean
    std = predictions['prob_std'][0]    # Day 1 std dev
    
    # Calculate 95% confidence interval
    lower_95 = mean - 1.96 * std
    upper_95 = mean + 1.96 * std
    print(f"95% CI: [${lower_95:.2f}, ${upper_95:.2f}]")
```

### Normalization Types

Two normalization methods are supported:

#### Relative Normalization (Default)
- Normalizes by last values in the window
- Better preserves relative relationships
- Recommended for stock prices

```python
predictions = predictor.predict(data, normalization_type='relative')
```

#### Z-Score Normalization
- Standardizes using mean and std dev
- Better for data with different scales
- May be more stable for volatile stocks

```python
predictions = predictor.predict(data, normalization_type='zscore')
```

### Pretty Printing

Display predictions in a human-readable format:

```python
predictions = predictor.predict_from_ticker_data("AAPL", data)
predictor.print_predictions(predictions, ticker="AAPL")
```

Output example:
```
============================================================
Predictions for AAPL
============================================================

📈 Regression Predictions (Point Estimates):
  Day 1: $150.25
  Day 2: $151.30
  Day 3: $149.80
  ...

📊 Quantile Predictions (Uncertainty Intervals):
  Day 1:
    10.0% quantile: $145.50
    50.0% quantile: $150.00
    90.0% quantile: $154.50
  ...

🎲 Probabilistic Predictions (Gaussian Distribution):
  Day 1:
    Mean: $150.20
    Std Dev: $2.50
    68% CI: [$147.70, $152.70]
    95% CI: [$145.30, $155.10]
  ...
============================================================
```

## Advanced Examples

### Example 1: Ensemble Predictions

Combine predictions from multiple models:

```python
checkpoint_paths = [
    "checkpoints/model1.pt",
    "checkpoints/model2.pt",
    "checkpoints/model3.pt"
]

all_predictions = []
for ckpt in checkpoint_paths:
    predictor = StockPredictor(ckpt)
    pred = predictor.predict_from_ticker_data("AAPL", data)
    all_predictions.append(pred)

# Average regression predictions
if 'regression' in all_predictions[0]:
    ensemble_pred = np.mean([p['regression'] for p in all_predictions], axis=0)
    print(f"Ensemble prediction: {ensemble_pred}")
```

### Example 2: Custom Data Source

Make predictions from any data source (CSV, API, database):

```python
import pandas as pd

# Load from CSV
custom_data = pd.read_csv("recent_prices.csv", parse_dates=['Date'], index_col='Date')
custom_data = custom_data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Ensure correct length
assert len(custom_data) == predictor.window_size

# Make prediction
predictions = predictor.predict(custom_data)
```

### Example 3: Real-time Predictions

Fetch live data and make predictions:

```python
import yfinance as yf
from datetime import datetime, timedelta

# Fetch recent data
ticker_symbol = "AAPL"
end_date = datetime.now()
start_date = end_date - timedelta(days=100)  # Get extra for window

stock = yf.Ticker(ticker_symbol)
df = stock.history(start=start_date, end=end_date)

# Get last window
window_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[-predictor.window_size:]

# Predict
predictions = predictor.predict(window_data)
predictor.print_predictions(predictions, ticker=ticker_symbol)
```

### Example 4: Batch Evaluation

Evaluate model on multiple tickers:

```python
import numpy as np

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
results = {}

for ticker in tickers:
    if ticker not in data:
        continue
    
    predictions = predictor.predict_from_ticker_data(
        ticker, data, use_last_window=True
    )
    
    results[ticker] = {
        'ticker': ticker,
        'predictions': predictions,
        'mean_price': predictions.get('regression', [0]).mean() if 'regression' in predictions else None
    }

# Analyze results
for ticker, result in results.items():
    print(f"{ticker}: ${result['mean_price']:.2f}")
```

## API Reference

### StockPredictor

#### `__init__(checkpoint_path, device=None)`
Initialize predictor with a checkpoint.

**Parameters:**
- `checkpoint_path` (str): Path to checkpoint file
- `device` (str, optional): 'cuda' or 'cpu', auto-detects if None

#### `predict(data, normalization_type='relative')`
Make predictions on DataFrame.

**Parameters:**
- `data` (pd.DataFrame): DataFrame with window_size rows and OHLCV columns
- `normalization_type` (str): 'relative' or 'zscore'

**Returns:** Dictionary with predictions (denormalized)

#### `predict_from_ticker_data(ticker, data_dict, normalization_type='relative', use_last_window=True)`
Make predictions for a specific ticker.

**Parameters:**
- `ticker` (str): Stock ticker symbol
- `data_dict` (dict): Dictionary of ticker -> DataFrame
- `normalization_type` (str): 'relative' or 'zscore'
- `use_last_window` (bool): Use last window or random sample

**Returns:** Dictionary with predictions (denormalized)

#### `batch_predict(data_list, normalization_type='relative', batch_size=32)`
Make predictions on multiple samples.

**Parameters:**
- `data_list` (list): List of DataFrames
- `normalization_type` (str): 'relative' or 'zscore'
- `batch_size` (int): Batch size for inference

**Returns:** List of prediction dictionaries

#### `print_predictions(predictions, ticker=None)`
Pretty print predictions.

**Parameters:**
- `predictions` (dict): Prediction dictionary
- `ticker` (str, optional): Ticker symbol for display

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Pandas
- yfinance (optional, for real-time data)

## Troubleshooting

### Issue: "Data must have exactly N timesteps"
**Solution:** Ensure your DataFrame has exactly `window_size` rows. Check `predictor.window_size`.

### Issue: "Ticker not found in data"
**Solution:** Verify the ticker exists in your data dictionary. Check `data.keys()`.

### Issue: Predictions seem incorrect
**Solution:** 
- Ensure normalization type matches training (usually 'relative')
- Check that data is properly formatted with OHLCV columns
- Verify checkpoint is loading correctly

### Issue: Out of memory on GPU
**Solution:** Use `device='cpu'` or reduce `batch_size` in `batch_predict()`.

## Notes

- All predictions are automatically denormalized to original price scale
- The checkpoint contains all configuration (no need to specify manually)
- Normalization type should match what was used during training
- For best results, use data from the same distribution as training data

## Examples

See `inference_example.py` for complete working examples covering:
1. Single ticker prediction
2. Batch predictions
3. Custom data prediction
4. Comparing normalization methods
5. Accessing raw prediction values
6. Ensemble predictions

Run examples:
```bash
python inference_example.py
```

