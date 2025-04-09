# Financial Investment Analyzer

A comprehensive web application for analyzing stocks, generating investment recommendations, and tracking financial performance using technical indicators and fundamental analysis.

## 🚀 Features

- **Stock Data Dashboard**: Track and visualize key stock metrics including price, volumeb, and historical performance
- **Investment Recommendations**: Get actionable BUY, SELL, or HOLD recommendations based on multiple factors
- **Technical Analysis**: Leverage powerful indicators including:
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Moving Averages (50-day, 200-day)
  - Volume Analysis
  - Beta Calculation
  - Pattern Recognition (Double Bottom)
- **Fundamental Analysis**: Evaluate stocks based on:
  - Investor Consensus
  - News Sentiment
  - Valuation Metrics
  - Financial Health
- **Portfolio Management**: Track your investments and monitor performance

## 📊 Screenshots

![Dashboard](docs/images/dashboard.png)
![Recommendations](docs/images/recommendations.png)
![Stock Analysis](docs/images/stock_analysis.png)

## 🔧 Installation

### Option 1: Using pip (requirements.txt)

1. Clone the repository:
   ```bash
   git clone https://github.com/int-smart/finances.git
   cd finances
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using conda (environment.yml)

1. Clone the repository:
   ```bash
   git clone https://github.com/int-smart/finances.git
   cd finances
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate finance-analysis  # The name specified in environment.yml
   ```

### Configuration

1. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

2. Run the application:
   ```bash
   python app.py
   ```
   or schedule it with
   ```bash
   python -m src.run --schedule
   ```

3. Open your browser and navigate to `http://localhost:5000`

## 📦 Dependencies

The application requires several Python packages:

- **Web Framework**: Flask
- **Data Processing**: pandas, numpy, scipy
- **Data Visualization**: matplotlib, seaborn
- **Financial APIs**: yfinance, alpha_vantage
- **Machine Learning**: scikit-learn
- **Database**: SQLAlchemy

For the complete list of dependencies:
- See `requirements.txt` for pip installation
- See `environment.yml` for conda installation

## 📝 Usage

### Viewing Stock Data

Navigate to the Stocks page to see a summary of all tracked stocks. Click on any ticker symbol to view detailed information about that specific stock.

### Generating Recommendations

1. Go to the Recommendations page
2. Click "Refresh Recommendations" to generate new investment recommendations
3. Review the recommendations table which includes:
   - Overall recommendation (BUY, SELL, HOLD)
   - Technical indicators
   - Fundamental metrics
   - Score ranking

### Technical Analysis

The application calculates various technical indicators to help you make informed decisions:

- **RSI**: Values below 30 indicate oversold conditions (potential buy), values above 70 indicate overbought conditions (potential sell)
- **MACD**: Bullish crossovers suggest upward momentum, bearish crossovers suggest downward momentum
- **Bollinger Bands**: Price at lower band may indicate oversold conditions, price at upper band may indicate overbought conditions

## 🔄 Data Sources

The application uses multiple data sources to provide comprehensive analysis:
- Market price data
- Financial statements
- Analyst recommendations
- News sentiment analysis
- Technical indicators

## 🛠️ Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, JavaScript, Bootstrap
- **Data Processing**: Pandas, NumPy
- **Data Visualization**: Chart.js
- **Database**: SQLite/PostgreSQL

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

If you have any questions or feedback, please open an issue on GitHub or contact the repository owner.

---

**Disclaimer**: This application is for informational purposes only and does not constitute investment advice. Always conduct your own research before making investment decisions.