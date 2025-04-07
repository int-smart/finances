import os
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, redirect, url_for, jsonify
from datetime import datetime

# Import your existing modules
from src.investor_tracker import InvestorTracker
from src.news_tracker import NewsTracker
from src.stock_tracker import StockTracker
from src.fundamentals_tracker import FundamentalsTracker
from src.decision_engine import DecisionEngine
from src.config import COMPANIES, INVESTORS

app = Flask(__name__)
app.config['DATA_DIR'] = 'data'

# Helper function to load pickle data
def load_pickle(filename):
    filepath = os.path.join(app.config['DATA_DIR'], filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

# Helper function to check if data is fresh (less than 24 hours old)
def is_data_fresh(filename):
    filepath = os.path.join(app.config['DATA_DIR'], filename)
    if not os.path.exists(filepath):
        return False
    
    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
    current_time = datetime.now()
    time_diff = current_time - file_time
    
    # Return True if file is less than 24 hours old
    return time_diff.total_seconds() < 86400

@app.route('/')
def index():
    """Main dashboard page"""
    # Check if we have fresh data
    stock_data_fresh = is_data_fresh('stock_data.pkl')
    investor_data_fresh = is_data_fresh('investor_data.pkl')
    news_data_fresh = is_data_fresh('news_data.pkl')
    fundamentals_data_fresh = is_data_fresh('fundamentals_data.pkl')
    recommendations_fresh = is_data_fresh('recommendations.pkl')
    
    # Load recommendations if available
    recommendations = load_pickle('recommendations.pkl')
    
    return render_template('index.html',
                          stock_data_fresh=stock_data_fresh,
                          investor_data_fresh=investor_data_fresh,
                          news_data_fresh=news_data_fresh,
                          fundamentals_data_fresh=fundamentals_data_fresh,
                          recommendations_fresh=recommendations_fresh,
                          recommendations=recommendations,
                          tickers=COMPANIES)

@app.route('/refresh_data', methods=['POST'])
def refresh_data():
    """Refresh all data or specific data types"""
    data_type = request.form.get('data_type', 'all')
    
    if data_type == 'stock' or data_type == 'all':
        # Refresh stock data
        stock_tracker = StockTracker()
        stock_data = stock_tracker.track()
        with open(os.path.join(app.config['DATA_DIR'], 'stock_data.pkl'), 'wb') as f:
            pickle.dump(stock_data, f)
    
    if data_type == 'investor' or data_type == 'all':
        # Refresh investor data
        investor_tracker = InvestorTracker()
        investor_tracker.track_all_investors()
        investor_tracker.identify_position_changes()
        investor_tracker.save_data(os.path.join(app.config['DATA_DIR'], 'investor_data.pkl'))
    
    if data_type == 'news' or data_type == 'all':
        # Refresh news data
        news_tracker = NewsTracker()
        news_data = news_tracker.track()
        with open(os.path.join(app.config['DATA_DIR'], 'news_data.pkl'), 'wb') as f:
            pickle.dump(news_data, f)
    
    if data_type == 'fundamentals' or data_type == 'all':
        # Refresh fundamentals data
        fundamentals_tracker = FundamentalsTracker()
        fundamentals_data = fundamentals_tracker.analyze_all_companies(COMPANIES)
        with open(os.path.join(app.config['DATA_DIR'], 'fundamentals_data.pkl'), 'wb') as f:
            pickle.dump(fundamentals_data, f)
    
    if data_type == 'recommendations' or data_type == 'all':
        # Generate new recommendations
        decision_engine = DecisionEngine(
            investor_data=load_pickle('investor_data.pkl'),
            news_data=load_pickle('news_data.pkl'),
            stock_data=load_pickle('stock_data.pkl'),
            fundamentals_data=load_pickle('fundamentals_data.pkl')
        )
        recommendations = decision_engine.generate_recommendations()
        with open(os.path.join(app.config['DATA_DIR'], 'recommendations.pkl'), 'wb') as f:
            pickle.dump(recommendations, f)
    
    return redirect(url_for('index'))

@app.route('/stocks')
def stocks():
    """Stock data page"""
    stock_data = load_pickle('stock_data.pkl')
    return render_template('stocks.html', stock_data=stock_data)

@app.route('/investors')
def investors():
    """Investor data page"""
    investor_data = load_pickle('investor_data.pkl')
    return render_template('investors.html', investor_data=investor_data, investors=INVESTORS)

@app.route('/news')
def news():
    """News data page"""
    news_data = load_pickle('news_data.pkl')
    return render_template('news.html', news_data=news_data)

@app.route('/fundamentals')
def fundamentals():
    """Fundamentals data page"""
    fundamentals_data = load_pickle('fundamentals_data.pkl')
    return render_template('fundamentals.html', fundamentals_data=fundamentals_data)

@app.route('/recommendations')
def recommendations():
    """Recommendations page"""
    recommendations = load_pickle('recommendations.pkl')
    return render_template('recommendations.html', recommendations=recommendations)

@app.route('/stock/<ticker>')
def stock_detail(ticker):
    """Individual stock detail page"""
    stock_data = load_pickle('stock_data.pkl')
    news_data = load_pickle('news_data.pkl')
    fundamentals_data = load_pickle('fundamentals_data.pkl')
    recommendations = load_pickle('recommendations.pkl')
    
    # Get stock-specific data
    ticker_stock_data = stock_data.get('stocks', {}).get(ticker, {}) if stock_data else {}
    ticker_news = news_data.get(ticker, []) if news_data else []
    ticker_fundamentals = fundamentals_data.get(ticker, {}) if fundamentals_data else {}
    ticker_recommendation = recommendations.get(ticker, {}) if recommendations else {}
    
    return render_template('stock_detail.html',
                          ticker=ticker,
                          stock_data=ticker_stock_data,
                          news=ticker_news,
                          fundamentals=ticker_fundamentals,
                          recommendation=ticker_recommendation)

@app.route('/api/stock_chart/<ticker>')
def stock_chart_data(ticker):
    """API endpoint for stock chart data"""
    stock_data = load_pickle('stock_data.pkl')
    if not stock_data or 'stocks' not in stock_data or ticker not in stock_data['stocks']:
        return jsonify({'error': 'Stock data not available'})
    
    ticker_data = stock_data['stocks'][ticker]
    if 'history' not in ticker_data:
        return jsonify({'error': 'Historical data not available'})
    
    # Convert to list of dictionaries for JSON
    history = ticker_data['history']
    if isinstance(history, pd.DataFrame):
        # Reset index to make date a column
        history = history.reset_index()
        # Convert to records format
        chart_data = history.to_dict(orient='records')
    else:
        chart_data = history
    
    return jsonify(chart_data)

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs(app.config['DATA_DIR'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
