import os
import pickle
import pandas as pd
import plotly.express as pxb
import plotly.graph_objects as go
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from datetime import datetime
from src.news_summarizer import NewsSummarizer

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
    
    # Load current recommendations
    recommendations = load_pickle('recommendations.pkl')
    
    # Load historical recommendations
    history_file = os.path.join(app.config['DATA_DIR'], 'recommendations_history.pkl')
    historical_dates = []
    
    if os.path.exists(history_file):
        try:
            with open(history_file, 'rb') as f:
                historical_recommendations = pickle.load(f)
                historical_dates = sorted(historical_recommendations.keys(), reverse=True)
        except Exception as e:
            print(f"Error loading historical recommendations: {e}")
    
    # Get selected date from query parameter, default to most recent
    selected_date = request.args.get('date', None)
    
    # If a date is selected and historical data exists, use that date's recommendations
    if selected_date and os.path.exists(history_file):
        try:
            with open(history_file, 'rb') as f:
                historical_recommendations = pickle.load(f)
                if selected_date in historical_recommendations:
                    recommendations = historical_recommendations[selected_date]
        except Exception as e:
            print(f"Error loading recommendations for date {selected_date}: {e}")
    
    return render_template('index.html',
                          stock_data_fresh=stock_data_fresh,
                          investor_data_fresh=investor_data_fresh,
                          news_data_fresh=news_data_fresh,
                          fundamentals_data_fresh=fundamentals_data_fresh,
                          recommendations_fresh=recommendations_fresh,
                          recommendations=recommendations,
                          historical_dates=historical_dates,
                          selected_date=selected_date,
                          tickers=COMPANIES)

@app.route('/refresh_data', methods=['POST'])
def refresh_data():
    """Refresh all data or specific data types"""
    data_type = request.form.get('data_type', 'all')
    tickers = COMPANIES
    # # Refresh investor data
    # for investor, data in investor_tracker.holdings_data.items():
    #     for quarter, companies in data.items():
    #         for company in companies:
    #             ticker = find_ticker(company)
    #             tickers.append(ticker)
    # print(investor_tracker.holdings_data)
    if data_type == 'investor' or data_type == 'all':
        investor_tracker = InvestorTracker()
        investor_tracker.track_all_investors()
        investor_tracker.identify_position_changes()
        investor_tracker.save_data(os.path.join(app.config['DATA_DIR'], 'investor_data.pkl'))

    if data_type == 'stock' or data_type == 'all':
        # Refresh stock data
        stock_tracker = StockTracker()
        stock_data = stock_tracker.track(tickers=tickers)
        with open(os.path.join(app.config['DATA_DIR'], 'stock_data.pkl'), 'wb') as f:
            pickle.dump(stock_data, f)
        
    if data_type == 'news' or data_type == 'all':
        # Refresh news data
        news_tracker = NewsTracker()
        news_data = news_tracker.track(tickers=tickers)
        news_tracker.save_data(os.path.join(app.config['DATA_DIR'], 'news_data.pkl'))

    
    if data_type == 'fundamentals':
        # Refresh fundamentals data
        fundamentals_tracker = FundamentalsTracker()
        fundamentals_data = fundamentals_tracker.analyze_all_companies(tickers=tickers)
        fundamentals_tracker.save_data(os.path.join(app.config['DATA_DIR'], 'fundamentals_data.pkl'))
    
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
    investor_data_history = load_pickle('investor_data.pkl')
    
    # Get all available dates from the history
    available_dates = []
    if investor_data_history:
        available_dates = sorted(investor_data_history.keys(), reverse=True)
        
    # Get selected date from query parameter, default to most recent
    selected_date = request.args.get('date', None)
    
    # If no date is selected or the selected date doesn't exist, use the most recent
    if not selected_date or selected_date not in available_dates:
        selected_date = available_dates[0] if available_dates else None
        
    # Get the data for the selected date
    investor_data = investor_data_history.get(selected_date, {}) if selected_date else {}
    return render_template('investors.html', 
                          investor_data=investor_data, 
                          investors=INVESTORS,
                          available_dates=available_dates,
                          selected_date=selected_date)

@app.route('/news')
def news():
    """News data page"""
    news_data_history = load_pickle('news_data.pkl')
    
    # Get all available dates from the history
    available_dates = []
    if news_data_history:
        available_dates = sorted(news_data_history.keys(), reverse=True)
        
    # Get selected date from query parameter, default to most recent
    selected_date = request.args.get('date', None)
    
    # If no date is selected or the selected date doesn't exist, use the most recent
    if not selected_date or selected_date not in available_dates:
        selected_date = available_dates[0] if available_dates else None
        
    # Get the data for the selected date
    news_data = news_data_history.get(selected_date, {}) if selected_date else {}
    
    return render_template('news.html', 
                          news_data=news_data,
                          available_dates=available_dates,
                          selected_date=selected_date)

@app.route('/fundamentals')
def fundamentals():
    """Fundamentals data page"""
    fundamentals_data = load_pickle('fundamentals_data.pkl')
    return render_template('fundamentals.html', fundamentals_data=fundamentals_data)

@app.route('/recommendations')
def recommendations():
    """Recommendations page"""
    # Load historical recommendations
    history_file = os.path.join(app.config['DATA_DIR'], 'recommendations_history.pkl')
    historical_dates = []
    
    if os.path.exists(history_file):
        try:
            with open(history_file, 'rb') as f:
                historical_recommendations = pickle.load(f)
                historical_dates = sorted(historical_recommendations.keys(), reverse=True)
        except Exception as e:
            print(f"Error loading historical recommendations: {e}")
    
    # Get selected date from query parameter, default to most recent
    selected_date = request.args.get('date', None)
    
    # Load recommendations based on selected date or default to current
    if selected_date and os.path.exists(history_file):
        try:
            with open(history_file, 'rb') as f:
                historical_recommendations = pickle.load(f)
                if selected_date in historical_recommendations:
                    recommendations = historical_recommendations[selected_date]
                else:
                    recommendations = load_pickle('recommendations.pkl')
        except Exception as e:
            print(f"Error loading recommendations for date {selected_date}: {e}")
            recommendations = load_pickle('recommendations.pkl')
    else:
        recommendations = load_pickle('recommendations.pkl')
    
    return render_template('recommendations.html', 
                          recommendations=recommendations,
                          historical_dates=historical_dates,
                          selected_date=selected_date)

def load_news_summary(ticker):
    """Load news summary for a specific ticker"""
    summary_file = os.path.join(app.config['DATA_DIR'], 'news_summaries', f"{ticker}.pkl")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'rb') as f:
                summaries = pickle.load(f)
                # Get the most recent summary (should be the latest date)
                return summaries
        except Exception as e:
            print(f"Error loading news summary for {ticker}: {e}")
    return None

@app.route('/stock/<ticker>')
def stock_detail(ticker):
    """Stock detail page"""
    # Load stock data
    stock_data = load_pickle('stock_data.pkl')
    
    # Extract data for the specific ticker
    ticker_data = stock_data.get('stocks', {}).get(ticker, {}) if stock_data else {}
    
    # Load news data
    news_data_history = load_pickle('news_data.pkl')
    if news_data_history:
        latest_date = max(news_data_history.keys()) if news_data_history.keys() else None
        news_data = news_data_history[latest_date] if latest_date else {}
    news = news_data.get(ticker, []) if news_data else []
    
    # Load recommendations data
    recommendations = load_pickle('recommendations.pkl')
    recommendation = recommendations.get(ticker, {}) if recommendations else {}
    
    # Load fundamentals data
    fundamentals_data = load_pickle('fundamentals_data.pkl')
    fundamentals = fundamentals_data.get(ticker, {}) if fundamentals_data else {}
    
    # Load news summary
    news_summary = load_news_summary(ticker)

    return render_template('stock_detail.html', 
                          ticker=ticker,
                          stock_data=ticker_data,
                          news=news,
                          recommendation=recommendation,
                          fundamentals=fundamentals,
                          news_summary=news_summary)

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

@app.route('/refresh_summary/<ticker>')
def refresh_summary(ticker):
    """Generate a fresh news summary for a ticker"""
    # Load news data
    news_data_history = load_pickle('news_data.pkl')
    if news_data_history:
        latest_date = max(news_data_history.keys()) if news_data_history.keys() else None
        news_data = news_data_history[latest_date] if latest_date else {}
    articles = news_data.get(ticker, []) if news_data else []
    
    if not articles:
        flash(f"No news articles found for {ticker}.", "warning")
        return redirect(url_for('stock_detail', ticker=ticker))
    
    # Initialize the news summarizer
    summarizer = NewsSummarizer()
    
    # Generate the summary
    summary = summarizer.summarize_news(ticker, articles)
    
    flash(f"News summary for {ticker} has been refreshed.", "success")
    return redirect(url_for('stock_detail', ticker=ticker))

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs(app.config['DATA_DIR'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
