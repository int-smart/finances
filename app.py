import os
import pickle
import pandas as pd
import plotly.express as pxb
import plotly.graph_objects as go
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from datetime import datetime
from src.news_summarizer import NewsSummarizer
from src.reddit_tracker import RedditTracker

# Import your existing modules
from src.investor_tracker import InvestorTracker
from src.news_tracker import NewsTracker
from src.stock_tracker import StockTracker
from src.fundamentals_tracker import FundamentalsTracker
from src.decision_engine import DecisionEngine
from src.config import COMPANIES, INVESTORS
from src.strategies.strategy_manager import StrategyManager
from gist_storage_python import GistStorage

app = Flask(__name__)
app.config['DATA_DIR'] = 'data'

# Initialize strategy manager
strategy_manager = StrategyManager(data_dir=app.config['DATA_DIR'])

# Helper function to load pickle data
def load_pickle(filename):
    filepath = os.path.join(app.config['DATA_DIR'], filename)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

# Helper function to convert new stock data structure to old format for templates
def convert_stock_data_for_template(stock_data, selected_date=None):
    """Convert new date-based stock data structure to format expected by templates"""
    if not stock_data or 'stocks' not in stock_data:
        return stock_data
    
    converted_data = {'stocks': {}, 'commodities': {}, 'options': {}}
    
    # Convert stocks
    for ticker, ticker_data in stock_data['stocks'].items():
        if 'dates' in ticker_data and ticker_data['dates']:
            # Use selected date or most recent date
            if selected_date and selected_date in ticker_data['dates']:
                date_to_use = selected_date
            else:
                date_to_use = max(ticker_data['dates'].keys())
            
            latest_data = ticker_data['dates'][date_to_use]
            
            # Combine with history for the template
            converted_data['stocks'][ticker] = {
                'history': ticker_data.get('history', pd.DataFrame()),
                **latest_data
            }
    
    # Convert commodities
    if 'commodities' in stock_data:
        for commodity, commodity_data in stock_data['commodities'].items():
            if 'dates' in commodity_data and commodity_data['dates']:
                # Use selected date or most recent date
                if selected_date and selected_date in commodity_data['dates']:
                    date_to_use = selected_date
                else:
                    date_to_use = max(commodity_data['dates'].keys())
                
                latest_data = commodity_data['dates'][date_to_use]
                
                # Combine with history for the template
                converted_data['commodities'][commodity] = {
                    'history': commodity_data.get('history', pd.DataFrame()),
                    **latest_data
                }
    
    # Convert options (use selected date or most recent)
    if 'options' in stock_data:
        if selected_date and selected_date in stock_data['options']:
            converted_data['options'] = stock_data['options'][selected_date]
        elif stock_data['options']:
            latest_options_date = max(stock_data['options'].keys())
            converted_data['options'] = stock_data['options'][latest_options_date]
    
    return converted_data

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
    reddit_data_fresh = is_data_fresh('reddit_data.pkl')
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
                          reddit_data_fresh=reddit_data_fresh,
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

    if data_type == 'reddit' or data_type == 'all':
        # Refresh Reddit data
        reddit_tracker = RedditTracker()
        reddit_data = reddit_tracker.track_all_tickers(tickers=tickers)
        reddit_tracker.save_data(os.path.join(app.config['DATA_DIR'], 'reddit_data.pkl'), reddit_data)
        
        # Also analyze popular subreddits
        subreddit_data = reddit_tracker.analyze_popular_subreddits(['wallstreetbets', 'stocks'])
        reddit_tracker.save_data(os.path.join(app.config['DATA_DIR'], 'subreddit_data.pkl'), subreddit_data)
    
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
            fundamentals_data=load_pickle('fundamentals_data.pkl'),
            reddit_data=load_pickle('reddit_data.pkl')
        )
        recommendations = decision_engine.generate_recommendations()
        with open(os.path.join(app.config['DATA_DIR'], 'recommendations.pkl'), 'wb') as f:
            pickle.dump(recommendations, f)
    
    return redirect(url_for('index'))

@app.route('/refresh_gist', methods=['POST'])
def refresh_gist():
    """Upload all current data to gist storage"""
    try:
        storage = GistStorage(token=os.environ.get('TOKEN_GIST'), repo_owner="int-smart", repo_name="finances")
        
        # Upload stock data from latest file
        stock_data_latest = load_pickle('stock_data_latest.pkl')
        if stock_data_latest:
            storage.upload_pickle(stock_data_latest, 'stock_data')
        
        # Upload investor data from latest file
        investor_data_latest = load_pickle('investor_data_latest.pkl')
        if investor_data_latest:
            storage.upload_pickle(investor_data_latest, 'investor_data')
        
        # Upload news data from latest file
        news_data_latest = load_pickle('news_data_latest.pkl')
        if news_data_latest:
            storage.upload_pickle(news_data_latest, 'news_data')
        
        # Upload fundamentals data from latest file
        fundamentals_data_latest = load_pickle('fundamentals_data_latest.pkl')
        if fundamentals_data_latest:
            storage.upload_pickle(fundamentals_data_latest, 'fundamentals_data')
        
        # Upload recommendations from latest file
        recommendations_latest = load_pickle('recommendations_latest.pkl')
        if recommendations_latest:
            storage.upload_pickle(recommendations_latest, 'recommendations')
        
        print("All data uploaded to gist storage successfully")
        
    except Exception as e:
        print(f"Error uploading data to gist: {e}")
    
    return redirect(url_for('index'))

@app.route('/stocks')
def stocks():
    """Stock data page"""
    stock_data = load_pickle('stock_data.pkl')
    
    # Get available dates
    available_dates = []
    if stock_data and 'stocks' in stock_data:
        # Get all unique dates from all tickers
        all_dates = set()
        for ticker_data in stock_data['stocks'].values():
            if 'dates' in ticker_data:
                all_dates.update(ticker_data['dates'].keys())
        available_dates = sorted(all_dates, reverse=True)
    
    # Get selected date from query parameter
    selected_date = request.args.get('date', None)
    if not selected_date and available_dates:
        selected_date = available_dates[0]
    
    # Convert new data structure to format expected by template
    converted_stock_data = convert_stock_data_for_template(stock_data, selected_date)
    
    return render_template('stocks.html', 
                          stock_data=converted_stock_data,
                          available_dates=available_dates,
                          selected_date=selected_date)

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

@app.route('/reddit')
def reddit():
    """Reddit data page"""
    reddit_data_history = load_pickle('reddit_data.pkl')
    
    # Get all available dates from the history
    available_dates = []
    if reddit_data_history:
        available_dates = sorted(reddit_data_history.keys(), reverse=True)
        
    # Get selected date from query parameter, default to most recent
    selected_date = request.args.get('date', None)
    
    # If no date is selected or the selected date doesn't exist, use the most recent
    if not selected_date or selected_date not in available_dates:
        selected_date = available_dates[0] if available_dates else None
        
    # Get the data for the selected date
    reddit_data = reddit_data_history.get(selected_date, {}) if selected_date else {}
    
    return render_template('reddit.html', 
                          reddit_data=reddit_data,
                          available_dates=available_dates,
                          selected_date=selected_date)

@app.route('/subreddits')
def subreddits():
    """Subreddit analysis page"""
    subreddit_data_history = load_pickle('subreddit_data.pkl')
    
    # Get all available dates from the history
    available_dates = []
    if subreddit_data_history:
        available_dates = sorted(subreddit_data_history.keys(), reverse=True)
        
    # Get selected date from query parameter, default to most recent
    selected_date = request.args.get('date', None)
    
    # If no date is selected or the selected date doesn't exist, use the most recent
    if not selected_date or selected_date not in available_dates:
        selected_date = available_dates[0] if available_dates else None
        
    # Get the data for the selected date
    subreddit_data = subreddit_data_history.get(selected_date, {}) if selected_date else {}
    
    return render_template('subreddits.html', 
                          subreddit_data=subreddit_data,
                          available_dates=available_dates,
                          selected_date=selected_date)

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

@app.route('/trading-strategies')
def trading_strategies():
    """Trading strategies page"""
    return render_template('trading_strategies.html', 
                         tickers=COMPANIES)

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

def load_reddit_summary(ticker):
    """Load Reddit summary for a specific ticker"""
    summary_file = os.path.join(app.config['DATA_DIR'], 'reddit_summaries', f"{ticker}.pkl")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'rb') as f:
                summaries = pickle.load(f)
                # Get the most recent summary (should be the latest date)
                return summaries
        except Exception as e:
            print(f"Error loading Reddit summary for {ticker}: {e}")
    return None

@app.route('/stock/<ticker>')
def stock_detail(ticker):
    """Stock detail page"""
    # Load stock data
    stock_data = load_pickle('stock_data.pkl')
    
    # Convert new data structure to format expected by template
    converted_stock_data = convert_stock_data_for_template(stock_data)
    
    # Extract data for the specific ticker
    ticker_data = converted_stock_data.get('stocks', {}).get(ticker, {}) if converted_stock_data else {}
    
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
    
    # Load Reddit data
    reddit_data_history = load_pickle('reddit_data.pkl')
    reddit_posts = []
    if reddit_data_history:
        latest_date = max(reddit_data_history.keys()) if reddit_data_history.keys() else None
        reddit_data = reddit_data_history[latest_date] if latest_date else {}
        ticker_reddit_data = reddit_data.get(ticker, {})
        if 'posts' in ticker_reddit_data:
            reddit_posts = ticker_reddit_data['posts']
    
    # Load Reddit summary
    reddit_summary = load_reddit_summary(ticker)
    
    # Load strategy data for charts
    strategy_data = strategy_manager.get_strategy_data_for_ticker(ticker)
    
    return render_template('stock_detail.html', 
                          ticker=ticker,
                          stock_data=ticker_data,
                          news=news,
                          recommendation=recommendation,
                          fundamentals=fundamentals,
                          news_summary=news_summary,
                          reddit_posts=reddit_posts,
                          reddit_summary=reddit_summary,
                          strategy_data=strategy_data)

@app.route('/api/stock_chart/<ticker>')
def stock_chart_data(ticker):
    """API endpoint for stock chart data"""
    stock_data = load_pickle('stock_data.pkl')
    if not stock_data or 'stocks' not in stock_data or ticker not in stock_data['stocks']:
        return jsonify({'error': 'Stock data not available'})
    
    ticker_info = stock_data['stocks'][ticker]
    if 'history' not in ticker_info:
        return jsonify({'error': 'Historical data not available'})
    
    # Convert to list of dictionaries for JSON
    history = ticker_info['history']
    if isinstance(history, pd.DataFrame):
        # Remove duplicates and sort by date
        history = history.drop_duplicates()
        history = history.sort_index()
        
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

@app.route('/refresh_reddit_summary/<ticker>')
def refresh_reddit_summary(ticker):
    """Generate a fresh Reddit summary for a ticker"""
    # Load Reddit data
    reddit_data_history = load_pickle('reddit_data.pkl')
    if reddit_data_history:
        latest_date = max(reddit_data_history.keys()) if reddit_data_history.keys() else None
        reddit_data = reddit_data_history[latest_date] if latest_date else {}
    
    ticker_reddit_data = reddit_data.get(ticker, {})
    posts = ticker_reddit_data.get('posts', []) if ticker_reddit_data else []
    
    if not posts:
        flash(f"No Reddit posts found for {ticker}.", "warning")
        return redirect(url_for('stock_detail', ticker=ticker))
    
    # Initialize the Reddit tracker which will handle summarization
    reddit_tracker = RedditTracker()
    
    # Generate a new summary using the tracker's built-in functionality
    summary = reddit_tracker.generate_ticker_summary(ticker, posts)
    
    # Save the summary manually
    current_date = datetime.now().strftime("%Y-%m-%d")
    summary_data = {
        current_date: {
            "summary": summary.summary,
            "sentiment": summary.sentiment,
            "price_impact": summary.price_impact,
            "positive_factors": [{"factor": f.get("factor", ""), "metrics": f.get("metrics", "")} for f in summary.positive_factors],
            "negative_factors": [{"factor": f.get("factor", ""), "metrics": f.get("metrics", "")} for f in summary.negative_factors],
            "earnings_boosters": summary.earnings_boosters,
            "earnings_sinks": summary.earnings_sinks,
            "source": "Reddit",
            "total_posts": summary.total_posts,
            "subreddits": list(summary.subreddit_breakdown.keys()),
            "avg_score": sum(p.score for p in posts) / len(posts) if posts else 0,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }
    
    # Save to file
    import os
    summary_file = os.path.join(app.config['DATA_DIR'], 'reddit_summaries', f"{ticker}.pkl")
    os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'rb') as f:
                existing_data = pickle.load(f)
                existing_data.update(summary_data)
        except:
            existing_data = summary_data
        with open(summary_file, 'wb') as f:
            pickle.dump(existing_data, f)
    else:
        with open(summary_file, 'wb') as f:
            pickle.dump(summary_data, f)
    
    flash(f"Reddit summary for {ticker} has been refreshed.", "success")
    return redirect(url_for('stock_detail', ticker=ticker))

@app.route('/api/strategies/<ticker>')
def api_get_strategy_data(ticker):
    """API endpoint to get saved strategy data for a ticker"""
    try:
        strategy_name = request.args.get('strategy', None)
        strategy_type = request.args.get('type', None)
        
        data = strategy_manager.get_strategy_data_for_ticker(
            ticker, strategy_name, strategy_type
        )
        
        if 'error' in data:
            return jsonify({'success': False, 'error': data['error']})
        
        return jsonify({'success': True, 'data': data})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/strategies/signals')
def api_get_all_signals():
    """API endpoint to get current signals for all tickers"""
    try:
        date = request.args.get('date', None)
        
        if date:
            # Get signals for specific date
            historical_data = strategy_manager.get_strategy_data_for_date(date)
            if historical_data and 'strategies' in historical_data:
                signals = {}
                for strategy_name, strategy_data in historical_data['strategies'].items():
                    signals[strategy_name] = {}
                    for strategy_type, type_data in strategy_data.items():
                        signals[strategy_name][strategy_type] = {}
                        for ticker, ticker_data in type_data.items():
                            if ticker_data.get('success'):
                                signals[strategy_name][strategy_type][ticker] = ticker_data.get('current_signal', 'hold')
                return jsonify({'success': True, 'signals': signals, 'date': date})
            else:
                return jsonify({'success': False, 'error': f'No strategy data found for date {date}'})
        else:
            # Get latest signals
            signals = strategy_manager.get_all_current_signals()
            return jsonify({'success': True, 'signals': signals})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/strategies/dates')
def api_get_strategy_dates():
    """API endpoint to get available strategy data dates"""
    try:
        dates = strategy_manager.get_available_strategy_dates()
        return jsonify({'success': True, 'dates': dates})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/strategies/calculate', methods=['POST'])
def api_calculate_strategy():
    """API endpoint for on-demand strategy calculation with custom parameters"""
    try:
        data = request.get_json()
        
        ticker = data.get('ticker', 'AAPL')
        strategy_name = data.get('strategy_name', 'moving_average')
        strategy_type = data.get('strategy_type', 'double')
        period = data.get('period', '1y')
        
        # Extract parameters based on strategy type
        parameters = {}
        
        if strategy_name == 'moving_average':
            if strategy_type == 'single':
                parameters['ema_window'] = int(data.get('ema_window', 20))
            elif strategy_type == 'double':
                parameters['short_window'] = int(data.get('short_window', 12))
                parameters['long_window'] = int(data.get('long_window', 26))
            elif strategy_type == 'triple':
                parameters['window1'] = int(data.get('window1', 5))
                parameters['window2'] = int(data.get('window2', 12))
                parameters['window3'] = int(data.get('window3', 26))
        
        # Optional parameters
        if 'alpha' in data and data['alpha']:
            parameters['alpha'] = float(data['alpha'])
        
        # Calculate strategy on-demand
        result = strategy_manager.calculate_strategy_on_demand(
            ticker, strategy_name, strategy_type, parameters, period
        )
        
        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']})
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs(app.config['DATA_DIR'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
