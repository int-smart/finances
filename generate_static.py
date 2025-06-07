import os
import json
import pickle
from datetime import datetime, timedelta
from flask import Flask, render_template
from src.decision_engine import DecisionEngine
from src.run import main as run_data_collection

def check_data_freshness():
    """Check if data files are fresh (updated within last 24 hours)"""
    data_files = {
        'stock_data': 'data/stock_data.pkl',
        'investor_data': 'data/investor_data.pkl', 
        'news_data': 'data/news_data.pkl',
        'fundamentals_data': 'data/fundamentals_data.pkl',
        'recommendations': 'data/recommendations.pkl'
    }
    
    freshness = {}
    current_time = datetime.now()
    
    for data_type, file_path in data_files.items():
        if os.path.exists(file_path):
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            freshness[f"{data_type}_fresh"] = (current_time - file_time) < timedelta(hours=24)
        else:
            freshness[f"{data_type}_fresh"] = False
    
    return freshness

def load_historical_recommendations():
    """Load historical recommendations data"""
    history_file = "data/recommendations_history.pkl"
    if os.path.exists(history_file):
        try:
            with open(history_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading historical recommendations: {e}")
    return {}

def generate_static_site():
    """Generate static HTML files from the Flask app"""
    
    # First, run the data collection
    print("Running data collection...")
    run_data_collection()
    
    # Initialize Flask app
    app = Flask(__name__)
    
    # Load the latest data
    engine = DecisionEngine()
    engine.load_data()
    
    # Create static directory
    static_dir = "docs"  # GitHub Pages serves from docs/ folder
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(f"{static_dir}/static", exist_ok=True)
    
    # Get data freshness status
    freshness_data = check_data_freshness()
    
    # Load historical data
    historical_recommendations = load_historical_recommendations()
    historical_dates = sorted(historical_recommendations.keys(), reverse=True)
    
    with app.app_context():
        # Generate index page
        recommendations = getattr(engine, 'recommendations', {})
        with open(f"{static_dir}/index.html", "w") as f:
            html_content = render_template('static_index.html', 
                                         recommendations=recommendations,
                                         historical_recommendations=historical_recommendations,
                                         historical_dates=historical_dates,
                                         selected_date=None,
                                         last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                         **freshness_data)
            f.write(html_content)
        
        # Generate stocks page
        stock_data = getattr(engine, 'stock_data', {})
        with open(f"{static_dir}/stocks.html", "w") as f:
            html_content = render_template('static_stocks.html', 
                                         stocks=stock_data.get('stocks', {}),
                                         last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            f.write(html_content)
        
        # Generate recommendations page
        with open(f"{static_dir}/recommendations.html", "w") as f:
            html_content = render_template('static_recommendations.html', 
                                         recommendations=recommendations,
                                         last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            f.write(html_content)
    
    # Copy static assets (CSS, JS)
    import shutil
    if os.path.exists("static"):
        shutil.copytree("static", f"{static_dir}/static", dirs_exist_ok=True)
    
    print("Static site generated successfully!")

if __name__ == "__main__":
    generate_static_site()