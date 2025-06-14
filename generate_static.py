import os
import json
import pickle
from datetime import datetime, timedelta
from flask import Flask, render_template
from src.decision_engine import DecisionEngine
from src.storage_helper import GistStorage

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

def load_data_from_cloud():
    """Load latest data from cloud storage"""
    try:
        storage = GistStorage()
    except ValueError as e:
        print(f"Error initializing storage: {e}")
        return {}
    
    data = {}
    data_types = ['stock_data', 'investor_data', 'news_data', 'fundamentals_data', 'recommendations']
    
    for data_type in data_types:
        try:
            cloud_data = storage.download_pickle(data_type)
            if cloud_data:
                data[data_type] = cloud_data
                print(f"Loaded {data_type} from cloud")
            else:
                print(f"Failed to load {data_type} from cloud")
        except Exception as e:
            print(f"Error loading {data_type}: {e}")
    
    return data

def generate_static_site():
    """Generate static HTML files from cloud data"""
    print("Loading data from cloud storage...")
    cloud_data = load_data_from_cloud()
    
    # Initialize Flask app
    app = Flask(__name__)
    
    # Create static directory
    static_dir = "docs"
    os.makedirs(static_dir, exist_ok=True)
    os.makedirs(f"{static_dir}/static", exist_ok=True)
    
    # Extract data
    stock_data = cloud_data.get('stock_data', {})
    investor_data = cloud_data.get('investor_data', {})
    news_data = cloud_data.get('news_data', {})
    fundamentals_data = cloud_data.get('fundamentals_data', {})
    recommendations = cloud_data.get('recommendations', {})

    # Generate news summaries
    news_summaries = {
        company_name: {
            "summary": company_data.get("detailed_news_summary"),
            "positive_factors": company_data.get("news_positive_factors"),
            "negative_factors": company_data.get("news_negative_factors"),
            "score": company_data.get("score"),
        }
        for company_name, company_data in recommendations.items()
    }
    
    # Get data freshness status
    freshness_data = check_data_freshness()
    
    # Load historical recommendations
    historical_recommendations = load_historical_recommendations()
    historical_dates = sorted(historical_recommendations.keys(), reverse=True) if historical_recommendations else []
    
    with app.app_context():
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate index page
        with open(f"{static_dir}/index.html", "w") as f:
            html_content = render_template('static_index.html', 
                                         recommendations=recommendations,
                                         historical_recommendations=historical_recommendations,
                                         historical_dates=historical_dates,
                                         selected_date=None,
                                         last_updated=last_updated,
                                         **freshness_data)
            f.write(html_content)
        
        # Generate stocks page
        with open(f"{static_dir}/stocks.html", "w") as f:
            html_content = render_template('static_stocks.html', 
                                         stocks=stock_data.get('stocks', {}),
                                         news_data=news_data,
                                         last_updated=last_updated)
            f.write(html_content)
        
        # Generate recommendations page
        with open(f"{static_dir}/recommendations.html", "w") as f:
            html_content = render_template('static_recommendations.html', 
                                         recommendations=recommendations,
                                         last_updated=last_updated)
            f.write(html_content)
        
        # Generate fundamentals page
        with open(f"{static_dir}/fundamentals.html", "w") as f:
            html_content = render_template('static_fundamentals.html', 
                                         fundamentals_data=fundamentals_data,
                                         last_updated=last_updated)
            f.write(html_content)
        
        # Generate investors page
        with open(f"{static_dir}/investors.html", "w") as f:
            investor_holdings = {}
            investor_changes = {}
            
            if investor_data:
                if isinstance(investor_data, dict):
                    latest_date = max(investor_data.keys()) if investor_data.keys() else None
                    if latest_date:
                        latest_data = investor_data[latest_date]
                        investor_holdings = latest_data.get('holdings', {})
                        investor_changes = latest_data.get('changes', {})
            
            html_content = render_template('static_investors.html', 
                                         investor_holdings=investor_holdings,
                                         investor_changes=investor_changes,
                                         last_updated=last_updated)
            f.write(html_content)
        
        # Generate news page
        with open(f"{static_dir}/news.html", "w") as f:
            html_content = render_template('static_news.html', 
                                         news_data=news_data,
                                         last_updated=last_updated)
            f.write(html_content)
        
        # Generate news summaries page
        with open(f"{static_dir}/news_summaries.html", "w") as f:
            html_content = render_template('static_news_summaries.html', 
                                         news_summaries=news_summaries,
                                         news_data=news_data,
                                         last_updated=last_updated)
            f.write(html_content)
    
    # Copy static assets (CSS, JS)
    import shutil
    if os.path.exists("static"):
        shutil.copytree("static", f"{static_dir}/static", dirs_exist_ok=True)
    
    print("Static site generated successfully!")
    print(f"Generated pages:")
    print(f"  - index.html")
    print(f"  - stocks.html")
    print(f"  - recommendations.html")
    print(f"  - fundamentals.html")
    print(f"  - investors.html")
    print(f"  - news.html")
    print(f"  - news_summaries.html")

if __name__ == "__main__":
    generate_static_site()