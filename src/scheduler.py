#!/usr/bin/env python3
import os
import pickle
import time
import pandas as pd
from datetime import datetime, timedelta
from src.investor_tracker import InvestorTracker
from src.stock_tracker import StockTracker
from src.news_tracker import NewsTracker
from src.fundamentals_tracker import FundamentalsTracker
from src.decision_engine import DecisionEngine
from src.config import COMPANIES, INVESTORS
import numpy as np
import schedule
class TaskScheduler:
    """Scheduler for investment research tasks that can run on demand or scheduled"""
    
    def __init__(self, load_cached=False, skip_news=False, tickers=None, output_dir="output"):
        """Initialize the task scheduler with configuration options"""
        self.load_cached = load_cached
        self.skip_news = skip_news
        self.tickers = tickers if tickers else COMPANIES
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize trackers and engines
        self.investor_tracker = InvestorTracker()
        self.stock_tracker = StockTracker()
        self.news_tracker = NewsTracker(tickers=self.tickers)
        self.fundamentals_tracker = FundamentalsTracker()
        self.decision_engine = DecisionEngine()
        
        # Data storage
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)
        
        print(f"TaskScheduler initialized with: tickers={self.tickers}")
    
    def track_investor_positions(self):
        """Track investor positions and identify changes"""
        investor_data = {}
        
        if self.load_cached:
            print("Loading cached investor data...")
            loaded = self.investor_tracker.load_data()
            if loaded:
                investor_data = {
                    "holdings": self.investor_tracker.holdings_data,
                    "changes": self.investor_tracker.changes
                }
        
        if not investor_data:
            print("Tracking investor positions...")
            self.investor_tracker.track_all_investors()
            self.investor_tracker.identify_position_changes()
            self.investor_tracker.save_data(f"{self.data_dir}/investor_data.pkl")
            
            investor_data = {
                "holdings": self.investor_tracker.holdings_data,
                "changes": self.investor_tracker.changes
            }
        
        return investor_data

    def collect_stock_data(self):
        """Collect stock price data and statistics"""
        print("Collecting stock data...")
        stock_data = self.stock_tracker.track(tickers=self.tickers)
        
        with open(f"{self.data_dir}/stock_data.pkl", 'wb') as f:
            pickle.dump(stock_data, f)
        
        return stock_data
    
    def collect_fundamentals(self):
        """Collect and analyze company fundamentals"""
        print("Analyzing company fundamentals...")
        fundamentals_data = self.fundamentals_tracker.analyze_all_companies(tickers=self.tickers)
        self.fundamentals_tracker.save_data(f"{self.data_dir}/fundamentals_data.pkl")
        return fundamentals_data
    
    def collect_news(self):
        """Collect and track news articles"""
        news_data = {}
        
        if not self.skip_news:
            print("Collecting news articles...")
            news_data = self.news_tracker.track(tickers=self.tickers)
            self.news_tracker.save_data(f"{self.data_dir}/news_data.pkl")
        else:
            print("Skipping news collection as requested.")
        
        return news_data

    def generate_recommendations(self, investor_data, stock_data, fundamentals_data, news_data):
        """Generate investment recommendations based on collected data"""
        print("Generating investment recommendations...")
        self.decision_engine.investor_data = investor_data
        self.decision_engine.news_data = news_data
        self.decision_engine.stock_data = stock_data
        self.decision_engine.fundamentals_data = fundamentals_data
    
        # Generate recommendations incorporating news summaries
        recommendations = self.decision_engine.generate_recommendations()
        
        with open(f"{self.data_dir}/recommendations.pkl", 'wb') as f:
            pickle.dump(recommendations, f)
        
        return recommendations
    
    def generate_report(self, investor_data, stock_data, fundamentals_data, news_data, recommendations):
        """Generate a detailed investment report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.output_dir, f"investment_report_{timestamp}.txt")
        
        with open(report_file, 'w') as f:
            f.write("INVESTMENT RESEARCH REPORT\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 80 + "\n\n")
            
            f.write("INVESTMENT RECOMMENDATIONS:\n")
            for ticker, rec in recommendations.items():
                f.write(f"\n{ticker}: {rec['final_recommendation']} (Score: {rec['score']})\n")
                f.write(f"  Investor Consensus: {rec['investor_consensus']}\n")
                f.write(f"  News Sentiment: {rec['news_sentiment']}\n")
                f.write(f"  Valuation: {rec['valuation']}\n")
                f.write(f"  Financial Health: {rec['financial_health']}\n")
                f.write(f"  Price Trend: {rec['price_trend']}\n")
                
                # Include news summary if available
                if 'detailed_news_summary' in rec:
                    f.write(f"  News Summary: {rec['detailed_news_summary']}\n")
                    if 'news_positive_factors' in rec:
                        f.write("  Positive Factors:\n")
                        for factor in rec['news_positive_factors']:
                            f.write(f"    - {factor}\n")
                    if 'news_negative_factors' in rec:
                        f.write("  Negative Factors:\n")
                        for factor in rec['news_negative_factors']:
                            f.write(f"    - {factor}\n")
            
            f.write("\n" + "-" * 80 + "\n")
            f.write("INVESTOR POSITION CHANGES:\n")
            for investor, changes in investor_data.get('changes', {}).items():
                if changes:
                    f.write(f"\n{investor}:\n")
                    for change in changes:
                        f.write(f"  {change['ticker']}: {change['change_type']} - Old: {change['old_position']}, New: {change['new_position']}\n")
                else:
                    f.write(f"\n{investor}: No significant changes\n")
            
            # Include stock data summary
            if stock_data and 'stocks' in stock_data:
                f.write("\n" + "-" * 80 + "\n")
                f.write("STOCK DATA SUMMARY:\n")
                for ticker, data in stock_data['stocks'].items():
                    if 'summary' in data:
                        f.write(f"\n{ticker}:\n")
                        for key, value in data['summary'].items():
                            if isinstance(value, float):
                                f.write(f"  {key}: {value:.2f}\n")
                            else:
                                f.write(f"  {key}: {value}\n")
        
        print(f"\nFull report saved to: {report_file}")
        return report_file
    
    def run_daily_tasks(self):
        """Run tasks that should happen daily"""
        print(f"Running daily tasks at {datetime.now()}")
        
        # 1. Collect stock data
        stock_data = self.collect_stock_data()
        
        # 2. Collect news
        news_data = self.collect_news()
        
        # 3. Load cached investor and fundamentals data
        try:
            with open(f"{self.data_dir}/investor_data.pkl", 'rb') as f:
                investor_data = pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError):
            investor_data = {}
        
        try:
            with open(f"{self.data_dir}/fundamentals_data.pkl", 'rb') as f:
                fundamentals_data = pickle.load(f)
        except (FileNotFoundError, pickle.UnpicklingError):
            fundamentals_data = {}
        
        # 4. Generate recommendations
        recommendations = self.generate_recommendations(
            investor_data, stock_data, fundamentals_data, news_data
        )
        
        # 5. Generate report
        self.generate_report(
            investor_data, stock_data, fundamentals_data, news_data, recommendations
        )
        
        print(f"Daily tasks completed at {datetime.now()}")
    
    def run_monthly_tasks(self):
        if datetime.now().day != 1:
            return
        """Run tasks that should happen monthly"""
        print(f"Running monthly tasks at {datetime.now()}")
        
        # 1. Track investor positions
        investor_data = self.track_investor_positions()
        
        # 2. Collect fundamentals
        fundamentals_data = self.collect_fundamentals()
        
        # 3. Collect stock data
        stock_data = self.collect_stock_data()
        
        # 4. Collect news
        news_data = self.collect_news()
        
        # 5. Generate recommendations
        recommendations = self.generate_recommendations(
            investor_data, stock_data, fundamentals_data, news_data
        )
        
        # 6. Generate report
        self.generate_report(
            investor_data, stock_data, fundamentals_data, news_data, recommendations
        )
        
        print(f"Monthly tasks completed at {datetime.now()}")
    
    def run_all_tasks(self):
        """Run all tasks in one go (for one-time execution)"""
        print(f"Running full analysis at {datetime.now()}")
        
        # 1. Track investor positions
        investor_data = self.track_investor_positions()

        # 2. Collect stock data
        stock_data = self.collect_stock_data()
        
        # 3. Collect fundamentals
        fundamentals_data = self.collect_fundamentals()
        
        # 4. Collect news
        news_data = self.collect_news()
        
        # 5. Generate recommendations
        recommendations = self.generate_recommendations(
            investor_data, stock_data, fundamentals_data, news_data
        )
        
        # 6. Generate report
        self.generate_report(
            investor_data, stock_data, fundamentals_data, news_data, recommendations
        )
        
        print(f"Full analysis completed at {datetime.now()}")
    
    def schedule_tasks(self):
        """Schedule daily and monthly tasks with startup compensation"""
        # Check if we missed the daily run due to system being powered off
        if self.check_missed_daily_run():
            self.run_daily_tasks()
        
        # Schedule daily tasks at 8 AM
        schedule.every().day.at("08:00").do(self.run_daily_tasks)
        
        # Schedule monthly tasks on the 1st of each month
        schedule.every().day.at("10:00").do(self.run_monthly_tasks)
        
        print("Scheduled tasks:")
        print("- Daily tasks at 8:00 AM")
        print("- Monthly tasks on the 1st of each month at 10:00 AM")
        print("- Startup compensation for missed daily runs")
        
        print("Scheduler running continuously. Press Ctrl+C to exit.")
        
        # Keep the scheduler running
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            print("Scheduler stopped by user.")
