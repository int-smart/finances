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
from src.storage_helper import GistStorage
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
        
        # File to track last daily run
        self.last_run_file = os.path.join(self.data_dir, "last_daily_run.txt")
        
        print(f"TaskScheduler initialized with: tickers={self.tickers}")
    
    def check_missed_daily_run(self):
        """
        Check if we missed the daily run due to system being powered off or other issues.
        Returns True if we missed a daily run, False otherwise.
        """
        try:
            if not os.path.exists(self.last_run_file):
                print("No previous daily run record found. Assuming first run.")
                return True
            
            # Read the last run timestamp
            with open(self.last_run_file, 'r') as f:
                last_run_str = f.read().strip()
            
            if not last_run_str:
                print("Empty last run file. Assuming missed run.")
                return True
            
            # Parse the timestamp
            try:
                last_run_time = datetime.fromisoformat(last_run_str)
            except ValueError:
                # Try parsing with different format if needed
                try:
                    last_run_time = datetime.strptime(last_run_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    print(f"Could not parse last run timestamp: {last_run_str}")
                    return True
            
            current_time = datetime.now()
            time_since_last_run = current_time - last_run_time
            
            # Check if more than 25 hours have passed (giving 1 hour buffer)
            # This accounts for the daily run scheduled at 8 AM
            if time_since_last_run > timedelta(hours=25):
                print(f"Missed daily run detected. Last run: {last_run_time}, Current time: {current_time}")
                print(f"Time since last run: {time_since_last_run}")
                return True
            else:
                print(f"Daily run is up to date. Last run: {last_run_time}")
                return False
                
        except Exception as e:
            print(f"Error checking last daily run: {e}")
            # If there's any error, assume we missed a run to be safe
            return True
    
    def update_last_daily_run(self):
        """Update the timestamp of the last daily run"""
        try:
            current_time = datetime.now()
            with open(self.last_run_file, 'w') as f:
                f.write(current_time.isoformat())
            print(f"Updated last daily run timestamp: {current_time.isoformat()}")
        except Exception as e:
            print(f"Error updating last daily run timestamp: {e}")
    
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
        
        try:
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
            
            # 6. Update the last run timestamp
            self.update_last_daily_run()
            
            # 7. Save and upload latest data
            self.save_all_latest_data()
            print(f"Daily tasks completed successfully at {datetime.now()}")
            
        except Exception as e:
            print(f"Error during daily tasks: {e}")
            # Still update timestamp even if there were errors, to avoid repeated retries
            self.update_last_daily_run()
    
    def save_all_latest_data(self):
        """Save and upload all latest data"""
        print("Saving and uploading latest data...")
        
        # Save latest data from each tracker
        if hasattr(self, 'stock_tracker'):
            self.stock_tracker.save_latest_data()
        
        if hasattr(self, 'investor_tracker'):
            self.investor_tracker.save_latest_data()
        
        if hasattr(self, 'news_tracker'):
            self.news_tracker.save_latest_data()
        
        if hasattr(self, 'fundamentals_tracker'):
            self.fundamentals_tracker.save_latest_data()
        
        # Upload recommendations
        if hasattr(self, 'decision_engine') and self.decision_engine.recommendations:
            with open('data/recommendations_latest.pkl', 'wb') as f:
                pickle.dump(self.decision_engine.recommendations, f)
            storage = GistStorage()
            storage.upload_pickle(self.decision_engine.recommendations, 'recommendations')
            
    def run_monthly_tasks(self):
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

        # 6. Update the last run timestamp
        self.update_last_daily_run()

        # 7. Save and upload latest data
        self.save_all_latest_data()
        print(f"Monthly tasks completed at {datetime.now()}")
    
    def run_all_tasks(self):
        """Run all tasks in one go (for one-time execution)"""
        print(f"Running full analysis at {datetime.now()}")
        
        # 1. Track investor positions
        # investor_data = self.track_investor_positions()

        # # 2. Collect stock data
        # stock_data = self.collect_stock_data()
        
        # # 3. Collect fundamentals
        # fundamentals_data = self.collect_fundamentals()
        
        # # 4. Collect news
        # news_data = self.collect_news()
        
        # # 5. Generate recommendations
        # recommendations = self.generate_recommendations(
        #     investor_data, stock_data, fundamentals_data, news_data
        # )
        
        # # 6. Generate report
        # self.generate_report(
        #     investor_data, stock_data, fundamentals_data, news_data, recommendations
        # )
        
        # 7. Save and upload latest data
        self.save_all_latest_data()
        
        # 8. Update the last run timestamp
        # self.update_last_daily_run()
        
        print(f"Full analysis completed at {datetime.now()}")
    
    def schedule_tasks(self):
        """Schedule daily and monthly tasks with startup compensation"""
        # Check if we missed the daily run due to system being powered off
        if self.check_missed_daily_run():
            print("Missed daily run detected. Running daily tasks now...")
            self.run_daily_tasks()
    
        if datetime.now().day == 1:
            # Schedule monthly tasks on the 1st of each month
            schedule.every().day.at("10:00").do(self.run_monthly_tasks)
        else:
            # Schedule daily tasks at 8 AM
            schedule.every().day.at("08:00").do(self.run_daily_tasks)
        
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
