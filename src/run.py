#!/usr/bin/env python3
import os
import argparse
from dotenv import load_dotenv
from src.scheduler import TaskScheduler
from datetime import datetime

def parse_arguments():
    """Parse command line arguments for application configuration"""
    parser = argparse.ArgumentParser(description='Investment Research Tool')
    parser.add_argument('--load-cached', action='store_true', 
                        help='Load cached investor data instead of fetching new data')
    parser.add_argument('--skip-news', action='store_true',
                        help='Skip news scraping (which can be slow)')
    parser.add_argument('--tickers', type=str, 
                        help='Comma-separated list of tickers to analyze (overrides config)')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for reports')
    parser.add_argument('--schedule', action='store_true',
                        help='Run in scheduled mode instead of one-time execution')
    
    return parser.parse_args()

def setup_environment():
    """Set up environment variables and directories"""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if Together API key is set
    if not os.environ.get("TOGETHER_API_KEY"):
        print("Warning: TOGETHER_API_KEY environment variable not set. News summarization may not work.")
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    os.makedirs("data", exist_ok=True)

def main():
    """Main entry point for the application"""
    # Setup environment
    setup_environment()
    
    # Parse command line arguments
    args = parse_arguments()
    
    if args.schedule:
        # Run in scheduled mode
        print(f"Starting scheduled tasks at {datetime.now()}")
        scheduler = TaskScheduler(
            load_cached=args.load_cached,
            skip_news=args.skip_news,
            tickers=args.tickers.split(',') if args.tickers else None,
            output_dir=args.output
        )
        scheduler.schedule_tasks()
    else:
        # Run one-time execution
        print(f"Running one-time analysis at {datetime.now()}")
        scheduler = TaskScheduler(
            load_cached=args.load_cached,
            skip_news=args.skip_news,
            tickers=args.tickers.split(',') if args.tickers else None,
            output_dir=args.output
        )
        scheduler.run_all_tasks()

if __name__ == "__main__":
    main()
