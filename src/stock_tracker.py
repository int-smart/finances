import pandas as pd
import yfinance as yf
from src.config import COMPANIES, COMMODITIES, REQUEST_DELAY
import time
import os
from dotenv import load_dotenv

load_dotenv()
class StockTracker:
    def __init__(self):
        self.companies = COMPANIES
        self.commodities = COMMODITIES
        self.stock_data = {}
        self.commodity_data = {}
    
    def get_stock_data(self, ticker, period="1y", interval="1d", max_retries=3, retry_delay=5):
        """Get historical stock price data for a given ticker"""
        import requests

        def fetch_from_fmp(ticker):
            api_key = os.environ.get("FMP_API_KEY")  # Replace with your actual API key
            base_url = "https://financialmodelingprep.com/stable"
            # Historical price endpoint
            hist_url = f"{base_url}/historical-price-eod/full?symbol=^{ticker}&serietype=line&apikey={api_key}"
            info_url = f"{base_url}/sec-profile?symbol={ticker}&apikey={api_key}"
            # holders_url = f"{base_url}/institutional-holder/{ticker}?apikey={api_key}"

            hist_resp = requests.get(hist_url)
            info_resp = requests.get(info_url)
            # holders_resp = requests.get(holders_url)

            if hist_resp.status_code != 200 or info_resp.status_code != 200:
                raise Exception("FMP API error")

            hist_data = hist_resp.json().get("historical", [])
            info_data = info_resp.json()[0] if info_resp.json() else {}
            # holders_data = holders_resp.json() if holders_resp.status_code == 200 else []

            # FMP does not provide major holders, so leave as None
            return {
                "history": hist_data,
                "info": info_data,
                "institutional_holders": None,
                "major_holders": None
            }

        print(f"Fetching stock data for {ticker}...")
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                
                hist = stock.history(period=period, interval=interval)
                
                # Get additional info
                info = stock.info
                
                # Get institutional and major holders
                institutional_holders = stock.institutional_holders
                major_holders = stock.major_holders
                
                self.stock_data[ticker] = {
                    "history": hist,
                    "info": info,
                    "institutional_holders": institutional_holders,
                    "major_holders": major_holders
                }
                
                time.sleep(REQUEST_DELAY)  # Avoid rate limiting
                return self.stock_data[ticker]
            except Exception as e:
                print(f"Error fetching data for {ticker} (attempt {attempt + 1}/{max_retries}): {e}")
                print("Attempting to fetch data from Financial Modeling Prep API...")
                try:
                    fmp_data = fetch_from_fmp(ticker)
                    self.stock_data[ticker] = fmp_data
                    time.sleep(REQUEST_DELAY)
                    return self.stock_data[ticker]
                except Exception as fmp_e:
                    print(f"FMP API error for {ticker}: {fmp_e}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        return None
    
    def get_commodity_data(self, commodity, period="1y", interval="1d", max_retries=3, retry_delay=5):
        """Get historical commodity price data"""
        print(f"Fetching commodity data for {commodity}...")
        for attempt in range(max_retries):
            try:
                data = yf.Ticker(commodity)
                hist = data.history(period=period, interval=interval)
                
                self.commodity_data[commodity] = {
                    "history": hist,
                    "name": data.info.get('shortName', commodity)
                }
                
                time.sleep(REQUEST_DELAY)  # Avoid rate limiting
                return self.commodity_data[commodity]
            except Exception as e:
                print(f"Error fetching data for {commodity} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    return None
    
    def track(self, tickers=None, commodities=None, period="1y", interval="1d"):
        if not tickers:
            tickers = self.companies
        if not commodities:
            commodities = self.commodities

        # Collect company data
        for ticker in tickers:
            self.get_stock_data(ticker, period, interval)
        
        # Collect commodity data
        for commodity in commodities:
            self.get_commodity_data(commodity, period, interval)

        summary = self.get_summary_stats()
        for ticker in tickers:
            if ticker in self.stock_data and ticker in summary["stocks"]:
                self.stock_data[ticker].update(summary["stocks"][ticker])
            elif ticker not in self.stock_data:
                print(f"Warning: {ticker} not found in stock_data")
            elif ticker not in summary["stocks"]:
                print(f"Warning: {ticker} not found in summary stats")
        for commodity in commodities:
            if commodity in self.commodity_data and commodity in summary["commodities"]:
                self.commodity_data[commodity].update(summary["commodities"][commodity])
            elif commodity not in self.commodity_data:
                print(f"Warning: {commodity} not found in commodity_data")
            elif commodity not in summary["commodities"]:
                print(f"Warning: {commodity} not found in summary stats")
        
        return {
            "stocks": self.stock_data,
            "commodities": self.commodity_data
        }
    
    def get_summary_stats(self):
        """Get summary statistics for all collected data"""
        summary = {
            "stocks": {},
            "commodities": {}
        }
        
        # Stock summary
        for ticker, data in self.stock_data.items():
            if 'history' in data and not data['history'].empty:
                hist = data['history']
                latest_close = hist['Close'].iloc[-1] if not hist.empty else None
                change_1d = hist['Close'].pct_change().iloc[-1] * 100 if len(hist) > 1 else None
                change_30d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-min(30, len(hist))]) * 100 - 100 if len(hist) > 30 else None
                
                summary["stocks"][ticker] = {
                    "latest_price": latest_close,
                    "1d_change_%": change_1d,
                    "30d_change_%": change_30d,
                    "52w_high": hist['High'].max(),
                    "52w_low": hist['Low'].min(),
                    "volume": hist['Volume'].iloc[-1]
                }
        
        # Commodity summary
        for commodity, data in self.commodity_data.items():
            if 'history' in data and not data['history'].empty:
                hist = data['history']
                latest_close = hist['Close'].iloc[-1] if not hist.empty else None
                change_1d = hist['Close'].pct_change().iloc[-1] * 100 if len(hist) > 1 else None
                change_30d = (hist['Close'].iloc[-1] / hist['Close'].iloc[-min(30, len(hist))]) * 100 - 100 if len(hist) > 30 else None
                
                summary["commodities"][commodity] = {
                    "latest_price": latest_close,
                    "1d_change_%": change_1d,
                    "30d_change_%": change_30d,
                    "52w_high": hist['High'].max(),
                    "52w_low": hist['Low'].min(),
                    "volume": hist['Volume'].iloc[-1]
                }
        
        return summary
