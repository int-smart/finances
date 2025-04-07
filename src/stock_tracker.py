import pandas as pd
import yfinance as yf
from src.config import COMPANIES, COMMODITIES, REQUEST_DELAY
import time

class StockTracker:
    def __init__(self):
        self.companies = COMPANIES
        self.commodities = COMMODITIES
        self.stock_data = {}
        self.commodity_data = {}
    
    def get_stock_data(self, ticker, period="1y", interval="1d"):
        """Get historical stock price data for a given ticker"""
        print(f"Fetching stock data for {ticker}...")
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
            print(f"Error fetching data for {ticker}: {e}")
            return None
    
    def get_commodity_data(self, commodity, period="1y", interval="1d"):
        """Get historical commodity price data"""
        print(f"Fetching commodity data for {commodity}...")
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
            print(f"Error fetching data for {commodity}: {e}")
            return None
    
    def track(self, period="1y", interval="1d"):
        """Collect data for all configured companies and commodities"""
        # Collect company data
        for ticker in self.companies:
            self.get_stock_data(ticker, period, interval)
        
        # Collect commodity data
        for commodity in self.commodities:
            self.get_commodity_data(commodity, period, interval)

        summary = self.get_summary_stats()
        for ticker in self.companies:
            self.stock_data[ticker].update(summary["stocks"][ticker])
        for commodity in self.commodities:
            self.commodity_data[commodity].update(summary["commodities"][commodity])
        
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
