import pandas as pd
import yfinance as yf
from src.config import COMPANIES, COMMODITIES, REQUEST_DELAY
import time
import os
from dotenv import load_dotenv
from gist_storage_python import GistStorage
from datetime import datetime
import pickle
import numpy as np

load_dotenv()
class StockTracker:
    def __init__(self):
        self.companies = COMPANIES
        self.commodities = COMMODITIES
        self.stock_data = {}
        self.commodity_data = {}
        self.options_data = {}
    
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
    
    def get_options_data(self, ticker, max_retries=3, retry_delay=5, save_full_chains=True, max_expirations=3):
        """
        Get comprehensive options data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            save_full_chains: Whether to save complete options chains or just summary data
            max_expirations: Maximum number of expirations to process for performance
            
        Returns:
            Dictionary containing comprehensive options data or None if failed
        """
        print(f"Fetching options data for {ticker}...")
        
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                
                # Get current stock price
                try:
                    hist = stock.history(period='1d')
                    if len(hist) > 0:
                        current_price = hist['Close'].iloc[-1]
                    else:
                        print(f"No price data found for {ticker}")
                        return None
                except Exception as e:
                    print(f"Failed to get price for {ticker}: {e}")
                    return None
                
                # Get available expiration dates
                try:
                    expirations = stock.options
                    if len(expirations) == 0:
                        print(f"No options expirations found for {ticker}")
                        return None
                except Exception as e:
                    print(f"Failed to get options expirations for {ticker}: {e}")
                    return None
                
                # Initialize options data structure
                options_data = {
                    'stock_price': current_price,
                    'expirations': list(expirations),
                    'chains': {},
                    'summary': {
                        'atm_call_iv': None,
                        'atm_put_iv': None,
                        'atm_call_strike': None,
                        'atm_put_strike': None,
                        'nearest_expiration': None,
                        'iv_skew': None,  # Difference between ATM call and put IV
                        'total_call_volume': 0,
                        'total_put_volume': 0,
                        'total_call_oi': 0,
                        'total_put_oi': 0
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                # Process each expiration (limit for performance)
                processed_expirations = 0
                total_call_volume = 0
                total_put_volume = 0
                total_call_oi = 0
                total_put_oi = 0
                
                for exp_date in expirations:
                    if processed_expirations >= max_expirations:
                        break
                        
                    try:
                        # Get options chain for this expiration
                        opt_chain = stock.option_chain(exp_date)
                        calls = opt_chain.calls
                        puts = opt_chain.puts
                        
                        if len(calls) == 0 or len(puts) == 0:
                            continue
                        
                        # Calculate volumes and open interest
                        call_volume = calls['volume'].fillna(0).sum()
                        put_volume = puts['volume'].fillna(0).sum()
                        call_oi = calls['openInterest'].fillna(0).sum()
                        put_oi = puts['openInterest'].fillna(0).sum()
                        
                        total_call_volume += call_volume
                        total_put_volume += put_volume
                        total_call_oi += call_oi
                        total_put_oi += put_oi
                        
                        # Store complete chains if requested
                        if save_full_chains:
                            options_data['chains'][exp_date] = {
                                'calls': calls.to_dict('records'),
                                'puts': puts.to_dict('records'),
                                'call_volume': call_volume,
                                'put_volume': put_volume,
                                'call_oi': call_oi,
                                'put_oi': put_oi
                            }
                        
                        # Find ATM options for the first (nearest) expiration
                        if options_data['summary']['atm_call_iv'] is None:
                            # Find strikes closest to current price
                            calls['price_diff'] = abs(calls['strike'] - current_price)
                            puts['price_diff'] = abs(puts['strike'] - current_price)
                            
                            # Get liquid options (volume > 0 or openInterest > 0)
                            liquid_calls = calls[
                                (calls['volume'].fillna(0) > 0) | 
                                (calls['openInterest'].fillna(0) > 0)
                            ]
                            liquid_puts = puts[
                                (puts['volume'].fillna(0) > 0) | 
                                (puts['openInterest'].fillna(0) > 0)
                            ]
                            
                            # Fallback to all options if no liquid ones found
                            if len(liquid_calls) == 0:
                                liquid_calls = calls
                            if len(liquid_puts) == 0:
                                liquid_puts = puts
                            
                            if len(liquid_calls) > 0 and len(liquid_puts) > 0:
                                # Find ATM options
                                atm_call = liquid_calls.loc[liquid_calls['price_diff'].idxmin()]
                                atm_put = liquid_puts.loc[liquid_puts['price_diff'].idxmin()]
                                
                                # Extract implied volatilities
                                if ('impliedVolatility' in atm_call and 'impliedVolatility' in atm_put and
                                    pd.notna(atm_call['impliedVolatility']) and pd.notna(atm_put['impliedVolatility']) and
                                    atm_call['impliedVolatility'] > 0 and atm_put['impliedVolatility'] > 0):
                                    
                                    options_data['summary']['atm_call_iv'] = float(atm_call['impliedVolatility'])
                                    options_data['summary']['atm_put_iv'] = float(atm_put['impliedVolatility'])
                                    options_data['summary']['atm_call_strike'] = float(atm_call['strike'])
                                    options_data['summary']['atm_put_strike'] = float(atm_put['strike'])
                                    options_data['summary']['nearest_expiration'] = exp_date
                                    
                                    # Calculate IV skew (call IV - put IV)
                                    options_data['summary']['iv_skew'] = (
                                        options_data['summary']['atm_call_iv'] - 
                                        options_data['summary']['atm_put_iv']
                                    )
                        
                        processed_expirations += 1
                        
                    except Exception as e:
                        print(f"   Warning: Error processing {exp_date} for {ticker}: {str(e)[:50]}")
                        continue
                
                # Update total volumes and open interest
                options_data['summary']['total_call_volume'] = total_call_volume
                options_data['summary']['total_put_volume'] = total_put_volume
                options_data['summary']['total_call_oi'] = total_call_oi
                options_data['summary']['total_put_oi'] = total_put_oi
                
                # Store the options data
                self.options_data[ticker] = options_data
                
                # Print summary
                if options_data['summary']['atm_call_iv'] is not None:
                    print(f"✅ {ticker}: Call IV = {options_data['summary']['atm_call_iv']:.3f}, "
                          f"Put IV = {options_data['summary']['atm_put_iv']:.3f}, "
                          f"IV Skew = {options_data['summary']['iv_skew']:.3f}, "
                          f"Expirations = {processed_expirations}")
                else:
                    print(f"⚠️  {ticker}: Options data collected but no valid IV found")
                
                time.sleep(REQUEST_DELAY)  # Avoid rate limiting
                return options_data
                
            except Exception as e:
                print(f"Error fetching options data for {ticker} (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to fetch options data for {ticker} after {max_retries} attempts")
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
    
    def track(self, tickers=None, commodities=None, period="1y", interval="1d", 
              include_options=True, save_full_options_chains=True, max_options_expirations=10):
        """
        Track stocks, commodities, and options data.
        
        Args:
            tickers: List of stock tickers to track
            commodities: List of commodities to track
            period: Historical data period
            interval: Data interval
            include_options: Whether to fetch options data for stocks
            save_full_options_chains: Whether to save complete options chains
            max_options_expirations: Max number of option expirations to process
        """
        if not tickers:
            tickers = self.companies
        if not commodities:
            commodities = self.commodities

        # Collect company data
        for ticker in tickers:
            self.get_stock_data(ticker, period, interval)
        
        # Collect options data for stocks (if requested)
        if include_options:
            print(f"\n📊 Fetching options data for {len(tickers)} stocks...")
            for ticker in tickers:
                # Only fetch options if we successfully got stock data
                if ticker in self.stock_data:
                    self.get_options_data(
                        ticker, 
                        save_full_chains=save_full_options_chains,
                        max_expirations=max_options_expirations
                    )
        
        # Collect commodity data
        for commodity in commodities:
            self.get_commodity_data(commodity, period, interval)

        # Get current date for organizing data
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Organize data by date
        return self.organize_data_by_date(current_date, tickers, commodities)
    
    def get_summary_stats(self):
        """Get summary statistics for all collected data"""
        summary = {
            "stocks": {},
            "commodities": {},
            "options": {}
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
        
        # Options summary
        for ticker, data in self.options_data.items():
            if 'summary' in data:
                options_summary = data['summary']
                summary["options"][ticker] = {
                    "atm_call_iv": options_summary.get('atm_call_iv'),
                    "atm_put_iv": options_summary.get('atm_put_iv'),
                    "iv_skew": options_summary.get('iv_skew'),
                    "atm_call_strike": options_summary.get('atm_call_strike'),
                    "atm_put_strike": options_summary.get('atm_put_strike'),
                    "nearest_expiration": options_summary.get('nearest_expiration'),
                    "total_call_volume": options_summary.get('total_call_volume', 0),
                    "total_put_volume": options_summary.get('total_put_volume', 0),
                    "total_call_oi": options_summary.get('total_call_oi', 0),
                    "total_put_oi": options_summary.get('total_put_oi', 0),
                    "put_call_ratio_volume": (
                        options_summary.get('total_put_volume', 0) / 
                        options_summary.get('total_call_volume', 1) 
                        if options_summary.get('total_call_volume', 0) > 0 else None
                    ),
                    "put_call_ratio_oi": (
                        options_summary.get('total_put_oi', 0) / 
                        options_summary.get('total_call_oi', 1) 
                        if options_summary.get('total_call_oi', 0) > 0 else None
                    ),
                    "stock_price": data.get('stock_price'),
                    "expirations_count": len(data.get('expirations', [])),
                    "timestamp": data.get('timestamp')
                }
        
        return summary

    def organize_data_by_date(self, current_date, tickers, commodities):
        """Organize data by date, separating history from daily data"""
        # Load existing data structure if it exists
        existing_data = self.load_existing_stock_data()
        
        # Get summary stats for current data
        summary = self.get_summary_stats()
        
        # Organize stocks data
        for ticker in tickers:
            if ticker in self.stock_data:
                # Extract history and other data
                ticker_data = self.stock_data[ticker]
                history = ticker_data.get('history', pd.DataFrame())
                
                # Initialize ticker in existing data if not present
                if ticker not in existing_data['stocks']:
                    existing_data['stocks'][ticker] = {
                        'history': history,
                        'dates': {}
                    }
                else:
                    # Ensure dates key exists
                    if 'dates' not in existing_data['stocks'][ticker]:
                        existing_data['stocks'][ticker]['dates'] = {}
                    
                    # Append new history data to existing history
                    if not history.empty:
                        existing_history = existing_data['stocks'][ticker].get('history', pd.DataFrame())
                        if not existing_history.empty:
                            # Combine histories, removing duplicates by index (date)
                            combined_history = pd.concat([existing_history, history])
                            combined_history = combined_history[~combined_history.index.duplicated(keep='last')]
                            combined_history = combined_history.sort_index()
                            existing_data['stocks'][ticker]['history'] = combined_history
                        else:
                            existing_data['stocks'][ticker]['history'] = history
                
                # Store current date's data (everything except history)
                current_data = {
                    'info': ticker_data.get('info', {}),
                    'institutional_holders': ticker_data.get('institutional_holders'),
                    'major_holders': ticker_data.get('major_holders'),
                    'summary': summary['stocks'].get(ticker, {}),
                    'timestamp': datetime.now().isoformat()
                }
                
                existing_data['stocks'][ticker]['dates'][current_date] = current_data
        
        # Organize commodities data
        for commodity in commodities:
            if commodity in self.commodity_data:
                commodity_data = self.commodity_data[commodity]
                history = commodity_data.get('history', pd.DataFrame())
                
                # Initialize commodity in existing data if not present
                if commodity not in existing_data['commodities']:
                    existing_data['commodities'][commodity] = {
                        'history': history,
                        'dates': {}
                    }
                else:
                    # Ensure dates key exists
                    if 'dates' not in existing_data['commodities'][commodity]:
                        existing_data['commodities'][commodity]['dates'] = {}
                    
                    # Append new history data to existing history
                    if not history.empty:
                        existing_history = existing_data['commodities'][commodity].get('history', pd.DataFrame())
                        if not existing_history.empty:
                            # Combine histories, removing duplicates by index (date)
                            combined_history = pd.concat([existing_history, history])
                            combined_history = combined_history[~combined_history.index.duplicated(keep='last')]
                            combined_history = combined_history.sort_index()
                            existing_data['commodities'][commodity]['history'] = combined_history
                        else:
                            existing_data['commodities'][commodity]['history'] = history
                
                # Store current date's data
                current_data = {
                    'name': commodity_data.get('name', commodity),
                    'summary': summary['commodities'].get(commodity, {}),
                    'timestamp': datetime.now().isoformat()
                }
                
                existing_data['commodities'][commodity]['dates'][current_date] = current_data
        
        # Store options data by date (full options data each day)
        existing_data['options'][current_date] = self.options_data
        
        return existing_data

    def load_existing_stock_data(self, filepath="data/stock_data.pkl"):
        """Load existing stock data structure"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading existing stock data: {e}")
        
        # Return default structure if file doesn't exist or can't be loaded
        return {
            'stocks': {},
            'commodities': {},
            'options': {}
        }

    def save_latest_data(self, filepath="data/stock_data_latest.pkl"):
        """Save only the latest data including options (for cloud upload)"""
        latest_data = {
            "stocks": self.stock_data,
            "commodities": self.commodity_data,
            "options": self.options_data,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(latest_data, f)
        
        # Upload to cloud storage
        storage = GistStorage(token=os.environ.get('TOKEN_GIST'), repo_owner="int-smart", repo_name="finances")
        storage.upload_pickle(latest_data, 'stock_data')
        
        print(f"Latest stock, commodity, and options data saved and uploaded")
        
        # Print summary statistics
        if self.options_data:
            options_count = len(self.options_data)
            successful_options = sum(1 for data in self.options_data.values() 
                                   if data.get('summary', {}).get('atm_call_iv') is not None)
            print(f"Options data: {successful_options}/{options_count} stocks with valid IV data")
