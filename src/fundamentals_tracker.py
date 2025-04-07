import re
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from src.config import SEC_API_HEADERS, REQUEST_DELAY
import time

class FundamentalsTracker:
    def __init__(self):
        self.fundamentals = {}
    
    def get_financial_ratios(self, ticker):
        """Get key financial ratios for a company"""
        print(f"Fetching financial ratios for {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            
            # Get financial statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cash_flow
            
            # Calculate key ratios
            ratios = {}

            # Only process if we have data
            if not income_stmt.empty and not balance_sheet.empty:
                # import pdb
                # pdb.set_trace()
                # Latest fiscal year
                latest_year = income_stmt.columns[0]
                
                # P/E Ratio
                try:
                    net_income = income_stmt.loc['Net Income', latest_year]
                    market_cap = stock.info.get('marketCap', None)
                    if market_cap and net_income > 0:
                        ratios['P/E'] = market_cap / net_income
                    else:
                        ratios['P/E'] = None
                except:
                    ratios['P/E'] = None
                
                # Debt-to-Equity Ratio
                try:
                    total_debt = balance_sheet.loc['Total Debt', latest_year] if 'Total Debt' in balance_sheet.index else None
                    if total_debt is None:
                        long_term_debt = balance_sheet.loc.get('Long Term Debt', pd.Series([0]))[latest_year]
                        short_term_debt = balance_sheet.loc.get('Short Term Debt', pd.Series([0]))[latest_year]
                        total_debt = long_term_debt + short_term_debt
                    
                    total_equity = balance_sheet.loc["Stockholders Equity", latest_year]
                    if total_equity != 0:
                        ratios['Debt/Equity'] = total_debt / total_equity
                    else:
                        ratios['Debt/Equity'] = None
                except:
                    ratios['Debt/Equity'] = None
                
                # Return on Equity (ROE)
                try:
                    net_income = income_stmt.loc['Net Income', latest_year]
                    total_equity = balance_sheet.loc['Stockholders Equity', latest_year]
                    if total_equity != 0:
                        ratios['ROE'] = net_income / total_equity
                    else:
                        ratios['ROE'] = None
                except:
                    ratios['ROE'] = None
                
                # Profit Margin
                try:
                    net_income = income_stmt.loc['Net Income', latest_year]
                    total_revenue = income_stmt.loc['Total Revenue', latest_year]
                    if total_revenue != 0:
                        ratios['Profit Margin'] = net_income / total_revenue
                    else:
                        ratios['Profit Margin'] = None
                except:
                    ratios['Profit Margin'] = None

            # Get additional info from stock.info
            info_ratios = {
                'Forward P/E': stock.info.get('forwardPE', None),
                'PEG Ratio': stock.info.get('pegRatio', None),
                'Price/Book': stock.info.get('priceToBook', None),
                'Dividend Yield': stock.info.get('dividendYield', None),
                'Beta': stock.info.get('beta', None),
                'EPS': stock.info.get('trailingEps', None)
            }
            
            ratios.update(info_ratios)
            self.fundamentals[ticker] = {
                "ratios": ratios,
                "income_statement": income_stmt,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow
            }
            
            time.sleep(REQUEST_DELAY)  # Avoid rate limiting
            return self.fundamentals[ticker]
        except Exception as e:
            print(f"Error fetching fundamentals for {ticker}: {e}")
            return None
    
    def get_annual_report_links(self, ticker):
        """Get links to the company's annual reports (10-K filings)"""
        print(f"Fetching annual report links for {ticker}...")
        try:
            # Get CIK number
            ticker_url = f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={ticker}&owner=exclude&action=getcompany"
            response = requests.get(ticker_url, headers=SEC_API_HEADERS)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            cik = None
            cik_text = soup.find('span', {'class': 'companyName'})
            if cik_text:
                cik_match = re.search(r'CIK#: (\d+)', cik_text.text)
                if cik_match:
                    cik = cik_match.group(1)
            
            if not cik:
                print(f"Could not find CIK for {ticker}")
                return []
            
            # Get 10-K filings
            edgar_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&count=5"
            response = requests.get(edgar_url, headers=SEC_API_HEADERS)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            links = []
            for row in soup.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) > 2 and '10-K' in cells[0].text:
                    doc_link = cells[1].find('a')['href']
                    filing_date = cells[3].text
                    full_link = f"https://www.sec.gov{doc_link}"
                    links.append({
                        'filing_type': '10-K',
                        'filing_date': filing_date,
                        'link': full_link
                    })
            # Iterate over links and fetch the complete text file
            for link_info in links:
                link_response = requests.get(link_info['link'], headers=SEC_API_HEADERS)
                link_soup = BeautifulSoup(link_response.text, 'html.parser')
                for row in link_soup.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) > 2 and 'txt' in cells[2].text:
                        doc_link = cells[2].find('a')['href']
                    link_response = requests.get(f"https://www.sec.gov{doc_link}", headers=SEC_API_HEADERS)
                    link_info['text_file'] = link_response.text
            
            return links
        except Exception as e:
            print(f"Error fetching annual report links for {ticker}: {e}")
            return []
    
    def analyze_all_companies(self, companies):
        """Analyze fundamentals for all given companies"""
        for ticker in companies:
            self.get_financial_ratios(ticker)
            # self.get_annual_report_links(ticker)
        
        return self.fundamentals
