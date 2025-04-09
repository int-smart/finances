import os
from dotenv import load_dotenv
from together import Together
import re
import json
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from src.config import SEC_API_HEADERS, REQUEST_DELAY
import time

class FundamentalsTracker:
    def __init__(self):
        self.fundamentals = {}
        self.links = {}
        load_dotenv()
        api_key = os.environ.get("TOGETHER_API_KEY")
        if not api_key:
            print("Warning: TOGETHER_API_KEY environment variable not set. Summarization may not work.")
        self.client = Together(api_key=api_key)
        # self.model = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
        self.model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
    
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
                latest_year.strftime('%Y-%m-%d'): {
                    "ratios": ratios,
                    "income_statement": income_stmt,
                    "balance_sheet": balance_sheet,
                    "cash_flow": cash_flow
                }
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
            
            if ticker not in self.links:
                self.links[ticker] = []
            
            for row in soup.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) > 2 and '10-K' in cells[0].text:
                    doc_link = cells[1].find('a')['href']
                    filing_date = cells[3].text
                    full_link = f"https://www.sec.gov{doc_link}"
                    self.links[ticker].append({
                        'filing_type': '10-K',
                        'filing_date': filing_date,
                        'link': full_link
                    })
            # Iterate over links and fetch the complete text file
            for link_info in self.links[ticker]:
                link_response = requests.get(link_info['link'], headers=SEC_API_HEADERS)
                link_soup = BeautifulSoup(link_response.text, 'html.parser')
                for row in link_soup.find_all('tr'):
                    cells = row.find_all('td')
                    if len(cells) > 2 and 'txt' in cells[2].text:
                        doc_link = cells[2].find('a')['href']
                        link_response = requests.get(f"https://www.sec.gov{doc_link}", headers=SEC_API_HEADERS)
                        link_info['text_file'] = link_response.text
            
            return self.links[ticker]
        except Exception as e:
            print(f"Error fetching annual report links for {ticker}: {e}")
            return []
    
    def summarize_10k(self, ticker):
        """Extract and summarize substantive text from a 10-K filing."""
        if ticker not in self.links or "text_file" not in self.links[ticker][0]:
            print(f"No 10-K filings found for {ticker}")
            return None
        
        # Get raw text
        raw_text = self.links[ticker][0]['text_file']
        
        # Extract all substantive content
        all_substantive_blocks = []
        
        # More precise regex to find the 10-K document section
        # This looks for the structure shown in SEC filings where TYPE is 10-K
        # Find the first DOCUMENT section that contains TYPE 10-K
        # Split the raw text into document sections
        document_sections = re.findall(r'<DOCUMENT>.*?</DOCUMENT>', raw_text, re.DOTALL | re.IGNORECASE)

        # Find the first section that contains TYPE 10-K
        ten_k_document = None
        for section in document_sections:
            if re.search(r'<TYPE>10-K', section, re.IGNORECASE):
                ten_k_document = section
                break
        
        if ten_k_document:
            # We found the 10-K document section
            doc_section = ten_k_document  # This gives everything inside the DOCUMENT tags
            
            # Look for HTML content within the 10-K document
            html_match = re.search(r'<html.*?>(.*?)</html>', doc_section, re.DOTALL | re.IGNORECASE)
            
            if html_match:
                # We found HTML content - parse it
                html_content = html_match.group(0)  # Get the entire HTML including tags
                soup = BeautifulSoup(html_content, 'html.parser')
            else:
                # No HTML found, use the whole document section
                soup = BeautifulSoup(doc_section, 'html.parser')
            
            # Extract substantive text
            threshold = 5  # Minimum number of words
            
            # Find all tags with text
            for tag in soup.find_all('span', recursive=True):
                # Get text and check if it's substantial
                text = tag.get_text().strip()
                if text and len(text.split()) >= threshold:
                    all_substantive_blocks.append(text)
        else:
            print(f"No 10-K document section found for {ticker}")
            return None
        
        print(f"Found {len(all_substantive_blocks)} substantive text blocks")
        
        # Combine the blocks, but limit total size to avoid token limits
        combined_text = ""
        
        for block in all_substantive_blocks:
            combined_text += block + "\n\n"
        
        if not combined_text:
            print(f"No substantive text blocks found for {ticker}")
            return None
        
        # Generate prompt for summarization
        prompt = f"""
        Analyze the following content from a 10-K filing for {ticker} and provide a summary focusing on:
        
        1. Risks involved in the business
        2. Positive factors
        3. Earning boosters
        4. Earning sinks
        
        Here is the text from the 10-K:
        {combined_text}
        
        Please format your response as JSON with the following structure:
        {{
            "risks": ["risk1", "risk2", ...],
            "positive_factors": ["factor1", "factor2", ...],
            "earning_boosters": ["booster1", "booster2", ...],
            "earning_sinks": ["sink1", "sink2", ...]
        }}
        """
        
        try:
            # Generate the summary using the LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = response.choices[0].message.content
            
            # Remove <think>...</think> blocks
            text = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
            
            # Check if there's a JSON code block
            json_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text, re.DOTALL)
            
            if json_block_match:
                # Extract the content inside the code block
                json_str = json_block_match.group(1).strip()
            else:
                # If no code block, use the entire text after removing think tags
                json_str = text.strip()
            
            # Try to parse as JSON
            try:
                if not self.links[ticker][0]['filing_date'] in self.fundamentals[ticker]:
                    self.fundamentals[ticker][self.links[ticker][0]['filing_date']] = {}
                self.fundamentals[ticker][self.links[ticker][0]['filing_date']]['10K_summary'] = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                return {"text": text, "error": str(e)}
            
        except Exception as e:
            return {"summary": f"Error generating summary: {str(e)}", "error": str(e)}
    
    def extract_substantive_text(self, soup, threshold=5):
        """Extract all text blocks with substantial content from a BeautifulSoup object."""
        substantive_blocks = []
        
        # Find all tags with text
        for tag in soup.find_all(True, recursive=True):
            # Skip script, style, etc.
            if tag.name in ['script', 'style', 'meta', 'link', 'head']:
                continue
            
            # Get text and check if it's substantial
            text = tag.get_text().strip()
            if text and len(text.split()) >= threshold:
                # Avoid duplicates
                if not any(text in block for block in substantive_blocks):
                    substantive_blocks.append(text)
        
        return substantive_blocks

    def analyze_all_companies(self, companies):
        """Analyze fundamentals for all given companies"""
        for ticker in companies:
            self.get_financial_ratios(ticker)
            self.get_annual_report_links(ticker)
            self.summarize_10k(ticker)
        
        return self.fundamentals
