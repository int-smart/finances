import os
import pickle
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from src.config import INVESTORS, SEC_API_HEADERS, QUARTERS_TO_TRACK, REQUEST_DELAY
from lxml import etree
from src.storage_helper import GistStorage

class InvestorTracker:
    def __init__(self):
        self.investors = INVESTORS
        self.holdings_data = {}
        self.changes = {}
    
    def get_13f_holdings(self, cik, name):
        """
        Get 13F filings for a specific investor by CIK number,
        properly handling both 13F-HR and 13F-HR/A filings
        """
        print(f"Fetching holdings for {name}...")
        
        # First, use the RSS feed approach for more reliable extraction
        rss_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=13F-HR&count={QUARTERS_TO_TRACK*10}&output=atom"
        response = requests.get(rss_url, headers=SEC_API_HEADERS)
        
        if response.status_code != 200:
            print(f"Failed to fetch RSS data for {name}. Status code: {response.status_code}")
            print("Falling back to traditional EDGAR search...")
            return self._get_13f_holdings_legacy(cik, name)
        
        # Parse the RSS feed
        filings_by_quarter = {}
        try:
            root = etree.fromstring(response.text.encode())
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            # Extract all entries from the RSS feed
            entries = root.findall(".//atom:entry", namespaces=ns)
            
            for entry in entries:
                # Get filing type, date, and link information
                try:
                    filing_type = entry.find(".//atom:category", namespaces=ns).get('term')
                    filing_date = entry.findtext(".//atom:updated", namespaces=ns)
                    if filing_date:
                        filing_date = filing_date.split('T')[0]  # Get just the date portion
                        
                    # Only process 13F-HR and 13F-HR/A filings
                    if filing_type and ('13F-HR' in filing_type):
                        links = entry.findall(".//atom:link", namespaces=ns)
                        filing_url = None
                        for link in links:
                            if link.get('rel') == 'alternate':
                                filing_url = link.get('href')
                                break
                        
                        if filing_url:
                            # Determine the quarter from the filing date
                            if filing_date:
                                try:
                                    date_obj = datetime.strptime(filing_date, '%Y-%m-%d')
                                    if 'A' in filing_type:
                                        quarter = f"{date_obj.year}-Q{(date_obj.month-1)//3}"
                                    else:
                                        quarter = f"{date_obj.year}-Q{(date_obj.month-1)//3 + 1}"
                                    
                                    # Initialize the quarter entry if it doesn't exist
                                    if quarter not in filings_by_quarter:
                                        filings_by_quarter[quarter] = {
                                            'original': None,
                                            'amendment': None
                                        }
                                    
                                    # Store both original and amended filings
                                    if 'A' in filing_type:  # It's an amendment
                                        filings_by_quarter[quarter]['amendment'] = {
                                            'url': filing_url,
                                            'date': filing_date,
                                            'type': filing_type
                                        }
                                    else:  # It's an original filing
                                        filings_by_quarter[quarter]['original'] = {
                                            'url': filing_url,
                                            'date': filing_date,
                                            'type': filing_type
                                        }
                                except ValueError:
                                    print(f"Could not parse date: {filing_date}")
                except Exception as e:
                    print(f"Error processing entry: {e}")
                    continue
            
            # Process the most recent quarters
            all_holdings = {}
            quarters = sorted(filings_by_quarter.keys(), reverse=True)[:QUARTERS_TO_TRACK]
            
            for i, quarter in enumerate(quarters):
                quarter_filings = filings_by_quarter[quarter]
                time.sleep(REQUEST_DELAY)  # Respect SEC's rate limits
                
                quarter_label = quarter
                all_holdings[quarter_label] = {}
                
                # Process amendment first if available
                if quarter_filings['amendment']:
                    xml_link = self.find_xml_link(name, quarter_filings['amendment']['url'])
                    if xml_link:
                        amended_holdings = self.process_xml_holding(name, xml_link)
                        # Store these as amended holdings
                        for ticker, data in amended_holdings.items():
                            if ticker not in all_holdings[quarter_label]:
                                all_holdings[quarter_label][ticker] = {}
                            all_holdings[quarter_label][ticker] = data.copy()
                            all_holdings[quarter_label][ticker]['amended'] = True
                
                # Process original filing if available
                if quarter_filings['original']:
                    xml_link = self.find_xml_link(name, quarter_filings['original']['url'])
                    if xml_link:
                        original_holdings = self.process_xml_holding(name, xml_link)
                        # For original holdings, only add tickers that weren't in the amendment
                        for ticker, data in original_holdings.items():
                            if ticker not in all_holdings[quarter_label]:
                                all_holdings[quarter_label][ticker] = data.copy()
                                all_holdings[quarter_label][ticker]['amended'] = False
                
                # If we couldn't get either filing, report empty holdings
                if not all_holdings[quarter_label]:
                    print(f"Warning: No valid holdings found for {name} in {quarter}")
            
            return all_holdings
            
        except Exception as e:
            print(f"Error processing RSS feed for {name}: {e}")
            # Fall back to traditional EDGAR search
            return self._get_13f_holdings_legacy(cik, name)
    def _get_13f_holdings_legacy(self, cik, name):
        """
        Get 13F filings for a specific investor by CIK number
        """
        print(f"Fetching holdings for {name}...")
        base_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=13F-HR&count={QUARTERS_TO_TRACK*10}"
        response = requests.get(base_url, headers=SEC_API_HEADERS)
        if response.status_code != 200:
            print(f"Failed to fetch data for {name}. Status code: {response.status_code}")
            return {}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract links to 13F-HR filings
        filing_links = []
        for link in soup.find_all('a', href=True):
            if 'Documents' in link.text and 'Archives' in link['href']:
                filing_links.append('https://www.sec.gov' + link['href'])
        
        if not filing_links:
            print(f"No 13F-HR filings found for {name}.")
            return {}
        
        # Process each filing to extract holdings
        all_holdings = {}

        # import pdb
        # pdb.set_trace()
        for i, link in enumerate(filing_links[:QUARTERS_TO_TRACK]):
            # Get the document page
            time.sleep(REQUEST_DELAY)  # Respect SEC's rate limits
            # quarter_label = f"Q{link.split('/')[-1].split('-')[1]}-{link.split('/')[-1].split('-')[2][:3]}"            
            quarter_label = f"Q{i+1}"            
            xml_link = self.find_xml_link(name, link)
            all_holdings[quarter_label] = {}

            if xml_link:
                all_holdings[quarter_label] = self.process_xml_holding(name, xml_link)
        
        return all_holdings
    
    def find_xml_link(self, name, link):
        # Find the XML file
        xml_link = None
        response = requests.get(link, headers=SEC_API_HEADERS)
        if response.status_code != 200:
            print(f"Failed to fetch filing document for {name}. Status code: {response.status_code}")
            return xml_link
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for link in soup.find_all('a', href=True):
            if link.text.endswith('.xml') and 'infotable' in link.text.lower():
                xml_link = 'https://www.sec.gov' + link['href']
                break
        if xml_link is None:
            # Find all table rows
            rows = soup.find_all('tr')
            for row in rows:
                # Check if this row contains "INFORMATION TABLE" text
                row_text = row.get_text().lower()
                if 'information table' in row_text:
                    # Look for XML links within this row
                    xml_links = row.find_all('a', href=True)
                    for link in xml_links:
                        if link.text.endswith('.xml'):
                            xml_link = 'https://www.sec.gov' + link['href']
                            break
                    if xml_link:
                        break
        return xml_link
    
    def process_xml_holding(self, name, xml_link):
        quarter_holdings = {}
        # Get the XML content
        time.sleep(REQUEST_DELAY)  # Respect SEC's rate limits
        response = requests.get(xml_link, headers=SEC_API_HEADERS)
        if response.status_code != 200:
            print(f"Failed to fetch XML for {name}. Status code: {response.status_code}")
            return quarter_holdings
        
        try:
            # Namespace from the XML root
            # Parse XML with lxml
            root = etree.fromstring(response.text.encode())
            ns = {'ns': root.nsmap[None]}  # Pull default namespace

            # Find all <infoTable> entries
            entries = root.findall(".//ns:infoTable", namespaces=ns)

            data = []

            for entry in entries:
                row = {
                    "nameOfIssuer": entry.findtext("ns:nameOfIssuer", namespaces=ns),
                    "titleOfClass": entry.findtext("ns:titleOfClass", namespaces=ns),
                    "cusip": entry.findtext("ns:cusip", namespaces=ns),
                    "value": int(entry.findtext("ns:value", default="0", namespaces=ns)),
                    "sshPrnamt": int(entry.find("ns:shrsOrPrnAmt/ns:sshPrnamt", namespaces=ns).text),
                    "sshPrnamtType": entry.find("ns:shrsOrPrnAmt/ns:sshPrnamtType", namespaces=ns).text,
                    "investmentDiscretion": entry.findtext("ns:investmentDiscretion", namespaces=ns),
                    "otherManager": entry.findtext("ns:otherManager", namespaces=ns),
                    "votingSole": int(entry.find("ns:votingAuthority/ns:Sole", namespaces=ns).text),
                    "votingShared": int(entry.find("ns:votingAuthority/ns:Shared", namespaces=ns).text),
                    "votingNone": int(entry.find("ns:votingAuthority/ns:None", namespaces=ns).text)
                }
                data.append(row)

            holdings_data = pd.DataFrame(data)
            # Convert to a more manageable format
            # Aggregate holdings
            for _, row in holdings_data.iterrows():
                ticker = row.get('nameOfIssuer', 'Unknown')
                value_in_dollar = row.get('value', 0)
                shares = row.get('sshPrnamt', 0)

                # If already exists, sum both value and shares
                if ticker in quarter_holdings:
                    quarter_holdings[ticker]['value'] += value_in_dollar
                    quarter_holdings[ticker]['shares'] += shares
                else:
                    quarter_holdings[ticker] = {
                        'value': value_in_dollar,
                        'shares': shares
                    }

            # Compute total portfolio value
            total_value = sum(holding['value'] for holding in quarter_holdings.values())

            # Compute percentage per stock and update structure
            for ticker, data in quarter_holdings.items():
                value = data['value']
                shares = data['shares']
                percentage = (value / total_value) * 100 if total_value > 0 else 0

                quarter_holdings[ticker] = {
                    'value': value,
                    'shares': shares,
                    'percentage': round(percentage, 2)
                }
        except Exception as e:
            print(f"Error parsing XML for {name}: {e}")
        return quarter_holdings
            
    def get_10k_holdings(self, cik, name):
        """Get 10-K holdings for a given investor"""
        print(f"Fetching 10-K data for {name}...")
        try:
            filings = yf.Ticker(cik).filings
            for filing in filings:
                if filing.type == "10-K":
                    return filing.accessionNumber
        except Exception as e:
            print(f"Error fetching 10-K data for {name}: {e}")
        return None
    
    def track_all_investors(self):
        """
        Track holdings for all configured investors
        """
        for name, cik in self.investors.items():
            try:
                self.holdings_data[name] = self.get_13f_holdings(cik, name)
                time.sleep(REQUEST_DELAY)  # Avoid rate limits
            except Exception as e:
                print(f"Error tracking {name}: {e}")
        
        return self.holdings_data
    
    def identify_position_changes(self):
        """
        Identify changes in investor positions between the most recent quarters
        """
        for investor, quarters in self.holdings_data.items():
            self.changes[investor] = []
            
            if len(quarters) < 2:
                continue
                
            sorted_quarters = sorted(quarters.keys(), reverse=True)
            q1_holdings = quarters.get(sorted_quarters[0], {})
            q2_holdings = quarters.get(sorted_quarters[1], {})

            # Find new positions
            for ticker, data in q1_holdings.items():
                shares = data.get('shares', 0)
                if ticker not in q2_holdings:
                    self.changes[investor].append({
                        "ticker": ticker,
                        "change_type": "NEW POSITION",
                        "old_position": 0,
                        "new_position": shares
                    })
                else:
                    # Check for significant changes (more than 10%)
                    old_shares = q2_holdings[ticker].get('shares', 0)
                    if shares > 0 and old_shares > 0:
                        percent_change = (shares - old_shares) / old_shares * 100
                        if abs(percent_change) > 0:
                            self.changes[investor].append({
                                "ticker": ticker,
                                "change_type": "INCREASE" if percent_change > 0 else "DECREASE",
                                "old_position": old_shares,
                                "new_position": shares,
                                "percent_change": round(percent_change, 2)
                            })
            
            # Find exited positions
            for ticker, data in q2_holdings.items():
                shares = data.get('shares', 0)
                if ticker not in q1_holdings:
                    self.changes[investor].append({
                        "ticker": ticker,
                        "change_type": "EXITED POSITION",
                        "old_position": shares,
                        "new_position": 0
                    })
        
        return self.changes
    
    def save_data(self, filepath="data/investor_data.pkl"):
        """Save the tracked data to a pickle file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        new_data = {
            current_date: {
                "holdings": self.holdings_data,
                "changes": self.changes,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                existing_data = pickle.load(f)
                existing_data.update(new_data)
            with open(filepath, 'wb') as f:
                pickle.dump(existing_data, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(new_data, f)
        
        print(f"Data saved to {filepath}")
        
    def load_data(self, filepath="data/investor_data.pkl"):
        """Load previously saved data"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.holdings_data = data.get("holdings", {})
                self.changes = data.get("changes", {})
                print(f"Data loaded from {filepath}")
                return True
        except (FileNotFoundError, pickle.UnpicklingError):
            print(f"No valid data found at {filepath}")
            return False

    def save_latest_data(self, filepath="data/investor_data_latest.pkl"):
        """Save only the latest data (for cloud upload)"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        latest_data = {
            current_date: {
                "holdings": self.holdings_data,
                "changes": self.changes,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(latest_data, f)
        
        # Upload to cloud storage
        storage = GistStorage()
        storage.upload_pickle(latest_data, 'investor_data')
        
        print(f"Latest investor data saved and uploaded")