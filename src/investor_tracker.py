import os
import pickle
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
from src.config import INVESTORS, SEC_API_HEADERS, QUARTERS_TO_TRACK, REQUEST_DELAY
from lxml import etree

class InvestorTracker:
    def __init__(self):
        self.investors = INVESTORS
        self.holdings_data = {}
        self.changes = {}
        
    def get_13f_holdings(self, cik, name):
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
            
            response = requests.get(link, headers=SEC_API_HEADERS)
            if response.status_code != 200:
                print(f"Failed to fetch filing document for {name}. Status code: {response.status_code}")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the XML file
            xml_link = None
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
            all_holdings[quarter_label] = {}

            if xml_link:
                # Get the XML content
                time.sleep(REQUEST_DELAY)  # Respect SEC's rate limits
                response = requests.get(xml_link, headers=SEC_API_HEADERS)
                if response.status_code != 200:
                    print(f"Failed to fetch XML for {name}. Status code: {response.status_code}")
                    continue
                
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
                        if ticker in all_holdings[quarter_label]:
                            all_holdings[quarter_label][ticker]['value'] += value_in_dollar
                            all_holdings[quarter_label][ticker]['shares'] += shares
                        else:
                            all_holdings[quarter_label][ticker] = {
                                'value': value_in_dollar,
                                'shares': shares
                            }

                    # Compute total portfolio value
                    total_value = sum(holding['value'] for holding in all_holdings[quarter_label].values())

                    # Compute percentage per stock and update structure
                    for ticker, data in all_holdings[quarter_label].items():
                        value = data['value']
                        shares = data['shares']
                        percentage = (value / total_value) * 100 if total_value > 0 else 0

                        all_holdings[quarter_label][ticker] = {
                            'value': value,
                            'shares': shares,
                            'percentage': round(percentage, 2)
                        }
                except Exception as e:
                    print(f"Error parsing XML for {name}: {e}")
        
        return all_holdings

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
                
            q1_holdings = quarters.get("Q1", {})
            q2_holdings = quarters.get("Q2", {})
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
        
        data = {
            "holdings": self.holdings_data,
            "changes": self.changes,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
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