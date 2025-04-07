import os
import json
import pickle
import pandas as pd
from datetime import datetime

def convert_json_to_pickle():
    """Convert JSON data files to pickle format for the web app"""
    data_dir = "data"
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Map of JSON files to their pickle equivalents
    file_map = {
        'investor_data.json': 'investor_data.pkl',
        'stock_data.json': 'stock_data.pkl',
        'news_data.json': 'news_data.pkl',
        'fundamentals_data.json': 'fundamentals_data.pkl',
        'recommendations.json': 'recommendations.pkl'
    }
    
    for json_file, pickle_file in file_map.items():
        json_path = os.path.join(data_dir, json_file)
        pickle_path = os.path.join(data_dir, pickle_file)
        
        if os.path.exists(json_path):
            try:
                print(f"Converting {json_file} to {pickle_file}...")
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Convert any special data structures if needed
                # For example, convert date strings to datetime objects
                if json_file == 'stock_data.json' and 'stocks' in data:
                    for ticker, stock_data in data['stocks'].items():
                        if 'history' in stock_data and isinstance(stock_data['history'], list):
                            # Convert list of records to DataFrame
                            df = pd.DataFrame(stock_data['history'])
                            if 'Date' in df.columns:
                                df['Date'] = pd.to_datetime(df['Date'])
                                df.set_index('Date', inplace=True)
                            stock_data['history'] = df
                
                # Save as pickle
                with open(pickle_path, 'wb') as f:
                    pickle.dump(data, f)
                
                print(f"Successfully converted {json_file} to {pickle_file}")
            except Exception as e:
                print(f"Error converting {json_file}: {e}")
        else:
            print(f"Warning: {json_file} not found, skipping conversion")

if __name__ == "__main__":
    convert_json_to_pickle()
