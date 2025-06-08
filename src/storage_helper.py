import requests
import pickle
import json
import os
import base64
import re
from datetime import datetime
from dotenv import load_dotenv

# Add this at the top
load_dotenv()
class GistStorage:
    def __init__(self, token=None):
        # Prioritize gist-specific token
        self.token = token or os.environ.get('TOKEN_GIST') or os.environ.get('GITHUB_TOKEN')
        
        if not self.token:
            raise ValueError("No GitHub token found. Set GITHUB_TOKEN_GIST environment variable.")
            
        self.base_url = "https://api.github.com/gists"
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        # Store gist IDs for each data type
        self.gist_ids = {
            'stock_data': None,
            'investor_data': None,
            'news_data': None,
            'fundamentals_data': None,
            'recommendations': None
        }
        self.load_gist_config()
    
    def load_gist_config(self):
        """Load gist IDs from config file"""
        config_file = "gist_config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.gist_ids.update(json.load(f))
    
    def save_gist_config(self):
        """Save gist IDs to config file"""
        with open("gist_config.json", 'w') as f:
            json.dump(self.gist_ids, f)
    
    def _fix_base64_padding(self, data):
        """Fix base64 padding and clean the string"""
        # Remove any whitespace
        data = re.sub(r'\s+', '', data)
        
        # Add padding if needed
        missing_padding = len(data) % 4
        if missing_padding:
            data += '=' * (4 - missing_padding)
        
        return data
    
    def upload_pickle(self, data, data_type, description=None):
        """Upload pickle data to GitHub Gist"""
        try:
            # Serialize data
            pickled_data = pickle.dumps(data)
            
            # Convert to base64
            encoded_data = base64.b64encode(pickled_data).decode('utf-8')
            
            filename = f"{data_type}_{datetime.now().strftime('%Y%m%d')}.pkl"
            
            gist_data = {
                "description": description or f"Financial data - {data_type}",
                "public": False,
                "files": {
                    filename: {
                        "content": encoded_data
                    },
                    "metadata.json": {
                        "content": json.dumps({
                            "data_type": data_type,
                            "timestamp": datetime.now().isoformat(),
                            "size_bytes": len(pickled_data)
                        })
                    }
                }
            }
            
            if self.gist_ids[data_type]:
                # Update existing gist
                url = f"{self.base_url}/{self.gist_ids[data_type]}"
                response = requests.patch(url, headers=self.headers, json=gist_data)
            else:
                # Create new gist
                response = requests.post(self.base_url, headers=self.headers, json=gist_data)
                if response.status_code == 201:
                    self.gist_ids[data_type] = response.json()['id']
                    self.save_gist_config()
            
            if response.status_code in [200, 201]:
                print(f"Successfully uploaded {data_type} to gist")
                return response.json()['html_url']
            else:
                print(f"Failed to upload {data_type}: {response.status_code}")
                print(response.text)
                return None
                
        except Exception as e:
            print(f"Error uploading {data_type}: {e}")
            return None
    
    def download_pickle(self, data_type):
        """Download pickle data from GitHub Gist"""
        if not self.gist_ids[data_type]:
            print(f"No gist ID found for {data_type}")
            return None
        
        try:
            url = f"{self.base_url}/{self.gist_ids[data_type]}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code != 200:
                print(f"Failed to download {data_type}: {response.status_code}")
                print(response.text)
                return None
            
            gist_data = response.json()
            
            # Find the pickle file
            for filename, file_data in gist_data['files'].items():
                if filename.endswith('.pkl'):
                    # Get and fix the base64 content
                    encoded_content = file_data['content']
                    encoded_content = self._fix_base64_padding(encoded_content)
                    
                    # Decode base64 to get pickled data
                    pickled_data = base64.b64decode(encoded_content)
                    
                    # Deserialize pickle
                    data = pickle.loads(pickled_data)
                    print(f"Successfully downloaded {data_type} from gist")
                    return data
            
            print(f"No pickle file found for {data_type}")
            return None
            
        except Exception as e:
            print(f"Error downloading {data_type}: {e}")
            return None

    @classmethod
    def download_without_auth(cls, gist_id, filename):
        """Download from public gist without authentication"""
        url = f"https://api.github.com/gists/{gist_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            gist_data = response.json()
            if filename in gist_data['files']:
                encoded_content = gist_data['files'][filename]['content']
                # Fix padding issues
                encoded_content = re.sub(r'\s+', '', encoded_content)
                missing_padding = len(encoded_content) % 4
                if missing_padding:
                    encoded_content += '=' * (4 - missing_padding)
                
                pickled_data = base64.b64decode(encoded_content)
                return pickle.loads(pickled_data)
        return None