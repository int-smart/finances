import requests
import pickle
import json
import os
from datetime import datetime

class GistStorage:
    def __init__(self, token=None):
        # Prioritize gist-specific token
        self.token = token or os.environ.get('TOKEN_GIST') or os.environ.get('GITHUB_TOKEN')
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
    
    def upload_pickle(self, data, data_type, description=None):
        """Upload pickle data to GitHub Gist"""
        # Serialize data
        pickled_data = pickle.dumps(data)
        
        # Convert to base64 for JSON transport
        import base64
        encoded_data = base64.b64encode(pickled_data).decode('utf-8')
        
        filename = f"{data_type}_{datetime.now().strftime('%Y%m%d')}.pkl"
        
        gist_data = {
            "description": description or f"Financial data - {data_type}",
            "public": False,  # Private gist
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
    
    def download_pickle(self, data_type):
        """Download pickle data from GitHub Gist"""
        if not self.gist_ids[data_type]:
            print(f"No gist ID found for {data_type}")
            return None
        
        url = f"{self.base_url}/{self.gist_ids[data_type]}"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            gist_data = response.json()
            
            # Find the pickle file
            for filename, file_data in gist_data['files'].items():
                if filename.endswith('.pkl'):
                    # Decode base64 data
                    import base64
                    encoded_content = file_data['content']
                    pickled_data = base64.b64decode(encoded_content.encode('utf-8'))
                    
                    # Deserialize pickle
                    data = pickle.loads(pickled_data)
                    print(f"Successfully downloaded {data_type} from gist")
                    return data
        
        print(f"Failed to download {data_type}: {response.status_code}")
        return None

    @classmethod
    def download_without_auth(cls, gist_id, filename):
        """Download from public gist without authentication"""
        url = f"https://api.github.com/gists/{gist_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            gist_data = response.json()
            if filename in gist_data['files']:
                import base64
                encoded_content = gist_data['files'][filename]['content']
                pickled_data = base64.b64decode(encoded_content.encode('utf-8'))
                return pickle.loads(pickled_data)
        return None