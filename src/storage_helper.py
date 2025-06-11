import requests
import pickle
import json
import os
import tempfile
import zipfile
from datetime import datetime
from dotenv import load_dotenv

# Add this at the top
load_dotenv()

class GitHubReleaseStorage:
    def __init__(self, token=None, repo_owner="int-smart", repo_name="finances"):
        # Prioritize specific tokens and repo info
        self.token = token or os.environ.get('TOKEN_GIST')
        self.repo_owner = repo_owner or os.environ.get('GITHUB_REPO_OWNER')
        self.repo_name = repo_name or os.environ.get('GITHUB_REPO_NAME')
        
        if not self.token:
            raise ValueError("No GitHub token found. Set GITHUB_TOKEN environment variable.")
        if not self.repo_owner or not self.repo_name:
            raise ValueError("Repository info missing. Set GITHUB_REPO_OWNER and GITHUB_REPO_NAME environment variables.")
            
        self.base_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
        self.headers = {
            'Authorization': f'token {self.token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        
        # Release tag for data storage
        self.release_tag = "data-storage"
        self.release_name = "Financial Data Storage"
        
        # Ensure release exists
        self.ensure_release_exists()
    
    def ensure_release_exists(self):
        """Ensure the data storage release exists"""
        try:
            # Check if release exists
            url = f"{self.base_url}/releases/tags/{self.release_tag}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 404:
                # Create release
                release_data = {
                    "tag_name": self.release_tag,
                    "name": self.release_name,
                    "body": "Automated storage for financial data files",
                    "draft": False,
                    "prerelease": True
                }
                
                url = f"{self.base_url}/releases"
                response = requests.post(url, headers=self.headers, json=release_data)
                
                if response.status_code == 201:
                    print(f"Created release: {self.release_tag}")
                else:
                    print(f"Failed to create release: {response.status_code}")
                    print(response.text)
            elif response.status_code == 200:
                print(f"Release {self.release_tag} exists")
            else:
                print(f"Error checking release: {response.status_code}")
                
        except Exception as e:
            print(f"Error ensuring release exists: {e}")
    
    def get_release_info(self):
        """Get release information"""
        try:
            url = f"{self.base_url}/releases/tags/{self.release_tag}"
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get release info: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error getting release info: {e}")
            return None
    
    def delete_existing_asset(self, filename):
        """Delete existing asset with the same name"""
        try:
            release_info = self.get_release_info()
            if not release_info:
                return
            
            for asset in release_info.get('assets', []):
                if asset['name'] == filename:
                    delete_url = f"{self.base_url}/releases/assets/{asset['id']}"
                    response = requests.delete(delete_url, headers=self.headers)
                    if response.status_code == 204:
                        print(f"Deleted existing asset: {filename}")
                    else:
                        print(f"Failed to delete asset {filename}: {response.status_code}")
                    break
                    
        except Exception as e:
            print(f"Error deleting existing asset: {e}")
    
    def upload_pickle(self, data, data_type, description=None):
        """Upload pickle data to GitHub Release"""
        try:
            release_info = self.get_release_info()
            if not release_info:
                return None
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as temp_file:
                pickle.dump(data, temp_file)
                temp_file_path = temp_file.name
            
            filename = f"{data_type}.pkl"
            
            # Delete existing asset if it exists
            self.delete_existing_asset(filename)
            
            # Upload new asset
            upload_url = release_info['upload_url'].replace('{?name,label}', f'?name={filename}')
            
            with open(temp_file_path, 'rb') as f:
                files = {'file': (filename, f, 'application/octet-stream')}
                
                upload_headers = self.headers.copy()
                upload_headers['Content-Type'] = 'application/octet-stream'
                
                response = requests.post(
                    upload_url,
                    headers=upload_headers,
                    data=f.read()
                )
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            if response.status_code == 201:
                print(f"Successfully uploaded {data_type} to release")
                return response.json()['browser_download_url']
            else:
                print(f"Failed to upload {data_type}: {response.status_code}")
                print(response.text)
                return None
                
        except Exception as e:
            print(f"Error uploading {data_type}: {e}")
            return None
    
    def download_pickle(self, data_type):
        """Download pickle data from GitHub Release"""
        try:
            release_info = self.get_release_info()
            if not release_info:
                return None
            
            filename = f"{data_type}.pkl"
            download_url = None
            
            # Find the asset
            for asset in release_info.get('assets', []):
                if asset['name'] == filename:
                    download_url = asset['browser_download_url']
                    break
            
            if not download_url:
                print(f"No asset found for {data_type}")
                return None
            
            # Download the file
            response = requests.get(download_url)
            
            if response.status_code == 200:
                # Load pickle data directly from response content
                data = pickle.loads(response.content)
                print(f"Successfully downloaded {data_type} from release")
                return data
            else:
                print(f"Failed to download {data_type}: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error downloading {data_type}: {e}")
            return None
    
    def list_assets(self):
        """List all assets in the release"""
        try:
            release_info = self.get_release_info()
            if not release_info:
                return []
            
            assets = []
            for asset in release_info.get('assets', []):
                assets.append({
                    'name': asset['name'],
                    'size': asset['size'],
                    'download_count': asset['download_count'],
                    'created_at': asset['created_at'],
                    'updated_at': asset['updated_at'],
                    'download_url': asset['browser_download_url']
                })
            
            return assets
            
        except Exception as e:
            print(f"Error listing assets: {e}")
            return []
    
    @classmethod
    def download_without_auth(cls, repo_owner, repo_name, data_type, release_tag="data-storage"):
        """Download from public release without authentication"""
        try:
            base_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}"
            url = f"{base_url}/releases/tags/{release_tag}"
            
            response = requests.get(url)
            
            if response.status_code == 200:
                release_info = response.json()
                filename = f"{data_type}.pkl"
                
                # Find the asset
                for asset in release_info.get('assets', []):
                    if asset['name'] == filename:
                        download_url = asset['browser_download_url']
                        
                        # Download the file
                        download_response = requests.get(download_url)
                        if download_response.status_code == 200:
                            return pickle.loads(download_response.content)
                        break
            
            return None
            
        except Exception as e:
            print(f"Error downloading without auth: {e}")
            return None

# Backward compatibility alias
GistStorage = GitHubReleaseStorage