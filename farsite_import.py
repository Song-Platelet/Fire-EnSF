import os
import requests
import subprocess
from util import *

def download_repo_contents(repo_owner, repo_name, commit_sha, path, local_dir):
    """
    Recursively downloads all files from a GitHub repository directory at a specific commit.
    """
    # Set up headers with optional GitHub token for higher rate limit
    headers = {}
    if 'GITHUB_TOKEN' in os.environ:
        headers['Authorization'] = f'token {os.environ["GITHUB_TOKEN"]}'
    
    # Construct API URL for the current path and commit
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{path}"
    params = {'ref': commit_sha}
    
    # Fetch directory contents
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        print(f"Error fetching {path}: Status code {response.status_code}")
        return
    
    entries = response.json()
    
    for entry in entries:
        entry_name = entry['name']
        entry_type = entry['type']
        entry_path = entry['path']
        
        if entry_type == 'file':
            # Download file content
            download_url = entry['download_url']
            file_response = requests.get(download_url, headers=headers)
            
            # Save file locally
            os.makedirs(local_dir, exist_ok=True)
            file_local_path = os.path.join(local_dir, entry_name)
            with open(file_local_path, 'wb') as f:
                f.write(file_response.content)
            print(f"Downloaded: {file_local_path}")
            
        elif entry_type == 'dir':
            # Recursively download directory
            new_local_dir = os.path.join(local_dir, entry_name)
            os.makedirs(new_local_dir, exist_ok=True)
            download_repo_contents(repo_owner, repo_name, commit_sha, entry_path, new_local_dir)

if __name__ == "__main__":
    
    # Repository details
    repo_owner = "mbedward"
    repo_name = "farsite"
    commit_sha = "4537d60ab013fa91d0b49d8efb385ffb1b3ddd13"
    target_path = r"src"  # The directory to download
    local_base_dir = "src"  # Local directory to save files
    create_folder(local_base_dir)
    # Start download
    download_repo_contents(repo_owner, repo_name, commit_sha, target_path, local_base_dir)
    print("All files downloaded successfully!")
    result = subprocess.run(
            ['make', 'all'],  # Correctly passes 'make' as the command and 'all' as its argument
            cwd = 'src', # This is crucial: it tells subprocess to run 'make' inside 'src/'
            capture_output = True,    # Captures standard output and standard error
            text = True,              # Decodes stdout/stderr as text (UTF-8 by default)
            check = True              # If 'make all' returns a non-zero exit code (i.e., fails), this will raise a CalledProcessError
        )
    print(result.stdout)
    print(result.stderr)