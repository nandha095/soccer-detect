import requests
import os
import json
from tqdm import tqdm

def upload_video(video_path, api_url):
    """
    Upload a video file to the specified API endpoint using the correct schema
    """
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} not found")
        return False
    
    file_size = os.path.getsize(video_path)
    filename = os.path.basename(video_path)
    
    print(f"Uploading {filename} ({file_size/1024/1024:.2f} MB) to {api_url}")
    
    try:
        with open(video_path, 'rb') as file:
            files = {'file': (filename, file, 'video/mp4')}
            
            # Include optional metadata according to schema
            data = {
                'chunkMetadata': json.dumps({
                    'frames': []  # This will be populated by the server
                })
            }
            
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename) as pbar:
                response = requests.post(
                    api_url,
                    files=files,
                    data=data,
                    headers={},
                    stream=True
                )
                
        if response.status_code in [200, 201]:
            print(f"Upload successful!")
            try:
                response_data = response.json()
                print(f"Status Code: {response_data.get('statusCode')}")
                print(f"Message: {response_data.get('message')}")
                
                # Check if data is available and is a dictionary
                if 'data' in response_data and isinstance(response_data['data'], dict):
                    data = response_data['data']
                    print(f"Chunk ID: {data.get('chunkId')}")
                    print(f"Chunk Path: {data.get('chunkPath')}")
                    print(f"Frames Count: {len(data.get('frames', []))}")
                elif 'data' in response_data:
                    # Handle case where data is a string (URL)
                    print(f"Data: {response_data['data']}")
            except ValueError:
                print(f"Response not in JSON format: {response.text}")
            return True
        else:
            print(f"Upload failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        return False

def upload_json(json_path, api_url):
    """
    Upload a JSON file to the specified API endpoint
    """
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found")
        return False
    
    file_size = os.path.getsize(json_path)
    filename = os.path.basename(json_path)
    
    print(f"Uploading {filename} ({file_size/1024:.2f} KB) to {api_url}")
    
    try:
        with open(json_path, 'rb') as file:
            files = {'file': (filename, file, 'application/json')}
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=filename) as pbar:
                response = requests.post(
                    api_url,
                    files=files,
                    headers={},
                    stream=True
                )
                
        if response.status_code in [200, 201]:
            print(f"Upload successful!")
            try:
                response_data = response.json()
                print(f"Status Code: {response_data.get('statusCode')}")
                print(f"Message: {response_data.get('message')}")
                if 'data' in response_data:
                    print(f"Data: {response_data['data']}")
            except ValueError:
                print(f"Response not in JSON format: {response.text}")
            return True
        else:
            print(f"Upload failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error during upload: {str(e)}")
        return False

if __name__ == "__main__":
    # File paths
    video_path = os.path.join(os.path.dirname(__file__), "input.mp4")
    json_path = os.path.join(os.path.dirname(__file__), "tracking_results_20250412_232318.json")
    
    # API endpoints
    video_api_url = "https://development7.promena.in/api/Chunks/UploadLargeFile"
    json_api_url = "https://development7.promena.in/api/Chunks/UploadLargeFile"
    
    # Upload video
    print("Starting video upload...")
    video_success = upload_video(video_path, video_api_url)
    
    # Upload JSON
    print("\nStarting JSON upload...")
    json_success = upload_json(json_path, json_api_url)
    
    # Summary
    print("\nUpload Summary:")
    print(f"Video upload: {'Success' if video_success else 'Failed'}")
    print(f"JSON upload: {'Success' if json_success else 'Failed'}")
