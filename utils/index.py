import json
import os
import logging
import shutil
import mimetypes
import requests

# Function to write frames_zoom_data to a JSON file as floats
def write_frames_zoom_data_as_float(frames_zoom_data, filename):
    try:
        with open(filename, 'w') as file:
            json.dump(frames_zoom_data, file)
    except Exception as e:
        logging.error(f"Error writing frames_zoom_data to {filename}: {e}")

# Function to read frames_zoom_data from a JSON file
def read_frames_zoom_data_as_float(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"File not found: {filename}")
    except Exception as e:
        logging.error(f"Error reading frames_zoom_data from {filename}: {e}")
        return None
    
def check_file_type(file_path):
    """
    Check if the input file is an audio or video file based on its MIME type.
    
    Parameters:
        file_path (str): Path to the file to be checked.
    
    Returns:
        str: 'audio' if the file is an audio file, 'video' if it is a video file, 
            'unknown' if the file type could not be determined.
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type is None:
        return 'unknown'
    
    if mime_type.startswith('image'):
        return 'image'
    elif mime_type.startswith('video'):
        return 'video'
    else:
        return 'unknown'
    
def clean_folder(folder_path):
    try:
        if os.path.exists(folder_path):
            # Iterate over the contents of the folder
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            print(f"The folder '{folder_path}' does not exist.")
    except Exception as e:
        print(f"Error cleaning folder '{folder_path}'. Reason: {e}")
        
# Function to authenticate with an API
def authenticate(domain, token):
    try:
        if not domain or not token:
            raise ValueError("Domain, token must be provided for authentication")

        headers = {
            'Content-Type': 'application/json',
            'Authorization' : 'Bearer ' + token
        }
        response = requests.get(domain, headers=headers)
        response.raise_for_status()  # Raise HTTPError for bad response codes

        return response.status_code, response.reason
    except ValueError as ve:
        logging.error(f"Authentication failed: {ve}")
        return None
    except requests.exceptions.RequestException as re:
        logging.error(f"Request error during authentication: {re}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during authentication: {e}")
        return None