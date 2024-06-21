import gc
import os
import cv2
import json
import mimetypes
import numpy as np
import argparse
import librosa
import requests
import moviepy.editor as mp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks, savgol_filter
import multiprocessing
from pydub import AudioSegment
import logging
import threading

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global stop event
stop_event = threading.Event()

DEFAULT_FRAME_FPS = 30
DEFAULT_BASS_AUDIO_FILENAME = 'bass_only_audio.mp3'
DEFAULT_TREBEL_AUDIO_FILENAME = 'trebel_only_audio.mp3'
DEFAULT_MID_AUDIO_FILENAME = 'mid_only_audio.mp3'
DEFAULT_HPSS_AUDIO_FILENAME = 'hpss_only_audio.mp3'


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


# Function to write frames_zoom_data to a JSON file as floats
def write_frames_zoom_data_as_float(frames_zoom_data, filename):
    try:
        with open(filename, 'w') as file:
            json.dump(frames_zoom_data, file)
        logging.info(f"Successfully wrote frames_zoom_data to {filename}")
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


# Function to apply zoom effect to a frame
def apply_zoom_effect_video(frame, zoom_factor, target_size):
    try:
        width, height = frame.shape[1], frame.shape[0]
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Calculate cropping margins to keep zoom centered
        x = (new_width - target_size[0]) // 2
        y = (new_height - target_size[1]) // 2
        
        # Crop to the target size
        cropped_frame = resized_frame[y:y + target_size[1], x:x + target_size[0]]
        
        return cropped_frame
    except Exception as e:
            logging.error(f"Error applying zoom effect: {e}")
            return None

# Function to process a batch of frames
def process_batch(args):
    start_idx, end_idx, clip_fps, interpolated_zoom_factors, batch_size, video_path, batch_folder = args
    clip = mp.VideoFileClip(video_path)
    processed_frames = []
    
    # Get the target size from the original clip
    target_size = (clip.size[0], clip.size[1])
    
    for idx in range(start_idx, end_idx):
        frame = clip.get_frame(idx / clip_fps)
        zoom_factor = interpolated_zoom_factors[idx]
        processed_frame = apply_zoom_effect_video(frame, zoom_factor, target_size)
        processed_frames.append(processed_frame)
    
    # Create a video clip from the processed batch of frames
    batch_clip = mp.ImageSequenceClip(processed_frames, fps=clip_fps)
    batch_clip_file = os.path.join(batch_folder, f"batch_{start_idx//batch_size}.mp4")
    batch_clip.write_videofile(batch_clip_file, codec='hevc_nvenc', audio_codec='aac')
    
    # Clear the processed frames to free up memory
    del processed_frames
    clip.close()  # Ensure the clip is properly closed
    del batch_clip
    gc.collect()  # Explicitly call the garbage collector
    
    return batch_clip_file

# Function to apply zoom effect to an image frame
def apply_zoom_effect_image(frame, zoom_factor):
    try:
        height, width = frame.shape[:2]
        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)

        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Calculate cropping margins to keep zoom centered
        x = (new_width - width) // 2
        y = (new_height - height) // 2

        # Crop to original size
        cropped_frame = resized_frame[y:y + height, x:x + width]

        return cropped_frame
    except Exception as e:
        logging.error(f"Error applying zoom effect: {e}")
        return None

# Function to parse command-line arguments
def parse_arguments():
    try:
        parser = argparse.ArgumentParser(description='Audio-Image Synchronization')
        parser.add_argument('--audio', type=str, help='Path to the audio file')
        parser.add_argument('--input', type=str, help='Path to the input file: Image or Video')
        parser.add_argument('--mode', choices=['bass', 'treble', 'mid', 'hpss'], help='Audio analysis mode: bass, treble, mid, hpss')
        parser.add_argument('--range', type=int, default=15, help='Window length for smoothing RMS')
        parser.add_argument('--output', type=str, default="output_video.mp4",help='Output video file name')
        parser.add_argument('--threads', type=int, default=3, help='Number of threads for multiprocessing')
        parser.add_argument('--domain', type=str, help='Domain for API authentication')
        parser.add_argument('--token', type=str, help='Token for API authentication')
        parser.add_argument('--effect', choices=['zoom', 'blur', 'brightness', 'rgb'], help='Effect reaction for audio amplitude')
        return parser.parse_args()
    except argparse.ArgumentError as e:
        logging.error(f"Argument parsing error: {e}")
        return None

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

        logging.info(f"Authentication successful with domain {domain}")
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

# Function to design a Butterworth lowpass filter
def butter_lowpass(cutoff, fs, order=5):
    try:
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    except Exception as e:
        logging.error(f"Error designing Butterworth lowpass filter: {e}")
        return None, None

# Function to apply a lowpass filter to data
def apply_lowpass_filter(data, cutoff, fs, order=5):
    try:
        b, a = butter_lowpass(cutoff, fs, order=order)
        if b is None or a is None:
            return None
        y = lfilter(b, a, data)
        return y
    except Exception as e:
        logging.error(f"Error applying lowpass filter: {e}")
        return None

# Function to analyze an audio component based on mode
def analyze_audio_component(y, sr, mode):
    try:
        if mode == 'bass':
            cutoff_freq = 150
            y_analysis = apply_lowpass_filter(y, cutoff_freq, sr)
        elif mode == 'treble':
            cutoff_freq = 4000
            y_analysis = librosa.effects.harmonic(y=y)  # Using harmonic component for treble
        elif mode == 'mid':
            cutoff_freq = [200, 4000]
            y_analysis = librosa.effects.bandpass(y, cutoff_freq[0], cutoff_freq[1], sr=sr)
        elif mode == 'hpss':
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            y_analysis = y_harmonic + y_percussive  # Combining both harmonic and percussive components
        else:
            raise ValueError(f'Unsupported mode: {mode}')

        return y_analysis
    except ValueError as ve:
        logging.error(f"Value error during audio analysis: {ve}")
        return None
    except Exception as e:
        logging.error(f"Error analyzing audio component: {e}")
        return None

# Function to process a frame with zoom effect
def process_frame(params):
    try:
        idx, image, zoom_factor_function = params
        frame = image.copy()  # Copy original image frame
        time = idx / DEFAULT_FRAME_FPS  # Calculate time in seconds
        zoom_factor = float(zoom_factor_function(time))
        resized_frame = apply_zoom_effect_image(frame, zoom_factor)
        max_width, max_height = image.shape[1], image.shape[0]
        resized_frame = cv2.resize(resized_frame, (max_width, max_height))  # Resize to original image dimensions
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        return resized_frame
    except AssertionError as e:
        logging.error(f"AssertionError processing frame {idx}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error processing frame {idx}: {e}")
        return None
    
def plot_rms_data(rms, smoothed_rms, sr, hop_length):
    try:
        times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
        plt.figure(figsize=(14, 6))
        plt.plot(times, rms, label='RMS', alpha=0.6)
        plt.plot(times, smoothed_rms, label='Smoothed RMS', alpha=0.8, linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('RMS')
        plt.title('RMS and Smoothed RMS Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting RMS data: {e}")
        
        
        
        
# ------------- START: RMS data for different audio components (bass, treble, mid, and hpss), save the filtered audio to separate files -------------#



def butter_filter(data, cutoff, sr, filter_type='low', order=5, lowcut=None, highcut=None):
    nyquist = 0.5 * sr
    if filter_type == 'low':
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
    elif filter_type == 'high':
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
    elif filter_type == 'band':
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
    else:
        raise ValueError(f"Invalid filter type: {filter_type}")
    y = lfilter(b, a, data)
    return y

def save_audio(data, sr, output_path):
    data_int16 = np.int16(data / np.max(np.abs(data)) * 32767)
    sample_width = data_int16.dtype.itemsize
    audio = AudioSegment(
        data_int16.tobytes(),
        frame_rate=sr,
        sample_width=sample_width,
        channels=1
    )
    audio.export(output_path, format="mp3")
    logging.info(f"Audio saved to {output_path}")

def calculate_rms(data, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)[0]
    return rms

def save_rms(rms, output_path):
    with open(output_path, 'w') as file:
        json.dump(rms.tolist(), file)
    logging.info(f"RMS data saved to {output_path}")

def generate_separate_frequency_to_file_audio(origin_file_path, mode):
    audio = AudioSegment.from_mp3(origin_file_path)
    y = np.array(audio.get_array_of_samples(), dtype=np.float32)
    sr = audio.frame_rate

    if audio.channels == 2:
        y = y.reshape((-1, 2))
        y = y.mean(axis=1)

    if mode == 'bass':
        y_filtered = butter_filter(y, cutoff=150.0, sr=sr, filter_type='low')
        output_audio = DEFAULT_BASS_AUDIO_FILENAME
        # output_rms = 'rms_bass.json'
    elif mode == 'treble':
        y_filtered = butter_filter(y, cutoff=4000.0, sr=sr, filter_type='high')
        output_audio = DEFAULT_TREBEL_AUDIO_FILENAME
        # output_rms = 'rms_trebel.json'
    elif mode == 'mid':
        y_filtered = butter_filter(y, lowcut=200.0, highcut=4000.0, sr=sr, filter_type='band')
        output_audio = DEFAULT_MID_AUDIO_FILENAME
        # output_rms = 'rms_mid.json'
    elif mode == 'hpss':
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        y_filtered = y_harmonic + y_percussive
        output_audio = DEFAULT_HPSS_AUDIO_FILENAME
        # output_rms = 'rms_hpss.json'
    else:
        raise ValueError(f"Invalid mode: {mode}")

    save_audio(y_filtered, sr, output_audio)
    return output_audio
    # rms = calculate_rms(y_filtered)
    # save_rms(rms, output_rms)



# ------------- END: RMS data for different audio components (bass, treble, mid, and hpss), save the filtered audio to separate files -------------#
        


def render_video(args):
    # Handling exceptions when missing or invalid arguments
    try:
        # Load the audio file
        audio_file_path = args.audio
        y, sr = librosa.load(audio_file_path, sr=None)

        # Load video clip
        video_file_path = args.input
        output_video = args.output
        clip = mp.VideoFileClip(video_file_path)

        # Create a subfolder for batch files
        batch_folder = "batch_videos"
        os.makedirs(batch_folder, exist_ok=True)

        # Get the FPS of the video
        video_fps = clip.fps

        # Frame length and hop length
        frame_length = 2048
        hop_length = 512
        
        # Analyze selected audio component
        y_analysis = analyze_audio_component(y, sr, args.mode)

        # Compute the RMS value for each frame
        rms = librosa.feature.rms(y=y_analysis, frame_length=frame_length, hop_length=hop_length)[0]

        # Smooth the RMS values using the Savitzky-Golay filter
        smoothed_rms = savgol_filter(rms, window_length=args.range, polyorder=4)

        # Find peaks and troughs in the smoothed RMS values
        peaks, _ = find_peaks(smoothed_rms)
        troughs, _ = find_peaks(-smoothed_rms)

        # Convert the peak and trough frames to time
        peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
        peak_values = smoothed_rms[peaks]

        trough_times = librosa.frames_to_time(troughs, sr=sr, hop_length=hop_length)
        trough_values = smoothed_rms[troughs]

        # Group peaks and troughs into alternating peak-trough pairs
        peak_trough_pairs = []
        p_idx = 0
        t_idx = 0

        while p_idx < len(peaks) and t_idx < len(troughs):
            if peak_times[p_idx] < trough_times[t_idx]:
                peak_trough_pairs.append((peak_times[p_idx], 'peak'))
                p_idx += 1
            else:
                peak_trough_pairs.append((trough_times[t_idx], 'trough'))
                t_idx += 1

        # Append remaining peaks or troughs if any
        while p_idx < len(peaks):
            peak_trough_pairs.append((peak_times[p_idx], 'peak'))
            p_idx += 1

        while t_idx < len(troughs):
            peak_trough_pairs.append((trough_times[t_idx], 'trough'))
            t_idx += 1

        # Sort peak-trough pairs by time
        peak_trough_pairs.sort(key=lambda x: x[0])

        # Create a dictionary of frames and corresponding zoom factors
        frames_zoom_data = {}
        for time, marker in peak_trough_pairs:
            frame_index = round(time * video_fps)
            smallest_zoom_size = 1.05
            calculated_zoom_factor = smoothed_rms[np.argmin(np.abs(librosa.times_like(rms, sr=sr, hop_length=hop_length) - time))]
            if calculated_zoom_factor <= 0:
                frames_zoom_data[frame_index] = smallest_zoom_size
            if calculated_zoom_factor > 0:
                frames_zoom_data[frame_index] = round((calculated_zoom_factor/3 + smallest_zoom_size), 3)

        # Write frames_zoom_data to file as floats
        output_filename = f'video_zoom_data_{args.mode}.json'
        write_frames_zoom_data_as_float(frames_zoom_data, output_filename)

        # Read frames_zoom_data from file
        frames_zoom_data = read_frames_zoom_data_as_float(output_filename)

        # Convert keys to integers and values to floats (in case JSON loaded them as strings)
        frames_zoom_data = {int(k): float(v) for k, v in frames_zoom_data.items()}

        # Extract frames and zoom factors
        frames = list(frames_zoom_data.keys())
        zoom_factors = list(frames_zoom_data.values())

        # Create an interpolation function
        interpolation_function = interp1d(frames, zoom_factors, kind='linear', fill_value='extrapolate')

        # Define the total number of frames in the video
        total_frames = int(clip.fps * clip.duration)

        # Generate interpolated zoom factors for each frame
        interpolated_zoom_factors = interpolation_function(np.arange(total_frames))

        # Process frames and apply zoom effect in smaller batches using multiprocessing
        batch_size = 100  # Process frames in batches of 100
        process_core_cpu = 4
        batch_indices = [(start_idx, min(start_idx + batch_size, total_frames), video_fps, interpolated_zoom_factors, batch_size, video_file_path, batch_folder) 
                            for start_idx in range(0, total_frames, batch_size)]

        # Use a limited number of processes to avoid overloading the system
        num_processes = min(process_core_cpu, multiprocessing.cpu_count())  # Adjust the number of processes as needed
        with multiprocessing.Pool(processes=num_processes) as pool:
            batch_clips = pool.map(process_batch, batch_indices)

        # Combine all batch videos into the final video
        final_clips = [mp.VideoFileClip(f) for f in batch_clips]
        final_clip = mp.concatenate_videoclips(final_clips)

        # Load audio from original video clip
        audio_clip = clip.audio

        # Set audio for the final video clip
        final_clip = final_clip.set_audio(audio_clip)

        # Write the final processed video clip to a file
        final_clip.write_videofile(output_video, codec='hevc_nvenc', audio_codec='aac')

        # Clean up temporary batch video files after the final video is rendered
        for batch_clip_file in batch_clips:
            try:
                clip = mp.VideoFileClip(batch_clip_file)
                clip.close()  # Ensure the clip is properly closed
            except Exception as e:
                print(f"Failed to close clip {batch_clip_file}: {e}")
            try:
                os.remove(batch_clip_file)
            except Exception as e:
                print(f"Failed to remove {batch_clip_file}: {e}")

        # Delete the batch video folder
        try:
            os.rmdir(batch_folder)
            print(f"Deleted folder {batch_folder}")
        except Exception as e:
            print(f"Failed to delete {batch_folder}: {e}")

    except SystemExit as e:
        if e.code != 0:
            print("Error: Missing or invalid arguments.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def render_image(args):
    try:
        if args is None:
            return
        
        print(f"Arguments: {args}")
        
        # Check if audio and image paths are provided and valid
        if not os.path.isfile(args.audio):
            raise FileNotFoundError(f"Audio file not found: {args.audio}")
        if not os.path.isfile(args.input):
            raise FileNotFoundError(f"Input file not found: {args.input}")

        # Authenticate if email and password are provided
        if args.token:
            status_code, message = authenticate(args.domain, args.token)
            if status_code != 200:
                raise Exception("Authentication failed")

            print(f"Authenticated successfully. Response: {message}")

        logging.info(f"Step 1/2: Process for getting data from audio....")

        # RMS data for different audio components (bass, treble, mid, and hpss)
        audio_file_path = generate_separate_frequency_to_file_audio(args.audio, args.mode)
        
        # Load the audio file
        origin_audio_file_path = args.audio
        y, sr = librosa.load(audio_file_path, sr=None)
        if y is None:
            raise ValueError(f"Error loading audio file: {args.audio}")

        # Load the image file
        image_file_path = args.input
        output_video = args.output
        image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Error loading image file: {args.input}")

        # Frame length and hop length
        frame_length = 2048
        hop_length = 512

        # Analyze selected audio component
        y_analysis = analyze_audio_component(y, sr, args.mode)

        # Compute the RMS value for each frame
        rms = librosa.feature.rms(y=y_analysis, frame_length=frame_length, hop_length=hop_length)[0]

        # Smooth the RMS values using the Savitzky-Golay filter
        smoothed_rms = savgol_filter(rms, window_length=args.range, polyorder=4)
        
        # Visualize Plot RMS and Smoothed RMS
        # plot_rms_data(rms, smoothed_rms, sr, hop_length)

        # Find peaks and troughs in the smoothed RMS values
        peaks, _ = find_peaks(smoothed_rms)
        troughs, _ = find_peaks(-smoothed_rms)

        # Convert the peak and trough frames to time
        peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
        # peak_values = smoothed_rms[peaks]

        trough_times = librosa.frames_to_time(troughs, sr=sr, hop_length=hop_length)
        # trough_values = smoothed_rms[troughs]

        # Group peaks and troughs into alternating peak-trough pairs
        peak_trough_pairs = []
        p_idx = 0
        t_idx = 0

        while p_idx < len(peaks) and t_idx < len(troughs):
            if peak_times[p_idx] < trough_times[t_idx]:
                peak_trough_pairs.append((peak_times[p_idx], 'peak'))
                p_idx += 1
            else:
                peak_trough_pairs.append((trough_times[t_idx], 'trough'))
                t_idx += 1

        # Append remaining peaks or troughs if any
        while p_idx < len(peaks):
            peak_trough_pairs.append((peak_times[p_idx], 'peak'))
            p_idx += 1

        while t_idx < len(troughs):
            peak_trough_pairs.append((trough_times[t_idx], 'trough'))
            t_idx += 1

        # Sort peak-trough pairs by time
        peak_trough_pairs.sort(key=lambda x: x[0])

        # Create a dictionary of frames and corresponding zoom factors
        frames_zoom_data = {}
        for time, marker in peak_trough_pairs:
            calculated_zoom_factor = smoothed_rms[np.argmin(np.abs(librosa.times_like(rms, sr=sr, hop_length=hop_length) - time))]
            smallest_zoom_size = 1.05

            if calculated_zoom_factor <= 0:
                frames_zoom_data[time] = smallest_zoom_size
            if calculated_zoom_factor > 0:
                frames_zoom_data[time] = round((calculated_zoom_factor/3 + smallest_zoom_size), 3)

        # Write frames_zoom_data to file as floats
        output_filename = f'frames_zoom_data_{args.mode}.json'
        write_frames_zoom_data_as_float(frames_zoom_data, output_filename)
        print("OK")
        logging.info(f"Write file {output_filename}")

        # Read frames_zoom_data from file
        frames_zoom_data = read_frames_zoom_data_as_float(output_filename)
        logging.info(f"Step 2/2: Start to render video with zoom effect....")
        
        # Convert keys to float and values to float (in case JSON loaded them as strings)
        frames_zoom_data = {float(k): float(v) for k, v in frames_zoom_data.items()}
        
        # Extract frames and zoom factors
        frames = list(frames_zoom_data.keys())
        zoom_factors = list(frames_zoom_data.values())
        
        # Calculate total_frames based on audio duration and 30 frames per second
        total_frames = int(librosa.get_duration(y=y, sr=sr) * DEFAULT_FRAME_FPS)
        
        
        # Use multiprocessing to process frames
        max_width, max_height = image.shape[1], image.shape[0]
        zoom_factor_function = interp1d(frames, zoom_factors, kind='quadratic', fill_value='extrapolate')
        
        
        with multiprocessing.Pool(args.threads) as pool:
            try:
                zoomed_images = pool.map(process_frame, [(idx, image, zoom_factor_function) for idx in range(total_frames)])
            except Exception as e:
                logging.error(f"Error in multiprocessing: {e}")
                pool.terminate()  # Terminate pool on exception
                pool.join()  # Ensure all processes are cleaned up
            
        # Remove None frames due to errors
        zoomed_images = [frame for frame in zoomed_images if frame is not None]
        
        # Perform garbage collection
        gc.collect()

        # Create a video clip from the zoomed frames
        video_clip = mp.ImageSequenceClip(zoomed_images, fps=DEFAULT_FRAME_FPS)

        # Load audio from original audio file
        audio_clip = mp.AudioFileClip(origin_audio_file_path)

        # Set audio for the video clip
        video_clip = video_clip.set_audio(audio_clip)

        # Write the final processed video clip to a file
        video_clip.write_videofile(output_video, codec='hevc_nvenc', audio_codec='aac', threads=args.threads, ffmpeg_params=['-pix_fmt', 'yuv420p'])

        logging.info(f"Video with zoom effect saved to {output_video}")
        
        # # Cleanup: delete temporary audio and JSON files
        # os.remove(audio_filename)
        # os.remove(json_filename)
        # logging.info("Temporary files deleted")

    except FileNotFoundError as fe:
        logging.error(fe)
    except ValueError as ve:
        logging.error(ve)
    except SystemExit as e:
        print('System Exit', e)
        if e.code != 0:
            print("Error: Missing or invalid arguments.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up any remaining resources
        multiprocessing.active_children()  # Ensure all child processes are terminated
        

if __name__ == '__main__':
    try:
        multiprocessing.freeze_support()  # For Windows support
        
        args = parse_arguments()
        
        # Check if the provided input file is an audio or a video file
        file_path = args.input
        file_type = check_file_type(file_path)
        
        if file_type == 'video':
            render_video(args)
        elif file_type == 'image':
            render_image(args)
        else:
            raise ValueError(f"Unsupported input file type. Please provide an image or video file.")
    except Exception as e:
        multiprocessing.active_children()
        print(f"An unexpected error occurred: {e}")
