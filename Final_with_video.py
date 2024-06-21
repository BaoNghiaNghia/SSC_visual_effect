import argparse
import moviepy.editor as mp
import numpy as np
import librosa
from scipy.signal import butter, lfilter, find_peaks, savgol_filter
from scipy.interpolate import interp1d
import json
import cv2
import os
import gc
from multiprocessing import Pool, cpu_count

# Function to write frames_zoom_data to a JSON file as floats
def write_frames_zoom_data_as_float(frames_zoom_data, filename):
    with open(filename, 'w') as file:
        json.dump(frames_zoom_data, file)

# Function to read frames_zoom_data from a JSON file
def read_frames_zoom_data_as_float(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# Function to apply zoom effect to a frame
def apply_zoom_effect(frame, zoom_factor):
    width, height = frame.shape[1], frame.shape[0]
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

# Function to process a batch of frames
def process_batch(args):
    start_idx, end_idx, clip_fps, interpolated_zoom_factors, batch_size, video_path, batch_folder = args
    clip = mp.VideoFileClip(video_path)
    processed_frames = []
    
    for idx in range(start_idx, end_idx):
        frame = clip.get_frame(idx / clip_fps)
        zoom_factor = interpolated_zoom_factors[idx]
        processed_frame = apply_zoom_effect(frame, zoom_factor)
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


def parse_arguments():
    # Parsing the Arguments from cmd line arguments
    parser = argparse.ArgumentParser(description='Audio-Video Synchronization')
    parser.add_argument('--audio', type=str, required=True, help='Path to the audio file')
    parser.add_argument('--image', type=str, required=True, help='Path to the video file')
    parser.add_argument('--mode', choices=['bass', 'treble', 'mid', 'hpss'], required=True, help='Audio analysis mode: bass, treble, mid, hpss')
    parser.add_argument('--range', type=int, default=15, help='Window length for smoothing RMS')
    return parser.parse_args()


####### --------------------------------  GENERATE RMS DATA WITH ANALYSIS TYPE -------------------------------- ####### 
def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def apply_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def analyze_audio_component(y, sr, mode):
    if mode == 'bass':
        cutoff_freq = 200
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


def render_video():
    # Handling exceptions when missing or invalid arguments
    try:
        args = parse_arguments()
        
        # Load the audio file
        audio_file_path = args.audio
        y, sr = librosa.load(audio_file_path, sr=None)

        # Load video clip
        video_file_path = args.image
        output_video = 'output_circular_bars_final.mp4'
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
            rms_value = smoothed_rms[np.argmin(np.abs(librosa.times_like(rms, sr=sr, hop_length=hop_length) - time))]
            frames_zoom_data[frame_index] = round(rms_value, 4) + 1

        # Write frames_zoom_data to file as floats
        output_filename = f'frames_zoom_data_{args.mode}.json'
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
        num_processes = min(process_core_cpu, cpu_count())  # Adjust the number of processes as needed
        with Pool(processes=num_processes) as pool:
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

if __name__ == '__main__':
    try:
        render_video()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
