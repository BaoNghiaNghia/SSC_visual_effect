import argparse
import cv2
import json
import numpy as np
import librosa
import moviepy.editor as mp
from scipy.interpolate import interp1d
from scipy.signal import butter, lfilter, find_peaks, savgol_filter


# Function to write frames_zoom_data to a JSON file as floats
def write_frames_zoom_data_as_float(frames_zoom_data, filename):
    with open(filename, 'w') as file:
        json.dump(frames_zoom_data, file)


# Function to read frames_zoom_data from a JSON file
def read_frames_zoom_data_as_float(filename):
    with open(filename, 'r') as file:
        return json.load(file)


# Function to apply zoom effect to an image frame
def apply_zoom_effect(frame, zoom_factor):
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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Audio-Image Synchronization')
    parser.add_argument('--audio', type=str, required=True, help='Path to the audio file')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--mode', choices=['bass', 'treble', 'mid', 'hpss'], required=True,
                        help='Audio analysis mode: bass, treble, mid, hpss')
    parser.add_argument('--range', type=int, default=15, help='Window length for smoothing RMS')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='Output video file name')
    return parser.parse_args()


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


def main():
    try:
        args = parse_arguments()

        # Load the audio file
        audio_file_path = args.audio
        y, sr = librosa.load(audio_file_path, sr=None)

        # Load the image file
        image_file_path = args.image
        output_video = args.output
        image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)

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
            frames_zoom_data[time] = smoothed_rms[np.argmin(np.abs(librosa.times_like(rms, sr=sr, hop_length=hop_length) - time))] + 1

        print(frames_zoom_data)
        # Write frames_zoom_data to file as floats
        output_filename = f'frames_zoom_data_{args.mode}.json'
        write_frames_zoom_data_as_float(frames_zoom_data, output_filename)

        # Read frames_zoom_data from file
        frames_zoom_data = read_frames_zoom_data_as_float(output_filename)

        # Convert keys to float and values to float (in case JSON loaded them as strings)
        frames_zoom_data = {float(k): float(v) for k, v in frames_zoom_data.items()}

        # Extract frames and zoom factors
        frames = list(frames_zoom_data.keys())
        zoom_factors = list(frames_zoom_data.values())

        # Calculate total_frames based on audio duration and 30 frames per second
        total_frames = int(librosa.get_duration(y=y, sr=sr) * 30)

        # Apply zoom effect to the image based on the calculated zoom factors
        zoomed_images = []
        max_width, max_height = image.shape[1], image.shape[0]
        zoom_factor_function = interp1d(frames, zoom_factors, kind='linear', fill_value='extrapolate')
        for idx in range(total_frames):
            frame = image.copy()  # Copy original image frame
            time = idx / 30.0  # Calculate time in seconds
            zoom_factor = float(zoom_factor_function(time))
            resized_frame = apply_zoom_effect(frame, zoom_factor)
            resized_frame = cv2.resize(resized_frame, (max_width, max_height))  # Resize to original image dimensions
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            zoomed_images.append(resized_frame)

        # Create a video clip from the zoomed frames
        video_clip = mp.ImageSequenceClip(zoomed_images, fps=30)

        # Load audio from original audio file
        audio_clip = mp.AudioFileClip(audio_file_path)

        # Set audio for the video clip
        video_clip = video_clip.set_audio(audio_clip)

        # Write the final processed video clip to a file
        video_clip.write_videofile(output_video, codec='hevc_nvenc', audio_codec='aac', threads=4, ffmpeg_params=['-pix_fmt', 'yuv420p'])

        print(f"Video with zoom effect saved to {output_video}")

    except SystemExit as e:
        if e.code != 0:
            print("Error: Missing or invalid arguments.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    main()
