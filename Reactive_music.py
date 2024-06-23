import gc
import os
import sys
import cv2
import argparse
import librosa
import logging
import numpy as np
import multiprocessing
import moviepy.editor as mp
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from concurrent.futures import ThreadPoolExecutor

from utils.index import write_frames_zoom_data_as_float, read_frames_zoom_data_as_float, check_file_type, clean_folder, authenticate, cleanup_resources
from constants.index import BATCH_IMAGE_SIZE, BATCH_VIDEO_SIZE, CACHE_DATA_FOLDER, CACHE_VIDEO_FOLDER, DEFAULT_FRAME_FPS, SMALLEST_ZOOM_SIZE, FFMPEG_PARAM_RENDER_DEFAULT
from modules.audio_analyze import generate_separate_frequency_to_file_audio, analyze_audio_component

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
def process_batch_video(args):
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
    batch_clip.write_videofile(
        batch_clip_file,
        codec='hevc_nvenc',
        audio_codec='aac',
        ffmpeg_params=FFMPEG_PARAM_RENDER_DEFAULT,
        logger=None
    )
    
    # Clear the processed frames to free up memory
    del processed_frames
    clip.close()  # Ensure the clip is properly closed
    del batch_clip
    gc.collect()  # Explicitly call the garbage collector
    
    logging.info(f"Completed {start_idx//batch_size} batches video")
    
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
        parser.add_argument('--layer1', type=str, help='Path to the layer 1 file: Image or Video')
        parser.add_argument('--layer2', type=str, help='Path to the layer 2 file: Image or Video')
        parser.add_argument('--layer3', type=str, help='Path to the layer 3 file: Image or Video')
        parser.add_argument('--mode', choices=['bass', 'treble', 'mid', 'hpss'], help='Audio analysis mode: bass, treble, mid, hpss')
        parser.add_argument('--range', type=int, default=15, help='Window length for smoothing RMS')
        parser.add_argument('--output', type=str, default="output_video.mp4",help='Output video file name')
        parser.add_argument('--threads', type=int, help='Number of threads for multiprocessing')
        parser.add_argument('--domain', type=str, help='Domain for API authentication')
        parser.add_argument('--token', type=str, help='Token for API authentication')
        parser.add_argument('--effect', choices=['zoom', 'blur', 'brightness', 'rgb'], default="zoom", help='Effect reaction for audio amplitude')
        return parser.parse_args()
    except argparse.ArgumentError as e:
        logging.error(f"Argument parsing error: {e}")
        return None
    
def process_frame_temp(params):
    try:
        start_idx, end_idx, image, zoom_factor_function = params
        batch_frames = []

        for idx in range(start_idx, end_idx):
            frame = image.copy()  # Copy original image frame
            time = idx / DEFAULT_FRAME_FPS  # Calculate time in seconds
            zoom_factor = float(zoom_factor_function(time))
            resized_frame = apply_zoom_effect_image(frame, zoom_factor)
            max_width, max_height = image.shape[1], image.shape[0]
            resized_frame = cv2.resize(resized_frame, (max_width, max_height))  # Resize to original image dimensions
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

            batch_frames.append(resized_frame)

        # Perform garbage collection
        gc.collect()

        return batch_frames
    except Exception as e:
        logging.error(f"Error processing frames {start_idx} to {end_idx}: {e}")
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
        
        # Perform garbage collection
        gc.collect()
        
        return resized_frame
    except AssertionError as e:
        logging.error(f"AssertionError processing frame {idx}: {e}")
        cleanup_resources()
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error processing frame {idx}: {e}")
        cleanup_resources()
        sys.exit(1)

def create_video_from_frames(frames_folder, output_video, origin_audio_file_path):
    try:
        # Get all image files in the frames_folder
        image_files = [os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith('.jpg')]
        image_files.sort()  # Sort files in numerical order if necessary

        # Load frames into ImageSequenceClip
        frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in image_files]
        video_clip = mp.ImageSequenceClip(frames, fps=DEFAULT_FRAME_FPS)

        # Load audio from original audio file
        audio_clip = mp.AudioFileClip(origin_audio_file_path)

        # Set audio for the video clip
        video_clip = video_clip.set_audio(audio_clip)

        # Write the final processed video clip to a file
        video_clip.write_videofile(
            output_video,
            codec='hevc_nvenc',
            audio_codec='aac',
            threads=args.threads,
            ffmpeg_params=FFMPEG_PARAM_RENDER_DEFAULT,
            logger=None
        )

        logging.info(f"Video with zoom effect saved to {output_video}")
        print(f"Video created from zoomed frames: {output_video}")
    except Exception as e:
        logging.error(f"Error creating video from frames: {e}")

def render_batch_frames_to_video(batch_start, batch_end, total_frames, image, zoom_factor_function, CACHE_VIDEO_FOLDER):
    try:            
        zoomed_images = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for idx in range(batch_start, batch_end, BATCH_IMAGE_SIZE):
                futures.append(executor.submit(process_frame_temp, (idx, min(idx + BATCH_IMAGE_SIZE, total_frames), image, zoom_factor_function)))

            for future in futures:
                result = future.result()
                if result:
                    zoomed_images.extend(result)

        # Create a video clip from the zoomed frames
        batch_clip = mp.ImageSequenceClip(zoomed_images, fps=DEFAULT_FRAME_FPS)
        batch_clip_file = os.path.join(CACHE_VIDEO_FOLDER, f"batch_{batch_start // BATCH_IMAGE_SIZE}.mp4")
        batch_clip.write_videofile(
            batch_clip_file,
            codec='hevc_nvenc',
            audio=False,
            ffmpeg_params=FFMPEG_PARAM_RENDER_DEFAULT,
            verbose=False,
            logger=None
        )

        # Clean up
        del zoomed_images
        gc.collect()
        
        current_file = len([name for name in os.listdir(CACHE_VIDEO_FOLDER) if os.path.isfile(os.path.join(CACHE_VIDEO_FOLDER, name))])
        
        logging.info(f"Completed {current_file}/{total_frames//BATCH_IMAGE_SIZE + 1} batches video")

        return batch_clip_file

    except Exception as e:
        logging.error(f"Error processing batch from {batch_start} to {batch_end}: {e}")
        return None

def render_video(args):
    start_time = datetime.now()  # Capture the start time

    try:
        logging.info(f"Step 1/3: Process for getting data from audio....")
        
        # Load the audio file
        origin_audio_file_path = args.audio
        
        # RMS data for different audio components (bass, treble, mid, and hpss)
        audio_file_path = generate_separate_frequency_to_file_audio(origin_audio_file_path, args.mode)
        y, sr = librosa.load(audio_file_path, sr=None)

        # Load video clip
        video_file_path = args.layer1
        output_video = args.output
        clip = mp.VideoFileClip(video_file_path)

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

        trough_times = librosa.frames_to_time(troughs, sr=sr, hop_length=hop_length)

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
            calculated_zoom_factor = smoothed_rms[np.argmin(np.abs(librosa.times_like(rms, sr=sr, hop_length=hop_length) - time))]
            if calculated_zoom_factor <= 0:
                frames_zoom_data[frame_index] = SMALLEST_ZOOM_SIZE
            if calculated_zoom_factor > 0:
                frames_zoom_data[frame_index] = round((calculated_zoom_factor/3 + SMALLEST_ZOOM_SIZE), 3)

        # Write frames_zoom_data to file as floats
        output_filename = os.path.join(CACHE_DATA_FOLDER, f'video_zoom_data_{args.mode}.json')
        write_frames_zoom_data_as_float(frames_zoom_data, output_filename)

        # Read frames_zoom_data from file
        frames_zoom_data = read_frames_zoom_data_as_float(output_filename)
        
        logging.info(f"Step 2/3: Start to render batches video with zoom effect....")

        # Convert keys to integers and values to floats (in case JSON loaded them as strings)
        frames_zoom_data = {int(k): float(v) for k, v in frames_zoom_data.items()}

        # Extract frames and zoom factors
        frames = list(frames_zoom_data.keys())
        zoom_factors = list(frames_zoom_data.values())

        # Create an interpolation function
        interpolation_function = interp1d(frames, zoom_factors, kind='quadratic', fill_value='extrapolate')

        # Define the total number of frames in the video
        total_frames = int(clip.fps * clip.duration)

        logging.info(f"Total: {total_frames//BATCH_VIDEO_SIZE + 1} batches video")

        # Generate interpolated zoom factors for each frame
        interpolated_zoom_factors = interpolation_function(np.arange(total_frames))

        # Process frames and apply zoom effect in smaller batches using multiprocessing
        batch_indices = [(start_idx, min(start_idx + BATCH_VIDEO_SIZE, total_frames), video_fps, interpolated_zoom_factors, BATCH_VIDEO_SIZE, video_file_path, CACHE_VIDEO_FOLDER) for start_idx in range(0, total_frames, BATCH_VIDEO_SIZE)]

        # Use a limited number of processes to avoid overloading the system
        num_processes = min(args.threads, multiprocessing.cpu_count())  # Adjust the number of processes as needed
        with multiprocessing.Pool(processes=num_processes) as pool:
            batch_clips = pool.map(process_batch_video, batch_indices)

        # Combine all batch videos into the final video
        final_clips = [mp.VideoFileClip(f) for f in batch_clips]
        final_clip = mp.concatenate_videoclips(final_clips)

        # Load audio from original video clip
        audio_clip = clip.audio

        # Set audio for the final video clip
        final_clip = final_clip.set_audio(audio_clip)
        
        logging.info(f"Step 3/3: Rendering video final...")

        # Write the final processed video clip to a file
        final_clip.write_videofile(
            output_video,
            codec='hevc_nvenc',
            audio_codec='aac',
            ffmpeg_params=FFMPEG_PARAM_RENDER_DEFAULT,
            logger=None
        )

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
        
        end_time = datetime.now()  # Capture the end time
        duration = end_time - start_time  # Calculate the duration
        duration_in_seconds = duration.total_seconds()  # Convert duration to seconds
        
        logging.info(f"Completed with {duration_in_seconds:.1f}s. Video with {args.effect} effect saved to {output_video}")

    except SystemExit as e:
        if e.code != 0:
            print("Error: Missing or invalid arguments.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def render_image(args):
    start_time = datetime.now()  # Capture the start time
    
    try:
        logging.info(f"Step 1/3: Process for getting data from audio....")

        # RMS data for different audio components (bass, treble, mid, and hpss)
        audio_file_path = generate_separate_frequency_to_file_audio(args.audio, args.mode)
        
        # Load the audio file
        origin_audio_file_path = args.audio
        y, sr = librosa.load(audio_file_path, sr=None)
        if y is None:
            raise ValueError(f"Error loading audio file: {args.audio}")

        # Load the image file
        image_file_path = args.layer1
        output_video = args.output
        image = cv2.imread(image_file_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Error loading image file: {args.layer1}")

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
        trough_times = librosa.frames_to_time(troughs, sr=sr, hop_length=hop_length)

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
            if calculated_zoom_factor <= 0:
                frames_zoom_data[time] = SMALLEST_ZOOM_SIZE
            if calculated_zoom_factor > 0:
                frames_zoom_data[time] = round((calculated_zoom_factor/3 + SMALLEST_ZOOM_SIZE), 3)

        # Write frames_zoom_data to file as floats
        output_filename = os.path.join(CACHE_DATA_FOLDER, f'frames_zoom_data_{args.mode}.json')
        write_frames_zoom_data_as_float(frames_zoom_data, output_filename)
        logging.info(f"Write file {output_filename}")

        # Read frames_zoom_data from file
        frames_zoom_data = read_frames_zoom_data_as_float(output_filename)
        logging.info(f"Step 2/3: Start to render batches video with {args.effect} effect....")

        # Convert keys to float and values to float (in case JSON loaded them as strings)
        frames_zoom_data = {float(k): float(v) for k, v in frames_zoom_data.items()}
        
        # Extract frames and zoom factors
        frames = list(frames_zoom_data.keys())
        zoom_factors = list(frames_zoom_data.values())
        
        # Calculate total_frames based on audio duration and 30 frames per second
        total_frames = int(librosa.get_duration(y=y, sr=sr) * DEFAULT_FRAME_FPS)
        
        logging.info(f"Total: {total_frames//BATCH_IMAGE_SIZE + 1} batches video")
        
        # Use multiprocessing to process frames
        zoom_factor_function = interp1d(frames, zoom_factors, kind='quadratic', fill_value='extrapolate')
        
        # Create a list to store batch video file paths
        batch_video_files = []
        num_processes = min(args.threads, multiprocessing.cpu_count())  # Adjust the number of processes as needed

        # Create a pool of processes
        with multiprocessing.Pool(processes=num_processes) as pool:
            try:
                # Map rendering of each batch to the pool
                results = [pool.apply_async(render_batch_frames_to_video, (batch_start, min(batch_start + BATCH_IMAGE_SIZE, total_frames), total_frames, image, zoom_factor_function, CACHE_VIDEO_FOLDER)) for batch_start in range(0, total_frames, BATCH_IMAGE_SIZE)]

                # Collect results
                for result in results:
                    batch_clip_file = result.get(timeout=None)  # No timeout; wait indefinitely for each result
                    if batch_clip_file:
                        batch_video_files.append(batch_clip_file)

            except Exception as e:
                logging.error(f"Error in multiprocessing: {e}")

        cleanup_resources()
        # Concatenate all batch clips into a final video
        final_clip = mp.concatenate_videoclips([mp.VideoFileClip(file) for file in batch_video_files])

        # Load audio from original audio file
        audio_clip = mp.AudioFileClip(origin_audio_file_path)

        # Set audio for the video clip
        final_clip = final_clip.set_audio(audio_clip)
        
        logging.info(f"Step 3/3: Rendering video final...")

        # Write the final processed video clip to a file
        final_clip.write_videofile(
            output_video,
            codec='hevc_nvenc',
            audio_codec='aac',
            ffmpeg_params=FFMPEG_PARAM_RENDER_DEFAULT,
            logger=None
        )
        
        # Clean up temporary batch video files after the final video is rendered
        for batch_clip_file in batch_video_files:
            try:
                clip = mp.VideoFileClip(batch_clip_file)
                clip.close()  # Ensure the clip is properly closed
                os.remove(batch_clip_file)
            except Exception as e:
                logging.error(f"Failed to close clip {batch_clip_file}: {e}")
                
        end_time = datetime.now()  # Capture the end time
        duration = end_time - start_time  # Calculate the duration
        duration_in_seconds = duration.total_seconds()  # Convert duration to seconds
        logging.info(f"Completed with {duration_in_seconds:.1f}s. Video with {args.effect} effect saved to {output_video}")

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



if __name__ == '__main__':
    try:
        multiprocessing.freeze_support()  # For Windows support

        args = parse_arguments()
        
        # Check if audio and image paths are provided and valid
        if not os.path.isfile(args.audio):
            raise FileNotFoundError(f"Audio file not found: {args.audio}")
        if not os.path.isfile(args.layer1):
            raise FileNotFoundError(f"Input file not found: {args.layer1}")
        
        if not os.path.exists(CACHE_DATA_FOLDER):
            os.makedirs(CACHE_DATA_FOLDER, exist_ok=True)
        
        if not os.path.exists(CACHE_VIDEO_FOLDER):
            os.makedirs(CACHE_VIDEO_FOLDER, exist_ok=True)
        
        # Authenticate if email and password are provided
        if args.token:
            status_code, message = authenticate(args.domain, args.token)
            if status_code != 200:
                raise Exception("Authentication failed")

            logging.info(f"Authentication successful with domain {args.domain}")

        # Check if the provided input file is an audio or a video file
        file_path = args.layer1
        file_type = check_file_type(file_path)
        
        # Clean CACHE_DATA_FOLDER && CACHE_VIDEO_FOLDER
        clean_folder(CACHE_VIDEO_FOLDER)
        clean_folder(CACHE_DATA_FOLDER)

        if file_type == 'video':
            render_video(args)
        elif file_type == 'image':
            render_image(args)
        else:
            raise ValueError(f"Unsupported input file type. Please provide an image or video file.")
        
        # Clean CACHE_DATA_FOLDER && CACHE_VIDEO_FOLDER
        clean_folder(CACHE_VIDEO_FOLDER)
        clean_folder(CACHE_DATA_FOLDER)

    except Exception as e:
        multiprocessing.active_children()
        print(f"An unexpected error occurred: {e}")
    except KeyboardInterrupt:
        logging.info("Main process received KeyboardInterrupt. Terminating child processes...")
        cleanup_resources()
        logging.info("All child processes terminated.")
        sys.exit(1)  # Exit cleanly
    finally:
        multiprocessing.active_children()
        cleanup_resources()
        sys.exit(1)
