import os
import json
import numpy as np
import librosa
from scipy.signal import butter, lfilter
from pydub import AudioSegment
import logging
from constants.index import CACHE_DATA_FOLDER, DEFAULT_BASS_AUDIO_FILENAME, DEFAULT_HPSS_AUDIO_FILENAME, DEFAULT_MID_AUDIO_FILENAME, DEFAULT_TREBEL_AUDIO_FILENAME

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
        output_audio = os.path.join(CACHE_DATA_FOLDER, DEFAULT_BASS_AUDIO_FILENAME)
    elif mode == 'treble':
        y_filtered = butter_filter(y, cutoff=4000.0, sr=sr, filter_type='high')
        output_audio = os.path.join(CACHE_DATA_FOLDER, DEFAULT_TREBEL_AUDIO_FILENAME)
    elif mode == 'mid':
        y_filtered = butter_filter(y, lowcut=200.0, highcut=4000.0, sr=sr, filter_type='band')
        output_audio = os.path.join(CACHE_DATA_FOLDER, DEFAULT_MID_AUDIO_FILENAME)
    elif mode == 'hpss':
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        y_filtered = y_harmonic + y_percussive
        output_audio = os.path.join(CACHE_DATA_FOLDER, DEFAULT_HPSS_AUDIO_FILENAME)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    save_audio(y_filtered, sr, output_audio)
    return output_audio



# ------------- END: RMS data for different audio components (bass, treble, mid, and hpss), save the filtered audio to separate files -------------#


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
