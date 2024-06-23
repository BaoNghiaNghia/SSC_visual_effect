import librosa
import logging
import matplotlib.pyplot as plt

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
