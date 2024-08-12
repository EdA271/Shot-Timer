import moviepy.editor as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Load the video and extract audio
def extract_audio_from_video(video_file):
    video = mp.VideoFileClip(video_file)
    audio = video.audio
    return audio

# Convert audio to a numpy array
def audio_to_array(audio):
    audio_samples = audio.to_soundarray(fps=44100)
    # Convert to mono by averaging the channels if it's stereo
    if audio_samples.shape[1] == 2:
        audio_samples = np.mean(audio_samples, axis=1)
    return audio_samples

# Run FFT on the audio data
def run_fft(audio_samples, sample_rate):
    N = len(audio_samples)
    yf = fft(audio_samples)
    xf = fftfreq(N, 1 / sample_rate)
    return xf, np.abs(yf)

# Plot the FFT result
def plot_fft(xf, yf, sample_rate):
    plt.figure(figsize=(12, 6))
    plt.plot(xf[:len(xf)//2], yf[:len(yf)//2])
    plt.title("FFT of Audio Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

def plot_audio_waveform(video_file):
    # Load the video and extract audio
    video = mp.VideoFileClip(video_file)
    audio = video.audio

    # Convert audio to a numpy array
    audio_samples = audio.to_soundarray(fps=44100)
    
    # Convert to mono by averaging the channels if it's stereo
    if audio_samples.shape[1] == 2:
        audio_samples = np.mean(audio_samples, axis=1)
    
    # Calculate the time axis for the plot
    duration = audio.duration  # Duration in seconds
    time_axis = np.linspace(0, duration, num=len(audio_samples))
    
    # Plot the audio waveform
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, audio_samples, color='blue')
    plt.title("Audio Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    video_file = "test1.mp4"  # Replace with your video file path
    sample_rate = 44100  # Sample rate in Hz

    # Extract and process audio
    audio = extract_audio_from_video(video_file)
    audio_samples = audio_to_array(audio)

    # Run FFT and plot
    xf, yf = run_fft(audio_samples, sample_rate)
    # print(xf)
    # print(yf)
    # plot_fft(xf, yf, sample_rate)
    plot_audio_waveform(video_file)