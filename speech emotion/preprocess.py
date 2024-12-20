import os
import numpy as np
from pydub import AudioSegment
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# Function to load and preprocess audio files
def preprocess_audio_files(input_folder, emotion):
    print(f"Processing {emotion} emotion...")
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            wav_file_path = os.path.join(input_folder, filename)
            print(f"Processing file: {filename}")

            # Load audio using pydub
            audio = AudioSegment.from_wav(wav_file_path)
            print(f"Loaded {filename} with duration: {audio.duration_seconds} seconds")

            # Normalize audio volume
            normalized_audio = audio.apply_gain(-audio.dBFS)
            print(f"Normalized {filename} to 0 dBFS")

            # Resample the audio to a consistent sample rate (e.g., 16000 Hz)
            normalized_audio = normalized_audio.set_frame_rate(16000)
            print(f"Resampled {filename} to 16000 Hz")

            # Convert audio to numpy array for further processing
            samples = np.array(normalized_audio.get_array_of_samples())

            # Extract features (e.g., MFCC using librosa)
            print(f"Extracting features for {filename}...")
            y, sr = librosa.load(wav_file_path, sr=16000)  # Load with 16kHz sample rate
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            print(f"Extracted MFCC features for {filename}: shape {mfcc.shape}")

            # Save processed audio file if needed (optional)
            output_wav_path = os.path.join(input_folder, f'processed_{emotion}_{filename}')
            write(output_wav_path, 16000, samples)
            print(f"Saved processed file as: {output_wav_path}")

            # Optional: Save a visual representation of MFCC
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfcc, x_axis='time', sr=sr)
            plt.colorbar()
            plt.title(f'MFCC for {filename}')
            plt.savefig(os.path.join(input_folder, f'mfcc_{emotion}_{filename}.png'))
            plt.close()
            print(f"MFCC plot saved for {filename}")

# Define paths for your emotion folders
happy_folder = r'C:\Users\anbum\OneDrive\Desktop\speech emotion\datasets\happy'
neutral_folder = r'C:\Users\anbum\OneDrive\Desktop\speech emotion\datasets\neutral'
sad_folder = r'C:\Users\anbum\OneDrive\Desktop\speech emotion\datasets\sad'

# Preprocess each emotion folder
preprocess_audio_files(happy_folder, 'happy')
preprocess_audio_files(neutral_folder, 'neutral')
preprocess_audio_files(sad_folder, 'sad')
