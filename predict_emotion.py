import os
import librosa
import numpy as np
import joblib  # For loading the saved model

# Function to extract features (MFCC + delta + delta-delta) from audio file
def extract_features(file_path):
    try:
        print(f"Loading audio file: {file_path}")
        audio, sr = librosa.load(file_path, sr=None)
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        # Extract delta features (first derivative of MFCCs)
        delta_mfccs = librosa.feature.delta(mfccs)
        # Extract delta-delta features (second derivative of MFCCs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        # Concatenate MFCC, delta, and delta-delta features and calculate their mean
        features = np.mean(np.hstack((mfccs, delta_mfccs, delta2_mfccs)), axis=1)
        # Add standard deviation for more variation capture
        std_features = np.std(np.hstack((mfccs, delta_mfccs, delta2_mfccs)), axis=1)
        return np.concatenate((features, std_features))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load the trained model (assuming you saved it as 'speech_emotion_classifier.pkl')
model = joblib.load('speech_emotion_classifier.pkl')  # Adjust the path if necessary
print("Model loaded successfully.")

# Function to predict emotion from a new audio file
def predict_emotion(file_path):
    print(f"Processing file: {file_path}")  # Debugging line
    feature = extract_features(file_path)
    if feature is not None:
        print(f"Feature extracted successfully.")  # Debugging line
        feature = feature.reshape(1, -1)  # Reshape for prediction
        prediction = model.predict(feature)
        print(f"Prediction: {prediction[0]}")  # Debugging line
        return prediction[0]
    else:
        print("Error: Feature extraction failed.")  # Debugging line
        return None

# Example usage:
if __name__ == "__main__":
    # Replace with the path to your processed audio file
    audio_file = "C:/Users/anbum/OneDrive/Desktop/speech emotion/processed_speech_data/sad/processed_sad_WhatsApp Audio 2024-12-20 at 2.56.05 PM (1).wav"  # Example path
    emotion = predict_emotion(audio_file)
    if emotion:
        print(f"The detected emotion is: {emotion}")
    else:
        print("Could not process the audio file.")
