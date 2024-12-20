import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score
import matplotlib.pyplot as plt
import joblib  # Import joblib to save the model

# Define directories
happy_dir = "C:/Users/anbum/OneDrive/Desktop/speech emotion/processed_speech_data/happy"
neutral_dir = "C:/Users/anbum/OneDrive/Desktop/speech emotion/processed_speech_data/neutral"
sad_dir = "C:/Users/anbum/OneDrive/Desktop/speech emotion/processed_speech_data/sad"

# Function to extract features (MFCC + delta + delta-delta) from audio file
def extract_features(file_path):
    try:
        # Load audio file
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

# Data augmentation function for happy class
def augment_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        # Time stretching
        stretched = librosa.effects.time_stretch(audio, rate=0.9)
        # Pitch shifting
        pitched = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
        return [stretched, pitched]
    except Exception as e:
        print(f"Error augmenting {file_path}: {e}")
        return []

# Prepare dataset
features = []
labels = []

# Function to process files from each emotion directory
def process_files(emotion, dir_path, augment=False):
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(dir_path, file_name)
            feature = extract_features(file_path)
            if feature is not None:
                features.append(feature)
                labels.append(emotion)
            # Apply augmentation if enabled
            if augment and emotion == 'happy':
                augmented_audios = augment_audio(file_path)
                for augmented_audio in augmented_audios:
                    try:
                        mfccs = librosa.feature.mfcc(y=augmented_audio, sr=None, n_mfcc=13)
                        delta_mfccs = librosa.feature.delta(mfccs)
                        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
                        aug_features = np.mean(np.hstack((mfccs, delta_mfccs, delta2_mfccs)), axis=1)
                        aug_std_features = np.std(np.hstack((mfccs, delta_mfccs, delta2_mfccs)), axis=1)
                        features.append(np.concatenate((aug_features, aug_std_features)))
                        labels.append(emotion)
                    except Exception as e:
                        print(f"Error processing augmented audio: {e}")

# Process all audio files
process_files('happy', happy_dir, augment=True)
process_files('neutral', neutral_dir)
process_files('sad', sad_dir)

# Convert features and labels to numpy arrays
X = np.array(features)
y = np.array(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model from GridSearch
model = grid_search.best_estimator_

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred, labels=['happy', 'neutral', 'sad'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['happy', 'neutral', 'sad'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix for Emotion Classification")
plt.show()

# Precision Calculation
precision_happy = precision_score(y_test, y_pred, average=None, labels=['happy'])[0]
precision_neutral = precision_score(y_test, y_pred, average=None, labels=['neutral'])[0]
precision_sad = precision_score(y_test, y_pred, average=None, labels=['sad'])[0]

# Print precision values
print(f"Precision for Happy: {precision_happy:.2f}")
print(f"Precision for Neutral: {precision_neutral:.2f}")
print(f"Precision for Sad: {precision_sad:.2f}")

# Save the trained model as a .pkl file
joblib.dump(model, 'speech_emotion_classifier.pkl')
print("Model saved as 'speech_emotion_classifier.pkl'")
