import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import wiener
from scipy.stats import ttest_rel
import librosa
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Step 1: Noise Reduction using Wiener Filtering
def apply_noise_reduction(audio_path):
    rate, signal = wav.read(audio_path)
    if signal.size == 0:
        raise ValueError(f"Audio file {audio_path} is empty.")
    denoised_signal = wiener(signal)
    denoised_signal = np.nan_to_num(denoised_signal)
    return rate, denoised_signal

# Step 2: MFCC Feature Extraction
def extract_mfcc(rate, signal, n_mfcc=20):
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    mfcc_features = librosa.feature.mfcc(y=signal.astype(float), sr=rate, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc_features.T, axis=0)  # Taking mean across time frames for simplicity
    return mfcc_mean

# Step 3: Train SVM Model
def train_svm_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"SVM Model Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return clf, y_test, y_pred

# Step 4: Statistical Comparison of Models
def perform_statistical_test(X, y, model1, model2):
    scores_model1 = cross_val_score(model1, X, y, cv=5)
    scores_model2 = cross_val_score(model2, X, y, cv=5)
    t_stat, p_value = ttest_rel(scores_model1, scores_model2)
    print("Scores for Model 1:", scores_model1)
    print("Scores for Model 2:", scores_model2)
    print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Significant difference between models (p < 0.05).")
    else:
        print("No significant difference between models (p >= 0.05).")

# Step 5: Speaker Verification Pipeline
def speaker_verification_pipeline(dataset_path):
    X = []  # Feature list
    y = []  # Label list
    for speaker_folder in os.listdir(dataset_path):
        speaker_label = speaker_folder
        speaker_path = os.path.join(dataset_path, speaker_folder)
        for audio_file in os.listdir(speaker_path):
            audio_path = os.path.join(speaker_path, audio_file)
            rate, denoised_signal = apply_noise_reduction(audio_path)
            mfcc_features = extract_mfcc(rate, denoised_signal)
            X.append(mfcc_features)
            y.append(speaker_label)
    clf, y_test, y_pred = train_svm_model(X, y)
    return clf, X, y, y_test, y_pred

# Step 6: Plot Confusion Matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Step 7: Test New Audio
def test_new_audio(model, audio_path):
    rate, denoised_signal = apply_noise_reduction(audio_path)
    mfcc_features = extract_mfcc(rate, denoised_signal)
    predicted_speaker = model.predict([mfcc_features])
    print(f"Predicted Speaker: {predicted_speaker[0]}")

# Main Execution
dataset_path = '/content/drive/MyDrive/VoxForge/DataSet'  # Replace with your dataset path
speaker_verification_model, X, y, y_test, y_pred = speaker_verification_pipeline(dataset_path)
plot_confusion_matrix(y_test, y_pred)

# Statistical Testing
model1 = svm.SVC(kernel='linear')  # SVM
model2 = RandomForestClassifier()  # Example second model
perform_statistical_test(X, y, model1, model2)

# Test with a new audio file
test_audio_path = "/content/drive/MyDrive/VoxForge/TestData/TestSpeaker4/a0487.wav"  # Replace with the test file path
test_new_audio(speaker_verification_model, test_audio_path)
