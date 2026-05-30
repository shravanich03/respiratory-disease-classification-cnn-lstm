import numpy as np
import librosa
from scipy.signal import butter, sosfilt

TARGET_SR = 22050
TARGET_DURATION = 3.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)
N_MFCC = 40
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512


def load_and_resample(file_path):
    """Load audio file and resample to 22,050 Hz."""
    audio, sr = librosa.load(file_path, sr=None)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio


def apply_bandpass_filter(audio, lowcut=100, highcut=2000, order=4):
    """Apply Butterworth bandpass filter (100–2000 Hz)."""
    sos = butter(order, [lowcut, highcut], btype='band', fs=TARGET_SR, output='sos')
    return sosfilt(sos, audio)


def normalize_amplitude(audio):
    """Normalize audio amplitude to range [-1, 1]."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio


def pad_or_trim(audio):
    """Pad or trim audio to exactly TARGET_SAMPLES length."""
    if len(audio) < TARGET_SAMPLES:
        audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))
    else:
        audio = audio[:TARGET_SAMPLES]
    return audio


def extract_mfcc(audio):
    """Extract 40 MFCC coefficients. Output shape: (40, 128)."""
    mfcc = librosa.feature.mfcc(y=audio, sr=TARGET_SR, n_mfcc=N_MFCC,
                                  n_fft=N_FFT, hop_length=HOP_LENGTH)
    # Ensure fixed shape (40, 128)
    if mfcc.shape[1] < 128:
        mfcc = np.pad(mfcc, ((0, 0), (0, 128 - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :128]
    return mfcc


def extract_mel_spectrogram(audio):
    """Extract log Mel-spectrogram. Output shape: (128, 128)."""
    mel = librosa.feature.melspectrogram(y=audio, sr=TARGET_SR, n_mels=N_MELS,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] < 128:
        mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :128]
    return mel_db


def extract_spectral_features(audio):
    """Extract spectral features: centroid, rolloff, bandwidth, contrast, ZCR, RMS, chroma, tonnetz."""
    features = {}
    features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=TARGET_SR))
    features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=TARGET_SR))
    features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=TARGET_SR))
    features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio))
    features['rms'] = np.mean(librosa.feature.rms(y=audio))
    features['chroma'] = np.mean(librosa.feature.chroma_stft(y=audio, sr=TARGET_SR))
    return features


def preprocess_audio(file_path):
    """
    Full preprocessing pipeline.
    Returns: mfcc (40x128), mel_spectrogram (128x128), spectral_features (dict)
    """
    audio = load_and_resample(file_path)
    audio = apply_bandpass_filter(audio)
    audio = normalize_amplitude(audio)
    audio = pad_or_trim(audio)

    mfcc = extract_mfcc(audio)
    mel_spec = extract_mel_spectrogram(audio)
    spectral = extract_spectral_features(audio)

    return mfcc, mel_spec, spectral
