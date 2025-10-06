import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import lfilter

def get_pitch(y, sr):
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    if f0 is not None:
        f0 = f0[~np.isnan(f0)]
        if len(f0) > 0:
            return np.mean(f0), np.std(f0)
    return 0, 0


def get_energy(y):
    energy = np.sum(y ** 2) / len(y)
    amplitude = np.mean(np.abs(y))
    return energy, amplitude


def get_speech_rate(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    duration_sec = librosa.get_duration(y=y, sr=sr)
    return len(beats) / duration_sec if duration_sec > 0 else 0


def get_mfcc(y, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)


import numpy as np
from scipy.signal import lfilter

def get_lpc_coeffs(y, order=12):
    from scipy.signal import lfilter
    y_preemph = lfilter([1, -0.97], 1, y)

    # Autocorrelation
    autocorr = np.correlate(y_preemph, y_preemph, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    R = autocorr[:order+1]
    if R[0] == 0:
        return np.zeros(order+1)

    # Levinson-Durbin recursion (This I didn't know)
    a = np.zeros(order+1)
    e = R[0]
    a[0] = 1.0

    for i in range(1, order+1):
        acc = R[i]
        for j in range(1, i):
            acc -= a[j] * R[i-j]
        k = acc / e

        # Update coefficients
        a_prev = a.copy()
        for j in range(1, i):
            a[j] = a_prev[j] - k * a_prev[i-j]
        a[i] = k
        e *= 1 - k**2

    return a




def get_temporal_features(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    if len(onset_times) < 2:
        return 0, 0, 0
    durations = np.diff(onset_times)
    irregularity = np.std(durations)
    duration_var = np.var(durations)
    onset_count = len(onset_times)
    return irregularity, duration_var, onset_count


#Main Extraction + Merge Function

def extract_features_from_phonations(base_folder, patient_excel, output_csv="training_features.csv"):
    # Load patient metadata
    patient_data = pd.read_excel(patient_excel)
    patient_data.columns = [c.strip() for c in patient_data.columns]

    # Detect ID column name (case-insensitive)
    id_col = [c for c in patient_data.columns if "id" in c.lower()][0]
    patient_data[id_col] = patient_data[id_col].astype(str).str.strip()

    data = []

    # Traverse phonation folders
    for phonation in sorted(os.listdir(base_folder)):
        phonation_path = os.path.join(base_folder, phonation)
        if not os.path.isdir(phonation_path):
            continue

        for file in sorted(os.listdir(phonation_path)):
            if file.lower().endswith(".wav"):
                filepath = os.path.join(phonation_path, file)
                patient_id = os.path.splitext(file)[0].split('_')[0].strip()

                print(f"Processing {phonation}/{file} for patient {patient_id}...")

                # Load audio safely
                try:
                    y, sr = librosa.load(filepath, sr=None)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue

                #Acoustic features
                mean_pitch, std_pitch = get_pitch(y, sr)
                energy, amplitude = get_energy(y)
                speech_rate = get_speech_rate(y, sr)

                #Spectral features
                mfccs = get_mfcc(y, sr)
                lpc = get_lpc_coeffs(y)

                #Temporal features
                irregularity, duration_var, onset_count = get_temporal_features(y, sr)

                # ollect all features
                features = {
                    "Patient_ID": patient_id,
                    "Phonation": phonation,
                    "Pitch_Mean": mean_pitch,
                    "Pitch_Std": std_pitch,
                    "Energy": energy,
                    "Amplitude": amplitude,
                    "Speech_Rate": speech_rate,
                    "Rhythm_Irregularity": irregularity,
                    "Duration_Variability": duration_var,
                    "Onset_Count": onset_count,
                }

                for i, val in enumerate(mfccs):
                    features[f"MFCC_{i+1}"] = val
                for i, val in enumerate(lpc):
                    features[f"LPC_{i+1}"] = val

                data.append(features)

    # Convert extracted features to DataFrame
    features_df = pd.DataFrame(data)

    # If no data was found
    if features_df.empty:
        print("No audio features extracted. Check your folder paths and file names.")
        return

    # Clean and merge with patient metadata
    features_df["Patient_ID"] = features_df["Patient_ID"].astype(str).str.strip()

    merged_df = pd.merge(
        features_df,
        patient_data,
        how="left",
        left_on="Patient_ID",
        right_on=id_col
    )

    # Save to CSV
    merged_df.to_csv(output_csv, index=False)
    print(f"\nFeature extraction and merge complete. Saved to '{output_csv}'.")


#Running main function
if __name__ == "__main__":
    base_folder = "task1/training" 
    patient_excel = "task1/sand_task_1.xlsx" 
    extract_features_from_phonations(base_folder, patient_excel)

