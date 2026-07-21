import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft


def process_audio(AUDIO_FILE, window_size=2, export_csv=False, output_csv_path=None):
    """
    Führt FFT/STFT auf einem Audiofile aus und gibt dominante Frequenzen zurück.

    Parameters:
        AUDIO_FILE (str): Pfad zur Audiodatei
        window_size (float): Fenstergröße in Sekunden
        export_csv (bool): optional CSV speichern
        output_csv_path (str): optionaler CSV Pfad

    Returns:
        pd.DataFrame mit Spalten: Time_s, Frequency_Hz
    """

    # ========================
    # Audio laden
    # ========================
    fs, data = wavfile.read(AUDIO_FILE)

    # Stereo → Mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Normalisieren (robust)
    max_val = np.max(np.abs(data))
    if max_val == 0:
        print(f"❌ Audio leer: {AUDIO_FILE}")
        return None

    data = data / max_val

    # ========================
    # STFT
    # ========================
    nperseg = int(window_size * fs)

    f, t, Zxx = stft(data, fs=fs, nperseg=nperseg)

    Z_mag = np.abs(Zxx)

    # ========================
    # Dominante Frequenz
    # ========================
    dominant_freqs = []

    for i in range(Z_mag.shape[1]):
        idx = np.argmax(Z_mag[:, i])
        dominant_freqs.append(f[idx])

    dominant_freqs = np.array(dominant_freqs)

    # ========================
    # DataFrame
    # ========================
    df = pd.DataFrame({
        "Time_s": t,
        "Frequency_Hz": dominant_freqs
    })

    if df.empty:
        print(f"❌ Keine FFT-Daten erzeugt aus {AUDIO_FILE}")
        return None

    # ========================
    # Optional CSV Export
    # ========================
    if export_csv:
        if output_csv_path is None:
            output_csv_path = AUDIO_FILE.replace(".wav", "_fft.csv")

        df.to_csv(output_csv_path, index=False)
        print(f"✅ FFT CSV erstellt: {output_csv_path}")

    return df


# ========================
# Standalone Test (optional)
# ========================
if __name__ == "__main__":

    AUDIO_FILE = r"Test_Ton_Haubengeometrien_v3.wav"

    df = process_audio(AUDIO_FILE, window_size=2, export_csv=True)

    if df is not None:
        print(df.head())