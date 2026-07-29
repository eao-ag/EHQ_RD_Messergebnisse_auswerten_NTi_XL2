import os
import glob
import re
import numpy as np
import librosa

# ==============================================================================
# EINSTELLUNGEN
# ==============================================================================

KALIBRIER_OFFSET = 134.22

# Fundamental- und Harmonischen-Band
BANDWIDTH_HZ = 60

USE_A_WEIGHTING = True

# ==============================================================================
# A-GEWICHTUNG
# ==============================================================================


def a_weighting_db(freq_hz):
    """
    IEC 61672 A-Gewichtung in dB
    """

    f = np.maximum(freq_hz, 1e-6)

    ra = (
        (12194.0 ** 2) * (f ** 4)
    ) / (
        ((f ** 2) + (20.6 ** 2))
        * np.sqrt(
            ((f ** 2) + (107.7 ** 2))
            * ((f ** 2) + (737.9 ** 2))
        )
        * ((f ** 2) + (12194.0 ** 2))
    )

    return 20.0 * np.log10(ra) + 2.0


# ==============================================================================
# SWEEP ANALYSE
# ==============================================================================

def analyze_pure_sweep(
    file_path,
    window_duration_sec=0.2,
    step_duration_sec=0.05
):
    """
    Rückgabe:

    Frequenz
    Peak SPL
    Gesamt-SPL(A)
    THD [%]
    """

    data, fs = librosa.load(
        file_path,
        sr=None,
        mono=True
    )

    window_size = int(
        window_duration_sec * fs
    )

    step_size = int(
        step_duration_sec * fs
    )

    hann_window = np.hanning(
        window_size
    )

    frequencies = []
    peak_band_db = []
    total_a_db = []
    thd_values = []

    for start in range(
        0,
        len(data) - window_size,
        step_size
    ):

        end = start + window_size

        chunk = (
            data[start:end]
            * hann_window
        )

        fft_complex = np.fft.rfft(
            chunk
        )

        fft_vals = np.abs(
            fft_complex
        )

        fft_freqs = np.fft.rfftfreq(
            window_size,
            d=1.0 / fs
        )

        peak_idx = np.argmax(
            fft_vals
        )

        peak_freq = fft_freqs[
            peak_idx
        ]

        if peak_freq < 200:
            continue

        if peak_freq > 3100:
            continue

        # ==========================================================
        # FUNDAMENTAL
        # ==========================================================

        fundamental_mask = (
            (fft_freqs >= peak_freq - BANDWIDTH_HZ)
            &
            (fft_freqs <= peak_freq + BANDWIDTH_HZ)
        )

        fundamental_energy = np.sqrt(
            np.sum(
                fft_vals[
                    fundamental_mask
                ] ** 2
            )
        )

        if fundamental_energy <= 0:
            continue

        # ==========================================================
        # PEAK SPL
        # ==========================================================

        peak_amplitude = (
            fundamental_energy
            / (window_size / 2)
        )

        peak_db = 20 * np.log10(
            peak_amplitude + 1e-12
        )

        # ==========================================================
        # GESAMT SPL(A)
        # ==========================================================

        if USE_A_WEIGHTING:

            a_db = a_weighting_db(
                fft_freqs
            )

            a_lin = (
                10.0 ** (
                    a_db / 20.0
                )
            )

            weighted_fft = (
                fft_vals * a_lin
            )

        else:

            weighted_fft = fft_vals

        total_energy_a = np.sqrt(
            np.sum(
                weighted_fft ** 2
            )
        )

        total_amplitude_a = (
            total_energy_a
            / (window_size / 2)
        )

        total_db = 20 * np.log10(
            total_amplitude_a
            + 1e-12
        )

        # ==========================================================
        # THD AUS HARMONISCHEN
        # ==========================================================

        harmonic_energy_sq = 0.0

        for harmonic in range(
            2,
            6
        ):

            harmonic_freq = (
                peak_freq * harmonic
            )

            if harmonic_freq >= fft_freqs[-1]:
                continue

            harmonic_mask = (
                (
                    fft_freqs
                    >=
                    harmonic_freq
                    - BANDWIDTH_HZ
                )
                &
                (
                    fft_freqs
                    <=
                    harmonic_freq
                    + BANDWIDTH_HZ
                )
            )

            harmonic_energy_sq += np.sum(
                fft_vals[
                    harmonic_mask
                ] ** 2
            )

        harmonic_energy = np.sqrt(
            harmonic_energy_sq
        )

        thd_percent = (
            harmonic_energy
            /
            (
                fundamental_energy
                + 1e-12
            )
        ) * 100.0

        frequencies.append(
            peak_freq
        )

        peak_band_db.append(
            peak_db
        )

        total_a_db.append(
            total_db
        )

        thd_values.append(
            thd_percent
        )

    return (
        np.array(
            frequencies
        ),
        np.array(
            peak_band_db
        ),
        np.array(
            total_a_db
        ),
        np.array(
            thd_values
        )
    )


# ==============================================================================
# GLÄTTUNG
# ==============================================================================

def moving_average(
    data,
    window_size=5
):

    if len(data) < window_size:
        return data

    return np.convolve(
        data,
        np.ones(window_size)
        / window_size,
        mode="same"
    )


# ==============================================================================
# HAUPTFUNKTION
# ==============================================================================

def get_fft_curve(
    audio_file
):
    """
    Rückgabe:

    freqs
    peak_spl
    total_a_spl
    thd_percent
    """

    (
        freqs,
        peak_db,
        total_a_db,
        thd_percent
    ) = analyze_pure_sweep(
        audio_file
    )

    if len(freqs) == 0:

        raise ValueError(
            f"Keine gültigen Frequenzpunkte gefunden: {audio_file}"
        )

    peak_db = (
        peak_db
        + KALIBRIER_OFFSET
    )

    total_a_db = (
        total_a_db
        + KALIBRIER_OFFSET
    )

    peak_db = moving_average(
        peak_db,
        25
    )

    total_a_db = moving_average(
        total_a_db,
        25
    )

    thd_percent = moving_average(
        thd_percent,
        5
    )

    return (
        freqs,
        peak_db,
        total_a_db,
        thd_percent
    )


# ==============================================================================
# ORDNERVERARBEITUNG
# ==============================================================================

def process_audio_folder(
    folder_path
):

    results = {}

    search_pattern = os.path.join(
        folder_path,
        "**",
        "*Audio_*.wav"
    )

    wav_files = glob.glob(
        search_pattern,
        recursive=True
    )

    wav_files.sort(
        key=lambda var: [
            int(x)
            if x.isdigit()
            else x
            for x in re.split(
                r"(\d+)",
                var
            )
        ]
    )

    print("")
    print("=" * 60)
    print(
        f"Gefundene WAV-Dateien: {len(wav_files)}"
    )
    print("=" * 60)

    for wav_file in wav_files:

        try:

            (
                freqs,
                peak_spl,
                total_a_spl,
                thd_percent
            ) = get_fft_curve(
                wav_file
            )

            results[wav_file] = {

                "freqs":
                    freqs,

                "peak_spl":
                    peak_spl,

                "total_a_spl":
                    total_a_spl,

                "thdn_percent":
                    thd_percent

            }

            print(
                f"✅ {os.path.basename(wav_file)}"
            )

        except Exception as e:

            print(
                f"❌ Fehler bei "
                f"{os.path.basename(wav_file)}"
            )

            print(e)

    return results