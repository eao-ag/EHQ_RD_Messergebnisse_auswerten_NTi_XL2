import os
import re

import librosa
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.signal import stft

from Audio_filter_and_SPL_Frequency_sweep_plot_bandwidth_subordinate_master2 import get_fft_curve

from script1_log import process_logfile
from script3_plot import process_and_map

# ==============================================================================
# EINSTELLUNGEN
# ==============================================================================

INPUT_FOLDER = r"C:\Users\marco.hubacher\Desktop\sus"

MIN_FREQ = 100
MAX_FREQ = 2900

# ==============================================================================


def process_audio(
    AUDIO_FILE,
    window_size=2
):
    try:

        data, fs = librosa.load(
            AUDIO_FILE,
            sr=None,
            mono=True
        )

    except Exception as e:

        print(
            f"❌ Audio konnte nicht geladen werden:\n{AUDIO_FILE}"
        )

        print(e)

        return None

    max_val = np.max(np.abs(data))

    if max_val == 0:

        print(
            f"❌ Audio leer:\n{AUDIO_FILE}"
        )

        return None

    data = data / max_val

    nperseg = int(window_size * fs)

    f, t, Zxx = stft(
        data,
        fs=fs,
        nperseg=nperseg
    )

    Z_mag = np.abs(Zxx)

    dominant_freqs = []

    for i in range(Z_mag.shape[1]):

        idx = np.argmax(
            Z_mag[:, i]
        )

        dominant_freqs.append(
            f[idx]
        )

    dominant_freqs = np.array(
        dominant_freqs
    )

    df = pd.DataFrame(
        {
            "Time_s": t,
            "Frequency_Hz": dominant_freqs
        }
    )

    if df.empty:

        print(
            f"❌ Keine FFT-Daten erzeugt aus:\n{AUDIO_FILE}"
        )

        return None

    return df


audio_files = {}
log_files = {}

all_files = os.listdir(INPUT_FOLDER)

for file in all_files:

    full_path = os.path.join(
        INPUT_FOLDER,
        file
    )

    filename_no_ext = os.path.splitext(
        file
    )[0]

    # Audiodateien
    if file.lower().endswith(".wav"):

        audio_files[
            filename_no_ext
        ] = full_path

    # Logdateien
    elif file.lower().endswith("_log.txt"):

        log_base = re.sub(
            r"_log$",
            "",
            filename_no_ext,
            flags=re.IGNORECASE
        )

        log_files[
            log_base
        ] = full_path


print("\nAudio Files:")
for k in audio_files:
    print(k)

print("\nLog Files:")
for k in log_files:
    print(k)

measurement_ids = sorted(
    audio_files.keys()
)

print("")
print("=" * 80)
print("GEFUNDENE MESSUNGEN")
print("=" * 80)

for measurement_id in measurement_ids:

    print("")
    print(measurement_id)

    print(
        "Audio:",
        os.path.basename(
            audio_files[measurement_id]
        )
    )

    found_log = None

    match = re.match(
        r"(\d{4}-\d{2}-\d{2}_SLM_\d{3})",
        measurement_id
    )

    if match:

        audio_prefix = match.group(1)

        for log_key, log_path in log_files.items():

            if log_key.startswith(
                audio_prefix
            ):

                found_log = log_path
                break

    if found_log is not None:

        print(
            "Log  :",
            os.path.basename(
                found_log
            )
        )

    else:

        print(
            "Log  : KEIN LOGFILE"
        )

print("")
print("=" * 80)
print(
    f"Anzahl Messungen: {len(measurement_ids)}"
)
print("=" * 80)

overlay_data = []

THD_WARNING_LIMIT = 10.0

for measurement_id in measurement_ids:

    print("")
    print("=" * 80)
    print(
        f"VERARBEITE: {measurement_id}"
    )
    print("=" * 80)

    audio_file = audio_files[
        measurement_id
    ]

    # ==========================================================
    # Passendes Logfile suchen (optional)
    # ==========================================================

    log_file = None

    match = re.match(
        r"(\d{4}-\d{2}-\d{2}_SLM_\d{3})",
        measurement_id
    )

    if match:

        audio_prefix = match.group(1)

        for log_key, log_path in log_files.items():

            if log_key.startswith(
                audio_prefix
            ):

                log_file = log_path

                print(
                    f"✅ Logfile gefunden: "
                    f"{os.path.basename(log_file)}"
                )

                break

    if log_file is None:

        print(
            f"⚠️ Kein Logfile für "
            f"{measurement_id}"
        )

    try:

        (
            fft_freqs,
            fft_peak_spl,
            fft_total_a_spl,
            fft_thdn_percent
        ) = get_fft_curve(
            audio_file
        )

        fft_df = process_audio(
            audio_file
        )

        if fft_df is None:

            print(
                "⚠️ FFT Mapping fehlgeschlagen"
            )

            continue

        # ==========================================================
        # LOGFILE OPTIONAL
        # ==========================================================

        if log_file is not None:

            log_df = process_logfile(
                log_file
            )

            if log_df is None:

                print(
                    "⚠️ Keine XL2 Daten gefunden"
                )

                xl2_freqs = np.array([])
                xl2_spl = np.array([])

            else:

                xl2_freqs, xl2_spl = process_and_map(
                    log_df,
                    fft_df
                )

        else:

            print(
                "⚠️ Kein Logfile gefunden"
            )

            xl2_freqs = np.array([])
            xl2_spl = np.array([])

        # ==========================================================
        # THD Plot
        # ==========================================================

        plt.figure(
            figsize=(12, 6)
        )

        plt.plot(
            fft_freqs,
            fft_thdn_percent,
            color="red",
            linewidth=2,
            label="THD"
        )

        plt.axhline(
            y=THD_WARNING_LIMIT,
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Warnschwelle ({THD_WARNING_LIMIT:.0f} %)"
        )

        plt.title(
            f"{measurement_id} - THD"
        )

        plt.xlabel(
            "Frequenz [Hz]"
        )

        plt.ylabel(
            "THD [%]"
        )

        plt.xlim(
            MIN_FREQ,
            MAX_FREQ
        )

        plt.yscale(
            "log"
        )

        plt.grid(
            True,
            which="both",
            linestyle="--",
            alpha=0.5
        )

        plt.legend()

        plt.tight_layout()
        plt.show()

        overlay_data.append(
            (
                measurement_id,
                fft_freqs,
                fft_peak_spl,
                fft_thdn_percent,
                xl2_freqs,
                xl2_spl
            )
        )

    except Exception as e:

        print("")
        print(
            f"❌ Fehler bei {measurement_id}"
        )

        print(e)

# ==============================================================================
# THD Overlay
# ==============================================================================

plt.figure(
    figsize=(14, 8)
)

for (
    measurement_id,
    fft_freqs,
    fft_peak_spl,
    fft_thdn_percent,
    xl2_freqs,
    xl2_spl
) in overlay_data:

    plt.plot(
        fft_freqs,
        fft_thdn_percent,
        linewidth=2,
        label=measurement_id
    )

plt.axhline(
    y=THD_WARNING_LIMIT,
    color="orange",
    linestyle="--",
    linewidth=2,
    label=f"Warnschwelle ({THD_WARNING_LIMIT:.0f} %)"
)

plt.title(
    "THD aller Messungen"
)

plt.xlabel(
    "Frequenz [Hz]"
)

plt.ylabel(
    "THD [%]"
)

plt.xlim(
    MIN_FREQ,
    MAX_FREQ
)

plt.yscale(
    "log"
)

plt.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.5
)

plt.legend(
    fontsize=8
)

plt.tight_layout()
plt.show()

print("")
print("🎉 Alle Messungen verarbeitet")