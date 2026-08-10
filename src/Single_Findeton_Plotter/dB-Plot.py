import os
import re
import numpy as np
import librosa
import matplotlib.pyplot as plt

# ============================================================
# Einstellungen
# ============================================================

AUDIO_FOLDER = r"C:\Users\marco.hubacher\OneDrive - EAO AG\S56 MTSMa Retrofit _ PJ - General\01_Creation\400_Akustische Optimierung S57 Haube\03_Berechnungen\Python Scripts\Findetöne Messungen\Findeton_Messungen_07_08_2026"

EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac")

HOP_LENGTH = 512
FRAME_LENGTH = 2048

PEAK_DROP_DB = 10
PEAK_SEARCH_THRESHOLD_DB = 6

MAX_DURATION = 7.81  # Sekunden

TARGET_LEVEL = 65.0  # dB gemäss Norm

# ============================================================
# LASmax_dt aus XL2-Logdatei lesen
# ============================================================

def read_lasmax_values(logfile):

    values = []

    with open(logfile, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    data_started = False

    for line in lines:

        if "# Broadband LOG Results" in line:
            data_started = True
            continue

        if not data_started:
            continue

        if line.startswith("#"):
            break

        cols = line.strip().split()

        if (
            len(cols) > 6
            and re.match(r"\d{4}-\d{2}-\d{2}", cols[0])
        ):
            try:
                # LASmax_dt = Spalte 4
                lasmax = float(cols[5])
                values.append(lasmax)

            except:
                pass

    return values

# ============================================================
# Speicherung der Resultate
# ============================================================

all_ls = []

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(14, 7))

for filename in sorted(os.listdir(AUDIO_FOLDER)):

    if not filename.lower().endswith(EXTENSIONS):
        continue

    filepath = os.path.join(AUDIO_FOLDER, filename)

    print("\n" + "=" * 70)
    print(f"Lade: {filename}")

    try:

        # ====================================================
        # Audio laden
        # ====================================================

        y, sr = librosa.load(
            filepath,
            sr=None,
            mono=True
        )

        # ====================================================
        # 1 s vorne weg
        # ====================================================

        start_sample = int(1 * sr)

        # ====================================================
        # 3 s hinten weg
        # ====================================================

        end_sample = len(y) - int(3 * sr)

        if end_sample <= start_sample:
            print("Datei zu kurz")
            continue

        y = y[start_sample:end_sample]

        # ====================================================
        # RMS berechnen
        # ====================================================

        rms = librosa.feature.rms(
            y=y,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )[0]

        # ====================================================
        # dBFS (nicht normiert!)
        # ====================================================

        db = 20 * np.log10(
            rms + 1e-12
        )

        # ====================================================
        # Ersten relevanten Peak finden
        # ====================================================

        global_max = np.max(db)

        threshold = (
            global_max
            - PEAK_SEARCH_THRESHOLD_DB
        )

        frames_100ms = int(
            0.1 * sr / HOP_LENGTH
        )

        peak_idx = None

        for i in range(len(db) - frames_100ms):

            if db[i] >= threshold:

                current_level = db[i]
                level_after = db[i + frames_100ms]

                if (
                    current_level
                    - level_after
                ) > PEAK_DROP_DB:

                    peak_idx = i

                    print(
                        f"Peak gefunden bei "
                        f"{i * HOP_LENGTH / sr:.2f} s"
                    )

                    break

        if peak_idx is None:

            print(
                "Kein passender Peak gefunden."
            )

            continue

        # ====================================================
        # Peak-Zeit bestimmen
        # ====================================================

        peak_time_sec = (
            peak_idx
            * HOP_LENGTH
            / sr
        )

        # ====================================================
        # Ab Peak schneiden
        # ====================================================

        db = db[peak_idx:]

        # ====================================================
        # Auf 7.81 s begrenzen
        # ====================================================

        max_frames = int(
            MAX_DURATION
            * sr
            / HOP_LENGTH
        )

        db = db[:max_frames]

        # ====================================================
        # Passende Logdatei suchen
        # ====================================================

        match = re.search(
            r"SLM_(\d+)",
            filename
        )

        log_path = None
        slm_number = None

        if match:

            slm_number = match.group(1)

            for candidate in os.listdir(AUDIO_FOLDER):

                if (
                    candidate.endswith("_Log.txt")
                    and f"SLM_{slm_number}" in candidate
                ):
                    log_path = os.path.join(
                        AUDIO_FOLDER,
                        candidate
                    )
                    break

        # ====================================================
        # Kalibrierung
        # ====================================================

        if log_path is not None:

            las_values = read_lasmax_values(
                log_path
            )

            if len(las_values) > 0:

                window_mid_time = (
                    peak_time_sec
                    + MAX_DURATION / 2
                )

                log_index = int(
                    round(window_mid_time)
                )

                log_index = min(
                    max(log_index, 0),
                    len(las_values) - 1
                )

                las_reference = (
                    las_values[log_index]
                )

                audio_peak = np.max(db)

                offset = (
                    las_reference
                    - audio_peak
                )

                db = db + offset

                print(
                    f"Logdatei: "
                    f"{os.path.basename(log_path)}"
                )

                print(
                    f"LAS Referenz: "
                    f"{las_reference:.1f} dB(A)"
                )

                print(
                    f"Kalibrier-Offset: "
                    f"{offset:.1f} dB"
                )

        else:

            print(
                f"Keine Logdatei für "
                f"SLM_{slm_number} gefunden"
            )

        # ====================================================
        # Ls gemäss Normbewertung
        # ====================================================

        Ls = 10 * np.log10(
            np.mean(
                10 ** (db / 10)
            )
        )

        all_ls.append(Ls)

        print(
            f"Ls = {Ls:.1f} dB(A)"
        )

        print(
            f"Abweichung zu {TARGET_LEVEL:.0f} dB: "
            f"{Ls - TARGET_LEVEL:+.1f} dB"
        )

        # ====================================================
        # Zeitachse
        # ====================================================

        times = librosa.times_like(
            db,
            sr=sr,
            hop_length=HOP_LENGTH
        )

        # ====================================================
        # Plot
        # ====================================================

        plt.plot(
            times,
            db,
            linewidth=2,
            label=f"{os.path.splitext(filename)[0]} | Ls={Ls:.1f} dB"
        )

    except Exception as e:

        print(
            f"Fehler bei {filename}: {e}"
        )

# ============================================================
# Gesamtauswertung
# ============================================================

print("\n")
print("=" * 70)

if len(all_ls) > 0:

    mean_ls = np.mean(all_ls)

    print(
        f"Durchschnitt aller Single-Findetöne: "
        f"{mean_ls:.1f} dB(A)"
    )

    print(
        f"Abweichung zu {TARGET_LEVEL:.0f} dB: "
        f"{mean_ls - TARGET_LEVEL:+.1f} dB"
    )

print("=" * 70)

# ============================================================
# Darstellung
# ============================================================

plt.title(
    "Findetonvergleich (XL2 LASmax-kalibriert)"
)

plt.xlabel(
    "Zeit [s]"
)

plt.ylabel(
    "Pegel [dB(A)]"
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend(
    fontsize=8
)

plt.tight_layout()
plt.show()