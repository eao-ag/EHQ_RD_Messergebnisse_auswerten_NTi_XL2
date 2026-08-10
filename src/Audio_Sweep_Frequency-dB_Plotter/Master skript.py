import os
import matplotlib.pyplot as plt

from script1_log import process_logfile
from script2_fft import process_audio
from script3_plot import generate_plot, process_and_map

# ========================
# SETTINGS
# ========================
INPUT_FOLDER = r"C:\Users\marco.hubacher\Desktop\Messergebnisse_Hochpass_v5"
AUDIO_FILE = r"C:\Users\marco.hubacher\Desktop\Test_Ton_Haubengeometrien_v3.wav"
OUTPUT_FOLDER = r"C:\Users\marco.hubacher\Desktop\sol_v18_Hochpass_v5"

print(repr(INPUT_FOLDER))

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ========================
# 1. Audio nur einmal verarbeiten
# ========================
print("🔊 Verarbeite Audio...")

fft_df = process_audio(AUDIO_FILE)

if fft_df is None:
    print("❌ FFT Verarbeitung fehlgeschlagen → Abbruch")
    exit()

print("✅ Audio erfolgreich verarbeitet")

# ========================
# Speicher für Overlay
# ========================
all_results = []
labels = []

# ========================
# 2. Alle Logfiles durchlaufen
# ========================
files = sorted(os.listdir(INPUT_FOLDER))

for file in files:

    if not (file.endswith(".txt") or file.endswith(".csv")):
        continue

    filepath = os.path.join(INPUT_FOLDER, file)

    print(f"\n📂 Verarbeite: {file}")

    try:
        # ========================
        # Logfile einlesen
        # ========================
        log_df = process_logfile(filepath)

        if log_df is None:
            print(f"⚠️ Überspringe (keine Daten): {file}")
            continue

        # ========================
        # Einzelplot speichern
        # ========================
        output_name = os.path.splitext(file)[0] + ".png"
        output_path = os.path.join(OUTPUT_FOLDER, output_name)

        #generate_plot(log_df, fft_df, output_path)

        print(f"✅ Plot gespeichert: {output_path}")

        # ========================
        # Daten für Overlay sammeln
        # ========================
        x, y = process_and_map(log_df, fft_df)

        all_results.append((x, y))
        labels.append(os.path.splitext(file)[0])

    except Exception as e:
        print(f"❌ Fehler bei {file}: {e}")

# ========================
# ✅ Overlay Plot (nur Anzeige)
# ========================
if len(all_results) > 0:
    print("\n📊 Zeige kombinierten Plot...")

    plt.figure(figsize=(10, 6))

    for i, (x, y) in enumerate(all_results):
        plt.plot(x, y, label=labels[i])

    # ========================
    # Balkenmarker für Sollbereiche
    # ========================
    marker = [
        (630, 65, 2, 2, "630 Hz: 65 ± 2 dB"),
        (550, 70, 2, 2, "550 Hz: 70 ± 2 dB"),
        (750, 70, 2, 2, None), 
        (1760, 70, 0, 6, "1760/2200 Hz: 70 bis 76 dB"),
        (2200, 70, 0, 6, None),
        (1900, 70, 0, 6, "1900 Hz: 70 bis 76 dB"),
    ]

    for frequenz, sollwert, minus, plus, beschriftung in marker:
        plt.errorbar(
            frequenz,
            sollwert,
            yerr=[[minus], [plus]],
            fmt="o",
            markersize=7,
            color="red",
            ecolor="red",
            elinewidth=5,
            capsize=8,
            capthick=2,
            label=beschriftung,
            zorder=10,
        )


    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("Schalldruckpegel [dB]")
    plt.title("Vergleich aller Messungen")

    plt.xlim(260, 3000)
    plt.ylim(50, 105)

    plt.legend()
    plt.grid()
    plt.tight_layout()

    # 👉 NUR anzeigen, NICHT speichern
    plt.show()

print("\n🎉 Alle Dateien verarbeitet!")