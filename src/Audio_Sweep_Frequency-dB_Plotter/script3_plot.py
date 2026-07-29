import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def process_and_map(log_df, fft_df):
    """
    Mappt FFT und Log-Daten und gibt x (Frequenz) und y (dB) zurück
    """

    log_df = log_df.copy()
    fft_df = fft_df.copy()

    # ========================
    # Zeit konvertieren
    # ========================
    log_df["Time"] = pd.to_datetime(log_df["Time"])

    start_time = log_df["Time"].min()
    fft_df["Time"] = start_time + pd.to_timedelta(fft_df["Time_s"], unit="s")

    # ========================
    # Endpunkt finden
    # ========================
    laf = log_df["Laf_max"].values
    peaks = np.where(laf > 40)[0]

    if len(peaks) == 0:
        print("⚠️ Kein Peak gefunden")
        return None, None

    if len(peaks) < 2:
        valid_end_index = peaks[-1]
    else:
        valid_end_index = peaks[-2]

    end_time = log_df.loc[valid_end_index, "Time"]

    # ========================
    # Startpunkt
    # ========================
    valid_start_index = max(valid_end_index - 293, 0)
    start_time_clean = log_df.loc[valid_start_index, "Time"]

    # ========================
    # Trim
    # ========================
    log_df = log_df[
        (log_df["Time"] >= start_time_clean) &
        (log_df["Time"] <= end_time)
    ]

    fft_df = fft_df[
        (fft_df["Time"] >= start_time_clean) &
        (fft_df["Time"] <= end_time)
    ]

    # ========================
    # Mapping
    # ========================
    merged = pd.merge_asof(
        fft_df.sort_values("Time"),
        log_df.sort_values("Time"),
        on="Time",
        direction="backward"
    )

    # ========================
    # Sortieren
    # ========================
    merged_sorted = merged.sort_values("Frequency_Hz")

    plot_df = merged_sorted[
        (merged_sorted["Frequency_Hz"] >= 0) &
        (merged_sorted["Frequency_Hz"] <= 5000)
    ]

    x = plot_df["Frequency_Hz"].values
    y = plot_df["Laq"].values

    return x, y


# ========================
# Einzelplot
# ========================
def generate_plot(log_df, fft_df, output_path):

    x, y = process_and_map(log_df, fft_df)

    if x is None:
        print("⚠️ Kein Plot möglich")
        return

    plt.figure(figsize=(10, 5))

    plt.scatter(x, y, s=10, color="red", label="Messdaten")
    plt.plot(x, y, label="Messdaten")

    plt.xlabel("Frequenz [Hz]")
    plt.ylabel("Schalldruckpegel [dB]")
    plt.title("Frequenzgang")

    plt.xlim(260, 3000)
    plt.ylim(50, 105)

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()

    print(f"✅ Plot gespeichert: {output_path}")


# ========================
# Standalone Test
# ========================
if __name__ == "__main__":

    log_df = pd.read_csv("xl2_output.csv")
    fft_df = pd.read_csv("Test_Ton_Haubengeometrien_v1_fft.csv")

    generate_plot(log_df, fft_df, "test_plot.png")