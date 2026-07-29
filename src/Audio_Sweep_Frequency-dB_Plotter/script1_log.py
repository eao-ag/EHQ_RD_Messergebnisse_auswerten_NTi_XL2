import pandas as pd


def process_logfile(LOGFILE, export_csv=False, output_csv_path=None):
    """
    Liest ein XL2 Logfile ein und gibt ein DataFrame zurück.

    Parameters:
        LOGFILE (str): Pfad zum Logfile
        export_csv (bool): optional CSV speichern
        output_csv_path (str): optionaler Pfad für CSV

    Returns:
        pd.DataFrame mit Spalten: Time, Laq, Laf_max
    """

    data = []
    reading = False

    with open(LOGFILE, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            # Start/Stop erkennen
            if "# Broadband LOG Results" in line:
                reading = True
                continue

            if "#CheckSum" in line:
                break

            if not reading:
                continue

            # Header überspringen
            if line.startswith("Date") or line.startswith("["):
                continue

            parts = line.split()

            if len(parts) < 10:
                continue

            try:
                date = parts[0]
                time = parts[1]

                laf_max = float(parts[5])   # LAFmax_dt
                laq     = float(parts[7])   # LAeq_dt

                timestamp = pd.to_datetime(f"{date} {time}")

                data.append([timestamp, laq, laf_max])

            except ValueError:
                continue

    df = pd.DataFrame(data, columns=["Time", "Laq", "Laf_max"])

    if df.empty:
        print(f"❌ Keine Daten extrahiert aus {LOGFILE}")
        return None

    # ========================
    # Optional CSV Export
    # ========================
    if export_csv:
        if output_csv_path is None:
            output_csv_path = LOGFILE.replace(".txt", "_parsed.csv")

        df.to_csv(output_csv_path, index=False)
        print(f"✅ CSV erstellt: {output_csv_path}")

    return df


# ========================
# Standalone Test (optional)
# ========================
if __name__ == "__main__":
    LOGFILE = r"Messergebnisse_v3_03_06_2026\SLM\2026-06-03_SLM_001_123_Log.txt"

    df = process_logfile(LOGFILE, export_csv=True)

    if df is not None:
        print(df.head())