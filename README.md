# Audio Sweep Plotter

Dieses Repository enthält verschiedene Python-Skripte zur Auswertung von XL2-Logfiles und Audiodaten für akustische Messungen.

## Projektstruktur

- `src/Audio_Sweep_Frequency-dB_Plotter/` – Haupt-Workflow für den Audio-Sweep-Plotter
  - `Master skript.py`
  - `script1_log.py`
  - `script2_fft.py`
  - `script3_plot.py`
- `src/Single_Findeton_Plotter/` – einzelner Plotter für Findeton-Auswertung
  - `dB-Plot.py`
- `src/THD[+N]_Plotter/` – zusätzliche THD-/SPL-Analyse-Skripte
  - `Audio_filter_and_SPL_Frequency_sweep_plot_bandwidth_subordinate_master2.py`
  - `Master2 skript FFT & SPL Overlay.py`

## Voraussetzungen

Erstelle und aktiviere eine lokale virtuelle Umgebung direkt im Projektordner:

```powershell
cd EHQ_RD_Audio_Sweep_Plotter
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Abhängigkeiten

Die folgenden Pakete werden für die Skripte benötigt:

- pandas
- numpy
- scipy
- matplotlib
- librosa

## Nutzung

Die Skripte können je nach Bedarf aus ihren jeweiligen Ordnern gestartet werden. Beispiel:

```powershell
python "src/Audio_Sweep_Frequency-dB_Plotter/Master skript.py"
```

## Hinweise

- Die virtuellen Umgebungen werden lokal im Projektordner unter `.venv/` gehalten.
- Die Skripte erwarten Eingabedateien und Ausgabeverzeichnisse, die je nach Nutzung angepasst werden müssen.
- Für neue Erweiterungen können weitere Skripte in den bestehenden Ordnern unter `src/` ergänzt werden.