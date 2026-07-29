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
Die Empfehlung ist jedoch das repo zu klonen und direkt in den Codes die Pfade auf die Messdaten Ordner anzupassen und diese lokal zu starten.
Wenn in einem Unterordner mehrere Files liegen, navigiere zum "Master Skript" und passe dort den Pfad an.
In den jeweiligen Unter-Scripts können feinheiten angepasst und Parameter verändert werden.
Für die effiziente Nutzung können folgende Dokumente nützlich sein: https://eaogroup.sharepoint.com/:f:/r/sites/pj-s56-mtsma-retrofit/Shared%20Documents/General/01_Creation/400_Akustische%20Optimierung%20S57%20Haube/05_Dokumentation/Anleitung%20Akustikbox?csf=1&web=1&e=R1q8Qx

## Hinweise

- Die virtuellen Umgebungen werden lokal im Projektordner unter `.venv/` gehalten.
- Die Skripte erwarten Eingabedateien und Ausgabeverzeichnisse, die je nach Nutzung angepasst werden müssen.
- Für neue Erweiterungen können weitere Skripte in den bestehenden Ordnern unter `src/` ergänzt werden.