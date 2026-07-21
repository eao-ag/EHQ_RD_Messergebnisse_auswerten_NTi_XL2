# Akustische Optimierung S57 Haube

Dieses Repository enthält Python-Skripte zur Auswertung von XL2-Logfiles und Audiodaten für akustische Messungen an Haubengeometrien.

## Inhalt

- `src/Master skript.py` - Hauptskript zum Verarbeiten aller Logfiles, Erzeugen einzelner Plots und Anzeigen eines kombinierten Overlay-Plots.
- `src/script1_log.py` - Ließt XL2-Logfiles ein und extrahiert Messdaten in ein `pandas.DataFrame`.
- `src/script2_fft.py` - Verarbeitet Audiodateien mit STFT/FFT und bestimmt dominante Frequenzen.
- `src/script3_plot.py` - Verknüpft Log- und FFT-Daten, erstellt Einzeldarstellungen und bereitet Plot-Daten auf.

## Voraussetzungen

Installiere die benötigten Python-Pakete:

```bash
python -m pip install -r requirements.txt
```

### Anforderungen

- pandas
- numpy
- scipy
- matplotlib

## Projektstruktur

- `Messungen/` - Ordner für Rohdaten und Messdateien.
- `Solutions/` - Ausgabeverzeichnis für Ergebnisse, falls verwendet.
- `src/` - Python-Skripte des Projekts.

## Nutzung

1. Lege die Audiodatei und die Logfiles in die entsprechenden Eingabeordner.
2. Öffne `src/Master skript.py` und passe bei Bedarf die Variablen `INPUT_FOLDER`, `AUDIO_FILE` und `OUTPUT_FOLDER` an.
3. Starte das Hauptskript:

```bash
python "src/Master skript.py"
```

4. Das Skript erzeugt PNG-Dateien im Zielordner und zeigt am Ende einen kombinierten Overlay-Plot an.

## Hinweise

- `Master skript.py` verarbeitet das Audio nur einmal und nutzt dann die gleichen FFT-Daten für alle Logfiles.
- Die einzelnen Messplots werden in `OUTPUT_FOLDER` gespeichert.
- Der finale Overlay-Plot wird zur Kontrolle angezeigt, aber nicht automatisch gespeichert.

## Erweiterung

- Für CSV-Exports kann in den Modul-Funktionen `process_logfile` und `process_audio` die Option `export_csv=True` genutzt werden.
- Bei Bedarf können weitere Logfile-Formate ergänzt oder die Zeit-Mapping-Logik in `src/script3_plot.py` angepasst werden.