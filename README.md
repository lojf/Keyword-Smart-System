# Keyword-Smart-System

Et live Keyword Spotting (KWS) projekt designet til microcontrollere (Particle). Systemet bruger en PDM-mikrofon og et letvægts neuralt netværk (MLP) til at lytte efter og genkende stemmekommandoer i realtid (Edge AI). 

Modellen er i øjeblikket trænet til at genkende 6 klasser: **"On", "Off", "Go", "Stop", "Unknown" og "Noise"**.

## Nøglefunktioner
* **Real-time Inferens:** Modellen kører 100% lokalt på enheden uden brug af cloud-tjenester.
* **Energy Gating:** Systemet måler lydniveauet og kører kun Machine Learning-delen, når der rent faktisk er lyd. Det sparer processorkraft og eliminerer falske gæt i et stille rum.
* **Hardware-Accelereret MFCC:** Udtrækning af lydens fingeraftryk (Mel-Frequency Cepstral Coefficients) bruger ARM CMSIS-DSP, hvilket gør beregningerne lynhurtige på Cortex-M processorer.
* **Sliding Window:** Sikrer kontinuerlig lytning ved at gemme dele af lydbufferen, så ord ikke bliver klippet over på midten.
* **Kalibreret Pipeline:** C++ DSP-koden er præcist kalibreret til at matche Python-biblioteket `librosa`, hvilket sikrer, at modellen opfører sig ens under træning og live-brug.

---

## Projektstruktur

Projektet er delt op i to dele: Træning (Python) og Live-kørsel (C++).

### 1. Machine Learning & Træning (Python)
* `script.py`: Scriptet læser lydfiler (`.wav`) fra mappen `lyd_data/`, udfører *Data Augmentation* (tidsforskydning og støjtilsætning) for at gøre modellen mere robust, og træner et Multi-Layer Perceptron (MLP) neuralt netværk via Scikit-Learn.
* **emlearn**: Til sidst konverterer scriptet automatisk den trænede model til en C-header fil (`model.h`), som kan lægges direkte over på microcontrolleren. 
* Scriptet udregner også StandardScaler-værdier (`scaler_values.h`), så live-lyden normaliseres korrekt.

### 2. Microcontroller Software (C++)
* `Keyword-spotting.cpp`: Hovedprogrammet. Styrer PDM-mikrofonen, opsamler lyd (1 sekund, 16000 Hz), kører *Energy Gating*, udtrækker features i et sliding window (63 rammer af 13 MFCC features), og spørger den konverterede model om et gæt.
* `Mfcc.cpp` / `Mfcc.h`: Indeholder CMSIS-DSP logikken til at omregne rå lydbølger til features.

---

## Kom godt i gang

### Træn din egen model
1. Opret en mappe der hedder `lyd_data/` i projektets rod.
2. Lav undermapper for hver klasse (f.eks. `On`, `Off`, `Noise`) og læg dine `.wav`-filer (16kHz, 1-sekund) derind.
3. Kør scriptet for at træne modellen:
```bash
   python script.py