# Keyword-Smart-System

A live Keyword Spotting (KWS) project designed for microcontrollers (Particle). The system uses a PDM microphone and a lightweight Neural Network (MLP) to listen for and recognize voice commands in real-time (Edge AI). 

The model is currently trained to recognize 6 classes: **"On", "Off", "Go", "Stop", "Unknown", and "Noise"**.

## Key Features
* **Real-time Inference:** The model runs 100% locally on the device without relying on any cloud services.
* **Energy Gating:** The system continuously measures the ambient audio energy level and skips Machine Learning inference when the room is quiet. This saves processing power and eliminates false positives in silent environments.
* **Hardware-Accelerated MFCC:** Extracting the audio fingerprints (Mel-Frequency Cepstral Coefficients) utilizes ARM CMSIS-DSP, making calculations extremely fast on Cortex-M processors.
* **Sliding Window:** Ensures continuous, non-stop listening by shifting and preserving parts of the audio buffer so words aren't accidentally cut in half.
* **Calibrated Pipeline:** The C++ DSP code is strictly calibrated to match the Python library `librosa`, ensuring the model behaves identically during training and live hardware deployment.

---

## Project Structure

The project is split into two main parts: Training (Python) and Live Deployment (C++).

### 1. Machine Learning & Training (Python)
* `script.py`: This script reads `.wav` audio files from the `lyd_data/` directory, performs *Data Augmentation* (time-shifting and white noise injection) to make the model robust, and trains a Multi-Layer Perceptron (MLP) neural network using Scikit-Learn.
* **emlearn**: Finally, the script automatically converts the trained model into a C-header file (`model.h`), which can be embedded directly into the microcontroller code.
* The script also computes the StandardScaler parameters (`scaler_values.h`) to properly normalize the live incoming audio.

### 2. Microcontroller Software (C++)
* `Keyword-spotting.cpp`: The main firmware file. It manages the PDM microphone, gathers 1 second of audio at 16000 Hz, runs the *Energy Gating* check, extracts features using a sliding window (62 frames of 13 MFCC features), and feeds them into the embedded model for an instant prediction.
* `Mfcc.cpp` / `Mfcc.h`: Contains the CMSIS-DSP logic used to convert raw time-domain audio waves into the frequency-domain feature vectors.

---

## Getting Started

### Train your own model
1. Create a folder named `lyd_data/` in the root of the project.
2. Create subfolders for each individual class (e.g., `On`, `Off`, `Noise`) and place your 1-second, 16kHz `.wav` files inside them.
3. Run the script to start the training pipeline:
```bash
   python script.py