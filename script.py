import os
import json
import librosa
import numpy as np
import emlearn
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Konfiguration
SR = 16000
DURATION = 1.0
SAMPLES = int(SR * DURATION) 
N_MFCC = 13
TEST_SIZE = 0.2
RANDOM_STATE = 42

klasser = ["On", "Off", "Go", "Stop", "Unknown", "Noise"]

X = []
y = []

# Data Augmentation Funktioner
def time_shift(audio, max_shift=2000):
    """Forskyder lyden tilfældigt frem eller tilbage (tidsforskydning)."""
    shift = np.random.randint(-max_shift, max_shift)
    return np.roll(audio, shift)

def add_noise(audio, noise_level=0.005):
    """Tilføjer en lille smule tilfældigt hvid støj til vores klip."""
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise

# -------------------------------------

print("Starter feature-ekstraktion til MLP med Data Augmentation...")

for klasse_idx, klasse_navn in enumerate(klasser):
    mappe = f"./lyd_data/{klasse_navn}/"
    
    if not os.path.exists(mappe):
        print(f"Mappen '{mappe}' findes ikke. Springer over.")
        continue
        
    for fil in os.listdir(mappe):
        if not fil.lower().endswith(".wav"):
            continue
            
        fil_sti = os.path.join(mappe, fil)
        try:
            # 1. Indlæs den originale lyd
            y_sound, sr_val = librosa.load(fil_sti, sr=SR, mono=True)
            
            # 2. Sørger for at arrayet er PRÆCIS 16.000 samples langt.
            if len(y_sound) < SAMPLES:
                y_sound = np.pad(y_sound, (0, SAMPLES - len(y_sound)), mode='constant')
            else:
                y_sound = y_sound[:SAMPLES]
            
            # Den originale lydfil
            mf_orig = librosa.feature.mfcc(y=y_sound, sr=sr_val, n_mfcc=N_MFCC, n_fft=512, hop_length=250)
            X.append(mf_orig.flatten().tolist())
            y.append(klasse_idx)
            
            # Vi forstærker jeres keywords, da Unknown og Noise allerede har andre variationer
            if klasse_navn not in ["Unknown", "Noise"]:
                
                # 1. Augmentering: Tidsforskydning (Time Shift)
                y_shifted = time_shift(y_sound, max_shift=1600) # op til ~100ms forskydning
                mf_shifted = librosa.feature.mfcc(y=y_shifted, sr=sr_val, n_mfcc=N_MFCC, n_fft=512, hop_length=250)
                X.append(mf_shifted.flatten().tolist())
                y.append(klasse_idx)
                
                # 2. Augmentering: Tidsforskydning + Tilfældig støj
                y_noisy = add_noise(y_shifted, noise_level=0.003)
                mf_noisy = librosa.feature.mfcc(y=y_noisy, sr=sr_val, n_mfcc=N_MFCC, n_fft=512, hop_length=250)
                X.append(mf_noisy.flatten().tolist())
                y.append(klasse_idx)
                
        except Exception as e:
            pass # Ignorer filer der ikke kan indlæses

X = np.array(X)
y = np.array(y)

# Datastørrelse efter augmentation
print(f"Færdig med at udtrække features! Data shape: {X.shape}") 

if len(X) > 0:
    print("\nNy klassefordeling i datasættet efter Data Augmentation:")
    total_counts = Counter(y)
    for k, v in sorted(total_counts.items()):
        print(f" - {klasser[k]}: {v} samples")
    print("-" * 40)

    # Opdel data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    # Skaler data
    print("Skalerer features med StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Gem scaler-værdierne
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    with open("scaler_values.json", "w") as f:
        json.dump(scaler_data, f)
    print("Gemte scalerens 'mean' og 'scale' i 'scaler_values.json'.")
    
    # Træner MLP
    print("\nTræner MLP (Neuralt Netværk)...")
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64), 
        activation='relu',
        solver='adam',
        alpha=0.01,                  
        learning_rate_init=0.001,
        max_iter=500,
        random_state=RANDOM_STATE,
        early_stopping=True,         
        n_iter_no_change=15
    )
    clf.fit(X_train, y_train)
    
    # Generer Træningshistorik Plot (Loss & Validation Accuracy)
    print("\nGenererer trænings- og valideringsplots...")
    plt.figure(figsize=(12, 5))

    # Training Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(clf.loss_curve_, label='Training Loss', color='blue', lw=2)
    plt.title('Model Trænings-Loss')
    plt.ylabel('Loss')
    plt.xlabel('Iterationer / Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    # Validation Accuracy Curve
    if hasattr(clf, 'validation_scores_') and clf.validation_scores_ is not None:
        plt.subplot(1, 2, 2)
        plt.plot(clf.validation_scores_, label='Validation Accuracy', color='orange', lw=2)
        plt.title('Model Validerings-Accuracy')
        plt.ylabel('Accuracy (Andel rigtige)')
        plt.xlabel('Iterationer / Epochs')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    else:
        print("Kunne ikke generere valideringsplot (kræver early_stopping=True)")

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=300)
    print("Træningshistorik blev gemt som 'training_history.png'!")
    
    # Evaluer model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy på testsæt: {acc * 100:.2f}%")
    print("\nKlassifikationsrapport:")
    print(classification_report(y_test, y_pred, target_names=klasser, digits=3))
    
    # Generer og gem Confusion Matrix
    print("\nGenererer Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=klasser)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title("Confusion Matrix - Keyword Spotting (Augmented MLP)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    print("Confusion Matrix blev gemt som 'confusion_matrix.png'!")
    
    # Konverter modellen til C++
    print("\nKonverterer MLP model til C++ med emlearn...")
    c_model = emlearn.convert(clf)
    c_model.save(file="model.h")
    print("Modellen blev gemt som 'model.h'!")
else:
    print("Ingen data fundet. Tjek jeres .wav-filer.")