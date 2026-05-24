#include "Particle.h"
#include "Microphone_PDM.h"
#include "mfcc.h"
#include "model.h"
#include "scaler_values.h"
#include <string.h>
#include <math.h>
#include <float.h>

// Konfiguration
#define SAMPLE_RATE 16000
#define DURATION_SEC 1
#define TOTAL_SAMPLES (SAMPLE_RATE * DURATION_SEC) // 16000
#define TOTAL_FEATURES 806

// Globale buffere (+512 for at forhindre buffer overflow i sliding window)
int16_t audio_raw_buffer[TOTAL_SAMPLES + 512] = {0}; 
float feature_vector[TOTAL_FEATURES];

size_t sample_counter = 0;
bool buffer_ready = false;

char prediction_string[32] = "Venter på lyd...";

// Initialiser MFCC objektet (13 features, 512 ramme-længde)
MFCC mfcc(13, 512, 12);

SYSTEM_THREAD(ENABLED);
SYSTEM_MODE(AUTOMATIC);

SerialLogHandler logHandler(LOG_LEVEL_INFO);

// LED og Servo Setup
const int ledPin = D7;       
const int servoPin = D14;     

Servo minServo;              
bool ledState = false;        
bool servoRunning = false;
int vinkel = 0;

void setup() {
    Serial.begin(115200);
    Particle.variable("kws_live", prediction_string);
    
    // LED og Servo setup
    pinMode(ledPin, OUTPUT);
    digitalWrite(ledPin, LOW);
    minServo.attach(servoPin);
    minServo.write(0);
    
    int err = Microphone_PDM::instance()
        .withOutputSize(Microphone_PDM::OutputSize::SIGNED_16)
        .withRange(Microphone_PDM::Range::RANGE_2048)
        .withSampleRate(SAMPLE_RATE)
        .init();

    if (err) Log.error("PDM decoder init err=%d", err);

    err = Microphone_PDM::instance().start();
    if (err) Log.error("PDM decoder start err=%d", err);
}

void loop() {
    // 1. Opsamling af lyd
    if (!buffer_ready) {
        Microphone_PDM::instance().noCopySamples([](void *pSamples, size_t numSamples) {
            int16_t *incomingData = (int16_t *)pSamples;
            size_t count = Microphone_PDM::instance().getBufferSizeInBytes() / 2;
            
            noInterrupts();
            for (size_t i = 0; i < count; i++) {
                if (sample_counter < TOTAL_SAMPLES) {
                    audio_raw_buffer[sample_counter] = incomingData[i];
                    sample_counter++;
                }
                if (sample_counter >= TOTAL_SAMPLES) {
                    buffer_ready = true;
                    break;
                }
            }
            interrupts();
        });
    }


    static unsigned long last_print = 0;
    if (millis() - last_print > 1000) { // Print en gang i sekundet
        int32_t sum = 0;
        // Vi tager gennemsnittet af det vi har indsamlet indtil nu
        size_t current_len = (sample_counter > 0) ? sample_counter : 1;
        for (size_t i = 0; i < current_len; i++) {
            sum += abs(audio_raw_buffer[i]);
        }
        Log.info("Live Mikrofon-Energi: %ld", sum / current_len);
        last_print = millis();
    }

    // 2. Machine Learning Inference
    if (buffer_ready) {
        // --- 1. Tjek lydstyrken (Energy Gate) ---
        int32_t sum = 0;
        for (size_t i = 0; i < TOTAL_SAMPLES; i++) {
            sum += abs(audio_raw_buffer[i]);
        }
        int32_t current_energy = sum / TOTAL_SAMPLES;

        // Hvis energien er lav (stille i lokalet), springer vi ML over!
        if (current_energy < 200) { 
            // Flyt bufferen 50% ligesom normalt, men skip ML
            size_t overlap_samples = TOTAL_SAMPLES / 2; 
            noInterrupts();
            for(size_t i = 0; i < overlap_samples; i++) {
                audio_raw_buffer[i] = audio_raw_buffer[i + overlap_samples];
            }
            sample_counter = overlap_samples;
            interrupts();
            buffer_ready = false;
            return; // Afbryd og start forfra med at lytte
        }

        int num_frames = 62; // Skal passe med Python output (63 * 13 = 819)
        int step_size = 250; // Skubber vinduet lidt frem ad gangen

        // Sliding Window: Udregn MFCC for hver af de 63 små blokke af 512 samples
        for (int f = 0; f < num_frames; f++) {
            int start_index = f * step_size;
            // Spytter 13 features ud direkte ind på den rigtige plads i feature_vector
            mfcc.mfcc_compute(&audio_raw_buffer[start_index], &feature_vector[f * 13]);
        }

        float scaled_features[TOTAL_FEATURES];

        for (int i = 0; i < TOTAL_FEATURES; i++) {
            if (fabsf(scaler_scale[i]) < 1e-8f) {
                scaled_features[i] = 0.0f;
            } else {
                scaled_features[i] =
                    (feature_vector[i] - scaler_mean[i]) / scaler_scale[i];
            }
        }

        float debug_sum = 0;
        for(int i=0; i<10; i++) debug_sum += scaled_features[i];

        const char* klasser[] = {"On", "Off", "Go", "Stop", "Unknown", "Noise"};

        // Hent procenter for alle 6 klasser
        float probabilities[6];
        model_regress(scaled_features, TOTAL_FEATURES, probabilities, 6);

        // Find den klasse med højest procent
        int best_class = 0;
        float max_prob = probabilities[0];
        for(int i = 1; i < 6; i++) {
            if(probabilities[i] > max_prob) {
                max_prob = probabilities[i];
                best_class = i;
            }
        }

        bool keyword_found = false;

        // Tjek om vi er mindst 75% sikre!
        if (max_prob > 0.75f) {
            Log.info(">>> SIKKERT MATCH: %s (%.1f%%) <<<", klasser[best_class], max_prob * 100.0f);
            
            // Vi reagerer kun hvis det er klasse 0, 1, 2 eller 3 (On, Off, Go, Stop)
            if (best_class < 4) {
                snprintf(prediction_string, sizeof(prediction_string), "%s", klasser[best_class]);
                Particle.publish("keyword_detect", klasser[best_class], PRIVATE);
                keyword_found = true; // Sæt flag for at vi fandt et ord!

                switch(best_class) {
                    case 0: // On
                        digitalWrite(ledPin, HIGH);
                        ledState = true;
                        Log.info("LED On!");
                        break;
                    case 1: // Off
                        digitalWrite(ledPin, LOW);
                        ledState = false;
                        Log.info("LED Off!");
                        break;
                    case 2: // Go
                        servoRunning = true;
                        Log.info("Servo started!");
                        break;
                    case 3: // Stop
                        servoRunning = false;
                        Log.info("Servo stopped!");
                        break;

                }
            }
        } else {
            Log.info("Ignoreret: Gættede på %s, men var kun %.1f%% sikker.", klasser[best_class], max_prob * 100.0f);
        }

        // Buffer Logik
        if (keyword_found) {
            // Vi har lige hørt et ord! Tøm bufferen helt for at undgå at "ekkoet" bliver gættet på næste gang.
            noInterrupts();
            sample_counter = 0;
            interrupts();
            // Log.info("Tømmer buffer for at undgå dobbelt-gæt.");
        } else {
            // Intet ord fundet. Rul bufferen 50% videre for ikke at klippe et ord over.
            size_t overlap_samples = TOTAL_SAMPLES / 4; 
            noInterrupts();
            for(size_t i = 0; i < overlap_samples; i++) {
                audio_raw_buffer[i] = audio_raw_buffer[i + overlap_samples];
            }
            sample_counter = overlap_samples;
            interrupts();
        }
        
        buffer_ready = false;
    }

    // SERVO KONTROL
    static unsigned long lastServoUpdate = 0;
    static bool sweepingUp = true;
    
    if (servoRunning && (millis() - lastServoUpdate > 15)) {
        if (sweepingUp) {
            vinkel++;
            if (vinkel >= 180) {
                vinkel = 180;
                sweepingUp = false;
            }
        } else {
            vinkel--;
            if (vinkel <= 0) {
                vinkel = 0;
                sweepingUp = true;
            }
        }
        minServo.write(vinkel);
        lastServoUpdate = millis();
    }
}