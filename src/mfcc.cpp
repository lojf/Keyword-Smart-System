/**
 * ============================================================================
 * KILDEHENVISNING OG OPEN-SOURCE CREDITS
 * 
 * Denne implementation bruger ARM CMSIS-DSP (arm_rfft_fast_f32) til at 
 * hardware-accelerere FFT-beregningen på Cortex-M processoren.
 * MFCC implementering er fra https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Deployment/Source/MFCC/mfcc.cpp
 * ============================================================================
 */

#include "mfcc.h"
#include <float.h>

// MFCC KLASSE IMPLEMENTERING
MFCC::MFCC(int num_mfcc_features, int frame_len, int mfcc_dec_bits) 
:num_mfcc_features(num_mfcc_features), frame_len(frame_len), mfcc_dec_bits(mfcc_dec_bits) {
    frame_len_padded = 512;
    frame = new float[frame_len_padded];
    buffer = new float[frame_len_padded];
    fft_output = new float[frame_len_padded]; // Ny buffer til ARMs output
    mel_energies = new float[NUM_FBANK_BINS];

    window_func = new float[frame_len];
    for (int i = 0; i < frame_len; i++)
        window_func[i] = 0.5f - 0.5f * cosf(2.0f * M_PI * ((float)i) / (frame_len));

    fbank_filter_first = new int32_t[NUM_FBANK_BINS];
    fbank_filter_last = new int32_t[NUM_FBANK_BINS];
    mel_fbank = create_mel_fbank();
    dct_matrix = create_dct_matrix(NUM_FBANK_BINS, num_mfcc_features);

    // Initialiser ARM CMSIS-DSP FFT instansen
    arm_rfft_fast_init_f32(&fft_instance, frame_len_padded);
}

MFCC::~MFCC() {
    delete [] frame; delete [] buffer; delete [] fft_output; 
    delete [] mel_energies; delete [] window_func;
    delete [] fbank_filter_first; delete [] fbank_filter_last; delete [] dct_matrix;
    for(int i=0; i<NUM_FBANK_BINS; i++) delete [] mel_fbank[i];
    delete [] mel_fbank;
}

float * MFCC::create_dct_matrix(int32_t input_length, int32_t coefficient_count) {
    float * M = new float[input_length*coefficient_count];
    float normalizer = sqrtf(2.0f / (float)input_length);
    for (int k = 0; k < coefficient_count; k++) {
        for (int n = 0; n < input_length; n++) {
            M[k*input_length+n] = normalizer * cosf( (float)M_PI / input_length * (n + 0.5f) * k );
        }
    }
    return M;
}

float ** MFCC::create_mel_fbank() {
    int32_t num_fft_bins = frame_len_padded/2;
    float fft_bin_width = (float)SAMP_FREQ / frame_len_padded;
    float mel_low_freq = MelScale((float)MEL_LOW_FREQ);
    float mel_high_freq = MelScale((float)MEL_HIGH_FREQ); 
    float mel_freq_delta = (mel_high_freq - mel_low_freq) / (NUM_FBANK_BINS+1);

    float *this_bin = new float[num_fft_bins];
    float ** m_fbank =  new float*[NUM_FBANK_BINS];

    for (int bin = 0; bin < NUM_FBANK_BINS; bin++) {
        float left_mel = mel_low_freq + bin * mel_freq_delta;
        float center_mel = mel_low_freq + (bin + 1) * mel_freq_delta;
        float right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;
        int32_t first_index = -1, last_index = -1;

        for (int i = 0; i < num_fft_bins; i++) {
            float freq = (fft_bin_width * i);
            float mel = MelScale(freq);
            this_bin[i] = 0.0f;

            if (mel > left_mel && mel < right_mel) {
                float weight = (mel <= center_mel) ? 
                    (mel - left_mel) / (center_mel - left_mel) : 
                    (right_mel - mel) / (right_mel - center_mel);
                this_bin[i] = weight;
                if (first_index == -1) first_index = i;
                last_index = i;
            }
        }

        fbank_filter_first[bin] = first_index;
        fbank_filter_last[bin] = last_index;
        m_fbank[bin] = new float[last_index-first_index+1]; 

        int32_t j = 0;
        for (int i = first_index; i <= last_index; i++) {
            m_fbank[bin][j++] = this_bin[i];
        }
    }
    delete [] this_bin;
    return m_fbank;
}

void MFCC::mfcc_compute(const int16_t * audio_data, float* mfcc_out) {
    // Digital gain
    float volume_gain = 1.0f; 
    for (int i = 0; i < frame_len; i++) {
        float sample = ((float)audio_data[i] / 32768.0f) * volume_gain;
        if (sample > 1.0f) sample = 1.0f;
        if (sample < -1.0f) sample = -1.0f;
        
        frame[i] = sample * window_func[i];
    }
    for (int i = frame_len; i < frame_len_padded; i++) {
        frame[i] = 0.0f;
    }

    // --- UDFØR ARM HARDWARE FFT ---
    // 0 betyder "Forward FFT" (og ikke Inverse FFT)
    arm_rfft_fast_f32(&fft_instance, frame, fft_output, 0);

    // Beregn Power Spectrum ud fra ARMs "pakkede" format
    int32_t half_dim = frame_len_padded / 2;
    
    float scale = 1.0f / (float)(frame_len_padded * frame_len_padded);

    buffer[0] = fft_output[0] * fft_output[0] * scale;

    for (int i = 1; i < half_dim; i++) {
        float real = fft_output[2 * i];
        float imag = fft_output[2 * i + 1];
        buffer[i] = ((real * real) + (imag * imag)) * scale;
    }
 
    for (int bin = 0; bin < NUM_FBANK_BINS; bin++) {
        int32_t j = 0;
        float mel_energy = 0;
        int32_t first_index = fbank_filter_first[bin];
        int32_t last_index = fbank_filter_last[bin];
        for (int i = first_index; i <= last_index; i++) {
            mel_energy += buffer[i] * mel_fbank[bin][j++];
        }
        mel_energies[bin] = (mel_energy == 0.0f) ? FLT_MIN : mel_energy;
    }

    for (int bin = 0; bin < NUM_FBANK_BINS; bin++)
        mel_energies[bin] = 10.0f * log10f(mel_energies[bin]);

    for (int i = 0; i < num_mfcc_features; i++) {
        float sum = 0.0f;
        for (int j = 0; j < NUM_FBANK_BINS; j++) {
            sum += dct_matrix[i*NUM_FBANK_BINS+j] * mel_energies[j];
        }
        mfcc_out[i] = sum;
    }
}