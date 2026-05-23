/*
 * Copyright (C) 2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MFCC_H
#define MFCC_H

#define EIDSP_USE_CMSIS_DSP 1
#define EIDSP_LOAD_CMSIS_DSP_SOURCES 1


#include "Particle.h"
#include <stdint.h>
#include <math.h>

#undef A0
#undef A1
#undef A2

#ifndef __STATIC_FORCEINLINE
#define __STATIC_FORCEINLINE static __attribute__((always_inline)) inline
#endif
#ifndef __STATIC_INLINE
#define __STATIC_INLINE static inline
#endif

#include "edge-impulse-sdk/CMSIS/DSP/Include/arm_math.h"

#define NUM_FBANK_BINS 26
#define MEL_LOW_FREQ 20
#define MEL_HIGH_FREQ 4000
#define SAMP_FREQ 16000

class MFCC {
public:
    MFCC(int num_mfcc_features, int frame_len, int mfcc_dec_bits);
    ~MFCC();
    void mfcc_compute(const int16_t *audio_data, float* mfcc_out);

private:
    int num_mfcc_features;
    int frame_len;
    int mfcc_dec_bits;
    int frame_len_padded;
    float *frame;
    float *buffer;
    float *mel_energies;
    float *window_func;
    int32_t *fbank_filter_first;
    int32_t *fbank_filter_last;
    float **mel_fbank;
    float *dct_matrix;

    arm_rfft_fast_instance_f32 fft_instance;
    float *fft_output;

    float *create_dct_matrix(int32_t input_length, int32_t coefficient_count);
    float **create_mel_fbank();
    inline float MelScale(float freq) {
        return 1127.0f * logf(1.0f + freq / 700.0f);
    }
};

#endif