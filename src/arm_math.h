#ifndef ARM_MATH_WRAPPER_H
#define ARM_MATH_WRAPPER_H

// Tving CMSIS-DSP aktiveret før config.hpp læses
#define EIDSP_USE_CMSIS_DSP 1
#define EIDSP_LOAD_CMSIS_DSP_SOURCES 1

#ifndef __STATIC_FORCEINLINE
#define __STATIC_FORCEINLINE static __attribute__((always_inline)) inline
#endif
#ifndef __STATIC_INLINE
#define __STATIC_INLINE static inline
#endif

#include "edge-impulse-sdk/CMSIS/DSP/Include/arm_math.h"

#endif // ARM_MATH_WRAPPER_H