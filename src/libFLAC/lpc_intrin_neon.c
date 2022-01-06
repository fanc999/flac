#include "private/cpu.h"

#ifndef FLAC__INTEGER_ONLY_LIBRARY
#ifndef FLAC__NO_ASM
#if defined FLAC__CPU_ARM64 && FLAC__HAS_NEONINTRIN
#include "private/lpc.h"
#include "FLAC/assert.h"
#include "FLAC/format.h"
#include "private/macros.h"
#include <arm_neon.h>

void FLAC__lpc_compute_autocorrelation_intrin_neon_lag_4(const FLAC__real data[], uint32_t data_len, uint32_t lag, FLAC__real autoc[]) {
    int i = 0;
    int limit = data_len - 4;
    
    // sum0 vector each of the lanes are used to store the lags. lane 0 is for lag 0, ... lane 3 is for lag 3
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    
    (void)lag;
    FLAC__ASSERT(lag <= 4);
    FLAC__ASSERT(lag <= data_len);
    
    // Processing all the data till data_len - lag (4)
    for ( ; i <= limit; i++) {
        float32x4_t d0 = vld1q_f32(data + i);
        
        // sum0 accumulates all 4 autocorelation lags: 0,..,3
        sum0 = vfmaq_laneq_f32(sum0, d0, d0, 0);
    }
    
    // Handling the last 4 data elements
    float32x4_t d0 = vdupq_n_f32(0.f);
    for ( ; i < (long)data_len; i++) {
        float32_t d = data[i];
        d0 = vextq_f32(d0, d0, 3);
        d0 = vsetq_lane_f32(d, d0, 0);
        sum0 = vfmaq_laneq_f32(sum0, d0, d0, 0);
    }
    
    // Storing the autocorelation results
    vst1q_f32(autoc, sum0);
}

void FLAC__lpc_compute_autocorrelation_intrin_neon_lag_8(const FLAC__real data[], uint32_t data_len, uint32_t lag, FLAC__real autoc[]) {
    int i;
    int limit = data_len - 8;
    
    // sum0 and sum1 vectors lanes (total 8) are used to store the lags. lane 0 is for lag 0, ... lane 7 is for lag 7
    float32x4_t sum0 = vdupq_n_f32(0.f);
    float32x4_t sum1 = vdupq_n_f32(0.f);
    const uint32x4_t vecMaskL0 = {0xffffffff, 0, 0, 0};

    (void)lag;
    FLAC__ASSERT(lag <= 8);
    FLAC__ASSERT(lag <= data_len);

    // Processing all the data till data_len - lag (8)
    for (i = 0; i <= limit; i++) {
        float32x4_t d0, d1;
        float32_t d;
        d0 = vld1q_f32(data + i);
        d1 = vld1q_f32(data + i + 4);
        d = vgetq_lane_f32(d0, 0);

        // sum0 and sum1 accumulate all 8 autocorelation lags: 0,..,7
        sum0 = vmlaq_n_f32(sum0, d0, d);
        sum1 = vmlaq_n_f32(sum1, d1, d);
    }

    // Handling the last 8 data elements
    float32x4_t d0 = vdupq_n_f32(0.f);
    float32x4_t d1 = vdupq_n_f32(0.f);
    for (; i < (long)data_len; i++) {
        float d = data[i];
        d1 = vextq_f32(d1, d1, 3);
        d0 = vextq_f32(d0, d0, 3);
        
        d1 = vbslq_f32(vecMaskL0, d0, d1);
        d0 = vsetq_lane_f32(d, d0, 0);
        
        sum1 = vmlaq_n_f32(sum1, d0, d);
        sum0 = vmlaq_n_f32(sum0, d1, d);
    }

    // Storing the autocorelation results.
    vst1q_f32(autoc, sum0);
    vst1q_f32(autoc + 4, sum1);
}

void FLAC__lpc_compute_autocorrelation_intrin_neon_lag_12(const FLAC__real data[], uint32_t data_len, uint32_t lag, FLAC__real autoc[]) {
    int i;
    int limit = data_len - 12;

    // sum0, sum1 and sum2 vectors lanes (total 12) are used to store the lags. lane 0 is for lag 0, ... lane 11 is for lag 11
    float32x4_t sum0 = vdupq_n_f32(0.f);
    float32x4_t sum1 = vdupq_n_f32(0.f);
    float32x4_t sum2 = vdupq_n_f32(0.f);
    const uint32x4_t vecMaskL0 = {0xffffffff, 0, 0, 0};

    (void)lag;
    FLAC__ASSERT(lag <= 12);
    FLAC__ASSERT(lag <= data_len);

    // Processing all the data till data_len - lag (12)
    for (i = 0; i <= limit; i++) {
        float32x4_t d0, d1, d2;
        float32_t d;

        d0 = vld1q_f32(data + i);
        d = vgetq_lane_f32(d0, 0);
        d1 = vld1q_f32(data + i + 4);
        d2 = vld1q_f32(data + i + 8);

        // sum0, sum1 and sum2 accumulate all 12 autocorelation lags: 0,..,11
        sum0 = vmlaq_n_f32(sum0, d0, d);
        sum1 = vmlaq_n_f32(sum1, d1, d);
        sum2 = vmlaq_n_f32(sum2, d2, d);
    }
    
    // Handling the last 12 data elements
    float32x4_t d0 = vdupq_n_f32(0.f);
    float32x4_t d1 = vdupq_n_f32(0.f);
    float32x4_t d2 = vdupq_n_f32(0.f);
    for (; i < (long)data_len; i++) {
        float d = data[i];
        d2 = vextq_f32(d2, d2, 3);
        d1 = vextq_f32(d1, d1, 3);
        d0 = vextq_f32(d0, d0, 3);
        
        d2 = vbslq_f32(vecMaskL0, d1, d2);
        d1 = vbslq_f32(vecMaskL0, d0, d1);
        d0 = vsetq_lane_f32(d, d0, 0);
        
        sum2 = vmlaq_n_f32(sum2, d2, d);
        sum1 = vmlaq_n_f32(sum1, d1, d);
        sum0 = vmlaq_n_f32(sum0, d0, d);
    }

    // Storing the autocorelation results
    vst1q_f32(autoc, sum0);
    vst1q_f32(autoc + 4, sum1);
    vst1q_f32(autoc + 8, sum2);
}

void FLAC__lpc_compute_autocorrelation_intrin_neon_lag_16(const FLAC__real data[], uint32_t data_len, uint32_t lag, FLAC__real autoc[]) {
    int i = 0;
    int j = 0;
    int limit = (data_len - 16);
    int limit4 = (limit >> 2) << 2;
    
    // The sumV 16 vectors are used in the 1st phase.
    // vector[i] stores 4 same lags of 4 adajacent data elements. The sum of its 4 lanes equals to the autocorrelation of that lag.
    float32x4_t sumV[16];
    
    // Next 4 vector are used in the 2nd phase.
    // sum0, sum1, sum3 and sum4 vectors lanes (total 16) are used to store the lags. lane 0 is for lag 0, ... lane 11 is for lag 16
    float32x4_t sumLanes[4];
    float32x4_t dataV[16];
    float32x4_t anchor;
	
    (void)lag;
    FLAC__ASSERT(lag <= 16);
    FLAC__ASSERT(lag <= data_len);

    // Next code is valid only in case data_len size is at least 16 + 3 (as we load last lag vector dataV[15])
    if (limit4 > 3)
    {
		// Epilogue: zeroing the 16 lag macc vectors + loading all 16 lags vectors.
        for (j = 0; j < 16; j++) {
            sumV[j] = vdupq_n_f32(0.f);
            dataV[j] = vld1q_f32(data + j);
        }
        // We start from 4 as the prolog of the loop sums all lags values.
        // We iterate one vector count elements (4) less as we avoid buffer over-read.
        for ( ;i < limit; i += 4) {
            anchor = dataV[0];
            // All first 12 lags can use previous loaded registers.
            for (j = 0; j < 12; j++) {
                sumV[j] = vfmaq_f32(sumV[j], anchor, dataV[j]);
                dataV[j] = dataV[j + 4];
            }
            // Last 4 lags are loaded.
            for (; j < 16; j++) {
                sumV[j] = vfmaq_f32(sumV[j], anchor, dataV[j]);
                dataV[j] = vld1q_f32(data + j + i);
            }
        }
		
        // Prologue: summing all the lags that were loaded.
        for (j = 0; j < 16; j++) {
            sumV[j] = vfmaq_f32(sumV[j], dataV[0], dataV[j]);
        }
        // Converting the 16 lag vectors to 4 vectors - each lane is the sum of the 4 lane per lag
        for (j = 0; j < 4; j++) {
            sumLanes[j] = vpaddq_f32(vpaddq_f32(sumV[j * 4], sumV[j * 4 + 1]), vpaddq_f32(sumV[j * 4 + 2], sumV[j * 4 + 3]));
        }
    }
    else {
        // Initializing the 4 sum vectors
		for (j = 0; j < 4; j++) {
            sumLanes[j] = vdupq_n_f32(0.f);
        }
    }
    
    // Next code iterates over the tail of the lags (same as the implementation of the lag4/8/12 above).
    // Notice that each k-th lag has k less accumlated elements than the 0 (first) lag.
    
    float32x4_t d0 = vdupq_n_f32(0.f);
    float32x4_t d1 = vdupq_n_f32(0.f);
    float32x4_t d2 = vdupq_n_f32(0.f);
    float32x4_t d3 = vdupq_n_f32(0.f);
    const uint32x4_t vecMaskL0 = {0xffffffff, 0, 0, 0};
    
    for (; i < (long)data_len; i++) {
		float d = data[i];
        d3 = vextq_f32(d3, d3, 3);
        d2 = vextq_f32(d2, d2, 3);
        d1 = vextq_f32(d1, d1, 3);
        d0 = vextq_f32(d0, d0, 3);
        
        d3 = vbslq_f32(vecMaskL0, d2, d3);
        d2 = vbslq_f32(vecMaskL0, d1, d2);
        d1 = vbslq_f32(vecMaskL0, d0, d1);
        d0 = vsetq_lane_f32(d, d0, 0);
        
        sumLanes[3] = vmlaq_n_f32(sumLanes[3], d3, d);
        sumLanes[2] = vmlaq_n_f32(sumLanes[2], d2, d);
        sumLanes[1] = vmlaq_n_f32(sumLanes[1], d1, d);
        sumLanes[0] = vmlaq_n_f32(sumLanes[0], d0, d);
    }
    
    // Storing the autocorelation results
    for (j = 0; j < 4; j++) {
        vst1q_f32(autoc + j * 4, sumLanes[j]);
    }
}



void FLAC__lpc_compute_residual_from_qlp_coefficients_intrin_neon(const FLAC__int32 *data, uint32_t data_len, const FLAC__int32 qlp_coeff[], uint32_t order, int lp_quantization, FLAC__int32 residual[])
{
    int i;
    FLAC__int32 sum;
    
    FLAC__ASSERT(order > 0);
    FLAC__ASSERT(order <= 32);

    
    // Using prologue reads is valid as encoder->private_->local_lpc_compute_residual_from_qlp_coefficients(signal+order,....)
    if(order <= 12) {
        if(order > 8) {
            if(order > 10) {
                if(order == 12) {
                    int32x4_t tData[12];
                    tData[0] = vld1q_s32(data - 12);
                    tData[1] = vld1q_s32(data - 11);
                    tData[2] = vld1q_s32(data - 10);
                    tData[3] = vld1q_s32(data - 9);
                    tData[4] = vld1q_s32(data - 8);
                    tData[5] = vld1q_s32(data - 7);
                    tData[6] = vld1q_s32(data - 6);
                    tData[7] = vld1q_s32(data - 5);
                    
                    for(i = 0; i < (int)data_len - 3; i+=4) {
                        int32x4_t  summ, mull;
                        
                        // Reading the 4 last data vectors.
                        tData[ 8] = vld1q_s32(data+i-4);
                        tData[ 9] = vld1q_s32(data+i-3);
                        tData[10] = vld1q_s32(data+i-2);
                        tData[11] = vld1q_s32(data+i-1);
                        
                        summ = vmulq_n_s32(tData[ 0],qlp_coeff[11]);
                        mull = vmulq_n_s32(tData[ 1],qlp_coeff[10]); summ += mull;
                        mull = vmulq_n_s32(tData[ 2],qlp_coeff[ 9]); summ += mull;
                        mull = vmulq_n_s32(tData[ 3],qlp_coeff[ 8]); summ += mull;
                        mull = vmulq_n_s32(tData[ 4],qlp_coeff[ 7]); summ += mull;
                        mull = vmulq_n_s32(tData[ 5],qlp_coeff[ 6]); summ += mull;
                        mull = vmulq_n_s32(tData[ 6],qlp_coeff[ 5]); summ += mull;
                        mull = vmulq_n_s32(tData[ 7],qlp_coeff[ 4]); summ += mull;
                        mull = vmulq_n_s32(tData[ 8],qlp_coeff[ 3]); summ += mull;
                        mull = vmulq_n_s32(tData[ 9],qlp_coeff[ 2]); summ += mull;
                        mull = vmulq_n_s32(tData[10],qlp_coeff[ 1]); summ += mull;
                        mull = vmulq_n_s32(tData[11],qlp_coeff[ 0]); summ += mull;
                        
                        summ >>= lp_quantization;
                        vst1q_s32(residual+i, vld1q_s32(data+i) - summ);
                        
                        // Moving the data registers while avoiding loads..
                        tData[0] = tData[4];
                        tData[1] = tData[5];
                        tData[2] = tData[6];
                        tData[3] = tData[7];
                        tData[4] = tData[8];
                        tData[5] = tData[9];
                        tData[6] = tData[10];
                        tData[7] = tData[11];
                    }
                }
                else { /* order == 11 */
                    int32x4_t tData[11];
                    tData[0] = vld1q_s32(data - 11);
                    tData[1] = vld1q_s32(data - 10);
                    tData[2] = vld1q_s32(data - 9);
                    tData[3] = vld1q_s32(data - 8);
                    tData[4] = vld1q_s32(data - 7);
                    tData[5] = vld1q_s32(data - 6);
                    tData[6] = vld1q_s32(data - 5);
                    
                    for(i = 0; i < (int)data_len - 3; i+=4) {
                        int32x4_t  summ, mull;
                        
                        // Reading the 4 last data vectors.
                        tData[ 7] = vld1q_s32(data+i-4);
                        tData[ 8] = vld1q_s32(data+i-3);
                        tData[ 9] = vld1q_s32(data+i-2);
                        tData[10] = vld1q_s32(data+i-1);
                        
                        summ = vmulq_n_s32(tData[ 0],qlp_coeff[10]);
                        mull = vmulq_n_s32(tData[ 1],qlp_coeff[ 9]); summ += mull;
                        mull = vmulq_n_s32(tData[ 2],qlp_coeff[ 8]); summ += mull;
                        mull = vmulq_n_s32(tData[ 3],qlp_coeff[ 7]); summ += mull;
                        mull = vmulq_n_s32(tData[ 4],qlp_coeff[ 6]); summ += mull;
                        mull = vmulq_n_s32(tData[ 5],qlp_coeff[ 5]); summ += mull;
                        mull = vmulq_n_s32(tData[ 6],qlp_coeff[ 4]); summ += mull;
                        mull = vmulq_n_s32(tData[ 7],qlp_coeff[ 3]); summ += mull;
                        mull = vmulq_n_s32(tData[ 8],qlp_coeff[ 2]); summ += mull;
                        mull = vmulq_n_s32(tData[ 9],qlp_coeff[ 1]); summ += mull;
                        mull = vmulq_n_s32(tData[10],qlp_coeff[ 0]); summ += mull;
                        
                        summ >>= lp_quantization;
                        vst1q_s32(residual+i, vld1q_s32(data+i) - summ);
                        
                        // Moving the data registers while avoiding loads..
                        tData[0] = tData[4];
                        tData[1] = tData[5];
                        tData[2] = tData[6];
                        tData[3] = tData[7];
                        tData[4] = tData[8];
                        tData[5] = tData[9];
                        tData[6] = tData[10];
                    }
                }
            }
            else {
                if(order == 10) {
                    int32x4_t tData[10];
                    tData[0] = vld1q_s32(data - 10);
                    tData[1] = vld1q_s32(data - 9);
                    tData[2] = vld1q_s32(data - 8);
                    tData[3] = vld1q_s32(data - 7);
                    tData[4] = vld1q_s32(data - 6);
                    tData[5] = vld1q_s32(data - 5);
                    
                    for(i = 0; i < (int)data_len - 3; i+=4) {
                        int32x4_t  summ, mull;
                        
                        // Reading the 4 last data vectors.
                        tData[ 6] = vld1q_s32(data+i-4);
                        tData[ 7] = vld1q_s32(data+i-3);
                        tData[ 8] = vld1q_s32(data+i-2);
                        tData[ 9] = vld1q_s32(data+i-1);
                        
                        summ = vmulq_n_s32(tData[0],qlp_coeff[9]);
                        mull = vmulq_n_s32(tData[1],qlp_coeff[8]); summ += mull;
                        mull = vmulq_n_s32(tData[2],qlp_coeff[7]); summ += mull;
                        mull = vmulq_n_s32(tData[3],qlp_coeff[6]); summ += mull;
                        mull = vmulq_n_s32(tData[4],qlp_coeff[5]); summ += mull;
                        mull = vmulq_n_s32(tData[5],qlp_coeff[4]); summ += mull;
                        mull = vmulq_n_s32(tData[6],qlp_coeff[3]); summ += mull;
                        mull = vmulq_n_s32(tData[7],qlp_coeff[2]); summ += mull;
                        mull = vmulq_n_s32(tData[8],qlp_coeff[1]); summ += mull;
                        mull = vmulq_n_s32(tData[9],qlp_coeff[0]); summ += mull;
                        
                        summ >>= lp_quantization;
                        vst1q_s32(residual+i, vld1q_s32(data+i) - summ);
                        
                        // Moving the data registers while avoiding loads..
                        tData[0] = tData[4];
                        tData[1] = tData[5];
                        tData[2] = tData[6];
                        tData[3] = tData[7];
                        tData[4] = tData[8];
                        tData[5] = tData[9];
                    }
                }
                else { /* order == 9 */
                    int32x4_t tData[9];
                    tData[0] = vld1q_s32(data - 9 );
                    tData[1] = vld1q_s32(data - 8 );
                    tData[2] = vld1q_s32(data - 7 );
                    tData[3] = vld1q_s32(data - 6);
                    tData[4] = vld1q_s32(data - 5);
                    
                    for(i = 0; i < (int)data_len - 3; i+=4) {
                        int32x4_t  summ, mull;
                        
                        // Reading the 4 last data vectors.
                        tData[ 5] = vld1q_s32(data+i-4);
                        tData[ 6] = vld1q_s32(data+i-3);
                        tData[ 7] = vld1q_s32(data+i-2);
                        tData[ 8] = vld1q_s32(data+i-1);
                        
                        summ = vmulq_n_s32(tData[0],qlp_coeff[8]);
                        mull = vmulq_n_s32(tData[1],qlp_coeff[7]); summ += mull;
                        mull = vmulq_n_s32(tData[2],qlp_coeff[6]); summ += mull;
                        mull = vmulq_n_s32(tData[3],qlp_coeff[5]); summ += mull;
                        mull = vmulq_n_s32(tData[4],qlp_coeff[4]); summ += mull;
                        mull = vmulq_n_s32(tData[5],qlp_coeff[3]); summ += mull;
                        mull = vmulq_n_s32(tData[6],qlp_coeff[2]); summ += mull;
                        mull = vmulq_n_s32(tData[7],qlp_coeff[1]); summ += mull;
                        mull = vmulq_n_s32(tData[8],qlp_coeff[0]); summ += mull;
                        
                        summ >>= lp_quantization;
                        vst1q_s32(residual+i, vld1q_s32(data+i) - summ);
                        
                        // Moving the data registers while avoiding loads..
                        tData[0] = tData[4];
                        tData[1] = tData[5];
                        tData[2] = tData[6];
                        tData[3] = tData[7];
                        tData[4] = tData[8];
                    }
                }
            }
        }
        else if(order > 4) {
            if(order > 6) {
                if(order == 8) {
            
                    for(i = 0; i < (int)data_len - 3; i+=4) {
                        int32x4_t  summ, mull;
                        summ = vmulq_n_s32(vld1q_s32(data+i-8),qlp_coeff[7]);
                        mull = vmulq_n_s32(vld1q_s32(data+i-7),qlp_coeff[6]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-6),qlp_coeff[5]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-5),qlp_coeff[4]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-4),qlp_coeff[3]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-3),qlp_coeff[2]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-2),qlp_coeff[1]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-1),qlp_coeff[0]); summ += mull;
                        summ >>= lp_quantization;
                        vst1q_s32(residual+i, vld1q_s32(data+i) - summ);
                    }
                }
                else { /* order == 7 */
            
                    for(i = 0; i < (int)data_len - 3; i+=4) {
                        int32x4_t  summ, mull;
                        summ = vmulq_n_s32(vld1q_s32(data+i-7),qlp_coeff[6]);
                        mull = vmulq_n_s32(vld1q_s32(data+i-6),qlp_coeff[5]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-5),qlp_coeff[4]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-4),qlp_coeff[3]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-3),qlp_coeff[2]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-2),qlp_coeff[1]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-1),qlp_coeff[0]); summ += mull;
                        summ >>= lp_quantization;
                        vst1q_s32(residual+i, vld1q_s32(data+i) - summ);
                    }
                }
            }
            else {
                if(order == 6) {
            
                    for(i = 0; i < (int)data_len - 3; i+=4) {
                        int32x4_t  summ, mull;
                        summ = vmulq_n_s32(vld1q_s32(data+i-6),qlp_coeff[5]);
                        mull = vmulq_n_s32(vld1q_s32(data+i-5),qlp_coeff[4]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-4),qlp_coeff[3]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-3),qlp_coeff[2]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-2),qlp_coeff[1]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-1),qlp_coeff[0]); summ += mull;
                        summ >>= lp_quantization;
                        vst1q_s32(residual+i, vld1q_s32(data+i) - summ);
                    }
                }
                else { /* order == 5 */
            
                    for(i = 0; i < (int)data_len - 3; i+=4) {
                        int32x4_t  summ, mull;
                        summ = vmulq_n_s32(vld1q_s32(data+i-5),qlp_coeff[4]);
                        mull = vmulq_n_s32(vld1q_s32(data+i-4),qlp_coeff[3]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-3),qlp_coeff[2]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-2),qlp_coeff[1]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-1),qlp_coeff[0]); summ += mull;
                        summ >>= lp_quantization;
                        vst1q_s32(residual+i, vld1q_s32(data+i) - summ);
                    }
                }
            }
        }
        else {
            if(order > 2) {
                if(order == 4) {
            
                    for(i = 0; i < (int)data_len - 3; i+=4) {
                        int32x4_t  summ, mull;
                        summ = vmulq_n_s32(vld1q_s32(data+i-4),qlp_coeff[3]);
                        mull = vmulq_n_s32(vld1q_s32(data+i-3),qlp_coeff[2]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-2),qlp_coeff[1]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-1),qlp_coeff[0]); summ += mull;
                        summ >>= lp_quantization;
                        vst1q_s32(residual+i, vld1q_s32(data+i) - summ);
                    }
                }
                else { /* order == 3 */
            
                    for(i = 0; i < (int)data_len - 3; i+=4) {
                        int32x4_t  summ, mull;
                        summ = vmulq_n_s32(vld1q_s32(data+i-3),qlp_coeff[2]);
                        mull = vmulq_n_s32(vld1q_s32(data+i-2),qlp_coeff[1]); summ += mull;
                        mull = vmulq_n_s32(vld1q_s32(data+i-1),qlp_coeff[0]); summ += mull;
                        summ >>= lp_quantization;
                        vst1q_s32(residual+i, vld1q_s32(data+i) - summ);
                    }
                }
            }
            else {
                if(order == 2) {
            
                    for(i = 0; i < (int)data_len - 3; i+=4) {
                        int32x4_t  summ, mull;
                        summ = vmulq_n_s32(vld1q_s32(data+i-2),qlp_coeff[1]);
                        mull = vmulq_n_s32(vld1q_s32(data+i-1),qlp_coeff[0]); summ += mull;
                        summ >>= lp_quantization;
                        vst1q_s32(residual+i, vld1q_s32(data+i) - summ);
                    }
                }
                else { /* order == 1 */
            
                    for(i = 0; i < (int)data_len - 3; i+=4) {
                        int32x4_t  summ;
                        summ = vmulq_n_s32(vld1q_s32(data+i-1),qlp_coeff[0]);
                        summ >>= lp_quantization;
                        vst1q_s32(residual+i, vld1q_s32(data+i) - summ);
                    }
                }
            }
        }
        for(; i < (int)data_len; i++) {
            sum = 0;
            switch(order) {
                case 12: sum += qlp_coeff[11] * data[i-12]; /* Falls through. */
                case 11: sum += qlp_coeff[10] * data[i-11]; /* Falls through. */
                case 10: sum += qlp_coeff[ 9] * data[i-10]; /* Falls through. */
                case 9:  sum += qlp_coeff[ 8] * data[i- 9]; /* Falls through. */
                case 8:  sum += qlp_coeff[ 7] * data[i- 8]; /* Falls through. */
                case 7:  sum += qlp_coeff[ 6] * data[i- 7]; /* Falls through. */
                case 6:  sum += qlp_coeff[ 5] * data[i- 6]; /* Falls through. */
                case 5:  sum += qlp_coeff[ 4] * data[i- 5]; /* Falls through. */
                case 4:  sum += qlp_coeff[ 3] * data[i- 4]; /* Falls through. */
                case 3:  sum += qlp_coeff[ 2] * data[i- 3]; /* Falls through. */
                case 2:  sum += qlp_coeff[ 1] * data[i- 2]; /* Falls through. */
                case 1:  sum += qlp_coeff[ 0] * data[i- 1];
            }
            residual[i] = data[i] - (sum >> lp_quantization);
        }
    }
    else { /* order > 12 */
        for(i = 0; i < (int)data_len; i++) {
            sum = 0;
            switch(order) {
                case 32: sum += qlp_coeff[31] * data[i-32]; /* Falls through. */
                case 31: sum += qlp_coeff[30] * data[i-31]; /* Falls through. */
                case 30: sum += qlp_coeff[29] * data[i-30]; /* Falls through. */
                case 29: sum += qlp_coeff[28] * data[i-29]; /* Falls through. */
                case 28: sum += qlp_coeff[27] * data[i-28]; /* Falls through. */
                case 27: sum += qlp_coeff[26] * data[i-27]; /* Falls through. */
                case 26: sum += qlp_coeff[25] * data[i-26]; /* Falls through. */
                case 25: sum += qlp_coeff[24] * data[i-25]; /* Falls through. */
                case 24: sum += qlp_coeff[23] * data[i-24]; /* Falls through. */
                case 23: sum += qlp_coeff[22] * data[i-23]; /* Falls through. */
                case 22: sum += qlp_coeff[21] * data[i-22]; /* Falls through. */
                case 21: sum += qlp_coeff[20] * data[i-21]; /* Falls through. */
                case 20: sum += qlp_coeff[19] * data[i-20]; /* Falls through. */
                case 19: sum += qlp_coeff[18] * data[i-19]; /* Falls through. */
                case 18: sum += qlp_coeff[17] * data[i-18]; /* Falls through. */
                case 17: sum += qlp_coeff[16] * data[i-17]; /* Falls through. */
                case 16: sum += qlp_coeff[15] * data[i-16]; /* Falls through. */
                case 15: sum += qlp_coeff[14] * data[i-15]; /* Falls through. */
                case 14: sum += qlp_coeff[13] * data[i-14]; /* Falls through. */
                case 13: sum += qlp_coeff[12] * data[i-13];
                         sum += qlp_coeff[11] * data[i-12];
                         sum += qlp_coeff[10] * data[i-11];
                         sum += qlp_coeff[ 9] * data[i-10];
                         sum += qlp_coeff[ 8] * data[i- 9];
                         sum += qlp_coeff[ 7] * data[i- 8];
                         sum += qlp_coeff[ 6] * data[i- 7];
                         sum += qlp_coeff[ 5] * data[i- 6];
                         sum += qlp_coeff[ 4] * data[i- 5];
                         sum += qlp_coeff[ 3] * data[i- 4];
                         sum += qlp_coeff[ 2] * data[i- 3];
                         sum += qlp_coeff[ 1] * data[i- 2];
                         sum += qlp_coeff[ 0] * data[i- 1];
            }
            residual[i] = data[i] - (sum >> lp_quantization);
        }
    }
}



void FLAC__lpc_compute_residual_from_qlp_coefficients_wide_intrin_neon(const FLAC__int32 *data, uint32_t data_len, const FLAC__int32 qlp_coeff[], uint32_t order, int lp_quantization, FLAC__int32 residual[]) {
	int i;
	FLAC__int64 sum;
	
	FLAC__ASSERT(order > 0);
	FLAC__ASSERT(order <= 32);

    // Using prologue reads is valid as encoder->private_->local_lpc_compute_residual_from_qlp_coefficients_64bit(signal+order,....)
	if(order <= 12) {
		if(order > 8) {
			if(order > 10) {
				if(order == 12) {
                    int32x2_t tData[12];
                    tData[0] = vld1_s32(data-12);
                    tData[1] = vld1_s32(data-11);
                    tData[2] = vld1_s32(data-10);
                    tData[3] = vld1_s32(data-9);
                    tData[4] = vld1_s32(data-8);
                    tData[5] = vld1_s32(data-7);
                    tData[6] = vld1_s32(data-6);
                    tData[7] = vld1_s32(data-5);
                    tData[8] = vld1_s32(data-4);
                    tData[9] = vld1_s32(data-3);
                    
					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t  summ, mull;
						int32x2_t summ32;
                        
                        // Reading the 2 last data vectors.
                        tData[10] = vld1_s32(data+i-2);
                        tData[11] = vld1_s32(data+i-1);
                        
                        summ = vmull_n_s32(tData[ 0],qlp_coeff[11]);
						mull = vmull_n_s32(tData[ 1],qlp_coeff[10]);    summ += mull;
						mull = vmull_n_s32(tData[ 2],qlp_coeff[ 9]);    summ += mull;
						mull = vmull_n_s32(tData[ 3],qlp_coeff[ 8]);    summ += mull;
						mull = vmull_n_s32(tData[ 4],qlp_coeff[ 7]);    summ += mull;
						mull = vmull_n_s32(tData[ 5],qlp_coeff[ 6]);    summ += mull;
						mull = vmull_n_s32(tData[ 6],qlp_coeff[ 5]);    summ += mull;
						mull = vmull_n_s32(tData[ 7],qlp_coeff[ 4]);    summ += mull;
						mull = vmull_n_s32(tData[ 8],qlp_coeff[ 3]);    summ += mull;
						mull = vmull_n_s32(tData[ 9],qlp_coeff[ 2]);    summ += mull;
						mull = vmull_n_s32(tData[10],qlp_coeff[ 1]);    summ += mull;
						mull = vmull_n_s32(tData[11],qlp_coeff[ 0]);    summ += mull;
						
                        summ32 = vqmovn_s64(summ >> lp_quantization);
						vst1_s32(residual+i, vld1_s32(data+i) - summ32);
                        
                        // Moving the data registers while avoiding loads..
                        tData[0] = tData[2];
                        tData[1] = tData[3];
                        tData[2] = tData[4];
                        tData[3] = tData[5];
                        tData[4] = tData[6];
                        tData[5] = tData[7];
                        tData[6] = tData[8];
                        tData[7] = tData[9];
                        tData[8] = tData[10];
                        tData[9] = tData[11];                
					}
				}
				else { /* order == 11 */
			
                    int32x2_t tData[11];
                    tData[0] = vld1_s32(data-11);
                    tData[1] = vld1_s32(data-10);
                    tData[2] = vld1_s32(data-9);
                    tData[3] = vld1_s32(data-8);
                    tData[4] = vld1_s32(data-7);
                    tData[5] = vld1_s32(data-6);
                    tData[6] = vld1_s32(data-5);
                    tData[7] = vld1_s32(data-4);
                    tData[8] = vld1_s32(data-3);
                    
					for(i = 0; i < (int)data_len-1; i+=2) {
                        int64x2_t  summ, mull;
                        int32x2_t summ32;
                        
                        // Reading the 2 last data vectors.
                        tData[9] = vld1_s32(data+i-2);
                        tData[10] = vld1_s32(data+i-1);
                        
                        summ = vmull_n_s32(tData[ 0],qlp_coeff[10]);
                        mull = vmull_n_s32(tData[ 1],qlp_coeff[ 9]);    summ += mull;
                        mull = vmull_n_s32(tData[ 2],qlp_coeff[ 8]);    summ += mull;
                        mull = vmull_n_s32(tData[ 3],qlp_coeff[ 7]);    summ += mull;
                        mull = vmull_n_s32(tData[ 4],qlp_coeff[ 6]);    summ += mull;
                        mull = vmull_n_s32(tData[ 5],qlp_coeff[ 5]);    summ += mull;
                        mull = vmull_n_s32(tData[ 6],qlp_coeff[ 4]);    summ += mull;
                        mull = vmull_n_s32(tData[ 7],qlp_coeff[ 3]);    summ += mull;
                        mull = vmull_n_s32(tData[ 8],qlp_coeff[ 2]);    summ += mull;
                        mull = vmull_n_s32(tData[ 9],qlp_coeff[ 1]);    summ += mull;
                        mull = vmull_n_s32(tData[10],qlp_coeff[ 0]);    summ += mull;
                        
                        summ32 = vqmovn_s64(summ >> lp_quantization);
                        vst1_s32(residual+i, vld1_s32(data+i) - summ32);
                        
                        // Moving the data registers while avoiding loads..
                        tData[0] = tData[2];
                        tData[1] = tData[3];
                        tData[2] = tData[4];
                        tData[3] = tData[5];
                        tData[4] = tData[6];
                        tData[5] = tData[7];
                        tData[6] = tData[8];
                        tData[7] = tData[9];
                        tData[8] = tData[10];
					}
				}
			}
			else {
				if(order == 10) {
                    int32x2_t tData[10];
                    tData[0] = vld1_s32(data-10);
                    tData[1] = vld1_s32(data-9);
                    tData[2] = vld1_s32(data-8);
                    tData[3] = vld1_s32(data-7);
                    tData[4] = vld1_s32(data-6);
                    tData[5] = vld1_s32(data-5);
                    tData[6] = vld1_s32(data-4);
                    tData[7] = vld1_s32(data-3);
                    
                    for(i = 0; i < (int)data_len-1; i+=2) {
                        int64x2_t  summ, mull;
                        int32x2_t summ32;
                        
                        // Reading the 2 last data vectors.
                        tData[8] = vld1_s32(data+i-2);
                        tData[9] = vld1_s32(data+i-1);
                        
                        summ = vmull_n_s32(tData[ 0],qlp_coeff[9]);
                        mull = vmull_n_s32(tData[ 1],qlp_coeff[8]);    summ += mull;
                        mull = vmull_n_s32(tData[ 2],qlp_coeff[7]);    summ += mull;
                        mull = vmull_n_s32(tData[ 3],qlp_coeff[6]);    summ += mull;
                        mull = vmull_n_s32(tData[ 4],qlp_coeff[5]);    summ += mull;
                        mull = vmull_n_s32(tData[ 5],qlp_coeff[4]);    summ += mull;
                        mull = vmull_n_s32(tData[ 6],qlp_coeff[3]);    summ += mull;
                        mull = vmull_n_s32(tData[ 7],qlp_coeff[2]);    summ += mull;
                        mull = vmull_n_s32(tData[ 8],qlp_coeff[1]);    summ += mull;
                        mull = vmull_n_s32(tData[ 9],qlp_coeff[0]);    summ += mull;
                        
                        summ32 = vqmovn_s64(summ >> lp_quantization);
                        vst1_s32(residual+i, vld1_s32(data+i) - summ32);
                        
                        // Moving the data registers while avoiding loads..
                        tData[0] = tData[2];
                        tData[1] = tData[3];
                        tData[2] = tData[4];
                        tData[3] = tData[5];
                        tData[4] = tData[6];
                        tData[5] = tData[7];
                        tData[6] = tData[8];
                        tData[7] = tData[9];
					}
				}
				else { /* order == 9 */
                    int32x2_t tData[9];
                    tData[0] = vld1_s32(data-9);
                    tData[1] = vld1_s32(data-8);
                    tData[2] = vld1_s32(data-7);
                    tData[3] = vld1_s32(data-6);
                    tData[4] = vld1_s32(data-5);
                    tData[5] = vld1_s32(data-4);
                    tData[6] = vld1_s32(data-3);
                    
                    for(i = 0; i < (int)data_len-1; i+=2) {
                        int64x2_t  summ, mull;
                        int32x2_t summ32;
                        
                        // Reading the 2 last data vectors.
                        tData[7] = vld1_s32(data+i-2);
                        tData[8] = vld1_s32(data+i-1);
                        
                        summ = vmull_n_s32(tData[0],qlp_coeff[8]);
                        mull = vmull_n_s32(tData[1],qlp_coeff[7]);    summ += mull;
                        mull = vmull_n_s32(tData[2],qlp_coeff[6]);    summ += mull;
                        mull = vmull_n_s32(tData[3],qlp_coeff[5]);    summ += mull;
                        mull = vmull_n_s32(tData[4],qlp_coeff[4]);    summ += mull;
                        mull = vmull_n_s32(tData[5],qlp_coeff[3]);    summ += mull;
                        mull = vmull_n_s32(tData[6],qlp_coeff[2]);    summ += mull;
                        mull = vmull_n_s32(tData[7],qlp_coeff[1]);    summ += mull;
                        mull = vmull_n_s32(tData[8],qlp_coeff[0]);    summ += mull;
                        
                        summ32 = vqmovn_s64(summ >> lp_quantization);
                        vst1_s32(residual+i, vld1_s32(data+i) - summ32);
                        
                        // Moving the data registers while avoiding loads..
                        tData[0] = tData[2];
                        tData[1] = tData[3];
                        tData[2] = tData[4];
                        tData[3] = tData[5];
                        tData[4] = tData[6];
                        tData[5] = tData[7];
                        tData[6] = tData[8];
					}
				}
			}
		}
		else if(order > 4) {
			if(order > 6) {
				if(order == 8) {
			
					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t  summ, mull;
						int32x2_t summ32;
						summ = vmull_n_s32(vld1_s32(data+i-8),qlp_coeff[7]);
						mull = vmull_n_s32(vld1_s32(data+i-7),qlp_coeff[6]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-6),qlp_coeff[5]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-5),qlp_coeff[4]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-4),qlp_coeff[3]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-3),qlp_coeff[2]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-2),qlp_coeff[1]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-1),qlp_coeff[0]); summ += mull;
						
						summ32 = vqmovn_s64(summ >> lp_quantization);
						vst1_s32(residual+i, vld1_s32(data+i) - summ32);
					}
				}
				else { /* order == 7 */
			
					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t  summ, mull;
						int32x2_t summ32;
						summ = vmull_n_s32(vld1_s32(data+i-7),qlp_coeff[6]);
						mull = vmull_n_s32(vld1_s32(data+i-6),qlp_coeff[5]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-5),qlp_coeff[4]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-4),qlp_coeff[3]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-3),qlp_coeff[2]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-2),qlp_coeff[1]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-1),qlp_coeff[0]); summ += mull;
						
						summ32 = vqmovn_s64(summ >> lp_quantization);
						vst1_s32(residual+i, vld1_s32(data+i) - summ32);
					}
				}
			}
			else {
				if(order == 6) {
			
					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t  summ, mull;
						int32x2_t summ32;
						summ = vmull_n_s32(vld1_s32(data+i-6),qlp_coeff[5]);
						mull = vmull_n_s32(vld1_s32(data+i-5),qlp_coeff[4]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-4),qlp_coeff[3]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-3),qlp_coeff[2]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-2),qlp_coeff[1]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-1),qlp_coeff[0]); summ += mull;
						
						summ32 = vqmovn_s64(summ >> lp_quantization);
						vst1_s32(residual+i, vld1_s32(data+i) - summ32);
					}
				}
				else { /* order == 5 */
			
					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t  summ, mull;
						int32x2_t summ32;
						summ = vmull_n_s32(vld1_s32(data+i-5),qlp_coeff[4]);
						mull = vmull_n_s32(vld1_s32(data+i-4),qlp_coeff[3]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-3),qlp_coeff[2]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-2),qlp_coeff[1]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-1),qlp_coeff[0]); summ += mull;
						
						summ32 = vqmovn_s64(summ >> lp_quantization);
						vst1_s32(residual+i, vld1_s32(data+i) - summ32);
					}
				}
			}
		}
		else {
			if(order > 2) {
				if(order == 4) {
			
					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t  summ, mull;
						int32x2_t summ32;
						summ = vmull_n_s32(vld1_s32(data+i-4),qlp_coeff[3]);
						mull = vmull_n_s32(vld1_s32(data+i-3),qlp_coeff[2]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-2),qlp_coeff[1]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-1),qlp_coeff[0]); summ += mull;
						
						summ32 = vqmovn_s64(summ >> lp_quantization);
						vst1_s32(residual+i, vld1_s32(data+i) - summ32);
					}
				}
				else { /* order == 3 */
			
					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t  summ, mull;
						int32x2_t summ32;
						summ = vmull_n_s32(vld1_s32(data+i-3),qlp_coeff[2]);
						mull = vmull_n_s32(vld1_s32(data+i-2),qlp_coeff[1]); summ += mull;
						mull = vmull_n_s32(vld1_s32(data+i-1),qlp_coeff[0]); summ += mull;
						
						summ32 = vqmovn_s64(summ >> lp_quantization);
						vst1_s32(residual+i, vld1_s32(data+i) - summ32);
					}
				}
			}
			else {
				if(order == 2) {
			
					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t  summ, mull;
						int32x2_t summ32;
						summ = vmull_n_s32(vld1_s32(data+i-2),qlp_coeff[1]);
						mull = vmull_n_s32(vld1_s32(data+i-1),qlp_coeff[0]); summ += mull;
						
						summ32 = vqmovn_s64(summ >> lp_quantization);
						vst1_s32(residual+i, vld1_s32(data+i) - summ32);
					}
				}
				else { /* order == 1 */
			
					for(i = 0; i < (int)data_len-1; i+=2) {
						int64x2_t  summ;
						int32x2_t summ32;
						summ = vmull_n_s32(vld1_s32(data+i-1),qlp_coeff[0]);
						
						summ32 = vqmovn_s64(summ >> lp_quantization);
						vst1_s32(residual+i, vld1_s32(data+i) - summ32);
					}
				}
			}
		}
		for(; i < (int)data_len; i++) {
			sum = 0;
			switch(order) {
				case 12: sum += qlp_coeff[11] * data[i-12]; /* Falls through. */
				case 11: sum += qlp_coeff[10] * data[i-11]; /* Falls through. */
				case 10: sum += qlp_coeff[ 9] * data[i-10]; /* Falls through. */
				case 9:  sum += qlp_coeff[ 8] * data[i- 9]; /* Falls through. */
				case 8:  sum += qlp_coeff[ 7] * data[i- 8]; /* Falls through. */
				case 7:  sum += qlp_coeff[ 6] * data[i- 7]; /* Falls through. */
				case 6:  sum += qlp_coeff[ 5] * data[i- 6]; /* Falls through. */
				case 5:  sum += qlp_coeff[ 4] * data[i- 5]; /* Falls through. */
				case 4:  sum += qlp_coeff[ 3] * data[i- 4]; /* Falls through. */
				case 3:  sum += qlp_coeff[ 2] * data[i- 3]; /* Falls through. */
				case 2:  sum += qlp_coeff[ 1] * data[i- 2]; /* Falls through. */
				case 1:  sum += qlp_coeff[ 0] * data[i- 1];
			}
			residual[i] = data[i] - (sum >> lp_quantization);
		}
	}
	else { /* order > 12 */
		for(i = 0; i < (int)data_len; i++) {
			sum = 0;
			switch(order) {
				case 32: sum += qlp_coeff[31] * data[i-32]; /* Falls through. */
				case 31: sum += qlp_coeff[30] * data[i-31]; /* Falls through. */
				case 30: sum += qlp_coeff[29] * data[i-30]; /* Falls through. */
				case 29: sum += qlp_coeff[28] * data[i-29]; /* Falls through. */
				case 28: sum += qlp_coeff[27] * data[i-28]; /* Falls through. */
				case 27: sum += qlp_coeff[26] * data[i-27]; /* Falls through. */
				case 26: sum += qlp_coeff[25] * data[i-26]; /* Falls through. */
				case 25: sum += qlp_coeff[24] * data[i-25]; /* Falls through. */
				case 24: sum += qlp_coeff[23] * data[i-24]; /* Falls through. */
				case 23: sum += qlp_coeff[22] * data[i-23]; /* Falls through. */
				case 22: sum += qlp_coeff[21] * data[i-22]; /* Falls through. */
				case 21: sum += qlp_coeff[20] * data[i-21]; /* Falls through. */
				case 20: sum += qlp_coeff[19] * data[i-20]; /* Falls through. */
				case 19: sum += qlp_coeff[18] * data[i-19]; /* Falls through. */
				case 18: sum += qlp_coeff[17] * data[i-18]; /* Falls through. */
				case 17: sum += qlp_coeff[16] * data[i-17]; /* Falls through. */
				case 16: sum += qlp_coeff[15] * data[i-16]; /* Falls through. */
				case 15: sum += qlp_coeff[14] * data[i-15]; /* Falls through. */
				case 14: sum += qlp_coeff[13] * data[i-14]; /* Falls through. */
				case 13: sum += qlp_coeff[12] * data[i-13];
				         sum += qlp_coeff[11] * data[i-12];
				         sum += qlp_coeff[10] * data[i-11];
				         sum += qlp_coeff[ 9] * data[i-10];
				         sum += qlp_coeff[ 8] * data[i- 9];
				         sum += qlp_coeff[ 7] * data[i- 8];
				         sum += qlp_coeff[ 6] * data[i- 7];
				         sum += qlp_coeff[ 5] * data[i- 6];
				         sum += qlp_coeff[ 4] * data[i- 5];
				         sum += qlp_coeff[ 3] * data[i- 4];
				         sum += qlp_coeff[ 2] * data[i- 3];
				         sum += qlp_coeff[ 1] * data[i- 2];
				         sum += qlp_coeff[ 0] * data[i- 1];
			}
			residual[i] = data[i] - (sum >> lp_quantization);
		}
	}
    
    return;
}



#endif /* FLAC__CPU_ARM64 && FLAC__HAS_ARCH64INTRIN */
#endif /* FLAC__NO_ASM */
#endif /* FLAC__INTEGER_ONLY_LIBRARY */
