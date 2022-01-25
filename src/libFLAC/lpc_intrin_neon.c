#include "private/cpu.h"

#ifndef FLAC__INTEGER_ONLY_LIBRARY
#ifndef FLAC__NO_ASM
#if defined FLAC__CPU_ARM64 && FLAC__HAS_NEONINTRIN
#include "private/lpc.h"
#include "FLAC/assert.h"
#include "FLAC/format.h"
#include "private/macros.h"

/*
 * Sadly, the Visual Studio 2017 headers do not allow us to include arm_neon.h for ARM64
 * builds, so we must include arm64_neon.h instead.  Interestingly, Visual Studio 2019
 * and later allow us to include arm_neon.h for ARM64 builds, which will in turn include
 * arm64_neon.h
 */
#if defined (_MSC_VER) && (_MSC_VER < 1920)
# include <arm64_neon.h>
#else
# include <arm_neon.h>
#endif

/*
 * The Visual Studio headers may not have vmlaq_laneq_s32()/vmulq_laneq_s32() defined-
 * This is copied from Visual Studio headers that do have them defined
 */
#ifdef _MSC_VER
# ifndef vmlaq_laneq_s32
#  define vmlaq_laneq_s32(src1, src2, src3, lane) neon_mlaqvind32q(src1, src2, src3, lane)
# endif
# ifndef vmulq_laneq_s32
#  define vmulq_laneq_s32(src1, src2, lane) neon_mulqvind32q(src1, src2, lane)
# endif
#endif

#if defined (_MSC_VER) && !defined (__clang__)
  /*
   * The .n128_u64 field is first.  Combine pairs of 32-bit integers in little-endian order.
   * (we need to avoid a GCCism on Visual Studio)
   */
# define INIT_uint32x4_t(w,x,y,z) { .n128_u32 = {w, x, y, z} }
# define INIT_int32x4_t(w,x,y,z) { .n128_i32 = {w, x, y, z} }
#else
# define INIT_uint32x4_t(w,x,y,z) { (w), (x), (y), (z) }
# define INIT_int32x4_t(w,x,y,z) { (w), (x), (y), (z) }
#endif

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
    const uint32x4_t vecMaskL0 = INIT_uint32x4_t(0xffffffff, 0, 0, 0);

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
    const uint32x4_t vecMaskL0 = INIT_uint32x4_t(0xffffffff, 0, 0, 0);

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
    const uint32x4_t vecMaskL0 = INIT_uint32x4_t(0xffffffff, 0, 0, 0);
    
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



#define MUL_32_BIT_LOOP_UNROOL_3(qlp_coeff_vec, lane) \
                        summ_0 = vmulq_laneq_s32(tmp_vec[0], qlp_coeff_vec, lane); \
                        summ_1 = vmulq_laneq_s32(tmp_vec[4], qlp_coeff_vec, lane); \
                        summ_2 = vmulq_laneq_s32(tmp_vec[8], qlp_coeff_vec, lane); 
                        

#define MACC_32BIT_LOOP_UNROOL_3(tmp_vec_ind, qlp_coeff_vec, lane) \
                        summ_0 = vmlaq_laneq_s32(summ_0,tmp_vec[tmp_vec_ind] ,qlp_coeff_vec, lane); \
                        summ_1 = vmlaq_laneq_s32(summ_1,tmp_vec[tmp_vec_ind+4] ,qlp_coeff_vec, lane); \
                        summ_2 = vmlaq_laneq_s32(summ_2,tmp_vec[tmp_vec_ind+8] ,qlp_coeff_vec, lane);
                        
void FLAC__lpc_compute_residual_from_qlp_coefficients_intrin_neon(const FLAC__int32 *data, uint32_t data_len, const FLAC__int32 qlp_coeff[], uint32_t order, int lp_quantization, FLAC__int32 residual[])
{
    int i;
    FLAC__int32 sum;
    FLAC__ASSERT(order > 0);
    FLAC__ASSERT(order <= 32);

    int32x4_t tmp_vec[20];

    // Using prologue reads is valid as encoder->private_->local_lpc_compute_residual_from_qlp_coefficients(signal+order,....)
    if(order <= 12) {
        if(order > 8) {
            if(order > 10) {
                if (order == 12) {
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4], qlp_coeff[5], qlp_coeff[6], qlp_coeff[7]);
                    int32x4_t qlp_coeff_2 = INIT_int32x4_t(qlp_coeff[8], qlp_coeff[9], qlp_coeff[10], qlp_coeff[11]);

                    tmp_vec[0] = vld1q_s32(data - 12);
                    tmp_vec[1] = vld1q_s32(data - 11);
                    tmp_vec[2] = vld1q_s32(data - 10);
                    tmp_vec[3] = vld1q_s32(data - 9);
                    tmp_vec[4] = vld1q_s32(data - 8);
                    tmp_vec[5] = vld1q_s32(data - 7);
                    tmp_vec[6] = vld1q_s32(data - 6);
                    tmp_vec[7] = vld1q_s32(data - 5);

                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int32x4_t summ_0, summ_1, summ_2;

                        tmp_vec[8] = vld1q_s32(data + i - 4);
                        tmp_vec[9] = vld1q_s32(data+i-3);
                        tmp_vec[10] = vld1q_s32(data+i-2);
                        tmp_vec[11] = vld1q_s32(data+i-1);
                        tmp_vec[12] = vld1q_s32(data+i);
                        tmp_vec[13] = vld1q_s32(data+i+1);
                        tmp_vec[14] = vld1q_s32(data+i+2);
                        tmp_vec[15] = vld1q_s32(data+i+3);
                        tmp_vec[16] = vld1q_s32(data + i + 4);
                        tmp_vec[17] = vld1q_s32(data + i + 5);
                        tmp_vec[18] = vld1q_s32(data + i + 6);
                        tmp_vec[19] = vld1q_s32(data + i + 7);

                        MUL_32_BIT_LOOP_UNROOL_3(qlp_coeff_2, 3)
                        MACC_32BIT_LOOP_UNROOL_3(1, qlp_coeff_2, 2)
                        MACC_32BIT_LOOP_UNROOL_3(2, qlp_coeff_2, 1)
                        MACC_32BIT_LOOP_UNROOL_3(3, qlp_coeff_2, 0)
                        MACC_32BIT_LOOP_UNROOL_3(4, qlp_coeff_1, 3)
                        MACC_32BIT_LOOP_UNROOL_3(5, qlp_coeff_1, 2)
                        MACC_32BIT_LOOP_UNROOL_3(6, qlp_coeff_1, 1)
                        MACC_32BIT_LOOP_UNROOL_3(7, qlp_coeff_1, 0)
                        MACC_32BIT_LOOP_UNROOL_3(8, qlp_coeff_0, 3)
                        MACC_32BIT_LOOP_UNROOL_3(9, qlp_coeff_0, 2)
                        MACC_32BIT_LOOP_UNROOL_3(10, qlp_coeff_0, 1)
                        MACC_32BIT_LOOP_UNROOL_3(11, qlp_coeff_0, 0)

                        vst1q_s32(residual+i + 0, vsubq_s32(vld1q_s32(data+i + 0) , vshlq_s32(summ_0,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 4, vsubq_s32(vld1q_s32(data+i + 4) , vshlq_s32(summ_1,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 8, vsubq_s32(vld1q_s32(data+i + 8) , vshlq_s32(summ_2,vdupq_n_s32(-lp_quantization))));

                        tmp_vec[0] = tmp_vec[12];
                        tmp_vec[1] = tmp_vec[13];
                        tmp_vec[2] = tmp_vec[14];
                        tmp_vec[3] = tmp_vec[15];
                        tmp_vec[4] = tmp_vec[16];
                        tmp_vec[5] = tmp_vec[17];
                        tmp_vec[6] = tmp_vec[18];
                        tmp_vec[7] = tmp_vec[19];
                    }
                }

                else { /* order == 11 */
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4], qlp_coeff[5], qlp_coeff[6], qlp_coeff[7]);
                    int32x4_t qlp_coeff_2 = INIT_int32x4_t(qlp_coeff[8], qlp_coeff[9], qlp_coeff[10], 0);

                    tmp_vec[0] = vld1q_s32(data - 11);
                    tmp_vec[1] = vld1q_s32(data - 10);
                    tmp_vec[2] = vld1q_s32(data - 9);
                    tmp_vec[3] = vld1q_s32(data - 8);
                    tmp_vec[4] = vld1q_s32(data - 7);
                    tmp_vec[5] = vld1q_s32(data - 6);
                    tmp_vec[6] = vld1q_s32(data - 5);
                    
                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int32x4_t summ_0, summ_1, summ_2;
                        tmp_vec[7] = vld1q_s32(data + i - 4);
                        tmp_vec[8] = vld1q_s32(data + i - 3);
                        tmp_vec[9] = vld1q_s32(data + i - 2);
                        tmp_vec[10] = vld1q_s32(data + i - 1);
                        tmp_vec[11] = vld1q_s32(data + i - 0);
                        tmp_vec[12] = vld1q_s32(data + i + 1);
                        tmp_vec[13] = vld1q_s32(data + i + 2);
                        tmp_vec[14] = vld1q_s32(data + i + 3);
                        tmp_vec[15] = vld1q_s32(data + i + 4);
                        tmp_vec[16] = vld1q_s32(data + i + 5);
                        tmp_vec[17] = vld1q_s32(data + i + 6);
                        tmp_vec[18] = vld1q_s32(data + i + 7);
                        
                      
                        MUL_32_BIT_LOOP_UNROOL_3(qlp_coeff_2, 2)
                        MACC_32BIT_LOOP_UNROOL_3(1, qlp_coeff_2, 1)
                        MACC_32BIT_LOOP_UNROOL_3(2, qlp_coeff_2, 0)
                        MACC_32BIT_LOOP_UNROOL_3(3, qlp_coeff_1, 3)
                        MACC_32BIT_LOOP_UNROOL_3(4, qlp_coeff_1, 2)
                        MACC_32BIT_LOOP_UNROOL_3(5, qlp_coeff_1, 1)
                        MACC_32BIT_LOOP_UNROOL_3(6, qlp_coeff_1, 0)
                        MACC_32BIT_LOOP_UNROOL_3(7, qlp_coeff_0, 3)
                        MACC_32BIT_LOOP_UNROOL_3(8, qlp_coeff_0, 2)
                        MACC_32BIT_LOOP_UNROOL_3(9, qlp_coeff_0, 1)
                        MACC_32BIT_LOOP_UNROOL_3(10, qlp_coeff_0, 0)
                        
                        vst1q_s32(residual+i + 0, vsubq_s32(vld1q_s32(data+i + 0) , vshlq_s32(summ_0,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 4, vsubq_s32(vld1q_s32(data+i + 4) , vshlq_s32(summ_1,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 8, vsubq_s32(vld1q_s32(data+i + 8) , vshlq_s32(summ_2,vdupq_n_s32(-lp_quantization))));

                        
                        tmp_vec[0] = tmp_vec[12];
                        tmp_vec[1] = tmp_vec[13];
                        tmp_vec[2] = tmp_vec[14];
                        tmp_vec[3] = tmp_vec[15];
                        tmp_vec[4] = tmp_vec[16];
                        tmp_vec[5] = tmp_vec[17];
                        tmp_vec[6] = tmp_vec[18];
                    }
                }
            }
            else {
                if(order == 10) {
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4], qlp_coeff[5], qlp_coeff[6], qlp_coeff[7]);
                    int32x4_t qlp_coeff_2 = INIT_int32x4_t(qlp_coeff[8], qlp_coeff[9], 0, 0);

                    tmp_vec[0] = vld1q_s32(data - 10);
                    tmp_vec[1] = vld1q_s32(data - 9);
                    tmp_vec[2] = vld1q_s32(data - 8);
                    tmp_vec[3] = vld1q_s32(data - 7);
                    tmp_vec[4] = vld1q_s32(data - 6);
                    tmp_vec[5] = vld1q_s32(data - 5);
                    
                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int32x4_t summ_0, summ_1, summ_2;
                        tmp_vec[6] = vld1q_s32(data + i - 4);
                        tmp_vec[7] = vld1q_s32(data + i - 3);
                        tmp_vec[8] = vld1q_s32(data + i - 2);
                        tmp_vec[9] = vld1q_s32(data + i - 1);
                        tmp_vec[10] = vld1q_s32(data + i - 0);
                        tmp_vec[11] = vld1q_s32(data + i + 1);
                        tmp_vec[12] = vld1q_s32(data + i + 2);
                        tmp_vec[13] = vld1q_s32(data + i + 3);
                        tmp_vec[14] = vld1q_s32(data + i + 4);
                        tmp_vec[15] = vld1q_s32(data + i + 5);
                        tmp_vec[16] = vld1q_s32(data + i + 6);
                        tmp_vec[17] = vld1q_s32(data + i + 7);
                        
                            
                        MUL_32_BIT_LOOP_UNROOL_3(qlp_coeff_2, 1)
                        MACC_32BIT_LOOP_UNROOL_3(1, qlp_coeff_2, 0)
                        MACC_32BIT_LOOP_UNROOL_3(2, qlp_coeff_1, 3)
                        MACC_32BIT_LOOP_UNROOL_3(3, qlp_coeff_1, 2)
                        MACC_32BIT_LOOP_UNROOL_3(4, qlp_coeff_1, 1)
                        MACC_32BIT_LOOP_UNROOL_3(5, qlp_coeff_1, 0)
                        MACC_32BIT_LOOP_UNROOL_3(6, qlp_coeff_0, 3)
                        MACC_32BIT_LOOP_UNROOL_3(7, qlp_coeff_0, 2)
                        MACC_32BIT_LOOP_UNROOL_3(8, qlp_coeff_0, 1)
                        MACC_32BIT_LOOP_UNROOL_3(9, qlp_coeff_0, 0)
                        
                        vst1q_s32(residual+i + 0, vsubq_s32(vld1q_s32(data+i + 0) , vshlq_s32(summ_0,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 4, vsubq_s32(vld1q_s32(data+i + 4) , vshlq_s32(summ_1,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 8, vsubq_s32(vld1q_s32(data+i + 8) , vshlq_s32(summ_2,vdupq_n_s32(-lp_quantization))));

                        
                        tmp_vec[0] = tmp_vec[12];
                        tmp_vec[1] = tmp_vec[13];
                        tmp_vec[2] = tmp_vec[14];
                        tmp_vec[3] = tmp_vec[15];
                        tmp_vec[4] = tmp_vec[16];
                        tmp_vec[5] = tmp_vec[17];
                    }
                }
                else { /* order == 9 */
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4], qlp_coeff[5], qlp_coeff[6], qlp_coeff[7]);
                    int32x4_t qlp_coeff_2 = INIT_int32x4_t(qlp_coeff[8], 0, 0, 0);

                    tmp_vec[0] = vld1q_s32(data - 9);
                    tmp_vec[1] = vld1q_s32(data - 8);
                    tmp_vec[2] = vld1q_s32(data - 7);
                    tmp_vec[3] = vld1q_s32(data - 6);
                    tmp_vec[4] = vld1q_s32(data - 5);
                    
                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int32x4_t summ_0, summ_1, summ_2;
                        tmp_vec[5] = vld1q_s32(data + i - 4);
                        tmp_vec[6] = vld1q_s32(data + i - 3);
                        tmp_vec[7] = vld1q_s32(data + i - 2);
                        tmp_vec[8] = vld1q_s32(data + i - 1);
                        tmp_vec[9] = vld1q_s32(data + i - 0);
                        tmp_vec[10] = vld1q_s32(data + i + 1);
                        tmp_vec[11] = vld1q_s32(data + i + 2);
                        tmp_vec[12] = vld1q_s32(data + i + 3);
                        tmp_vec[13] = vld1q_s32(data + i + 4);
                        tmp_vec[14] = vld1q_s32(data + i + 5);
                        tmp_vec[15] = vld1q_s32(data + i + 6);
                        tmp_vec[16] = vld1q_s32(data + i + 7);
                        
                        MUL_32_BIT_LOOP_UNROOL_3(qlp_coeff_2, 0)
                        MACC_32BIT_LOOP_UNROOL_3(1, qlp_coeff_1, 3)
                        MACC_32BIT_LOOP_UNROOL_3(2, qlp_coeff_1, 2)
                        MACC_32BIT_LOOP_UNROOL_3(3, qlp_coeff_1, 1)
                        MACC_32BIT_LOOP_UNROOL_3(4, qlp_coeff_1, 0)
                        MACC_32BIT_LOOP_UNROOL_3(5, qlp_coeff_0, 3)
                        MACC_32BIT_LOOP_UNROOL_3(6, qlp_coeff_0, 2)
                        MACC_32BIT_LOOP_UNROOL_3(7, qlp_coeff_0, 1)
                        MACC_32BIT_LOOP_UNROOL_3(8, qlp_coeff_0, 0)

                        vst1q_s32(residual+i + 0, vsubq_s32(vld1q_s32(data+i + 0) , vshlq_s32(summ_0,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 4, vsubq_s32(vld1q_s32(data+i + 4) , vshlq_s32(summ_1,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 8, vsubq_s32(vld1q_s32(data+i + 8) , vshlq_s32(summ_2,vdupq_n_s32(-lp_quantization))));

                        tmp_vec[0] = tmp_vec[12];
                        tmp_vec[1] = tmp_vec[13];
                        tmp_vec[2] = tmp_vec[14];
                        tmp_vec[3] = tmp_vec[15];
                        tmp_vec[4] = tmp_vec[16];
                    }
                }
            }
        }
        else if(order > 4) {
            if(order > 6) {
                if(order == 8) {
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4], qlp_coeff[5], qlp_coeff[6], qlp_coeff[7]);

                    tmp_vec[0] = vld1q_s32(data - 8);
                    tmp_vec[1] = vld1q_s32(data - 7);
                    tmp_vec[2] = vld1q_s32(data - 6);
                    tmp_vec[3] = vld1q_s32(data - 5);
                    
                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int32x4_t summ_0, summ_1, summ_2;
                        tmp_vec[4] = vld1q_s32(data + i - 4);
                        tmp_vec[5] = vld1q_s32(data + i - 3);
                        tmp_vec[6] = vld1q_s32(data + i - 2);
                        tmp_vec[7] = vld1q_s32(data + i - 1);
                        tmp_vec[8] = vld1q_s32(data + i - 0);
                        tmp_vec[9] = vld1q_s32(data + i + 1);
                        tmp_vec[10] = vld1q_s32(data + i + 2);
                        tmp_vec[11] = vld1q_s32(data + i + 3);
                        tmp_vec[12] = vld1q_s32(data + i + 4);
                        tmp_vec[13] = vld1q_s32(data + i + 5);
                        tmp_vec[14] = vld1q_s32(data + i + 6);
                        tmp_vec[15] = vld1q_s32(data + i + 7);
                        
                        MUL_32_BIT_LOOP_UNROOL_3(qlp_coeff_1, 3)
                        MACC_32BIT_LOOP_UNROOL_3(1, qlp_coeff_1, 2)
                        MACC_32BIT_LOOP_UNROOL_3(2, qlp_coeff_1, 1)
                        MACC_32BIT_LOOP_UNROOL_3(3, qlp_coeff_1, 0)
                        MACC_32BIT_LOOP_UNROOL_3(4, qlp_coeff_0, 3)
                        MACC_32BIT_LOOP_UNROOL_3(5, qlp_coeff_0, 2)
                        MACC_32BIT_LOOP_UNROOL_3(6, qlp_coeff_0, 1)
                        MACC_32BIT_LOOP_UNROOL_3(7, qlp_coeff_0, 0)

                        vst1q_s32(residual+i + 0, vsubq_s32(vld1q_s32(data+i + 0) , vshlq_s32(summ_0,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 4, vsubq_s32(vld1q_s32(data+i + 4) , vshlq_s32(summ_1,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 8, vsubq_s32(vld1q_s32(data+i + 8) , vshlq_s32(summ_2,vdupq_n_s32(-lp_quantization))));

                        tmp_vec[0] = tmp_vec[12];
                        tmp_vec[1] = tmp_vec[13];
                        tmp_vec[2] = tmp_vec[14];
                        tmp_vec[3] = tmp_vec[15];
                    }
                }
                else { /* order == 7 */
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4], qlp_coeff[5], qlp_coeff[6], 0);

                    tmp_vec[0] = vld1q_s32(data - 7);
                    tmp_vec[1] = vld1q_s32(data - 6);
                    tmp_vec[2] = vld1q_s32(data - 5);
                    
                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int32x4_t summ_0, summ_1, summ_2;
                        tmp_vec[3] = vld1q_s32(data + i - 4);
                        tmp_vec[4] = vld1q_s32(data + i - 3);
                        tmp_vec[5] = vld1q_s32(data + i - 2);
                        tmp_vec[6] = vld1q_s32(data + i - 1);
                        tmp_vec[7] = vld1q_s32(data + i - 0);
                        tmp_vec[8] = vld1q_s32(data + i + 1);
                        tmp_vec[9] = vld1q_s32(data + i + 2);
                        tmp_vec[10] = vld1q_s32(data + i + 3);
                        tmp_vec[11] = vld1q_s32(data + i + 4);
                        tmp_vec[12] = vld1q_s32(data + i + 5);
                        tmp_vec[13] = vld1q_s32(data + i + 6);
                        tmp_vec[14] = vld1q_s32(data + i + 7);
                        
                        MUL_32_BIT_LOOP_UNROOL_3(qlp_coeff_1, 2)
                        MACC_32BIT_LOOP_UNROOL_3(1, qlp_coeff_1, 1)
                        MACC_32BIT_LOOP_UNROOL_3(2, qlp_coeff_1, 0)
                        MACC_32BIT_LOOP_UNROOL_3(3, qlp_coeff_0, 3)
                        MACC_32BIT_LOOP_UNROOL_3(4, qlp_coeff_0, 2)
                        MACC_32BIT_LOOP_UNROOL_3(5, qlp_coeff_0, 1)
                        MACC_32BIT_LOOP_UNROOL_3(6, qlp_coeff_0, 0)
                        
                        vst1q_s32(residual+i + 0, vsubq_s32(vld1q_s32(data+i + 0) , vshlq_s32(summ_0,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 4, vsubq_s32(vld1q_s32(data+i + 4) , vshlq_s32(summ_1,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 8, vsubq_s32(vld1q_s32(data+i + 8) , vshlq_s32(summ_2,vdupq_n_s32(-lp_quantization))));

                        tmp_vec[0] = tmp_vec[12];
                        tmp_vec[1] = tmp_vec[13];
                        tmp_vec[2] = tmp_vec[14];
                    }
                }
            }
            else {
                if(order == 6) {
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4], qlp_coeff[5], 0, 0);

                    tmp_vec[0] = vld1q_s32(data - 6);
                    tmp_vec[1] = vld1q_s32(data - 5);
                    
                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int32x4_t summ_0, summ_1, summ_2;
                        tmp_vec[2] = vld1q_s32(data + i - 4);
                        tmp_vec[3] = vld1q_s32(data + i - 3);
                        tmp_vec[4] = vld1q_s32(data + i - 2);
                        tmp_vec[5] = vld1q_s32(data + i - 1);
                        tmp_vec[6] = vld1q_s32(data + i - 0);
                        tmp_vec[7] = vld1q_s32(data + i + 1);
                        tmp_vec[8] = vld1q_s32(data + i + 2);
                        tmp_vec[9] = vld1q_s32(data + i + 3);
                        tmp_vec[10] = vld1q_s32(data + i + 4);
                        tmp_vec[11] = vld1q_s32(data + i + 5);
                        tmp_vec[12] = vld1q_s32(data + i + 6);
                        tmp_vec[13] = vld1q_s32(data + i + 7);
                        
                        MUL_32_BIT_LOOP_UNROOL_3(qlp_coeff_1, 1)
                        MACC_32BIT_LOOP_UNROOL_3(1, qlp_coeff_1, 0)
                        MACC_32BIT_LOOP_UNROOL_3(2, qlp_coeff_0, 3)
                        MACC_32BIT_LOOP_UNROOL_3(3, qlp_coeff_0, 2)
                        MACC_32BIT_LOOP_UNROOL_3(4, qlp_coeff_0, 1)
                        MACC_32BIT_LOOP_UNROOL_3(5, qlp_coeff_0, 0)
                        
                        vst1q_s32(residual+i + 0, vsubq_s32(vld1q_s32(data+i + 0) , vshlq_s32(summ_0,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 4, vsubq_s32(vld1q_s32(data+i + 4) , vshlq_s32(summ_1,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 8, vsubq_s32(vld1q_s32(data+i + 8) , vshlq_s32(summ_2,vdupq_n_s32(-lp_quantization))));

                        tmp_vec[0] = tmp_vec[12];
                        tmp_vec[1] = tmp_vec[13];
                    }
                }
                else { /* order == 5 */
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4], 0, 0, 0);

                    tmp_vec[0] = vld1q_s32(data - 5);
                    
                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int32x4_t summ_0, summ_1, summ_2;

                        tmp_vec[1] = vld1q_s32(data + i - 4);
                        tmp_vec[2] = vld1q_s32(data + i - 3);
                        tmp_vec[3] = vld1q_s32(data + i - 2);
                        tmp_vec[4] = vld1q_s32(data + i - 1);
                        tmp_vec[5] = vld1q_s32(data + i - 0);
                        tmp_vec[6] = vld1q_s32(data + i + 1);
                        tmp_vec[7] = vld1q_s32(data + i + 2);
                        tmp_vec[8] = vld1q_s32(data + i + 3);
                        tmp_vec[9] = vld1q_s32(data + i + 4);
                        tmp_vec[10] = vld1q_s32(data + i + 5);
                        tmp_vec[11] = vld1q_s32(data + i + 6);
                        tmp_vec[12] = vld1q_s32(data + i + 7);
                        
                        MUL_32_BIT_LOOP_UNROOL_3(qlp_coeff_1, 0)
                        MACC_32BIT_LOOP_UNROOL_3(1, qlp_coeff_0, 3)
                        MACC_32BIT_LOOP_UNROOL_3(2, qlp_coeff_0, 2)
                        MACC_32BIT_LOOP_UNROOL_3(3, qlp_coeff_0, 1)
                        MACC_32BIT_LOOP_UNROOL_3(4, qlp_coeff_0, 0)
                        
                        vst1q_s32(residual+i + 0, vsubq_s32(vld1q_s32(data+i + 0) , vshlq_s32(summ_0,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 4, vsubq_s32(vld1q_s32(data+i + 4) , vshlq_s32(summ_1,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 8, vsubq_s32(vld1q_s32(data+i + 8) , vshlq_s32(summ_2,vdupq_n_s32(-lp_quantization))));

                        tmp_vec[0] = tmp_vec[12];
                    }
                }
            }
        }
        else {
            if(order > 2) {
                if(order == 4) {
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    
                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int32x4_t summ_0, summ_1, summ_2;
                        tmp_vec[0] = vld1q_s32(data + i - 4);
                        tmp_vec[1] = vld1q_s32(data + i - 3);
                        tmp_vec[2] = vld1q_s32(data + i - 2);
                        tmp_vec[3] = vld1q_s32(data + i - 1);
                        tmp_vec[4] = vld1q_s32(data + i - 0);
                        tmp_vec[5] = vld1q_s32(data + i + 1);
                        tmp_vec[6] = vld1q_s32(data + i + 2);
                        tmp_vec[7] = vld1q_s32(data + i + 3);
                        tmp_vec[8] = vld1q_s32(data + i + 4);
                        tmp_vec[9] = vld1q_s32(data + i + 5);
                        tmp_vec[10] = vld1q_s32(data + i + 6);
                        tmp_vec[11] = vld1q_s32(data + i + 7);
                    
                        MUL_32_BIT_LOOP_UNROOL_3(qlp_coeff_0, 3)
                        MACC_32BIT_LOOP_UNROOL_3(1, qlp_coeff_0, 2)
                        MACC_32BIT_LOOP_UNROOL_3(2, qlp_coeff_0, 1)
                        MACC_32BIT_LOOP_UNROOL_3(3, qlp_coeff_0, 0)
                        
                        vst1q_s32(residual+i + 0, vsubq_s32(vld1q_s32(data+i + 0) , vshlq_s32(summ_0,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 4, vsubq_s32(vld1q_s32(data+i + 4) , vshlq_s32(summ_1,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 8, vsubq_s32(vld1q_s32(data+i + 8) , vshlq_s32(summ_2,vdupq_n_s32(-lp_quantization))));
                    }
                }
                else { /* order == 3 */
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], 0);

                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int32x4_t summ_0, summ_1, summ_2;
                        tmp_vec[0] = vld1q_s32(data + i - 3);
                        tmp_vec[1] = vld1q_s32(data + i - 2);
                        tmp_vec[2] = vld1q_s32(data + i - 1);
                        tmp_vec[4] = vld1q_s32(data + i + 1);
                        tmp_vec[5] = vld1q_s32(data + i + 2);
                        tmp_vec[6] = vld1q_s32(data + i + 3);
                        tmp_vec[8] = vld1q_s32(data + i + 5);
                        tmp_vec[9] = vld1q_s32(data + i + 6);
                        tmp_vec[10] = vld1q_s32(data + i + 7);
                        
                        MUL_32_BIT_LOOP_UNROOL_3(qlp_coeff_0, 2)
                        MACC_32BIT_LOOP_UNROOL_3(1, qlp_coeff_0, 1)
                        MACC_32BIT_LOOP_UNROOL_3(2, qlp_coeff_0, 0)

                        vst1q_s32(residual+i + 0, vsubq_s32(vld1q_s32(data+i + 0) , vshlq_s32(summ_0,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 4, vsubq_s32(vld1q_s32(data+i + 4) , vshlq_s32(summ_1,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 8, vsubq_s32(vld1q_s32(data+i + 8) , vshlq_s32(summ_2,vdupq_n_s32(-lp_quantization))));
                    }
                }
            }
            else {
                if(order == 2) {
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], 0, 0);

                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int32x4_t summ_0, summ_1, summ_2;
                        tmp_vec[0] = vld1q_s32(data + i - 2);
                        tmp_vec[1] = vld1q_s32(data + i - 1);
                        tmp_vec[4] = vld1q_s32(data + i + 2);
                        tmp_vec[5] = vld1q_s32(data + i + 3);
                        tmp_vec[8] = vld1q_s32(data + i + 6);
                        tmp_vec[9] = vld1q_s32(data + i + 7);
                        
                        MUL_32_BIT_LOOP_UNROOL_3(qlp_coeff_0, 1)
                        MACC_32BIT_LOOP_UNROOL_3(1, qlp_coeff_0, 0)
                        
                        vst1q_s32(residual+i + 0, vsubq_s32(vld1q_s32(data+i + 0) , vshlq_s32(summ_0,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 4, vsubq_s32(vld1q_s32(data+i + 4) , vshlq_s32(summ_1,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 8, vsubq_s32(vld1q_s32(data+i + 8) , vshlq_s32(summ_2,vdupq_n_s32(-lp_quantization))));
                    }
                }
                else { /* order == 1 */
                    int32x4_t qlp_coeff_0 = vdupq_n_s32(qlp_coeff[0]);

                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int32x4_t summ_0, summ_1, summ_2;
                        tmp_vec[0] = vld1q_s32(data + i - 1);
                        tmp_vec[4] = vld1q_s32(data + i + 3);
                        tmp_vec[8] = vld1q_s32(data + i + 7);
                        
                        summ_0 = vmulq_s32(tmp_vec[0], qlp_coeff_0);
                        summ_1 = vmulq_s32(tmp_vec[4], qlp_coeff_0);
                        summ_2 = vmulq_s32(tmp_vec[8], qlp_coeff_0);

                        vst1q_s32(residual+i + 0, vsubq_s32(vld1q_s32(data+i + 0) , vshlq_s32(summ_0,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 4, vsubq_s32(vld1q_s32(data+i + 4) , vshlq_s32(summ_1,vdupq_n_s32(-lp_quantization))));
                        vst1q_s32(residual+i + 8, vsubq_s32(vld1q_s32(data+i + 8) , vshlq_s32(summ_2,vdupq_n_s32(-lp_quantization))));
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



#define MUL_64_BIT_LOOP_UNROOL_3(qlp_coeff_vec, lane) \
                        summ_l_0 = vmull_laneq_s32(vget_low_s32(tmp_vec[0]),qlp_coeff_vec, lane); \
                        summ_h_0 = vmull_high_laneq_s32(tmp_vec[0], qlp_coeff_vec, lane);\
                        summ_l_1 = vmull_laneq_s32(vget_low_s32(tmp_vec[4]),qlp_coeff_vec, lane); \
                        summ_h_1 = vmull_high_laneq_s32(tmp_vec[4], qlp_coeff_vec, lane);\
                        summ_l_2 = vmull_laneq_s32(vget_low_s32(tmp_vec[8]),qlp_coeff_vec, lane);\
                        summ_h_2 = vmull_high_laneq_s32(tmp_vec[8], qlp_coeff_vec, lane);


#define MACC_64_BIT_LOOP_UNROOL_3(tmp_vec_ind, qlp_coeff_vec, lane) \
                        summ_l_0 = vmlal_laneq_s32(summ_l_0,vget_low_s32(tmp_vec[tmp_vec_ind]),qlp_coeff_vec, lane); \
                        summ_h_0 = vmlal_high_laneq_s32(summ_h_0, tmp_vec[tmp_vec_ind], qlp_coeff_vec, lane); \
                        summ_l_1 = vmlal_laneq_s32(summ_l_1, vget_low_s32(tmp_vec[tmp_vec_ind+4]),qlp_coeff_vec, lane); \
                        summ_h_1 = vmlal_high_laneq_s32(summ_h_1, tmp_vec[tmp_vec_ind+4], qlp_coeff_vec, lane); \
                        summ_l_2 = vmlal_laneq_s32(summ_l_2, vget_low_s32(tmp_vec[tmp_vec_ind+8]),qlp_coeff_vec, lane);\
                        summ_h_2 = vmlal_high_laneq_s32(summ_h_2,tmp_vec[tmp_vec_ind+8], qlp_coeff_vec, lane);

#define SHIFT_SUMS_64BITS_AND_STORE_SUB() \
                        res0 = vuzp1q_s32(vreinterpretq_s32_s64(vshlq_s64(summ_l_0,lp_quantization_vec)), vreinterpretq_s32_s64(vshlq_s64(summ_h_0,lp_quantization_vec))); \
                        res1 = vuzp1q_s32(vreinterpretq_s32_s64(vshlq_s64(summ_l_1,lp_quantization_vec)), vreinterpretq_s32_s64(vshlq_s64(summ_h_1,lp_quantization_vec))); \
                        res2 = vuzp1q_s32(vreinterpretq_s32_s64(vshlq_s64(summ_l_2,lp_quantization_vec)), vreinterpretq_s32_s64(vshlq_s64(summ_h_2,lp_quantization_vec))); \
                        vst1q_s32(residual+i+0, vsubq_s32(vld1q_s32(data+i+0), res0));\
                        vst1q_s32(residual+i+4, vsubq_s32(vld1q_s32(data+i+4), res1));\
                        vst1q_s32(residual+i+8, vsubq_s32(vld1q_s32(data+i+8), res2));

void FLAC__lpc_compute_residual_from_qlp_coefficients_wide_intrin_neon(const FLAC__int32 *data, uint32_t data_len, const FLAC__int32 qlp_coeff[], uint32_t order, int lp_quantization, FLAC__int32 residual[]) {
	int i;
	FLAC__int64 sum;
	
    int32x4_t tmp_vec[20];
    int32x4_t res0, res1, res2;
    int64x2_t  lp_quantization_vec = vdupq_n_s64(-lp_quantization);

    FLAC__ASSERT(order > 0);
	FLAC__ASSERT(order <= 32);
    
    // Using prologue reads is valid as encoder->private_->local_lpc_compute_residual_from_qlp_coefficients_64bit(signal+order,....)
	if(order <= 12) {
		if(order > 8) {
			if(order > 10) {
				if(order == 12) {
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4],qlp_coeff[5],qlp_coeff[6],qlp_coeff[7]);
                    int32x4_t qlp_coeff_2 = INIT_int32x4_t(qlp_coeff[8],qlp_coeff[9],qlp_coeff[10],qlp_coeff[11]);

                    tmp_vec[0] = vld1q_s32(data - 12);
                    tmp_vec[1] = vld1q_s32(data - 11);
                    tmp_vec[2] = vld1q_s32(data - 10);
                    tmp_vec[3] = vld1q_s32(data - 9);
                    tmp_vec[4] = vld1q_s32(data - 8);
                    tmp_vec[5] = vld1q_s32(data - 7);
                    tmp_vec[6] = vld1q_s32(data - 6);
                    tmp_vec[7] = vld1q_s32(data - 5);

                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int64x2_t  summ_l_0, summ_h_0, summ_l_1, summ_h_1, summ_l_2, summ_h_2;
                        
                        tmp_vec[8] = vld1q_s32(data+i-4);
                        tmp_vec[9] = vld1q_s32(data+i-3);
                        tmp_vec[10] = vld1q_s32(data+i-2);
                        tmp_vec[11] = vld1q_s32(data+i-1);
                        tmp_vec[12] = vld1q_s32(data+i);
                        tmp_vec[13] = vld1q_s32(data+i+1);
                        tmp_vec[14] = vld1q_s32(data+i+2);
                        tmp_vec[15] = vld1q_s32(data+i+3);
                        tmp_vec[16] = vld1q_s32(data + i + 4);
                        tmp_vec[17] = vld1q_s32(data + i + 5);
                        tmp_vec[18] = vld1q_s32(data + i + 6);
                        tmp_vec[19] = vld1q_s32(data + i + 7);

                        MUL_64_BIT_LOOP_UNROOL_3(qlp_coeff_2, 3)
                        MACC_64_BIT_LOOP_UNROOL_3(1, qlp_coeff_2, 2) 
                        MACC_64_BIT_LOOP_UNROOL_3(2, qlp_coeff_2, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(3, qlp_coeff_2, 0) 
                        MACC_64_BIT_LOOP_UNROOL_3(4, qlp_coeff_1, 3) 
                        MACC_64_BIT_LOOP_UNROOL_3(5, qlp_coeff_1, 2) 
                        MACC_64_BIT_LOOP_UNROOL_3(6, qlp_coeff_1, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(7, qlp_coeff_1, 0) 
                        MACC_64_BIT_LOOP_UNROOL_3(8, qlp_coeff_0, 3) 
                        MACC_64_BIT_LOOP_UNROOL_3(9, qlp_coeff_0, 2) 
                        MACC_64_BIT_LOOP_UNROOL_3(10,qlp_coeff_0, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(11,qlp_coeff_0, 0) 

                        SHIFT_SUMS_64BITS_AND_STORE_SUB()
                        
                        tmp_vec[0] = tmp_vec[12];
                        tmp_vec[1] = tmp_vec[13];
                        tmp_vec[2] = tmp_vec[14];
                        tmp_vec[3] = tmp_vec[15];
                        tmp_vec[4] = tmp_vec[16];
                        tmp_vec[5] = tmp_vec[17];
                        tmp_vec[6] = tmp_vec[18];
                        tmp_vec[7] = tmp_vec[19];
                    }
                }
				else { /* order == 11 */			
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4],qlp_coeff[5],qlp_coeff[6],qlp_coeff[7]);
                    int32x4_t qlp_coeff_2 = INIT_int32x4_t(qlp_coeff[8],qlp_coeff[9],qlp_coeff[10],0);

                    tmp_vec[0] = vld1q_s32(data - 11);
                    tmp_vec[1] = vld1q_s32(data - 10);
                    tmp_vec[2] = vld1q_s32(data - 9);
                    tmp_vec[3] = vld1q_s32(data - 8);
                    tmp_vec[4] = vld1q_s32(data - 7);
                    tmp_vec[5] = vld1q_s32(data - 6);
                    tmp_vec[6] = vld1q_s32(data - 5);

                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int64x2_t  summ_l_0, summ_h_0, summ_l_1, summ_h_1, summ_l_2, summ_h_2;
                        
                        tmp_vec[7] = vld1q_s32(data+i-4);
                        tmp_vec[8] = vld1q_s32(data+i-3);
                        tmp_vec[9] = vld1q_s32(data+i-2);
                        tmp_vec[10] = vld1q_s32(data+i-1);
                        tmp_vec[11] = vld1q_s32(data+i);
                        tmp_vec[12] = vld1q_s32(data+i+1);
                        tmp_vec[13] = vld1q_s32(data+i+2);
                        tmp_vec[14] = vld1q_s32(data+i+3);
                        tmp_vec[15] = vld1q_s32(data + i + 4);
                        tmp_vec[16] = vld1q_s32(data + i + 5);
                        tmp_vec[17] = vld1q_s32(data + i + 6);
                        tmp_vec[18] = vld1q_s32(data + i + 7);

                        MUL_64_BIT_LOOP_UNROOL_3(qlp_coeff_2, 2)
                        MACC_64_BIT_LOOP_UNROOL_3(1, qlp_coeff_2, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(2, qlp_coeff_2, 0) 
                        MACC_64_BIT_LOOP_UNROOL_3(3, qlp_coeff_1, 3) 
                        MACC_64_BIT_LOOP_UNROOL_3(4, qlp_coeff_1, 2) 
                        MACC_64_BIT_LOOP_UNROOL_3(5, qlp_coeff_1, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(6, qlp_coeff_1, 0) 
                        MACC_64_BIT_LOOP_UNROOL_3(7, qlp_coeff_0, 3) 
                        MACC_64_BIT_LOOP_UNROOL_3(8, qlp_coeff_0, 2) 
                        MACC_64_BIT_LOOP_UNROOL_3(9, qlp_coeff_0, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(10,qlp_coeff_0, 0) 

                        SHIFT_SUMS_64BITS_AND_STORE_SUB()
                        
                        tmp_vec[0] = tmp_vec[12];
                        tmp_vec[1] = tmp_vec[13];
                        tmp_vec[2] = tmp_vec[14];
                        tmp_vec[3] = tmp_vec[15];
                        tmp_vec[4] = tmp_vec[16];
                        tmp_vec[5] = tmp_vec[17];
                        tmp_vec[6] = tmp_vec[18];
                    }
                }
            }
            else
            {
                if (order == 10) {
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4], qlp_coeff[5], qlp_coeff[6], qlp_coeff[7]);
                    int32x4_t qlp_coeff_2 = INIT_int32x4_t(qlp_coeff[8], qlp_coeff[9], 0, 0);

                    tmp_vec[0] = vld1q_s32(data - 10);
                    tmp_vec[1] = vld1q_s32(data - 9);
                    tmp_vec[2] = vld1q_s32(data - 8);
                    tmp_vec[3] = vld1q_s32(data - 7);
                    tmp_vec[4] = vld1q_s32(data - 6);
                    tmp_vec[5] = vld1q_s32(data - 5);
                    

                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int64x2_t summ_l_0, summ_h_0, summ_l_1, summ_h_1, summ_l_2, summ_h_2;
                        
                        tmp_vec[6] = vld1q_s32(data + i - 4);
                        tmp_vec[7] = vld1q_s32(data + i - 3);
                        tmp_vec[8] = vld1q_s32(data + i - 2);
                        tmp_vec[9] = vld1q_s32(data + i - 1);
                        tmp_vec[10] = vld1q_s32(data + i - 0);
                        tmp_vec[11] = vld1q_s32(data + i + 1);
                        tmp_vec[12] = vld1q_s32(data + i + 2);
                        tmp_vec[13] = vld1q_s32(data + i + 3);
                        tmp_vec[14] = vld1q_s32(data + i + 4);
                        tmp_vec[15] = vld1q_s32(data + i + 5);
                        tmp_vec[16] = vld1q_s32(data + i + 6);
                        tmp_vec[17] = vld1q_s32(data + i + 7);
                        
                        MUL_64_BIT_LOOP_UNROOL_3(qlp_coeff_2, 1)
                        MACC_64_BIT_LOOP_UNROOL_3(1, qlp_coeff_2, 0) 
                        MACC_64_BIT_LOOP_UNROOL_3(2, qlp_coeff_1, 3) 
                        MACC_64_BIT_LOOP_UNROOL_3(3, qlp_coeff_1, 2) 
                        MACC_64_BIT_LOOP_UNROOL_3(4, qlp_coeff_1, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(5, qlp_coeff_1, 0) 
                        MACC_64_BIT_LOOP_UNROOL_3(6, qlp_coeff_0, 3) 
                        MACC_64_BIT_LOOP_UNROOL_3(7, qlp_coeff_0, 2) 
                        MACC_64_BIT_LOOP_UNROOL_3(8, qlp_coeff_0, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(9, qlp_coeff_0, 0) 

                        SHIFT_SUMS_64BITS_AND_STORE_SUB()
                        
                        tmp_vec[0] = tmp_vec[12];
                        tmp_vec[1] = tmp_vec[13];
                        tmp_vec[2] = tmp_vec[14];
                        tmp_vec[3] = tmp_vec[15];
                        tmp_vec[4] = tmp_vec[16];
                        tmp_vec[5] = tmp_vec[17];
                    }
                }

                else /* order == 9 */ {
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4], qlp_coeff[5], qlp_coeff[6], qlp_coeff[7]);
                    int32x4_t qlp_coeff_2 = INIT_int32x4_t(qlp_coeff[8], 0, 0, 0);

                    tmp_vec[0] = vld1q_s32(data - 9);
                    tmp_vec[1] = vld1q_s32(data - 8);
                    tmp_vec[2] = vld1q_s32(data - 7);
                    tmp_vec[3] = vld1q_s32(data - 6);
                    tmp_vec[4] = vld1q_s32(data - 5);

                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int64x2_t summ_l_0, summ_h_0, summ_l_1, summ_h_1, summ_l_2, summ_h_2;

                        tmp_vec[5] = vld1q_s32(data + i - 4);
                        tmp_vec[6] = vld1q_s32(data + i - 3);
                        tmp_vec[7] = vld1q_s32(data + i - 2);
                        tmp_vec[8] = vld1q_s32(data + i - 1);
                        tmp_vec[9] = vld1q_s32(data + i - 0);
                        tmp_vec[10] = vld1q_s32(data + i + 1);
                        tmp_vec[11] = vld1q_s32(data + i + 2);
                        tmp_vec[12] = vld1q_s32(data + i + 3);
                        tmp_vec[13] = vld1q_s32(data + i + 4);
                        tmp_vec[14] = vld1q_s32(data + i + 5);
                        tmp_vec[15] = vld1q_s32(data + i + 6);
                        tmp_vec[16] = vld1q_s32(data + i + 7);

                        MUL_64_BIT_LOOP_UNROOL_3(qlp_coeff_2, 0)
                        MACC_64_BIT_LOOP_UNROOL_3(1, qlp_coeff_1, 3) 
                        MACC_64_BIT_LOOP_UNROOL_3(2, qlp_coeff_1, 2) 
                        MACC_64_BIT_LOOP_UNROOL_3(3, qlp_coeff_1, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(4, qlp_coeff_1, 0) 
                        MACC_64_BIT_LOOP_UNROOL_3(5, qlp_coeff_0, 3) 
                        MACC_64_BIT_LOOP_UNROOL_3(6, qlp_coeff_0, 2) 
                        MACC_64_BIT_LOOP_UNROOL_3(7, qlp_coeff_0, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(8, qlp_coeff_0, 0) 

                        SHIFT_SUMS_64BITS_AND_STORE_SUB()
                        
                        tmp_vec[0] = tmp_vec[12];
                        tmp_vec[1] = tmp_vec[13];
                        tmp_vec[2] = tmp_vec[14];
                        tmp_vec[3] = tmp_vec[15];
                        tmp_vec[4] = tmp_vec[16];
                    }
                }
            }
        }
        else if (order > 4)
        {
            if (order > 6)
            {
                if (order == 8)
                {
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4], qlp_coeff[5], qlp_coeff[6], qlp_coeff[7]);
                 
                    tmp_vec[0] = vld1q_s32(data - 8);
                    tmp_vec[1] = vld1q_s32(data - 7);
                    tmp_vec[2] = vld1q_s32(data - 6);
                    tmp_vec[3] = vld1q_s32(data - 5);

                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int64x2_t summ_l_0, summ_h_0, summ_l_1, summ_h_1, summ_l_2, summ_h_2;

                        tmp_vec[4] = vld1q_s32(data + i - 4);
                        tmp_vec[5] = vld1q_s32(data + i - 3);
                        tmp_vec[6] = vld1q_s32(data + i - 2);
                        tmp_vec[7] = vld1q_s32(data + i - 1);
                        tmp_vec[8] = vld1q_s32(data + i - 0);
                        tmp_vec[9] = vld1q_s32(data + i + 1);
                        tmp_vec[10] = vld1q_s32(data + i + 2);
                        tmp_vec[11] = vld1q_s32(data + i + 3);
                        tmp_vec[12] = vld1q_s32(data + i + 4);
                        tmp_vec[13] = vld1q_s32(data + i + 5);
                        tmp_vec[14] = vld1q_s32(data + i + 6);
                        tmp_vec[15] = vld1q_s32(data + i + 7);
                        
                      
                        MUL_64_BIT_LOOP_UNROOL_3(qlp_coeff_1, 3)
                        MACC_64_BIT_LOOP_UNROOL_3(1, qlp_coeff_1, 2) 
                        MACC_64_BIT_LOOP_UNROOL_3(2, qlp_coeff_1, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(3, qlp_coeff_1, 0) 
                        MACC_64_BIT_LOOP_UNROOL_3(4, qlp_coeff_0, 3) 
                        MACC_64_BIT_LOOP_UNROOL_3(5, qlp_coeff_0, 2) 
                        MACC_64_BIT_LOOP_UNROOL_3(6, qlp_coeff_0, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(7, qlp_coeff_0, 0) 

                        SHIFT_SUMS_64BITS_AND_STORE_SUB()
                        
                        tmp_vec[0] = tmp_vec[12];
                        tmp_vec[1] = tmp_vec[13];
                        tmp_vec[2] = tmp_vec[14];
                        tmp_vec[3] = tmp_vec[15];
                    }
                }
                else /* order == 7 */
                {
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4], qlp_coeff[5], qlp_coeff[6], 0);

                    tmp_vec[0] = vld1q_s32(data - 7);
                    tmp_vec[1] = vld1q_s32(data - 6);
                    tmp_vec[2] = vld1q_s32(data - 5);
                    

                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int64x2_t summ_l_0, summ_h_0, summ_l_1, summ_h_1, summ_l_2, summ_h_2;
                        tmp_vec[3] = vld1q_s32(data +i - 4);
                        tmp_vec[4] = vld1q_s32(data + i - 3);
                        tmp_vec[5] = vld1q_s32(data + i - 2);
                        tmp_vec[6] = vld1q_s32(data + i - 1);
                        tmp_vec[7] = vld1q_s32(data + i - 0);
                        tmp_vec[8] = vld1q_s32(data + i + 1);
                        tmp_vec[9] = vld1q_s32(data + i + 2);
                        tmp_vec[10] = vld1q_s32(data + i + 3);
                        tmp_vec[11] = vld1q_s32(data + i + 4);
                        tmp_vec[12] = vld1q_s32(data + i + 5);
                        tmp_vec[13] = vld1q_s32(data + i + 6);
                        tmp_vec[14] = vld1q_s32(data + i + 7);
                                              
                      
                        MUL_64_BIT_LOOP_UNROOL_3(qlp_coeff_1, 2)
                        MACC_64_BIT_LOOP_UNROOL_3(1, qlp_coeff_1, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(2, qlp_coeff_1, 0) 
                        MACC_64_BIT_LOOP_UNROOL_3(3, qlp_coeff_0, 3) 
                        MACC_64_BIT_LOOP_UNROOL_3(4, qlp_coeff_0, 2) 
                        MACC_64_BIT_LOOP_UNROOL_3(5, qlp_coeff_0, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(6, qlp_coeff_0, 0) 

                        SHIFT_SUMS_64BITS_AND_STORE_SUB()
                        
                        tmp_vec[0] = tmp_vec[12];
                        tmp_vec[1] = tmp_vec[13];
                        tmp_vec[2] = tmp_vec[14];
                    }
                }
            }
            else
            {
                if (order == 6) {
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4], qlp_coeff[5], 0, 0);

                    tmp_vec[0] = vld1q_s32(data - 6);
                    tmp_vec[1] = vld1q_s32(data - 5);
                    
                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int64x2_t summ_l_0, summ_h_0, summ_l_1, summ_h_1, summ_l_2, summ_h_2;

                        tmp_vec[2] = vld1q_s32(data + i - 4);
                        tmp_vec[3] = vld1q_s32(data + i - 3);
                        tmp_vec[4] = vld1q_s32(data + i - 2);
                        tmp_vec[5] = vld1q_s32(data + i - 1);
                        tmp_vec[6] = vld1q_s32(data + i - 0);
                        tmp_vec[7] = vld1q_s32(data + i + 1);
                        tmp_vec[8] = vld1q_s32(data + i + 2);
                        tmp_vec[9] = vld1q_s32(data + i + 3);
                        tmp_vec[10] = vld1q_s32(data + i + 4);
                        tmp_vec[11] = vld1q_s32(data + i + 5);
                        tmp_vec[12] = vld1q_s32(data + i + 6);
                        tmp_vec[13] = vld1q_s32(data + i + 7);
                        
                       
                        MUL_64_BIT_LOOP_UNROOL_3(qlp_coeff_1, 1)
                        MACC_64_BIT_LOOP_UNROOL_3(1, qlp_coeff_1, 0) 
                        MACC_64_BIT_LOOP_UNROOL_3(2, qlp_coeff_0, 3) 
                        MACC_64_BIT_LOOP_UNROOL_3(3, qlp_coeff_0, 2) 
                        MACC_64_BIT_LOOP_UNROOL_3(4, qlp_coeff_0, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(5, qlp_coeff_0, 0) 
                        
                        SHIFT_SUMS_64BITS_AND_STORE_SUB()
                        
                        tmp_vec[0] = tmp_vec[12];
                        tmp_vec[1] = tmp_vec[13];
                    }
                }

                else
                { /* order == 5 */
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    int32x4_t qlp_coeff_1 = INIT_int32x4_t(qlp_coeff[4], 0, 0, 0);

                    tmp_vec[0] = vld1q_s32(data - 5);
                    
                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int64x2_t summ_l_0, summ_h_0, summ_l_1, summ_h_1, summ_l_2, summ_h_2;
                        tmp_vec[1] = vld1q_s32(data + i - 4);
                        tmp_vec[2] = vld1q_s32(data + i - 3);
                        tmp_vec[3] = vld1q_s32(data + i - 2);
                        tmp_vec[4] = vld1q_s32(data + i - 1);
                        tmp_vec[5] = vld1q_s32(data + i - 0);
                        tmp_vec[6] = vld1q_s32(data + i + 1);
                        tmp_vec[7] = vld1q_s32(data + i + 2);
                        tmp_vec[8] = vld1q_s32(data + i + 3);
                        tmp_vec[9] = vld1q_s32(data + i + 4);
                        tmp_vec[10] = vld1q_s32(data + i + 5);
                        tmp_vec[11] = vld1q_s32(data + i + 6);
                        tmp_vec[12] = vld1q_s32(data + i + 7);
                        
                        MUL_64_BIT_LOOP_UNROOL_3(qlp_coeff_1, 0)
                        MACC_64_BIT_LOOP_UNROOL_3(1, qlp_coeff_0, 3) 
                        MACC_64_BIT_LOOP_UNROOL_3(2, qlp_coeff_0, 2) 
                        MACC_64_BIT_LOOP_UNROOL_3(3, qlp_coeff_0, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(4, qlp_coeff_0, 0) 
                        
                        SHIFT_SUMS_64BITS_AND_STORE_SUB()
                        
                        tmp_vec[0] = tmp_vec[12];
                    }
                }
            }
        }
        else
        {
            if (order > 2)
            {
                if (order == 4)
                {
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], qlp_coeff[3]);
                    
                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int64x2_t summ_l_0, summ_h_0, summ_l_1, summ_h_1, summ_l_2, summ_h_2;
                        tmp_vec[0] = vld1q_s32(data + i - 4);
                        tmp_vec[1] = vld1q_s32(data + i - 3);
                        tmp_vec[2] = vld1q_s32(data + i - 2);
                        tmp_vec[3] = vld1q_s32(data + i - 1);
                        tmp_vec[4] = vld1q_s32(data + i - 0);
                        tmp_vec[5] = vld1q_s32(data + i + 1);
                        tmp_vec[6] = vld1q_s32(data + i + 2);
                        tmp_vec[7] = vld1q_s32(data + i + 3);
                        tmp_vec[8] = vld1q_s32(data + i + 4);
                        tmp_vec[9] = vld1q_s32(data + i + 5);
                        tmp_vec[10] = vld1q_s32(data + i + 6);
                        tmp_vec[11] = vld1q_s32(data + i + 7);
                        
                        MUL_64_BIT_LOOP_UNROOL_3(qlp_coeff_0, 3)
                        MACC_64_BIT_LOOP_UNROOL_3(1, qlp_coeff_0, 2) 
                        MACC_64_BIT_LOOP_UNROOL_3(2, qlp_coeff_0, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(3, qlp_coeff_0, 0) 
                                               
                        SHIFT_SUMS_64BITS_AND_STORE_SUB()                        
                    }
                }
                else
                { /* order == 3 */

                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], qlp_coeff[2], 0);
                    
                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int64x2_t summ_l_0, summ_h_0, summ_l_1, summ_h_1, summ_l_2, summ_h_2;
                        tmp_vec[0] = vld1q_s32(data + i - 3);
                        tmp_vec[1] = vld1q_s32(data + i - 2);
                        tmp_vec[2] = vld1q_s32(data + i - 1);
                        tmp_vec[4] = vld1q_s32(data + i + 1);
                        tmp_vec[5] = vld1q_s32(data + i + 2);
                        tmp_vec[6] = vld1q_s32(data + i + 3);
                        tmp_vec[8] = vld1q_s32(data + i + 5);
                        tmp_vec[9] = vld1q_s32(data + i + 6);
                        tmp_vec[10] = vld1q_s32(data + i + 7);
                        
                        MUL_64_BIT_LOOP_UNROOL_3(qlp_coeff_0, 2)
                        MACC_64_BIT_LOOP_UNROOL_3(1, qlp_coeff_0, 1) 
                        MACC_64_BIT_LOOP_UNROOL_3(2, qlp_coeff_0, 0) 
                        
                        SHIFT_SUMS_64BITS_AND_STORE_SUB()                        
                    }
                }
            }
            else
            {
                if (order == 2)
                {
                    int32x4_t qlp_coeff_0 = INIT_int32x4_t(qlp_coeff[0], qlp_coeff[1], 0, 0);

                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int64x2_t summ_l_0, summ_h_0, summ_l_1, summ_h_1, summ_l_2, summ_h_2;
                        tmp_vec[0] = vld1q_s32(data + i - 2);
                        tmp_vec[1] = vld1q_s32(data + i - 1);
                        tmp_vec[4] = vld1q_s32(data + i + 2);
                        tmp_vec[5] = vld1q_s32(data + i + 3);
                        tmp_vec[8] = vld1q_s32(data + i + 6);
                        tmp_vec[9] = vld1q_s32(data + i + 7);
                        
                        MUL_64_BIT_LOOP_UNROOL_3(qlp_coeff_0, 1)
                        MACC_64_BIT_LOOP_UNROOL_3(1, qlp_coeff_0, 0) 

                        SHIFT_SUMS_64BITS_AND_STORE_SUB()                        
                    }
                }

                else
                { /* order == 1 */

                    int32x2_t qlp_coeff_0_2 = vdup_n_s32(qlp_coeff[0]);
                    int32x4_t qlp_coeff_0_4 = vdupq_n_s32(qlp_coeff[0]);

                    for (i = 0; i < (int)data_len - 11; i += 12)
                    {
                        int64x2_t summ_l_0, summ_h_0, summ_l_1, summ_h_1, summ_l_2, summ_h_2;
                        tmp_vec[0] = vld1q_s32(data + i - 1);
                        tmp_vec[4] = vld1q_s32(data + i + 3);
                        tmp_vec[8] = vld1q_s32(data + i + 7);
                        
                        summ_l_0 = vmull_s32(vget_low_s32(tmp_vec[0]), qlp_coeff_0_2);
                        summ_h_0 = vmull_high_s32(tmp_vec[0], qlp_coeff_0_4);

                        summ_l_1 = vmull_s32(vget_low_s32(tmp_vec[4]), qlp_coeff_0_2);
                        summ_h_1 = vmull_high_s32(tmp_vec[4], qlp_coeff_0_4);

                        summ_l_2 = vmull_s32(vget_low_s32(tmp_vec[8]), qlp_coeff_0_2);
                        summ_h_2 = vmull_high_s32(tmp_vec[8], qlp_coeff_0_4);

                        SHIFT_SUMS_64BITS_AND_STORE_SUB()                        
                    }
                }
            }
        }
        for (; i < (int)data_len; i++)
        {
            sum = 0;
            switch (order)
            {
            case 12:
                sum += qlp_coeff[11] * (FLAC__int64)data[i - 12]; /* Falls through. */
            case 11:
                sum += qlp_coeff[10] * (FLAC__int64)data[i - 11]; /* Falls through. */
            case 10:
                sum += qlp_coeff[9] * (FLAC__int64)data[i - 10]; /* Falls through. */
            case 9:
                sum += qlp_coeff[8] * (FLAC__int64)data[i - 9]; /* Falls through. */
            case 8:
                sum += qlp_coeff[7] * (FLAC__int64)data[i - 8]; /* Falls through. */
            case 7:
                sum += qlp_coeff[6] * (FLAC__int64)data[i - 7]; /* Falls through. */
            case 6:
                sum += qlp_coeff[5] * (FLAC__int64)data[i - 6]; /* Falls through. */
            case 5:
                sum += qlp_coeff[4] * (FLAC__int64)data[i - 5]; /* Falls through. */
            case 4:
                sum += qlp_coeff[3] * (FLAC__int64)data[i - 4]; /* Falls through. */
            case 3:
                sum += qlp_coeff[2] * (FLAC__int64)data[i - 3]; /* Falls through. */
            case 2:
                sum += qlp_coeff[1] * (FLAC__int64)data[i - 2]; /* Falls through. */
            case 1:
                sum += qlp_coeff[0] * (FLAC__int64)data[i - 1];
            }
            residual[i] = data[i] - (sum >> lp_quantization);
        }
    }
    else
    { /* order > 12 */
        for (i = 0; i < (int)data_len; i++)
        {
            sum = 0;
            switch (order)
            {
            case 32:
                sum += qlp_coeff[31] * (FLAC__int64)data[i - 32]; /* Falls through. */
            case 31:
                sum += qlp_coeff[30] * (FLAC__int64)data[i - 31]; /* Falls through. */
            case 30:
                sum += qlp_coeff[29] * (FLAC__int64)data[i - 30]; /* Falls through. */
            case 29:
                sum += qlp_coeff[28] * (FLAC__int64)data[i - 29]; /* Falls through. */
            case 28:
                sum += qlp_coeff[27] * (FLAC__int64)data[i - 28]; /* Falls through. */
            case 27:
                sum += qlp_coeff[26] * (FLAC__int64)data[i - 27]; /* Falls through. */
            case 26:
                sum += qlp_coeff[25] * (FLAC__int64)data[i - 26]; /* Falls through. */
            case 25:
                sum += qlp_coeff[24] * (FLAC__int64)data[i - 25]; /* Falls through. */
            case 24:
                sum += qlp_coeff[23] * (FLAC__int64)data[i - 24]; /* Falls through. */
            case 23:
                sum += qlp_coeff[22] * (FLAC__int64)data[i - 23]; /* Falls through. */
            case 22:
                sum += qlp_coeff[21] * (FLAC__int64)data[i - 22]; /* Falls through. */
            case 21:
                sum += qlp_coeff[20] * (FLAC__int64)data[i - 21]; /* Falls through. */
            case 20:
                sum += qlp_coeff[19] * (FLAC__int64)data[i - 20]; /* Falls through. */
            case 19:
                sum += qlp_coeff[18] * (FLAC__int64)data[i - 19]; /* Falls through. */
            case 18:
                sum += qlp_coeff[17] * (FLAC__int64)data[i - 18]; /* Falls through. */
            case 17:
                sum += qlp_coeff[16] * (FLAC__int64)data[i - 17]; /* Falls through. */
            case 16:
                sum += qlp_coeff[15] * (FLAC__int64)data[i - 16]; /* Falls through. */
            case 15:
                sum += qlp_coeff[14] * (FLAC__int64)data[i - 15]; /* Falls through. */
            case 14:
                sum += qlp_coeff[13] * (FLAC__int64)data[i - 14]; /* Falls through. */
            case 13:
                sum += qlp_coeff[12] * (FLAC__int64)data[i - 13];
                sum += qlp_coeff[11] * (FLAC__int64)data[i - 12];
                sum += qlp_coeff[10] * (FLAC__int64)data[i - 11];
                sum += qlp_coeff[9] * (FLAC__int64)data[i - 10];	
                sum += qlp_coeff[8] * (FLAC__int64)data[i - 9];
                sum += qlp_coeff[7] * (FLAC__int64)data[i - 8];
                sum += qlp_coeff[6] * (FLAC__int64)data[i - 7];
                sum += qlp_coeff[5] * (FLAC__int64)data[i - 6];
                sum += qlp_coeff[4] * (FLAC__int64)data[i - 5];
                sum += qlp_coeff[3] * (FLAC__int64)data[i - 4];
                sum += qlp_coeff[2] * (FLAC__int64)data[i - 3];
                sum += qlp_coeff[1] * (FLAC__int64)data[i - 2];
                sum += qlp_coeff[0] * (FLAC__int64)data[i - 1];
            }
            residual[i] = data[i] - (sum >> lp_quantization);
        }
    }

    return;
}

#endif /* FLAC__CPU_ARM64 && FLAC__HAS_ARCH64INTRIN */
#endif /* FLAC__NO_ASM */
#endif /* FLAC__INTEGER_ONLY_LIBRARY */
