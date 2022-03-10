//===- glang_math_forward_declares.h - Prototypes of evice__ math fns --===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===

#ifndef gLANG__OPENMP_MATH_FORWARD_DECLARES_H__
#define gLANG__OPENMP_MATH_FORWARD_DECLARES_H__

#if !defined(_OPENMP)
#error "This file is for OpenMP compilation only."
#endif

#pragma push_macro("__DEVICE__")
#define __DEVICE__

#if defined(__cplusplus)
extern "C" {
#endif

__DEVICE__ int __omp_abs(int);
__DEVICE__ double __omp_fabs(double);
__DEVICE__ double __omp_acos(double);
__DEVICE__ float __omp_acosf(float);
__DEVICE__ double __omp_acosh(double);
__DEVICE__ float __omp_acoshf(float);
__DEVICE__ double __omp_asin(double);
__DEVICE__ float __omp_asinf(float);
__DEVICE__ double __omp_asinh(double);
__DEVICE__ float __omp_asinhf(float);
__DEVICE__ double __omp_atan(double);
__DEVICE__ double __omp_atan2(double, double);
__DEVICE__ float __omp_atan2f(float, float);
__DEVICE__ float __omp_atanf(float);
__DEVICE__ double __omp_atanh(double);
__DEVICE__ float __omp_atanhf(float);
__DEVICE__ double __omp_cbrt(double);
__DEVICE__ float __omp_cbrtf(float);
__DEVICE__ double __omp_ceil(double);
__DEVICE__ float __omp_ceilf(float);
__DEVICE__ double __omp_copysign(double, double);
__DEVICE__ float __omp_copysignf(float, float);
__DEVICE__ double __omp_cos(double);
__DEVICE__ float __omp_cosf(float);
__DEVICE__ double __omp_cosh(double);
__DEVICE__ float __omp_coshf(float);
__DEVICE__ double __omp_cospi(double);
__DEVICE__ float __omp_cospif(float);
__DEVICE__ double __omp_cyl_bessel_i0(double);
__DEVICE__ float __omp_cyl_bessel_i0f(float);
__DEVICE__ double __omp_cyl_bessel_i1(double);
__DEVICE__ float __omp_cyl_bessel_i1f(float);
__DEVICE__ double __omp_erf(double);
__DEVICE__ double __omp_erfc(double);
__DEVICE__ float __omp_erfcf(float);
__DEVICE__ double __omp_erfcinv(double);
__DEVICE__ float __omp_erfcinvf(float);
__DEVICE__ double __omp_erfcx(double);
__DEVICE__ float __omp_erfcxf(float);
__DEVICE__ float __omp_erff(float);
__DEVICE__ double __omp_erfinv(double);
__DEVICE__ float __omp_erfinvf(float);
__DEVICE__ double __omp_exp(double);
__DEVICE__ double __omp_exp10(double);
__DEVICE__ float __omp_exp10f(float);
__DEVICE__ double __omp_exp2(double);
__DEVICE__ float __omp_exp2f(float);
__DEVICE__ float __omp_expf(float);
__DEVICE__ double __omp_expm1(double);
__DEVICE__ float __omp_expm1f(float);
__DEVICE__ float __omp_fabsf(float);
__DEVICE__ double __omp_fdim(double, double);
__DEVICE__ float __omp_fdimf(float, float);
__DEVICE__ double __omp_fdivide(double, double);
__DEVICE__ float __omp_fdividef(float, float);
__DEVICE__ double __omp_floor(double __f);
__DEVICE__ float __omp_floorf(float __f);
__DEVICE__ double __omp_fma(double, double, double);
__DEVICE__ float __omp_fmaf(float, float, float);
__DEVICE__ double __omp_fmax(double, double);
__DEVICE__ float __omp_fmaxf(float, float);
__DEVICE__ double __omp_fmin(double, double);
__DEVICE__ float __omp_fminf(float, float);
__DEVICE__ double __omp_fmod(double, double);
__DEVICE__ float __omp_fmodf(float, float);
__DEVICE__ double __omp_frexp(double, int *);
__DEVICE__ float __omp_frexpf(float, int *);
__DEVICE__ double __omp_hypot(double, double);
__DEVICE__ float __omp_hypotf(float, float);
__DEVICE__ int __omp_ilogb(double);
__DEVICE__ int __omp_ilogbf(float);
__DEVICE__ double __omp_j0(double);
__DEVICE__ float __omp_j0f(float);
__DEVICE__ double __omp_j1(double);
__DEVICE__ float __omp_j1f(float);
__DEVICE__ double __omp_jn(int __n, double);
__DEVICE__ float __omp_jnf(int __n, float);
__DEVICE__ long __omp_labs(long);
__DEVICE__ double __omp_ldexp(double, int);
__DEVICE__ float __omp_ldexpf(float, int);
__DEVICE__ double __omp_lgamma(double);
__DEVICE__ float __omp_lgammaf(float);
__DEVICE__ long long __omp_llabs(long long);
__DEVICE__ long long __omp_llmax(long long, long long);
__DEVICE__ long long __omp_llmin(long long, long long);
__DEVICE__ long long __omp_llrint(double);
__DEVICE__ long long __omp_llrintf(float);
__DEVICE__ long long __omp_llround(double);
__DEVICE__ long long __omp_llroundf(float);
__DEVICE__ double __omp_round(double);
__DEVICE__ float __omp_roundf(float);
__DEVICE__ double __omp_log(double);
__DEVICE__ double __omp_log10(double);
__DEVICE__ float __omp_log10f(float);
__DEVICE__ double __omp_log1p(double);
__DEVICE__ float __omp_log1pf(float);
__DEVICE__ double __omp_log2(double);
__DEVICE__ float __omp_log2f(float);
__DEVICE__ double __omp_logb(double);
__DEVICE__ float __omp_logbf(float);
__DEVICE__ float __omp_logf(float);
__DEVICE__ long __omp_lrint(double);
__DEVICE__ long __omp_lrintf(float);
__DEVICE__ long __omp_lround(double);
__DEVICE__ long __omp_lroundf(float);
__DEVICE__ int __omp_max(int, int);
__DEVICE__ int __omp_min(int, int);
__DEVICE__ double __omp_modf(double, double *);
__DEVICE__ float __omp_modff(float, float *);
__DEVICE__ double __omp_nearbyint(double);
__DEVICE__ float __omp_nearbyintf(float);
__DEVICE__ double __omp_nextafter(double, double);
__DEVICE__ float __omp_nextafterf(float, float);
__DEVICE__ double __omp_norm(int im, const double *);
__DEVICE__ double __omp_norm3d(double, double, double);
__DEVICE__ float __omp_norm3df(float, float, float);
__DEVICE__ double __omp_norm4d(double, double, double, double);
__DEVICE__ float __omp_norm4df(float, float, float, float);
__DEVICE__ double __omp_normcdf(double);
__DEVICE__ float __omp_normcdff(float);
__DEVICE__ double __omp_normcdfinv(double);
__DEVICE__ float __omp_normcdfinvf(float);
__DEVICE__ float __omp_normf(int im, const float *);
__DEVICE__ double __omp_pow(double, double);
__DEVICE__ float __omp_powf(float, float);
__DEVICE__ double __omp_powi(double, int);
__DEVICE__ float __omp_powif(float, int);
__DEVICE__ double __omp_rcbrt(double);
__DEVICE__ float __omp_rcbrtf(float);
__DEVICE__ double __omp_remainder(double, double);
__DEVICE__ float __omp_remainderf(float, float);
__DEVICE__ double __omp_remquo(double, double, int *);
__DEVICE__ float __omp_remquof(float, float, int *);
__DEVICE__ double __omp_rhypot(double, double);
__DEVICE__ float __omp_rhypotf(float, float);
__DEVICE__ double __omp_rint(double);
__DEVICE__ float __omp_rintf(float);
__DEVICE__ double __omp_rnorm(int, const double *);
__DEVICE__ double __omp_rnorm3d(double, double, double);
__DEVICE__ float __omp_rnorm3df(float, float, float);
__DEVICE__ double __omp_rnorm4d(double, double, double, double);
__DEVICE__ float __omp_rnorm4df(float, float, float, float);
__DEVICE__ float __omp_rnormf(int im, const float *);
__DEVICE__ double __omp_rsqrt(double);
__DEVICE__ float __omp_rsqrtf(float);
__DEVICE__ double __omp_scalbn(double, int);
__DEVICE__ float __omp_scalbnf(float, int);
__DEVICE__ double __omp_scalbln(double, long);
__DEVICE__ float __omp_scalblnf(float, long);
__DEVICE__ double __omp_sin(double);
__DEVICE__ void __omp_sincos(double, double *, double *);
__DEVICE__ void __omp_sincosf(float, float *, float *);
__DEVICE__ void __omp_sincospi(double, double *, double *);
__DEVICE__ void __omp_sincospif(float, float *, float *);
__DEVICE__ float __omp_sinf(float);
__DEVICE__ double __omp_sinh(double);
__DEVICE__ float __omp_sinhf(float);
__DEVICE__ double __omp_sinpi(double);
__DEVICE__ float __omp_sinpif(float);
__DEVICE__ double __omp_sqrt(double);
__DEVICE__ float __omp_sqrtf(float);
__DEVICE__ double __omp_tan(double);
__DEVICE__ float __omp_tanf(float);
__DEVICE__ double __omp_tanh(double);
__DEVICE__ float __omp_tanhf(float);
__DEVICE__ double __omp_tgamma(double);
__DEVICE__ float __omp_tgammaf(float);
__DEVICE__ double __omp_trunc(double);
__DEVICE__ float __omp_truncf(float);
__DEVICE__ unsigned long long __omp_ullmax(unsigned long long,
                                           unsigned long long);
__DEVICE__ unsigned long long __omp_ullmin(unsigned long long,
                                           unsigned long long);
__DEVICE__ unsigned int __omp_umax(unsigned int, unsigned int);
__DEVICE__ unsigned int __omp_umin(unsigned int, unsigned int);
__DEVICE__ double __omp_y0(double);
__DEVICE__ float __omp_y0f(float);
__DEVICE__ double __omp_y1(double);
__DEVICE__ float __omp_y1f(float);
__DEVICE__ double __omp_yn(int, double);
__DEVICE__ float __omp_ynf(int, float);

#if defined(__cplusplus)
}
#endif

#pragma pop_macro("__DEVICE__")

#endif
