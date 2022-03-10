//===-------- OpenMPMath.cpp - Implementation of OpenMP math fns -----------===
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------------===

#include <OpenMPMath.h>

#define __DEVICE__ __attribute__((always_inline, nothrow))

extern "C" {

__DEVICE__ int __omp_abs(int __a) { return abs(__a); }
__DEVICE__ double __omp_fabs(double __a) { return fabs(__a); }
__DEVICE__ double __omp_acos(double __a) { return acos(__a); }
__DEVICE__ float __omp_acosf(float __a) { return acosf(__a); }
__DEVICE__ double __omp_acosh(double __a) { return acosh(__a); }
__DEVICE__ float __omp_acoshf(float __a) { return acoshf(__a); }
__DEVICE__ double __omp_asin(double __a) { return asin(__a); }
__DEVICE__ float __omp_asinf(float __a) { return asinf(__a); }
__DEVICE__ double __omp_asinh(double __a) { return asinh(__a); }
__DEVICE__ float __omp_asinhf(float __a) { return asinhf(__a); }
__DEVICE__ double __omp_atan(double __a) { return atan(__a); }
__DEVICE__ double __omp_atan2(double __a, double __b) {
  return atan2(__a, __b);
}
__DEVICE__ float __omp_atan2f(float __a, float __b) { return atan2f(__a, __b); }
__DEVICE__ float __omp_atanf(float __a) { return atanf(__a); }
__DEVICE__ double __omp_atanh(double __a) { return atanh(__a); }
__DEVICE__ float __omp_atanhf(float __a) { return atanhf(__a); }
__DEVICE__ double __omp_cbrt(double __a) { return cbrt(__a); }
__DEVICE__ float __omp_cbrtf(float __a) { return cbrtf(__a); }
__DEVICE__ double __omp_ceil(double __a) { return ceil(__a); }
__DEVICE__ float __omp_ceilf(float __a) { return ceilf(__a); }
__DEVICE__ double __omp_copysign(double __a, double __b) {
  return copysign(__a, __b);
}
__DEVICE__ float __omp_copysignf(float __a, float __b) {
  return copysignf(__a, __b);
}
__DEVICE__ double __omp_cos(double __a) { return cos(__a); }
__DEVICE__ float __omp_cosf(float __a) { return cosf(__a); }
__DEVICE__ double __omp_cosh(double __a) { return cosh(__a); }
__DEVICE__ float __omp_coshf(float __a) { return coshf(__a); }
__DEVICE__ double __omp_cospi(double __a) { return cospi(__a); }
__DEVICE__ float __omp_cospif(float __a) { return cospif(__a); }
__DEVICE__ double __omp_cyl_bessel_i0(double __a) { return cyl_bessel_i0(__a); }
__DEVICE__ float __omp_cyl_bessel_i0f(float __a) { return cyl_bessel_i0f(__a); }
__DEVICE__ double __omp_cyl_bessel_i1(double __a) { return cyl_bessel_i1(__a); }
__DEVICE__ float __omp_cyl_bessel_i1f(float __a) { return cyl_bessel_i1f(__a); }
__DEVICE__ double __omp_erf(double __a) { return erf(__a); }
__DEVICE__ double __omp_erfc(double __a) { return erfc(__a); }
__DEVICE__ float __omp_erfcf(float __a) { return erfcf(__a); }
__DEVICE__ double __omp_erfcinv(double __a) { return erfcinv(__a); }
__DEVICE__ float __omp_erfcinvf(float __a) { return erfcinvf(__a); }
__DEVICE__ double __omp_erfcx(double __a) { return erfcx(__a); }
__DEVICE__ float __omp_erfcxf(float __a) { return erfcxf(__a); }
__DEVICE__ float __omp_erff(float __a) { return erff(__a); }
__DEVICE__ double __omp_erfinv(double __a) { return erfinv(__a); }
__DEVICE__ float __omp_erfinvf(float __a) { return erfinvf(__a); }
__DEVICE__ double __omp_exp(double __a) { return exp(__a); }
__DEVICE__ double __omp_exp10(double __a) { return exp10(__a); }
__DEVICE__ float __omp_exp10f(float __a) { return exp10f(__a); }
__DEVICE__ double __omp_exp2(double __a) { return exp2(__a); }
__DEVICE__ float __omp_exp2f(float __a) { return exp2f(__a); }
__DEVICE__ float __omp_expf(float __a) { return expf(__a); }
__DEVICE__ double __omp_expm1(double __a) { return expm1(__a); }
__DEVICE__ float __omp_expm1f(float __a) { return expm1f(__a); }
__DEVICE__ float __omp_fabsf(float __a) { return fabsf(__a); }
__DEVICE__ double __omp_fdim(double __a, double __b) { return fdim(__a, __b); }
__DEVICE__ float __omp_fdimf(float __a, float __b) { return fdimf(__a, __b); }
__DEVICE__ double __omp_fdivide(double __a, double __b) { return __a / __b; }
__DEVICE__ float __omp_fdividef(float __a, float __b) { return __a / __b; }
__DEVICE__ double __omp_floor(double __f) { return floor(__f); }
__DEVICE__ float __omp_floorf(float __f) { return floorf(__f); }
__DEVICE__ double __omp_fma(double __a, double __b, double __c) {
  return fma(__a, __b, __c);
}
__DEVICE__ float __omp_fmaf(float __a, float __b, float __c) {
  return fmaf(__a, __b, __c);
}
__DEVICE__ double __omp_fmax(double __a, double __b) { return fmax(__a, __b); }
__DEVICE__ float __omp_fmaxf(float __a, float __b) { return fmaxf(__a, __b); }
__DEVICE__ double __omp_fmin(double __a, double __b) { return fmin(__a, __b); }
__DEVICE__ float __omp_fminf(float __a, float __b) { return fminf(__a, __b); }
__DEVICE__ double __omp_fmod(double __a, double __b) { return fmod(__a, __b); }
__DEVICE__ float __omp_fmodf(float __a, float __b) { return fmodf(__a, __b); }
__DEVICE__ double __omp_frexp(double __a, int *__b) { return frexp(__a, __b); }
__DEVICE__ float __omp_frexpf(float __a, int *__b) { return frexpf(__a, __b); }
__DEVICE__ double __omp_hypot(double __a, double __b) {
  return hypot(__a, __b);
}
__DEVICE__ float __omp_hypotf(float __a, float __b) { return hypotf(__a, __b); }
__DEVICE__ int __omp_ilogb(double __a) { return ilogb(__a); }
__DEVICE__ int __omp_ilogbf(float __a) { return ilogbf(__a); }
__DEVICE__ double __omp_j0(double __a) { return j0(__a); }
__DEVICE__ float __omp_j0f(float __a) { return j0f(__a); }
__DEVICE__ double __omp_j1(double __a) { return j1(__a); }
__DEVICE__ float __omp_j1f(float __a) { return j1f(__a); }
__DEVICE__ double __omp_jn(int __n, double __a) { return jn(__n, __a); }
__DEVICE__ float __omp_jnf(int __n, float __a) { return jnf(__n, __a); }
#if defined(__LP64__) || defined(_WIN64)
__DEVICE__ long __omp_labs(long __a) { return llabs(__a); };
#else
__DEVICE__ long __omp_labs(long __a) { return abs(__a); };
#endif
__DEVICE__ double __omp_ldexp(double __a, int __b) { return ldexp(__a, __b); }
__DEVICE__ float __omp_ldexpf(float __a, int __b) { return ldexpf(__a, __b); }
__DEVICE__ double __omp_lgamma(double __a) { return lgamma(__a); }
__DEVICE__ float __omp_lgammaf(float __a) { return lgammaf(__a); }
__DEVICE__ long long __omp_llabs(long long __a) { return llabs(__a); }
__DEVICE__ long long __omp_llmax(long long __a, long long __b) {
  return llmax(__a, __b);
}
__DEVICE__ long long __omp_llmin(long long __a, long long __b) {
  return llmin(__a, __b);
}
__DEVICE__ long long __omp_llrint(double __a) { return llrint(__a); }
__DEVICE__ long long __omp_llrintf(float __a) { return llrintf(__a); }
__DEVICE__ long long __omp_llround(double __a) { return llround(__a); }
__DEVICE__ long long __omp_llroundf(float __a) { return llroundf(__a); }
__DEVICE__ double __omp_round(double __a) { return round(__a); }
__DEVICE__ float __omp_roundf(float __a) { return roundf(__a); }
__DEVICE__ double __omp_log(double __a) { return log(__a); }
__DEVICE__ double __omp_log10(double __a) { return log10(__a); }
__DEVICE__ float __omp_log10f(float __a) { return log10f(__a); }
__DEVICE__ double __omp_log1p(double __a) { return log1p(__a); }
__DEVICE__ float __omp_log1pf(float __a) { return log1pf(__a); }
__DEVICE__ double __omp_log2(double __a) { return log2(__a); }
__DEVICE__ float __omp_log2f(float __a) { return log2f(__a); }
__DEVICE__ double __omp_logb(double __a) { return logb(__a); }
__DEVICE__ float __omp_logbf(float __a) { return logbf(__a); }
__DEVICE__ float __omp_logf(float __a) { return logf(__a); }
__DEVICE__ long __omp_lrint(double __a) { return lrint(__a); }
__DEVICE__ long __omp_lrintf(float __a) { return lrintf(__a); }
__DEVICE__ long __omp_lround(double __a) { return lround(__a); }
__DEVICE__ long __omp_lroundf(float __a) { return lroundf(__a); }
__DEVICE__ int __omp_max(int __a, int __b) { return max(__a, __b); }
__DEVICE__ int __omp_min(int __a, int __b) { return min(__a, __b); }
__DEVICE__ double __omp_modf(double __a, double *__b) { return modf(__a, __b); }
__DEVICE__ float __omp_modff(float __a, float *__b) { return modff(__a, __b); }
__DEVICE__ double __omp_nearbyint(double __a) { return nearbyint(__a); }
__DEVICE__ float __omp_nearbyintf(float __a) { return nearbyintf(__a); }
__DEVICE__ double __omp_nextafter(double __a, double __b) {
  return nextafter(__a, __b);
}
__DEVICE__ float __omp_nextafterf(float __a, float __b) {
  return nextafterf(__a, __b);
}
__DEVICE__ double __omp_norm(int __dim, const double *__t) {
  return norm(__dim, __t);
}
__DEVICE__ double __omp_norm3d(double __a, double __b, double __c) {
  return norm3d(__a, __b, __c);
}
__DEVICE__ float __omp_norm3df(float __a, float __b, float __c) {
  return norm3df(__a, __b, __c);
}
__DEVICE__ double __omp_norm4d(double __a, double __b, double __c, double __d) {
  return norm4d(__a, __b, __c, __d);
}
__DEVICE__ float __omp_norm4df(float __a, float __b, float __c, float __d) {
  return norm4df(__a, __b, __c, __d);
}
__DEVICE__ double __omp_normcdf(double __a) { return normcdf(__a); }
__DEVICE__ float __omp_normcdff(float __a) { return normcdff(__a); }
__DEVICE__ double __omp_normcdfinv(double __a) { return normcdfinv(__a); }
__DEVICE__ float __omp_normcdfinvf(float __a) { return normcdfinvf(__a); }
__DEVICE__ float __omp_normf(int __dim, const float *__t) {
  return normf(__dim, __t);
}
__DEVICE__ double __omp_pow(double __a, double __b) { return pow(__a, __b); }
__DEVICE__ float __omp_powf(float __a, float __b) { return powf(__a, __b); }
__DEVICE__ double __omp_powi(double __a, int __b) { return powi(__a, __b); }
__DEVICE__ float __omp_powif(float __a, int __b) { return powif(__a, __b); }
__DEVICE__ double __omp_rcbrt(double __a) { return rcbrt(__a); }
__DEVICE__ float __omp_rcbrtf(float __a) { return rcbrtf(__a); }
__DEVICE__ double __omp_remainder(double __a, double __b) {
  return remainder(__a, __b);
}
__DEVICE__ float __omp_remainderf(float __a, float __b) {
  return remainderf(__a, __b);
}
__DEVICE__ double __omp_remquo(double __a, double __b, int *__c) {
  return remquo(__a, __b, __c);
}
__DEVICE__ float __omp_remquof(float __a, float __b, int *__c) {
  return remquof(__a, __b, __c);
}
__DEVICE__ double __omp_rhypot(double __a, double __b) {
  return rhypot(__a, __b);
}
__DEVICE__ float __omp_rhypotf(float __a, float __b) {
  return rhypotf(__a, __b);
}
__DEVICE__ double __omp_rint(double __a) { return rint(__a); }
__DEVICE__ float __omp_rintf(float __a) { return rintf(__a); }
__DEVICE__ double __omp_rnorm(int __a, const double *__b) {
  return rnorm(__a, __b);
}
__DEVICE__ double __omp_rnorm3d(double __a, double __b, double __c) {
  return rnorm3d(__a, __b, __c);
}
__DEVICE__ float __omp_rnorm3df(float __a, float __b, float __c) {
  return rnorm3df(__a, __b, __c);
}
__DEVICE__ double __omp_rnorm4d(double __a, double __b, double __c,
                                double __d) {
  return rnorm4d(__a, __b, __c, __d);
}
__DEVICE__ float __omp_rnorm4df(float __a, float __b, float __c, float __d) {
  return rnorm4df(__a, __b, __c, __d);
}
__DEVICE__ float __omp_rnormf(int __dim, const float *__t) {
  return rnormf(__dim, __t);
}
__DEVICE__ double __omp_rsqrt(double __a) { return rsqrt(__a); }
__DEVICE__ float __omp_rsqrtf(float __a) { return rsqrtf(__a); }
__DEVICE__ double __omp_scalbn(double __a, int __b) { return scalbn(__a, __b); }
__DEVICE__ float __omp_scalbnf(float __a, int __b) { return scalbnf(__a, __b); }
__DEVICE__ double __omp_scalbln(double __a, long __b) {
  return scalbn(__a, (int)__b);
}
__DEVICE__ float __omp_scalblnf(float __a, long __b) {
  return scalbnf(__a, (int)__b);
}
__DEVICE__ double __omp_sin(double __a) { return sin(__a); }
__DEVICE__ void __omp_sincos(double __a, double *__s, double *__c) {
  return sincos(__a, __s, __c);
}
__DEVICE__ void __omp_sincosf(float __a, float *__s, float *__c) {
  return sincosf(__a, __s, __c);
}
__DEVICE__ void __omp_sincospi(double __a, double *__s, double *__c) {
  return sincospi(__a, __s, __c);
}
__DEVICE__ void __omp_sincospif(float __a, float *__s, float *__c) {
  return sincospif(__a, __s, __c);
}
__DEVICE__ float __omp_sinf(float __a) { return sinf(__a); }
__DEVICE__ double __omp_sinh(double __a) { return sinh(__a); }
__DEVICE__ float __omp_sinhf(float __a) { return sinhf(__a); }
__DEVICE__ double __omp_sinpi(double __a) { return sinpi(__a); }
__DEVICE__ float __omp_sinpif(float __a) { return sinpif(__a); }
__DEVICE__ double __omp_sqrt(double __a) { return sqrt(__a); }
__DEVICE__ float __omp_sqrtf(float __a) { return sqrtf(__a); }
__DEVICE__ double __omp_tan(double __a) { return tan(__a); }
__DEVICE__ float __omp_tanf(float __a) { return tanf(__a); }
__DEVICE__ double __omp_tanh(double __a) { return tanh(__a); }
__DEVICE__ float __omp_tanhf(float __a) { return tanhf(__a); }
__DEVICE__ double __omp_tgamma(double __a) { return tgamma(__a); }
__DEVICE__ float __omp_tgammaf(float __a) { return tgammaf(__a); }
__DEVICE__ double __omp_trunc(double __a) { return trunc(__a); }
__DEVICE__ float __omp_truncf(float __a) { return truncf(__a); }
__DEVICE__ unsigned long long __omp_ullmax(unsigned long long __a,
                                           unsigned long long __b) {
  return ullmax(__a, __b);
}
__DEVICE__ unsigned long long __omp_ullmin(unsigned long long __a,
                                           unsigned long long __b) {
  return ullmin(__a, __b);
}
__DEVICE__ unsigned int __omp_umax(unsigned int __a, unsigned int __b) {
  return umax(__a, __b);
}
__DEVICE__ unsigned int __omp_umin(unsigned int __a, unsigned int __b) {
  return umin(__a, __b);
}
__DEVICE__ double __omp_y0(double __a) { return y0(__a); }
__DEVICE__ float __omp_y0f(float __a) { return y0f(__a); }
__DEVICE__ double __omp_y1(double __a) { return y1(__a); }
__DEVICE__ float __omp_y1f(float __a) { return y1f(__a); }
__DEVICE__ double __omp_yn(int __a, double __b) { return yn(__a, __b); }
__DEVICE__ float __omp_ynf(int __a, float __b) { return ynf(__a, __b); }
}
