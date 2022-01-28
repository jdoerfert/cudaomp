//===--- omp_math.cpp - Implementation of the math API for targets -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

//#include <limits.h>
#include <math.h>
#include <stdlib.h>

#include "target_impl.h"
#include "target/omp_math.h"

#define USED __attribute__((used))

#pragma omp declare target


extern "C" {

INLINE USED int __nv_abs(int __a) { return abs(__a); }
INLINE USED double __nv_fabs(double __a) { return fabs(__a); }
INLINE USED double __nv_acos(double __a) { return acos(__a); }
INLINE USED float __nv_acosf(float __a) { return acosf(__a); }
INLINE USED double __nv_acosh(double __a) { return acosh(__a); }
INLINE USED float __nv_acoshf(float __a) { return acoshf(__a); }
INLINE USED double __nv_asin(double __a) { return asin(__a); }
INLINE USED float __nv_asinf(float __a) { return asinf(__a); }
INLINE USED double __nv_asinh(double __a) { return asinh(__a); }
INLINE USED float __nv_asinhf(float __a) { return asinhf(__a); }
INLINE USED double __nv_atan(double __a) { return atan(__a); }
INLINE USED double __nv_atan2(double __a, double __b) { return atan2(__a, __b); }
INLINE USED float __nv_atan2f(float __a, float __b) { return atan2f(__a, __b); }
INLINE USED float __nv_atanf(float __a) { return atanf(__a); }
INLINE USED double __nv_atanh(double __a) { return atanh(__a); }
INLINE USED float __nv_atanhf(float __a) { return atanhf(__a); }
INLINE USED double __nv_cbrt(double __a) { return cbrt(__a); }
INLINE USED float __nv_cbrtf(float __a) { return cbrtf(__a); }
INLINE USED double __nv_ceil(double __a) { return ceil(__a); }
INLINE USED float __nv_ceilf(float __a) { return ceilf(__a); }
INLINE USED double __nv_copysign(double __a, double __b) {
  return copysign(__a, __b);
}
INLINE USED float __nv_copysignf(float __a, float __b) {
  return copysignf(__a, __b);
}
INLINE USED double __nv_cos(double __a) { return cos(__a); }
INLINE USED float __nv_fast_cosf(float __a) {
  return cosf(__a);
}
INLINE USED float __nv_cosf(float __a) {
  return cosf(__a);
}
INLINE USED double __nv_cosh(double __a) { return cosh(__a); }
INLINE USED float __nv_coshf(float __a) { return coshf(__a); }
INLINE USED double __nv_cospi(double __a) { return cospi(__a); }
INLINE USED float __nv_cospif(float __a) { return cospif(__a); }
INLINE USED double __nv_cyl_bessel_i0(double __a) { return cyl_bessel_i0(__a); }
INLINE USED float __nv_cyl_bessel_i0f(float __a) { return cyl_bessel_i0f(__a); }
INLINE USED double __nv_cyl_bessel_i1(double __a) { return cyl_bessel_i1(__a); }
INLINE USED float __nv_cyl_bessel_i1f(float __a) { return cyl_bessel_i1f(__a); }
INLINE USED double __nv_erf(double __a) { return erf(__a); }
INLINE USED double __nv_erfc(double __a) { return erfc(__a); }
INLINE USED float __nv_erfcf(float __a) { return erfcf(__a); }
INLINE USED double __nv_erfcinv(double __a) { return erfcinv(__a); }
INLINE USED float __nv_erfcinvf(float __a) { return erfcinvf(__a); }
INLINE USED double __nv_erfcx(double __a) { return erfcx(__a); }
INLINE USED float __nv_erfcxf(float __a) { return erfcxf(__a); }
INLINE USED float __nv_erff(float __a) { return erff(__a); }
INLINE USED double __nv_erfinv(double __a) { return erfinv(__a); }
INLINE USED float __nv_erfinvf(float __a) { return erfinvf(__a); }
INLINE USED double __nv_exp(double __a) { return exp(__a); }
INLINE USED double __nv_exp10(double __a) { return exp10(__a); }
INLINE USED float __nv_exp10f(float __a) { return exp10f(__a); }
INLINE USED double __nv_exp2(double __a) { return exp2(__a); }
INLINE USED float __nv_exp2f(float __a) { return exp2f(__a); }
INLINE USED float __nv_expf(float __a) { return expf(__a); }
INLINE USED double __nv_expm1(double __a) { return expm1(__a); }
INLINE USED float __nv_expm1f(float __a) { return expm1f(__a); }
INLINE USED float __nv_fabsf(float __a) { return fabsf(__a); }
INLINE USED double __nv_fdim(double __a, double __b) { return fdim(__a, __b); }
INLINE USED float __nv_fdimf(float __a, float __b) { return fdimf(__a, __b); }
// TODO: No need to override fdivide, resolves to straight division?
//double fdivide(double __a, double __b) { return __a / __b; }
INLINE USED float __nv_fast_fdividef(float __a, float __b) {
  return fdividef(__a, __b);
}
INLINE USED double __nv_floor(double __f) { return floor(__f); }
INLINE USED float __nv_floorf(float __f) { return floorf(__f); }
INLINE USED double __nv_fma(double __a, double __b, double __c) {
  return fma(__a, __b, __c);
}
INLINE USED float __nv_fmaf(float __a, float __b, float __c) {
  return fmaf(__a, __b, __c);
}
INLINE USED double __nv_fmax(double __a, double __b) { return fmax(__a, __b); }
INLINE USED float __nv_fmaxf(float __a, float __b) { return fmaxf(__a, __b); }
INLINE USED double __nv_fmin(double __a, double __b) { return fmin(__a, __b); }
INLINE USED float __nv_fminf(float __a, float __b) { return fminf(__a, __b); }
INLINE USED double __nv_fmod(double __a, double __b) { return fmod(__a, __b); }
INLINE USED float __nv_fmodf(float __a, float __b) { return fmodf(__a, __b); }
INLINE USED double __nv_frexp(double __a, int *__b) { return frexp(__a, __b); }
INLINE USED float __nv_frexpf(float __a, int *__b) { return frexpf(__a, __b); }
INLINE USED double __nv_hypot(double __a, double __b) { return hypot(__a, __b); }
INLINE USED float __nv_hypotf(float __a, float __b) { return hypotf(__a, __b); }
INLINE USED int __nv_ilogb(double __a) { return ilogb(__a); }
INLINE USED int __nv_ilogbf(float __a) { return ilogbf(__a); }
INLINE USED double __nv_j0(double __a) { return j0(__a); }
INLINE USED float __nv_j0f(float __a) { return j0f(__a); }
INLINE USED double __nv_j1(double __a) { return j1(__a); }
INLINE USED float __nv_j1f(float __a) { return j1f(__a); }
INLINE USED double __nv_jn(int __n, double __a) { return jn(__n, __a); }
INLINE USED float __nv_jnf(int __n, float __a) { return jnf(__n, __a); }
INLINE USED double __nv_ldexp(double __a, int __b) { return ldexp(__a, __b); }
INLINE USED float __nv_ldexpf(float __a, int __b) { return ldexpf(__a, __b); }
INLINE USED double __nv_lgamma(double __a) { return lgamma(__a); }
INLINE USED float __nv_lgammaf(float __a) { return lgammaf(__a); }
INLINE USED long long __nv_llabs(long long __a) { return llabs(__a); }
INLINE USED long long __nv_llmax(long long __a, long long __b) {
  return llmax(__a, __b);
}
INLINE USED long long __nv_llmin(long long __a, long long __b) {
  return llmin(__a, __b);
}
INLINE USED long long __nv_llrint(double __a) { return llrint(__a); }
INLINE USED long long __nv_llrintf(float __a) { return llrintf(__a); }
INLINE USED long long __nv_llround(double __a) { return llround(__a); }
INLINE USED long long __nv_llroundf(float __a) { return llroundf(__a); }
INLINE USED double __nv_round(double __a) { return round(__a); }
INLINE USED float __nv_roundf(float __a) { return roundf(__a); }
INLINE USED double __nv_log(double __a) { return log(__a); }
INLINE USED double __nv_log10(double __a) { return log10(__a); }
INLINE USED float __nv_log10f(float __a) { return log10f(__a); }
INLINE USED double __nv_log1p(double __a) { return log1p(__a); }
INLINE USED float __nv_log1pf(float __a) { return log1pf(__a); }
INLINE USED double __nv_log2(double __a) { return log2(__a); }
INLINE USED float __nv_fast_log2f(float __a) {
  return log2f(__a);
}
INLINE USED float __nv_log2f(float __a) {
  return log2f(__a);
}
INLINE USED double __nv_logb(double __a) { return logb(__a); }
INLINE USED float __nv_logbf(float __a) { return logbf(__a); }
INLINE USED float __nv_fast_logf(float __a) {
  return logf(__a);
}
INLINE USED float __nv_logf(float __a) {
  return logf(__a);
}
// TODO: not needed since they resolve to overriden ones?
//long llrint(double __a) { return lrint(__a); }
//long __float2ll_rn(float __a) { return lrintf(__a); }
//long llround(double __a) { return lround(__a); }
//long llroundf(float __a) { return lroundf(__a); }
//long rint(double __a) { return lrint(__a); }
//long __float2int_rn(float __a) { return lrintf(__a); }
//long round(double __a) { return lround(__a); }
//long roundf(float __a) { return lroundf(__a); }
INLINE USED int __nv_max(int __a, int __b) { return max(__a, __b); }
INLINE USED int __nv_min(int __a, int __b) { return min(__a, __b); }
INLINE USED double __nv_modf(double __a, double *__b) { return modf(__a, __b); }
INLINE USED float __nv_modff(float __a, float *__b) { return modff(__a, __b); }
// TODO: fix builtins
//INLINE USED double __builtin_nearbyint(double __a) { return nearbyint(__a); }
//INLINE USED float __builtin_nearbyintf(float __a) { return nearbyintf(__a); }
INLINE USED double __nv_nextafter(double __a, double __b) {
  return nextafter(__a, __b);
}
INLINE USED float __nv_nextafterf(float __a, float __b) {
  return nextafterf(__a, __b);
}
INLINE USED double __nv_norm(int __dim, const double *__t) {
  return norm(__dim, __t);
}
INLINE USED double __nv_norm3d(double __a, double __b, double __c) {
  return norm3d(__a, __b, __c);
}
INLINE USED float __nv_norm3df(float __a, float __b, float __c) {
  return norm3df(__a, __b, __c);
}
INLINE USED double __nv_norm4d(double __a, double __b, double __c, double __d) {
  return norm4d(__a, __b, __c, __d);
}
INLINE USED float __nv_norm4df(float __a, float __b, float __c, float __d) {
  return norm4df(__a, __b, __c, __d);
}
INLINE USED double __nv_normcdf(double __a) { return normcdf(__a); }
INLINE USED float __nv_normcdff(float __a) { return normcdff(__a); }
INLINE USED double __nv_normcdfinv(double __a) { return normcdfinv(__a); }
INLINE USED float __nv_normcdfinvf(float __a) { return normcdfinvf(__a); }
INLINE USED float __nv_normf(int __dim, const float *__t) {
  return normf(__dim, __t);
}
INLINE USED double __nv_pow(double __a, double __b) { return pow(__a, __b); }
INLINE USED float __nv_powf(float __a, float __b) { return powf(__a, __b); }
INLINE USED double __nv_powi(double __a, int __b) { return powi(__a, __b); }
INLINE USED float __nv_powif(float __a, int __b) { return powif(__a, __b); }
INLINE USED double __nv_rcbrt(double __a) { return rcbrt(__a); }
INLINE USED float __nv_rcbrtf(float __a) { return rcbrtf(__a); }
INLINE USED double __nv_remainder(double __a, double __b) {
  return remainder(__a, __b);
}
INLINE USED float __nv_remainderf(float __a, float __b) {
  return remainderf(__a, __b);
}
INLINE USED double __nv_remquo(double __a, double __b, int *__c) {
  return remquo(__a, __b, __c);
}
INLINE USED float __nv_remquof(float __a, float __b, int *__c) {
  return remquof(__a, __b, __c);
}
INLINE USED double __nv_rhypot(double __a, double __b) {
  return rhypot(__a, __b);
}
INLINE USED float __nv_rhypotf(float __a, float __b) {
  return rhypotf(__a, __b);
}
// TODO: fix builtins
//double __builtin_rint(double __a) { return rint(__a); }
//float __builtin_rintf(float __a) { return rintf(__a); }
INLINE USED double __nv_rnorm(int __a, const double *__b) {
  return rnorm(__a, __b);
}
INLINE USED double __nv_rnorm3d(double __a, double __b, double __c) {
  return rnorm3d(__a, __b, __c);
}
INLINE USED float __nv_rnorm3df(float __a, float __b, float __c) {
  return rnorm3df(__a, __b, __c);
}
INLINE USED double __nv_rnorm4d(double __a, double __b, double __c, double __d) {
  return rnorm4d(__a, __b, __c, __d);
}
INLINE USED float __nv_rnorm4df(float __a, float __b, float __c, float __d) {
  return rnorm4df(__a, __b, __c, __d);
}
INLINE USED float __nv_rnormf(int __dim, const float *__t) {
  return rnormf(__dim, __t);
}
INLINE USED double __nv_rsqrt(double __a) { return rsqrt(__a); }
INLINE USED float __nv_rsqrtf(float __a) { return rsqrtf(__a); }
INLINE USED double __nv_scalbn(double __a, int __b) { return scalbn(__a, __b); }
INLINE USED float __nv_scalbnf(float __a, int __b) { return scalbnf(__a, __b); }
// TODO: We don't need to override those since they will resolve to existing
// overriden functions?
// double scalbn(double __a, long __b) {
//  if (__b > INT_MAX)
//    return __a > 0 ? HUGE_VAL : -HUGE_VAL;
//  if (__b < INT_MIN)
//    return __a > 0 ? 0.0 : -0.0;
//  return scalbln(__a, (int)__b);
//}
// float scalbnf(float __a, long __b) {
//  if (__b > INT_MAX)
//    return __a > 0 ? HUGE_VALF : -HUGE_VALF;
//  if (__b < INT_MIN)
//    return __a > 0 ? 0.f : -0.f;
//  return scalblnf(__a, (int)__b);
//}
INLINE USED double __nv_sin(double __a) { return sin(__a); }
INLINE USED void __nv_sincos(double __a, double *__s, double *__c) {
  return sincos(__a, __s, __c);
}
INLINE USED void __nv_fast_sincosf(float __a, float *__s, float *__c) {
  return sincosf(__a, __s, __c);
}
INLINE USED void __nv_sincosf(float __a, float *__s, float *__c) {
  return sincosf(__a, __s, __c);
}
INLINE USED void __nv_sincospi(double __a, double *__s, double *__c) {
  return sincospi(__a, __s, __c);
}
INLINE USED void __nv_sincospif(float __a, float *__s, float *__c) {
  return sincospif(__a, __s, __c);
}
INLINE USED float __nv_fast_sinf(float __a) {
  return sinf(__a);
}
INLINE USED float __nv_sinf(float __a) {
  return sinf(__a);
}
INLINE USED double __nv_sinh(double __a) { return sinh(__a); }
INLINE USED float __nv_sinhf(float __a) { return sinhf(__a); }
INLINE USED double __nv_sinpi(double __a) { return sinpi(__a); }
INLINE USED float __nv_sinpif(float __a) { return sinpif(__a); }
INLINE USED double __nv_sqrt(double __a) { return sqrt(__a); }
INLINE USED float __nv_sqrtf(float __a) { return sqrtf(__a); }
INLINE USED double __nv_tan(double __a) { return tan(__a); }
INLINE USED float __nv_tanf(float __a) { return tanf(__a); }
INLINE USED double __nv_tanh(double __a) { return tanh(__a); }
INLINE USED float __nv_tanhf(float __a) { return tanhf(__a); }
INLINE USED double __nv_tgamma(double __a) { return tgamma(__a); }
INLINE USED float __nv_tgammaf(float __a) { return tgammaf(__a); }
INLINE USED double __nv_trunc(double __a) { return trunc(__a); }
INLINE USED float __nv_truncf(float __a) { return truncf(__a); }
INLINE USED unsigned long long __nv_ullmax(unsigned long long __a,
                                     unsigned long long __b) {
  return ullmax(__a, __b);
}
INLINE USED unsigned long long __nv_ullmin(unsigned long long __a,
                                     unsigned long long __b) {
  return ullmin(__a, __b);
}
INLINE USED unsigned int __nv_umax(unsigned int __a, unsigned int __b) {
  return umax(__a, __b);
}
INLINE USED unsigned int __nv_umin(unsigned int __a, unsigned int __b) {
  return umin(__a, __b);
}
INLINE USED double __nv_y0(double __a) { return y0(__a); }
INLINE USED float __nv_y0f(float __a) { return y0f(__a); }
INLINE USED double __nv_y1(double __a) { return y1(__a); }
INLINE USED float __nv_y1f(float __a) { return y1f(__a); }
INLINE USED double __nv_yn(int __a, double __b) { return yn(__a, __b); }
INLINE USED float __nv_ynf(int __a, float __b) { return ynf(__a, __b); }

}

#pragma omp end declare target