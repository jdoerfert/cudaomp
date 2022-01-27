#define __VECTOR_FUNCTIONS_DECL__ static __inline__ __host__ __device__
#include "host_defines.h"
#include "vector_types.h"

#define __VECTOR_FUNCTIONS_DECL__ static __inline__ __host__ __device__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

__VECTOR_FUNCTIONS_DECL__ char1 make_char1(signed char x)
{
  char1 t; t.x = x; return t;
}

__VECTOR_FUNCTIONS_DECL__ uchar1 make_uchar1(unsigned char x)
{
  uchar1 t; t.x = x; return t;
}

__VECTOR_FUNCTIONS_DECL__ char2 make_char2(signed char x, signed char y)
{
  char2 t; t.x = x; t.y = y; return t;
}

__VECTOR_FUNCTIONS_DECL__ uchar2 make_uchar2(unsigned char x, unsigned char y)
{
  uchar2 t; t.x = x; t.y = y; return t;
}

__VECTOR_FUNCTIONS_DECL__ char3 make_char3(signed char x, signed char y, signed char z)
{
  char3 t; t.x = x; t.y = y; t.z = z; return t;
}

__VECTOR_FUNCTIONS_DECL__ uchar3 make_uchar3(unsigned char x, unsigned char y, unsigned char z)
{
  uchar3 t; t.x = x; t.y = y; t.z = z; return t;
}

__VECTOR_FUNCTIONS_DECL__ char4 make_char4(signed char x, signed char y, signed char z, signed char w)
{
  char4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__VECTOR_FUNCTIONS_DECL__ uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
{
  uchar4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__VECTOR_FUNCTIONS_DECL__ short1 make_short1(short x)
{
  short1 t; t.x = x; return t;
}

__VECTOR_FUNCTIONS_DECL__ ushort1 make_ushort1(unsigned short x)
{
  ushort1 t; t.x = x; return t;
}

__VECTOR_FUNCTIONS_DECL__ short2 make_short2(short x, short y)
{
  short2 t; t.x = x; t.y = y; return t;
}

__VECTOR_FUNCTIONS_DECL__ ushort2 make_ushort2(unsigned short x, unsigned short y)
{
  ushort2 t; t.x = x; t.y = y; return t;
}

__VECTOR_FUNCTIONS_DECL__ short3 make_short3(short x,short y, short z)
{ 
  short3 t; t.x = x; t.y = y; t.z = z; return t;
}

__VECTOR_FUNCTIONS_DECL__ ushort3 make_ushort3(unsigned short x, unsigned short y, unsigned short z)
{
  ushort3 t; t.x = x; t.y = y; t.z = z; return t;
}

__VECTOR_FUNCTIONS_DECL__ short4 make_short4(short x, short y, short z, short w)
{
  short4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__VECTOR_FUNCTIONS_DECL__ ushort4 make_ushort4(unsigned short x, unsigned short y, unsigned short z, unsigned short w)
{
  ushort4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__VECTOR_FUNCTIONS_DECL__ int1 make_int1(int x)
{
  int1 t; t.x = x; return t;
}

__VECTOR_FUNCTIONS_DECL__ uint1 make_uint1(unsigned int x)
{
  uint1 t; t.x = x; return t;
}

__VECTOR_FUNCTIONS_DECL__ int2 make_int2(int x, int y)
{
  int2 t; t.x = x; t.y = y; return t;
}

__VECTOR_FUNCTIONS_DECL__ uint2 make_uint2(unsigned int x, unsigned int y)
{
  uint2 t; t.x = x; t.y = y; return t;
}

__VECTOR_FUNCTIONS_DECL__ int3 make_int3(int x, int y, int z)
{
  int3 t; t.x = x; t.y = y; t.z = z; return t;
}

__VECTOR_FUNCTIONS_DECL__ uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z)
{
  uint3 t; t.x = x; t.y = y; t.z = z; return t;
}

__VECTOR_FUNCTIONS_DECL__ int4 make_int4(int x, int y, int z, int w)
{
  int4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__VECTOR_FUNCTIONS_DECL__ uint4 make_uint4(unsigned int x, unsigned int y, unsigned int z, unsigned int w)
{
  uint4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__VECTOR_FUNCTIONS_DECL__ long1 make_long1(long int x)
{
  long1 t; t.x = x; return t;
}

__VECTOR_FUNCTIONS_DECL__ ulong1 make_ulong1(unsigned long int x)
{
  ulong1 t; t.x = x; return t;
}

__VECTOR_FUNCTIONS_DECL__ long2 make_long2(long int x, long int y)
{
  long2 t; t.x = x; t.y = y; return t;
}

__VECTOR_FUNCTIONS_DECL__ ulong2 make_ulong2(unsigned long int x, unsigned long int y)
{
  ulong2 t; t.x = x; t.y = y; return t;
}

__VECTOR_FUNCTIONS_DECL__ long3 make_long3(long int x, long int y, long int z)
{
  long3 t; t.x = x; t.y = y; t.z = z; return t;
}

__VECTOR_FUNCTIONS_DECL__ ulong3 make_ulong3(unsigned long int x, unsigned long int y, unsigned long int z)
{
  ulong3 t; t.x = x; t.y = y; t.z = z; return t;
}

__VECTOR_FUNCTIONS_DECL__ long4 make_long4(long int x, long int y, long int z, long int w)
{
  long4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__VECTOR_FUNCTIONS_DECL__ ulong4 make_ulong4(unsigned long int x, unsigned long int y, unsigned long int z, unsigned long int w)
{
  ulong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__VECTOR_FUNCTIONS_DECL__ float1 make_float1(float x)
{
  float1 t; t.x = x; return t;
}

__VECTOR_FUNCTIONS_DECL__ float2 make_float2(float x, float y)
{
  float2 t; t.x = x; t.y = y; return t;
}

__VECTOR_FUNCTIONS_DECL__ float3 make_float3(float x, float y, float z)
{
  float3 t; t.x = x; t.y = y; t.z = z; return t;
}

__VECTOR_FUNCTIONS_DECL__ float4 make_float4(float x, float y, float z, float w)
{
  float4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__VECTOR_FUNCTIONS_DECL__ longlong1 make_longlong1(long long int x)
{
  longlong1 t; t.x = x; return t;
}

__VECTOR_FUNCTIONS_DECL__ ulonglong1 make_ulonglong1(unsigned long long int x)
{
  ulonglong1 t; t.x = x; return t;
}

__VECTOR_FUNCTIONS_DECL__ longlong2 make_longlong2(long long int x, long long int y)
{
  longlong2 t; t.x = x; t.y = y; return t;
}

__VECTOR_FUNCTIONS_DECL__ ulonglong2 make_ulonglong2(unsigned long long int x, unsigned long long int y)
{
  ulonglong2 t; t.x = x; t.y = y; return t;
}

__VECTOR_FUNCTIONS_DECL__ longlong3 make_longlong3(long long int x, long long int y, long long int z)
{
  longlong3 t; t.x = x; t.y = y; t.z = z; return t;
}

__VECTOR_FUNCTIONS_DECL__ ulonglong3 make_ulonglong3(unsigned long long int x, unsigned long long int y, unsigned long long int z)
{
  ulonglong3 t; t.x = x; t.y = y; t.z = z; return t;
}

__VECTOR_FUNCTIONS_DECL__ longlong4 make_longlong4(long long int x, long long int y, long long int z, long long int w)
{
  longlong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__VECTOR_FUNCTIONS_DECL__ ulonglong4 make_ulonglong4(unsigned long long int x, unsigned long long int y, unsigned long long int z, unsigned long long int w)
{
  ulonglong4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

__VECTOR_FUNCTIONS_DECL__ double1 make_double1(double x)
{
  double1 t; t.x = x; return t;
}

__VECTOR_FUNCTIONS_DECL__ double2 make_double2(double x, double y)
{
  double2 t; t.x = x; t.y = y; return t;
}

__VECTOR_FUNCTIONS_DECL__ double3 make_double3(double x, double y, double z)
{
  double3 t; t.x = x; t.y = y; t.z = z; return t;
}

__VECTOR_FUNCTIONS_DECL__ double4 make_double4(double x, double y, double z, double w)
{
  double4 t; t.x = x; t.y = y; t.z = z; t.w = w; return t;
}

#undef __VECTOR_FUNCTIONS_DECL__

#endif /* !__VECTOR_FUNCTIONS_HPP__ */
