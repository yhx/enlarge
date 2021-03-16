/* This header file is writen by qp09
 * usually just for fun
 * Thu October 22 2015
 */
#ifndef CONSTANT_H
#define CONSTANT_H

#include <stddef.h>
#include <limits.h>
#include "mpi.h"

// typedef unsigned long long uinteger_t;
// #define UINTEGER_T_MAX ULL_MAX;
// #define MPI_UINTEGER_T MPI_UNSIGNED_LONG_LONG

typedef unsigned int uinteger_t;
#define UINTEGER_T_MAX UINT_MAX;
#define MPI_UINTEGER_T MPI_UNSIGNED

typedef int integer_t;
#define INTEGER_T_MAX INT_MAX;
#define MPI_INTEGER_T MPI_INT;

#ifndef USE_DOUBLE
typedef float real;
#else
typedef double real;
#endif

#ifndef USE_DOUBLE
#define MPI_U_REAL MPI_FLOAT
#else
#define MPI_U_REAL MPI_DOUBLE
#endif

#if SIZE_MAX == UCHAR_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
   #define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
   #define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
   #error "unsupported size_t"
#endif

const real ZERO = 1e-10;

const int MAX_BLOCK_SIZE = 1024;
const int WARP_SIZE = 32;

const int DECAY_MULTIPLE_TAU = 5;

const int DATA_TAG = 0x1000000;
const int MSG_TAG = 0;
const int DNET_TAG = 100000;
const int NET_TAG = 10000;
const int TYPE_TAG = 200;
const int CONN_TAG = 200;

#endif /* CONSTANT_H */

