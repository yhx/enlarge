/* This header file is writen by qp09
 * usually just for fun
 * Thu October 22 2015
 */
#ifndef CONSTANT_H
#define CONSTANT_H

#include <stddef.h>

#define MPI_CREAL MPI_FLOAT

#ifndef USE_DOUBLE
typedef float real;
#else
typedef double real;
#endif

#ifndef USE_DOUBLE
typedef MPI_FLOAT MPI_U_REAL;
#else
typedef MPI_DOUBLE MPI_U_REAL;
#endif

typedef int ID;

const real ZERO = 1e-10;

const int MAX_BLOCK_SIZE = 1024;
const int WARP_SIZE = 32;

const int DECAY_MULTIPLE_TAU = 5;

const int DATA_TAG = 0x80000000;
const int MSG_TAG = 0;
const int DNET_TAG = 100000;
const int NET_TAG = 10000;
const int TYPE_TAG = 100;

#endif /* CONSTANT_H */

