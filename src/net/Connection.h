/* This header file is writen by qp09
 * usually just for fun
 * Thu December 29 2016
 */
#ifndef CONNECTION_H
#define CONNECTION_H

#include <mpi.h>
#include <stdio.h>
#include <string>

#include "../base/constant.h"
#include "../base/type.h"

using std::string;

#define access_(array, a, b) ((array)[a*num + b])

struct Connection {
	//int *pSynapsesIdx; 
	//int synapsesNum; 
	size_t nNum;
	size_t sNum;

	unsigned int maxDelay;
	unsigned int minDelay;

    uinteger_t *pDelayStart;
	uinteger_t *pDelayNum;
	uinteger_t *pSidMap;
    uinteger_t *dst;

	uinteger_t *pDelayStartRev;
	uinteger_t *pDelayNumRev; 
	uinteger_t *pSidMapRev;
};

Connection * allocConnection(size_t nNum, size_t sNum, unsigned int maxDelay, unsigned int minDelay);

int freeConnection(Connection * pCPU);

Connection * cudaAllocConnection(Connection * pCPU);
int cudaFetchConnection(Connection *pCPU, Connection *pGPU);
int cudaFreeConnection(Connection *pGPU);

int saveConnection(Connection *conn, const string &path, const Type &type);
Connection * loadConnection(const string &path, const Type &type);

bool isEqualConnection(Connection *c1, Connection *c2);

int sendConnection(Connection *conn, int dest, int tag, MPI_Comm comm);
Connection *recvConnection(int src, int tag, MPI_Comm comm);

#endif /* CONNECTION_H */

