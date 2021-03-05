/* This header file is writen by qp09
 * usually just for fun
 * Thu December 29 2016
 */
#ifndef CONNECTION_H
#define CONNECTION_H

#include <mpi.h>
#include <stdio.h>

struct Connection {
	//int *pSynapsesIdx; 
	//int synapsesNum; 
	size_t nNum;
	size_t sNum;

	unsigned int maxDelay;
	unsigned int minDelay;

    size_t *pDelayStart;
	size_t *pDelayNum;
	size_t *pSidMap;
    size_t *dst;

	size_t *pDelayStartRev;
	size_t *pDelayNumRev; 
	size_t *pSidMapRev;
};

Connection * allocConnection(size_t nNum, size_t sNum, unsigned int maxDelay, unsigned int minDelay);

int freeConnection(Connection * pCPU);

Connection * cudaAllocConnection(Connection * pCPU);
int cudaFetchConnection(Connection *pCPU, Connection *pGPU);
int cudaFreeConnection(Connection *pGPU);

int saveConnection(Connection *conn, FILE *f);
Connection * loadConnection(FILE *f);

bool isEqualConnection(Connection *c1, Connection *c2);

int sendConnection(Connection *conn, int dest, int tag, MPI_Comm comm);
Connection *recvConnection(int src, int tag, MPI_Comm comm);

#endif /* CONNECTION_H */

