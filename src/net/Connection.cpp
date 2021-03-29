
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "../utils/helper_c.h"
#include "../utils/utils.h"
#include "Connection.h"

Connection * allocConnection(size_t nNum, size_t sNum, unsigned int maxDelay, unsigned int minDelay)
{
	Connection *ret = (Connection*)malloc(sizeof(Connection));
	assert(ret != NULL);

	ret->nNum = nNum;
	ret->sNum = sNum;
	ret->maxDelay = maxDelay;
	ret->minDelay = minDelay;

    size_t length = (maxDelay - minDelay + 1) * nNum;

	ret->pDelayStart = malloc_c<uinteger_t>(length+1);
	ret->pDelayNum = malloc_c<uinteger_t>(length);
	ret->pSidMap = malloc_c<uinteger_t>(sNum);
	ret->dst = malloc_c<uinteger_t>(sNum);

	ret->pDelayStartRev = malloc_c<uinteger_t>(length+1);
	ret->pDelayNumRev = malloc_c<uinteger_t>(length);
	ret->pSidMapRev = malloc_c<uinteger_t>(sNum);

	return ret;
}

int freeConnection(Connection *pCPU)
{
	free(pCPU->pDelayStart);
	free(pCPU->pDelayNum);
	free(pCPU->pSidMap);
	free(pCPU->dst);
	free(pCPU->pDelayStartRev);
	free(pCPU->pDelayNumRev);
	free(pCPU->pSidMapRev);
	free(pCPU);
	pCPU = NULL;
	return 0;
}

int saveConnection(Connection *conn, FILE *f)
{
	fwrite_c(&(conn->nNum), 1, f);
	fwrite_c(&(conn->sNum), 1, f);
	fwrite_c(&(conn->maxDelay), 1, f);
	fwrite_c(&(conn->minDelay), 1, f);

	size_t length = (conn->maxDelay - conn->minDelay + 1) * conn->nNum;

	fwrite_c(conn->pDelayStart, length+1, f);
	fwrite_c(conn->pDelayNum, length, f);
	fwrite_c(conn->pSidMap, conn->sNum, f);
	fwrite_c(conn->dst, conn->sNum, f);

	fwrite_c(conn->pDelayStartRev, length+1, f);
	fwrite_c(conn->pDelayNumRev, length, f);
	fwrite_c(conn->pSidMapRev, conn->sNum, f);

	return 0;
}

Connection * loadConnection(FILE *f)
{
	Connection *conn = (Connection *)malloc(sizeof(Connection));

	fread_c(&(conn->nNum), 1, f);
	fread_c(&(conn->sNum), 1, f);
	fread_c(&(conn->maxDelay), 1, f);
	fread_c(&(conn->minDelay), 1, f);

	size_t length = (conn->maxDelay - conn->minDelay + 1) * conn->nNum;
	size_t sNum = conn->sNum;

	conn->pDelayStart = malloc_c<uinteger_t>(length+1);
	conn->pDelayNum = malloc_c<uinteger_t>(length);
	conn->pSidMap = malloc_c<uinteger_t>(sNum);
	conn->dst = malloc_c<uinteger_t>(sNum);

	conn->pDelayStartRev = malloc_c<uinteger_t>(length+1);
	conn->pDelayNumRev = malloc_c<uinteger_t>(length);
	conn->pSidMapRev = malloc_c<uinteger_t>(sNum);

	fread_c(conn->pDelayStart, length+1, f);
	fread_c(conn->pDelayNum, length, f);
	fread_c(conn->pSidMap, conn->sNum, f);
	fread_c(conn->dst, conn->sNum, f);

	fread_c(conn->pDelayStartRev, length+1, f);
	fread_c(conn->pDelayNumRev, length, f);
	fread_c(conn->pSidMapRev, conn->sNum, f);

	return conn;
}


bool isEqualConnection(Connection *c1, Connection *c2)
{
	bool ret = true;
	ret = ret && (c1->nNum == c2->nNum);
	ret = ret && (c1->sNum == c2->sNum);
	ret = ret && (c1->maxDelay == c2->maxDelay);
	ret = ret && (c1->minDelay == c2->minDelay);

	size_t length = (c1->maxDelay - c1->minDelay + 1) * c1->nNum;

	ret = ret && isEqualArray(c1->pDelayStart, c2->pDelayStart, length+1);
	ret = ret && isEqualArray(c1->pDelayNum, c2->pDelayNum, length);
	// ret = ret && isEqualArray(c1->pSidMap, c2->pSidMap, c1->sNum);
	ret = ret && isEqualArray(c1->dst, c2->dst, c1->sNum);

	// ret = ret && isEqualArray(c1->pDelayStartRev, c2->pDelayStartRev, length+1);
	// ret = ret && isEqualArray(c1->pDelayNumRev, c2->pDelayNumRev, length);

	// ret = ret && isEqualArray(c1->pSidMapRev, c2->pSidMapRev, c1->sNum);

	return ret;
}


int sendConnection(Connection *conn, int dest, int tag, MPI_Comm comm)
{
	int ret = 0;
	ret = MPI_Send(conn, sizeof(Connection), MPI_UNSIGNED_CHAR, dest, tag, comm);
	assert(ret == MPI_SUCCESS);

	unsigned int length = (conn->maxDelay - conn->minDelay + 1) * conn->nNum;

	ret = MPI_Send(conn->pDelayStart, length+1, MPI_UINTEGER_T, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(conn->pDelayNum, length, MPI_UINTEGER_T, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(conn->pSidMap, conn->sNum, MPI_UINTEGER_T, dest, tag+3, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(conn->dst, conn->sNum, MPI_UINTEGER_T, dest, tag+4, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(conn->pDelayStartRev, length+1, MPI_UINTEGER_T, dest, tag+5, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(conn->pDelayNumRev, length, MPI_UINTEGER_T, dest, tag+6, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(conn->pSidMapRev, conn->sNum, MPI_UINTEGER_T, dest, tag+7, comm);
	assert(ret == MPI_SUCCESS);
	
	return ret;
}

Connection *recvConnection(int src, int tag, MPI_Comm comm)
{
	Connection *conn = (Connection *)malloc(sizeof(Connection));
	int ret = 0;
	MPI_Status status;

	ret = MPI_Recv(conn, sizeof(Connection), MPI_UNSIGNED_CHAR, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);

	int length = (conn->maxDelay - conn->minDelay + 1) * (conn->nNum);

	conn->pDelayStart = malloc_c<uinteger_t>(length+1);
	conn->pDelayNum = malloc_c<uinteger_t>(length);

	ret = MPI_Recv(conn->pDelayStart, length+1, MPI_UINTEGER_T, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(conn->pDelayNum, length, MPI_UINTEGER_T, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);

	conn->pSidMap = malloc_c<uinteger_t>(conn->sNum);
	conn->dst = malloc_c<uinteger_t>(conn->sNum);

	ret = MPI_Recv(conn->pSidMap, conn->sNum, MPI_UINTEGER_T, src, tag+3, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(conn->dst, conn->sNum, MPI_UINTEGER_T, src, tag+4, comm, &status);
	assert(ret==MPI_SUCCESS);

	conn->pDelayStartRev = malloc_c<uinteger_t>(length+1);
	conn->pDelayNumRev = malloc_c<uinteger_t>(length);

	ret = MPI_Recv(conn->pDelayStartRev, length+1, MPI_UINTEGER_T, src, tag+5, comm, &status);
	assert(ret==MPI_SUCCESS);
	ret = MPI_Recv(conn->pDelayNumRev, length, MPI_UINTEGER_T, src, tag+6, comm, &status);
	assert(ret==MPI_SUCCESS);

	conn->pSidMapRev = malloc_c<uinteger_t>(conn->sNum);

	ret = MPI_Recv(conn->pSidMapRev, conn->sNum, MPI_UINTEGER_T, src, tag+7, comm, &status);
	assert(ret==MPI_SUCCESS);

	return conn;
}
