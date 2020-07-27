
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "../utils/utils.h"

#include "Connection.h"

Connection * allocConnection(int nNum, int sNum, int maxDelay, int minDelay)
{
	Connection *ret = (Connection*)malloc(sizeof(Connection));
	assert(ret != NULL);

	ret->nNum = nNum;
	ret->sNum = sNum;
	ret->maxDelay = maxDelay;
	ret->minDelay = minDelay;

	int length = (maxDelay - minDelay + 1) * nNum;

	ret->pDelayStart = (int*)malloc(sizeof(int)*length);
	assert(ret->pDelayStart != NULL);
	memset(ret->pDelayStart, 0, sizeof(int)*length);
	ret->pDelayNum = (int*)malloc(sizeof(int)*length);
	assert(ret->pDelayNum != NULL);
	memset(ret->pDelayNum, 0, sizeof(int)*length);

	ret->pDelayStartRev = (int*)malloc(sizeof(int)*length);
	assert(ret->pDelayStartRev != NULL);
	memset(ret->pDelayStartRev, 0, sizeof(int)*length);
	ret->pDelayNumRev = (int*)malloc(sizeof(int)*length);
	assert(ret->pDelayNumRev != NULL);
	memset(ret->pDelayNumRev, 0, sizeof(int)*length);
	ret->pSidMapRev = (int*)malloc(sizeof(int)*sNum);
	assert(ret->pSidMapRev != NULL);
	memset(ret->pSidMapRev, 0, sizeof(int)*sNum);

	return ret;
}

int freeConnection(Connection *pCPU)
{
	free(pCPU->pDelayStart);
	free(pCPU->pDelayNum);
	free(pCPU->pDelayStartRev);
	free(pCPU->pDelayNumRev);
	free(pCPU->pSidMapRev);
	free(pCPU);
	pCPU = NULL;
	return 0;
}

int saveConnection(Connection *conn, FILE *f)
{
	fwrite(&(conn->nNum), sizeof(int), 1, f);
	fwrite(&(conn->sNum), sizeof(int), 1, f);
	fwrite(&(conn->maxDelay), sizeof(int), 1, f);
	fwrite(&(conn->minDelay), sizeof(int), 1, f);

	int length = (conn->maxDelay - conn->minDelay + 1) * conn->nNum;

	fwrite(conn->pDelayStart, sizeof(int), length, f);
	fwrite(conn->pDelayNum, sizeof(int), length, f);

	fwrite(conn->pDelayStartRev, sizeof(int), length, f);
	fwrite(conn->pDelayNumRev, sizeof(int), length, f);
	fwrite(conn->pSidMapRev, sizeof(int), conn->sNum, f);

	return 0;
}

Connection * loadConnection(FILE *f)
{
	Connection *conn = (Connection *)malloc(sizeof(Connection));

	fread(&(conn->nNum), sizeof(int), 1, f);
	fread(&(conn->sNum), sizeof(int), 1, f);
	fread(&(conn->maxDelay), sizeof(int), 1, f);
	fread(&(conn->minDelay), sizeof(int), 1, f);

	int length = (conn->maxDelay - conn->minDelay + 1) * conn->nNum;

	conn->pDelayStart = (int*)malloc(sizeof(int)*length);
	conn->pDelayNum = (int*)malloc(sizeof(int)*length);

	conn->pDelayStartRev = (int*)malloc(sizeof(int)*length);
	conn->pDelayNumRev = (int*)malloc(sizeof(int)*length);
	conn->pSidMapRev = (int*)malloc(sizeof(int)*conn->sNum);

	fread(conn->pDelayStart, sizeof(int), length, f);
	fread(conn->pDelayNum, sizeof(int), length, f);

	fread(conn->pDelayStartRev, sizeof(int), length, f);
	fread(conn->pDelayNumRev, sizeof(int), length, f);
	fread(conn->pSidMapRev, sizeof(int), conn->sNum, f);

	return conn;
}


bool isEqualConnection(Connection *c1, Connection *c2)
{
	bool ret = true;
	ret = ret && (c1->nNum == c2->nNum);
	ret = ret && (c1->sNum == c2->sNum);
	ret = ret && (c1->maxDelay == c2->maxDelay);
	ret = ret && (c1->minDelay == c2->minDelay);

	int length = (c1->maxDelay - c1->minDelay + 1) * c1->nNum;

	ret = ret && isEqualArray(c1->pDelayStart, c2->pDelayStart, length);
	ret = ret && isEqualArray(c1->pDelayNum, c2->pDelayNum, length);

	ret = ret && isEqualArray(c1->pDelayStartRev, c2->pDelayStartRev, length);
	ret = ret && isEqualArray(c1->pDelayNumRev, c2->pDelayNumRev, length);

	ret = ret && isEqualArray(c1->pSidMapRev, c2->pSidMapRev, c1->sNum);

	return ret;
}


int sendConnection(Connection *conn, int dest, int tag, MPI_Comm comm)
{
	int ret = 0;
	ret = MPI_Send(conn, sizeof(Connection), MPI_UNSIGNED_CHAR, dest, tag, comm);
	assert(ret == MPI_SUCCESS);

	int length = (conn->maxDelay - conn->minDelay + 1) * conn->nNum;

	ret = MPI_Send(conn->pDelayStart, length, MPI_INT, dest+1, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(conn->pDelayNum, length, MPI_INT, dest+2, tag, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(conn->pDelayStartRev, length, MPI_INT, dest+3, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(conn->pDelayNumRev, length, MPI_INT, dest+4, tag, comm);
	assert(ret == MPI_SUCCESS);

	ret = MPI_Send(conn->pSidMapRev, conn->sNum, MPI_INT, dest+5, tag, comm);
	assert(ret == MPI_SUCCESS);
	
	return ret;
}

Connection *recvConnection(int src, int tag, MPI_Comm comm)
{
	Connection *ret = (Connection *)malloc(sizeof(Connection));
	MPI_Status status;
	MPI_Recv(ret, sizeof(Connection), MPI_UNSIGNED_CHAR, src, tag, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);

	int length = (ret->maxDelay - ret->minDelay + 1) * (ret->nNum);

	ret->pDelayStart = (int *)malloc(sizeof(int) * length);
	ret->pDelayNum = (int *)malloc(sizeof(int) * length);

	MPI_Recv(ret->pDelayStart, length, MPI_INT, src, tag+1, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);
	MPI_Recv(ret->pDelayNum, length, MPI_INT, src, tag+2, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);

	ret->pDelayStartRev = (int *)malloc(sizeof(int) * length);
	ret->pDelayNumRev = (int *)malloc(sizeof(int) * length);

	MPI_Recv(ret->pDelayStartRev, length, MPI_INT, src, tag+3, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);
	MPI_Recv(ret->pDelayNumRev, length, MPI_INT, src, tag+4, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);

	ret->pSidMapRev = (int *)malloc(sizeof(int) * (ret->sNum));

	MPI_Recv(ret->pSidMapRev, ret->sNum, MPI_INT, src, tag+5, comm, &status);
	assert(status.MPI_ERROR==MPI_SUCCESS);

	return ret;
}
