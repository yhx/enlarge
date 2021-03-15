
#include <stdlib.h>
#include <assert.h>

#include "../utils/utils.h"
#include "CrossNodeMap.h"

CrossNodeMap * allocCNM(size_t num, size_t cross_num, unsigned node_num)
{
	CrossNodeMap* ret = (CrossNodeMap*)malloc(sizeof(CrossNodeMap));
	assert(ret != NULL);
	ret->_num = num;

	ret->_idx2index = (size_t*)malloc(sizeof(size_t)*num);
	assert(ret->_idx2index != NULL);
	std::fill(ret->_idx2index, ret->_idx2index + num, SIZE_MAX);

	if (cross_num > 0) {
		ret->_crossnodeIndex2idx = (size_t*)malloc(sizeof(size_t) * cross_num * node_num);
	} else {
		ret->_crossnodeIndex2idx = NULL;
	}
	ret->_crossSize = node_num * cross_num;

	return ret;
}

int sendMap(CrossNodeMap *map_, int dest, int tag, MPI_Comm comm) 
{
	int ret = 0;
	ret = MPI_Send(map_, sizeof(CrossNodeMap), MPI_UNSIGNED_CHAR, dest, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(map_->_idx2index, map_->_num, MPI_SIZE_T, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(map_->_crossnodeIndex2idx, map_->_crossSize, MPI_SIZE_T, dest, tag+2, comm);
	assert(ret == MPI_SUCCESS);
	return ret;
}

CrossNodeMap * recvMap(int src, int tag, MPI_Comm comm)
{
	CrossNodeMap *net = (CrossNodeMap *)malloc(sizeof(CrossNodeMap));
	int ret = 0;
	MPI_Status status;
	ret = MPI_Recv(net, sizeof(CrossNodeMap), MPI_UNSIGNED_CHAR, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);
	net->_idx2index = (size_t *)malloc(sizeof(size_t)*net->_num);
	ret = MPI_Recv(net->_idx2index, net->_num, MPI_SIZE_T, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
	net->_crossnodeIndex2idx = (size_t *)malloc(sizeof(size_t)*(net->_crossSize));
	ret = MPI_Recv(net->_crossnodeIndex2idx, net->_crossSize, MPI_SIZE_T, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);

	return net;
}

int saveCNM(CrossNodeMap *map, FILE *f)
{
	fwrite(&(map->_num), sizeof(size_t), 1, f);
	fwrite(&(map->_crossSize), sizeof(size_t), 1, f);
	fwrite(map->_idx2index, sizeof(size_t), map->_num, f);
	fwrite(map->_crossnodeIndex2idx, sizeof(size_t), map->_crossSize, f);

	return 0;
}

CrossNodeMap *loadCNM(FILE *f)
{
	CrossNodeMap *map = (CrossNodeMap*)malloc(sizeof(CrossNodeMap));
	fread(&(map->_num), sizeof(size_t), 1, f);
	fread(&(map->_crossSize), sizeof(size_t), 1, f);
	map->_idx2index = (size_t*)malloc(sizeof(size_t) * map->_num);
	map->_crossnodeIndex2idx = (size_t*)malloc(sizeof(size_t) * map->_crossSize);
	fwrite(map->_idx2index, sizeof(size_t), map->_num, f);
	fwrite(map->_crossnodeIndex2idx, sizeof(size_t), map->_crossSize, f);

	return map;
}

int compareCNM(CrossNodeMap *m1, CrossNodeMap *m2)
{
	bool equal = true;
	equal = (m1->_num == m2->_num) && equal;
	equal = (m1->_crossSize== m2->_crossSize) && equal;
	equal = compareArray(m1->_idx2index, m2->_idx2index, m1->_num) && equal;
	equal = compareArray(m1->_crossnodeIndex2idx, m2->_crossnodeIndex2idx, m1->_crossSize) && equal;
	
	return equal;
}

