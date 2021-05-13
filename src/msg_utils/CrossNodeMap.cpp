
#include <stdlib.h>
#include <assert.h>

#include "../utils/utils.h"
#include "../utils/helper_c.h"
#include "CrossNodeMap.h"

CrossNodeMap * allocCNM(size_t num, size_t cross_num, unsigned node_num)
{
	return allocCNM(num, cross_num * node_num);
}

CrossNodeMap * allocCNM(size_t num, size_t cross_size)
{
	CrossNodeMap* ret = malloc_c<CrossNodeMap>(1);
	assert(ret != NULL);
	ret->_num = num;

	ret->_idx2index = malloc_c<integer_t>(num);
	std::fill(ret->_idx2index, ret->_idx2index + num, -1);

	ret->_crossSize = cross_size;
	if (ret->_crossSize > 0) {
		ret->_crossnodeIndex2idx = malloc_c<integer_t>(cross_size);
		std::fill(ret->_crossnodeIndex2idx, ret->_crossnodeIndex2idx + cross_size, -1);
	} else {
		ret->_crossnodeIndex2idx = NULL;
	}

	return ret;
}

int sendMap(CrossNodeMap *map_, int dest, int tag, MPI_Comm comm) 
{
	int ret = 0;
	ret = MPI_Send(map_, sizeof(CrossNodeMap), MPI_UNSIGNED_CHAR, dest, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(map_->_idx2index, map_->_num, MPI_INTEGER_T, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(map_->_crossnodeIndex2idx, map_->_crossSize, MPI_INTEGER_T, dest, tag+2, comm);
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
	net->_idx2index = malloc_c<integer_t>(net->_num);
	ret = MPI_Recv(net->_idx2index, net->_num, MPI_INTEGER_T, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);
	net->_crossnodeIndex2idx = malloc_c<integer_t>(net->_crossSize);
	ret = MPI_Recv(net->_crossnodeIndex2idx, net->_crossSize, MPI_INTEGER_T, src, tag+2, comm, &status);
	assert(ret==MPI_SUCCESS);

	return net;
}

int saveCNM(CrossNodeMap *map, const string &path)
{
	string name = path + "/cross.map";
	FILE *f = fopen_c(name.c_str(), "w");

	fwrite_c(&(map->_num), 1, f);
	fwrite_c(&(map->_crossSize), 1, f);
	fwrite_c(map->_idx2index, map->_num, f);
	if (map->_crossSize > 0) {
		fwrite_c(map->_crossnodeIndex2idx, map->_crossSize, f);
	}

	fclose_c(f);

	return 0;
}

CrossNodeMap *loadCNM(const string &path)
{
	string name = path + "/cross.map";
	FILE *f = fopen_c(name.c_str(), "r");

	size_t num = 0, cross_size = 0;
	fread_c(&(num), 1, f);
	fread_c(&(cross_size), 1, f);

	CrossNodeMap *map = allocCNM(num, cross_size);
	fread_c(map->_idx2index, map->_num, f);
	if (cross_size > 0) {
		fread_c(map->_crossnodeIndex2idx, map->_crossSize, f);
	}

	fclose_c(f);

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

