#include <assert.h>
#include "../utils/helper_c.h"
#include "CrossMap.h"


CrossMap::CrossMap()
{
	_is_view = false;
	_num = 0;
	_crossSize = 0;


	_gpu_array = NULL;
	_crossnodeIndex2idx = NULL;
	_idx2index = NULL;


}


CrossMap::CrossMap(size_t num) {
	alloc(num);
}
CrossMap::~CrossMap() {
	if (_num > 0 && !_is_view) {
		delete [] _crossSize;
		delete [] _num;


		delete [] _crossnodeIndex2idx;
		delete [] _idx2index;


	}
#ifdef USE_GPU
	free_gpu();
#else
	assert(!_gpu);
#endif
}


void CrossMap::alloc(size_t num)
{
	_is_view = false;
	_num = num;
	_crossSize = new uinteger_t[num]();
	_num = new uinteger_t[num]();


	_crossnodeIndex2idx = new uinteger_t[num]();
	_idx2index = new uinteger_t[num]();


	_gpu_array = NULL;
}


int CrossMap::save(FILE *f, size_t num, size_t offset=0)
{
	if (num <=0 || num > _num) {
		num = _num;
	}
	fwrite_c(&num, 1, f);
	fwrite_c(&_crossSize, num, f);


	fwrite_c(&_crossnodeIndex2idx+offset, num, f);
	fwrite_c(&_idx2index+offset, num, f);


	return 0;
}


int CrossMap::load(FILE *f)
{
	size_t num;
	fread_c(&num, 1, f);
	if (_num != 0) {
		printf("Reconstruct Data is not supported!\n");
		return -1;
	}
	alloc(num);
	fread_c(&_crossSize, 1, f);


	fread_c(_crossnodeIndex2idx, num, f);
	fread_c(_idx2index, num, f);


}


#ifdef USE_MPI
int CrossMap::send(int dest, int tag, MPI_Comm comm, int offset, size_t num) {
	if (offset >= _num || offset + num > _num) {
		printf("Wrong offset %d and num %d\n", offset, num);
		return -1;
	}
	if (num <= 0) {
		num = _num - offset;
	}
	ret = MPI_Send(&(num), 1, MPI_SIZE_T, dest, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(&(_crossSize), 1, MPI_SIZE_T, dest, tag+1, comm);


	ret = MPI_Send(_crossnodeIndex2idx + offset, num, MPI_INTEGER_T, dest, tag+3, comm);
	ret = MPI_Send(_idx2index + offset, num, MPI_INTEGER_T, dest, tag+4, comm);


	assert(ret == MPI_SUCCESS);
	return 0;
}


int CrossMap::recv(int dest, int tag, MPI_Comm comm=MPI_COMM_WORLD, int offset) {
	size_t num = 0;
	MPI_Status status;
	ret = MPI_Recv(&(_num), 1, MPI_SIZE_T, src, tag, comm, &status);
	assert(ret==MPI_SUCCESS);
	if (_num == 0) {
		alloc(offset + num);
	}
	if (_num < offset + num) {
		printf("Wrong offset %d and num %d\n", offset, num);
		return -1;
	}
	ret = MPI_Recv(&(_crossSize), num, MPI_SIZE_T, src, tag+1, comm, &status);
	ret = MPI_Recv(_num + offset, num, MPI_SIZE_T, src, tag+2, comm, &status);


	ret = MPI_Recv(_crossnodeIndex2idx + offset, num, MPI_INTEGER_T, src, tag+3, comm, &status);
	ret = MPI_Recv(_idx2index + offset, num, MPI_INTEGER_T *, src, tag+4, comm, &status);


	assert(ret==MPI_SUCCESS);
	return 0;
}


#else
int CrossMap::send() 
{
	printf("MPI not enabled!\n");
	return -1;
}
int CrossMap::recv() 
{
	printf("MPI not enabled!\n");
	return -1;
}
#endif


bool CrossMap::is_equal(Data *p, size_t *shuffle1, size_t *shuffle2) 
{
	Data *d = dynamic_cast<Data *>(p)
	bool ret = _num == d.num;
	ret = ret && isEqualArray(_crossSize, d->_crossSize, _num, shuffle1, shuffle2);
	ret = ret && isEqualArray(_num, d->_num, _num, shuffle1, shuffle2);


	ret = ret && isEqualArray(_crossnodeIndex2idx, d->_crossnodeIndex2idx, _num, shuffle1, shuffle2);
	ret = ret && isEqualArray(_idx2index, d->_idx2index, _num, shuffle1, shuffle2);


	return ret;
}
#ifndef USE_GPU
void CrossMap::free_gpu
{
	printf("GPU not enabled!\n");
}
int CrossMap::to_gpu()
{
	printf("GPU not enabled!\n");
	return -1;
}
int CrossMap::from_gpu()
{
	printf("GPU not enabled!\n");
	return -1;
}
#endif
