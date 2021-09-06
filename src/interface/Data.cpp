
#include <assert.h>

#include "../../msg_utils/helper/helper_c.h"
#include "Data.h"

Data::Data() {
	_is_view = false;
	_num = 0;

	// _data = NULL;

	_gpu = NULL;
}


Data::Data(size_t num) {
	alloc(num);
}

Data::~Data() {
	if (_num > 0 && !_is_view) {
		// delete [] _data;
	}
#ifdef USE_GPU
	if (_gpu) {
		//freeGPU(_gpu>_data);
		_gpu->_num = 0;
		_gpu = NULL;
	}
#else
	assert(!_gpu);
#endif
}

int Data::alloc(size_t num)
{
	_is_view = false;
	_num = num;

	// _data = new uinteger_t[num]();

	_gpu = NULL;

	return 0;
}


int Data::save(FILE *f, size_t num)
{
	if (num <=0 || num > _num) {
		num = _num;
	}

	fwrite_c(&num, 1, f);

	// fwrite_c(&_data, _num, f);

	return 0;
}

int Data::load(FILE *f)
{
	size_t num;
	fread_c(&num, 1, f);
	if (_num != 0) {
		printf("Reconstruct Data is not supported!\n");
		return -1;
	}

	alloc(num);

	// fread_c(_data, num, f);
	
	return 0;
}

#ifdef USE_MPI
int Data::send(int dest, int tag, MPI_Comm comm, int offset, size_t num) 
{
	if (offset >= _num || offset + num > _num) {
		printf("Wrong offset %d and num %d\n", offset, num);
		return -1;
	}

	if (num <= 0) {
		num = _num - offset;
	}

	ret = MPI_Send(&(num), 1, MPI_SIZE_T, dest, tag, comm);
	assert(ret == MPI_SUCCESS);
	ret = MPI_Send(_data + offset, num, MPI_UINTETER_T, dest, tag+1, comm);
	assert(ret == MPI_SUCCESS);

	return 0;
}

int Data::recv(int dest, int tag, MPI_Comm comm, int offset) 
{
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

	ret = MPI_Recv(_data + offset, num, MPI_UINTEGER_T, src, tag+1, comm, &status);
	assert(ret==MPI_SUCCESS);

	return 0;
}
#else
int Data::send(int dest, int tag, MPI_Comm comm, int offset, size_t num) 
{
	printf("MPI not enabled!\n");
	return -1;
}

int Data::recv(int dest, int tag, MPI_Comm comm, int offset) 
{
	printf("MPI not enabled!\n");
	return -1;
}

#endif

#ifndef USE_GPU
int Data::to_gpu() 
{
	printf("GPU not enabled!\n");
	return -1;
}
int Data::fetch() 
{
	printf("GPU not enabled!\n");
	return -1;
}
#endif

bool Data::is_equal(Data *p, size_t *shuffle1, size_t *shuffle2) 
{
	Data *d = dynamic_cast<Data *>(p);
	bool ret = _num == d->_num;

	// ret = ret && isEqualArray(_data, d->_data, _num, shuffle1, shuffle2);

	return ret;
}

