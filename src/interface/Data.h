#ifndef DATA_H
#define DATA_H

#include "../base/constant.h"

class Data {
public:
	Data();
	Data(size_t num);
	~Data();

	virtual int save(FILE *f, size_t num) = 0;
	virtual int load(FILE *f) = 0;

	virtual int send(int dest, int tag, MPI_Comm comm=MPI_COMM_WORLD, int offset=0, size_t num=0) = 0;
	virtual int recv(int dest, int tag, MPI_Comm comm=MPI_COMM_WORLD, int offset=0) = 0;

	virtual int to_gpu() = 0;
	virtual int fetch() = 0;

	virtual bool is_equal(Data *p, size_t *shuffle1=NULL, size_t *shuffle2=NULL) = 0;

	bool _is_view;
	size_t _num;

	Data * _gpu;

protected:
	int alloc(size_t num);
};

#endif // DATA_H
