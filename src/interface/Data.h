#ifndef DATA_H
#define DATA_H

#include "../base/constant.h"

class Data : Father {
public:
	Data();
	Data(size_t num);
	~Data();

	virtual int save() override;
	virtual int load() override;

	virtual int send(int dest, int tag, MPI_Comm comm=MPI_COMM_WORLD, int offset=0, size_t num=0) override;
	virtual int recv(int dest, int tag, MPI_Comm comm=MPI_COMM_WORLD, int offset=0) override;

	virtual int to_gpu() override;
	virtual int fetch() override;

	template<typename T>
	virtual bool is_equal(Father *p, T *shuffle1=NULL, T *shuffle2=NULL) override;

	bool _is_view;
	size_t _num;
	
	uinteger_t *_data;

	Data * _gpu;

protected:
	int alloc(size_t num);

};

#endif // DATA_H
