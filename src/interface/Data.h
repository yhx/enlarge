#ifndef DATA_H
#define DATA_H

#include "../base/constant.h"

class Data : Father {
public:
	Data();
	Data(size_t num);
	~Data();

	int save();
	int load();

	int send(int dest, int tag, MPI_Comm comm=MPI_COMM_WORLD, int offset=0, size_t num=0);
	int recv(int dest, int tag, MPI_Comm comm=MPI_COMM_WORLD, int offset=0);
	int recv();

	int to_gpu();
	int fetch();

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
