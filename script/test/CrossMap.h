#ifndef CROSSMAP_H
#define CROSSMAP_H
#include "../base/constant.h"


class CrossMap::Data {
public:
	CrossMap();
	CrossMap(size_t num);
	~CrossMap();


	virtual int save() override;
	virtual int load() override;


	virtual int send(int dest, int tag, MPI_Comm comm=MPI_COMM_WORLD, int offset=0, size_t num=0) override;
	virtual int recv(int dest, int tag, MPI_Comm comm=MPI_COMM_WORLD, int offset=0) override;


	virtual int to_gpu() override;
	virtual int fetch() override;
	template<typename T>
	virtual bool is_equal(Data *p, T *shuffle1=NULL, T *shuffle2=NULL) override;
	bool _is_view;
	size_t _num;

    size_t  _crossSize;
	size_t  _num;

	integer_t *_crossnodeIndex2idx;
	uinteger_t *_idx2index;


	CrossMap *_gpu_aray;
protected:
	int alloc(size_t num);
	void free_gpu();
};
#endif
