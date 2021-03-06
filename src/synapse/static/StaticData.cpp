#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../../utils/utils.h"
#include "../../../msg_utils/helper/helper_c.h"


#include "StaticData.h"

size_t getStaticSize()
{
	return sizeof(StaticData);
}

void *mallocStatic()
{
	StaticData *p = (StaticData*)malloc(sizeof(StaticData)*1);
	memset(p, 0, sizeof(StaticData)*1);
	return (void*)p;
}

int allocStaticPara(void *pCPU, size_t num)
{
	StaticData *p = (StaticData*)pCPU;

	p->num = num;
	p->pWeight = malloc_c<real>(num);
	p->is_view = false;

	return 0;
}

void *allocStatic(size_t num)
{
	assert(num > 0);
	void *p = mallocStatic();
	allocStaticPara(p, num);
	return p;
}

int freeStaticPara(void *pCPU)
{
	StaticData *p = (StaticData*)pCPU;

	// free(p->pDst);
	// p->pDst = NULL;

	if (!p->is_view) {
		free(p->pWeight);
		p->pWeight = NULL;
	}

	return 0;
}

int freeStatic(void *pCPU)
{
	StaticData *p = (StaticData*)pCPU;

	freeStaticPara(p);
	free(p);
	p = NULL;
	return 0;
}

int saveStatic(void *pCPU, size_t num, const string &path)
{
	string name = path + "/static.synapse";
	FILE *f = fopen_c(name.c_str(), "w");

	StaticData *p = (StaticData*)pCPU;
	assert(num <= p->num);
	if (num <= 0) {
		num = p->num;
	}

	fwrite_c(&(num), 1, f);
	// fwrite(p->pDst, sizeof(int), num, f);
	fwrite_c(p->pWeight, num, f);

	fclose_c(f);

	return 0;
}

void *loadStatic(size_t num, const string &path)
{
	string name = path + "/static.synapse";
	FILE *f = fopen_c(name.c_str(), "r");

	StaticData *p = (StaticData*)allocStatic(num);


	fread_c(&(p->num), 1, f);
	assert(num == p->num);

	// fread(p->pDst, sizeof(int), num, f);
	fread_c(p->pWeight, num, f);

	fclose_c(f);

	return p;
}

bool isEqualStatic(void *p1, void *p2, size_t num, uinteger_t *shuffle1, uinteger_t *shuffle2)
{
	StaticData *t1 = (StaticData*)p1;
	StaticData *t2 = (StaticData*)p2;

	bool ret = true;
	// ret = ret && isEqualArray(t1->pDst, t2->pDst, num);

	ret = ret && isEqualArray(t1->pWeight, t2->pWeight, num, shuffle1, shuffle2);

	return ret;
}

int shuffleStatic(void *p, uinteger_t *shuffle, size_t num)
{
	StaticData *d = static_cast<StaticData *>(p);
	assert(num == d->num);

	real *tmp = malloc_c<real>(d->num);
	memcpy_c(tmp, d->pWeight, d->num);

	for (size_t i=0; i<num; i++) {
		d->pWeight[i] = tmp[shuffle[i]];
	}

	return num;
}


