#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../../utils/utils.h"
#include "../../../msg_utils/helper/helper_c.h"


#include "PoissonData.h"

size_t getPoissonSize()
{
	return sizeof(PoissonData);
}

void *mallocPoisson()
{
	PoissonData *p = (PoissonData*)malloc(sizeof(PoissonData)*1);
	memset(p, 0, sizeof(PoissonData)*1);
	return (void*)p;
}

int allocPoissonPara(void *pCPU, size_t num)
{
	PoissonData *p = (PoissonData*)pCPU;

	p->num = num;
	p->pWeight = malloc_c<real>(num);
	p->is_view = false;

	return 0;
}

void *allocPoisson(size_t num)
{
	assert(num > 0);
	void *p = mallocPoisson();
	allocPoissonPara(p, num);
	return p;
}

int freePoissonPara(void *pCPU)
{
	PoissonData *p = (PoissonData*)pCPU;

	// free(p->pDst);
	// p->pDst = NULL;

	if (!p->is_view) {
		free(p->pWeight);
		p->pWeight = NULL;
	}

	return 0;
}

int freePoisson(void *pCPU)
{
	PoissonData *p = (PoissonData*)pCPU;

	freePoissonPara(p);
	free(p);
	p = NULL;
	return 0;
}

int savePoisson(void *pCPU, size_t num, const string &path)
{
	string name = path + "/poisson.synapse";
	FILE *f = fopen_c(name.c_str(), "w");

	PoissonData *p = (PoissonData*)pCPU;
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

void *loadPoisson(size_t num, const string &path)
{
	string name = path + "/poisson.synapse";
	FILE *f = fopen_c(name.c_str(), "r");

	PoissonData *p = (PoissonData*)allocPoisson(num);


	fread_c(&(p->num), 1, f);
	assert(num == p->num);

	// fread(p->pDst, sizeof(int), num, f);
	fread_c(p->pWeight, num, f);

	fclose_c(f);

	return p;
}

bool isEqualPoisson(void *p1, void *p2, size_t num, uinteger_t *shuffle1, uinteger_t *shuffle2)
{
	PoissonData *t1 = (PoissonData*)p1;
	PoissonData *t2 = (PoissonData*)p2;

	bool ret = true;
	// ret = ret && isEqualArray(t1->pDst, t2->pDst, num);

	ret = ret && isEqualArray(t1->pWeight, t2->pWeight, num, shuffle1, shuffle2);

	return ret;
}

int shufflePoisson(void *p, uinteger_t *shuffle, size_t num)
{
	PoissonData *d = static_cast<PoissonData *>(p);
	assert(num == d->num);

	real *tmp = malloc_c<real>(d->num);
	memcpy_c(tmp, d->pWeight, d->num);

	for (size_t i=0; i<num; i++) {
		d->pWeight[i] = tmp[shuffle[i]];
	}

	return num;
}


