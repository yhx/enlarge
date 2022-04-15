
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../../utils/utils.h"
#include "../../../msg_utils/helper/helper_c.h"


#include "IAFData.h"

size_t getIAFSize()
{
	return sizeof(IAFData);
}

void *mallocIAF()
{
	IAFData *p = (IAFData*)malloc(sizeof(IAFData)*1);
	memset(p, 0, sizeof(IAFData)*1);
	return (void*)p;
}

int allocIAFPara(void *pCPU, size_t num)
{
	IAFData *p = (IAFData*)pCPU;

	p->num = num;

	p->pRefracTime = (int*)malloc(num*sizeof(int));
	memset(p->pRefracTime, 0, num*sizeof(int));
	p->pRefracStep = (int*)malloc(num*sizeof(int));
	memset(p->pRefracStep, 0, num*sizeof(int));

	// model param
	// p->pTau = (real*)malloc(num*sizeof(real));
	// memset(p->pTau, 0, num*sizeof(real));
	// p->pC = (real*)malloc(num*sizeof(real));
	// memset(p->pC, 0, num*sizeof(real));
	// p->ppt_refC = (real*)malloc(num*sizeof(real));
	// memset(p->pt_ref, 0, num*sizeof(real));
	p->pE_L = (real*)malloc(num*sizeof(real));
	memset(p->pE_L, 0, num*sizeof(real));
	p->pI_e = (real*)malloc(num*sizeof(real));
	memset(p->pI_e, 0, num*sizeof(real));
	p->pTheta = (real*)malloc(num*sizeof(real));
	memset(p->pTheta, 0, num*sizeof(real));
	p->pV_reset = (real*)malloc(num*sizeof(real));
	memset(p->pV_reset, 0, num*sizeof(real));
	// p->ptau_ex = (real*)malloc(num*sizeof(real));
	// memset(p->ptau_ex, 0, num*sizeof(real));
	// p->ptau_in = (real*)malloc(num*sizeof(real));
	// memset(p->ptau_in, 0, num*sizeof(real));
	// p->prho = (real*)malloc(num*sizeof(real));
	// memset(p->prho, 0, num*sizeof(real));
	// p->pdelta = (real*)malloc(num*sizeof(real));
	// memset(p->pdelta, 0, num*sizeof(real));

	// state param
	p->pi_0 = (real*)malloc(num*sizeof(real));
	memset(p->pi_0, 0, num*sizeof(real));
	p->pi_1 = (real*)malloc(num*sizeof(real));
	memset(p->pi_1, 0, num*sizeof(real));
	p->pi_syn_ex = (real*)malloc(num*sizeof(real));
	memset(p->pi_syn_ex, 0, num*sizeof(real));
	p->pi_syn_in = (real*)malloc(num*sizeof(real));
	memset(p->pi_syn_in, 0, num*sizeof(real));
	p->pV_m = (real*)malloc(num*sizeof(real));
	memset(p->pV_m, 0, num*sizeof(real));

	// internal param
	p->pP11ex = (real*)malloc(num*sizeof(real));
	memset(p->pP11ex, 0, num*sizeof(real));
	p->pP11in = (real*)malloc(num*sizeof(real));
	memset(p->pP11in, 0, num*sizeof(real));
	p->pP22 = (real*)malloc(num*sizeof(real));
	memset(p->pP22, 0, num*sizeof(real));
	p->pP21ex = (real*)malloc(num*sizeof(real));
	memset(p->pP21ex, 0, num*sizeof(real));
	p->pP21in = (real*)malloc(num*sizeof(real));
	memset(p->pP21in, 0, num*sizeof(real));
	p->pP20 = (real*)malloc(num*sizeof(real));
	memset(p->pP20, 0, num*sizeof(real));

	p->_fire_count = malloc_c<int>(num);
	
	p->is_view = false;

	return 0;
}

void *allocIAF(size_t num)
{
	assert(num > 0);
	void *p = mallocIAF();
	allocIAFPara(p, num);
	return p;
}

int freeIAFPara(void *pCPU)
{
	IAFData *p = (IAFData*)pCPU;

	p->num = 0;

	if (!p->is_view) {
		free(p->pRefracTime);
		p->pRefracTime = NULL;
		free(p->pRefracStep);
		p->pRefracStep = NULL;

		// model param
		// free(p->pTau);
		// p->pTau = NULL;
		// free(p->pC);
		// p->pC = NULL;
		// free(p->pt_ref);
		// p->pt_ref = NULL;
		free(p->pE_L);
		p->pE_L = NULL;
		free(p->pI_e);
		p->pI_e = NULL;
		free(p->pTheta);
		p->pTheta = NULL;
		free(p->pV_reset);
		p->pV_reset = NULL;
		// free(p->ptau_ex);
		// p->ptau_ex = NULL;
		// free(p->ptau_in);
		// p->ptau_in = NULL;
		// free(p->prho);
		// p->prho = NULL;
		// free(p->pdelta);
		// p->pdelta = NULL;

		// state param
		free(p->pi_0);
		p->pi_0 = NULL;
		free(p->pi_1);
		p->pi_1 = NULL;
		free(p->pi_syn_ex);
		p->pi_syn_ex = NULL;
		free(p->pi_syn_in);
		p->pi_syn_in = NULL;
		free(p->pV_m);
		p->pV_m = NULL;

		// internal param
		free(p->pP11ex);
		p->pP11ex = NULL;
		free(p->pP11in);
		p->pP11in = NULL;
		free(p->pP22);
		p->pP22 = NULL;
		free(p->pP21ex);
		p->pP21ex = NULL;
		free(p->pP21in);
		p->pP21in = NULL;
		free(p->pP20);
		p->pP20 = NULL;
	}
	free_c(p->_fire_count);

	return 0;
}

int freeIAF(void *pCPU)
{
	IAFData *p = (IAFData*)pCPU;

	freeIAFPara(p);
	free(p);
	p = NULL;
	return 0;
}

int saveIAF(void *pCPU, size_t num, const string &path)
{
	string name = path + "/iaf.neuron";
	FILE *f = fopen(name.c_str(), "w");

	IAFData *p = (IAFData*)pCPU;
	assert(num <= p->num);
	if (num <= 0)
		num = p->num;

	fwrite(&num, sizeof(size_t), 1, f);

	fwrite(p->pRefracTime, sizeof(int), num, f);
	fwrite(p->pRefracStep, sizeof(int), num, f);

	// fwrite(p->pTau, sizeof(real), num, f);
	// fwrite(p->pC, sizeof(real), num, f);
	// fwrite(p->pt_ref, sizeof(real), num, f);
	fwrite(p->pE_L, sizeof(real), num, f);
	fwrite(p->pI_e, sizeof(real), num, f);
	fwrite(p->pTheta, sizeof(real), num, f);
	fwrite(p->pV_reset, sizeof(real), num, f);
	// fwrite(p->ptau_ex, sizeof(real), num, f);
	// fwrite(p->ptau_in, sizeof(real), num, f);
	// fwrite(p->prho, sizeof(real), num, f);
	// fwrite(p->pdelta, sizeof(real), num, f);

	fwrite(p->pi_0, sizeof(real), num, f);
	fwrite(p->pi_1, sizeof(real), num, f);
	fwrite(p->pi_syn_ex, sizeof(real), num, f);
	fwrite(p->pi_syn_in, sizeof(real), num, f);
	fwrite(p->pV_m, sizeof(real), num, f);

	fwrite(p->pP11ex, sizeof(real), num, f);
	fwrite(p->pP11in, sizeof(real), num, f);
	fwrite(p->pP22, sizeof(real), num, f);
	fwrite(p->pP21ex, sizeof(real), num, f);
	fwrite(p->pP21in, sizeof(real), num, f);
	fwrite(p->pP20, sizeof(real), num, f);
	
	fwrite_c(p->_fire_count, num, f);

	fclose_c(f);

	return 0;
}

void *loadIAF(size_t num, const string &path)
{
	string name = path + "/iaf.neuron";
	FILE *f = fopen(name.c_str(), "r");

	IAFData *p = (IAFData*)allocIAF(num);

	fread_c(&num, 1, f);

	assert(num == p->num);

	fread(p->pRefracTime, sizeof(int), num, f);
	fread(p->pRefracStep, sizeof(int), num, f);

	// fread(p->pTau, sizeof(real), num, f);
	// fread(p->pC, sizeof(real), num, f);
	// fread(p->pt_ref, sizeof(real), num, f);
	fread(p->pE_L, sizeof(real), num, f);
	fread(p->pI_e, sizeof(real), num, f);
	fread(p->pTheta, sizeof(real), num, f);
	fread(p->pV_reset, sizeof(real), num, f);
	// fread(p->ptau_ex, sizeof(real), num, f);
	// fread(p->ptau_in, sizeof(real), num, f);
	// fread(p->prho, sizeof(real), num, f);
	// fread(p->pdelta, sizeof(real), num, f);

	fread(p->pi_0, sizeof(real), num, f);
	fread(p->pi_1, sizeof(real), num, f);
	fread(p->pi_syn_ex, sizeof(real), num, f);
	fread(p->pi_syn_in, sizeof(real), num, f);
	fread(p->pV_m, sizeof(real), num, f);

	fread(p->pP11ex, sizeof(real), num, f);
	fread(p->pP11in, sizeof(real), num, f);
	fread(p->pP22, sizeof(real), num, f);
	fread(p->pP21ex, sizeof(real), num, f);
	fread(p->pP21in, sizeof(real), num, f);
	fread(p->pP20, sizeof(real), num, f);
	
	fread_c(p->_fire_count, num, f);

	fclose_c(f);

	return p;
}

bool isEqualIAF(void *p1, void *p2, size_t num, uinteger_t *shuffle1, uinteger_t *shuffle2)
{
	IAFData *t1 = (IAFData*)p1;
	IAFData *t2 = (IAFData*)p2;

	bool ret = t1->num == t2->num;
	ret = ret && isEqualArray(t1->pRefracTime, t2->pRefracTime, num, shuffle1, shuffle2);
	ret = ret && isEqualArray(t1->pRefracStep, t2->pRefracStep, num, shuffle1, shuffle2);

	// model param
	// ret = ret && isEqualArray(t1->pTau, t2->pTau, num, shuffle1, shuffle2);
	// ret = ret && isEqualArray(t1->pC, t2->pC, num, shuffle1, shuffle2);
	// ret = ret && isEqualArray(t1->pt_ref, t2->pt_ref, num, shuffle1, shuffle2);
	ret = ret && isEqualArray(t1->pE_L, t2->pE_L, num, shuffle1, shuffle2);
	ret = ret && isEqualArray(t1->pI_e, t2->pI_e, num, shuffle1, shuffle2);
	ret = ret && isEqualArray(t1->pTheta, t2->pTheta, num, shuffle1, shuffle2);
	ret = ret && isEqualArray(t1->pV_reset, t2->pV_reset, num, shuffle1, shuffle2);
	// ret = ret && isEqualArray(t1->ptau_ex, t2->ptau_ex, num, shuffle1, shuffle2);
	// ret = ret && isEqualArray(t1->ptau_in, t2->ptau_in, num, shuffle1, shuffle2);
	// ret = ret && isEqualArray(t1->prho, t2->prho, num, shuffle1, shuffle2);
	// ret = ret && isEqualArray(t1->pdelta, t2->pdelta, num, shuffle1, shuffle2);

	// state param
	ret = ret && isEqualArray(t1->pi_0, t2->pi_0, num, shuffle1, shuffle2);
	ret = ret && isEqualArray(t1->pi_1, t2->pi_1, num, shuffle1, shuffle2);
	ret = ret && isEqualArray(t1->pi_syn_ex, t2->pi_syn_ex, num, shuffle1, shuffle2);
	ret = ret && isEqualArray(t1->pi_syn_in, t2->pi_syn_in, num, shuffle1, shuffle2);
	ret = ret && isEqualArray(t1->pV_m, t2->pV_m, num, shuffle1, shuffle2);

	// internal param
	ret = ret && isEqualArray(t1->pP11ex, t2->pP11ex, num, shuffle1, shuffle2);
	ret = ret && isEqualArray(t1->pP11in, t2->pP11in, num, shuffle1, shuffle2);
	ret = ret && isEqualArray(t1->pP22, t2->pP22, num, shuffle1, shuffle2);
	ret = ret && isEqualArray(t1->pP21ex, t2->pP21ex, num, shuffle1, shuffle2);
	ret = ret && isEqualArray(t1->pP21in, t2->pP21in, num, shuffle1, shuffle2);
	ret = ret && isEqualArray(t1->pP20, t2->pP20, num, shuffle1, shuffle2);

	return ret;
}

int copyIAF(void *p_src, size_t s_off, void *p_dst, size_t d_off) 
{
	IAFData *src = static_cast<IAFData *>(p_src);
	IAFData *dst = static_cast<IAFData *>(p_dst);

	dst->pRefracTime[d_off] = src->pRefracTime[s_off];
	return 0;
}

int logRateIAF(void *data, const char *name)
{
	char filename[512];
	sprintf(filename, "rate_%s.%s.log", name, "IAF");
	FILE *f = fopen_c(filename, "w+");
	IAFData *d = static_cast<IAFData*>(data);
	log_array(f, d->_fire_count, d->num);
	fclose_c(f);
	return 0;
}

real * getVIAF(void *data) {
	IAFData * p = static_cast<IAFData *>(data);
	return p->pV_m;
}
