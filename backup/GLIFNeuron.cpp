#include <stdlib.h>
#include <string.h>

#include "GLIFNeuron.h"

size_t getLIFSize()
{
	return sizeof(GLIFNeuron);
}

void *mallocLIF()
{
	GLIFNeuron *p = (GLIFNeuron*)malloc(sizeof(GLIFNeuron)*1);
	memset(p, 0, sizeof(GLIFNeuron)*1);
	return (void*)p;
}

int allocLIFPara(void *pCPU, int num)
{
	GLIFNeuron *p = (GLIFNeuron*)pCPU;

	p->pI_e = (real*)malloc(num*sizeof(real));
	p->pI_i = (real*)malloc(num*sizeof(real));
	p->pV_e = (real*)malloc(num*sizeof(real));
	p->pV_reset = (real*)malloc(num*sizeof(real));
	p->pC_i = (real*)malloc(num*sizeof(real));
	p->pC_m = (real*)malloc(num*sizeof(real));
	p->pV_m = (real*)malloc(num*sizeof(real));
	p->pV_i = (real*)malloc(num*sizeof(real));
	p->pV_tmp = (real*)malloc(num*sizeof(real));
	p->pC_e = (real*)malloc(num*sizeof(real));
	p->pCe = (real*)malloc(num*sizeof(real));
	p->pCi = (real*)malloc(num*sizeof(real));
	p->pV_thresh = (real*)malloc(num*sizeof(real));

	p->pRefracTime = (int*)malloc(num*sizeof(int));
	p->pRefracStep = (int*)malloc(num*sizeof(int));

	return 0;
}

void *allocLIF(int num)
{
	void *p = mallocLIF();
	allocLIFPara(p, num);
	return p;
}

int freeLIFPara(void *pCPU)
{
	GLIFNeuron *p = (GLIFNeuron*)pCPU;

	free(p->pI_e);
	p->pI_e = NULL;
	free(p->pI_i);
	p->pI_i = NULL;
	free(p->pV_e);
	p->pV_e = NULL;
	free(p->pV_reset);
	p->pV_reset = NULL;
	free(p->pC_i);
	p->pC_i = NULL;
	free(p->pC_m);
	p->pC_m = NULL;
	free(p->pV_m);
	p->pV_m = NULL;
	free(p->pV_i);
	p->pV_i = NULL;
	free(p->pV_tmp);
	p->pV_tmp = NULL;
	free(p->pC_e);
	p->pC_e = NULL;
	free(p->pCe);
	p->pCe = NULL;
	free(p->pCi);
	p->pCi = NULL;
	free(p->pV_thresh);
	p->pV_thresh = NULL;

	free(p->pRefracTime);
	p->pRefracTime = NULL;
	free(p->pRefracStep);
	p->pRefracStep = NULL;

	return 0;
}

int freeLIF(void *pCPU)
{
	GLIFNeuron *p = (GLIFNeuron*)pCPU;

	freeLIFPara(p);
	free(p);
	p = NULL;
	return 0;
}

int saveLIF(void *pCPU, int num, FILE *f)
{

	GLIFNeuron *p = (GLIFNeuron*)pCPU;
	fwrite(p->pI_e, sizeof(real), num, f);
	fwrite(p->pI_i, sizeof(real), num, f);
	fwrite(p->pV_e, sizeof(real), num, f);
	fwrite(p->pV_reset, sizeof(real), num, f);
	fwrite(p->pC_i, sizeof(real), num, f);
	fwrite(p->pC_m, sizeof(real), num, f);
	fwrite(p->pV_m, sizeof(real), num, f);
	fwrite(p->pV_i, sizeof(real), num, f);
	fwrite(p->pV_tmp, sizeof(real), num, f);
	fwrite(p->pC_e, sizeof(real), num, f);
	fwrite(p->pCe, sizeof(real), num, f);
	fwrite(p->pCi, sizeof(real), num, f);
	fwrite(p->pV_thresh, sizeof(real), num, f);

	fwrite(p->pRefracTime, sizeof(int), num, f);
	fwrite(p->pRefracStep, sizeof(int), num, f);

	return 0;
}

void *loadLIF(int num, FILE *f)
{
	GLIFNeuron *p = (GLIFNeuron*)malloc(sizeof(GLIFNeuron));

	fread(p->pI_e, sizeof(real), num, f);
	fread(p->pI_i, sizeof(real), num, f);
	fread(p->pV_e, sizeof(real), num, f);
	fread(p->pV_reset, sizeof(real), num, f);
	fread(p->pC_i, sizeof(real), num, f);
	fread(p->pC_m, sizeof(real), num, f);
	fread(p->pV_m, sizeof(real), num, f);
	fread(p->pV_i, sizeof(real), num, f);
	fread(p->pV_tmp, sizeof(real), num, f);
	fread(p->pC_e, sizeof(real), num, f);
	fread(p->pCe, sizeof(real), num, f);
	fread(p->pCi, sizeof(real), num, f);
	fread(p->pV_thresh, sizeof(real), num, f);

	fread(p->pRefracTime, sizeof(int), num, f);
	fread(p->pRefracStep, sizeof(int), num, f);

	return p;
}

