#include <stdlib.h>
#include <string.h>
#include "../../../msg_utils/helper/helper_gpu.h"
#include "LIFData.h"

void *cudaMallocLIF()
{
	void *ret = NULL;
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(LIFData)*1));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(LIFData)*1));
	return ret;
}

void *cudaAllocLIF(void *pCPU, size_t num)
{
	void *ret = cudaMallocLIF();
	void *tmp = cudaAllocLIFPara(pCPU, num);
	checkCudaErrors(cudaMemcpy(ret, tmp, sizeof(LIFData)*1, cudaMemcpyHostToDevice));
	free(tmp);
	tmp = NULL;
	return ret;
}

void *cudaAllocLIFPara(void *pCPU, size_t num)
{
	LIFData *p = (LIFData*)pCPU;
	LIFData *ret = (LIFData*)malloc(sizeof(LIFData)*1);
	memset(ret, 0, sizeof(LIFData)*1);

	checkCudaErrors(cudaMalloc((void**)&(ret->pRefracTime), sizeof(int)*num));
	checkCudaErrors(cudaMemset(ret->pRefracTime, 0, sizeof(int)*num));
	checkCudaErrors(cudaMemcpy(ret->pRefracTime, p->pRefracTime, sizeof(int)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(ret->pRefracStep), sizeof(int)*num));
	checkCudaErrors(cudaMemset(ret->pRefracStep, 0, sizeof(int)*num));
	checkCudaErrors(cudaMemcpy(ret->pRefracStep, p->pRefracStep, sizeof(int)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pI_e), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pI_e, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pI_e, p->pI_e, sizeof(real)*num, cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMalloc((void**)&(ret->pV_i), sizeof(real)*num));
	// checkCudaErrors(cudaMemset(ret->pV_i, 0, sizeof(real)*num));
	// checkCudaErrors(cudaMemcpy(ret->pV_i, p->pV_i, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(ret->pCe), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pCe, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pCe, p->pCe, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(ret->pV_reset), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pV_reset, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pV_reset, p->pV_reset, sizeof(real)*num, cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMalloc((void**)&(ret->pV_e), sizeof(real)*num));
	// checkCudaErrors(cudaMemset(ret->pV_e, 0, sizeof(real)*num));
	// checkCudaErrors(cudaMemcpy(ret->pV_e, p->pV_e, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(ret->pV_tmp), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pV_tmp, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pV_tmp, p->pV_tmp, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(ret->pI_i), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pI_i, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pI_i, p->pI_i, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(ret->pV_thresh), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pV_thresh, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pV_thresh, p->pV_thresh, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(ret->pCi), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pCi, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pCi, p->pCi, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(ret->pV_m), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pV_m, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pV_m, p->pV_m, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(ret->pC_e), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pC_e, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pC_e, p->pC_e, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(ret->pC_m), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pC_m, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pC_m, p->pC_m, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(ret->pC_i), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pC_i, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pC_i, p->pC_i, sizeof(real)*num, cudaMemcpyHostToDevice));

	ret->_fire_count = TOGPU(p->_fire_count, num);

	return ret;
}

int cudaFetchLIF(void *pCPU, void *pGPU, size_t num)
{
	LIFData *pTmp = (LIFData*)malloc(sizeof(LIFData)*1);
	memset(pTmp, 0, sizeof(LIFData)*1);
	checkCudaErrors(cudaMemcpy(pTmp, pGPU, sizeof(LIFData)*1, cudaMemcpyDeviceToHost));

	cudaLIFParaFromGPU(pCPU, pTmp, num);
	return 0;
}

int cudaLIFParaToGPU(void *pCPU, void *pGPU, size_t num)
{
	LIFData *pC = (LIFData*)pCPU;
	LIFData *pG = (LIFData*)pGPU;

	checkCudaErrors(cudaMemcpy(pG->pRefracTime, pC->pRefracTime, sizeof(int)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pRefracStep, pC->pRefracStep, sizeof(int)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(pG->pI_e, pC->pI_e, sizeof(real)*num, cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMemcpy(pG->pV_i, pC->pV_i, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pCe, pC->pCe, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pV_reset, pC->pV_reset, sizeof(real)*num, cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMemcpy(pG->pV_e, pC->pV_e, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pV_tmp, pC->pV_tmp, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pI_i, pC->pI_i, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pV_thresh, pC->pV_thresh, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pCi, pC->pCi, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pV_m, pC->pV_m, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pC_e, pC->pC_e, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pC_m, pC->pC_m, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pC_i, pC->pC_i, sizeof(real)*num, cudaMemcpyHostToDevice));

	COPYTOGPU(pG->_fire_count, pC->_fire_count, num);

	return 0;
}

int cudaLIFParaFromGPU(void *pCPU, void *pGPU, size_t num)
{
	LIFData *pC = (LIFData*)pCPU;
	LIFData *pG = (LIFData*)pGPU;

	checkCudaErrors(cudaMemcpy(pC->pRefracTime, pG->pRefracTime, sizeof(int)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pRefracStep, pG->pRefracStep, sizeof(int)*num, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(pC->pI_e, pG->pI_e, sizeof(real)*num, cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(pC->pV_i, pG->pV_i, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pCe, pG->pCe, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pV_reset, pG->pV_reset, sizeof(real)*num, cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(pC->pV_e, pG->pV_e, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pV_tmp, pG->pV_tmp, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pI_i, pG->pI_i, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pV_thresh, pG->pV_thresh, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pCi, pG->pCi, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pV_m, pG->pV_m, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pC_e, pG->pC_e, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pC_m, pG->pC_m, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pC_i, pG->pC_i, sizeof(real)*num, cudaMemcpyDeviceToHost));

	COPYFROMGPU(pC->_fire_count, pG->_fire_count, num);

	return 0;
}

int cudaFreeLIF(void *pGPU)
{
	LIFData *tmp = (LIFData*)malloc(sizeof(LIFData)*1);
	memset(tmp, 0, sizeof(LIFData)*1);
	checkCudaErrors(cudaMemcpy(tmp, pGPU, sizeof(LIFData)*1, cudaMemcpyDeviceToHost));
	cudaFreeLIFPara(tmp);
	free(tmp);
	tmp = NULL;
	cudaFree(pGPU);
	pGPU = NULL;
	return 0;
}

int cudaFreeLIFPara(void *pGPU)
{
	LIFData *p = (LIFData*)pGPU;
	cudaFree(p->pRefracTime);
	p->pRefracTime = NULL;
	cudaFree(p->pRefracStep);
	p->pRefracStep = NULL;

	cudaFree(p->pI_e);
	p->pI_e = NULL;
	// cudaFree(p->pV_i);
	// p->pV_i = NULL;
	cudaFree(p->pCe);
	p->pCe = NULL;
	cudaFree(p->pV_reset);
	p->pV_reset = NULL;
	// cudaFree(p->pV_e);
	// p->pV_e = NULL;
	cudaFree(p->pV_tmp);
	p->pV_tmp = NULL;
	cudaFree(p->pI_i);
	p->pI_i = NULL;
	cudaFree(p->pV_thresh);
	p->pV_thresh = NULL;
	cudaFree(p->pCi);
	p->pCi = NULL;
	cudaFree(p->pV_m);
	p->pV_m = NULL;
	cudaFree(p->pC_e);
	p->pC_e = NULL;
	cudaFree(p->pC_m);
	p->pC_m = NULL;
	cudaFree(p->pC_i);
	p->pC_i = NULL;

	gpuFree(p->_fire_count);

	return 0;
}

int cudaLogRateLIF(void *cpu, void *gpu, const char *name)
{
	LIFData *c = static_cast<LIFData *>(cpu);
	LIFData *g = static_cast<LIFData *>(gpu);

	LIFData *t = FROMGPU(g, 1);
	COPYFROMGPU(c->_fire_count, t->_fire_count, c->num);
	return logRateLIF(cpu, name);
}

real * cudaGetVLIF(void *data) {
	LIFData *c_g_lif = FROMGPU(static_cast<LIFData *>(data), 1);
	return c_g_lif->pV_m;
}
