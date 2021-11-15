#include <stdlib.h>
#include <string.h>
#include "../../../msg_utils/helper/helper_gpu.h"
#include "IAFData.h"

void *cudaMallocIAF()
{
	void *ret = NULL;
	checkCudaErrors(cudaMalloc((void**)&(ret), sizeof(IAFData)*1));
	checkCudaErrors(cudaMemset(ret, 0, sizeof(IAFData)*1));
	return ret;
}

void *cudaAllocIAF(void *pCPU, size_t num)
{
	void *ret = cudaMallocIAF();
	void *tmp = cudaAllocIAFPara(pCPU, num);
	checkCudaErrors(cudaMemcpy(ret, tmp, sizeof(IAFData)*1, cudaMemcpyHostToDevice));
	free(tmp);
	tmp = NULL;
	return ret;
}

void *cudaAllocIAFPara(void *pCPU, size_t num)
{
	IAFData *p = (IAFData*)pCPU;
	IAFData *ret = (IAFData*)malloc(sizeof(IAFData)*1);
	memset(ret, 0, sizeof(IAFData)*1);

	checkCudaErrors(cudaMalloc((void**)&(ret->pRefracTime), sizeof(int)*num));
	checkCudaErrors(cudaMemset(ret->pRefracTime, 0, sizeof(int)*num));
	checkCudaErrors(cudaMemcpy(ret->pRefracTime, p->pRefracTime, sizeof(int)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&(ret->pRefracStep), sizeof(int)*num));
	checkCudaErrors(cudaMemset(ret->pRefracStep, 0, sizeof(int)*num));
	checkCudaErrors(cudaMemcpy(ret->pRefracStep, p->pRefracStep, sizeof(int)*num, cudaMemcpyHostToDevice));

	// model param
	checkCudaErrors(cudaMalloc((void**)&(ret->pE_L), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pE_L, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pE_L, p->pE_L, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pI_e), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pI_e, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pI_e, p->pI_e, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pTheta), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pTheta, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pTheta, p->pTheta, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pV_reset), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pV_reset, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pV_reset, p->pV_reset, sizeof(real)*num, cudaMemcpyHostToDevice));

	// state param
	checkCudaErrors(cudaMalloc((void**)&(ret->pi_0), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pi_0, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pi_0, p->pi_0, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pi_1), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pi_1, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pi_1, p->pi_1, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pi_syn_ex), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pi_syn_ex, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pi_syn_ex, p->pi_syn_ex, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pi_syn_in), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pi_syn_in, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pi_syn_in, p->pi_syn_in, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pV_m), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pV_m, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pV_m, p->pV_m, sizeof(real)*num, cudaMemcpyHostToDevice));

	// internal param
	checkCudaErrors(cudaMalloc((void**)&(ret->pP11ex), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pP11ex, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pP11ex, p->pP11ex, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pP11in), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pP11in, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pP11in, p->pP11in, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pP22), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pP22, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pP22, p->pP22, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pP21ex), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pP21ex, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pP21ex, p->pP21ex, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pP21in), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pP21in, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pP21in, p->pP21in, sizeof(real)*num, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&(ret->pP20), sizeof(real)*num));
	checkCudaErrors(cudaMemset(ret->pP20, 0, sizeof(real)*num));
	checkCudaErrors(cudaMemcpy(ret->pP20, p->pP20, sizeof(real)*num, cudaMemcpyHostToDevice));


	ret->_fire_count = TOGPU(p->_fire_count, num);

	return ret;
}

int cudaFetchIAF(void *pCPU, void *pGPU, size_t num)
{
	IAFData *pTmp = (IAFData*)malloc(sizeof(IAFData)*1);
	memset(pTmp, 0, sizeof(IAFData)*1);
	checkCudaErrors(cudaMemcpy(pTmp, pGPU, sizeof(IAFData)*1, cudaMemcpyDeviceToHost));

	cudaIAFParaFromGPU(pCPU, pTmp, num);
	return 0;
}

int cudaIAFParaToGPU(void *pCPU, void *pGPU, size_t num)
{
	IAFData *pC = (IAFData*)pCPU;
	IAFData *pG = (IAFData*)pGPU;

	checkCudaErrors(cudaMemcpy(pG->pRefracTime, pC->pRefracTime, sizeof(int)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pRefracStep, pC->pRefracStep, sizeof(int)*num, cudaMemcpyHostToDevice));

	// model param
	checkCudaErrors(cudaMemcpy(pG->pE_L, pC->pE_L, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pI_e, pC->pI_e, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pTheta, pC->pTheta, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pV_reset, pC->pV_reset, sizeof(real)*num, cudaMemcpyHostToDevice));

	// state param
	checkCudaErrors(cudaMemcpy(pG->pi_0, pC->pi_0, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pi_1, pC->pi_1, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pi_syn_ex, pC->pi_syn_ex, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pi_syn_in, pC->pi_syn_in, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pV_m, pC->pV_m, sizeof(real)*num, cudaMemcpyHostToDevice));

	// internal param
	checkCudaErrors(cudaMemcpy(pG->pP11ex, pC->pP11ex, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pP11in, pC->pP11in, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pP22, pC->pP22, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pP21ex, pC->pP21ex, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pP21in, pC->pP21in, sizeof(real)*num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(pG->pP20, pC->pP20, sizeof(real)*num, cudaMemcpyHostToDevice));

	COPYTOGPU(pG->_fire_count, pC->_fire_count, num);

	return 0;
}

int cudaIAFParaFromGPU(void *pCPU, void *pGPU, size_t num)
{
	IAFData *pC = (IAFData*)pCPU;
	IAFData *pG = (IAFData*)pGPU;

	checkCudaErrors(cudaMemcpy(pC->pRefracTime, pG->pRefracTime, sizeof(int)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pRefracStep, pG->pRefracStep, sizeof(int)*num, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(pC->pE_L, pG->pE_L, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pI_e, pG->pI_e, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pTheta, pG->pTheta, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pV_reset, pG->pV_reset, sizeof(real)*num, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaMemcpy(pC->pi_0, pG->pi_0, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pi_1, pG->pi_1, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pi_syn_ex, pG->pi_syn_ex, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pi_syn_in, pG->pi_syn_in, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pV_m, pG->pV_m, sizeof(real)*num, cudaMemcpyDeviceToHost));
	
	checkCudaErrors(cudaMemcpy(pC->pP11ex, pG->pP11ex, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pP11in, pG->pP11in, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pP22, pG->pP22, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pP21ex, pG->pP21ex, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pP21in, pG->pP21in, sizeof(real)*num, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(pC->pP20, pG->pP20, sizeof(real)*num, cudaMemcpyDeviceToHost));

	COPYFROMGPU(pC->_fire_count, pG->_fire_count, num);

	return 0;
}

int cudaFreeIAF(void *pGPU)
{
	IAFData *tmp = (IAFData*)malloc(sizeof(IAFData)*1);
	memset(tmp, 0, sizeof(IAFData)*1);
	checkCudaErrors(cudaMemcpy(tmp, pGPU, sizeof(IAFData)*1, cudaMemcpyDeviceToHost));
	cudaFreeIAFPara(tmp);
	free(tmp);
	tmp = NULL;
	cudaFree(pGPU);
	pGPU = NULL;
	return 0;
}

int cudaFreeIAFPara(void *pGPU)
{
	IAFData *p = (IAFData*)pGPU;
	cudaFree(p->pRefracTime);
	p->pRefracTime = NULL;
	cudaFree(p->pRefracStep);
	p->pRefracStep = NULL;

	cudaFree(p->pE_L);
	p->pE_L = NULL;
	cudaFree(p->pI_e);
	p->pI_e = NULL;
	cudaFree(p->pTheta);
	p->pTheta = NULL;
	cudaFree(p->pV_reset);
	p->pV_reset = NULL;

	cudaFree(p->pi_0);
	p->pi_0 = NULL;
	cudaFree(p->pi_1);
	p->pi_1 = NULL;
	cudaFree(p->pi_syn_ex);
	p->pi_syn_ex = NULL;
	cudaFree(p->pi_syn_in);
	p->pi_syn_in = NULL;
	cudaFree(p->pV_m);
	p->pV_m = NULL;

	cudaFree(p->pP11ex);
	p->pP11ex = NULL;
	cudaFree(p->pP11in);
	p->pP11in = NULL;
	cudaFree(p->pP22);
	p->pP22 = NULL;
	cudaFree(p->pP21ex);
	p->pP21ex = NULL;
	cudaFree(p->pP21in);
	p->pP21in = NULL;
	cudaFree(p->pP20);
	p->pP20 = NULL;

	gpuFree(p->_fire_count);

	return 0;
}

int cudaLogRateIAF(void *cpu, void *gpu, const char *name)
{
	IAFData *c = static_cast<IAFData *>(cpu);
	IAFData *g = static_cast<IAFData *>(gpu);

	IAFData *t = FROMGPU(g, 1);
	COPYFROMGPU(c->_fire_count, t->_fire_count, c->num);
	return logRateIAF(cpu, name);
}

real * cudaGetVIAF(void *data) {
	IAFData *c_g_lif = FROMGPU(static_cast<IAFData *>(data), 1);
	return c_g_lif->pV_m;
}
