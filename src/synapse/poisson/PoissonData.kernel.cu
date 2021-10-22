
#include "../../gpu_utils/runtime.h"

#include "PoissonData.h"
#include <curand_kernel.h>


__global__ void update_dense_poisson_hit(Connection *connection, PoissonData *data, real *buffer, const uinteger_t *firedTable, const uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int time)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;  
	for (int idx = 0; idx < data->num; idx += blockDim.x * gridDim.x) {
		int sid = tid + idx;  // sid: synapse id
		if (sid < data->num) {
			curandState localState = data->pState[sid];
			int tmp = curand_poisson(&localState, data->pMean[connection->pSidMap[sid]]);
			data->pState[sid] = localState;
			atomicAdd(&(buffer[connection->dst[sid]]), tmp * data->pWeight[connection->pSidMap[sid]]);
		}
		// __syncthreads();
	}	
}

// __global__ void curand_setup_poisson_init_state(PoissonData *data)
// {
// 	int tid = threadIdx.x + blockIdx.x * blockDim.x;
// 	// if (id < num) {
// 	// 	curand_init(1234, id, 0, &state[id]);
// 	// }
// 	for (int idx = 0; idx < data->num; idx += blockDim.x * gridDim.x) {
// 		int sid = tid + idx;
// 		if (sid < data->num) {
// 			curand_init(666, sid, 0, &(data->pState[sid]));
// 		}
// 	}
// }

void cudaUpdatePoisson(Connection * connection, void *data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int time, BlockSize *pSize)
{
	std::cout << "poisson cuda kernel: " << std::endl;
	// pSize->gridSize = 1;
	// pSize->blockSize = 128;
	std::cout << pSize->gridSize << " " << pSize->blockSize << std::endl;
	// std::cout << "buffer: ";
	// for (int i = 0; i < num; ++i)
	// 	std::cout << buffer[i] << " ";
	// std::cout << std::endl;
	// curand_setup_poisson_init_state<<<pSize->gridSize, pSize->blockSize>>>((PoissonData *)data);
	update_dense_poisson_hit<<<pSize->gridSize, pSize->blockSize>>>((Connection *)connection,  (PoissonData *)data, buffer, firedTable, firedTableSizes, firedTableCap, num, start_id, time);
}
