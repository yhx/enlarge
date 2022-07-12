#include <assert.h>
#include <random>
#include <cstdlib>
#include <ctime>

#include "../../utils/runtime.h"
#include "../../net/Connection.h"

#include "PoissonData.h"

/**
 * poisson neuron 存储发放平均值
 * poisson synapse 根据poisson neuron的发放均值生成对应的泊松分布值，即发放脉冲的数量
 **/
void updatePoisson(Connection *connection, void *_data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int time)
{
	// std::default_random_engine generator;
	PoissonData *data = (PoissonData *)_data;
	// int delayLength = connection->maxDelay - connection->minDelay + 1;
	// for (int delta_t = 0; delta_t < delayLength; delta_t++)
	// {
	// 	int time_idx = (time + delayLength - delta_t) % (connection->maxDelay + 1);
	// std::cout << data->num << std::endl;
	// std::random_device rd;
    // std::mt19937 gen(rd());
	for (size_t i = 0; i < data->num; i++) {  // 对于所有poisson突触
		// std::poisson_distribution<int> pd(data->pMean[connection->pSidMap[i]]);
		buffer[connection->dst[i]] += (data->pPoissonGenerator[i])(data->pGenerator[i]) * data->pWeight[connection->pSidMap[i]];
		// buffer[connection->dst[i]] += 2 * data->pWeight[connection->pSidMap[i]];
		// std::cout << connection->dst[i] << std::endl; 
	}
	// }
}
