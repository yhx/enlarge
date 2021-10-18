
#include <assert.h>

#include "../../utils/runtime.h"
#include "../../net/Connection.h"

#include "PoissonData.h"

void updatePoisson(Connection *connection, void *_data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int time)
{
	PoissonData *data = (PoissonData *)_data;
	int delayLength = connection->maxDelay - connection->minDelay + 1;
	for (int delta_t = 0; delta_t < delayLength; delta_t++)
	{
		int time_idx = (time + delayLength - delta_t) % (connection->maxDelay + 1);
		size_t firedSize = firedTableSizes[time_idx];

		for (size_t i = 0; i < firedSize; i++)
		{
			size_t nid = firedTable[time_idx * firedTableCap + i];
			size_t startLoc = access_(connection->pDelayStart, delta_t, nid);
			size_t synapseNum = access_(connection->pDelayNum, delta_t, nid);
			// size_t startLoc = connection->pDelayStart[delta_t + nid * delayLength];
			// size_t synapseNum = connection->pDelayNum[delta_t + nid * delayLength];
			for (size_t j = 0; j < synapseNum; j++)
			{
				//int sid = connection->pSynapsesIdx[j+startLoc];
				size_t sid = j + startLoc;
				assert(sid < num);
				real weight = data->pWeight[connection->pSidMap[sid]];
				buffer[connection->dst[sid]] += weight;
			}
		}
	}
}
