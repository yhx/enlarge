
#include <assert.h>

#include "../../utils/runtime.h"
#include "../../net/Connection.h"

#include "STDPData.h"


void updateSTDP(Connection *connection, void *_data, real *currentE, real *currentI, int *firedTable, int *firedTableSizes, int firedTableCap, int num, int start_id, int time)
{
	STDPData * data = (STDPData *)_data;
	int delayLength = connection->maxDelay - connection->minDelay + 1;
	for (int delta_t = 0; delta_t<delayLength; delta_t++) {
		int time_idx = (time+delayLength-delta_t)%(connection->maxDelay+1);
		int firedSize = firedTableSizes[time_idx];

		for (int i=0; i<firedSize; i++) {
			int nid = firedTable[time_idx*firedTableCap + i];
			int startLoc = connection->pDelayStart[delta_t + nid * delayLength];
			int synapseNum = connection->pDelayNum[delta_t + nid * delayLength];
			for (int j=0; j<synapseNum; j++) {
				//int sid = connection->pSynapsesIdx[j+startLoc];
				int sid = j+startLoc;
				assert(sid < num);
				real weight = data->pWeight[sid];
				if (weight >= 0) {
					currentE[data->pDst[sid]] += weight;
				} else {
					currentI[data->pDst[sid]] += weight;
				}
			}
		}
	}
}
