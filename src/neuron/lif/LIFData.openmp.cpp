#include <atomic>

#include "../../utils/runtime.h"
#include "../../net/Connection.h"

#include "LIFData.h"

void updateOpenmpLIF(Connection *connection, void *_data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t offset, int time)
{
	LIFData *data = (LIFData *)_data;
	int currentIdx = time % (connection->maxDelay+1);
    // #pragma parallel for
	for (size_t nid=0; nid<num; nid++) {
		size_t gnid = offset + nid; 

		if (data->pRefracStep[nid] <= 0) {
			data->pV_m[nid] = data->pC_m[nid] * data->pV_m[nid] + data->pV_tmp[nid] + data->pI_e[nid] * data->pC_e[nid] + data->pI_i[nid] * data->pC_i[nid];

			//data->p_i_syn[nid] = 0;

			data->pI_e[nid] *= data->pCe[nid];
			data->pI_i[nid] *= data->pCi[nid];

			bool fired = data->pV_m[nid] >= data->pV_thresh[nid];
			data->_fire_count[gnid] += fired;

			if (fired) {
				firedTable[firedTableSizes[currentIdx] + firedTableCap * currentIdx] = gnid;
				firedTableSizes[currentIdx]++;

				data->pRefracStep[nid] = data->pRefracTime[nid] - 1;
				data->pV_m[nid] = data->pV_reset[nid];
			} else {
				data->pI_e[nid] += buffer[gnid];
				data->pI_i[nid] += buffer[num + gnid];
			}
	
		} else {
			data->pRefracStep[nid]--;
		}
		buffer[gnid] = 0;
		buffer[num + gnid] = 0;
	}
}


void updatePthreadLIF(Connection *connection, void *_data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t offset, int time, int thread_num, int thread_id, pthread_barrier_t &pthread_barrier)
{
}


