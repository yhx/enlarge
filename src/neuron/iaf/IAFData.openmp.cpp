#include <assert.h>
#include <pthread.h>
#include <atomic>

#include "../../utils/runtime.h"
#include "../../net/Connection.h"
#include "../../../msg_utils/helper/helper_c.h"

#include "IAFData.h"


/**
 * @brief Openmp version of multithread simulation
 * 
 * @param connection 
 * @param _data 
 * @param buffer 
 * @param firedTable 
 * @param firedTableSizes 
 * @param firedTableCap 
 * @param num 
 * @param offset 
 * @param time 
 */
void updateOpenmpIAF(Connection *connection, void *_data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t offset, int time)
{
	IAFData *data = (IAFData *)_data;
	int currentIdx = time % (connection->maxDelay+1);
    
    // #pragma omp parallel for
	for (size_t nid = 0; nid < num; nid++) {  // for all neuron number
		size_t gnid = offset + nid; 
		if (data->pRefracStep[nid] <= 0) {  // neuron not refractory, so evolve V
			data->pV_m[nid] = (data->pV_m[nid] - data->pE_L[nid]) * data->pP22[nid] 
								+ data->pi_syn_ex[nid] * data->pP21ex[nid]  // input excitatory current
								+ data->pi_syn_in[nid] * data->pP21in[nid]  // input inhibitory current
								+ ( data->pI_e[nid] + data->pi_0[nid] ) * data->pP20[nid] + data->pE_L[nid];
		} else {  // neuron is absolute refractory
			data->pRefracStep[nid]--;
		}

		// exponential decaying PSCs
		data->pi_syn_ex[nid] *= data->pP11ex[nid];
		data->pi_syn_in[nid] *= data->pP11in[nid];

		// add evolution of presynaptic input current
		data->pi_syn_ex[nid] += (1.0 - data->pP11ex[nid]) * data->pi_1[nid];
		
		// the spikes arriving at T+1 have an immediate effect on the state of the neuron
		real weighted_spikes_ex = buffer[gnid];
		real weighted_spikes_in = buffer[gnid + num];

		// std::cout << "buffer: " << buffer[gnid] << " " << buffer[gnid+num] << std::endl;

		data->pi_syn_ex[nid] += weighted_spikes_ex;
		data->pi_syn_in[nid] += weighted_spikes_in; 

		// data->pI_e[nid] += weighted_spikes_ex - weighted_spikes_in;

		bool fired = data->pV_m[nid] >= data->pTheta[nid];
		data->_fire_count[gnid] += fired;

		if (fired) {  // update fire table if fired
			firedTable[firedTableSizes[currentIdx] + firedTableCap * currentIdx] = gnid;
			firedTableSizes[currentIdx]++;

			data->pRefracStep[nid] = data->pRefracTime[nid];
			data->pV_m[nid] = data->pV_reset[nid];
		} 

		data->pi_0[nid] = 0;
		data->pi_1[nid] = 0;
		
		// clear buffer
		buffer[gnid] = 0;		
		buffer[num + gnid] = 0;
	}
}

void updatePthreadIAF(Connection *connection, void *_data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t offset, int time, int thread_num, int thread_id, pthread_barrier_t &pthread_barrier) {
	assert(thread_num > 0);

	pthread_barrier_wait(&pthread_barrier);

	IAFData *data = (IAFData *)_data;
	int currentIdx = time % (connection->maxDelay+1);

	for (size_t idx = 0; idx < num; idx += thread_num) {  // for all neuron number
		size_t nid = idx + thread_id;
		if (nid < num) {
			size_t gnid = offset + nid; 
			if (data->pRefracStep[nid] <= 0) {  // neuron not refractory, so evolve V
				data->pV_m[nid] = (data->pV_m[nid] - data->pE_L[nid]) * data->pP22[nid] 
									+ data->pi_syn_ex[nid] * data->pP21ex[nid]  // input excitatory current
									+ data->pi_syn_in[nid] * data->pP21in[nid]  // input inhibitory current
									+ ( data->pI_e[nid] + data->pi_0[nid] ) * data->pP20[nid] + data->pE_L[nid];
			} else {  // neuron is absolute refractory
				data->pRefracStep[nid]--;
			}

			// exponential decaying PSCs
			data->pi_syn_ex[nid] *= data->pP11ex[nid];
			data->pi_syn_in[nid] *= data->pP11in[nid];

			// add evolution of presynaptic input current
			data->pi_syn_ex[nid] += (1.0 - data->pP11ex[nid]) * data->pi_1[nid];
			
			// the spikes arriving at T+1 have an immediate effect on the state of the neuron
			real weighted_spikes_ex = buffer[gnid];
			real weighted_spikes_in = buffer[gnid + num];

			// std::cout << "buffer: " << buffer[gnid] << " " << buffer[gnid+num] << std::endl;

			data->pi_syn_ex[nid] += weighted_spikes_ex;
			data->pi_syn_in[nid] += weighted_spikes_in; 

			bool fired = data->pV_m[nid] >= data->pTheta[nid];
			data->_fire_count[gnid] += fired;

			if (fired) {  // update fire table if fired
				int testLoc = __atomic_fetch_add(&(firedTableSizes[currentIdx]), 1, __ATOMIC_SEQ_CST);
				firedTable[testLoc + firedTableCap * currentIdx] = gnid;
				// firedTableSizes[currentIdx]++;

				data->pRefracStep[nid] = data->pRefracTime[nid];
				data->pV_m[nid] = data->pV_reset[nid];
			} 

			data->pi_0[nid] = 0;
			data->pi_1[nid] = 0;
			
			// clear buffer
			buffer[gnid] = 0;		
			buffer[num + gnid] = 0;
		}
		// printf("nid: %ld %ld %ld %ld %d\n", nid, num, thread_num, idx, idx <  num);
		// pthread_barrier_wait(&pthread_barrier);
	}
	// printf("[%d finish!]\n", thread_id);
}
