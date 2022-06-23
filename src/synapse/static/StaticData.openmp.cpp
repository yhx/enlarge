
#include <assert.h>
#include <atomic>

#include "../../utils/runtime.h"
#include "../../net/Connection.h"
#include "../../../msg_utils/helper/helper_c.h"

#include "StaticData.h"


void updateOpenmpStatic(Connection *connection, void *_data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int time)
{
	// num是神经元数量
	StaticData * data = (StaticData *)_data;  									// 获取当前突触的连接权重等信息
	int delayLength = connection->maxDelay - connection->minDelay + 1;			// 计算所有可能的delay数量
	for (int delta_t = 0; delta_t < delayLength; delta_t++) {						
		int time_idx = (time+delayLength-delta_t)%(connection->maxDelay+1);		// 优先处理delay大的时间节点（time_idx）
		size_t firedSize = firedTableSizes[time_idx];  							// firedSize保存了firedTableSizes中在time_idx步发放的神经元数量

		for (size_t i = 0; i < firedSize; i++) {
		    size_t nid = firedTable[time_idx*firedTableCap + i];  				// 获取当前发放的神经元的neuron id即nid
			// size_t startLoc = access_(connection->pDelayStart, delta_t, nid);  	// 获取nid神经元连接突触的sid的最小值，它们都是连续存储的，可以通过startLoc+j来访问
			// size_t synapseNum = access_(connection->pDelayNum, delta_t, nid);	// 获取nid神经元的突触数量
			size_t startLoc = access_connection_(connection->pDelayStart, delta_t, connection->nNum, nid);
			size_t synapseNum = access_connection_(connection->pDelayNum, delta_t, connection->nNum, nid);
			// size_t startLoc = connection->pDelayStart[delta_t + nid * delayLength];
			// size_t synapseNum = connection->pDelayNum[delta_t + nid * delayLength];
			// #pragma parallel for
			for (size_t j = 0; j < synapseNum; j++) {							
				//int sid = connection->pSynapsesIdx[j+startLoc];
				size_t sid = j + startLoc;
				assert(sid < num);  											// 突触连接的数量 
				real weight = data->pWeight[connection->pSidMap[sid]];			// 获得当前连接的权重
				// std::cout << "connection->dst[sid]: " << connection->dst[sid] << std::endl;
			    buffer[connection->dst[sid]] += weight; 						// buffer中目的神经元的输入电流增加weight
			}
		}
	}
}

// void updatePthreadStatic(Connection *connection, void *_data, std::atomic<real> *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int time, int thread_num) {
void updatePthreadStatic(Connection *connection, void *_data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int time, int thread_num, int thread_id, pthread_barrier_t &pthread_barrier) {
	assert(thread_num > 0);
	pthread_barrier_wait(&pthread_barrier);
	// num是神经元数量
	StaticData * data = (StaticData *)_data;  									// 获取当前突触的连接权重等信息
	int delayLength = connection->maxDelay - connection->minDelay + 1;			// 计算所有可能的delay数量
	for (int delta_t = 0; delta_t < delayLength; delta_t++) {						
		int time_idx = (time+delayLength-delta_t)%(connection->maxDelay+1);		// 优先处理delay大的时间节点（time_idx）
		size_t firedSize = firedTableSizes[time_idx];  							// firedSize保存了firedTableSizes中在time_idx步发放的神经元数量

		for (size_t i = 0; i < firedSize; i++) {
		    size_t nid = firedTable[time_idx*firedTableCap + i];  				// 获取当前发放的神经元的neuron id即nid
			size_t startLoc = access_connection_(connection->pDelayStart, delta_t, connection->nNum, nid);
			size_t synapseNum = access_connection_(connection->pDelayNum, delta_t, connection->nNum, nid);
			
			for (size_t j = 0; j < synapseNum; j += thread_num) {							
				int idx = j + thread_id;
				if (idx < synapseNum) {
					size_t sid = idx + startLoc;
					assert(sid < num);  											// 突触连接的数量 
					real weight = data->pWeight[connection->pSidMap[sid]];			// 获得当前连接的权重
					// buffer[connection->dst[sid]] += weight; 						// buffer中目的神经元的输入电流增加weight
					
					long long *pSrc = (long long*)&(buffer[connection->dst[sid]]);
					// printf("buffer pSrc: %lf\n", buffer[connection->dst[sid]]);
					// printf("pSrc %lld\n", *pSrc);
					real pCur_tmp=0;
					long long *pCur = (long long*)&pCur_tmp, *pDes;
					real ret;
					// printf("before buffer: %lf, %lf\n", buffer[connection->dst[sid]], weight);
					do {
						__atomic_load(pSrc, pCur, __ATOMIC_ACQUIRE);
						ret = *(real*)pCur + weight;
						// printf("ret: %lf\n", ret);
						pDes = (long long*)&(ret);
					} while (!__atomic_compare_exchange(pSrc, pCur, pDes, true, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE));
					// printf("after buffer: %lf\n", buffer[connection->dst[sid]]);
					
					// do {
					// 	real expected = std::atomic_load(&(buffer[connection->dst[sid]]));
					// } while (!std::atomic_compare_exchange_weak(&(buffer[connection->dst[sid]]), &expected, expected + weight));
				}
			}
		}

		pthread_barrier_wait(&pthread_barrier);
	}
}
