
#include <assert.h>

#include "../../utils/runtime.h"
#include "../../net/Connection.h"

#include "StaticData.h"


void updateStatic(Connection *connection, void *_data, real *buffer, uinteger_t *firedTable, uinteger_t *firedTableSizes, size_t firedTableCap, size_t num, size_t start_id, int time)
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
			for (size_t j = 0; j < synapseNum; j++) {							
				//int sid = connection->pSynapsesIdx[j+startLoc];
				size_t sid = j+startLoc;
				assert(sid < num);  											// 突触连接的数量 
				real weight = data->pWeight[connection->pSidMap[sid]];			// 获得当前连接的权重
				// std::cout << "connection->dst[sid]: " << connection->dst[sid] << std::endl;
			    buffer[connection->dst[sid]] += weight; 						// buffer中目的神经元的输入电流增加weight
			}
		}
	}
}
