
#ifndef CONVERT_H
#define CONVERT_H

#include "../../msg_utils/msg_utils/CrossMap.h"
#include "../../msg_utils/msg_utils/CrossSpike.h"
#include "CrossNodeMap.h"
#include "CrossNodeData.h"

CrossMap * convert2crossmap(CrossNodeMap * cnm);
CrossSpike * convert2crossspike(CrossNodeData *cnd, int proc_rank, int gpu_num);
CrossSpike * convert2crossspike2(CrossNodeData *cnd, int proc_rank, int gpu_num, int thread_num);

#endif //CONVERT_H
