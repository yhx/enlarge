
#ifndef CONVERT_H
#define CONVERT_H

#include "../../msg_utils/msg_utils/CrossMap.h"
#include "../../msg_utils/msg_utils/CrossSpike.h"
#include "../../msg_utils/msg_utils/HybridCrossMap.h"
#include "../../msg_utils/msg_utils/HybridCrossSpike.h"
#include "CrossNodeMap.h"
#include "CrossNodeData.h"

CrossMap * convert2crossmap(CrossNodeMap * cnm);
CrossSpike * convert2crossspike(CrossNodeData *cnd, int proc_rank, int gpu_num);

// for hybrid simulator
HybridCrossMap * convert2hybridcrossmap(CrossNodeMap * cnm);
HybridCrossSpike * convert2hybridcrossspike(CrossNodeData *cnd, int proc_rank, int gpu_num);

#endif //CONVERT_H
