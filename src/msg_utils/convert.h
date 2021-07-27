
#ifdef CONVERT_H
#define CONVERT_H

#include "../../msg_utils/msg_utils/CrossMap.h"
#include "../../msg_utils/msg_utils/CrossSpike.h"

CrossMap * convert2crossmap(CrossNodeMap * cnm);
CrossSpike * convert2crossspike(CrossNodeData *cnd, int proc_rank, int gpu_rank = 0, int gpu_num = 1, int gpu_group = 0);

#endif //CONVERT_H
