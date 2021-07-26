
#ifdef CONVERT_H
#define CONVERT_H

#include "../../msg_utils/msg_utils/CrossMap.h"
#include "../../msg_utils/msg_utils/CrossSpike.h"

CrossMap * convert2crossmap(CrossNodeMap * cnm);
CrossSpike * convert2crossspike(CrossNodeData *cnd, int proc_rank, int gpu_rank, int gpu_num, int gpu_group);

#endif //CONVERT_H
